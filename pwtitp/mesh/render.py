"""First-order Ambisonic (SN3D: W, X, Y, Z = [1, u]) responses along a stored path of a mesh scene (bank.py).

Specular part: image sources up to 50 reflections with band gains, air absorption, exact visibility on the 1-mm path grid
(edges faded over the band's Fresnel zone) and a random source directivity. Point scatterers: single scattering by up to
60 small objects. Scattered field: plane-wave arrivals whose arrival times follow the diffuse-rain energy envelope and
whose directions follow its energy-weighted mean direction per time bin (von Mises-Fisher with the same mean resultant
length); each arrival is a plane wave, so it moves with the array (delay -u.dp/c). Every arrival is encoded with its own
direction u (unit vector from the array towards the arrival). A random transducer coloration is applied to all channels.
"""
import math
import torch
from .geometry import BANDS, AIR_DB_M, C
from .bank import DL

FS = 16000


_BITS = {}


def _unpack(bits, N, dev):
    """np.packbits(axis=1) inverse on the GPU: uint8 [M, ceil(N/8)] -> bool [M, N]"""
    if dev not in _BITS:
        _BITS[dev] = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=dev)
    b = bits.to(dev, non_blocking=True)
    return (b[..., None] & _BITS[dev]).ne(0).reshape(b.shape[0], -1)[:, :N]


def _band_masks(F, dev):
    """partition of unity over the 8 octave bands on the rfft grid (raised-cosine crossovers in log-frequency)"""
    f = torch.linspace(0, FS / 2, F, device=dev, dtype=torch.float64).clamp_min(1.0)
    lf = torch.log2(f)
    cen = torch.log2(torch.tensor(BANDS, device=dev, dtype=torch.float64))
    m = torch.zeros(8, F, device=dev, dtype=torch.float64)
    for b in range(8):
        if b < 7:  # transition to next band between centres
            x = ((lf - cen[b]) / (cen[b + 1] - cen[b])).clamp(0, 1)
            down = 0.5 + 0.5 * torch.cos(math.pi * x)
        else:
            down = torch.ones_like(lf)
        if b > 0:
            x = ((lf - cen[b - 1]) / (cen[b] - cen[b - 1])).clamp(0, 1)
            up = 0.5 - 0.5 * torch.cos(math.pi * x)
        else:
            up = torch.ones_like(lf)
        m[b] = torch.where(lf <= cen[b], up, down)
    return (m / m.sum(0, keepdim=True)).float()


def _box_rows(x, w):
    """moving average of each row of x [M, N] with per-row half-width w [M] (samples), edge-padded"""
    M, N = x.shape
    W = int(w.max())
    if W == 0:
        return x
    xp = torch.cat([x[:, :1].expand(M, W + 1), x, x[:, -1:].expand(M, W)], 1)
    c = xp.cumsum(1)
    i = torch.arange(N, device=x.device)[None] + W + 1
    hi = c.gather(1, i + w[:, None]); lo = c.gather(1, i - w[:, None] - 1)
    return (hi - lo) / (2 * w[:, None] + 1)


def _smooth_vis_bands(vis, r, gi, dev):
    """[8, J, M] band-wise smoothed visibility at grid indices gi; only rows whose visibility changes are smoothed"""
    J, M = gi.shape[0], vis.shape[0]
    out = vis[:, gi].T.float()[None].repeat(8, 1, 1)
    ch = (vis[:, 1:] != vis[:, :-1]).any(1).nonzero().squeeze(1)
    for c0 in range(0, len(ch), 4096):
        c = ch[c0:c0 + 4096]
        vf = vis[c].float()
        for b in range(8):
            out[b][:, c] = _smooth_vis(vf, r[c], b, dev)[:, gi].T
    return out


def _smooth_vis(vis, r, b, dev):
    """band-b smoothed visibility [M, N] (two box passes ~ triangle of half-width 2w); r [M] mean path length (m)"""
    lam = C / BANDS[b]
    w = (0.25 * math.sqrt(lam * 1.0) * r.sqrt() / DL).round().long().clamp(0, 400)
    return _box_rows(_box_rows(vis, w), w)


def _frac_render(delay, val, L, taps=16):
    """windowed-sinc fractional delay. delay [J, M], val [J, M, Ch] -> [J, L, Ch]"""
    J, M = delay.shape; Ch = val.shape[-1]
    i0 = delay.floor().long(); t = torch.arange(-taps + 1, taps + 1, device=delay.device)
    idx = i0[..., None] + t; x = t.float() - (delay - i0)[..., None]
    w = torch.sinc(x) * (0.5 + 0.5 * torch.cos(math.pi * x / taps)).clamp(min=0)
    ok = (idx >= 0) & (idx < L); idx = torch.where(ok, idx, L)
    out = torch.zeros(J, L + 1, Ch, device=delay.device)
    flat = (idx + (torch.arange(J, device=delay.device) * (L + 1))[:, None, None]).reshape(-1)
    out.view(-1, Ch).index_add_(0, flat, ((w * ok)[..., None] * val[:, :, None, :]).reshape(-1, Ch))
    return out[:, :L]


def _vmf(mu, kappa, gen=None):
    """unit vectors from von Mises-Fisher distributions with mean directions mu [n, 3] and concentrations kappa [n]"""
    n = mu.shape[0]; dev = mu.device
    xi = torch.rand(n, device=dev, dtype=torch.float64, generator=gen).clamp(1e-12, 1)
    k = kappa.double().clamp(1e-6, 500)
    w = 1 + torch.log(xi + (1 - xi) * torch.exp(-2 * k)) / k            # cosine to the mean direction
    w = torch.where(kappa < 1e-6, 2 * xi - 1, w).clamp(-1, 1)
    v = torch.randn(n, 3, device=dev, dtype=torch.float64, generator=gen)
    v = v - (v * mu).sum(-1, keepdim=True) * mu; v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return w[:, None] * mu + (1 - w ** 2).clamp_min(0).sqrt()[:, None] * v


def _kappa(R):
    """concentration of a 3-D von Mises-Fisher distribution with mean resultant length R (coth k - 1/k = R)"""
    R = R.clamp(0, 0.98)
    k = R * (3 - R ** 2) / (1 - R ** 2)                                   # Banerjee et al. (2005) approximation
    for _ in range(3):                                                    # Newton steps
        f = 1 / torch.tanh(k.clamp_min(1e-6)) - 1 / k.clamp_min(1e-6) - R
        df = 1 / k.clamp_min(1e-6) ** 2 - 1 / torch.sinh(k.clamp_min(1e-6)) ** 2
        k = (k - f / df.clamp_min(1e-9)).clamp_min(0)
    return torch.where(R < 1e-3, torch.zeros_like(k), k)


@torch.no_grad()
def render_foa(s, svals, K, dev, n_scat=(0, 60), n_arr=2400):
    """FOA field [J, K, 4] at arc-length coordinates svals [J] (m along the stored path) of scene s; also returns the
    positions [J, 3]. Random quantities (directivity, scatterer subset, diffuse arrivals, coloration) use torch's global
    generator, so a fixed torch seed gives a fixed field."""
    from ..sim import _rand_filter_resp
    U = lambda *sh, lo=0., hi=1.: torch.empty(*sh, device=dev).uniform_(lo, hi)
    J = svals.shape[0]; F = K + 1
    N = int(round(s['L'] / DL)) + 1
    gi = (svals / DL).round().long().clamp(0, N - 1)
    pth = s['path'].to(dev)
    xf = (svals.float() / DL).clamp(0, N - 1); i0 = xf.floor().long().clamp(max=N - 2); fr = (xf - i0)[:, None]
    pos = pth[i0] * (1 - fr) + pth[i0 + 1] * fr
    src = s['src'].to(dev)
    masks = _band_masks(F, dev)
    air = torch.tensor(AIR_DB_M, device=dev, dtype=torch.float32)
    sdir = torch.randn(3, device=dev); sdir = sdir / sdir.norm(); a_dir = float(U(1, hi=0.6))
    ip = torch.cat([src[None], s['img'].to(dev)])
    g = torch.cat([torch.ones(1, 8), s['gain']]).to(dev)
    vis = torch.cat([torch.ones(1, N, dtype=torch.bool, device=dev), _unpack(s['vis'], N, dev)]) if s['img'].shape[0] else torch.ones(1, N, dtype=torch.bool, device=dev)
    JC = 256
    dmin = torch.full((ip.shape[0],), float('inf'), device=dev); dsum = torch.zeros(ip.shape[0], device=dev)
    for j0 in range(0, J, JC):
        dj = (pos[j0:j0 + JC, None] - ip[None]).norm(dim=-1)
        dmin = torch.minimum(dmin, dj.min(0).values); dsum += dj.sum(0)
    keep = dmin < C * K / FS + 1
    ip, g, vis, r_all = ip[keep], g[keep], vis[keep], (dsum / J)[keep]
    taps = 16
    tt = torch.arange(-taps + 1, taps + 1, device=dev)
    spec = torch.zeros(J, F, 4, dtype=torch.complex64, device=dev)
    for j0 in range(0, J, JC):
        js = slice(j0, min(j0 + JC, J)); Jc = js.stop - js.start
        dj = (pos[js, None] - ip[None]).norm(dim=-1)
        kj = dj.min(0).values < C * K / FS + 1
        ipj, gj, visj, r_mean, dj = ip[kj], g[kj], vis[kj], r_all[kj], dj[:, kj]
        M = ipj.shape[0]
        u = (ipj[None] - pos[js, None]) / dj[..., None]                # towards the (image) source
        dirg = (1 - a_dir) + a_dir * (-u @ sdir)
        att = 10 ** (-air[:, None, None] * dj[None] / 20) * gj.T[:, None] * (dirg / (4 * math.pi * dj))[None]   # [8, Jc, M]
        delay = dj / C * FS
        vball = _smooth_vis_bands(visj, r_mean, gi[js], dev)
        Y = torch.cat([torch.ones_like(dj)[..., None], u], -1)          # [Jc, M, 4]
        for b in range(8):
            out = torch.zeros(Jc, 2 * K + 1, 4, device=dev)
            for c0 in range(0, M, 2048):
                sl = slice(c0, min(c0 + 2048, M))
                i0_ = delay[:, sl].floor().long(); x = tt.float() - (delay[:, sl] - i0_)[..., None]
                wgt = torch.sinc(x) * (0.5 + 0.5 * torch.cos(math.pi * x / taps)).clamp(min=0)
                idx = i0_[..., None] + tt
                ok_ = (idx >= 0) & (idx < 2 * K)
                idx = torch.where(ok_, idx, 2 * K)
                a = (att[b, :, sl] * vball[b, :, sl])[..., None] * Y[:, sl]          # [Jc, Mc, 4]
                vals = (wgt * ok_)[..., None] * a[:, :, None, :]                     # [Jc, Mc, 2T, 4]
                flat = (idx + (torch.arange(Jc, device=dev) * (2 * K + 1))[:, None, None]).reshape(-1)
                out.view(-1, 4).index_add_(0, flat, vals.reshape(-1, 4))
            spec[js] += torch.fft.rfft(out[:, :2 * K], n=2 * K, dim=1)[:, :F] * masks[b][None, :, None]
        del u, dirg, att, delay, vball, Y, out
    # point scatterers
    ns = int(U(1, lo=n_scat[0], hi=min(n_scat[1], s['scat'].shape[0]) + 1).item()) if s['scat'].shape[0] else 0
    if ns > 0:
        sel = torch.randperm(s['scat'].shape[0], device='cpu')[:ns]
        sp = s['scat'][sel].to(dev)
        sv = _unpack(s['scat_vis'][sel], N, dev)[:, gi].T.float()
        sg = U(ns, lo=0.02, hi=0.25) * torch.sign(torch.randn(ns, device=dev))
        d1 = (sp - src).norm(dim=-1)
        wall = (s['gain'].mean(0) if s['gain'].shape[0] else torch.ones(8)).to(dev)
        for j0 in range(0, J, JC):
            js = slice(j0, min(j0 + JC, J))
            vs = sp[None] - pos[js, None]; d2 = vs.norm(dim=-1)
            am = sg[None] * sv[js] / (4 * math.pi * d1[None] * d2)
            val = am[..., None] * torch.cat([torch.ones_like(d2)[..., None], vs / d2[..., None]], -1)
            r = _frac_render((d1[None] + d2) / C * FS, val, 2 * K)
            spec[js] += torch.fft.rfft(r, n=2 * K, dim=1)[:, :F] * (masks * wall[:, None]).sum(0)[None, :, None]
    spec += _diffuse(s, pos, svals, K, F, dev, masks, n_arr)
    col = _rand_filter_resp(1, F, dev)[0]
    return torch.fft.irfft(spec * col[None, :, None], n=2 * K, dim=1)[:, :K], pos


def _diffuse(s, pos, svals, K, F, dev, masks, n_arr):
    """scattered field: n_arr plane-wave arrivals; times ~ energy density of the mid-path envelope, directions ~ vMF
    around the energy-weighted mean direction of their time bin (500 Hz-4 kHz bands), band energies from the envelope,
    per-band level along the path from the interpolated envelopes."""
    J = pos.shape[0]
    H = s['rain'].to(dev).double(); Dm = s['rain_dir'].to(dev).double()      # [R, 8, nb], [R, 8, nb, 3]
    rs = s['rain_s'].to(dev).double(); nb = H.shape[-1]; bin_s = float(s['rain_bin'])
    x = svals.double().clamp(float(rs[0]), float(rs[-1]))
    i0 = (torch.searchsorted(rs, x, right=True) - 1).clamp(0, len(rs) - 2)
    w = ((x - rs[i0]) / (rs[i0 + 1] - rs[i0]).clamp_min(1e-6))[:, None, None]
    tot = (H[i0] * (1 - w) + H[i0 + 1] * w).sum(-1)                          # [J, 8]
    jr = J // 2
    xr = x[jr]; r0 = int((torch.searchsorted(rs, xr, right=True) - 1).clamp(0, len(rs) - 2)); wr = float((xr - rs[r0]) / (rs[r0 + 1] - rs[r0]).clamp_min(1e-6))
    Hr = H[r0] * (1 - wr) + H[r0 + 1] * wr                                    # [8, nb]
    Dr = Dm[r0] * (1 - wr) + Dm[r0 + 1] * wr                                  # [8, nb, 3]
    tr = Hr.sum(0)
    if float(tr.sum()) <= 0:
        return torch.zeros(J, F, 4, dtype=torch.complex64, device=dev)
    # mean direction per bin, smoothed over 3 bins (Monte-Carlo noise)
    m = Dr[3:7].sum(0); e = Hr[3:7].sum(0)
    ker = torch.ones(1, 1, 3, device=dev, dtype=torch.float64)
    m = torch.nn.functional.conv1d(m.T[:, None], ker, padding=1)[:, 0].T
    e = torch.nn.functional.conv1d(e[None, None], ker, padding=1)[0, 0]
    mv = m / e.clamp_min(1e-30)[:, None]; Rl = mv.norm(dim=-1).clamp(0, 1)
    mu = mv / Rl.clamp_min(1e-9)[:, None]
    cen = (torch.arange(nb, device=dev, dtype=torch.float64) + 0.5) * bin_s
    uu = torch.rand(n_arr, device=dev, dtype=torch.float64)
    ti = torch.searchsorted(tr.cumsum(0) / tr.sum(), uu).clamp(0, nb - 1)
    tv = cen[ti] + (torch.rand(n_arr, device=dev, dtype=torch.float64) - 0.5) * bin_s
    p = tr[ti] / tr.sum()
    A = (Hr[:, ti] / (len(ti) * p[None])).clamp_min(0).sqrt()
    A = A * (torch.randint(0, 2, (n_arr,), device=dev) * 2 - 1)[None]
    u = _vmf(mu[ti], _kappa(Rl[ti]))                                          # [n_arr, 3] towards the arrival
    delay = tv[None] * FS - ((pos.double() - pos[jr][None].double()) @ u.T) / C * FS
    gl = (tot / tot[jr][None].clamp_min(1e-30)).clamp(0.01, 100).sqrt()
    Y = torch.cat([torch.ones(n_arr, 1, device=dev, dtype=torch.float64), u], -1).float()   # [n_arr, 4]
    out = torch.zeros(J, F, 4, dtype=torch.complex64, device=dev)
    for j0 in range(0, J, 256):
        js = slice(j0, min(j0 + 256, J))
        for b in range(8):
            amp = (A[b][None] * gl[js, b:b + 1]).float()
            r = _frac_render(delay[js].float(), amp[..., None] * Y[None], 2 * K)
            out[js] += torch.fft.rfft(r, n=2 * K, dim=1)[:, :F] * masks[b][None, :, None]
    return out
