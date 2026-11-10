"""Scene bank for moving first-order Ambisonic recordings in furnished 3D-FRONT houses.

Sampler (grid -> fast-winding-number inside test -> clearance from the mesh): a source and smooth, irregular (curving,
slightly bobbing) array paths. The whole path must lie inside the house with `clear_path` metres to every surface, must
not cross geometry and must see the source directly at every point.
Scene = (source, path, image sources up to 50 reflections with band gains and their visibility along the path on a
1-mm grid, band energy envelopes and energy-weighted arrival directions of the scattered field up to 200 reflections at
points every 0.2 m along the path, candidate point scatterers with visibility, room statistics). Rendering at any speed
only interpolates along the stored path (render.py)."""
import math, os, glob
import numpy as np
import torch
from .geometry import MeshRoom, BANDS, AIR_DB_M, C, discover_tube, diffuse_rain

DL = 1e-3       # visibility grid along the path (m)
COARSE = 20     # coarse validation every COARSE*DL, refined where visibility changes
SPEC_ORDER = 50     # specular reflections (image sources)
DIFF_ORDER = 200    # diffuse reflections (diffuse rain)
RAIN_DS = 0.2       # diffuse envelopes every RAIN_DS m along the path
RAIN_BIN = 1e-3     # envelope time bin (s)
T_MAX = 0.105       # simulated time span (s): K = 1600 samples at 16 kHz + margin


# ------------------------------------------------------------------------------------------------------------ sampler
def free_points(room, step=0.2, z=(0.5, 2.0), clear=0.3, w_thr=0.4):
    lo, hi = room.bounds
    ax = [np.arange(lo[0] + step / 2, hi[0], step), np.arange(lo[1] + step / 2, hi[1], step),
          np.arange(max(lo[2] + z[0], lo[2]), min(lo[2] + z[1], hi[2]), step)]
    P = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    P = P[np.abs(room.winding(P)) > w_thr]
    if len(P):
        P = P[room.clearance(P) > clear]
    return P


def random_curves(rng, n, L, kappa_max=2.0, z_amp=0.12):
    """n smooth, irregular microphone trajectories of arc length L (m), sampled every DL: [n, N, 3], starting at 0.
    Heading: curvature kappa(s) = Gaussian-filtered noise (correlation length 0.1-0.5 m) with a random std in
    [0, kappa_max] 1/m, clipped to |kappa| <= 4 1/m (radius >= 0.25 m), so paths range from nearly straight to winding.
    Elevation angle: small offset + sinusoidal bob (amplitude <= z_amp rad, wavelength 0.4-1.5 m). The tangent is unit
    length, so the grid index is exact arc length (speed along the path = v)."""
    N = int(round(L / DL)) + 1
    s = np.arange(N) * DL
    nc = int(math.ceil(L / 0.05)) + 1
    out = np.empty((n, N, 3))
    for i in range(n):
        ell = rng.uniform(0.1, 0.5) / 0.05
        w = rng.standard_normal(nc + 40)
        k = np.exp(-0.5 * (np.arange(-int(3 * ell) - 1, int(3 * ell) + 2) / ell) ** 2)
        kc = np.convolve(w, k, mode='same')[20:20 + nc]
        kc = kc / (kc.std() + 1e-9) * rng.uniform(0, kappa_max)
        kap = np.clip(np.interp(s, np.arange(nc) * 0.05, kc), -4, 4)
        th = rng.uniform(0, 2 * np.pi) + np.concatenate([[0], np.cumsum(kap[:-1]) * DL])
        lam = rng.uniform(0.4, 1.5)
        ph = rng.uniform(-0.03, 0.03) + rng.uniform(0, z_amp) * np.sin(2 * np.pi * s / lam + rng.uniform(0, 2 * np.pi))
        tan = np.stack([np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)], 1)
        out[i, 0] = 0
        out[i, 1:] = np.cumsum((tan[1:] + tan[:-1]) / 2, 0) * DL
    return out


def sample_paths(room, src, L, rng, n_want, z=(0.8, 1.8), clear_path=0.25, min_src=0.5, step=0.05, n_cand=128, rounds=16,
                 free=None, w_thr=0.4, max_src=None, start_r=None):
    """curved trajectories of arc length L (m) from free start points (see random_curves). The whole path (checked every
    `step` m) must be inside the house, keep `clear_path` m from every surface, not cross geometry, see the source
    directly, and its closest distance to the source must lie in (min_src, max_src). Returns a list of [N, 3] arrays
    (DL grid)."""
    lo = room.bounds[0]
    st = free[(free[:, 2] >= lo[2] + z[0]) & (free[:, 2] <= lo[2] + z[1])]
    if start_r is not None:  # near-source paths: start within start_r m of the source
        st = st[np.linalg.norm(st - src, axis=1) < start_r]
    if len(st) == 0:
        return []
    out = []
    sub = int(round(step / DL))
    for _ in range(rounds):
        p0 = st[rng.integers(len(st), size=n_cand)] + rng.uniform(-0.1, 0.1, (n_cand, 3)) * np.array([1, 1, 0.5])
        curves = random_curves(rng, n_cand, L)
        ci = np.r_[np.arange(0, curves.shape[1], sub), curves.shape[1] - 1] if (curves.shape[1] - 1) % sub else np.arange(0, curves.shape[1], sub)
        pts = p0[:, None] + curves[:, ci]  # [n, ns, 3]
        ns = pts.shape[1]
        ok = (pts[..., 2] > lo[2] + 0.5).all(1) & (pts[..., 2] < lo[2] + 2.2).all(1)
        dmin = np.linalg.norm(pts - src, axis=-1).min(1)
        ok &= dmin > min_src
        if max_src is not None:
            ok &= dmin < max_src
        idx = np.nonzero(ok)[0]
        if len(idx) == 0:
            continue
        good = (np.abs(room.winding(pts[idx].reshape(-1, 3))) > w_thr).reshape(len(idx), ns).all(1)
        idx = idx[good]
        if len(idx) == 0:
            continue
        good = (room.clearance(pts[idx].reshape(-1, 3)) > clear_path).reshape(len(idx), ns).all(1)
        idx = idx[good]
        if len(idx) == 0:
            continue
        dev = room.dev
        P = torch.as_tensor(pts[idx], dtype=torch.float64, device=dev)  # [n, ns, 3]
        blk = room.segment_blocked(P[:, :-1].reshape(-1, 3), P[:, 1:].reshape(-1, 3)).view(len(idx), ns - 1).any(1)
        S = torch.as_tensor(src, dtype=torch.float64, device=dev).expand(P.shape[0] * ns, 3)
        los = ~room.segment_blocked(P.reshape(-1, 3), S).view(len(idx), ns).any(1)
        good = (~blk & los).cpu().numpy()
        for i in idx[good]:
            out.append(p0[i] + curves[i])
            if len(out) >= n_want:
                return out
    return out


# ------------------------------------------------------------------------------------------------------------ scenes
@torch.no_grad()
def path_visibility(room, src, chain, nlen, imgs, path):
    """visibility of every image (chains of mixed order, padded: chain [C, NMAX], nlen [C], imgs [C, NMAX, 3]) along the
    path [N, 3] (DL grid): coarse grid every COARSE samples + every grid point inside coarse intervals where the
    visibility changes. [C, N] bool"""
    N = path.shape[0]
    dev = room.dev
    sgrid = torch.arange(0, N, COARSE, device=dev)
    if sgrid[-1] != N - 1:
        sgrid = torch.cat([sgrid, torch.tensor([N - 1], device=dev)])
    pth = torch.as_tensor(path, dtype=torch.float64, device=dev)
    Cn, G = chain.shape[0], len(sgrid)
    pc = torch.arange(Cn, device=dev).repeat_interleave(G); pp = sgrid.repeat(Cn)
    vc = room.validate_padded(src, chain, nlen, imgs, pth, pc, pp).view(Cn, G)
    seg = (torch.searchsorted(sgrid, torch.arange(N, device=dev), right=True) - 1).clamp(max=G - 1)
    vis = vc[:, seg]
    ch_c, ch_i = torch.nonzero(vc[:, 1:] != vc[:, :-1], as_tuple=True)
    if len(ch_c):
        lo = sgrid[ch_i] + 1; cnt = sgrid[ch_i + 1] - lo
        m = cnt > 0
        lo, cnt, cc = lo[m], cnt[m], ch_c[m]
        if len(cc):
            rc = torch.repeat_interleave(cc, cnt)
            start = torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
            pts = torch.repeat_interleave(lo, cnt) + torch.arange(int(cnt.sum()), device=dev) - start
            vis[rc, pts] = room.validate_padded(src, chain, nlen, imgs, pth, rc, pts)
    return vis


def _pad_chains(room, src, chains, nmax):
    """list of [C_n, n] chains -> padded chain [C, nmax], nlen [C], images [C, nmax, 3]"""
    dev = room.dev
    Cs = sum(len(c) for c in chains)
    chain = torch.zeros(Cs, nmax, dtype=torch.long, device=dev); nlen = torch.zeros(Cs, dtype=torch.long, device=dev)
    imgs = torch.zeros(Cs, nmax, 3, dtype=torch.float64, device=dev)
    k = 0
    for n, c in enumerate(chains, start=1):
        if len(c):
            chain[k:k + len(c), :n] = c; nlen[k:k + len(c)] = n; imgs[k:k + len(c), :n] = room.images(src, c); k += len(c)
    return chain, nlen, imgs


@torch.no_grad()
def build_scenes(room, rng, n_src, n_far, n_near, L, far=(0.7, None), near=(0.25, 0.5), n_rays=400000, n_rain=20000,
                 n_scat=250, clear_src=0.3, max_escape=0.01):
    """n_src sources, each with n_far paths whose closest distance to the source is in `far` and up to n_near paths with
    the closest distance in `near` (all of arc length L). Per source: specular discovery up to SPEC_ORDER reflections (tube
    test against points every 0.1 m of all its paths, time span T_MAX), exact visibility of every candidate image along
    every path (sources whose rays escape the mesh more than max_escape are discarded); diffuse rain (DIFF_ORDER
    reflections, n_rain rays) with direction moments at points every RAIN_DS m along every path."""
    free = free_points(room, clear=clear_src)
    if len(free) < 10:
        return [], None
    scenes, st_all = [], None
    used = 0
    dev = room.dev
    for si in rng.permutation(len(free))[:max(6 * n_src, 30)]:
        if used >= n_src:
            break
        src = free[si] + rng.uniform(-0.1, 0.1, 3) * np.array([1, 1, 0.5])
        pf = sample_paths(room, src, L, rng, n_far, free=free, min_src=far[0], max_src=far[1])
        pn = sample_paths(room, src, L, rng, n_near, free=free, min_src=near[0], max_src=near[1], start_r=near[1] + L / 2,
                          rounds=32) if n_near else []
        if len(pf) < n_far:  # near-source paths are optional (not every source has free space around it)
            continue
        paths = pf + pn; kinds = ['far'] * len(pf) + ['near'] * len(pn)
        N = paths[0].shape[0]
        step = int(round(0.1 / DL))
        rpts = np.concatenate([p[::step] for p in paths])
        chains, st = discover_tube(room, src, rpts, n_rays=n_rays, max_order=SPEC_ORDER, t_max=T_MAX, tube=2.0,
                                   seed=int(rng.integers(1 << 30)))
        if st['escape'] > max_escape:  # the source sees a hole in the mesh: not a closed room
            continue
        used += 1
        st_all = st
        t60 = room.eyring_t60(st)
        chain, nlen, imgs = _pad_chains(room, src, chains, SPEC_ORDER)
        I = imgs[torch.arange(len(nlen), device=dev), (nlen - 1).clamp_min(0)]
        ri = np.unique(np.r_[np.arange(0, N, int(round(RAIN_DS / DL))), N - 1])
        rain_pts = np.concatenate([p[ri] for p in paths])
        H, _, D = diffuse_rain(room, src, rain_pts, n_rays=n_rain, max_depth=DIFF_ORDER, t_max=T_MAX, bin_s=RAIN_BIN,
                               seed=int(rng.integers(1 << 30)), moments=True)
        H = H.view(len(paths), len(ri), 8, -1); D = D.view(len(paths), len(ri), 8, -1, 3)
        for pi, (path, kind) in enumerate(zip(paths, kinds)):
            pth = torch.as_tensor(path, dtype=torch.float64, device=dev)
            near_ = (I[:, None] - pth[None, ::step]).norm(dim=-1).min(1).values < C * T_MAX
            idx = near_.nonzero().squeeze(1)
            vis = torch.zeros(0, N, dtype=torch.bool, device=dev)
            if len(idx):
                vis = path_visibility(room, src, chain[idx], nlen[idx], imgs[idx], path)
                k = vis.any(1)
                idx, vis = idx[k], vis[k]
            used_ = (torch.arange(SPEC_ORDER, device=dev)[None, :] < nlen[idx, None])[..., None]
            gains = torch.where(used_, room.prefl[chain[idx]], torch.ones((), dtype=torch.float64, device=dev)).prod(1)
            sp = free[rng.integers(len(free), size=4 * n_scat)] + rng.uniform(-0.1, 0.1, (4 * n_scat, 3))
            spt = torch.as_tensor(sp, dtype=torch.float64, device=dev)
            ok = ~room.segment_blocked(spt, torch.as_tensor(src, dtype=torch.float64, device=dev).expand(len(sp), 3))
            spt = spt[ok][:n_scat]
            sgrid = torch.arange(0, N, COARSE, device=dev)
            posc = pth[sgrid]
            S = spt.shape[0]
            sv = ~room.segment_blocked(spt[:, None].expand(S, len(sgrid), 3).reshape(-1, 3), posc[None].expand(S, -1, 3).reshape(-1, 3)).view(S, -1)
            seg = (torch.arange(N, device=dev) // COARSE).clamp(max=len(sgrid) - 1)
            sv = sv[:, seg]
            dsrc = np.linalg.norm(path - src, axis=1)
            scenes.append(dict(src=torch.tensor(src, dtype=torch.float32), path=torch.tensor(path, dtype=torch.float32), L=float(L),
                               kind=kind, dmin=float(dsrc.min()), s_min=float(np.argmin(dsrc) * DL),
                               img=I[idx].float().cpu(), gain=gains.float().cpu(), order=nlen[idx].to(torch.int8).cpu(),
                               vis=torch.from_numpy(np.packbits(vis.cpu().numpy(), axis=1)),
                               rain=H[pi].float().cpu(), rain_dir=D[pi].float().cpu(),
                               rain_s=torch.tensor(ri * DL, dtype=torch.float32), rain_bin=RAIN_BIN,
                               scat=spt.float().cpu(), scat_vis=torch.from_numpy(np.packbits(sv.cpu().numpy(), axis=1)),
                               t60=torch.tensor(t60, dtype=torch.float32), sbar=torch.tensor(st['s'], dtype=torch.float32),
                               abar=torch.tensor(st['a'], dtype=torch.float32), mfp=float(st['mfp'])))
    return scenes, st_all
