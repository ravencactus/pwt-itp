"""Moving first-order Ambisonic (SN3D B-format) recordings in random shoebox rooms (with env PWTITP_HOUSES: in furnished
3D-FRONT houses, see pwtitp/mesh/scenario.py).

Image sources to 100 ms with a broadband reflection coefficient per wall, a low-pass filter per reflection order,
gain perturbations above the second order, source directivity, up to 60 point scatterers and a random transducer
coloration; an isotropic diffuse tail (T60 0.2-1.2 s, to 0.5 s) continues the response and acts as unmodelled
interference.  The response is rendered every G = 16 samples along a straight path at constant speed, so it changes
continuously rather than at the knots of the estimators.
Speech excitation: LibriSpeech test-clean utterances listed in data/speech_manifest.json (env LIBRISPEECH = folder
that contains test-clean/).
"""
import os, json, math, numpy as np, torch
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEECH = None


def speech_pool():
    """Concatenated LibriSpeech test-clean utterances, 60 segments of 60 s, each normalized to unit variance."""
    global _SPEECH
    if _SPEECH is None:
        cache = os.path.join(_HERE, '..', 'data', 'speech_pool.npy')
        if os.path.exists(cache): _SPEECH = np.load(cache)
        else:
            man = json.load(open(os.path.join(_HERE, '..', 'data', 'speech_manifest.json')))
            root = os.environ.get('LIBRISPEECH', os.path.join(_HERE, '..', 'data', 'LibriSpeech'))
            out = []
            for u in man['utterances']:
                x = np.concatenate([sf.read(os.path.join(root, f), dtype='float32')[0] for f in u['files']])[:u['n']]
                out.append(x/(x.std() + 1e-9))
            _SPEECH = np.concatenate(out); np.save(cache, _SPEECH)
    return _SPEECH


def speech_test(n, rng):
    sp = speech_pool(); s = int(rng.integers(0, len(sp) - n)); x = sp[s:s+n]
    return x/(x.std() + 1e-9)


def _images(room, src, max_order, device):
    """all image sources up to max_order: positions [M,3], per-wall reflection counts [M,6], order [M]."""
    r = torch.arange(-max_order, max_order + 1, device=device)
    nx, ny, nz = torch.meshgrid(r, r, r, indexing='ij')
    n = torch.stack([nx, ny, nz], -1).reshape(-1, 3)
    out_p, out_c = [], []
    for q in torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], device=device):
        # coordinate: x_img = 2 n L + (1-2q) x_s
        p = 2 * n * room + (1 - 2 * q) * src
        # wall hit counts: low wall |n - q|, high wall |n|
        lo = (n - q).abs(); hi = n.abs()
        out_p.append(p); out_c.append(torch.cat([lo, hi], -1))
    p = torch.cat(out_p); c = torch.cat(out_c)
    order = c.sum(-1)
    keep = order <= max_order
    return p[keep].float(), c[keep].float(), order[keep]


def _rand_filter_resp(B, F, device, n_bq=3):
    """random smooth magnitude response on rfft grid [B, F] (loudspeaker/mic colouration, dB-domain smooth curve)."""
    f = torch.linspace(0, 1, F, device=device)
    # sum of a few random gaussian bumps in log-freq + highpass
    lf = torch.log10(f * 8000 + 20)
    g = torch.zeros(B, F, device=device)
    for _ in range(n_bq):
        c = torch.empty(B, 1, device=device).uniform_(math.log10(100), math.log10(7000))
        wdt = torch.empty(B, 1, device=device).uniform_(0.1, 0.6)
        a = torch.empty(B, 1, device=device).uniform_(-6, 6)
        g = g + a * torch.exp(-0.5 * ((lf - c) / wdt) ** 2)
    fc = torch.empty(B, 1, device=device).uniform_(40, 150)
    hp = 1 / (1 + (fc / (f * 8000 + 1e-3)) ** 4)
    return (10 ** (g / 20)) * hp.sqrt()


def _render(delay, val, L, taps=16):
    """delay (b,M), val (b,M,4) -> (b,L,4) windowed-sinc fractional delay."""
    b, M = delay.shape; i0 = delay.floor().long(); f = delay - i0
    t = torch.arange(-taps + 1, taps + 1, device=delay.device)
    idx = i0[..., None] + t; x = t.float() - f[..., None]
    w = torch.sinc(x)*(0.5 + 0.5*torch.cos(math.pi*x/taps)).clamp(min=0)
    ok = (idx >= 0) & (idx < L); idx = torch.where(ok, idx, L)
    out = torch.zeros(b, (L + 1), 4, device=delay.device)
    base = (torch.arange(b, device=delay.device)*(L + 1))[:, None, None]
    flat = (idx + base).reshape(-1); vals = (w*ok)[..., None]*val[:, :, None, :]
    out.view(-1, 4).index_add_(0, flat, vals.reshape(-1, 4))
    return out[:, :L]


def scenario(seed, v, kind='wn', T=None, fs=16000, K=1024, Kr=1600, Kt=8000, H=512, G=16, c=343., dev='cuda',
             dmin=0.7, dmax=1e9):   # closest source distance to the path in [dmin, dmax]
    if os.environ.get('PWTITP_HOUSES'):     # furnished 3D-FRONT houses (pwtitp/mesh); near-source paths pass 0.25-0.5 m from the source
        from .mesh.scenario import scenario_mesh
        return scenario_mesh(seed, v, kind, T, fs, K, Kr, Kt, H, G, c, dev, geo='near' if dmax < 1 else 'far')
    if T is None: T = min(4.0, 3.2/v)          # path length <= 3.2 m (fits the rooms); T = 4 s for v <= 0.8
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    U = lambda *s, lo=0., hi=1.: torch.empty(*s, device=dev).uniform_(lo, hi)
    room = torch.tensor([rng.uniform(3, 10), rng.uniform(3, 9), rng.uniform(2.4, 4.5)], device=dev, dtype=torch.float32)
    src = room*torch.tensor(rng.uniform(0.1, 0.9, 3), device=dev, dtype=torch.float32)
    J = int(T*fs//H) + 1; N = (J - 1)*H; length = v*N/fs
    for _ in range(1000):
        p0 = room*torch.tensor(rng.uniform(0.08, 0.92, 3), device=dev, dtype=torch.float32)
        phi = rng.uniform(0, 2*np.pi); th = rng.uniform(-0.3, 0.3)
        d = torch.tensor([np.cos(phi)*np.cos(th), np.sin(phi)*np.cos(th), np.sin(th)], device=dev, dtype=torch.float32)
        line = p0[None] + d[None]*torch.linspace(0, length, 32, device=dev)[:, None]
        if bool(((line > 0.05*room) & (line < 0.95*room)).all()) and dmin < float((line - src).norm(dim=-1).min()) < dmax: break
    else: raise RuntimeError('no valid trajectory')
    tg = torch.arange(0, N + G, G, device=dev, dtype=torch.float32)/fs
    traj = p0[None] + v*tg[:, None]*d[None]                                         # (Tg,3)
    Tg = traj.shape[0]
    mo = min(int(math.ceil(c*Kr/fs/float(room.min()))) + 1, 12)
    ip, cnt, order = _images(room, src, mo, dev)
    keep = (traj[[0, -1]][:, None] - ip[None]).norm(dim=-1).min(0).values < c*Kr/fs + length + 1
    ip, cnt, order = ip[keep], cnt[keep], order[keep]
    beta = U(6, lo=0.55, hi=0.97); g = torch.exp((cnt*torch.log(beta)).sum(-1))
    g = g*(1 + 0.3*torch.randn_like(g)*(order > 2))
    sdir = torch.randn(3, device=dev); sdir /= sdir.norm(); a_dir = float(rng.uniform(0, 0.6))
    ns = int(rng.integers(0, 61)); sp = room*U(ns, 3, lo=0.02, hi=0.98)
    sg = U(ns, lo=0.02, hi=0.25)*torch.sign(torch.randn(ns, device=dev)); d1 = (sp - src).norm(dim=-1)
    F = Kr + 1; fgrid = torch.linspace(0, 1, F, device=dev)
    fc = float(rng.uniform(0.15, 0.9)); wall = 1/torch.sqrt(1 + (fgrid/fc)**2)**float(rng.uniform(0.3, 1.0))
    col = _rand_filter_resp(1, F, dev)[0]
    he = torch.zeros(Tg, Kr, 4, device=dev); omax = int(order.max())
    for i in range(0, Tg, 64):
        R = traj[i:i+64]; vec = ip[None] - R[:, None]; r = vec.norm(dim=-1); u = vec/r[..., None]   # towards image
        amp = g[None]*((1 - a_dir) + a_dir*(-u @ sdir))/(4*math.pi*r)
        val = amp[..., None]*torch.cat([torch.ones_like(r)[..., None], u], -1)
        spec = 0
        for o in range(omax + 1):
            m = order == o
            if bool(m.any()): spec = spec + torch.fft.rfft(_render(r[:, m]/c*fs, val[:, m], 2*Kr), dim=1)*(wall**o)[None, :, None]
        if ns:
            vs = sp[None] - R[:, None]; d2 = vs.norm(dim=-1)
            vv = (sg[None]/(4*math.pi*d1[None]*d2))[..., None]*torch.cat([torch.ones_like(d2)[..., None], vs/d2[..., None]], -1)
            spec = spec + torch.fft.rfft(_render((d1[None] + d2)/c*fs, vv, 2*Kr), dim=1)*wall[None, :, None]
        he[i:i+64] = torch.fft.irfft(spec*col[None, :, None], n=2*Kr, dim=1)[:, :Kr]
    # isotropic diffuse tail on lags [Kr, Kt): decorrelates over ~lambda(2 kHz) of travel; SN3D: X,Y,Z carry 1/3 of W energy
    t60 = float(rng.uniform(0.2, 1.2)); Lt = Kt - Kr
    env = torch.exp(-6.91*torch.arange(Lt, device=dev)/(t60*fs))
    e_end = float((he[:, Kr-200:, 0]**2).mean())
    pos = v*tg/(c/2000.); nA = int(pos[-1]) + 2
    anc = torch.randn(nA, Lt, 4, device=dev)*torch.tensor([1, 3**-.5, 3**-.5, 3**-.5], device=dev)
    i0 = pos.floor().long().clamp(max=nA - 2); w = (pos - i0)[:, None, None]*math.pi/2
    tail = (torch.cos(w)*anc[i0] + torch.sin(w)*anc[i0 + 1])*(env*math.sqrt(e_end))[None, :, None]
    hf = torch.cat([he, tail], 1)                                                    # (Tg,Kt,4)
    n_x = N + Kt - 1
    xx = speech_test(n_x, rng) if kind == 'speech' else rng.standard_normal(n_x).astype(np.float32)
    xx = torch.tensor(xx, device=dev); nb = N//G; y = torch.zeros(nb, G, 4, device=dev)
    for b0 in range(0, nb, 64):
        b1 = min(b0 + 64, nb)
        X = xx[b0*G:b1*G + Kt - 1].unfold(0, Kt, 1).flip(1).reshape(b1 - b0, G, Kt)
        y[b0:b1] = torch.bmm(X, hf[b0:b1])
    y = y.reshape(N, 4); x = xx[Kt - K:]                                            # estimator convention: len N+K-1
    ht = he[::H//G][:J, :K].contiguous(); pk = traj[::H//G][:J]; dp = pk[1:] - pk[:-1]
    return dict(x=x, y=y, ht=ht, dp=dp, J=J, K=K, H=H, fs=fs, c=c, traj=traj, pk=pk, src=src.cpu().numpy(), G=G,
                t60=t60, room=room.tolist(), ns=ns)

def add_noise(y, snr, seed):
    if snr is None: return y
    g = torch.Generator(device=y.device).manual_seed(1000 + seed)
    return y + torch.randn(y.shape, generator=g, device=y.device)*torch.sqrt((y**2).mean()*10**(-snr/10))
