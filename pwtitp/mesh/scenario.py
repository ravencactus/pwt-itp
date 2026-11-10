"""Moving first-order Ambisonic recordings in furnished 3D-FRONT houses (dataset: env PWTITP_HOUSES = folder of
per-house .npz files of the released dataset, or of .pt files written by experiments/simulation/build_bank.py).

Room i (seed i or 100 i + 7) is the i-th house in index.txt order (md5 order of the id without it); a 'far' scenario uses the first path of that house whose
closest distance to the source is at least 0.7 m, a 'near' scenario the i-th path (over the bank) that passes 0.25-0.5 m
from the source, cut so that the closest approach lies inside the recording. The early response (K_r = 100 ms) is
rendered every G = 16 samples along the curved path at constant speed (render.py); a diffuse tail to K_t = 0.5 s with the
band T60s of the room (Eyring, from the same rays) continues it and acts as unmodelled interference. The array keeps its
orientation (pure translation)."""
import os, glob, math, hashlib
import numpy as np, torch
from .render import render_foa, _band_masks

_BANK = {}


def load_house(f):
    """Scenes of one house: list of dicts (tensors and scalars), from a released .npz or a .pt of build_bank.py."""
    if f.endswith('.pt'):
        return torch.load(f, mmap=True, weights_only=False)
    z = np.load(f)
    sc = {}
    for key in z.files:
        i, k = key.split('/', 1); a = z[key]
        sc.setdefault(int(i[1:]), {})[k] = a.item() if a.ndim == 0 else torch.from_numpy(a)
    return [sc[i] for i in sorted(sc)]


def house_order(root):
    """House files in experiment order: the order of index.txt if present (released dataset), else md5 of the id."""
    files = glob.glob(os.path.join(root, '*.npz')) + glob.glob(os.path.join(root, '*.pt'))
    by_id = {os.path.basename(f).rsplit('.', 1)[0]: f for f in files}
    idx = os.path.join(root, 'index.txt')
    if os.path.exists(idx):
        return [by_id[r] for r in open(idx).read().split() if r in by_id]
    return [by_id[r] for r in sorted(by_id, key=lambda r: int(hashlib.md5(r.encode()).hexdigest(), 16))]


def _bank(root, n=None):
    if root not in _BANK:
        far, near = [], []
        for f in house_order(root)[:n or int(os.environ.get('PWTITP_N_HOUSES', 40))]:
            sc = load_house(f)
            for s in sc: s['house'] = os.path.basename(f).rsplit('.', 1)[0]
            far.append(next(s for s in sc if s['kind'] == 'far'))
            near += [s for s in sc if s['kind'] == 'near']
        _BANK[root] = (far, near)
    return _BANK[root]


def scenario_mesh(seed, v, kind='wn', T=None, fs=16000, K=1024, Kr=1600, Kt=8000, H=512, G=16, c=343., dev='cuda',
                  geo='far', bank=None):
    from ..sim import speech_test
    if T is None: T = min(4.0, 3.2/v)
    far, near = _bank(bank or os.environ['PWTITP_HOUSES'])
    i = (seed//100 if seed >= 100 else seed) - 1                         # experiments use seed = 100 i + 7
    s = (far if geo == 'far' else near)[i]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    J = int(T*fs//H) + 1; N = (J - 1)*H; length = v*N/fs
    assert length <= s['L'] + 1e-6, 'path too short'
    if geo == 'near':      # keep the closest approach inside the recording
        lo, hi = max(0., s['s_min'] - 0.9*length), min(s['L'] - length, s['s_min'] - 0.1*length)
        off = float(rng.uniform(lo, max(lo, hi)))
    else:
        off = float(rng.uniform(0, s['L'] - length))
    tg = torch.arange(0, N + G, G, device=dev, dtype=torch.float32)/fs
    sv = off + v*tg
    if geo == 'far' and rng.uniform() < 0.5: sv = s['L'] - sv          # either direction along the stored path
    he, traj = render_foa(s, sv, Kr, dev)                                 # (Tg, Kr, 4), (Tg, 3)
    # diffuse tail on lags [Kr, Kt): decorrelates over ~lambda(2 kHz) of travel; band T60 of the room; SN3D: X, Y, Z
    # carry 1/3 of the W energy
    Lt = Kt - Kr; t60 = s['t60'].to(dev).clamp(0.1, 3.0)
    tl = torch.arange(Lt, device=dev, dtype=torch.float32)/fs
    env = torch.exp(-6.91*tl[None]/t60[:, None])                          # (8, Lt)
    e_end = float((he[:, Kr-200:, 0]**2).mean())
    pos = torch.cat([torch.zeros(1, device=dev), (traj[1:] - traj[:-1]).norm(dim=-1).cumsum(0)])/(c/2000.)
    nA = int(pos[-1]) + 2
    anc = torch.randn(nA, Lt, 4, device=dev)*torch.tensor([1, 3**-.5, 3**-.5, 3**-.5], device=dev)
    m = _band_masks(Lt//2 + 1, dev)                                       # (8, Lt/2+1) partition of unity
    Af = torch.fft.rfft(anc, n=Lt, dim=1)
    anc = sum(torch.fft.irfft(Af*m[b][None, :, None], n=Lt, dim=1)*env[b][None, :, None] for b in range(8))
    anc = anc/anc[..., 0].pow(2)[:, :200].mean().sqrt().clamp_min(1e-12)
    i0 = pos.floor().long().clamp(max=nA - 2); w = (pos - i0)[:, None, None]*math.pi/2
    tail = (torch.cos(w)*anc[i0] + torch.sin(w)*anc[i0 + 1])*math.sqrt(e_end)
    hf = torch.cat([he, tail], 1)
    n_x = N + Kt - 1
    xx = speech_test(n_x, rng) if kind == 'speech' else rng.standard_normal(n_x).astype(np.float32)
    xx = torch.tensor(xx, device=dev); nb = N//G; y = torch.zeros(nb, G, 4, device=dev)
    for b0 in range(0, nb, 64):
        b1 = min(b0 + 64, nb)
        X = xx[b0*G:b1*G + Kt - 1].unfold(0, Kt, 1).flip(1).reshape(b1 - b0, G, Kt)
        y[b0:b1] = torch.bmm(X, hf[b0:b1])
    y = y.reshape(N, 4); x = xx[Kt - K:]
    ht = he[::H//G][:J, :K].contiguous(); pk = traj[::H//G][:J]; dp = pk[1:] - pk[:-1]
    d_src = (traj - s['src'].to(dev)).norm(dim=-1)
    return dict(x=x, y=y, ht=ht, dp=dp, J=J, K=K, H=H, fs=fs, c=c, traj=traj, pk=pk, src=s['src'].numpy(), G=G,
                t60=s['t60'].tolist(), room=s['house'], dmin=float(d_src.min()))
