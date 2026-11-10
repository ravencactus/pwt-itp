"""trajectoRIR database (Damiano et al.), configuration MC3, FOA array A1 (Rode NT-SF1), straight rail segment P1-P28.

Protocol: signals at 8 kHz; K = 2048 (256 ms), H = 256 (32 ms); recordings and reference responses zero-phase
high-passed at 150 Hz (cart ego-noise); array orientation from the direct sound of the static responses of both
loudspeakers; NMSE at the 28 static positions (at the instants the cart passes them), between 150 Hz and 2 kHz.
Env TRAJECTORIR = database root (folders audio/, meta/).
"""
import os, csv, numpy as np, torch, soundfile as sf
from scipy.signal import butter, sosfiltfilt, resample_poly
from .model import Fwd

ROOT = os.environ.get('TRAJECTORIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'trajectoRIR'))
CAPS = ['LF', 'RF', 'LB', 'RB']
# tetrahedral A -> B (LFU, RFD, LBD, RBU)
MB = 0.5*np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]], float)
POS = list(range(1, 29))
FS, K, H, FHP, C = 8000, 2048, 256, 150., 343.
EVB = [(150, 500), (500, 1000), (1000, 2000), (150, 2000)]
SOS = butter(4, FHP, 'hp', fs=FS, output='sos')


def rcsv(name): return list(csv.DictReader(open(f'{ROOT}/meta/{name}')))
def rd(p, fs=FS): x, f0 = sf.read(f'{ROOT}/{p}'); return resample_poly(x, 1, f0//fs)


def spk(ls):
    r = [r for r in rcsv('cart_pose.csv') if r['label'] == ls][0]
    return np.array([float(r[k]) for k in ('x_coordinate', 'y_coordinate', 'z_coordinate')])


def mic_pos(arr='A1'):
    return {int(r['position']): np.array([float(r[k]) for k in ('x_coord', 'y_coord', 'z_coord')])
            for r in rcsv('mic_coordinates.csv') if r['configuration'] == 'MC3' and r['microphone'] == arr+'LF'}


def stamps(ls, v, sig='WN'):
    r = [r for r in rcsv('mov_timestamps.csv') if (r['signal'], r['loudspeaker'], r['configuration'], r['velocity']) == (sig, ls, 'MC3', v)][0]
    return {p: float(r[str(p)]) for p in POS if r[str(p)] not in ('', 'NaN')}


def static_B(ls, arr, Ks, fs=16000):
    """Static responses at P1-P28 in B-format (channel frame), (P, Ks, 4)."""
    h = np.stack([np.stack([rd(f'audio/STAT/RIR/{ls}/MC3/P{p}/{arr}{c}.wav', fs)[:Ks] for c in CAPS], 1) for p in POS])
    return h @ MB.T


def calibrate_joint(arr, Ks, lss=('SR', 'SL')):
    """Orientation L = s Q (det Q = +1) with L h_xyz ~= u h_W on the direct sound of the static responses."""
    M = np.zeros((3, 3)); den = 0; mp = mic_pos(arr)
    for ls in lss:
        hB = static_B(ls, arr, Ks); s0 = spk(ls)
        for i, p in enumerate(POS):
            k0 = np.argmax(np.abs(hB[i, :, 0])); seg = hB[i, k0-1:k0+2]; u = s0 - mp[p]; u /= np.linalg.norm(u)
            M += np.outer(u, (seg[:, 0:1]*seg[:, 1:]).sum(0)); den += (seg[:, 1:]**2).sum()
    U, S, Vt = np.linalg.svd(M); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); Q = U @ D @ Vt
    return np.trace(Q.T @ M)/den*Q


def doa_err(L, ls, arr, Ks):
    """Mean direction error (degrees) of the direct sound after applying L."""
    hB = static_B(ls, arr, Ks); mp = mic_pos(arr); s0 = spk(ls); e = []
    for i, p in enumerate(POS):
        k0 = np.argmax(np.abs(hB[i, :, 0])); v = L @ (hB[i, k0-1:k0+2, 0:1]*hB[i, k0-1:k0+2, 1:]).sum(0)
        u = s0 - mp[p]; e.append(np.degrees(np.arccos(np.clip(v @ u/np.linalg.norm(v)/np.linalg.norm(u), -1, 1))))
    return float(np.mean(e)), float(np.max(e))


def to_room(hB, L): o = hB.copy(); o[..., 1:] = hB[..., 1:] @ L.T; return o


def build(ls, v, L, arr='A1', dev=os.environ.get('REALDEV', 'cuda')):
    ts = stamps(ls, v); mp = mic_pos(arr); P = sorted(ts)
    x = rd('audio/SRC/WN.wav'); yA = np.stack([rd(f'audio/MOV/WN/{ls}/MC3/{v}/{arr}{c}.wav') for c in CAPS], 1)
    n0 = int(round(ts[P[0]]*FS)); n1 = int(round(ts[P[-1]]*FS)); J = (n1-n0)//H + 2; N = (J-1)*H
    yB = sosfiltfilt(SOS, to_room(yA @ MB.T, L), axis=0)
    a = x[n0:n0+N].std(); xs = x[n0-K+1:n0+N]/a; y = np.ascontiguousarray(yB[n0:n0+N]/a)
    tk = (n0 + np.arange(J)*H)/FS; T = np.array([ts[p] for p in P]); Pm = np.stack([mp[p] for p in P])
    pos = lambda t: np.stack([np.interp(t, T, Pm[:, d]) for d in range(3)], 1)
    # discrete responses scale with 1/fs; kept in the 16 kHz convention of the database
    hs = np.stack([np.stack([rd(f'audio/STAT/RIR/{ls}/MC3/P{p}/{arr}{c}.wav')[:K] for c in CAPS], 1) for p in P]) @ MB.T*(16000/FS)
    hg = sosfiltfilt(SOS, np.pad(to_room(hs, L), ((0, 0), (256, 0), (0, 0))), axis=1)[:, 256:].copy()
    F = Fwd(torch.tensor(xs, dtype=torch.float32), K, H, J, dev)
    return dict(F=F, y=torch.tensor(y, dtype=torch.float32, device=dev), dp=torch.tensor(np.diff(pos(tk), axis=0), dtype=torch.float32, device=dev),
                n=np.array([ts[p]*FS - n0 for p in P]), hg=hg, xs=xs, traj1k=pos((n0 + np.arange(0, N, FS//1000))/FS), tq=pos(T),
                src=spk(ls), J=J, N=N)


def bnd(e, t, fs=FS):
    f = np.fft.rfftfreq(e.shape[1], 1/fs); E = np.fft.rfft(e - t, axis=1); T = np.fft.rfft(t, axis=1)
    return {f'{a}-{b}': float(10*np.log10((np.abs(E[:, (f >= a) & (f < b)])**2).sum()/(np.abs(T[:, (f >= a) & (f < b)])**2).sum())) for a, b in EVB}


def lo1k(h): return 8*resample_poly(np.asarray(h, np.float64), 1, 8, axis=1)       # (P, K, 4) at 8 kHz -> 1 kHz


def low_ndb(e1k, t1k, Lw):                                                       # 150-500 Hz on the first Lw ms
    f = np.fft.rfftfreq(Lw, 1/1000); m = f >= 150
    E = np.fft.rfft(e1k[:, :Lw] - t1k[:, :Lw], axis=1); T = np.fft.rfft(t1k[:, :Lw], axis=1)
    return float(10*np.log10((np.abs(E[:, m])**2).sum()/(np.abs(T[:, m])**2).sum()))
