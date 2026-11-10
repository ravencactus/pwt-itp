"""trajectoRIR protocol diagnostics: (i) direct-sound direction error with SR-only vs joint (SR+SL) orientation
calibration; (ii) per-band residual of the recording when the interpolated static responses are used as the moving
response, compared with the random-walk MAP estimate.  usage: real_diagnostics.py out.jsonl"""
import _common
import sys, json, numpy as np, torch
from scipy.signal import butter, sosfiltfilt
from pwtitp.model import Solver
from pwtitp.pipeline import IT
from pwtitp.trajectorir import build, calibrate_joint, doa_err, EVB, FS

fo = open(sys.argv[1], 'w')
L1 = calibrate_joint('A1', 4096, ('SR',)); L2 = calibrate_joint('A1', 4096)
fo.write(json.dumps(dict(what='doa', sr_only=[doa_err(L1, 'SR', 'A1', 4096), doa_err(L1, 'SL', 'A1', 4096)],
                         joint=[doa_err(L2, 'SR', 'A1', 4096), doa_err(L2, 'SL', 'A1', 4096)])) + '\n'); fo.flush()
def resid(B, hk):
    F = B['F']; yy = B['y'].reshape(F.J-1, F.H, -1); r = (F.A(hk) - yy).reshape(-1, 4).cpu().numpy(); y = yy.reshape(-1, 4).cpu().numpy()
    out = {}
    for a, b in EVB:
        sos = butter(4, [a, b], 'bp', fs=FS, output='sos'); rb = sosfiltfilt(sos, r, axis=0); yb = sosfiltfilt(sos, y, axis=0)
        out[f'{a}-{b}'] = float(10*np.log10((rb**2).sum()/(yb**2).sum()))
    return out
for v in ['V1', 'V2', 'V3']:
    for ls in ['SR', 'SL']:
        B = build(ls, v, L2); F, n = B['F'], B['n']; hg = torch.tensor(B['hg'], dtype=torch.float32, device=B['y'].device); hk = []
        for j in range(F.J):
            t = j*F.H; i = np.clip(np.searchsorted(n, t), 1, len(n)-1); w = np.clip((t - n[i-1])/(n[i]-n[i-1]), 0, 1)
            hk.append((1-w)*hg[i-1] + w*hg[i])
        S = Solver(F, B['y'])
        fo.write(json.dumps(dict(what='resid', v=v, ls=ls, gt=resid(B, torch.stack(hk)), rw=resid(B, S.solve(F.H*1., iters=IT)))) + '\n'); fo.flush()
        print(v, ls, flush=True)
