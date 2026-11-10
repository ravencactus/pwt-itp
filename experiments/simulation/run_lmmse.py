"""First-order diffuse-field LMMSE transport prior ('LMMSE1', pwtitp/lmmse.py) over its hyperparameter grid on the simulated rooms.
Started from the random-walk estimate, as PWT. Records LMMSE1 [lam, rw].
usage: run_lmmse.py v kind out.jsonl [seeds]   (env SNRS as in run_baselines.py)"""
import _common
import os, sys, json, time, torch
from pwtitp.model import Fwd, Solver
from pwtitp.lmmse import lmmse_solve
from pwtitp.pipeline import rw_init, IT
from pwtitp.metrics import ndb, lo, band_ndb
from pwtitp.sim import scenario, add_noise

SNRS = [None, 20., 10., 0.] if 'SNRS' not in os.environ else [None if t == 'inf' else float(t) for t in os.environ['SNRS'].split(',')]
v = float(sys.argv[1]); kind = sys.argv[2]; fo = open(sys.argv[3], 'a')
seeds = [int(t) for t in sys.argv[4].split(',')] if len(sys.argv) > 4 else [1, 2, 3, 4]
for sd in seeds:
    s0 = scenario(100*sd + 7, v, kind, K=1600)
    for snr in SNRS:
        t0 = time.time(); s = s0; y = add_noise(s['y'], snr, sd)
        F = Fwd(s['x'], s['K'], s['H'], s['J']); S = Solver(F, y); H = s['H']
        htn = s['ht'].double().cpu().numpy(); htl = lo(htn)[:, :64]
        h0 = rw_init(S, H)
        for lam in [0.3, 1., 3., 10., 30., 100.]:
            for rw in [0.03, 0.1, 0.3]:
                e = lmmse_solve(S, s['dp'], s['fs'], s['c'], H*lam, rw=H*rw, h0=h0, iters=IT).double().cpu().numpy()
                fo.write(json.dumps(dict(v=v, kind=kind, snr=snr, seed=sd, method='LMMSE1', param=[lam, rw], full=ndb(e, htn),
                                         low=ndb(lo(e)[:, :64], htl), bands=band_ndb(e, htn))) + '\n'); fo.flush()
        del F, S; torch.cuda.empty_cache()
        print(v, kind, snr, sd, f'{time.time()-t0:.0f}s', flush=True)
