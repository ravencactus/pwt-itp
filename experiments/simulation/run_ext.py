"""Extension of the run_main.py grid: PWT weights lam in {1e5, 3e5} (the original grid ends at 3e4), and ITP with a single
outer iteration (outer = 1, intensity field taken once from the PWT estimate), with and without the plane-wave coupling.
Records: PWT [lam] for the new lam; ITP [li, rw, mu, 1]; PWT+ITP [lam, li, rw, mu, 1, fc] for every lam of the extended grid.
usage: run_ext.py v kind out.jsonl [seeds]   (env SNRS as in run_baselines.py)"""
import _common
import os, sys, json, time, torch
from pwtitp.model import Fwd, Solver
from pwtitp.pwt import PlaneWaveTransport
from pwtitp.pipeline import rw_init, pwt, itp_, split
from pwtitp.metrics import ndb, lo, band_ndb
from pwtitp.sim import scenario, add_noise

SNRS = [None, 20., 10., 0.] if 'SNRS' not in os.environ else [None if t == 'inf' else float(t) for t in os.environ['SNRS'].split(',')]
v = float(sys.argv[1]); kind = sys.argv[2]; fo = open(sys.argv[3], 'a')
seeds = [int(t) for t in sys.argv[4].split(',')] if len(sys.argv) > 4 else [1, 2, 3, 4]
OLD, NEW = [300, 1000, 3000, 10000, 30000], [100000, 300000]
for sd in seeds:
    s0 = scenario(100*sd + 7, v, kind, K=1600)
    for snr in SNRS:
        t0 = time.time(); s = s0; y = add_noise(s['y'], snr, sd)
        F = Fwd(s['x'], s['K'], s['H'], s['J']); S = Solver(F, y); H = s['H']; a = (s['dp'], s['fs'], s['c'])
        htn = s['ht'].double().cpu().numpy(); htl = lo(htn)[:, :64]
        def rec(m, p, e):
            e = e.double().cpu().numpy()
            fo.write(json.dumps(dict(v=v, kind=kind, snr=snr, seed=sd, method=m, param=p, full=ndb(e, htn), low=ndb(lo(e)[:, :64], htl), bands=band_ndb(e, htn))) + '\n'); fo.flush()
        h0 = rw_init(S, H); pw = PlaneWaveTransport(F, *a, 128); hp = {}
        for lam in OLD + NEW:
            hp[lam] = pwt(S, F, *a, H, lam, h0, pw=pw)
            if lam in NEW: rec('PWT', [lam], hp[lam])
        del pw; torch.cuda.empty_cache()
        for mu in [0.3, 0.]:
            for li in [0.3, 1.]:
                for rw in [0., 0.03, 0.1]:
                    e = itp_(S, *a, H, li, hp[3000], rw=rw, mu=mu, outer=1); rec('ITP', [li, rw, mu, 1], e)
                    for lam, el in hp.items():
                        for fc in [1000., 2000., 4000.]: rec('PWT+ITP', [lam, li, rw, mu, 1, fc], split(el, e, fc))
        print(v, kind, snr, sd, f'{time.time()-t0:.0f}s', flush=True)
