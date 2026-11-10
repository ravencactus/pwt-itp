"""Plane-wave vs exact spherical delay in PWT and ITP, near-source (closest distance 0.25-0.5 m) vs default geometry.
usage: run_near.py v kind geo out.jsonl    (geo: near | far, the latter with the default geometry)"""
import _common
import sys, json, time, torch
from pwtitp.model import Fwd, Solver
from pwtitp.pipeline import rw_init, pwt, itp_, split
from pwtitp.baselines import krr_lb
from pwtitp.metrics import ndb, lo, band_ndb
from pwtitp.sim import scenario, add_noise

v = float(sys.argv[1]); kind = sys.argv[2]; geo = sys.argv[3]; fo = open(sys.argv[4], 'a')
dd = dict(dmin=0.25, dmax=0.5) if geo == 'near' else {}
for sd in [1, 2, 3, 4]:
    s0 = scenario(100*sd + 7, v, kind, K=1600, **dd)
    for snr in [None, 20.]:
        t0 = time.time(); s = s0; y = add_noise(s['y'], snr, sd)
        F = Fwd(s['x'], s['K'], s['H'], s['J']); S = Solver(F, y); H = s['H']; dp, fs, c = s['dp'], s['fs'], s['c']
        htn = s['ht'].double().cpu().numpy(); htl = lo(htn)[:, :64]
        def rec(m, p, e=None, low=None):
            d = dict(v=v, kind=kind, geo=geo, snr=snr, seed=sd, method=m, param=p)
            if e is not None: e = e.double().cpu().numpy(); d.update(full=ndb(e, htn), low=ndb(lo(e)[:, :64], htl), bands=band_ndb(e, htn))
            else: d['low'] = ndb(low, htl)
            fo.write(json.dumps(d) + '\n'); fo.flush()
        for f in [0.1, 0.3, 1, 3, 10]: rec('RW-MAP', f, S.solve(H*f, iters=400))
        for t, e in krr_lb(s, y, 1., [-4, -3, -2, -1, 0]).items(): rec('KRR-MD', [1., t], low=e)
        h0 = rw_init(S, H)
        for sph in [False, True]:
            tg = 'sph' if sph else 'pw'; hp = {lam: pwt(S, F, dp, fs, c, H, lam, h0, sph=sph) for lam in [1000, 3000]}
            for lam, e in hp.items(): rec('PWT-' + tg, [lam], e)
            torch.cuda.empty_cache()
            for li in [0.3, 1.]:
                e = itp_(S, dp, fs, c, H, li, hp[3000], sph=sph); rec('ITP-' + tg, [li], e)
                for lam, el in hp.items():
                    for fc in [1000., 2000., 4000.]: rec('PWT+ITP-' + tg, [lam, li, fc], split(el, e, fc))
        print(v, kind, geo, snr, sd, f'{time.time()-t0:.0f}s', flush=True)
