"""Ablations of PWT+ITP on the development rooms (seeds 1-4; SNR inf/20/10): number of directions Q, transport
switched off (dp = 0), ITP components, Catmull-Rom instead of exact/windowed-sinc delays, exchanged bands,
crossover frequency.  usage: run_ablation.py v kind out.jsonl [seeds]"""
import _common
import sys, json, time, torch
from pwtitp.model import Fwd, Solver
from pwtitp.pwt import PlaneWaveTransport
from pwtitp.pipeline import rw_init, pwt, itp_, split
from pwtitp.metrics import ndb, lo, band_ndb
from pwtitp.sim import scenario, add_noise

v = float(sys.argv[1]); kind = sys.argv[2]; fo = open(sys.argv[3], 'a')
for sd in ([int(t) for t in sys.argv[4].split(',')] if len(sys.argv) > 4 else [1, 2, 3, 4]):
    s0 = scenario(100*sd + 7, v, kind, K=1600)
    for snr in [None, 20., 10.]:
        t0 = time.time(); s = s0; y = add_noise(s['y'], snr, sd)
        F = Fwd(s['x'], s['K'], s['H'], s['J']); S = Solver(F, y); H = s['H']; dp, fs, c = s['dp'], s['fs'], s['c']
        htn = s['ht'].double().cpu().numpy(); htl = lo(htn)[:, :64]
        def rec(m, p, e):
            e = e.double().cpu().numpy()
            fo.write(json.dumps(dict(v=v, kind=kind, snr=snr, seed=sd, method=m, param=p, full=ndb(e, htn), low=ndb(lo(e)[:, :64], htl), bands=band_ndb(e, htn))) + '\n'); fo.flush()
        h0 = rw_init(S, H); hp = {}
        for Q in [16, 32, 64, 128, 256]:
            pw = PlaneWaveTransport(F, dp, fs, c, Q)
            for lam in [1000, 3000]:
                hp[(Q, lam)] = pwt(S, F, dp, fs, c, H, lam, h0, pw=pw); rec('PWT', [Q, lam], hp[(Q, lam)])
            del pw; torch.cuda.empty_cache()
        hi = {li: itp_(S, dp, fs, c, H, li, hp[(128, 3000)]) for li in [0.3, 1.]}
        for li, e in hi.items(): rec('ITP', [li], e)
        for (Q, lam), el in hp.items():
            for li, eh in hi.items():
                for fc in [1000., 2000.]: rec('PWT+ITP', [Q, lam, li, fc], split(el, eh, fc))
        pw = PlaneWaveTransport(F, dp*0, fs, c, 128)                               # transport off in PWT
        for lam in [30, 100, 300, 1000, 3000]:
            e = pwt(S, F, dp, fs, c, H, lam, h0, pw=pw); rec('PWT-notau', [lam], e)
            for li, eh in hi.items(): rec('PWT-notau+ITP', [lam, li, 1000.], split(e, eh, 1000.))
        del pw; torch.cuda.empty_cache()
        for li in [0.3, 1.]:                                                       # ITP components
            rec('ITP-notau', [li], itp_(S, dp*0, fs, c, H, li, hp[(128, 3000)]))
            rec('ITP-noPW', [li], itp_(S, dp, fs, c, H, li, hp[(128, 3000)], mu=0.))
            rec('ITP-outer1', [li], itp_(S, dp, fs, c, H, li, hp[(128, 3000)], outer=1))
            rec('ITP-rwinit', [li], itp_(S, dp, fs, c, H, li, h0))
        pw = PlaneWaveTransport(F, dp, fs, c, 128, delay='cr')                     # Catmull-Rom delays (PWT and ITP)
        hc = {lam: pwt(S, F, dp, fs, c, H, lam, h0, pw=pw) for lam in [1000, 3000]}
        for lam, e in hc.items(): rec('PWT-cr', [lam], e)
        del pw; torch.cuda.empty_cache()
        for li in [0.3, 1.]:
            eh = itp_(S, dp, fs, c, H, li, hc[3000], kern='cr'); rec('ITP-cr', [li], eh)
            for lam, el in hc.items():
                for fc in [1000., 2000.]: rec('PWT+ITP-cr', [lam, li, fc], split(el, eh, fc))
        for lam in [1000, 3000]:
            for li, eh in hi.items():
                rec('swap', [lam, li, 1000.], split(eh, hp[(128, lam)], 1000.))
                for fc in [250., 500., 1000., 2000., 4000.]: rec('PWT+ITP-fc', [lam, li, fc], split(hp[(128, lam)], eh, fc))
        print(v, kind, snr, sd, f'{time.time()-t0:.0f}s', flush=True)
