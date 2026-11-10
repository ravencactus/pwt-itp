"""Reference methods on the simulated rooms: RW-MAP, NLMS, KF, FDKF, KRR (isotropic, 'KRR-M') and KRR-D ('KRR-MD').
usage: run_baselines.py v kind out.jsonl [seeds]      (kind: wn | speech; env SNRS, e.g. SNRS=20 or SNRS=inf,20,10,0)"""
import _common
import os, sys, json, time, torch
from pwtitp.model import Fwd, Solver
from pwtitp.baselines import nlms, kf_alpha, fdkf, krr_lb
from pwtitp.metrics import ndb, lo, band_ndb, band_pow
from pwtitp.sim import scenario, add_noise
from pwtitp.pipeline import IT

SNRS = [None, 20., 10., 0.] if 'SNRS' not in os.environ else \
    [None if t == 'inf' else float(t) for t in os.environ['SNRS'].split(',')]
K = 1600

def main():
    v = float(sys.argv[1]); kind = sys.argv[2]; fo = open(sys.argv[3], 'a')
    seeds = [int(t) for t in sys.argv[4].split(',')] if len(sys.argv) > 4 else [1, 2, 3, 4]
    for sd in seeds:
        s0 = scenario(100*sd + 7, v, kind, K=K)
        fo.write(json.dumps(dict(v=v, kind=kind, seed=sd, method='XPSD', xband=band_pow(s0['x'].cpu().numpy()))) + '\n')
        for snr in SNRS:
            t0 = time.time(); s = s0; y = add_noise(s['y'], snr, sd)
            F = Fwd(s['x'], s['K'], s['H'], s['J']); S = Solver(F, y); H, J = s['H'], s['J']
            htn = s['ht'].double().cpu().numpy(); htl = lo(htn)[:, :64]
            tq = [j*H for j in range(J-1)] + [(J-1)*H - 1]
            def rec(m, p, est=None, low=None):
                d = dict(v=v, kind=kind, snr=snr, seed=sd, method=m, param=p)
                if est is not None:
                    e = est.double().cpu().numpy(); d['full'] = ndb(e, htn); d['low'] = ndb(lo(e)[:, :64], htl)
                    d['bands'] = band_ndb(e, htn)
                else: d['low'] = ndb(low, htl)
                fo.write(json.dumps(d) + '\n'); fo.flush()
            for f in [0.03, 0.1, 0.3, 1, 3, 10]: rec('RW-MAP', f, S.solve(H*f, iters=IT))
            for mu in [0.05, 0.1, 0.3, 0.6, 1.0]: rec('NLMS', mu, nlms(s['x'], y, K, tq, mu))
            for q in [-70, -60, -50, -40]:
                for r in [1e-2, 1e-1, 1.]: rec('KF-alpha', [q, r], kf_alpha(s['x'], y, K, tq, q, r))
            xs = s['x'][K-1:K-1+y.shape[0]]
            for A in [0.99, 0.999, 0.9999]:
                for sm in [False, True]:
                    rec('FDKF', [A, sm], torch.stack([fdkf(xs, y[:, c], K, tq, A=A, smooth=sm) for c in range(4)], -1))
            for b in [0., 1., 2., 4.]:
                for t, e in krr_lb(s, y, b, [-4, -3, -2, -1, 0]).items():
                    rec('KRR-M' if b == 0 else 'KRR-MD', [b, t], low=e)
            print(v, kind, snr, sd, f'{time.time()-t0:.0f}s', flush=True)

if __name__ == '__main__': main()
