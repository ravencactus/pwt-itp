"""trajectoRIR (MC3/A1, white noise, P1-P28): all methods over their grids on both loudspeakers
(tuned on SR, reported on SL in paper/results.py).  usage: run_real.py V1|V2|V3 out.jsonl"""
import _common
import os, sys, json, time, numpy as np, torch
from scipy.signal import resample_poly
from pwtitp.model import Solver
from pwtitp.pwt import PlaneWaveTransport
from pwtitp.pipeline import rw_init, pwt, itp_, split, IT
from pwtitp.baselines.krr import KRR, F_fwd, phi_matrix
from pwtitp.trajectorir import build, calibrate_joint, bnd, lo1k, low_ndb, FS, K, H, C

DEV = os.environ.get('REALDEV', 'cuda')

def krr(B, beta, Lk, ts):
    xt = B['xs'][K-1:]; x1 = resample_poly(xt, 1, FS//1000); y1 = resample_poly(B['y'].cpu().numpy().astype(np.float64), 1, FS//1000, axis=0)
    n_idx = np.arange(Lk, min(len(x1), len(y1))); R = B['traj1k'][n_idx]; Rq = B['tq']
    sx = x1.std(); Phi = F_fwd(phi_matrix(x1/sx, n_idx, Lk))
    eta = Rq.mean(0) - B['src']; eta /= np.linalg.norm(eta)
    kr = KRR(L=Lk, fs=1000., c=C, beta=beta, eta=eta, s_eta=-1, dev=DEV); Km = kr.kernel_matrix(R, Phi)
    out = {}
    for t in ts:
        try: out[t] = np.stack([kr.estimate(kr.solve(Km, y1[n_idx, c], Lk*10.**t), R, Phi, Rq)/sx for c in range(4)], -1)
        except RuntimeError: pass                                               # Cholesky failure at too small lambda
    return out

def main():
    v = sys.argv[1]; fo = open(sys.argv[2], 'a'); L = calibrate_joint('A1', 4096)
    for ls in ['SR', 'SL']:
        t0 = time.time(); B = build(ls, v, L); F = B['F']; S = Solver(F, B['y']); hg = B['hg']; g1k = lo1k(hg)
        a = (B['dp'], FS, C)
        def rec(m, p, h=None, e1k=None):
            d = dict(v=v, ls=ls, method=m, param=p)
            if h is not None:
                e = np.stack([F.at_time(h, n).double().cpu().numpy() for n in B['n']]); d.update(bnd(e, hg)); e1k = lo1k(e)
            d['low100'] = low_ndb(e1k, g1k, 100)
            if e1k.shape[1] >= 256: d['low256'] = low_ndb(e1k, g1k, 256)
            fo.write(json.dumps(d) + '\n'); fo.flush()
        for f in [0.03, 0.1, 0.3, 1, 3, 10]: rec('RW-MAP', f, S.solve(H*f, iters=IT))
        for Lk in [100, 256]:
            for b in [0., 1.]:
                for t, e in krr(B, b, Lk, [-2, -1, 0, 1, 2, 3]).items():
                    rec(('KRR-M' if b == 0 else 'KRR-MD') + f'-L{Lk}', [b, t], e1k=e)
        h0 = rw_init(S, H); pw = PlaneWaveTransport(F, *a, 128); hp = {}
        for lam in [300, 1000, 3000, 10000, 30000]: hp[lam] = pwt(S, F, *a, H, lam, h0, pw=pw); rec('PWT', [lam], hp[lam])
        del pw; torch.cuda.empty_cache()
        for li in [0.1, 0.3, 1., 3.]:
            for rw in [0., 0.1]:
                for mu in [0., 0.3]:
                    e = itp_(S, *a, H, li, hp[3000], rw=rw, mu=mu); rec('ITP', [li, rw, mu], e)
                    for lam, el in hp.items():
                        for fc in [1000., 2000.]: rec('PWT+ITP', [lam, li, rw, mu, fc], split(el, e, fc, fs=FS))
        print(v, ls, f'{time.time()-t0:.0f}s', flush=True)

if __name__ == '__main__': main()
