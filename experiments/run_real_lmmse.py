"""trajectoRIR (MC3/A1, white noise, P1-P28): first-order diffuse-field LMMSE prior ('LMMSE1', pwtitp/lmmse.py) over the
same grid as in simulation, on both loudspeakers (tuned on SR, reported on SL), as run_real.py.
usage: run_real_lmmse.py V1|V2|V3 out.jsonl"""
import _common
import sys, json, time, numpy as np, torch
from pwtitp.model import Solver
from pwtitp.lmmse import lmmse_solve
from pwtitp.pipeline import rw_init, IT
from pwtitp.trajectorir import build, calibrate_joint, bnd, lo1k, low_ndb, FS, H, C

def main():
    v = sys.argv[1]; fo = open(sys.argv[2], 'a'); L = calibrate_joint('A1', 4096)
    for ls in ['SR', 'SL']:
        t0 = time.time(); B = build(ls, v, L); F = B['F']; S = Solver(F, B['y']); hg = B['hg']; g1k = lo1k(hg)
        h0 = rw_init(S, H)
        for lam in [0.3, 1., 3., 10., 30., 100.]:
            for rw in [0.03, 0.1, 0.3]:
                h = lmmse_solve(S, B['dp'], FS, C, H*lam, rw=H*rw, h0=h0, iters=IT)
                e = np.stack([F.at_time(h, n).double().cpu().numpy() for n in B['n']])
                d = dict(v=v, ls=ls, method='LMMSE1', param=[lam, rw]); d.update(bnd(e, hg)); e1k = lo1k(e)
                d['low100'] = low_ndb(e1k, g1k, 100)
                if e1k.shape[1] >= 256: d['low256'] = low_ndb(e1k, g1k, 256)
                fo.write(json.dumps(d) + '\n'); fo.flush()
        print(v, ls, f'{time.time()-t0:.0f}s', flush=True)

if __name__ == '__main__': main()
