"""Run time of the estimator stages for a 4-s recording (K = 1600, H = 512, 16 kHz)."""
import _common
import time, torch
from pwtitp.model import Fwd, Solver
from pwtitp.pipeline import rw_init, pwt, itp_
from pwtitp.sim import scenario, add_noise

s = scenario(507, 0.4, 'wn', K=1600); y = add_noise(s['y'], 20., 5)
F = Fwd(s['x'], s['K'], s['H'], s['J']); S = Solver(F, y); H = s['H']; a = (s['dp'], s['fs'], s['c'])
for rep in range(2):
    torch.cuda.synchronize(); t0 = time.time(); h0 = rw_init(S, H); torch.cuda.synchronize(); t1 = time.time()
    hp = pwt(S, F, *a, H, 3000, h0); torch.cuda.synchronize(); t2 = time.time()
    hi = itp_(S, *a, H, 1.0, hp, rw=0.03, mu=0.3); torch.cuda.synchronize(); t3 = time.time()
    print(f'J={s["J"]} K={s["K"]} RW {t1-t0:.1f}s  PWT {t2-t1:.1f}s  ITP {t3-t2:.1f}s', flush=True)
