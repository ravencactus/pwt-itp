"""Example estimate shown in the paper: test room (seed 5), v = 0.8 m/s, white noise, SNR 20 dB, with the
hyperparameters selected on the development rooms.  usage: example.py out.npz"""
import _common
import sys, numpy as np
from pwtitp.model import Fwd, Solver
from pwtitp.pipeline import rw_init, pwt, itp_, split
from pwtitp.sim import scenario, add_noise

s = scenario(100*5 + 7, 0.8, 'wn', K=1600); y = add_noise(s['y'], 20., 5)
F = Fwd(s['x'], s['K'], s['H'], s['J']); S = Solver(F, y); H = s['H']; a = (s['dp'], s['fs'], s['c'])
h0 = rw_init(S, H); hp = pwt(S, F, *a, H, 300, h0); hi = itp_(S, *a, H, 1.0, pwt(S, F, *a, H, 3000, h0), rw=0.03, mu=0.3)
he = split(hp, hi, 2000.)
np.savez_compressed(sys.argv[1], ht=s['ht'].cpu().numpy(), rw=h0.cpu().numpy(), pwt=hp.cpu().numpy(),
                    itp=hi.cpu().numpy(), est=he.cpu().numpy(), pk=s['pk'].cpu().numpy(), src=s['src'], room=np.array(s['room']), fs=s['fs'], H=H)
