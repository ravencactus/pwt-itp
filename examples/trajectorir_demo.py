"""Estimate the time-varying SRIR of one measured moving recording (trajectoRIR, FOA array A1, loudspeaker SL,
0.2 m/s, white noise) with the random-walk MAP estimator and with PWT+ITP, and compare both with the static
reference responses at the 28 rail positions (NMSE, 150 Hz - 2 kHz).
Hyperparameters are those selected on loudspeaker SR in the paper.  Env TRAJECTORIR = database root.
usage: python examples/trajectorir_demo.py [V1|V2|V3]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from pwtitp.model import Solver
from pwtitp.pipeline import rw_init, pwt, itp_, split
from pwtitp.trajectorir import build, calibrate_joint, bnd, FS, H, C

v = sys.argv[1] if len(sys.argv) > 1 else 'V1'
B = build('SL', v, calibrate_joint('A1', 4096))          # recording, excitation, knot displacements, references
F = B['F']; S = Solver(F, B['y']); a = (B['dp'], FS, C)  # forward operator (knots every H = 256 samples at 8 kHz)

h_rw = rw_init(S, H)                                       # random-walk MAP estimate, (J, K, 4) B-format knot responses
h_pwt = pwt(S, F, *a, H, 300, h_rw)                        # plane-wave transport prior (low band)
h_itp = itp_(S, *a, H, 0.1, pwt(S, F, *a, H, 3000, h_rw), rw=0., mu=0.)   # intensity-guided transport prior (high band)
h = split(h_pwt, h_itp, 1000., fs=FS)                      # PWT+ITP, crossover 1 kHz


def score(hk):                                             # response at the instants the array passes the references
    e = np.stack([F.at_time(hk, n).double().cpu().numpy() for n in B['n']]); return bnd(e, B['hg'])['150-2000']


print(f'{v}: NMSE 150 Hz - 2 kHz   RW {score(h_rw):.1f} dB   PWT+ITP {score(h):.1f} dB')
