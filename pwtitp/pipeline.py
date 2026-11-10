"""Complete estimator: random-walk MAP start -> PWT -> ITP (started from PWT) -> two-band combination.
Regularization weights are given per knot spacing H (lam_abs = H*lam), as in the hyperparameter grids."""
import torch
from .pwt import PlaneWaveTransport, pwt_solve
from .itp import itp

IT = 400


def split(hl, hh, fc, fs=16000, bw=0.5):
    """Zero-phase raised-cosine crossover along the lag axis (half an octave wide): low part of hl + high part of hh."""
    K = hl.shape[1]; f = torch.fft.rfftfreq(2*K, 1/fs, device=hl.device)
    a = ((f/fc).log2()/bw + 0.5).clamp(0, 1); g = (0.5 - 0.5*torch.cos(torch.pi*a))[None, :, None]   # high-pass gain
    Fl = torch.fft.rfft(hl, 2*K, dim=1); Fh = torch.fft.rfft(hh, 2*K, dim=1)
    return torch.fft.irfft((1-g)*Fl + g*Fh, 2*K, dim=1)[:, :K]


def rw_init(S, H): return S.solve(H*1., iters=IT)


def pwt(S, F, dp, fs, c, H, lam, h0, Q=128, sph=False, pw=None, delay=None):
    pw = pw or PlaneWaveTransport(F, dp, fs, c, Q, sph=sph, delay=delay); return pwt_solve(S, pw, H*lam, h0=h0, iters=IT)


def itp_(S, dp, fs, c, H, lam, h0, rw=0.03, mu=0.3, outer=6, sph=False, kern='lz32'):
    return itp(S, h0, H*lam, dp, fs, c, rw=H*rw, mu=H*mu, outer=outer, iters=IT, sph=sph, kern=kern)


def estimate(S, F, dp, fs, c, lam_p=3000, lam_i=1., rw=0.03, mu=0.3, fc=1000., Q=128, lam_p_init=3000):
    """PWT+ITP with given hyperparameters; ITP is started from the PWT estimate at lam_p_init.
    Returns (combined, PWT, ITP) knot responses (J, K, 4)."""
    H = F.H; h0 = rw_init(S, H); pw = PlaneWaveTransport(F, dp, fs, c, Q)
    hp = pwt(S, F, dp, fs, c, H, lam_p, h0, pw=pw)
    hs = hp if lam_p == lam_p_init else pwt(S, F, dp, fs, c, H, lam_p_init, h0, pw=pw)
    del pw; torch.cuda.empty_cache()
    hi = itp_(S, dp, fs, c, H, lam_i, hs, rw=rw, mu=mu)
    return split(hp, hi, fc, fs=fs), hp, hi
