"""Intensity-guided transport prior (ITP).

Each knot and lag is transported along the direction of the active intensity of the current estimate, weighted by
its non-diffuseness psi = |I|/E:  (W_j h)(k) = h(k - tau_j(k)),  tau_j(k) = -psi_j(k) u_j(k)^T dp_j fs/c.
    min ||y - A h||^2 + lam sum_j ||h_{j+1} - W_j h_j||^2 + rw sum_j ||h_{j+1} - h_j||^2
        + mu sum_{j,k} psi |V - u W|^2 + eps ||h||^2
solved by alternating between the intensity field (from the current estimate) and the quadratic problem.
The lag-dependent shift W_j uses a 64-tap Lanczos-windowed sinc interpolator (kern='lz32'; 'cr' = Catmull-Rom).
"""
import torch
from .model import cg, Warp, intensity_field, make_tau, pw_normal


def make_tau_sph(u, psi, dp, fs, c, rmin=0.1):
    """Exact spherical delay of the lag-k image (distance r_k = c k/fs) along u; tends to make_tau as r_k -> inf."""
    K = u.shape[1]; r = (torch.arange(K, device=u.device, dtype=u.dtype)*c/fs).clamp(min=rmin)[None, :, None]
    x = r*u[:-1] - dp[:, None, :]
    return psi[:-1]*(x.norm(dim=-1) - r[..., 0])*fs/c


def solve_itp(S, lam, warp, rw=0., mu=0., pw=None, eps=1e-3, h0=None, iters=400):
    F = S.F
    def DtD(h, W):
        r = h[1:] - (h[:-1] if W is None else W(h[:-1])); t = torch.zeros_like(h)
        t[1:] += r; t[:-1] -= (r if W is None else W.T(r)); return t
    def op(h):
        o = F.AT(F.A(h)) + eps*h
        if lam > 0: o = o + lam*DtD(h, warp)
        if rw > 0: o = o + rw*DtD(h, None)
        if mu > 0: o = o + mu*pw_normal(h, *pw)
        return o
    return cg(op, S.b, torch.zeros_like(S.b) if h0 is None else h0, iters)


def itp(S, h_init, lam, dp, fs, c, rw=0., mu=0., outer=6, iters=400, eps=1e-3, L=None, sph=False, kern='lz32'):
    h = h_init; Lm = torch.eye(3, device=h.device, dtype=h.dtype) if L is None else L
    for _ in range(outer):
        u, psi = intensity_field(h, Lm); W = Warp((make_tau_sph if sph else make_tau)(u, psi, dp, fs, c), kern)
        h = solve_itp(S, lam, W, rw, mu, (u, psi, Lm), eps, h0=h, iters=iters)
    return h
