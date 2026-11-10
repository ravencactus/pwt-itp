"""First-order diffuse-field LMMSE transport prior (order-1 baseline).

The knot response at j+1 is predicted from the four FOA channels at knot j by the diffuse-field LMMSE predictor of the
translation over dp_j (Proposition 1, extended to all four channels). Per frequency f of the lag axis, with
x = 2 pi f |dp_j|/c and n = dp_j/|dp_j| (SN3D, room frame):
    W' = j0 W + 3i j1 n^T V
    V' = i j1 n W + (j0 + j2) V - 3 j2 n (n^T V)
MAP estimate:
    min ||y - A h||^2 + lam sum_j ||h_{j+1} - P_j h_j||^2 + rw sum_j ||h_{j+1} - h_j||^2 + eps ||h||^2.
At large kd, P_j -> 0 and the first knot is constrained only by eps; the weak random-walk term (as in ITP) closes this gap.
P_j is applied as a zero-padded (2K) spectral multiplication; the adjoint uses the conjugate coefficients.
"""
import math
import torch
from .model import cg


def sph_j012(x):
    """Spherical Bessel functions j0, j1, j2 (series for small x)."""
    s = x < 1e-2; xs = torch.where(s, torch.ones_like(x), x)
    sn, cs = torch.sin(xs), torch.cos(xs)
    j0 = sn/xs; j1 = sn/xs**2 - cs/xs; j2 = (3/xs**2 - 1)*sn/xs - 3*cs/xs**2
    x2 = x**2
    j0 = torch.where(s, 1 - x2/6, j0); j1 = torch.where(s, x/3 - x*x2/30, j1); j2 = torch.where(s, x2/15, j2)
    return j0, j1, j2


class FOAPredictor:
    """Applies P_j (and its adjoint) to h (J-1, K, 4)."""
    def __init__(self, dp, K, fs, c):
        self.K = K
        f = torch.fft.rfftfreq(2*K, 1/fs, device=dp.device).to(dp.dtype)            # (B,)
        d = dp.norm(dim=-1).clamp(min=1e-12); self.n = (dp/d[:, None])[:, None, :]      # (J-1, 1, 3)
        x = 2*math.pi*f[None, :]*d[:, None]/c                                           # (J-1, B)
        j0, j1, j2 = sph_j012(x)
        j1 = j1.clone(); j1[:, -1] = 0.                                                  # Nyquist bin real
        self.j0, self.j1, self.j2 = j0[..., None], j1[..., None], j2[..., None]

    def _apply(self, h, conj):
        K = self.K; Hf = torch.fft.rfft(h, 2*K, dim=1)                                # (J-1, B, 4)
        W, V = Hf[..., :1], Hf[..., 1:]; n = self.n
        ij1 = (-1j if conj else 1j)*self.j1
        nV = (V*n).sum(-1, keepdim=True)
        W2 = self.j0*W + 3*ij1*nV
        if conj:   # adjoint of [[j0, 3i j1 n^T], [i j1 n, (j0+j2) I - 3 j2 n n^T]]: off-diagonal blocks swap
            W2 = self.j0*W + ij1*nV
            V2 = 3*ij1*n*W + (self.j0 + self.j2)*V - 3*self.j2*n*nV
        else:
            V2 = ij1*n*W + (self.j0 + self.j2)*V - 3*self.j2*n*nV
        return torch.fft.irfft(torch.cat([W2, V2], -1), 2*K, dim=1)[:, :K]

    def __call__(self, h): return self._apply(h, False)

    def T(self, h): return self._apply(h, True)


def lmmse_solve(S, dp, fs, c, lam, rw=0., eps=1e-3, h0=None, iters=400):
    F = S.F; P = FOAPredictor(dp, F.K, fs, c)
    def op(h):
        r = h[1:] - P(h[:-1]); g = torch.zeros_like(h); g[1:] += r; g[:-1] -= P.T(r)
        o = F.AT(F.A(h)) + eps*h + lam*g
        if rw > 0:
            r = h[1:] - h[:-1]; t = torch.zeros_like(h); t[1:] += r; t[:-1] -= r; o = o + rw*t
        return o
    return cg(op, S.b, torch.zeros_like(S.b) if h0 is None else h0, iters)
