"""Plane-wave transport prior (PWT).

The B-format knot response is written as a superposition of Q plane-wave components on a spherical Fibonacci grid,
    h_j(k) = sum_q a_{j,q}(k) b(u_q)^T,   b(u) = [1, u^T]^T   (SN3D, room frame),
and every component is transported exactly between knots,
    a_{j+1,q}(k) ~= a_{j,q}(k - tau_{j,q}),   tau_{j,q} = -u_q^T dp_j fs/c.
MAP estimate:
    min_a ||y - A B a||^2 + lam sum_{j,q} ||a_{j+1,q} - S_{j,q} a_{j,q}||^2 + eps ||a||^2,   h = B a.
With sph=True the delay of lag k is the exact spherical one, (|r_k u_q - dp_j| - r_k) fs/c with r_k = c k/fs.
The plane-wave delay does not depend on the lag, so it is applied exactly as a phase shift (delay='fft', zero-padded
to 2K, i.e. without circular wrap-around); with sph=True only the small
spherical residual is interpolated.
"""
import math
import torch
from .model import cg, interp_taps


def fib_dirs(Q, dev='cuda', dtype=torch.float32):
    i = torch.arange(Q, device=dev, dtype=dtype) + 0.5
    z = 1 - 2*i/Q; r = (1 - z**2).sqrt(); ph = math.pi*(1 + 5**0.5)*i
    return torch.stack([r*torch.cos(ph), r*torch.sin(ph), z], -1)            # (Q, 3)


class LazyWarp:
    """Interpolated shift as model.Warp, with the interpolation indices built per block of rows on the fly
    (memory proportional to tau only)."""
    def __init__(self, tau, kern='cr', chunk=2048):
        self.tau = tau; self.Jm, self.K = tau.shape; self.kern = kern; self.ch = chunk if kern == 'cr' else chunk//4

    def _iw(self, t):
        k = torch.arange(self.K, device=t.device, dtype=t.dtype); idx, w = interp_taps(k[None] - t, self.kern)
        w = w*((idx >= 0) & (idx < self.K)); return idx.clamp(0, self.K-1).reshape(t.shape[0], -1), w

    def __call__(self, h):
        C = h.shape[-1]; out = torch.empty_like(h)
        for a in range(0, self.Jm, self.ch):
            idx, w = self._iw(self.tau[a:a+self.ch]); n = idx.shape[0]; L = w.shape[-1]
            g = torch.gather(h[a:a+n], 1, idx[..., None].expand(-1, -1, C)).reshape(n, self.K, L, C)
            out[a:a+n] = (g*w[..., None]).sum(2)
        return out

    def T(self, r):
        C = r.shape[-1]; out = torch.zeros_like(r)
        for a in range(0, self.Jm, self.ch):
            idx, w = self._iw(self.tau[a:a+self.ch]); n = idx.shape[0]; L = w.shape[-1]
            out[a:a+n].scatter_add_(1, idx[..., None].expand(-1, -1, C), (r[a:a+n, :, None, :]*w[..., None]).reshape(n, self.K*L, C))
        return out


class FFTShift:
    """Exact fractional delay of each row by a lag-independent tau (zero-padded to 2K); adjoint = conjugate phase."""
    def __init__(self, tau, K):
        self.K = K; w = 2*math.pi*torch.fft.rfftfreq(2*K, device=tau.device)
        self.P = torch.exp(-1j*w[None, :]*tau[:, None]); self.P[:, -1] = self.P[:, -1].real.to(self.P.dtype)

    def __call__(self, z, conj=False):                                          # z (R, K, 1)
        Z = torch.fft.rfft(z[..., 0], 2*self.K, dim=1)
        return torch.fft.irfft(Z*(self.P.conj() if conj else self.P), 2*self.K, dim=1)[:, :self.K, None]

    def T(self, z): return self(z, True)


class ComposedShift:
    """Exact lag-independent delay followed by an interpolated small lag-dependent residual delay."""
    def __init__(self, fft, res): self.f = fft; self.r = res

    def __call__(self, z): return self.r(self.f(z))

    def T(self, z): return self.f.T(self.r.T(z))


class PlaneWaveTransport:
    """Synthesis/analysis on the direction grid and the transport penalty D^T D of the PWT prior."""
    def __init__(self, F, dp, fs, c, Q=128, dirs=None, sph=False, rmin=0.1, delay=None):
        """delay: 'fft' (default, plane wave only), 'cr' (Catmull-Rom) or 'lzA' (Lanczos sinc); with sph, 'fft' applies the
        plane-wave part exactly and interpolates only the small spherical residual."""
        self.F = F; self.U = fib_dirs(Q, dp.device) if dirs is None else dirs; self.Q = self.U.shape[0]
        self.Y = torch.cat([torch.ones(self.Q, 1, device=dp.device), self.U], 1)    # (Q, 4)
        Jm = dp.shape[0]
        if not sph:                                                             # plane wave: tau = -u.dp fs/c
            tau = (-(dp @ self.U.T)*fs/c).T.reshape(-1)                          # rows (q, j)
            delay = delay or 'fft'
            self.W = FFTShift(tau, F.K) if delay == 'fft' else LazyWarp(tau[:, None].expand(-1, F.K).contiguous(), delay)
        else:                                                                   # lag k <-> image at r_k = c k/fs along u_q
            r = (torch.arange(F.K, device=dp.device)*c/fs).clamp(min=rmin)
            x = r[None, None, :, None]*self.U[:, None, None, :] - dp[None, :, None, :]   # (Q, J-1, K, 3)
            ts = ((x.norm(dim=-1) - r)*fs/c).reshape(self.Q*Jm, F.K); t0 = (-(dp @ self.U.T)*fs/c).T.reshape(-1)
            if delay in (None, 'fft'):                                          # exact plane-wave part + residual
                self.W = ComposedShift(FFTShift(t0, F.K), LazyWarp((ts - t0[:, None]).contiguous()))
            else: self.W = LazyWarp(ts, delay)
        self.Jm = Jm

    def syn(self, a): return a @ self.Y                                         # (J, K, Q) -> (J, K, 4)

    def ana(self, h): return h @ self.Y.T

    def _sh(self, a, T=False):                                                  # a (J-1, K, Q) -> shifted
        Jm, K, Q = a.shape; z = a.permute(2, 0, 1).reshape(Q*Jm, K, 1)
        z = self.W.T(z) if T else self.W(z)
        return z.reshape(Q, Jm, K).permute(1, 2, 0)

    def reg(self, a):                                                           # D^T D a
        r = a[1:] - self._sh(a[:-1]); o = torch.zeros_like(a); o[1:] += r; o[:-1] -= self._sh(r, True); return o


def pwt_solve(S, pw, lam, eps=1e-3, h0=None, iters=400):
    """PWT MAP estimate (CG on the normal equations). h0: B-format start; its minimum-norm decomposition is used."""
    F = S.F
    def op(a):
        h = pw.syn(a); g = F.AT(F.A(h))
        return pw.ana(g) + lam*pw.reg(a) + eps*a
    b = pw.ana(S.b)
    a0 = torch.zeros(F.J, F.K, pw.Q, device=b.device) if h0 is None else h0 @ torch.linalg.pinv(pw.Y)
    return pw.syn(cg(op, b, a0, iters))
