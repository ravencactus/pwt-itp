"""Moving-microphone kernel ridge regression (Brunnstrom et al., IEEE TASLP 34, 2026, "Sound field estimation with
moving microphones using kernel ridge regression"), KRR-M (diffuse kernel) and KRR-MD (frequency-wise directional
weighting gamma(d) = exp(-beta eta^T d)).

Conventions (paper Sec. III-A, eqs. (3)-(5)):
  forward real DFT   (F v)_l = sum_n exp(+2 pi j n l / L) v_n        == conj(np.fft.rfft(v))
  inverse            (F^-1 V)_n = Re sum_l c_l exp(-2 pi j n l / L) V_l == np.fft.irfft(conj(V), L)
  c_0 = c_{L/2} = 1/L, c_l = 2/L otherwise; <U, V>_f = Re[V^H C U]  (F unitary).
Data model (2):  p(n) = <u(r_n), phi(n)>_t + s(n),  phi(n) = (x(n), x(n-1), ..., x(n-L+1)).
Kernel matrix (30): K_nm = sum_l c_l Re[ kappa_l(r_n, r_m) conj(Phi_l(n)) Phi_l(m) ],  Phi(n) = F phi(n).
Estimate (20):  F u(r) = sum_n Gamma(r, r_n) Phi(n) a_n,  a = (K + lam I)^-1 p.
Kernels, l < L/2:  kappa_l(r, r') = int_S2 exp(-j k_l (r - r')^T d) gamma(d) ds(d)  (normalised measure)
                     = j0( sqrt(xi^T xi) ),   xi = k_l (r - r') + s_eta * j beta eta,   (eq. (28) has s_eta = +1)
  j0(sqrt(w)) = sin(z)/z, z = sqrt(w), is an entire function of w (even in z), so no branch issue.
  With gamma = exp(-beta eta^T d) inserted in (27) one obtains s_eta = -1 (checked numerically on a free-field
  point source).  Nyquist bin: (kappa(r, r') + kappa(r, -r'))/2 with beta = 0.
"""
import numpy as np, torch
from scipy.signal import resample_poly


def cweights(L):
    Lf = L // 2 + 1
    c = np.full(Lf, 2.0 / L); c[0] = 1.0 / L
    if L % 2 == 0: c[-1] = 1.0 / L
    return c


def F_fwd(v, axis=-1):  # paper's forward DFT (3)
    return np.conj(np.fft.rfft(v, axis=axis))


def F_inv(V, L, axis=-1):  # paper's inverse DFT (3)
    return np.fft.irfft(np.conj(V), n=L, axis=axis)


def phi_matrix(x, n_idx, L):
    """phi(n)_k = x(n - k), k = 0..L-1, for n in n_idx (x(m) = 0 for m < 0)."""
    xp = np.concatenate([np.zeros(L), np.asarray(x, np.float64)])
    idx = n_idx[:, None] + L - np.arange(L)[None, :]
    return xp[idx]


def _j0sqrt(w):
    """j0(sqrt(w)) for complex w (entire function); series near w = 0."""
    z = torch.sqrt(w)
    small = w.abs() < 1e-6
    zs = torch.where(small, torch.ones_like(z), z)
    out = torch.sin(zs) / zs
    return torch.where(small, 1 - w / 6 + w * w / 120, out)


def _sinc_real(kd):  # j0(kd) for real kd >= 0
    small = kd < 1e-6
    kds = torch.where(small, torch.ones_like(kd), kd)
    return torch.where(small, 1 - kd * kd / 6, torch.sin(kds) / kds)


def geom(Ra, Rb, eta):
    """pairwise quantities for blocks Ra [A,3], Rb [B,3]: |ra-rb|, |ra+rb|, (ra-rb)^T eta."""
    d = ((Ra[:, None, :] - Rb[None, :, :]) ** 2).sum(-1).sqrt()
    dp = ((Ra[:, None, :] + Rb[None, :, :]) ** 2).sum(-1).sqrt()
    de = (Ra @ eta)[:, None] - (Rb @ eta)[None, :]
    return d, dp, de


def kappa_block(G, k, beta, s_eta, nyq):
    """kappa_l(r_a, r_b) from geom() output; real [A,B] if beta == 0 or Nyquist, else complex."""
    d, dp, de = G
    if nyq:
        return 0.5 * (_sinc_real(k * d) + _sinc_real(k * dp))
    if beta == 0:
        return _sinc_real(k * d)
    w = torch.complex(k * k * d * d - beta * beta, s_eta * 2 * k * beta * de)
    return _j0sqrt(w)


class KRR:
    """fs, L, c (speed of sound), beta (scalar, applied to all l < L/2), eta (unit 3-vector), s_eta (sign convention)."""

    def __init__(self, L=100, fs=1000.0, c=343.0, beta=0.0, eta=(1.0, 0.0, 0.0), s_eta=-1, dev='cuda:0', chunk=256):
        self.L, self.fs, self.c, self.beta, self.s_eta, self.dev, self.chunk = L, fs, c, float(beta), s_eta, dev, chunk
        self.Lf = L // 2 + 1
        self.cl = cweights(L)
        self.kl = 2 * np.pi * fs * np.arange(self.Lf) / L / c
        self.eta = torch.tensor(np.asarray(eta, np.float64) / np.linalg.norm(eta), device=dev)

    def _kap(self, G, l):
        nyq = (self.L % 2 == 0 and l == self.Lf - 1)
        return kappa_block(G, float(self.kl[l]), 0.0 if nyq else self.beta, self.s_eta, nyq)

    def kernel_matrix(self, R, Phi):
        """R [N,3] numpy, Phi [N,Lf] complex numpy (paper's F phi(n)). Returns K [N,N] float64 torch on CPU (pinned)."""
        N = len(R); dev = self.dev
        Rt = torch.tensor(R, device=dev, dtype=torch.float64)
        Pr = torch.tensor(Phi.real, device=dev); Pi = torch.tensor(Phi.imag, device=dev)
        Kc = torch.empty(N, N, dtype=torch.float64, pin_memory=True)
        for i0 in range(0, N, self.chunk):
            i1 = min(N, i0 + self.chunk)
            acc = torch.zeros(i1 - i0, N, dtype=torch.float64, device=dev)
            G = geom(Rt[i0:i1], Rt, self.eta)
            for l in range(self.Lf):
                kap = self._kap(G, l)
                # w = conj(Phi_n) Phi_m = Pre + j Pim
                Pre = Pr[i0:i1, l:l + 1] * Pr[None, :, l] + Pi[i0:i1, l:l + 1] * Pi[None, :, l]
                if torch.is_complex(kap):
                    Pim = Pr[i0:i1, l:l + 1] * Pi[None, :, l] - Pi[i0:i1, l:l + 1] * Pr[None, :, l]
                    acc += self.cl[l] * (kap.real * Pre - kap.imag * Pim)
                else:
                    acc += self.cl[l] * kap * Pre
            Kc[i0:i1] = acc.cpu()
            del acc, G, kap
        torch.cuda.empty_cache()
        return Kc

    def solve(self, Kc, p, lam):
        """a = (K + lam I)^-1 p via in-place GPU Cholesky (float64)."""
        A = Kc.to(self.dev, non_blocking=True)
        A.diagonal().add_(lam)
        info = torch.empty(0, dtype=torch.int32, device=self.dev)
        Lc, info = torch.linalg.cholesky_ex(A, out=(A, info))
        if int(info) != 0:
            raise RuntimeError(f'cholesky failed (info={int(info)}), lam={lam}')
        pt = torch.tensor(p, device=self.dev, dtype=torch.float64)[:, None]
        a = torch.cholesky_solve(pt, Lc)[:, 0]
        del A, Lc
        torch.cuda.empty_cache()
        return a

    def estimate(self, a, R, Phi, Rq):
        """u(r) at query points Rq [Q,3] via (20); returns [Q, L] numpy."""
        dev = self.dev
        Rt = torch.tensor(R, device=dev, dtype=torch.float64); Rqt = torch.tensor(Rq, device=dev, dtype=torch.float64)
        Pa = torch.tensor(Phi, device=dev) * a[:, None].to(torch.complex128)  # Phi_l(n) a_n
        U = torch.zeros(len(Rq), self.Lf, dtype=torch.complex128, device=dev)
        for j0 in range(0, len(R), 8192):
            G = geom(Rqt, Rt[j0:j0 + 8192], self.eta)
            for l in range(self.Lf):
                kap = self._kap(G, l)
                U[:, l] += kap.to(torch.complex128) @ Pa[j0:j0 + 8192, l]
        return F_inv(U.cpu().numpy(), self.L, axis=-1)


def interp_traj(t_nodes, P_nodes, t):
    """Piecewise-linear position at times t (linear extrapolation with the end segments)."""
    t_nodes = np.asarray(t_nodes); P_nodes = np.asarray(P_nodes)
    j = np.clip(np.searchsorted(t_nodes, t, side='right') - 1, 0, len(t_nodes) - 2)
    w = (t - t_nodes[j]) / (t_nodes[j + 1] - t_nodes[j])
    return P_nodes[j] + w[:, None] * (P_nodes[j + 1] - P_nodes[j])


def krr_lb(s, y, beta, ts, dev='cuda'):
    """Moving-microphone KRR at 1 kHz with L = 100 taps, applied to each B-format channel; beta = 0: isotropic kernel
    (KRR), beta > 0: directional weighting towards the true source (KRR-D). ts: log10 regularization factors.
    Returns {t: (J, 64, 4) low-band estimate at the knot positions}."""
    K = s['K']; xt = s['x'][K-1:].cpu().numpy().astype(np.float64); x1 = resample_poly(xt, 1, 16)
    y1 = resample_poly(y.cpu().numpy().astype(np.float64), 1, 16, axis=0)
    n_idx = np.arange(100, min(len(x1), len(y1))); tr = s['traj'].cpu().numpy()
    R = tr[n_idx]; Rq = tr[::s['H']//16][:s['J']]
    sx = x1.std(); Phi = F_fwd(phi_matrix(x1/sx, n_idx, 100))
    eta = Rq.mean(0) - s['src']; eta /= np.linalg.norm(eta)
    kr = KRR(L=100, fs=1000., c=s['c'], beta=beta, eta=eta, s_eta=-1, dev=dev); Km = kr.kernel_matrix(R, Phi)
    out = {}
    for t in ts:
        out[t] = np.stack([kr.estimate(kr.solve(Km, y1[n_idx, c], 100*10.**t), R, Phi, Rq)[:, :64]/sx
                           for c in range(4)], -1)
    return out
