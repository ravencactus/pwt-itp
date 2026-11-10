"""Time-varying FIR model of a moving B-format recording, conjugate-gradient solver, random-walk MAP estimator and
the shared building blocks of the transport priors.

Model:  y_c(n) = sum_k h_c(n, k) x(n - k) + e_c(n),  h_c(n, k) = sum_j l_j(n) h_{j,c}(k)
        (linear interpolation between knots spaced H samples apart; channels c = W, X, Y, Z in SN3D).
Shapes: knot responses h (J, K, C); recordings y (N, C) with N = (J - 1) H; excitation x of length N + K - 1.
"""
import math
import torch


def cg(op, b, x0, iters=300, tol=1e-6):
    """Conjugate gradients for op(x) = b (op symmetric positive definite)."""
    x = x0.clone(); r = b - op(x); p = r.clone(); rs = (r*r).sum(); b2 = (b*b).sum()
    for _ in range(iters):
        Ap = op(p); a = rs/(p*Ap).sum(); x += a*p; r -= a*Ap; rn = (r*r).sum()
        if rn < tol**2*b2: break
        p = r + (rn/rs)*p; rs = rn
    return x


class Fwd:
    """Forward operator A (convolution of the excitation with the knot-interpolated response), same for all channels."""
    def __init__(self, x, K, H, J, dev='cuda'):
        N = (J-1)*H; assert x.numel() >= N+K-1
        X = x[:N+K-1].unfold(0, K, 1).flip(1)            # X[n, k] = x(n - k)  (x stored with offset K - 1)
        self.X = X.reshape(J-1, H, K).contiguous().to(dev)
        self.a = (torch.arange(H, device=dev, dtype=x.dtype)/H)[None, :, None]
        self.J, self.K, self.H = J, K, H

    def A(self, h):                                       # (J, K, C) -> (J-1, H, C)
        return (1-self.a)*torch.bmm(self.X, h[:-1]) + self.a*torch.bmm(self.X, h[1:])

    def AT(self, r):
        g = torch.zeros(self.J, self.K, r.shape[-1], device=r.device, dtype=r.dtype)
        Xt = self.X.transpose(1, 2)
        g[:-1] += torch.bmm(Xt, (1-self.a)*r); g[1:] += torch.bmm(Xt, self.a*r)
        return g

    def at_time(self, h, n):                              # response at sample n, interpolated from the knots
        j = min(int(n//self.H), self.J-2); a = n/self.H - j
        return (1-a)*h[j] + a*h[j+1]


def interp_taps(s, kern='cr'):
    """Interpolation indices and weights for reading h at fractional positions s: Catmull-Rom (4 taps, kern='cr')
    or a Lanczos-windowed sinc with 2a taps (kern='lzA', e.g. 'lz8'), normalized to unit DC gain."""
    i0 = torch.floor(s); f = (s - i0)[..., None]
    if kern == 'cr':
        off = torch.tensor([-1, 0, 1, 2], device=s.device)
        w = torch.cat([(-f + 2*f**2 - f**3)/2, (2 - 5*f**2 + 3*f**3)/2,
                       (f + 4*f**2 - 3*f**3)/2, (-f**2 + f**3)/2], -1)
    else:
        a = int(kern[2:]); off = torch.arange(-a + 1, a + 1, device=s.device)
        x = f - off.to(s.dtype); w = torch.sinc(x)*torch.sinc(x/a); w = w/w.sum(-1, keepdim=True)
    return i0.long()[..., None] + off, w


class Warp:
    """Fractional shift per lag: (W h)(k) = h(k - tau(k)); tau (J-1, K); kern as in interp_taps."""
    def __init__(self, tau, kern='cr'):
        Jm, K = tau.shape; k = torch.arange(K, device=tau.device, dtype=tau.dtype)
        idx, w = interp_taps(k[None] - tau, kern); self.L = idx.shape[-1]
        valid = (idx >= 0) & (idx < K); w = w*valid
        self.idx = idx.clamp(0, K-1).reshape(Jm, K*self.L); self.w = w; self.Jm, self.K = Jm, K

    def __call__(self, h):                                # h (J-1, K, C)
        C = h.shape[-1]
        g = torch.gather(h, 1, self.idx[..., None].expand(-1, -1, C)).reshape(self.Jm, self.K, self.L, C)
        return (g*self.w[..., None]).sum(2)

    def T(self, r):
        C = r.shape[-1]; out = torch.zeros_like(r)
        out.scatter_add_(1, self.idx[..., None].expand(-1, -1, C),
                         (r[:, :, None, :]*self.w[..., None]).reshape(self.Jm, self.K*self.L, C))
        return out


def smooth(v, kk=9, kj=5):
    """Separable smoothing of v (J, K, ...): Hann window over kk lags, box over kj knots."""
    sh = v.shape; J, K = sh[:2]; z = v.reshape(J, K, -1).permute(2, 0, 1)[:, None]   # (B, 1, J, K)
    wk = torch.hann_window(kk+2, periodic=False, device=v.device, dtype=v.dtype)[1:-1]; wk = wk/wk.sum()
    wj = torch.ones(kj, device=v.device, dtype=v.dtype)/kj
    z = torch.nn.functional.conv2d(torch.nn.functional.pad(z, (kk//2, kk//2, 0, 0), mode='replicate'), wk.view(1, 1, 1, -1))
    z = torch.nn.functional.conv2d(torch.nn.functional.pad(z, (0, 0, kj//2, kj//2), mode='replicate'), wj.view(1, 1, -1, 1))
    return z[:, 0].permute(1, 2, 0).reshape(sh)


def intensity_field(h, L=None):
    """Active intensity of a B-format estimate h (J, K, 4).  L maps the channel-frame XYZ to the room frame.
    Returns u (J, K, 3), unit direction of arrival (pointing towards the source), and psi (J, K) = |I|/E in [0, 1]."""
    W = h[..., 0]; V = h[..., 1:] if L is None else h[..., 1:] @ L.T
    I = smooth(W[..., None]*V); E = smooth(0.5*(W**2 + (V**2).sum(-1)))
    nI = I.norm(dim=-1); u = I/(nI[..., None] + 1e-20)
    psi = (nI/(E + 1e-20)).clamp(0, 1)
    return u, psi


def make_tau(u, psi, dp, fs, c):
    """Plane-wave delay of each knot and lag along its intensity direction: tau = -psi u.dp fs/c; dp (J-1, 3)."""
    return -psi[:-1]*(u[:-1]*dp[:, None, :]).sum(-1)*fs/c


def pw_normal(h, u, psi, L):
    """Single-plane-wave channel coupling G(h) = sqrt(psi) (L h_xyz - u h_W); returns G^T G h."""
    s = psi.sqrt()[..., None]
    g = s*((h[..., 1:] @ L.T) - u*h[..., :1])
    o = torch.zeros_like(h); o[..., 1:] = (s*g) @ L; o[..., 0] = -(s*u*g).sum(-1)
    return o


class Solver:
    """Normal equations of the data term; solve() gives the random-walk MAP estimate
    argmin ||y - A h||^2 + lam sum_j ||h_{j+1} - h_j||^2 + eps ||h||^2."""
    def __init__(self, F, y):
        self.F = F; self.b = F.AT(y.reshape(F.J-1, F.H, -1))

    def solve(self, lam=0., eps=1e-3, h0=None, iters=300):
        F = self.F
        def DtD(h):
            r = h[1:] - h[:-1]; g = torch.zeros_like(h); g[1:] += r; g[:-1] -= r; return g
        def op(h):
            o = F.AT(F.A(h)) + eps*h
            if lam > 0: o = o + lam*DtD(h)
            return o
        h0 = torch.zeros_like(self.b) if h0 is None else h0
        return cg(op, self.b, h0, iters)


def nmse_db(e, t): return 10*math.log10(float((e-t).pow(2).sum()/t.pow(2).sum()))
