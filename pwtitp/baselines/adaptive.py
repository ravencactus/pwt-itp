"""Adaptive-filter reference methods, applied to each B-format channel and read out at the query samples tq:
NLMS, Kalman filter with random-walk state model (shared covariance), and the partitioned-block frequency-domain
Kalman filter (FDKF) with optional Rauch-Tung-Striebel smoothing."""
import numpy as np
import torch

DEV = 'cuda'


def regress(x, K, n):  # u_n[k] = x(n - k), x stored with offset K - 1
    return x[n + K - 1 - torch.arange(K, device=x.device)]


def nlms(x, y, K, tq, mu, delta=1e-6):
    w = torch.zeros(K, y.shape[1], device=DEV); out = {}; tq = set(tq)
    for n in range(max(tq) + 1):
        u = regress(x, K, n); e = y[n] - u @ w; w += (mu/(u @ u + delta))*u[:, None]*e[None]
        if n in tq: out[n] = w.clone()
    return torch.stack([out[t] for t in sorted(out)])


def kf_alpha(x, y, K, tq, q_db, r):  # covariance shared by the channels (depends only on x); channels at unit variance
    sy = y.std(0); yn = y/sy; q = 10**(q_db/10)
    h = torch.zeros(K, y.shape[1], device=DEV); P = torch.eye(K, device=DEV)*1e-2; out = {}; tq = set(tq)
    for n in range(max(tq) + 1):
        u = regress(x, K, n); P.diagonal().add_(q); Pu = P @ u; g = Pu/(u @ Pu + r)
        h += g[:, None]*(yn[n] - u @ h)[None]; P.addr_(g, Pu, alpha=-1.0)
        if n in tq: out[n] = h.clone()
    return torch.stack([out[t] for t in sorted(out)])*sy


def fdkf(x, y, K, tq, block=256, A=0.999, beta=1.0, smooth=False, alpha_n=0.9, device='cuda'):
    """Partitioned-block frequency-domain Kalman filter (Enzner & Vary 2006; Kuech et al. 2014), diagonal
    approximation, first-order Markov state model W(k+1) = A W(k) + dW with Psi_dW = (1 - A^2) E|W|^2.
    smooth=True adds a fixed-interval Rauch-Tung-Striebel smoother over all blocks (offline, uses the whole recording
    like the MAP estimators).  Returns the K-tap time-domain RIR at the query sample indices tq."""
    x = torch.as_tensor(x, device=device, dtype=torch.float32); y = torch.as_tensor(y, device=device, dtype=torch.float32)
    sx, sy = float(x.std()) + 1e-12, float(y.std()) + 1e-12  # scale-invariant: run on unit-variance signals
    x, y = x / sx, y / sy
    B = block; P = (K + B - 1) // B; N = 2 * B; F = B + 1
    nb = len(y) // B
    xp = torch.cat([torch.zeros(B * P, device=device), x])
    W = torch.zeros(P, F, dtype=torch.complex64, device=device)
    Pc = torch.ones(P, F, device=device) * (B / K)
    Psi_s = torch.ones(F, device=device) * B
    Ws, Pps, Pms = [], [], []
    for b in range(nb):
        # partitions p: input frames delayed by p blocks
        st = b * B + B * P - torch.arange(P, device=device) * B - B
        idx = st[:, None] + torch.arange(N, device=device)[None]
        X = torch.fft.rfft(xp[idx])  # [P, F]
        # prediction (time update) from previous block
        if b > 0:
            Pc = A ** 2 * Pc + (1 - A ** 2) * (W.abs() ** 2)
            W = A * W
        Pms.append(Pc.clone())
        Yh = (X * W).sum(0)
        yh = torch.fft.irfft(Yh, n=N)[B:]
        e = y[b * B:(b + 1) * B] - yh
        E = torch.fft.rfft(torch.cat([torch.zeros(B, device=device), e]))
        D = (X.abs() ** 2 * Pc).sum(0) + 2 * Psi_s + 1e-6
        G = Pc * X.conj() / D
        dw = torch.fft.irfft(G * E, n=N); dw[:, B:] = 0  # gradient (causality) constraint
        W = W + torch.fft.rfft(dw)
        Pc = (1 - (G * X).real / P).clamp(1e-3, 1) * Pc
        # observation-noise PSD (same DFT units as |E|^2) from the a-posteriori error, scaled by beta (tuned)
        Yp = (X * W).sum(0); ep = y[b * B:(b + 1) * B] - torch.fft.irfft(Yp, n=N)[B:]
        Ep = torch.fft.rfft(torch.cat([torch.zeros(B, device=device), ep]))
        Psi_s = alpha_n * Psi_s + (1 - alpha_n) * beta * Ep.abs() ** 2
        Ws.append(W.clone()); Pps.append(Pc.clone())
    if smooth:  # RTS: W_s(k) = W(k) + C (W_s(k+1) - A W(k)),  C = A P+(k) / P-(k+1)
        for b in range(nb - 2, -1, -1):
            C = A * Pps[b] / Pms[b + 1].clamp_min(1e-12)
            Ws[b] = Ws[b] + C * (Ws[b + 1] - A * Ws[b])
    taps = torch.stack([torch.fft.irfft(w, n=N)[:, :B].reshape(-1)[:K] for w in Ws])  # [nb, K], valid at block end
    tb = (torch.arange(nb, device=device) + 1) * B
    q = torch.as_tensor(np.asarray(tq), device=device).long()
    j = torch.clamp(torch.searchsorted(tb, q) , 0, nb - 1)
    return taps[j] * (sy / sx)


