"""Error measures: full-band and low-band NMSE, octave-band NMSE along the lag axis, excitation band power."""
import numpy as np
from scipy.signal import resample_poly

EDGES = [0, 250, 500, 1000, 2000, 4000, 8001]


def ndb(e, t): return float(10*np.log10(((e-t)**2).sum()/(t**2).sum()))


def lo(h, axis=1):
    """16 kHz -> 1 kHz (0-500 Hz), amplitude kept in the 16 kHz convention."""
    return 16*resample_poly(np.asarray(h, np.float64), 1, 16, axis=axis)


def _mask(n, fs=16000):
    f = np.fft.rfftfreq(n, 1/fs); return [(f >= a) & (f < b) for a, b in zip(EDGES[:-1], EDGES[1:])]


def band_ndb(e, t, fs=16000):
    E = np.fft.rfft(e - t, axis=1); T = np.fft.rfft(t, axis=1)
    return [float(10*np.log10((np.abs(E[:, m])**2).sum()/((np.abs(T[:, m])**2).sum() + 1e-30))) for m in _mask(e.shape[1], fs)]


def band_pow(x, fs=16000, n=4096):
    X = np.abs(np.fft.rfft(x[:len(x)//n*n].reshape(-1, n), axis=1))**2; P = X.mean(0)
    return [float(10*np.log10(P[m].mean()/P[1:].mean())) for m in _mask(n, fs)]
