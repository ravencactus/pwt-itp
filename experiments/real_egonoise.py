"""Octave-band ratio of recording power to cart ego-noise power (silence recordings) during the evaluated segment."""
import _common
import glob, numpy as np, soundfile as sf
from scipy.signal import welch
from pwtitp.trajectorir import ROOT, stamps

ED = [50, 125, 250, 500, 1000, 2000, 4000, 8000]
for v in ['V1', 'V2', 'V3']:
    ts = stamps('SL', v); t0, t1 = min(ts.values()), max(ts.values())
    fs_ = glob.glob(f'{ROOT}/audio/MOV/silence/SL/MC3/{v}/A1LF.wav') or glob.glob(f'{ROOT}/audio/MOV/silence/**/{v}/A1LF.wav', recursive=True)
    out = []
    for path in [fs_[0], f'{ROOT}/audio/MOV/WN/SL/MC3/{v}/A1LF.wav']:
        x, fs = sf.read(path); x = x[int(t0*fs):int(t1*fs)]; f, P = welch(x, fs, nperseg=8192); out.append(P)
    snr = [10*np.log10((out[1][(f >= a) & (f < b)].sum() - out[0][(f >= a) & (f < b)].sum())/out[0][(f >= a) & (f < b)].sum()) for a, b in zip(ED[:-1], ED[1:])]
    print(v, ' '.join(f'{a}-{b}:{s:5.1f}' for a, b, s in zip(ED[:-1], ED[1:], snr)))
