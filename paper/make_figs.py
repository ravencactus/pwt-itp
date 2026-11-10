# Figures of the paper (vector PDF; single column 3.5 in, double 7.16 in) -> ../outputs/figs/.
import json, gzip, os, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.special import spherical_jn as jn
from scipy.optimize import brentq
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'mathtext.fontset': 'stix', 'axes.linewidth': 0.6,
                     'lines.linewidth': 1.0, 'lines.markersize': 4, 'legend.fontsize': 6.5, 'legend.frameon': False,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', ''); F = P + 'outputs/figs/'; os.makedirs(F, exist_ok=True)
N = json.load(open(P + 'outputs/numbers.json'))
I = [json.loads(l) for l in gzip.open(P + 'results/raw/ident.jsonl.gz', 'rt')]
c, fs, H, K = 343.0, 16000., 512, 1600
ST = {'NLMS': ('0.6', ':', 'v'), 'KF-alpha': ('tab:gray', '--', '<'), 'FDKF': ('0.4', '-.', '>'), 'RW-MAP': ('tab:blue', '--', 'o'), 'LMMSE1': ('tab:cyan', '-.', 'P'),
      'KRR-M': ('tab:green', ':', 'x'), 'KRR-MD': ('tab:green', '-.', 's'), 'PWT': ('tab:orange', ':', '^'),
      'ITP': ('tab:purple', ':', 'D'), 'PWT+ITP': ('tab:red', '-', '*')}
LB = {'NLMS': 'NLMS', 'KF-alpha': 'KF', 'FDKF': 'FDKF', 'RW-MAP': 'RW', 'LMMSE1': 'FOA-LMMSE', 'KRR-M': 'KRR', 'KRR-MD': 'KRR-D',
      'PWT': 'PWT', 'ITP': 'ITP', 'PWT+ITP': 'PWT+ITP'}
def sty(m): col, ls, mk = ST[m]; return dict(color=col, ls=ls, marker=mk, label=LB[m])
def lab(a, s, y): a.set_title(s, y=y, fontsize=8)
db = lambda x: 10*np.log10(x)
def eps(N_, x): return 1 - sum((2*n + 1)*jn(n, x)**2 for n in range(N_ + 1))
def erw(x): return 2*(1 - jn(0, x))

E = np.load(P + 'results/example.npz')

# ---------- ident: (a) prediction error vs kd, theory and measurement; (b) identifiable bandwidth vs speed
x = np.logspace(-2.3, np.log10(30), 400)
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.4))
a = ax[0]
a.semilogx(x, db(erw(x)), color='tab:blue', ls='--', label='RW (theory)')
for N_, col in [(1, 'tab:red'), (2, 'tab:orange'), (4, 'tab:purple')]:
    a.semilogx(x, db(np.maximum(eps(N_, x), 1e-12)), color=col, label=f'$N={N_}$ (theory)')
kd = np.array([2*np.pi*f/c*r['d'] for r in I for f in r['fc']])
e0 = np.array([v for r in I for v in r['e0']]); e1 = np.array([v for r in I for v in r['e1p']])
a.scatter(kd, e0, s=3, color='tab:blue', alpha=0.25, lw=0, label='RW (houses)')
a.scatter(kd, e1, s=3, color='tab:red', alpha=0.25, lw=0, label='FOA (houses)')
a.axhline(db(H/K), color='k', lw=0.6, ls=':')
a.set_xlim(5e-3, 30); a.set_ylim(-70, 6); a.set_xlabel('$kd$'); a.set_ylabel('Prediction error (dB)')
a.grid(alpha=0.3, lw=0.4); a.legend(loc='lower right', ncol=1, fontsize=6); lab(a, '(a)', -0.38)
b = ax[1]; v = np.linspace(0.1, 1.6, 200)
star = {'RW': brentq(lambda z: erw(z) - H/K, 0.1, 3)}
for N_ in (1, 2, 4): star[N_] = brentq(lambda z: eps(N_, z) - H/K, 0.3, 20)
for key, col, ls in [('RW', 'tab:blue', '--'), (1, 'tab:red', '-'), (2, 'tab:orange', '-'), (4, 'tab:purple', '-')]:
    b.plot(v, np.minimum(star[key]*c*fs/(2*np.pi*v*H), 2e4)/1e3, color=col, ls=ls, label='RW' if key == 'RW' else f'$N={key}$')
b.axhline(fs/2e3, color='k', lw=0.6, ls=':'); b.set_ylim(0, 12); b.set_xlim(0.1, 1.6)
b.set_xlabel('Array speed $v$ (m/s)'); b.set_ylabel('$f_N$ (kHz)'); b.grid(alpha=0.3, lw=0.4); b.legend(loc='upper right')
lab(b, '(b)', -0.38)
fig.savefig(F + 'ident.pdf'); plt.close(fig)
print('kd*', star)

# ---------- kd collapse: octave-band NMSE (development houses, white noise) versus kd
A = N['abl']; fcb = np.array(I[0]['fc'])
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.4))
for vi, (vv, mk) in enumerate([('0.2', 'o'), ('0.4', 's'), ('0.8', '^')]):
    kdv = 2*np.pi*fcb/c*float(vv)*H/fs
    for m, key, col in [('RW-MAP', 'RW-MAP', 'tab:blue'), ('PWT', 'PWT Q=128', 'tab:orange'), ('ITP', 'ITP', 'tab:purple')]:
        ax[0].semilogx(kdv, A[key]['wn']['per_v'][vv]['bands'], color=col, marker=mk, ls='-', lw=0.7,
                       label=(LB[m] if vi == 0 else None))
    d_ = np.array(A['ITP']['wn']['per_v'][vv]['bands']) - np.array(A['PWT Q=128']['wn']['per_v'][vv]['bands'])
    ax[1].semilogx(kdv, d_, color='k', marker=mk, ls='-', lw=0.7, label=f'{vv} m/s')
for i, a in enumerate(ax):
    a.set_xlabel('$kd$ (octave-band center)'); a.grid(alpha=0.3, lw=0.4); a.legend(loc='best')
ax[0].set_ylabel('NMSE (dB)'); ax[1].set_ylabel('ITP $-$ PWT (dB)'); ax[1].axhline(0, color='k', lw=0.5)
h = [plt.Line2D([], [], color='0.3', marker=mk, ls='', label=f'{vv} m/s') for vv, mk in [('0.2', 'o'), ('0.4', 's'), ('0.8', '^')]]
ax[0].legend(handles=ax[0].get_legend_handles_labels()[0] + h, loc='upper left', ncol=2)
lab(ax[0], '(a)', -0.38); lab(ax[1], '(b)', -0.38)
fig.savefig(F + 'kd.pdf'); plt.close(fig)


# ---------- psi: per-house 4-8 kHz gain of ITP over PWT versus directionality (36 test houses)
fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.3))
Ir = {(r['seed'], r['v']): r for r in I}; rs = {}
for kind, vv, col, mk in [('wn', 0.4, 'tab:orange', 'o'), ('wn', 0.8, 'tab:red', 'o'), ('speech', 0.8, 'tab:purple', 's')]:
    M = N['main36'][f'{kind}|{vv}|20']
    g = np.array(M['PWT']['full']['bands'][5]) - np.array(M['ITP']['full']['bands'][5])
    ps = np.array([Ir[(s, vv)]['psi'][5] for s in range(5, 41)])
    r_ = np.corrcoef(ps, g)[0, 1]; rs[f'{kind}|{vv}'] = r_
    ax.scatter(ps, g, s=9, marker=mk, fc='none' if kind == 'speech' else col, ec=col, lw=0.6,
               label=f'{"White noise" if kind == "wn" else "Speech"}, {vv} m/s ($r={r_:.2f}$)')
    pf = np.polyfit(ps, g, 1); xx = np.array([ps.min(), ps.max()]); ax.plot(xx, np.polyval(pf, xx), color=col, lw=0.7)
r_ = rs; print('psi corr', rs)
ax.axhline(0, color='k', lw=0.5); ax.set_xlabel(r'Directionality $\bar\psi$ of the true response (4–8 kHz)'); ax.set_ylabel('ITP gain over PWT, 4–8 kHz (dB)')
ax.grid(alpha=0.3, lw=0.4); ax.legend(ncol=1, fontsize=6, loc='upper left')
fig.savefig(F + 'psi.pdf'); plt.close(fig)

# ---------- example: (a) per-knot NMSE along the path; (b)-(d) wavefronts of the W channel above 2 kHz along the path
from scipy.signal import butter, sosfiltfilt
ht = E['ht']
def nm(e, axis=(1, 2)): return db(((e - ht)**2).sum(axis)/(ht**2).sum(axis))
t = np.arange(ht.shape[0])*H/fs
fig, ax = plt.subplots(1, 4, figsize=(7.16, 2.2), gridspec_kw=dict(width_ratios=[1.25, 1, 1, 1], wspace=0.12))
for key, m in [('rw', 'RW-MAP'), ('itp', 'ITP'), ('est', 'PWT+ITP')]:
    s = sty(m); s.pop('marker'); ax[0].plot(t, nm(E[key]), **s)
ax[0].set_xlabel('Time (s)'); ax[0].set_ylabel('NMSE per knot (dB)'); ax[0].grid(alpha=0.3, lw=0.4)
ax[0].legend(ncol=1, loc='upper left'); lab(ax[0], '(a)', -0.42)
pos = ax[0].get_position(); ax[0].set_position([pos.x0, pos.y0, pos.width - 0.06, pos.height])
sos = butter(4, 2000, 'highpass', fs=fs, output='sos'); k0, k1 = int(0.002*fs), int(0.016*fs); kk = np.arange(k0, k1)/fs*1e3
Wf = {k: sosfiltfilt(sos, E[k][:, :, 0], axis=1)[:, k0:k1] for k in ['ht', 'est', 'rw']}; ref = np.abs(Wf['ht']).max()
for i, (k, l) in enumerate([('ht', '(b) True'), ('est', '(c) PWT+ITP'), ('rw', '(d) RW')]):
    a = ax[i + 1]; Z = 20*np.log10(np.abs(Wf[k])/ref + 1e-6)
    im = a.pcolormesh(t, kk, Z.T, vmin=-30, vmax=0, cmap='Greys', shading='auto', rasterized=True)
    a.set_xlabel('Time (s)'); lab(a, l, -0.42)
    if i == 0: a.set_ylabel('Lag (ms)')
    else: a.set_yticklabels([])
cb = fig.colorbar(im, ax=ax[1:], pad=0.015, fraction=0.03); cb.set_label('dB', fontsize=7)
fig.savefig(F + 'example.pdf', dpi=300); plt.close(fig)
print('example nmse', {k: float(db(((E[k]-ht)**2).sum()/(ht**2).sum())) for k in ['rw', 'pwt', 'itp', 'est']})

