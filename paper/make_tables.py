# LaTeX tables of the paper from ../outputs/numbers.json (written by collect.py) -> ../outputs/tables/.
import json, os, numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', '')
N = json.load(open(D + 'numbers.json')); T = D + 'tables/'; os.makedirs(T, exist_ok=True)
LB = {'NLMS': 'NLMS', 'KF-alpha': 'KF', 'FDKF': 'FDKF', 'RW-MAP': 'RW', 'LMMSE1': 'FOA-LMMSE', 'KRR-M': 'KRR', 'KRR-MD': 'KRR-D',
      'PWT': 'PWT', 'ITP': 'ITP', 'PWT+ITP': 'PWT+ITP'}
METH = list(LB)


def fmt(x, best=False):
    if x is None or not np.isfinite(x): return '--'
    s = f'{x:.1f}'; s = '0.0' if s == '-0.0' else s; s = s.replace('-', '$-$')
    return r'\textbf{' + s + '}' if best else s


def bold_rows(cols):
    """cols: list of dict method->value; returns per column the best method."""
    out = []
    for c in cols:
        v = {m: round(x, 1) for m, x in c.items() if x is not None and np.isfinite(x)}
        out.append({m for m in v if v[m] == min(v.values())})
    return out


# ---------------------------------------------------------------- Table I: SNR 20, 36 test houses
C36 = [f'{k}|{v}|20' for k in ['wn', 'speech'] for v in ['0.2', '0.4', '0.8']]
L = [r'\begin{table*}[t]', r'\centering',
     r'\caption{NMSE (dB) on the 36 test houses at 20\,dB SNR, mean over houses. Hyperparameters of every method are selected on the four development houses. Best value per column in bold. The last rows give the improvement of PWT+ITP over the most accurate reference method of each column, with equal weight per house (mean of dB) and pooled (dB of the mean NMSE), and the number of houses in which PWT+ITP is more accurate than this reference.}',
     r'\label{tab:main36}', r'\setlength{\tabcolsep}{4.2pt}',
     r'\begin{tabular}{l' + 'c'*6 + '@{\hspace{10pt}}' + 'c'*6 + '}', r'\toprule',
     r' & \multicolumn{6}{c}{Low band (0--500\,Hz)} & \multicolumn{6}{c}{Full band (0--8\,kHz)}\\',
     r'\cmidrule(lr){2-7}\cmidrule(lr){8-13}',
     r' & \multicolumn{3}{c}{White noise} & \multicolumn{3}{c}{Speech} & \multicolumn{3}{c}{White noise} & \multicolumn{3}{c}{Speech}\\',
     r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}',
     r'$v$ (m/s) & ' + ' & '.join(['0.2', '0.4', '0.8']*4) + r'\\', r'\midrule']
cols = [{m: N['main36'][c].get(m, {}).get(key, {}).get('eq') for m in METH} for key in ['low', 'full'] for c in C36]
best = bold_rows(cols)
for m in METH:
    if m == 'PWT': L.append(r'\midrule')
    L.append(LB[m] + ' & ' + ' & '.join(fmt(col[m], m in best[i]) for i, col in enumerate(cols)) + r'\\')
L.append(r'\midrule')
for lab, f in [('Gain, equal weight', lambda g, k: fmt(g[f'{k}_eq'])), ('Gain, pooled', lambda g, k: fmt(g[f'{k}_pool'])),
               ('Houses won', lambda g, k: '{}/{}'.format(*g[f'{k}_wins']))]:
    L.append(lab + ' & ' + ' & '.join(f(N['main36'][c]['gain'], k) for k in ['low', 'full'] for c in C36) + r'\\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
open(T + 'main36.tex', 'w').write('\n'.join(L) + '\n')

# ---------------------------------------------------------------- Table II: SNR dependence, houses 5-8, mean over speeds
SN = ['inf', '20', '10', '0']; SL = {'inf': r'$\infty$', '20': '20', '10': '10', '0': '0'}
def mspeed(kind, snr, m, key):
    x = [N['main8'][f'{kind}|{v}|{snr}'].get(m, {}).get(key, {}).get('eq') for v in ['0.2', '0.4', '0.8']]
    return None if any(t is None for t in x) else float(np.mean(x))
L = [r'\begin{table*}[t]', r'\centering',
     r'\caption{NMSE (dB) versus SNR on test houses 5--8, mean over houses and over the speeds 0.2, 0.4, and 0.8\,m/s. Selection as in Table~\ref{tab:main36}. The last rows give the improvement of PWT+ITP over the most accurate reference method of each speed (equal weight, mean over speeds) and the number of house--speed pairs (out of 12) in which PWT+ITP is more accurate.}',
     r'\label{tab:snr}', r'\setlength{\tabcolsep}{4.2pt}',
     r'\begin{tabular}{l' + 'c'*8 + '@{\hspace{10pt}}' + 'c'*8 + '}', r'\toprule',
     r' & \multicolumn{8}{c}{Low band (0--500\,Hz)} & \multicolumn{8}{c}{Full band (0--8\,kHz)}\\',
     r'\cmidrule(lr){2-9}\cmidrule(lr){10-17}',
     r' & \multicolumn{4}{c}{White noise} & \multicolumn{4}{c}{Speech} & \multicolumn{4}{c}{White noise} & \multicolumn{4}{c}{Speech}\\',
     r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}\cmidrule(lr){14-17}',
     r'SNR (dB) & ' + ' & '.join([SL[s] for s in SN]*4) + r'\\', r'\midrule']
cc = [(key, kind, s) for key in ['low', 'full'] for kind in ['wn', 'speech'] for s in SN]
cols = [{m: mspeed(kind, s, m, key) for m in METH} for key, kind, s in cc]
best = bold_rows(cols)
for m in METH:
    if m == 'PWT': L.append(r'\midrule')
    L.append(LB[m] + ' & ' + ' & '.join(fmt(col[m], m in best[i]) for i, col in enumerate(cols)) + r'\\')
L.append(r'\midrule')
def gsum(kind, s, key):
    G = [N['main8'][f'{kind}|{v}|{s}']['gain'] for v in ['0.2', '0.4', '0.8']]
    return float(np.mean([g[f'{key}_eq'] for g in G])), sum(g[f'{key}_wins'][0] for g in G), sum(g[f'{key}_wins'][1] for g in G)
L.append('Gain & ' + ' & '.join(fmt(gsum(kind, s, key)[0]) for key, kind, s in cc) + r'\\')
L.append('Won & ' + ' & '.join('{1}/{2}'.format(*gsum(kind, s, key)) for key, kind, s in cc) + r'\\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
open(T + 'snr.tex', 'w').write('\n'.join(L) + '\n')

# ---------------------------------------------------------------- Table III: ablation (dev houses, oracle per condition)
A = N['abl']
rows = [('PWT ($Q=128$)', 'PWT Q=128'), ('\\quad without transport ($\\tau=0$)', 'PWT-notau'), ('\\quad $Q=16$', 'PWT Q=16'),
        ('\\quad $Q=64$', 'PWT Q=64'), ('\\quad $Q=256$', 'PWT Q=256'), ('\\quad cubic delay', 'PWT-cr'),
        ('ITP', 'ITP'), ('\\quad without transport ($\\tau=0$)', 'ITP-notau'), ('\\quad without coupling ($\\mu=0$)', 'ITP-noPW'),
        ('\\quad started from RW', 'ITP-rwinit'), ('\\quad cubic delay', 'ITP-cr'),
        ('PWT+ITP ($f_c=2$\\,kHz)', 'PWT+ITP Q=128'), ('\\quad roles exchanged', 'swap'), ('\\quad $f_c=250$\\,Hz', 'fc=250'),
        ('\\quad $f_c=1$\\,kHz', 'fc=1000'), ('\\quad $f_c=4$\\,kHz', 'fc=4000'), ('RW', 'RW-MAP')]
L = [r'\begin{table}[t]', r'\centering',
     r'\caption{Ablation on the development houses: NMSE (dB), mean over the 18 conditions with SNR $\ge 10$\,dB per excitation (speeds 0.2--0.8\,m/s), each configuration at its best regularization weights per condition. Last two columns: 2--4 and 4--8\,kHz bands, white noise.}',
     r'\label{tab:abl}', r'\setlength{\tabcolsep}{3.2pt}', r'\begin{tabular}{lcccccc}', r'\toprule',
     r' & \multicolumn{2}{c}{White noise} & \multicolumn{2}{c}{Speech} & \multicolumn{2}{c}{Bands (WN)}\\',
     r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}', r' & Low & Full & Low & Full & 2--4k & 4--8k\\', r'\midrule']
for i, (lab, m) in enumerate(rows):
    if i in (6, 12, 17): L.append(r'\midrule')
    a = A[m]; L.append(lab + ' & ' + ' & '.join(fmt(x) for x in [a['wn']['low'], a['wn']['full'], a['speech']['low'], a['speech']['full'], a['wn']['bands'][4], a['wn']['bands'][5]]) + r'\\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open(T + 'abl.tex', 'w').write('\n'.join(L) + '\n')

# ---------------------------------------------------------------- Table IV: near-source paths
NE = N['near']
cc = [('wn', '0.4'), ('wn', '0.8'), ('speech', '0.4'), ('speech', '0.8')]
def nv(geo, kind, v, m, key):
    x = [NE.get(f'{geo}|{kind}|{v}|{s}', {}).get(m, {}).get(key) for s in ['inf', '20']]
    return None if any(t is None for t in x) else float(np.mean(x))
L = [r'\begin{table}[t]', r'\centering',
     r"\caption{Near-source (closest distance 0.25--0.5\,m) and default paths: NMSE (dB), mean over 4 houses and SNR $\infty$ and 20\,dB, best regularization weights per condition. ``pw'' and ``sph'': plane-wave and spherical transport delay.}",
     r'\label{tab:near}', r'\setlength{\tabcolsep}{3.0pt}', r'\begin{tabular}{llcccc}', r'\toprule',
     r' & & \multicolumn{2}{c}{White noise} & \multicolumn{2}{c}{Speech}\\', r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
     r' & & 0.4\,m/s & 0.8\,m/s & 0.4\,m/s & 0.8\,m/s\\', r'\midrule']
for key, kl in [('low', 'Low'), ('full', 'Full')]:
    for geo, gl in [('far', 'default'), ('near', 'near')]:
        ms = [('RW-MAP', 'RW'), ('KRR-MD', 'KRR-D'), ('PWT+ITP-pw', 'PWT+ITP, pw'), ('PWT+ITP-sph', 'PWT+ITP, sph')] if key == 'low' else \
             [('RW-MAP', 'RW'), ('PWT+ITP-pw', 'PWT+ITP, pw'), ('PWT+ITP-sph', 'PWT+ITP, sph')]
        for j, (m, ml) in enumerate(ms):
            L.append((f'{kl}, {gl}' if j == 0 else '') + ' & ' + ml + ' & ' + ' & '.join(fmt(nv(geo, k, v, m, key)) for k, v in cc) + r'\\')
        L.append(r'\midrule' if not (key == 'full' and geo == 'near') else '')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open(T + 'near.tex', 'w').write('\n'.join(L) + '\n')

# ---------------------------------------------------------------- Table V: measured recordings
RR = N['real']; VS = [('V1', '0.2'), ('V2', '0.4'), ('V3', '0.8')]
bands = [('150-500', '150--500'), ('500-1000', '0.5--1k'), ('1000-2000', '1--2k'), ('150-2000', '150--2k')]
L = [r'\begin{table}[t]', r'\centering',
     r"\caption{Measured recordings (trajectoRIR, FOA array A1, loudspeaker SL; hyperparameters selected on loudspeaker SR): NMSE (dB) at the 28 reference positions per band, and in the 0--500\,Hz band at 1\,kHz sampling (``Low'').}",
     r'\label{tab:real}', r'\setlength{\tabcolsep}{3.0pt}', r'\begin{tabular}{llccccc}', r'\toprule',
     r'$v$ & Method & 150--500 & 0.5--1k & 1--2k & 150--2k & Low\\', r'\midrule']
for vi, (V, vl) in enumerate(VS):
    ms = ['RW-MAP', 'LMMSE1', 'KRR-M', 'KRR-MD', 'PWT', 'ITP', 'PWT+ITP']
    keys = [b for b, _ in bands] + ['low100']
    colv = [{m: RR[V].get(m, {}).get(k, {}).get('test') for m in ms} for k in keys]
    best = bold_rows(colv)
    for j, m in enumerate(ms):
        L.append((f'{vl}\\,m/s' if j == 0 else '') + ' & ' + LB[m] + ' & ' + ' & '.join(fmt(colv[i][m], m in best[i]) for i in range(len(keys))) + r'\\')
    if vi < 2: L.append(r'\midrule')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open(T + 'real.tex', 'w').write('\n'.join(L) + '\n')
# ---------------------------------------------------------------- Table: speed dependence, houses 5-8, white noise, 20 dB
VSP = ['0.1', '0.2', '0.4', '0.6', '0.8', '1.2', '1.6']
def cspeed(v): k = f'wn|{v}|20'; return N['main8'][k] if k in N['main8'] else N['sweep8'][k]
L = [r'\begin{table*}[t]', r'\centering',
     r'\caption{NMSE (dB) versus array speed on test houses 5--8 (white noise, 20\,dB SNR, mean over houses). Hyperparameters of every method are selected on the development houses at each speed. Best value per column in bold. The last rows give the improvement of PWT+ITP over the most accurate reference method of each column (equal weight) and the number of houses (out of 4) in which PWT+ITP is more accurate.}',
     r'\label{tab:speed}', r'\setlength{\tabcolsep}{4.0pt}',
     r'\begin{tabular}{l' + 'c'*7 + '@{\hspace{10pt}}' + 'c'*7 + '}', r'\toprule',
     r' & \multicolumn{7}{c}{Low band (0--500\,Hz)} & \multicolumn{7}{c}{Full band (0--8\,kHz)}\\',
     r'\cmidrule(lr){2-8}\cmidrule(lr){9-15}',
     r'$v$ (m/s) & ' + ' & '.join(VSP*2) + r'\\', r'\midrule']
cols = [{m: cspeed(v).get(m, {}).get(key, {}).get('eq') for m in METH} for key in ['low', 'full'] for v in VSP]
best = bold_rows(cols)
for m in METH:
    if m == 'PWT': L.append(r'\midrule')
    L.append(LB[m] + ' & ' + ' & '.join(fmt(col[m], m in best[i]) for i, col in enumerate(cols)) + r'\\')
L.append(r'\midrule')
L.append('Gain & ' + ' & '.join(fmt(cspeed(v)['gain'][f'{key}_eq']) for key in ['low', 'full'] for v in VSP) + r'\\')
L.append('Houses won & ' + ' & '.join('{}/{}'.format(*cspeed(v)['gain'][f'{key}_wins']) for key in ['low', 'full'] for v in VSP) + r'\\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
open(T + 'speed.tex', 'w').write('\n'.join(L) + '\n')

print('tables ok')
