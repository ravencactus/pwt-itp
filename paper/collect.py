# Every number of the paper -> ../outputs/numbers.json, from the raw records in ../results/raw.
# Selection: every hyperparameter of every method is chosen per condition on dev houses 1-4 (mean of dB), then reported on
# the test houses: T8 = houses 5-8 (all SNRs), T36 = houses 5-40 (SNR 20). Two averages over test houses:
#   eq  = mean of the per-house NMSE in dB (equal weight per house)
#   pool = 10 log10 of the mean of the per-house linear NMSE (dominated by the hardest houses)
import json, glob, gzip, collections, os, numpy as np
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'raw', '')
R1, R2, R3 = RAW + 'sim/', RAW + 'sim_ext/', RAW + 'sim_ext6/'   # run_baselines/main/ablation/near; run_ext; run_ext6
R4 = RAW + 'lmmse/'          # first-order diffuse-field LMMSE prior (run_lmmse.py, run_real_lmmse.py)
R5 = RAW + 'real/'           # trajectoRIR (run_real.py)
OUTER = int(os.environ.get('OUTER', 6))          # default ITP: six outer iterations
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'numbers.json' if OUTER == 6 else f'numbers_outer{OUTER}.json')
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def lines(fn): return gzip.open(fn, 'rt') if fn.endswith('.gz') else open(fn)
DEV, T8, T36 = [1, 2, 3, 4], [5, 6, 7, 8], list(range(5, 41))
BASE = ['NLMS', 'KF-alpha', 'FDKF', 'RW-MAP', 'LMMSE1', 'KRR-M', 'KRR-MD']


def canon(fn, d):
    """(method, canonical param).  PWT (lam,); ITP (li, rw, mu, outer); PWT+ITP (lam, li, rw, mu, outer, fc)."""
    m, p = d['method'], d.get('param'); b = os.path.basename(fn)
    nopw = b.startswith(('mainb', 'mb20', 'sweepb')); ext = b.startswith(('ext', 'x20', 'x620'))
    if m == 'PWT': return m, (p[0],)
    if m == 'ITP': return m, (tuple(p) if ext else (p[0], p[1], 0. if nopw else .3, 6))
    if m == 'PWT+ITP':
        if ext: return m, tuple(p)
        return m, ((p[0], p[1], p[2], 0., 6, p[4]) if nopw else (p[0], p[1], p[2], .3, 6, p[3]))
    return m, json.dumps(p)


def load(globs):
    R = collections.defaultdict(lambda: collections.defaultdict(dict))
    for g in globs:
        for fn in sorted(glob.glob(g)):
            if not fn.endswith(('.jsonl', '.jsonl.gz')): continue
            for l in lines(fn):
                d = json.loads(l)
                if d['method'] == 'XPSD': continue
                m, p = canon(fn, d); R[(d['kind'], d['v'], d['snr'], m)][p][d['seed']] = d
    for k in [k for k in R if k[3] == 'PWT']:                 # f_c = inf: PWT alone is a point of the combination grid
        for p, r in R[k].items(): R[k[:3] + ('PWT+ITP',)][(p[0], None, None, None, OUTER, None)] = r
    return R


KEEP = {'PWT+ITP': lambda p: p[4] == OUTER, 'ITP': lambda p: p[3] == OUTER}


def val(r, key, b=None): return r[key] if b is None else r['bands'][b]


def select(rows, key, test, m):
    """Dev selection; returns (param, test values per house, bands per house).  Only grid points complete on dev+test."""
    keep = KEEP.get(m, lambda p: True)
    ok = {p: r for p, r in rows.items() if keep(p) and all(s in r and key in r[s] for s in DEV + test)}
    full = {p for p, r in rows.items() if keep(p) and all(s in r and key in r[s] for s in DEV)}
    if not ok: return None
    p = min(ok, key=lambda p: np.mean([ok[p][s][key] for s in DEV]))
    pd = min(full, key=lambda p: np.mean([rows[p][s][key] for s in DEV]))
    if pd != p: print('WARNING incomplete test coverage of the dev choice', m, key, pd, '->', p)
    bands = [[ok[p][s]['bands'][b] for s in test] for b in range(6)] if 'bands' in ok[p][test[0]] else None
    return dict(param=p, x=[ok[p][s][key] for s in test], bands=bands)


def stats(x):
    x = np.asarray(x); return dict(eq=float(x.mean()), pool=float(10*np.log10(np.mean(10**(x/10)))))


def table(R, test, snrs=None):
    conds = sorted({k[:3] for k in R if k[3] == 'PWT+ITP' and (snrs is None or k[2] in snrs)},
                   key=lambda c: (c[0] != 'wn', c[1], -1e9 if c[2] is None else -c[2]))
    tab = {}
    for c in conds:
        cs = f'{c[0]}|{c[1]}|{"inf" if c[2] is None else int(c[2])}'; tab[cs] = {}
        for m in BASE + ['PWT', 'ITP', 'PWT+ITP']:
            if c + (m,) not in R: continue
            for key in ['low', 'full']:
                r = select(R[c + (m,)], key, test, m)
                if r is None: continue
                r.update(stats(r['x']))
                if r['bands']: r['bands_eq'] = [float(np.mean(b)) for b in r['bands']]
                r['param'] = list(r['param']) if isinstance(r['param'], tuple) else r['param']
                tab[cs].setdefault(m, {})[key] = r
        # gain of PWT+ITP over the best reference method (oracle choice of the reference on the test houses)
        for key, pool in [('low', BASE), ('full', [m for m in BASE if not m.startswith('KRR')])]:
            cand = [tab[cs][m][key] for m in pool if key in tab[cs].get(m, {})]
            if not cand or key not in tab[cs].get('PWT+ITP', {}): continue
            for avg in ['eq', 'pool']:
                bb = min(cand, key=lambda r: r[avg]); prop = tab[cs]['PWT+ITP'][key]
                tab[cs].setdefault('gain', {})[f'{key}_{avg}'] = bb[avg] - prop[avg]
            bb = min(cand, key=lambda r: r['eq'])
            d = np.asarray(bb['x']) - np.asarray(tab[cs]['PWT+ITP'][key]['x'])
            tab[cs]['gain'][f'{key}_wins'] = [int((d > 0).sum()), len(d)]
            # per-house best reference (strictest: the best of all references in every house)
            per = np.min(np.stack([r['x'] for r in cand]), 0)
            dd = per - np.asarray(tab[cs]['PWT+ITP'][key]['x']); tab[cs]['gain'][f'{key}_wins_strict'] = [int((dd > 0).sum()), len(dd)]
    return tab


G_MAIN = [R1 + 'base_main_*', R1 + 'main_*', R1 + 'mainb_*', R2 + 'ext_*', R3 + 'ext6_*', R4 + 'lm_main_*']
G_SWEEP = [R1 + 'base_sweep_*', R1 + 'sweep_*', R1 + 'sweepb_*', R2 + 'extsweep_*', R3 + 'ext6sweep_*', R4 + 'lm_sweep_*']
G_20 = G_MAIN + [R2 + 'b20_*', R2 + 'm20_*', R2 + 'mb20_*', R2 + 'x20_*', R3 + 'x620_*', R4 + 'lm20_*']
out = {'main8': table(load(G_MAIN), T8)}
RS = load(G_SWEEP); out['sweep8'] = table(RS, T8)
out['main36'] = table(load(G_20), T36, snrs={20.})

# ---- ablation: oracle per condition on dev houses 1-4 (as in the batch-1 analysis), mean over SNR inf/20/10 and speeds
A = collections.defaultdict(lambda: collections.defaultdict(dict))
for fn in glob.glob(R1 + 'abl_*.jsonl*'):
    for l in lines(fn):
        d = json.loads(l); p = d['param'] if isinstance(d['param'], list) else [d['param']]; m = d['method']
        if m in ('PWT', 'PWT+ITP'): m = f'{m} Q={p[0]}'
        if m == 'PWT+ITP-fc': m = f'fc={int(p[2])}'
        A[(d['kind'], d['v'], d['snr'], m)][json.dumps(p)][d['seed']] = d
for fn in glob.glob(R1 + 'base_main_*.jsonl*'):
    for l in lines(fn):
        d = json.loads(l)
        if d['method'] == 'RW-MAP' and d['seed'] in DEV and d['snr'] != 0.:
            A[(d['kind'], d['v'], d['snr'], 'RW-MAP')][json.dumps(d['param'])][d['seed']] = d
ac = sorted({k[:3] for k in A if k[3] == 'ITP'}, key=str)
def abest(c, m, key, b=None):
    rows = A.get(c + (m,))
    if not rows: return np.nan
    return min(np.mean([val(r[s], key, b) for s in DEV]) for r in rows.values() if all(s in r for s in DEV))
abl = {}
for m in sorted({k[3] for k in A}):
    for kind in ['wn', 'speech']:
        cs = [c for c in ac if c[0] == kind]
        abl.setdefault(m, {})[kind] = dict(low=float(np.mean([abest(c, m, 'low') for c in cs])), full=float(np.mean([abest(c, m, 'full') for c in cs])),
                                         bands=[float(np.mean([abest(c, m, 'bands', b) for c in cs])) for b in range(6)],
                                         per_v={str(v): dict(full=float(np.mean([abest(c, m, 'full') for c in cs if c[1] == v])),
                                                             bands=[float(np.mean([abest(c, m, 'bands', b) for c in cs if c[1] == v])) for b in range(6)])
                                                for v in sorted({c[1] for c in cs})})
out['abl'] = abl
out['abl_conds'] = [list(map(str, c)) for c in ac]

# ---- near field: oracle over the small grid, houses of the near/far runs
N = collections.defaultdict(lambda: collections.defaultdict(list))
for fn in glob.glob(R1 + 'near_*.jsonl*'):
    for l in lines(fn):
        d = json.loads(l); N[(d['geo'], d['kind'], d['v'], d['snr'], d['method'])][json.dumps(d['param'])].append(d)
for k in [k for k in N if k[4] in ('PWT-pw', 'PWT-sph')]:
    for p, r in N[k].items(): N[k[:4] + ('PWT+ITP-' + k[4][4:],)][json.dumps(json.loads(p) + [None])] = r
near = {}
for k, rows in N.items():
    cs = f'{k[0]}|{k[1]}|{k[2]}|{"inf" if k[3] is None else int(k[3])}'
    for key in ['low', 'full']:
        ok = [np.mean([r[key] for r in rs]) for rs in rows.values() if all(key in r for r in rs)]
        if ok: near.setdefault(cs, {}).setdefault(k[4], {})[key] = float(min(ok))
out['near'] = near
out['near_seeds'] = sorted({r['seed'] for rows in N.values() for rs in rows.values() for r in rs})

# ---- measured data (trajectoRIR): tuned on SR, reported on SL; independent of the simulator
Rr = collections.defaultdict(lambda: collections.defaultdict(dict))
for fn in glob.glob(R5 + 'real_V*.jsonl*'):
    for l in lines(fn):
        d = json.loads(l); m = d['method']
        if m.startswith('KRR'): m = m.rsplit('-L', 1)[0]           # pool L in {100, 256} into one grid
        Rr[(d['v'], m)][json.dumps([d['method'], d['param']])][d['ls']] = d
for k in [k for k in Rr if k[1] == 'PWT']: Rr[(k[0], 'PWT+ITP')].update(Rr[k])   # f_c = inf, as in the simulations
out['real'] = {}
for (v, m), rows in Rr.items():
    for key in ['150-2000', '150-500', '500-1000', '1000-2000', 'low100']:
        ok = {p: r for p, r in rows.items() if 'SL' in r and 'SR' in r and key in r['SL'] and key in r['SR']}
        if ok:
            p = min(ok, key=lambda p: ok[p]['SR'][key]); out['real'].setdefault(v, {}).setdefault(m, {})[key] = dict(test=ok[p]['SL'][key], param=json.loads(p))
Rl = collections.defaultdict(lambda: collections.defaultdict(dict))       # LMMSE1 on the same data (run_real_lmmse.py), same rule
for fn in glob.glob(R4 + 'real_V*_lmmse.jsonl*'):
    for l in lines(fn):
        d = json.loads(l); Rl[d['v']][json.dumps(d['param'])][d['ls']] = d
for v, rows in Rl.items():
    for key in ['150-2000', '150-500', '500-1000', '1000-2000', 'low100']:
        ok = {p: r for p, r in rows.items() if 'SL' in r and 'SR' in r and key in r['SL'] and key in r['SR']}
        if ok:
            p = min(ok, key=lambda p: ok[p]['SR'][key])
            out['real'].setdefault(v, {}).setdefault('LMMSE1', {})[key] = dict(test=ok[p]['SL'][key], param=['LMMSE1', json.loads(p)])

# ---- identifiability diagnostic of the true responses
I = [json.loads(l) for l in lines(RAW + 'ident.jsonl.gz')]
out['ident'] = I
json.dump(out, open(OUT, 'w'), indent=1)
print('ok', {k: len(v) for k, v in out.items()})
