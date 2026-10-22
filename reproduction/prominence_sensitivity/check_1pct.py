"""The 1% column must reproduce the paper's stored per-fit results exactly (validity and n_peaks)."""
import os, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); T = os.path.join(HERE, '..', 'tables')
def load(sec):
    f = glob.glob(os.path.join(HERE, 'partials', f'{sec}_*.csv')); assert len(f) == 1, (sec, f)
    d = pd.read_csv(f[0], keep_default_na=False)
    d['valid'] = d['valid'].astype(str) == 'True'
    d['np1'] = [np.nan if not v else (len([x for x in r.split(';') if x and float(x) >= 0.01]) if r != 'NA' else -1)
                for v, r in zip(d.valid, d.ratios)]
    return d
def cmp(name, mine, ref, keys, vcol, ncol):
    j = mine.merge(ref, on=keys, how='inner', suffixes=('', '_ref'))
    vref = j[vcol].astype(str).isin(['True', '1', '1.0'])
    vm = (j.valid == vref).mean()
    both = j[j.valid & vref]
    nm = (both.np1.values == both[ncol].astype(float).values).mean() if len(both) else np.nan
    print(f'{name:28} matched {len(j):6d}/{len(mine):6d}  validity agree {vm:.4%}  n_peaks agree {nm:.4%}')
for short in ('su', 'sl', 'sb'):
    for n in (50, 200):
        m = load(f'mc_{short}_{n}').rename(columns={'model': 'Model'})
        ref = pd.read_csv(f'{T}/mc_perrep_{short}_n{n}.csv')
        cmp(f'MC {short} N={n}', m, ref, ['rep', 'Model', 'K'], 'valid_ref' if 'valid_ref' in ref else 'valid', 'n_peaks')
for n in (200,):                                  # the paper runs the inverse Gaussian at N = 200 only
    m = load(f'ig_{n}').rename(columns={'model': 'Model'})
    ref = pd.read_csv(f'{T}/invgauss/mc_perrep_ig_n{n}.csv')
    cmp(f'IG N={n}', m, ref, ['rep', 'Model', 'K'], 'valid', 'n_peaks')
for sec, f, rk in (('bimodal', 'bimodal_mc.csv', 'replication'), ('bootstrap', 'bootstrap_sweep.csv', 'b'),
                   ('hydrology', 'hydrology_sweep.csv', 'b'), ('geyser', 'geyser_sweep.csv', 'b')):
    m = load(sec)
    m[rk] = m['rep'] if sec == 'bimodal' else m['rep'] - 1
    ref = pd.read_csv(f'{T}/{f}')
    ncol = 'n_peaks' if 'n_peaks' in ref else 'n_modes'
    m = m.rename(columns={'valid': 'valid'}); ref = ref.rename(columns={'feasible': 'feas'})
    cmp(sec, m, ref, [rk, 'model', 'K'], 'feas', ncol)
for j in ('0.5', '1.0'):
    m = load(f'fish_{j}')
    ref = pd.read_csv(f'{T}/fish_summary_perreplicate_jitter{j}.csv')
    # The fish per-replicate files carry no replicate id and their rows are not in
    # replicate order (parallel workers), so compare per (model, K) counts of
    # valid fits and of each mode count instead of fit by fit.
    m = m.rename(columns={'model': 'Model'})
    vref = ref['valid'].astype(str).isin(['True', '1', '1.0'])
    mine = m[m.valid].assign(n=m.np1).groupby(['Model', 'K', 'n']).size()
    theirs = ref[vref].assign(n=ref.n_modes.astype(float)).groupby(['Model', 'K', 'n']).size()
    idx = mine.index.union(theirs.index)
    diff = int((mine.reindex(idx, fill_value=0) - theirs.reindex(idx, fill_value=0)).abs().sum())
    vm = (m.groupby(['Model', 'K']).valid.sum() == ref.assign(v=vref).groupby(['Model', 'K']).v.sum()).mean()
    print(f'{"fish jitter " + j:28} valid counts agree {vm:.4%}  mode-count distribution differences {diff}')
