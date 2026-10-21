"""
Fish -- sensitivity of Log Metalog validity and modality to sample size
(paper Table 11, Table11_fish_sample_size_sweep.csv), at K = 10 and K = 13,
1000 bootstrap replicates per size.

SAMPLES. Built on the paper's own fish replicates: replicate b is
fish_common.resample(x, 0.5, b) -- the same 1000 jittered (+-0.5 lb) draws
behind Table 10 (fish_summary.py). For sample size N the
fitted sample is the FIRST N values of replicate b in generation order.
resample() draws all n indices and then all n jitters, element by element,
so the first N entries are themselves an iid bootstrap draw of size N with
+-0.5 lb jitter. Consequences:

  * at N = 3474 the sample IS the paper's replicate b, so that row must
    reproduce Table 10's Log Metalog K = 10 and K = 13 rows exactly -- checked
    at the end of every run;
  * sizes are NESTED within a replicate (200 within 400 within ... 3474), so
    differences between rows reflect N rather than unrelated random draws.

CONVENTIONS (identical to fish_summary.py / Table 10):
  * fit: LogMetalog(lower_bound=LOWER_BOUND), Weibull positions i/(N+1);
    a fit is VALID when it constructs and Q increases on the 0.001..0.999 grid;
  * modes: eval_framework (grid-difference density, interior peaks, relative prominence >= 1%);
  * Validity % is over all 1000 replicates; the mode shares are over the
    valid fits; "1 mode" is n_modes <= 1 (no-peak fits count as unimodal,
    the paper convention).
"""
import os, sys, json, time, datetime, warnings
import multiprocessing as mp
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'code', 'common')); sys.path.insert(0, HERE)

from fish_common import load_fish, resample, P_GRID, N_BOOT, LOWER_BOUND
from metalog.metalog_v2 import LogMetalog
import eval_framework as EF
from table_rounding import split_pct   # rows sum to exactly 100.0

TAB = os.path.join(HERE, '..', 'tables')
LOG = os.path.join(HERE, '..', 'logs')
JITTER = 0.5
SIZES = [200, 400, 800, 1600, 3474]
KS = [10, 13]

_X = None


def _init():
    global _X
    _X, _ = load_fish()
    assert SIZES[-1] == len(_X), f'largest size must be the full sample ({len(_X)})'


def _one(b):
    if _X is None:
        _init()
    full = resample(_X, JITTER, b)            # generation order, NOT sorted
    out = []
    for N in SIZES:
        s = np.sort(full[:N])
        p = np.arange(1, N + 1) / (N + 1)
        for K in KS:
            try:
                m = LogMetalog(s, p, lower_bound=LOWER_BOUND, terms=K)
                ok = bool(EF.is_valid(m))
            except Exception:
                ok = False
            if not ok:
                out.append((N, K, b, False, np.nan)); continue
            qf = EF.quantile(m)
            nm, _l, _h = EF.modes(m)
            out.append((N, K, b, True, float(nm) if nm is not None else -1.0))
    return out


def main():
    t0 = time.time()
    print(f'fish sample-size sweep: N={SIZES}, K={KS}, {N_BOOT} replicates, '
          f'jitter +-{JITTER} lb', flush=True)
    rows = []
    with mp.Pool(4, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, range(N_BOOT), chunksize=10), 1):
            rows.extend(r)
            if i % 250 == 0:
                print(f'  {i}/{N_BOOT}  {time.time()-t0:.0f}s', flush=True)

    df = pd.DataFrame(rows, columns=['N', 'K', 'b', 'valid', 'n_modes'])
    df = df.sort_values(['K', 'N', 'b']).reset_index(drop=True)
    df.to_csv(os.path.join(TAB, 'fish_sample_size_sweep_perreplicate.csv'), index=False)

    out = []
    for K in KS:
        for N in SIZES:
            g = df[(df.K == K) & (df.N == N)]
            v = g[g.valid]
            vm = v[v.n_modes >= 0]
            out.append(dict(
                K=K, N=N, n_replicates=len(g), n_valid=len(v),
                Validity_pct=round(100 * len(v) / len(g), 1),
                **dict(zip(['Unimodal_pct', 'Bimodal_pct', 'GT2_pct'],
                           split_pct([(vm.n_modes <= 1).sum(), (vm.n_modes == 2).sum(),
                                      (vm.n_modes > 2).sum()])))))
    res = pd.DataFrame(out)
    dst = os.path.join(TAB, 'Table11_fish_sample_size_sweep.csv')
    res.to_csv(dst, index=False)
    print('\n' + res.to_string(index=False))

    # Anchor: the N = 3474 row must equal Table 10 (fish_summary, 0.5 lb).
    t9 = pd.read_csv(os.path.join(TAB, 'Table10_fish_summary_jitter0.5.csv'))
    print('\nN = 3474 vs Table 10 (Table10_fish_summary_jitter0.5.csv):')
    ok_all = True
    for K in KS:
        a = res[(res.K == K) & (res.N == SIZES[-1])].iloc[0]
        t = t9[(t9.Model == 'Log Metalog') & (t9.K == K)].iloc[0]
        same = all(np.isclose(a[c], t[c]) for c in
                   ['Validity_pct', 'Unimodal_pct', 'Bimodal_pct', 'GT2_pct'])
        ok_all &= same
        print(f'  K={K}: here {a.Validity_pct}/{a.Unimodal_pct}/{a.Bimodal_pct}/{a.GT2_pct}  '
              f'Table 10 {t.Validity_pct}/{t.Unimodal_pct}/{t.Bimodal_pct}/{t.GT2_pct}  '
              f'-> {"MATCH" if same else "MISMATCH"}')

    os.makedirs(LOG, exist_ok=True)
    json.dump({'generated': datetime.datetime.now().isoformat(timespec='seconds'),
               'replicates': f'first N of fish_common.resample(x, {JITTER}, b), b < {N_BOOT}',
               'sizes': SIZES, 'K': KS, 'model': 'Log Metalog',
               'unimodal_convention': 'n_modes <= 1 (no-peak fits count as unimodal)',
               'anchor_matches_table10': bool(ok_all),
               'runtime_sec': round(time.time() - t0, 1)},
              open(os.path.join(LOG, 'fish_sample_size_sweep_provenance.json'), 'w'), indent=2)
    print(f'\n-> {dst}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
