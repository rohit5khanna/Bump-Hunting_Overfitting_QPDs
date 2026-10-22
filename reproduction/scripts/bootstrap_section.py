"""
Canonical definition of the sample and resampling scheme for the bootstrap
section of the paper (Section 4.5, "Bootstrap Diagnostics of Structural
Stability": Tables 5 and 6, Figures 6, 7 and 8, and Figures O5-O7 in the
Online Supplement). The section's scripts import from here (bootstrap_hartigan.py
restates the same seed and rule so it can run standalone), so they all
describe the same realization and the same 1000 resamples.

THE REALIZATION. Seed 200110 -- replication 68 of the Johnson SU, N = 200
Monte Carlo. Table 6 (bootstrap_summary.py -> tables/Table06_bootstrap_summary.csv)
gives its bootstrap summary.

WHY 68 AND NOT 1. Replication 68 is feasible for all five QPDs at every order
3-13. On replication 1 (seed 200043) the Metalog ORIGINAL-sample fit is
infeasible at K = 11, 12 and 13 and QFlex-U from K = 9 up, which would leave
missing bars in the section's figures. Note that 68 was SELECTED on that
feasibility, so it is not an arbitrary draw.

    SEED      = 200110     (= 42 + 200*1000 + 68, i.e. replication 68)
    GENERATOR = jpse.JohnsonSU(xi=0, lam=1, gamma=0.5, delta=1.2)
    BOOTSTRAP = rng = np.random.default_rng(SEED*10_000 + b)
                x_boot = rng.choice(x_orig, size=200, replace=True)

WHY THE SAMPLE IS PINNED TO A FILE rather than regenerated. jpse draws via
legacy np.random.seed + inverse-CDF; regenerating reproduces the stored
sample to 2.2e-15 but not bit-exactly, so results could drift with a numpy
or jpse version change. data/su_n200_seed200110_x_orig.csv (sorted) and
data/su_n200_seed200110_x_orig_unsorted.csv (generation order) are
authoritative; load_original_sample() checks the sorted file against the
sha256 in data/su_n200_seed200110_manifest.json.

THE GENERATOR MATTERS. scipy.stats.johnsonsu.rvs with a default_rng yields a
DIFFERENT sample from the same seed than jpse does, so the realization must
come from jpse (or from the pinned files above).

verify() IS NOT RUNNABLE FROM THIS REPOSITORY. It compares the pinned sample
and the resampling rule against simulation checkpoints (CKPT, a parquet file
on an external drive) written by an earlier version of the code. Those
checkpoints are not in this repository. Nothing else in the pipeline needs
them: the sweep, the tables and the figures are all computed from the pinned
files. The checkpoints also do not contain per-resample Hartigan or Silverman
results, which is why bootstrap_hartigan.py and bootstrap_silverman.py compute
Table 5 directly.
"""
import os, json, hashlib
import numpy as np

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'data')

DIST_PARAMS = dict(xi=0, lam=1, gamma=0.5, delta=1.2)
SEED        = 200110
N           = 200

X_PATH          = os.path.join(DATA_DIR, f'su_n200_seed{SEED}_x_orig.csv')            # sorted
X_UNSORTED_PATH = os.path.join(DATA_DIR, f'su_n200_seed{SEED}_x_orig_unsorted.csv')
MANIFEST        = os.path.join(DATA_DIR, f'su_n200_seed{SEED}_manifest.json')
N_BOOTSTRAP = 1000
K_VALUES    = [4, 7, 10]
CKPT = os.path.expanduser(
    f'~/mnt/Backup Plus/OD Results/MC_Bootstrap/checkpoints/su_n200_seed{SEED}.parquet')


def load_original_sample(check=True):
    """The section's original realization, sorted. Verified against the
    sha256 recorded in the manifest unless check=False."""
    x = np.loadtxt(X_PATH, skiprows=1)
    if check:
        want = json.load(open(MANIFEST))['x_orig']['sha256_of_csv']
        got = hashlib.sha256(open(X_PATH, 'rb').read()).hexdigest()
        if got != want:
            raise ValueError(f'x_orig checksum mismatch: {got} != {want}')
    return x


def load_original_sample_unsorted():
    """The realization in GENERATION ORDER.

    bootstrap_resample() draws `rng.choice(x_orig, ...)` from the unsorted
    array, and choice() selects indices -- so the ordering changes which
    values a replicate contains. Sorting destroys that information, which is
    why the sorted file alone cannot reproduce the resamples.
    """
    return np.loadtxt(X_UNSORTED_PATH, skiprows=2)


def bootstrap_resample(x_orig_unsorted, b):
    """Bootstrap replicate b of the section. Pass the UNSORTED sample."""
    rng = np.random.default_rng(SEED * 10_000 + b)
    return rng.choice(x_orig_unsorted, size=len(x_orig_unsorted), replace=True)


P_GRID = np.linspace(0.01, 0.99, 1000)


#: The earlier checkpoints (see verify) stored W1 as mean(|.|) * 0.98. Used
#: only by w1_as_stored/verify; every W1 the paper reports comes from
#: eval_framework (a plain mean at the sample points).
STORED_W1_FACTOR = 0.98


def w1_as_stored(qf_on_pgrid, x_orig_sorted):
    """The W1 the earlier checkpoints hold (used only by verify()).

    Every replicate's fit is measured against the ORIGINAL realization's
    EQF -- not the resample's -- on P_GRID, as mean(|.|) * 0.98 (the 0.98
    makes the mean an integral over the grid's span). Returns (w1_distance,
    w1_normalized); the normalized one is divided by the ORIGINAL sample's
    interdecile range, as in Equation 6.
    """
    n = len(x_orig_sorted)
    p_orig = np.arange(1, n + 1) / (n + 1)
    emp_qf = np.interp(P_GRID, p_orig, x_orig_sorted)
    w1 = float(np.mean(np.abs(np.asarray(qf_on_pgrid) - emp_qf)) * STORED_W1_FACTOR)
    p90_p10 = float(np.interp(0.9, p_orig, x_orig_sorted) -
                    np.interp(0.1, p_orig, x_orig_sorted))
    return w1, (w1 / p90_p10 if p90_p10 > 0 else float('nan'))


def verify(n_check=25, verbose=True):
    """Self-test: pinned sample vs the checkpoint's stored sample and vs a
    fresh jpse draw, then the resampling scheme end-to-end by refitting
    replicates and comparing against the values stored in the checkpoint.

    Needs the simulation checkpoints (CKPT) written by an earlier version of
    the code. They are not in this repository, so this function cannot run
    from the repository alone."""
    import sys, warnings
    warnings.filterwarnings('ignore')
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'code', 'common'))
    import pyarrow.parquet as pq
    from jpse import JohnsonSU
    from metalog.metalog_v2 import Metalog

    x = load_original_sample()
    ok = True

    stored = np.asarray(pq.read_table(
        CKPT.replace('.parquet', '_samples.parquet')).to_pandas().x_sample.iloc[0], dtype=float)
    d_art = float(np.max(np.abs(x - stored)))
    d_gen = float(np.max(np.abs(x - np.sort(
        JohnsonSU(**DIST_PARAMS).rvs(size=N, random_state=SEED)))))
    if verbose:
        print(f'  pinned vs stored artifact : max|d| = {d_art:.3e}')
        print(f'  pinned vs fresh jpse draw : max|d| = {d_gen:.3e}')
    ok &= (d_art == 0.0) and (d_gen < 1e-12)

    cols = ['bootstrap_id', 'model_type', 'constraint_type', 'model_order',
            'is_feasible', 'w1_distance', 'w1_normalized', 'primary_mode_loc']
    df = pq.read_table(CKPT, columns=cols).to_pandas()
    df = df[(df.model_type == 'Metalog') & (df.model_order == 10) &
            (df.bootstrap_id < n_check)].sort_values('bootstrap_id')
    x_uns = load_original_sample_unsorted()
    p = np.arange(1, N + 1) / (N + 1)
    dw, dn, n_cmp = [], [], 0
    for _, r in df.iterrows():
        if not r.is_feasible:
            continue
        xb = np.sort(bootstrap_resample(x_uns, int(r.bootstrap_id)))
        try:
            m = Metalog(xb, p, terms=10)
            if not m.is_feasible:
                continue
            wa, wn = w1_as_stored(m.quantile(P_GRID), x)
        except Exception:
            continue
        dw.append(abs(wa - r.w1_distance))
        dn.append(abs(wn - r.w1_normalized))
        n_cmp += 1
    if verbose:
        print(f'  refit {n_cmp} replicates (Metalog K=10):')
        print(f'    max |w1_distance   - stored| = {max(dw) if dw else float("nan"):.3e}')
        print(f'    max |w1_normalized - stored| = {max(dn) if dn else float("nan"):.3e}')
    ok &= bool(dw) and max(dw) < 1e-8 and max(dn) < 1e-8
    print('  VERIFIED' if ok else '  MISMATCH')
    return ok


if __name__ == '__main__':
    print(f'Bootstrap section — Johnson SU N={N}, seed={SEED}, B={N_BOOTSTRAP}, K={K_VALUES}')
    verify()
