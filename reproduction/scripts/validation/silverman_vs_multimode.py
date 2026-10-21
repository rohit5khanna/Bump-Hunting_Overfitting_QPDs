"""Cross-check our Silverman test against R's multimode::modetest.

WHY. stats_tests.apply_silverman_test is OUR implementation. It was written
from Silverman's (1981) procedure, validated against samples with known mode
counts, and afterwards found to match the formula the multimode paper prints
for its submethod 1 -- the rescaled bootstrap,

    Y*_bi = (1 + h_k^2 / sigma^2)^(-1/2) X*_bi,   X* drawn from f-hat at h_k.

That is agreement with a PROSE DESCRIPTION, not with the reference code. This
script closes that gap: it writes fixed samples with known mode counts, records
our p-values, and leaves the same samples for the R side to score. Both are
bootstrap estimates, so they agree up to Monte Carlo error (~1/sqrt(B)), not
exactly -- the check is |difference| small and decisions identical, not equality.

Step 1 (here):   python3 silverman_vs_multimode.py
Step 2 (R):      Rscript silverman_vs_multimode.R
Step 3 (here):   python3 silverman_vs_multimode.py --compare
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..', 'code', 'common')))

SAMPLES_CSV = os.path.join(HERE, 'validation_samples.csv')
OURS_CSV = os.path.join(HERE, 'validation_ours.csv')
R_CSV = os.path.join(HERE, 'validation_multimode.csv')
B = 999
N = 200


def make_samples():
    """Fixed samples with known mode counts, one per column."""
    rng = np.random.default_rng(20261004)
    out = {}
    out['unimodal_normal'] = rng.normal(0, 1, N)
    for name, locs in (('bimodal_0_4', [0, 4]), ('trimodal_0_5_10', [0, 5, 10])):
        c = rng.integers(0, len(locs), N)
        out[name] = np.array([rng.normal(locs[i], 1.0) for i in c])
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..', 'code', 'common', 'jpse')))
    try:
        from jpse.johnson import JohnsonSU
        out['johnson_su_paper'] = np.asarray(
            JohnsonSU(xi=0, lam=1, gamma=0.5, delta=1.2).rvs(size=N, random_state=200110), float)
    except Exception as e:
        print(f'  (skipping Johnson SU sample: {e})')
    return {k: np.sort(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compare', action='store_true')
    a = ap.parse_args()

    import pandas as pd
    if not a.compare:
        from stats_tests import apply_silverman_test
        s = make_samples()
        pd.DataFrame(s).to_csv(SAMPLES_CSV, index=False)
        rows = []
        for name, x in s.items():
            for k in (1, 2, 3):
                np.random.seed(4242 + k)
                rows.append(dict(sample=name, k=k,
                                 p_ours=apply_silverman_test(x, n_bootstrap=B, k=k)))
                print(f'  {name:>18} k={k}  p={rows[-1]["p_ours"]:.4f}')
        pd.DataFrame(rows).to_csv(OURS_CSV, index=False)
        print(f'\nwrote {os.path.basename(SAMPLES_CSV)} and {os.path.basename(OURS_CSV)}')
        print('now run:  Rscript silverman_vs_multimode.R')
        return 0

    if not os.path.exists(R_CSV):
        sys.exit(f'{os.path.basename(R_CSV)} not found -- run the R step first')
    o = pd.read_csv(OURS_CSV); r = pd.read_csv(R_CSV)
    m = o.merge(r, on=['sample', 'k'])
    m['diff'] = (m.p_ours - m.p_multimode).abs()
    m['same_decision'] = (m.p_ours < 0.05) == (m.p_multimode < 0.05)
    print(m.to_string(index=False))
    print(f'\nmax |difference| = {m["diff"].max():.4f}   '
          f'(Monte Carlo error at B={B} is about {1/np.sqrt(B):.3f})')
    print(f'decisions agree at alpha=0.05: {int(m.same_decision.sum())}/{len(m)}')
    return 0 if m.same_decision.all() else 1


if __name__ == '__main__':
    sys.exit(main())
