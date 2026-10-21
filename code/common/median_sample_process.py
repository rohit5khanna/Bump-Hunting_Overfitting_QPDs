"""
Process C -- fit the QPD to the median bootstrap SAMPLE
=======================================================

Processes A and B (median_curve_processes.py) both fit a QPD to every
bootstrap resample first and then reduce the resulting ENSEMBLE OF CURVES to
one representative. Process C reverses the order: it reduces the ensemble of
RESAMPLES to one representative sample, and fits a QPD to that, once.

    1. Draw B bootstrap resamples of the observed data.
    2. Sort each resample. At each order statistic i, take the median across
       the B resamples:

           x_C(i) = median_b  X_(i)^(b),      i = 1 .. N

       which is the median of the i-th order statistic's bootstrap
       distribution. With Weibull plotting positions p_i = i/(N+1) this is
       the pointwise median of the bootstrap EQFs, evaluated at the same
       p-values the fit will use.
    3. Fit each QPD to the pseudo-sample {(p_i, x_C(i))}.

WHY THE PSEUDO-SAMPLE IS A LEGITIMATE SAMPLE. The median is monotone: if
X_(i)^(b) <= X_(i+1)^(b) holds within every resample -- which it does, they
are order statistics -- then the medians inherit the ordering, so x_C is
nondecreasing. It is a valid sorted sample of length N, and the fitting code
consumes it exactly as it consumes the real data. Nothing about the QPD fit
has to be special-cased.

HOW IT DIFFERS FROM PROCESS A, WHICH ALSO TAKES A POINTWISE MEDIAN. A takes
the median of B fitted quantile curves; C takes the median of B empirical
samples and fits once. So in A the smoothing happens after the model has
already been asked to interpolate each noisy resample -- every wiggle a
high-K fit invents on a single resample is in the ensemble before the median
sees it. In C the model never sees an individual resample: it is fitted to an
already-stabilised set of order statistics. C therefore shows what a QPD of
order K does to a *clean* target, which is the more direct question when the
point at issue is whether high K overfits noise. A and C also differ in what
they are: A's output is a curve with no model behind it (it satisfies no
QPD's functional form), while C's output is an actual fitted QPD of the
stated order, with parameters, feasibility and a closed-form density.

WHY THE MEDIAN IS TAKEN AT THE N PLOTTING POSITIONS AND NOT ON THE DENSE
DRAWING GRID. Step 3 needs (x, p) pairs to fit. Taking the median on a
1000-point grid would hand the fit 1000 pseudo-observations instead of N,
changing the effective weighting of the least-squares problem relative to
every other fit in the paper and making the K-vs-N story incomparable. The
N-point version keeps the fit's conditioning identical to the fit on the
observed data; only the y-values change.

JITTER. Where the example's bootstrap carries jitter (fish, geyser), the
jitter is redrawn inside each replicate before sorting, so it is inside the
median. The pseudo-sample reflects the jitter's effect on the ensemble
rather than one arbitrary realisation of it.
"""

import numpy as np


def median_bootstrap_sample(x_raw, n_boot=1000, jitter=None, seed=42,
                            lower_bound=0.0, return_order_stats=False):
    """The Process C pseudo-sample.

    Resampling matches bootstrap_curve_store.build_store exactly -- same
    per-replicate generator `default_rng(seed + b)`, same draw, same jitter
    amplitude convention Uniform(-jitter, +jitter) -- so the pseudo-sample is
    the median of the very same resamples the Process A/B curve stores were
    built from, and the three variants describe one ensemble rather than
    three.

    Returns the sorted length-N pseudo-sample, or (pseudo_sample, order_stats)
    with the full (B, N) matrix when return_order_stats is True.
    """
    x_raw = np.asarray(x_raw, dtype=float)
    N = len(x_raw)
    order = np.empty((n_boot, N), dtype=float)
    for b in range(n_boot):
        rng = np.random.default_rng(seed + b)
        draw = rng.choice(x_raw, size=N, replace=True)
        if jitter:
            draw = draw + rng.uniform(-jitter, jitter, size=N)
            draw = np.clip(draw, lower_bound + 1e-6, None)
        order[b] = np.sort(draw)
    med = np.median(order, axis=0)
    # Monotone by construction (see module docstring); assert rather than
    # sort, so that a violation surfaces as a bug instead of being hidden.
    if np.any(np.diff(med) < 0):
        raise AssertionError('median of order statistics came out non-monotone')
    return (med, order) if return_order_stats else med


def fit_to_median_sample(med_sample, specs, p_grid, lower_bound=0.0,
                         fit_spec=None, verbose=True):
    """Fit every spec to the pseudo-sample and evaluate on p_grid.

    Returns {key: {'qf', 'x', 'pdf', 'model'} or None}, in the same shape the
    Process A/B figure code expects, so the identical plotting functions can
    draw it.
    """
    if fit_spec is None:
        from bootstrap_curve_store import fit_spec as fit_spec
    med_sample = np.asarray(med_sample, dtype=float)
    N = len(med_sample)
    p_pos = np.arange(1, N + 1) / (N + 1)
    out = {}
    for spec in specs:
        key, label, K, _fam = spec
        m = fit_spec(med_sample, p_pos, spec, lower_bound=lower_bound)
        if m is None:
            out[key] = None
            if verbose:
                print(f'  {key}: infeasible on the median bootstrap sample', flush=True)
            continue
        try:
            qf = np.asarray(m.quantile(p_grid), dtype=float)
            pdf = np.clip(np.asarray(m.pdf(p_grid), dtype=float), 0, None)
        except Exception as e:
            out[key] = None
            if verbose:
                print(f'  {key}: evaluation failed ({e})', flush=True)
            continue
        out[key] = {'qf': qf, 'x': qf, 'pdf': pdf, 'model': m}
    return out
