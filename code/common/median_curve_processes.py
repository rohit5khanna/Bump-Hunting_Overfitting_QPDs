"""
Representative ("median") bootstrap curve selection
====================================================

Given B bootstrap QPD fits, each evaluated as a quantile curve Q_b(p) on a
shared probability grid, we often want ONE curve to plot as "the typical
fit". There is more than one defensible way to define that, and the choice
changes what the plotted curve means. This module implements two of them,
side by side, so figures can show both.

PROCESS A -- pointwise median quantile function
    At each p_i, take the median of {Q_1(p_i), ..., Q_B(p_i)}.

    Q_A(p_i) = median_b Q_b(p_i)

    Because every Q_b is nondecreasing in p, the pointwise median is too, so
    Q_A is always a valid quantile function. Its density is obtained from
    Q_A itself (f = 1 / (dQ/dp) evaluated at x = Q_A(p)), not by taking a
    pointwise median of the individual densities -- median-of-densities and
    the density-of-the-median-quantile-function are different objects, and
    only the latter is consistent with the curve being drawn.

    Caveat worth stating in any caption: Q_A is a synthetic curve. It is not
    one of the bootstrap fits, and in general no resample produces it. It can
    therefore show features (or smooth away features) that no individual fit
    has -- averaging two fits whose second mode sits at different locations
    can flatten the mode entirely.

PROCESS B -- medoid: the actual bootstrap fit closest to Process A
    Take Q_A from Process A, then choose the single real bootstrap fit whose
    curve is closest to it in W1:

    b* = argmin_b  W1(Q_b, Q_A),   W1 = integral |Q_b(p) - Q_A(p)| dp

    This returns an honest member of the bootstrap ensemble -- a fit some
    resample actually produced -- at the cost of not being exactly central.
    Its modality is real modality: whatever structure it shows, at least one
    resample genuinely implied.

Use A when the question is "where is the centre of the ensemble", B when the
question is "what does a typical individual fit look like".

Both functions take `qf_curves` of shape (B, n_p) and ignore non-finite rows.
"""

import numpy as np


def _finite_rows(qf_curves):
    qf = np.asarray(qf_curves, dtype=float)
    if qf.ndim != 2:
        raise ValueError(f'qf_curves must be 2-D (B, n_p); got shape {qf.shape}')
    ok = np.all(np.isfinite(qf), axis=1)
    if not ok.any():
        raise ValueError('no finite bootstrap curves supplied')
    return qf, ok


def density_from_quantile(p_grid, qf, clip_min=0.0, smooth_window=51, smooth_order=3):
    """Density implied by a quantile curve: f(x) = 1 / (dQ/dp) at x = Q(p).

    Returned as (x, f) sampled at the same p-grid points, so it can be
    plotted directly against a value axis.

    Why the smoothing matters here. A pointwise median quantile curve is not
    a smooth function: at each p the median may come from a different
    resample, so Q_A picks up grid-scale jitter that no individual fit has.
    Differentiating amplifies exactly that jitter, and 1/(dQ/dp) turns it
    into a violently spiky density that misrepresents how smooth the
    underlying fits are. The derivative is therefore taken with a
    Savitzky-Golay filter (a local polynomial fit) rather than raw central
    differences. This smooths the SLOPE only -- the plotted quantile curve
    itself is never altered.

    Pass smooth_window=0 to get the raw np.gradient derivative instead; the
    result is faithful to the median curve's literal slope but is normally
    too noisy to plot for Process A. Process B needs no smoothing at all,
    since there the curve is a single model's own output -- which is why it
    is given its stored density rather than a derived one.

    NON-UNIFORM GRIDS. The p-grid used for the empirical figures is a dense
    log-spaced ramp toward p -> 0 glued to a linear core, so spacing varies
    by orders of magnitude and savgol_filter's single `delta` argument does
    not apply. The derivative is therefore taken in INDEX space for both the
    curve and the grid, and combined by the chain rule:

        dQ/dp = (dQ/di) / (dp/di)

    which is exact for any spacing and reduces to the usual delta-scaled
    filter when the grid happens to be uniform.
    """
    p_grid = np.asarray(p_grid, dtype=float)
    qf = np.asarray(qf, dtype=float)
    n = len(qf)
    if smooth_window and n >= 7:
        from scipy.signal import savgol_filter
        w = int(min(smooth_window, n if n % 2 else n - 1))
        if w % 2 == 0:
            w -= 1
        w = max(w, smooth_order + 2 + (smooth_order % 2 == 0))
        if w % 2 == 0:
            w += 1
        dq_di = savgol_filter(qf, window_length=w, polyorder=smooth_order, deriv=1, delta=1.0)
        dp_di = savgol_filter(p_grid, window_length=w, polyorder=smooth_order, deriv=1, delta=1.0)
        dp_di = np.where(dp_di <= 0, np.nan, dp_di)
        dq_dp = dq_di / dp_di
    else:
        dq_dp = np.gradient(qf, p_grid)
    dq_dp = np.where(dq_dp <= 0, np.nan, dq_dp)
    pdf = 1.0 / dq_dp
    pdf = np.clip(np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0), clip_min, None)
    return qf, pdf


def process_a_pointwise_median(p_grid, qf_curves, smooth_window=51):
    """Process A. Returns dict with 'qf', 'pdf', 'x', 'n_used'.

    'qf' is the pointwise median quantile curve on p_grid -- exactly the
    median, unsmoothed. 'x'/'pdf' are the density implied by that curve, with
    its slope smoothed as described in density_from_quantile.
    """
    qf, ok = _finite_rows(qf_curves)
    med = np.median(qf[ok], axis=0)
    x, pdf = density_from_quantile(p_grid, med, smooth_window=smooth_window)
    return {'qf': med, 'x': x, 'pdf': pdf, 'n_used': int(ok.sum())}


def w1_to_reference(p_grid, qf_curves, reference_qf):
    """W1 distance of each bootstrap curve to a reference quantile curve.

    W1(Q_b, Q_ref) = integral_0^1 |Q_b(p) - Q_ref(p)| dp, approximated on
    p_grid by the trapezoid rule and normalised by the grid's p-span so the
    result is in data units regardless of how much of (0,1) the grid covers.
    Non-finite curves get np.inf.
    """
    p_grid = np.asarray(p_grid, dtype=float)
    qf, ok = _finite_rows(qf_curves)
    ref = np.asarray(reference_qf, dtype=float)
    span = p_grid[-1] - p_grid[0]
    out = np.full(qf.shape[0], np.inf)
    diff = np.abs(qf[ok] - ref[None, :])
    out[ok] = np.trapezoid(diff, p_grid, axis=1) / span
    return out


def process_b_medoid(p_grid, qf_curves, pdf_curves=None, mask=None):
    """Process B. Returns dict with 'index', 'qf', 'pdf', 'w1', 'n_used',
    and 'reference_qf' (the Process A curve it was matched against).

    'pdf' is the stored density of the selected bootstrap fit when
    `pdf_curves` is given (preferred -- that is the model's own density);
    otherwise it is derived from the selected quantile curve.
    """
    a = process_a_pointwise_median(p_grid, qf_curves)
    # `mask` restricts the DISTANCE to a working range while the returned
    # curves stay on the full grid. Without it, a grid extended toward p -> 0
    # for drawing would let extreme-tail behaviour -- where unstable fits
    # differ most and matter least -- decide which replicate is "closest to
    # the median".
    if mask is None:
        d = w1_to_reference(p_grid, qf_curves, a['qf'])
    else:
        m = np.asarray(mask, dtype=bool)
        d = w1_to_reference(np.asarray(p_grid)[m],
                            np.asarray(qf_curves)[:, m], a['qf'][m])
    idx = int(np.argmin(d))
    qf_sel = np.asarray(qf_curves, dtype=float)[idx]
    if pdf_curves is not None:
        pdf_sel = np.asarray(pdf_curves, dtype=float)[idx]
        x_sel = qf_sel
    else:
        x_sel, pdf_sel = density_from_quantile(p_grid, qf_sel)
    return {
        'index': idx, 'qf': qf_sel, 'x': x_sel, 'pdf': pdf_sel,
        'w1': float(d[idx]), 'n_used': a['n_used'], 'reference_qf': a['qf'],
    }


def both_processes(p_grid, qf_curves, pdf_curves=None):
    """Convenience: returns {'A': ..., 'B': ...} from one curve store."""
    return {
        'A': process_a_pointwise_median(p_grid, qf_curves),
        'B': process_b_medoid(p_grid, qf_curves, pdf_curves=pdf_curves),
    }
