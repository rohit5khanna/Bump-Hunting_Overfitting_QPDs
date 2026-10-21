"""
Shared modality-test implementations used across this repo's experiments.

The paper's modality-test results are produced by ONE Silverman
implementation, apply_silverman_test, at ONE mode-detection prominence
(KDE_PROMINENCE_FRAC, below). The legacy fish-jitter family further down is
not used for anything the paper reports.

  PAPER FAMILY  (apply_silverman_test, apply_hartigan_dip_test)
     Used by: reproduction/scripts/mc_modality_tests.py             (Table 3)
              reproduction/scripts/bootstrap_silverman.py           (Table 5, Silverman)
              reproduction/scripts/fish_modality_paper_replicates.py (Table 9)
              reproduction/scripts/geyser_modality_paper_replicates.py (Table 12)
              reproduction/scripts/hydrology_modality_tests.py      (text of Section 5.1)

     H0: the density has <= k modes. A 0-mode KDE satisfies H0 and every
     decision point tests "<= k", not "== k". In practice the case does not
     arise: a Gaussian KDE of a finite sample decays to zero in both tails
     and so always has at least one interior maximum.

     h_crit = inf{h : KDE(data, h) has <= k modes}, located by bisection with
     an expanding upper bracket. _critical_bandwidth returns the bracket's
     UPPER end, not the midpoint, so the null provably satisfies H0.

     Mode counting: binned Gaussian-KDE grid convolved via numpy.fft, then
     scipy.signal.find_peaks with prominence = KDE_PROMINENCE_FRAC of the
     tallest peak. numpy.fft rather than scipy.signal.fftconvolve: the latter
     deadlocks under multiprocessing fork on macOS. KDE_PROMINENCE_FRAC must
     match mode_utils.detect_modes_from_arrays, which counts modes of the
     fitted QPDs. _count_kde_modes_reference is the direct-evaluation
     definition the binned path is validated against.

     Null: the KDE of the data at h_crit itself, sampled by the rescaled
     smoothed bootstrap
         y* = xbar + (1 + h_crit^2/s^2)^(-1/2) (x_I - xbar + h_crit * eps),
     which preserves the sample variance. p = (1 + #{h*_crit >= h_crit})/(B+1).
     The reproduction scripts reject at strict p < ALPHA with ALPHA = 0.05;
     note that with B = 199 the p-value grid has an atom exactly at 0.05, so
     the strict comparison tests at an effective 0.045.

     SCOPE: this calibration is exact only for k = 1 (Hall & York 2001). The
     k = 2 and k = 3 columns are reported as Silverman's own sequential
     procedure, uncalibrated. hall_york_adjust / silverman_test_calibrated
     provide the k = 1 calibration and are used only as a secondary report in
     the hydrology script -- none of the tables above apply them.

  LEGACY FISH-JITTER FAMILY  (jitter_find_h_crit, jitter_silverman_pvalue)
     Nothing in reproduction/ and nothing in the paper calls these. They keep
     their own JITTER_PROMINENCE = 0.05 and their own max(1, n_peaks) mode
     floor, deliberately NOT unified with KDE_PROMINENCE_FRAC. Do not call
     these for new work -- use apply_silverman_test.

  Hartigan dip test: apply_hartigan_dip_test is a thin wrapper over
  diptest.diptest and is the only dip-test entry point reproduction/ uses.
"""

import numpy as np
from scipy.signal import find_peaks


# ══════════════════════════════════════════════════════════════════════════════
# PAPER FAMILY (apply_silverman_test, apply_hartigan_dip_test)
# ══════════════════════════════════════════════════════════════════════════════

_SQRT_2PI = np.sqrt(2.0 * np.pi)

#: Identifies the Silverman implementation that produced a result. Long runs are
#: checkpointed and resumed, and a resume that silently mixes rows from two
#: implementations is unrecoverable after the fact -- it happened here. Scripts
#: write this into their partials and refuse to resume across a change.
SILVERMAN_IMPL = '2026-10-04.null-at-hcrit.rescaled.hhi.prom1pct'

#: Mode-detection prominence, as a fraction of the tallest peak. Must match
#: mode_utils.detect_modes_from_arrays, which counts modes of the fitted QPDs.
KDE_PROMINENCE_FRAC = 0.01


def _count_kde_modes_reference(x_sample, bandwidth, grid_size=2000):
    """Reference mode counter: the KDE evaluated exactly, point by point.

    This is the definition the fast path below is verified against. It is no
    longer called in production -- keep it, and keep it correct, because every
    equivalence claim in this module is an equality against THIS function.

    The kernel is evaluated UNNORMALISED, as exp(-z^2/2) rather than
    scipy.stats.norm.pdf(z), with the prominence threshold scaled by the same
    constant. Since norm.pdf(z) = exp(-z^2/2)/sqrt(2*pi), both the density and
    the threshold are multiplied by sqrt(2*pi), so every peak clears or fails
    the test exactly as before -- an algebraic identity, not an approximation.
    """
    x_sorted = np.sort(x_sample)
    x_min, x_max = x_sorted[0], x_sorted[-1]
    x_grid = np.linspace(x_min - 2 * bandwidth, x_max + 2 * bandwidth, grid_size)
    z = (x_grid[:, None] - x_sorted[None, :]) / bandwidth
    pdf_vals = np.mean(np.exp(-0.5 * z * z), axis=1)
    # 'A mode' of the KDE: a peak whose prominence is at least
    # KDE_PROMINENCE_FRAC of the tallest peak, the rule of
    # mode_utils.detect_modes_from_arrays. This counter was on an
    # ABSOLUTE 1e-4 instead, so 'a mode' meant one thing in the fits and another
    # inside Silverman's statistic. A relative threshold is also scale-free --
    # pounds or grams give the same count -- and on fish replicates it breaks the
    # monotonicity the bisection needs LESS often than the absolute one did
    # (4 violations vs 6 over the bracket, all at ~100 modes, far from k<=3).
    peaks, _ = find_peaks(pdf_vals, prominence=KDE_PROMINENCE_FRAC * pdf_vals.max())
    return len(peaks)


# Fine-grid points per bandwidth for the binned KDE below. At 8 the binning
# error is already far under the prominence threshold for peaks of ordinary
# height, but it flipped 9 of 300 test bandwidths by one mode -- always at mode
# counts between 26 and 243, never anywhere near k, so never able to move
# h_crit for k = 1, 2 or 3. 32 removed every one of those: 800 bandwidths across
# six fish replicates and four Johnson SU replicates, at both grid sizes in use,
# zero disagreements with _count_kde_modes_reference. Exactness was preferred to
# the extra speed 8 would have bought.
KDE_OVERSAMPLE = 32

# The Gaussian is truncated where it contributes less than ~1e-14 of its peak,
# which is below double precision's reach in a sum of this length.
KDE_KERNEL_CUT = 8.0


def _count_kde_modes(x_sample, bandwidth, grid_size=2000):
    """Count modes in a KDE with given bandwidth -- binned, via FFT.

    Returns exactly what _count_kde_modes_reference returns, computed in
    O(grid log grid) instead of O(grid * n). The density is read off the SAME
    grid_size evaluation points over the SAME interval and tested against the
    SAME prominence threshold; only the route to those values differs.

    The sample is linearly binned onto a fine grid of (grid_size - 1) * m + 1
    points over the same interval, which CONTAINS every one of the grid_size
    evaluation points at stride m -- so the coarse values are recovered by
    exact subsampling, never by interpolation. m is chosen per call to keep the
    fine spacing at or below bandwidth / KDE_OVERSAMPLE, which is what makes
    the binning error negligible at the small bandwidths the bisection visits
    at the bottom of its bracket as well as at the large ones.

    Equivalence is verified, not assumed: 800 bandwidths spanning 1 to 245
    modes over the full bisection bracket on fish (N = 3474) and Johnson SU
    (N = 200) give identical counts, and apply_silverman_test returns bit-
    identical p-values on seven (replicate, k) cells with k = 1, 2, 3. It is
    about 15x faster per mode count and about 32x faster per p-value, because
    the bandwidths the bisection actually visits are mostly moderate ones where
    m is small. See reproduction/scripts/verify_kde_equivalence.py.
    """
    x_sorted = np.sort(x_sample)
    lo = x_sorted[0] - 2 * bandwidth
    hi = x_sorted[-1] + 2 * bandwidth
    span = hi - lo

    m = int(np.ceil((span / (grid_size - 1)) / (bandwidth / KDE_OVERSAMPLE)))
    m = int(min(max(m, 1), 512))
    n_fine = (grid_size - 1) * m + 1
    delta = span / (n_fine - 1)

    # Linear binning: each observation is split between its two neighbouring
    # fine-grid nodes in proportion to distance. Simple (nearest-node) binning
    # would be O(delta) accurate instead of O(delta^2) and is not good enough.
    pos = (x_sorted - lo) / delta
    i0 = np.floor(pos).astype(np.int64)
    np.clip(i0, 0, n_fine - 2, out=i0)
    frac = pos - i0
    weights = np.zeros(n_fine)
    np.add.at(weights, i0, 1.0 - frac)
    np.add.at(weights, i0 + 1, frac)

    half = int(np.ceil(KDE_KERNEL_CUT * bandwidth / delta))
    offsets = np.arange(-half, half + 1) * (delta / bandwidth)
    kernel = np.exp(-0.5 * offsets * offsets)

    # Convolve with numpy.fft rather than scipy.signal.fftconvolve. scipy's FFT
    # keeps an internal thread pool; these scripts force multiprocessing's
    # 'fork' start method, and on macOS a forked child that inherits a live
    # thread pool deadlocks the first time it touches it -- the SU sweep hung
    # indefinitely with zero chunks completed. numpy.fft is single-threaded and
    # holds no pool, so it is fork-safe. Same arithmetic either way: verified
    # equal to fftconvolve's 'same' output to machine precision, and the mode
    # counts and p-values are unchanged.
    n_w = weights.size
    n_k = kernel.size
    n_full = n_w + n_k - 1
    n_fft = 1 << (n_full - 1).bit_length()          # power of two: fast, exact
    spec = np.fft.rfft(weights, n_fft) * np.fft.rfft(kernel, n_fft)
    full = np.fft.irfft(spec, n_fft)[:n_full]
    start = (n_k - 1) // 2                           # centre, as mode='same'
    density = full[start:start + n_w] / len(x_sorted)
    pdf_vals = density[::m]
    # 'A mode' of the KDE: a peak whose prominence is at least
    # KDE_PROMINENCE_FRAC of the tallest peak, the rule of
    # mode_utils.detect_modes_from_arrays. This counter was on an
    # ABSOLUTE 1e-4 instead, so 'a mode' meant one thing in the fits and another
    # inside Silverman's statistic. A relative threshold is also scale-free --
    # pounds or grams give the same count -- and on fish replicates it breaks the
    # monotonicity the bisection needs LESS often than the absolute one did
    # (4 violations vs 6 over the bracket, all at ~100 modes, far from k<=3).
    peaks, _ = find_peaks(pdf_vals, prominence=KDE_PROMINENCE_FRAC * pdf_vals.max())
    return len(peaks)


def _critical_bandwidth(data, k, h_lo, h_hi, grid_size, iters=40):
    """Smallest bandwidth whose KDE has <= k modes, by bisection.

    The upper bracket EXPANDS if it is not already smoothing enough. The old
    code used a fixed [0.01*h0, 5*h0]: for a bootstrap sample rougher than the
    original, the true critical bandwidth can exceed 5*h0, and bisection then
    silently returned the top of the bracket as if it were the answer.
    """
    if _count_kde_modes(data, h_lo, grid_size) <= k:
        return h_lo
    tries = 0
    while _count_kde_modes(data, h_hi, grid_size) > k and tries < 12:
        h_hi *= 2.0
        tries += 1
    for _ in range(iters):
        h_mid = 0.5 * (h_lo + h_hi)
        if _count_kde_modes(data, h_mid, grid_size) > k:
            h_lo = h_mid
        else:
            h_hi = h_mid
    # h_hi, not the midpoint: h_lo always has MORE than k modes and h_hi always
    # has at most k, so only h_hi is guaranteed to satisfy H0. The null density
    # is built at this bandwidth, and a null that violates its own hypothesis is
    # indefensible even when the bracket is 1e-12 wide, as it is after 40 steps.
    return h_hi


def apply_silverman_test(x_sample, n_bootstrap=199, k=1, grid_size=2000):
    """Silverman (1981) critical-bandwidth test, H0: the density has <= k modes.

    UNCALIBRATED -- no Hall & York adjustment anywhere, by project decision.

    The statistic is h_crit, the smallest bandwidth whose KDE has at most k
    modes. Under H0 a large h_crit is unlikely, so p = P(h_crit* >= h_crit).

    THE NULL. h_crit* is drawn from the KDE AT h_crit -- the roughest density
    consistent with H0 -- with Silverman's variance correction,

        y* = xbar + (1 + h^2/s^2)^(-1/2) * (x_I - xbar + h*eps),

    which keeps the null's variance equal to the sample's. Both halves of this
    were wrong before and the errors interacted:

      1. The null bandwidth was h0 when the KDE at h0 already had <= k modes,
         and 1.5*h_crit otherwise. Neither is h_crit, and the BRANCH DEPENDS ON
         k: for the fish data the KDE at h0 has 2 modes, so k=1 took one path
         and k=2,3 took the other. That is what made the nested results
         incoherent -- 26.8 percent of replicates rejected "<= 2 modes" while
         accepting "<= 1 mode", which is impossible, and rejection rose with k
         when a harder null must be rejected less often.
      2. The draw went through scipy's gaussian_kde.resample, which omits the
         rescaling: it returns x_I + h*eps, whose variance is s^2 + h^2. At the
         bandwidths k=1 uses that inflated the null's sd by more than 2x.

    The inner draw uses numpy's LEGACY global state (np.random.*), because every
    caller seeds per replicate with np.random.seed() to make single cells
    reproducible. Do not switch to default_rng without changing them too.
    """
    try:
        x = np.asarray(x_sample, dtype=float)
        n = x.size
        s = float(np.std(x, ddof=1))
        if not np.isfinite(s) or s <= 0:
            return None
        h0 = 1.06 * s * n ** -0.2
        h_lo, h_hi = 0.01 * h0, 5.0 * h0

        h_crit = _critical_bandwidth(x, k, h_lo, h_hi, grid_size)

        xbar = float(np.mean(x))
        scale = 1.0 / np.sqrt(1.0 + (h_crit * h_crit) / (s * s))

        ge = 0
        for _ in range(n_bootstrap):
            idx = np.random.randint(0, n, n)
            eps = np.random.standard_normal(n)
            y = xbar + scale * (x[idx] - xbar + h_crit * eps)
            if _critical_bandwidth(y, k, h_lo, h_hi, grid_size) >= h_crit:
                ge += 1

        # (ge + 1) / (B + 1): the observed value is one of the draws under H0,
        # which also keeps p strictly positive without the old ad-hoc floor.
        return float((ge + 1.0) / (n_bootstrap + 1.0))
    except Exception:
        return None


def apply_hartigan_dip_test(x_sample):
    """Hartigan dip test. Returns p-value or None."""
    try:
        import diptest
        _, p_value = diptest.diptest(x_sample)
        return float(p_value)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY FISH-JITTER FAMILY -- not used by the paper (see module docstring)
# ══════════════════════════════════════════════════════════════════════════════

JITTER_N_GRID     = 128   # KDE evaluation grid points
JITTER_PROMINENCE = 0.05  # Peak prominence threshold as fraction of max PDF value


def _jitter_kde_pdf(data, h, n_grid=JITTER_N_GRID):
    """Evaluate Gaussian KDE on a uniform grid. Returns (n_grid,) PDF array."""
    x_min = data.min() - 3.0 * h
    x_max = data.max() + 3.0 * h
    x_grid = np.linspace(x_min, x_max, n_grid)
    diff = (x_grid[:, None] - data[None, :]) / h          # (n_grid, N)
    pdf  = np.exp(-0.5 * diff ** 2).sum(axis=1)            # unnormalised (fine for mode counting)
    return pdf


def jitter_count_modes(data, h, n_grid=JITTER_N_GRID):
    """Number of modes in Gaussian KDE of data at bandwidth h."""
    pdf = _jitter_kde_pdf(data, h, n_grid)
    prom = max(JITTER_PROMINENCE * pdf.max(), 1e-10)
    peaks, _ = find_peaks(pdf, prominence=prom)
    return max(1, len(peaks))


def jitter_find_h_crit(data, k=1, n_grid=JITTER_N_GRID, tol=1e-5):
    """
    Find h_crit = smallest bandwidth at which KDE(data, h) has <= k modes.
    Uses bisection between h_low (multimodal) and h_high (unimodal).

    Returns h_crit (the crossover bandwidth from multimodal to unimodal).
    If data is already unimodal at very small h, returns a small h_fallback.
    """
    n   = len(data)
    std = data.std(ddof=1)
    if std < 1e-10:
        return 1e-4   # constant data -- trivially unimodal

    # Silverman's rule of thumb as a starting upper bound
    h_max = 0.9 * std * n ** (-0.2)

    # Expand h_max until KDE is unimodal (at most <= k modes)
    for _ in range(30):
        if jitter_count_modes(data, h_max, n_grid) <= k:
            break
        h_max *= 2.0

    # h_min: small enough to be multimodal
    h_min = h_max * 0.001
    for _ in range(50):
        if jitter_count_modes(data, h_min, n_grid) > k:
            break
        h_min /= 2.0
        if h_min < 1e-8:
            # Data may be genuinely unimodal at all scales -- return small h
            return h_max * 0.001

    # Bisection
    for _ in range(60):
        h_mid = (h_min + h_max) / 2.0
        if jitter_count_modes(data, h_mid, n_grid) <= k:
            h_max = h_mid
        else:
            h_min = h_mid
        if (h_max - h_min) < tol * h_max:
            break

    return h_max   # smallest h with <= k modes


def jitter_silverman_pvalue(data, h_crit, k=1, n_inner=100, rng=None, n_grid=JITTER_N_GRID):
    """
    Bootstrap p-value for Silverman's test.

    Generates n_inner variance-preserving bootstrap samples from KDE(data, h_crit)
    and computes the fraction with h_crit* >= h_crit.

    Variance preservation (Hall & York 2001):
        x* = mu + (X_{b(i)} - mu + h_crit * Z_i) / sqrt(1 + h_crit^2 / s^2)
    where b(i) is a bootstrap index and Z_i ~ N(0,1).
    """
    if rng is None:
        rng = np.random.default_rng()

    n   = len(data)
    mu  = data.mean()
    s2  = data.var(ddof=1)
    c   = np.sqrt(1.0 + h_crit ** 2 / max(s2, 1e-10))

    exceed = 0
    for _ in range(n_inner):
        idx    = rng.integers(0, n, size=n)
        noise  = rng.standard_normal(size=n) * h_crit
        boot   = mu + (data[idx] - mu + noise) / c
        h_boot = jitter_find_h_crit(boot, k=k, n_grid=n_grid)
        if h_boot >= h_crit:
            exceed += 1

    return exceed / n_inner

# ══════════════════════════════════════════════════════════════════════════════
# HALL & YORK (2001) CALIBRATION
#
# Silverman (1981, §3) is explicit that his procedure gives "a conservative
# assessment of the significance": the reference density g0 is rescaled to the
# sample variance, which makes it the most extreme k-modal density consistent
# with the data. Hall & York (2001) quantified that conservatism -- the test's
# actual asymptotic level is well below the nominal one -- and proposed a
# calibration.
#
# NOTE ON ATTRIBUTION: the variance-preserving resampling formula
#     x* = mu + (X_b - mu + h*Z) / sqrt(1 + h^2/s^2)
# used by jitter_silverman_pvalue above is Silverman's OWN construction
# (Silverman 1981 eq. in §3, crediting Efron 1979 for the smoothed-bootstrap
# idea), NOT the Hall & York correction. Applying that formula does not
# calibrate the test. The calibration is the separate step implemented here.
#
# The mapping below is the tabulated asymptotic level of Silverman's test as a
# function of the nominal level, from Hall & York (2001), as distributed in the
# reference R implementation (`silvermantest::silverman.test`, argument
# `adjust=TRUE`). It is valid only for k = 1 (testing unimodality).
# Reading the table: using a nominal cutoff of 0.05 gives an actual asymptotic
# level of about 0.010, i.e. the uncalibrated test rejects far less often than
# advertised.
_HY_NOMINAL = np.array([0.00, 0.005, 0.010, 0.020, 0.030, 0.040, 0.050, 0.06,
                        0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15,
                        0.16, 0.17, 0.18, 0.19, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
_HY_ACTUAL = np.array([0.000, 0.000, 0.000, 0.002, 0.004, 0.006, 0.010, 0.012,
                       0.016, 0.021, 0.025, 0.032, 0.038, 0.043, 0.050, 0.057,
                       0.062, 0.070, 0.079, 0.088, 0.094, 0.102, 0.149, 0.202,
                       0.252, 0.308, 0.423])


def hall_york_adjust(p_silverman):
    """
    Hall & York (2001) calibration of a Silverman unimodality-test p-value.

    Maps the raw (conservative) Silverman p-value through the tabulated
    nominal-level -> actual-level curve, so the calibrated p-value can be
    compared against alpha directly. Because the raw test is conservative the
    adjusted p-value is SMALLER than the raw one; equivalently, at alpha = 0.05
    the calibrated test rejects when the raw p-value falls below roughly 0.13.

    Only valid for k = 1 (H0: unimodal). Returns None if given None.

    Uses a monotone (PCHIP) interpolant rather than a natural cubic spline so
    the mapping cannot overshoot and return a non-monotone or negative
    p-value at the ends of the table -- an unconstrained cubic through these
    knots does dip below zero near the origin.
    """
    if p_silverman is None:
        return None
    from scipy.interpolate import PchipInterpolator
    p = float(np.clip(p_silverman, 0.0, 1.0))
    interp = PchipInterpolator(_HY_NOMINAL, _HY_ACTUAL, extrapolate=False)
    if p <= _HY_NOMINAL[-1]:
        return float(np.clip(interp(p), 0.0, 1.0))
    # Beyond the tabulated range (p > 0.5) the test is nowhere near rejecting;
    # extend linearly from the last knot to (1, 1) rather than extrapolating
    # the polynomial.
    lo_x, lo_y = _HY_NOMINAL[-1], _HY_ACTUAL[-1]
    return float(lo_y + (p - lo_x) * (1.0 - lo_y) / (1.0 - lo_x))


def silverman_test_calibrated(x_sample, n_bootstrap=199, k=1, grid_size=2000):
    """Silverman test returning both the raw and the Hall & York calibrated
    p-value, as {'p_raw': ..., 'p_hall_york': ...}. The calibration applies
    only to k=1; for k>1 'p_hall_york' is None."""
    p_raw = apply_silverman_test(x_sample, n_bootstrap=n_bootstrap, k=k, grid_size=grid_size)
    return {
        'p_raw': p_raw,
        'p_hall_york': hall_york_adjust(p_raw) if (k == 1 and p_raw is not None) else None,
    }
