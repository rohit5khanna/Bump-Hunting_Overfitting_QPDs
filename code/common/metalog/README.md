# Metalog Distributions (Version 2.0)

A Python implementation of metalog quantile-parameterized distributions with updated basis function ordering for improved numerical stability.

## Overview

The metalog distribution is a flexible quantile-based distribution family introduced by Keelin (2016). This implementation uses the **Metalog 2.0** basis function ordering from "On the Properties of Metalog" which ensures full rank design matrices and avoids degeneracies that could occur in the original ordering.

The metalog supports unbounded, semibounded, and bounded distributions through log and logit transforms.

## Module Structure

| File | Purpose |
|------|---------|
| `metalog_distributions.py` | Core implementation with `Metalog`, `LogMetalog`, and `LogitMetalog` classes |
| `__init__.py` | Public API exports |

## Quick Start

```python
from metalog import Metalog, LogMetalog, LogitMetalog

# Fit an unbounded distribution
metalog = Metalog(x_data, y_data, terms=5)

# Semibounded distribution (e.g., fish weight, income)
log_metalog = LogMetalog(x_data, y_data, lower_bound=0, terms=7)

# Bounded distribution (e.g., proportions, bounded measurements)
logit_metalog = LogitMetalog(x_data, y_data, lower_bound=0, upper_bound=10, terms=5)

# Check feasibility
if metalog.is_feasible:
    print("Distribution is feasible (monotonic quantile function)")
    
# Evaluate quantile function
q = metalog.quantile(0.95)  # 95th percentile

# Evaluate PDF
pdf_value = metalog.pdf(5.0)

# Evaluate CDF
cdf_value = metalog.cdf(5.0)

# Get moments
mean = metalog.mean()
variance = metalog.variance()
```

## Basis Functions (Metalog 2.0)

The metalog uses two basis function families:

- **μ(y)**: Functions without logit transformation
  - `μ₁(y) = 1`
  - `μ₃(y) = y - 0.5`
  - For `n ≥ 5`: Selected terms based on position in sequence

- **s(y)**: Functions with logit transformation
  - `s₂(y) = ln(y/(1-y))`
  - `s₄(y) = (y - 0.5)·ln(y/(1-y))`
  - For `n ≥ 5`: Complementary terms to μ family

### Metalog 2.0 Ordering (Terms ≥ 5)

For term `n`, let `j = n - 4`, then:
- If `j mod 4 ∈ {1, 2}`: Use μ family, `k = (j + 1) // 2`
  - If `k` is even: `(y - 0.5)^k`
  - If `k` is odd: `(y - 0.5)^(k-1) × y^k`
- If `j mod 4 ∈ {3, 0}`: Use s family with same power rules

This ordering ensures the design matrix always has full rank.

## Class Overview

### `Metalog`
- **Domain**: Unbounded (-∞, ∞)
- **Use case**: General-purpose distributions without domain constraints
- **Quantile function**: Direct linear combination of basis functions

### `LogMetalog`
- **Domain**: Semibounded [lower_bound, ∞)
- **Use case**: Strictly positive quantities (income, weight, time)
- **Transform**: `z = ln(x - lower_bound)`
- **Quantile function**: `x = lower_bound + exp(Q_z(y))`

### `LogitMetalog`
- **Domain**: Bounded [lower_bound, upper_bound]
- **Use case**: Proportions, percentages, bounded measurements
- **Transform**: `z = ln((x - lower_bound)/(upper_bound - x))`
- **Quantile function**: `x = lower_bound + (upper_bound - lower_bound) / (1 + exp(-Q_z(y)))`

## Feasibility

A metalog distribution is feasible if:
1. The quantile function `Q(y)` is strictly monotonically increasing
2. The PDF `f(x) = 1 / Q'(y)` is strictly positive everywhere

Feasibility is automatically checked on a fine grid (999 points) during initialization.

## Key Features

- ✅ **Metalog 2.0 basis ordering** for numerical stability
- ✅ **Automatic feasibility checking** with detailed diagnostics
- ✅ **Efficient PDF/CDF computation** with adaptive numerical differentiation
- ✅ **Moment calculation** (mean, variance, skewness, kurtosis)
- ✅ **Support for unbounded, semibounded, and bounded domains**
- ✅ **Consistent with QFlex** for fair comparison

## W1 Distance Metric

The W1 normalized metric measures fit quality:

```
W1_norm = (∫|Q_fit(y) - Q_data(y)| dy) / (Q90 - Q10)
```

Lower W1 indicates better fit. Typical good values: 0.01-0.03.

## References

1. Keelin, T. W. (2016). "The Metalog Distributions." *Decision Analysis*, 13(4), 243-277.
2. "On the Properties of Metalog" - Updated basis function ordering to ensure full rank
3. See `Documentation/METALOG_2_0_UPDATE_SUMMARY.md` for details on the basis function changes

## Version History

- **v2.0.0**: Updated basis function ordering (Metalog 2.0), consistent feasibility grid with QFlex
- **v1.0.0**: Initial implementation based on Keelin (2016)
