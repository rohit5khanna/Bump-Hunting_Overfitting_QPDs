# Metalog 1.0 vs 2.0 - Version Comparison

## Overview

This package provides **both** versions of the Metalog distribution:
- **Metalog 1.0**: Original implementation from Keelin (2016)
- **Metalog 2.0**: Updated implementation from Baucells et al. (2024)

## Key Differences

### Terms 1-6: IDENTICAL ✓

Both versions use the same basis functions for terms 1-6:

| Term | Basis Function | Both Versions |
|------|----------------|---------------|
| 1 | 1 | ✓ |
| 2 | ℓ(y) | ✓ |
| 3 | (y-0.5)ℓ(y) | ✓ |
| 4 | (y-0.5) | ✓ |
| 5 | (y-0.5)² | ✓ |
| 6 | (y-0.5)²ℓ(y) | ✓ |

### Terms 7+: DIFFERENT ⚠️

Starting from term 7, the basis function assignment differs:

| Term | Metalog 1.0 (Keelin 2016) | Metalog 2.0 (Baucells 2024) |
|------|---------------------------|----------------------------|
| **7** | (y-0.5)³ | **(y-0.5)³ℓ(y)** ← SWAPPED |
| **8** | (y-0.5)³ℓ(y) | **(y-0.5)³** ← SWAPPED |
| 9 | (y-0.5)⁴ | (y-0.5)⁴ |
| 10 | (y-0.5)⁴ℓ(y) | (y-0.5)⁴ℓ(y) |
| **11** | (y-0.5)⁵ | **(y-0.5)⁵ℓ(y)** ← SWAPPED |
| **12** | (y-0.5)⁵ℓ(y) | **(y-0.5)⁵** ← SWAPPED |

Pattern: Terms {7, 8, 11, 12, 15, 16, ...} are swapped between versions.

## Mathematical Formulation

### Metalog 1.0 Rule (Original)

For terms i ≥ 5:
- power = (i-3)//2 + 1
- If (i-3) % 2 == 0: **(y-0.5)^power** (without logit)
- If (i-3) % 2 == 1: **(y-0.5)^power × ℓ(y)** (with logit)

### Metalog 2.0 Rule (Updated)

For all terms j:
- power = (j-1)//2
- If j % 4 ∈ {0, 1}: **(y-0.5)^power** (goes to μ, no logit)
- If j % 4 ∈ {2, 3}: **(y-0.5)^power × ℓ(y)** (goes to s, with logit)

## Why Metalog 2.0?

### Problem with 1.0

**Proposition 2** (Baucells et al. 2024) proves that Metalog 1.0 can produce:
- **Rank deficiency** for certain term counts (k ∈ {7, 11, 15, 19, ...})
- **Singular design matrices** → parameter estimation fails
- **Numerical instability** in edge cases

### Solution in 2.0

Metalog 2.0 is the **only balanced assignment** that guarantees:
- ✅ Full rank design matrix for all k ≤ n
- ✅ No degeneracies under symmetric percentiles
- ✅ Better numerical stability
- ✅ Explicit feasibility guarantees

## Usage

### Import Both Versions

```python
from metalog import Metalog_v1, Metalog_v2
from metalog import LogMetalog_v1, LogMetalog_v2
from metalog import LogitMetalog_v1, LogitMetalog_v2

# Default imports use v2 (recommended)
from metalog import Metalog, LogMetalog, LogitMetalog
```

### Example: Compare Results

```python
import numpy as np
from metalog import Metalog_v1, Metalog_v2

# Create test data
x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])

# Fit with both versions
m1 = Metalog_v1(x_data, y_data, terms=10)
m2 = Metalog_v2(x_data, y_data, terms=10)

# Compare coefficients
print(f"Coefficient 7 (v1): {m1.coefficients[6]:.4f}")
print(f"Coefficient 7 (v2): {m2.coefficients[6]:.4f}")
print(f"Difference: {abs(m1.coefficients[6] - m2.coefficients[6]):.4f}")

# Compare quantile predictions
y_test = np.array([0.25, 0.5, 0.75])
q1 = m1.quantile(y_test)
q2 = m2.quantile(y_test)
print(f"Max quantile difference: {np.max(np.abs(q1 - q2)):.4f}")
```

### When Results Differ

- **Terms ≤ 6**: Results are **IDENTICAL**
- **Terms ≥ 7**: Results **MAY DIFFER** due to different basis assignments
- **Practical impact**: Usually small for well-behaved data, but can be significant for edge cases

## Recommendations

### Use Metalog 2.0 (Default) When:
- ✅ Starting new projects
- ✅ Using 7+ terms
- ✅ Need guaranteed numerical stability
- ✅ Want to match latest research

### Use Metalog 1.0 Only When:
- ⚠️ Reproducing old results (backward compatibility)
- ⚠️ Comparing with legacy systems
- ⚠️ Historical analysis using original paper

### Safe for Either Version:
- ✓ Using ≤ 6 terms (results identical)
- ✓ Exploratory data analysis
- ✓ Teaching/learning metalog concepts

## Testing & Verification

Both implementations have been verified to:
- ✅ Produce identical results for terms 1-6
- ✅ Differ as expected for terms 7+
- ✅ Pass all unit tests
- ✅ Handle unbounded, semibounded, and bounded domains

## References

1. **Metalog 1.0**: Keelin, T. W. (2016). "The Metalog Distributions." *Decision Analysis*, 13(4), 243-277.
2. **Metalog 2.0**: Baucells, M., Chrisman, L., Keelin, T. W., & Xu, S. (2024). "On the Properties of the Metalog Distribution."
3. **Implementation details**: See `Documentation/METALOG_2_0_UPDATE_SUMMARY.md`

## Files in This Package

- **`metalog_v1.py`**: Metalog 1.0 implementation (Keelin 2016)
- **`metalog_v2.py`**: Metalog 2.0 implementation (Baucells et al. 2024)
- **`__init__.py`**: Package interface (defaults to v2)
- **`README.md`**: General documentation
- **`VERSION_COMPARISON.md`**: This file

---

**Recommendation**: Use **Metalog 2.0** (default) for all new work unless you specifically need backward compatibility with Metalog 1.0.
