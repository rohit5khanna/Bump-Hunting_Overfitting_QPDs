# Johnson Power-Series Expansions (JPSE)

Implementation of the Johnson distribution system using quantile functions as described in:

> Bickel, J. E. (2026). Quantile-based power-series expansions of the Johnson distribution system. *Communications in Statistics - Theory and Methods*. DOI: 10.1080/03610926.2025.2612230

## Overview

This module implements the three Johnson distribution families (SU, SL, SB) using their exact quantile function representations. The Johnson system is defined by transforming the standard normal distribution:

```
Z = c + d·g((X - η)/κ),  where Z ~ N(0,1)
```

Solving for X gives the quantile functions.

## Distribution Families

### 1. Johnson SU (Unbounded)

**Support**: X ∈ (-∞, ∞)

**Quantile Function**:
```
Q_SU(p) = η + κ · sinh((Φ^(-1)(p) - c) / d)
```

where:
- `sinh` is the hyperbolic sine function
- `Φ^(-1)(p)` is the standard normal quantile function

**Use Case**: Modeling unbounded continuous variables with flexible skewness and kurtosis.

### 2. Johnson SL (Semi-bounded)

**Support**: X ∈ (η, ∞)

**Quantile Function**:
```
Q_SL(p) = η + κ · exp((Φ^(-1)(p) - c) / d)
```

where:
- `exp` is the exponential function
- η is the lower bound

**Use Case**: Modeling positive continuous variables (like prices, waiting times).

### 3. Johnson SB (Bounded)

**Support**: X ∈ (η, η + κ)

**Quantile Function**:
```
Q_SB(p) = η + κ · logit^(-1)((Φ^(-1)(p) - c) / d)
```

where:
- `logit^(-1)(x) = 1/(1 + exp(-x))` is the logistic sigmoid function
- Support is bounded between η and η + κ

**Use Case**: Modeling proportions, probabilities, or any variable constrained to a finite interval.

## Parameters

All three distributions share the same four-parameter structure:

| Parameter | Symbol | Description | Constraints |
|-----------|--------|-------------|-------------|
| Location | η (eta) | Location parameter / lower bound | Real number |
| Scale | κ (kappa) | Scale parameter / width | κ > 0 |
| Shape | c | Shape parameter (affects center) | Real number |
| Shape/Scale | d | Shape/scale parameter | d > 0 |

## Installation & Usage

### Basic Usage

```python
from jpse import JohnsonSU, JohnsonSL, JohnsonSB
import numpy as np

# Figure 1 parameters from Bickel (2026)
params = {'eta': 0, 'kappa': 1, 'c': 0.5, 'd': 1.2}

# Create distributions
su = JohnsonSU(**params)
sl = JohnsonSL(**params)
sb = JohnsonSB(**params)

# Generate samples
samples_su = su.rvs(size=10000)
samples_sl = sl.rvs(size=10000)
samples_sb = sb.rvs(size=10000)

# Compute quantiles
median_su = su.quantile(0.5)
q95_sl = sl.quantile(0.95)

# Compute PDF at specific points
x = np.linspace(-2, 2, 100)
pdf_su = su.pdf(x)
```

### Replicating Figure 1

Run the replication script to generate the figures from the paper:

```bash
cd jpse
python replicate_figure1.py
```

This will:
1. Create the three Johnson distributions with Figure 1 parameters
2. Generate 100,000 samples from each distribution
3. Compute PDFs using the quantile-density relation
4. Generate plots comparing quantile-based PDFs with kernel density estimates
5. Save `figure1_combined.png` and `figure1_replication.png`

## Implementation Details

### Quantile-Based Simulation

Random samples are generated using the **probability integral transform**:

```python
U ~ Uniform(0, 1)
X = Q(U)  # Apply quantile function
```

This is mathematically equivalent to inverse transform sampling and ensures that X follows the Johnson distribution.

### PDF Computation

PDFs are computed using the **quantile-density relation**:

```
f(Q(p)) = 1 / q(p)
```

where `q(p) = dQ(p)/dp` is the quantile density function.

The steps are:
1. Create a fine grid of probability values p ∈ (0,1)
2. Compute Q(p) for each p
3. Numerically differentiate to get q(p) = dQ(p)/dp
4. Compute PDF as f(Q(p)) = 1/q(p)
5. Map from (p, f(Q(p))) space to (x, f(x)) space

This approach:
- ✅ Avoids numerical inversion of the quantile function
- ✅ Provides accurate PDFs directly from quantile functions
- ✅ Matches the paper's quantile-based framework

### Validation

The implementation includes validation via kernel density estimates (KDE) from simulated samples. The close agreement between quantile-based PDFs and KDEs confirms correctness.

## File Structure

```
jpse/
├── __init__.py              # Package initialization
├── johnson.py               # Core implementation of three Johnson families
├── replicate_figure1.py     # Script to replicate Figure 1 from paper
├── README.md                # This file
├── figure1_combined.png     # Generated: Combined plot of all three distributions
└── figure1_replication.png  # Generated: Three-panel validation plot
```

## Results

### Figure 1 Parameters

All three distributions in Figure 1 use:
- η = 0
- κ = 1
- c = 0.5
- d = 1.2

### Summary Statistics (100,000 samples)

**Johnson SU (unbounded)**:
- Sample mean: -0.610
- Sample median: -0.427
- Theoretical median: -0.429

**Johnson SL (semi-bounded)**:
- Sample mean: 0.937
- Sample median: 0.662
- Theoretical median: 0.659

**Johnson SB (bounded on (0,1))**:
- Sample mean: 0.411
- Sample median: 0.398
- Theoretical median: 0.397
- All samples in (0,1): ✓

## Mathematical Correspondence

The quantile functions implemented here correspond exactly to equations (9), (10), and (11) in Bickel (2026):

| Family | Equation in Paper | Implementation |
|--------|-------------------|----------------|
| SU | Eq. (9) | `eta + kappa * np.sinh(z)` |
| SL | Eq. (10) | `eta + kappa * np.exp(z)` |
| SB | Eq. (11) | `eta + kappa * expit(z)` |

where `z = (norm.ppf(p) - c) / d` is the centered and scaled standard normal quantile.

## Dependencies

- `numpy` - Array operations and numerical functions
- `scipy` - Statistical functions (norm.ppf, expit)
- `matplotlib` - Plotting (for replication script)

## References

1. Bickel, J. E. (2026). Quantile-based power-series expansions of the Johnson distribution system. *Communications in Statistics - Theory and Methods*. DOI: 10.1080/03610926.2025.2612230

2. Johnson, N. L. (1949). Systems of frequency curves generated by methods of translation. *Biometrika*, 36(1/2), 149-176.

3. Gilchrist, W. (2000). *Statistical Modelling with Quantile Functions*. Chapman and Hall/CRC.

## Future Extensions

Potential extensions for MC_UnimodalTruth experiments:
- Add wrapper functions for easy Monte Carlo simulation
- Implement analytical PDF derivatives (if available)
- Add parameter estimation methods
- Create visualization utilities for quantile function analysis
