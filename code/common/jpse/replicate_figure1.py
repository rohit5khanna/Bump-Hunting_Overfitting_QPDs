"""
Replicate Figure 1 from Bickel (2026)

This script generates the three Johnson distributions (SU, SL, SB) shown in
Figure 1 of the paper, using the exact parameter values and quantile functions
specified in the text.

Reference:
    Bickel, J. E. (2026). Quantile-based power-series expansions of the Johnson
    distribution system. Communications in Statistics - Theory and Methods.
    DOI: 10.1080/03610926.2025.2612230

Figure 1 Parameters (all three distributions):
    η = 0
    κ = 1
    c = 0.5
    d = 1.2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from johnson import JohnsonSU, JohnsonSL, JohnsonSB


def main():
    print("="*70)
    print("REPLICATING FIGURE 1 FROM BICKEL (2026)")
    print("="*70)

    # Figure 1 parameters (same for all three distributions)
    params = {
        'xi': 0.0,
        'lam': 1.0,
        'c': 0.5,
        'd': 1.2
    }

    print(f"\nParameters (from Figure 1):")
    print(f"  η = {params['xi']}")
    print(f"  κ = {params['lam']}")
    print(f"  c = {params['c']}")
    print(f"  d = {params['d']}")

    # Create the three Johnson distributions
    print("\n" + "="*70)
    print("DISTRIBUTION DEFINITIONS")
    print("="*70)

    su_dist = JohnsonSU(**params)
    sl_dist = JohnsonSL(**params)
    sb_dist = JohnsonSB(**params)

    print("\n1. Johnson SU (unbounded: X ∈ (-∞, ∞))")
    print(f"   Q_SU(p) = η + κ * sinh((Φ^(-1)(p) - c) / d)")
    print(f"   Q_SU(p) = sinh((Φ^(-1)(p) - 0.5) / 1.2)")
    print(f"   {su_dist}")

    print("\n2. Johnson SL (semi-bounded: X ∈ (η, ∞) = (0, ∞))")
    print(f"   Q_SL(p) = η + κ * exp((Φ^(-1)(p) - c) / d)")
    print(f"   Q_SL(p) = exp((Φ^(-1)(p) - 0.5) / 1.2)")
    print(f"   {sl_dist}")

    print("\n3. Johnson SB (bounded: X ∈ (η, η+κ) = (0, 1))")
    print(f"   Q_SB(p) = η + κ * logit^(-1)((Φ^(-1)(p) - c) / d)")
    print(f"   Q_SB(p) = logit^(-1)((Φ^(-1)(p) - 0.5) / 1.2)")
    print(f"   {sb_dist}")

    # Generate samples
    print("\n" + "="*70)
    print("GENERATING SAMPLES")
    print("="*70)

    n_samples = 100000
    np.random.seed(42)

    print(f"\nGenerating {n_samples:,} samples from each distribution...")

    samples_su = su_dist.rvs(size=n_samples)
    samples_sl = sl_dist.rvs(size=n_samples)
    samples_sb = sb_dist.rvs(size=n_samples)

    print(f"\n✓ SU samples: min={samples_su.min():.4f}, max={samples_su.max():.4f}, mean={samples_su.mean():.4f}")
    print(f"✓ SL samples: min={samples_sl.min():.4f}, max={samples_sl.max():.4f}, mean={samples_sl.mean():.4f}")
    print(f"✓ SB samples: min={samples_sb.min():.4f}, max={samples_sb.max():.4f}, mean={samples_sb.mean():.4f}")

    # Compute PDFs using quantile-density relation
    print("\n" + "="*70)
    print("COMPUTING PDFs VIA QUANTILE-DENSITY RELATION")
    print("="*70)

    print("\nMethod: f(Q(p)) = 1 / q(p), where q(p) = dQ(p)/dp")

    # Create fine grid of p values
    p_grid = np.linspace(0.001, 0.999, 2000)

    # Compute Q(p) and f(Q(p)) for each distribution
    print("\nComputing SU PDF...")
    x_su = su_dist.quantile(p_grid)
    pdf_su = su_dist.pdf_via_quantile_density(p_grid)

    print("Computing SL PDF...")
    x_sl = sl_dist.quantile(p_grid)
    pdf_sl = sl_dist.pdf_via_quantile_density(p_grid)

    print("Computing SB PDF...")
    x_sb = sb_dist.quantile(p_grid)
    pdf_sb = sb_dist.pdf_via_quantile_density(p_grid)

    # Compute kernel density estimates for validation
    print("\nComputing kernel density estimates from samples...")
    kde_su = gaussian_kde(samples_su)
    kde_sl = gaussian_kde(samples_sl)
    kde_sb = gaussian_kde(samples_sb)

    # Create Figure 1
    print("\n" + "="*70)
    print("CREATING FIGURE 1 REPLICATION")
    print("="*70)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Common plot settings
    colors = {'su': '#1f77b4', 'sl': '#ff7f0e', 'sb': '#2ca02c'}
    linewidth = 2.5

    # Panel 1: Johnson SU
    ax = axes[0]
    ax.plot(x_su, pdf_su, color=colors['su'], lw=linewidth, label='SU (quantile-based)')

    # Overlay KDE for validation
    x_kde = np.linspace(samples_su.min(), samples_su.max(), 500)
    ax.plot(x_kde, kde_su(x_kde), 'k--', lw=1.5, alpha=0.5, label='SU (KDE from samples)')

    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Johnson SU\n(unbounded)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Johnson SL
    ax = axes[1]
    ax.plot(x_sl, pdf_sl, color=colors['sl'], lw=linewidth, label='SL (quantile-based)')

    # Overlay KDE
    x_kde = np.linspace(samples_sl.min(), np.percentile(samples_sl, 99), 500)
    ax.plot(x_kde, kde_sl(x_kde), 'k--', lw=1.5, alpha=0.5, label='SL (KDE from samples)')

    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Johnson SL\n(semi-bounded)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, np.percentile(samples_sl, 99))

    # Panel 3: Johnson SB
    ax = axes[2]
    ax.plot(x_sb, pdf_sb, color=colors['sb'], lw=linewidth, label='SB (quantile-based)')

    # Overlay KDE
    x_kde = np.linspace(0.001, 0.999, 500)
    ax.plot(x_kde, kde_sb(x_kde), 'k--', lw=1.5, alpha=0.5, label='SB (KDE from samples)')

    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Johnson SB\n(bounded on (0,1))', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.suptitle('Figure 1 Replication: Johnson Distributions (η=0, κ=1, c=0.5, d=1.2)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    output_file = 'figure1_replication.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")

    # Create combined plot (like original Figure 1)
    print("\nCreating combined plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all three PDFs on same axes
    ax.plot(x_su, pdf_su, color=colors['su'], lw=linewidth, label='SU (unbounded)', zorder=3)
    ax.plot(x_sl, pdf_sl, color=colors['sl'], lw=linewidth, label='SL (semi-bounded)', zorder=2)
    ax.plot(x_sb, pdf_sb, color=colors['sb'], lw=linewidth, label='SB (bounded)', zorder=1)

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Johnson Distributions: SU, SL, and SB\n(η=0, κ=1, c=0.5, d=1.2)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 4)

    # Add support annotations
    ax.text(-1.5, 0.05, 'SU: (-∞, ∞)', fontsize=9, color=colors['su'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(2.5, 0.3, 'SL: (0, ∞)', fontsize=9, color=colors['sl'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.5, 1.2, 'SB: (0, 1)', fontsize=9, color=colors['sb'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    output_file = 'figure1_combined.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    print("\nJohnson SU (unbounded):")
    print(f"  Sample mean:   {samples_su.mean():.4f}")
    print(f"  Sample median: {np.median(samples_su):.4f}")
    print(f"  Sample std:    {samples_su.std():.4f}")
    print(f"  Theoretical median (Q(0.5)): {su_dist.quantile(0.5):.4f}")

    print("\nJohnson SL (semi-bounded):")
    print(f"  Sample mean:   {samples_sl.mean():.4f}")
    print(f"  Sample median: {np.median(samples_sl):.4f}")
    print(f"  Sample std:    {samples_sl.std():.4f}")
    print(f"  Theoretical median (Q(0.5)): {sl_dist.quantile(0.5):.4f}")

    print("\nJohnson SB (bounded):")
    print(f"  Sample mean:   {samples_sb.mean():.4f}")
    print(f"  Sample median: {np.median(samples_sb):.4f}")
    print(f"  Sample std:    {samples_sb.std():.4f}")
    print(f"  Theoretical median (Q(0.5)): {sb_dist.quantile(0.5):.4f}")
    print(f"  Support check: All samples in (0,1)? {(samples_sb > 0).all() and (samples_sb < 1).all()}")

    print("\n" + "="*70)
    print("✓ REPLICATION COMPLETE")
    print("="*70)
    print("\nNotes:")
    print("1. PDFs computed using quantile-density relation: f(Q(p)) = 1/q(p)")
    print("2. KDE overlays validate the quantile-based PDF computation")
    print("3. All distributions use the exact parameters from Figure 1")
    print("4. Quantile functions match equations (9), (10), (11) in the paper")

    # plt.show()  # Comment out to avoid blocking


if __name__ == "__main__":
    main()
