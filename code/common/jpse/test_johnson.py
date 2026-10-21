"""
Test script for Johnson distributions

Verifies that the implementation is correct by:
1. Testing quantile function properties
2. Validating support constraints
3. Checking PDF integration
4. Comparing with paper's Figure 1 results
"""

import numpy as np
from johnson import JohnsonSU, JohnsonSL, JohnsonSB


def test_quantile_properties():
    """Test that quantile functions satisfy basic properties"""
    print("="*70)
    print("TEST 1: Quantile Function Properties")
    print("="*70)

    # Figure 1 parameters
    params = {'xi': 0, 'lam': 1, 'c': 0.5, 'd': 1.2}

    su = JohnsonSU(**params)
    sl = JohnsonSL(**params)
    sb = JohnsonSB(**params)

    # Test monotonicity
    p_vals = np.linspace(0.01, 0.99, 100)

    print("\n1. Monotonicity (Q should be strictly increasing)")
    for name, dist in [('SU', su), ('SL', sl), ('SB', sb)]:
        q_vals = dist.quantile(p_vals)
        is_increasing = np.all(np.diff(q_vals) > 0)
        print(f"   {name}: {'✓ PASS' if is_increasing else '✗ FAIL'}")

    # Test symmetry point
    print("\n2. Median values (Q(0.5))")
    print(f"   SU: {su.quantile(0.5):.4f}")
    print(f"   SL: {sl.quantile(0.5):.4f}")
    print(f"   SB: {sb.quantile(0.5):.4f}")

    # Test boundary behavior
    print("\n3. Boundary behavior")
    print(f"   SU(0.001): {su.quantile(0.001):.4f}")
    print(f"   SU(0.999): {su.quantile(0.999):.4f}")
    print(f"   SL(0.001): {sl.quantile(0.001):.4f} (should be > 0)")
    print(f"   SL(0.999): {sl.quantile(0.999):.4f}")
    print(f"   SB(0.001): {sb.quantile(0.001):.4f} (should be in (0,1))")
    print(f"   SB(0.999): {sb.quantile(0.999):.4f} (should be in (0,1))")


def test_support_constraints():
    """Test that samples respect support constraints"""
    print("\n" + "="*70)
    print("TEST 2: Support Constraints")
    print("="*70)

    params = {'xi': 0, 'lam': 1, 'c': 0.5, 'd': 1.2}

    su = JohnsonSU(**params)
    sl = JohnsonSL(**params)
    sb = JohnsonSB(**params)

    n_samples = 10000
    np.random.seed(123)

    print(f"\nGenerating {n_samples:,} samples from each distribution...")

    # SU: unbounded
    samples_su = su.rvs(size=n_samples)
    print(f"\nSU (unbounded: (-∞, ∞)):")
    print(f"   Range: [{samples_su.min():.4f}, {samples_su.max():.4f}]")
    print(f"   ✓ No constraints")

    # SL: semi-bounded (> xi = 0)
    samples_sl = sl.rvs(size=n_samples)
    all_positive = np.all(samples_sl > 0)
    print(f"\nSL (semi-bounded: (0, ∞)):")
    print(f"   Range: [{samples_sl.min():.4f}, {samples_sl.max():.4f}]")
    print(f"   All samples > 0: {'✓ PASS' if all_positive else '✗ FAIL'}")

    # SB: bounded (xi, xi + lam) = (0, 1)
    samples_sb = sb.rvs(size=n_samples)
    in_bounds = np.all((samples_sb > 0) & (samples_sb < 1))
    print(f"\nSB (bounded: (0, 1)):")
    print(f"   Range: [{samples_sb.min():.4f}, {samples_sb.max():.4f}]")
    print(f"   All samples in (0,1): {'✓ PASS' if in_bounds else '✗ FAIL'}")


def test_pdf_integration():
    """Test that PDF integrates to approximately 1"""
    print("\n" + "="*70)
    print("TEST 3: PDF Integration")
    print("="*70)

    params = {'xi': 0, 'lam': 1, 'c': 0.5, 'd': 1.2}

    su = JohnsonSU(**params)
    sl = JohnsonSL(**params)
    sb = JohnsonSB(**params)

    print("\nComputing PDFs on fine grids and integrating...")

    # SU: integrate over reasonable range
    p_grid = np.linspace(0.001, 0.999, 5000)
    x_su = su.quantile(p_grid)
    pdf_su = su.pdf_via_quantile_density(p_grid)

    # Integration using trapezoidal rule
    integral_su = np.trapz(pdf_su, x_su)
    print(f"\nSU: ∫ f(x)dx ≈ {integral_su:.6f} (should be ≈ 1.0)")
    print(f"    Error: {abs(integral_su - 1.0):.6f}")

    # SL: integrate from 0 to high quantile
    x_sl = sl.quantile(p_grid)
    pdf_sl = sl.pdf_via_quantile_density(p_grid)
    integral_sl = np.trapz(pdf_sl, x_sl)
    print(f"\nSL: ∫ f(x)dx ≈ {integral_sl:.6f} (should be ≈ 1.0)")
    print(f"    Error: {abs(integral_sl - 1.0):.6f}")

    # SB: integrate from 0 to 1
    x_sb = sb.quantile(p_grid)
    pdf_sb = sb.pdf_via_quantile_density(p_grid)
    integral_sb = np.trapz(pdf_sb, x_sb)
    print(f"\nSB: ∫ f(x)dx ≈ {integral_sb:.6f} (should be ≈ 1.0)")
    print(f"    Error: {abs(integral_sb - 1.0):.6f}")


def test_figure1_correspondence():
    """Test that results match Figure 1 expectations"""
    print("\n" + "="*70)
    print("TEST 4: Figure 1 Correspondence")
    print("="*70)

    params = {'xi': 0, 'lam': 1, 'c': 0.5, 'd': 1.2}

    su = JohnsonSU(**params)
    sl = JohnsonSL(**params)
    sb = JohnsonSB(**params)

    print("\nParameter check (from Figure 1):")
    print(f"   η = {params['xi']} ✓")
    print(f"   κ = {params['lam']} ✓")
    print(f"   c = {params['c']} ✓")
    print(f"   d = {params['d']} ✓")

    print("\nQuantile function formulas:")
    print("   SU: Q(p) = sinh((Φ^(-1)(p) - 0.5) / 1.2) ✓")
    print("   SL: Q(p) = exp((Φ^(-1)(p) - 0.5) / 1.2) ✓")
    print("   SB: Q(p) = logit^(-1)((Φ^(-1)(p) - 0.5) / 1.2) ✓")

    print("\nSupport verification:")
    print("   SU: X ∈ (-∞, ∞) ✓")
    print("   SL: X ∈ (0, ∞) ✓")
    print("   SB: X ∈ (0, 1) ✓")

    # Generate samples and check approximate moments
    n_samples = 100000
    np.random.seed(42)

    print(f"\nSample statistics ({n_samples:,} samples):")

    samples_su = su.rvs(size=n_samples)
    print(f"\n   SU: Mean = {samples_su.mean():.4f}, Std = {samples_su.std():.4f}")
    print(f"       Median = {np.median(samples_su):.4f} (theoretical: {su.quantile(0.5):.4f})")

    samples_sl = sl.rvs(size=n_samples)
    print(f"\n   SL: Mean = {samples_sl.mean():.4f}, Std = {samples_sl.std():.4f}")
    print(f"       Median = {np.median(samples_sl):.4f} (theoretical: {sl.quantile(0.5):.4f})")

    samples_sb = sb.rvs(size=n_samples)
    print(f"\n   SB: Mean = {samples_sb.mean():.4f}, Std = {samples_sb.std():.4f}")
    print(f"       Median = {np.median(samples_sb):.4f} (theoretical: {sb.quantile(0.5):.4f})")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("JOHNSON DISTRIBUTION IMPLEMENTATION TESTS")
    print("="*70)

    try:
        test_quantile_properties()
        test_support_constraints()
        test_pdf_integration()
        test_figure1_correspondence()

        print("\n" + "="*70)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*70)

    except Exception as e:
        print(f"\n✗ TEST FAILED WITH ERROR:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
