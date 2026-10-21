"""
Rounding rule for mode-share columns (1 mode % / 2 modes % / >2 modes %).

The author's rule for the paper: the shares in a row are a PARTITION of the
valid fits, so the printed values must sum to exactly 100.0. Rounding each
cell to its nearest 0.1 independently does not guarantee that (a row can
sum to 99.9 or 100.1). This module applies largest-remainder (Hamilton)
rounding instead: every cell is first rounded DOWN, then the shortfall is
handed, 0.1 at a time, to the cells with the largest remainders.

Consequences worth knowing:
  * a row always sums to exactly 100.0;
  * any single cell is within 0.1 of its exact value, but may be 0.1 away
    from its nearest-rounded value (e.g. 7.6477 can print as 7.7);
  * when nearest rounding already sums to 100.0 the result is IDENTICAL to
    nearest rounding.

Integer arithmetic throughout (counts * 1000 // n), so there are no
floating-point floor/remainder slips. Ties in the remainder go to the
earlier column.
"""
import numpy as np


def split_pct(counts, decimals=1):
    """Percentages of a partition, rounded so they sum to exactly 100.

    counts: non-negative integers whose sum is the denominator.
    Returns a list of floats with `decimals` decimal places, or NaNs if the
    denominator is zero.
    """
    counts = [int(c) for c in counts]
    n = sum(counts)
    if n == 0:
        return [np.nan] * len(counts)
    total = 100 * 10 ** decimals                 # 1000 units of 0.1 %
    units = [total * c // n for c in counts]     # floor
    rem = [total * c % n for c in counts]        # exact integer remainders
    short = total - sum(units)
    for i in sorted(range(len(counts)), key=lambda i: -rem[i])[:short]:
        units[i] += 1
    return [round(u / 10 ** decimals, decimals) for u in units]


if __name__ == '__main__':
    assert split_pct([66, 745, 52]) == [7.7, 86.3, 6.0]
    assert split_pct([180, 820, 0]) == [18.0, 82.0, 0.0]
    assert abs(sum(split_pct([37, 349, 103])) - 100.0) < 1e-9
    print('table_rounding self-test: OK')
