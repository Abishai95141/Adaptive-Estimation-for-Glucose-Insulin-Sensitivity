"""F-1: The Stage-2 design matrix is exactly rank-deficient.

Demonstrates:
  (a) rank(X2) = 4 with 5 columns; the null vector is exact to machine precision
  (b) the reported tau changes when the V column is rescaled (causally meaningless)
  (c) the estimator is NOT unit-equivariant in the treatment variable

Run from the repository root:  python audit/repro/check_rank_deficiency.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import banner, center, ols, stable_ols_coef, two_stage  # noqa: E402

T = 288
rng = np.random.RandomState(0)
U = rng.randn(T)
Z = 0.8 * U + 0.2 * rng.randn(T)
W = 0.8 * (0.9 * U) + 0.2 * rng.randn(T)
A = -1.5 * U + 0.1 * rng.randn(T)
TAU_TRUE = -8.0
Y = TAU_TRUE * A + 15.0 * U + 2.0 * rng.randn(T)
Yr, Ar, Zr, Wr = map(center, (Y, A, Z, W))

# ---------------------------------------------------------------- (a) rank
banner("(a) Stage-2 design matrix  X2 = [1, A, Z, W, V]")
tau_cf, X2, b1, V = two_stage(Ar, Zr, Wr, Yr)
s = np.linalg.svd(X2, compute_uv=False)
print("shape:             ", X2.shape)
print("singular values:   ", np.array2string(s, precision=4))
print("condition number:   %.3e" % (s[0] / s[-1]))
print("numpy matrix_rank:  %d  of %d columns" % (np.linalg.matrix_rank(X2), X2.shape[1]))

null_vec = np.array([-b1[0], 1.0, -b1[1], -b1[2], -1.0])
print("||X2 @ null_vec||:  %.3e   (0 => exactly collinear)"
      % np.linalg.norm(X2 @ null_vec))

print("\nreported tau_cf (beta[1]) = %8.4f     [TRUE tau = %.1f]" % (tau_cf, TAU_TRUE))
print("textbook CF   Y~[1,A,V]   = %8.4f"
      % ols(np.column_stack([np.ones(T), Ar, V]), Yr)[1])
print("proxy-adj     Y~[1,A,Z,W] = %8.4f"
      % ols(np.column_stack([np.ones(T), Ar, Zr, Wr]), Yr)[1])
print("naive         Y~[1,A]     = %8.4f" % (np.sum(Ar * Yr) / np.sum(Ar ** 2)))

# --------------------------------------------------- (b) rescaling V changes tau
banner("(b) Rescaling the V column changes the answer (it should not)")
print("  %-16s %12s" % ("V scaled by", "reported tau"))
for k in [0.01, 0.1, 1.0, 10.0, 100.0]:
    beta, _ = stable_ols_coef(
        np.column_stack([np.ones(T), Ar, Zr, Wr, k * V]), Yr)
    print("  %-16g %12.4f" % (k, beta[1]))

# ------------------------------------------------- (c) unit-equivariance broken
banner("(c) Unit-equivariance: scaling A by k must leave tau*k invariant")
print("  %-16s %22s" % ("A scaled by", "tau * k (should be flat)"))
for k in [1.0, 2.0, 4.0]:
    tau_k, _, _, _ = two_stage(k * Ar, Zr, Wr, Yr)
    print("  %-16g %22.4f" % (k, tau_k * k))
print("\n=> The reported effect depends on whether insulin is measured in")
print("   Units or half-Units. It is not a physical quantity in mg/dL/U.")
