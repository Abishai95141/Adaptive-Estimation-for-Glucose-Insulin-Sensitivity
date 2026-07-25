"""F-2 / F-3: The correction is a halving, and it ignores proxy quality.

Section 1 reads the COMMITTED diagnostics CSVs -- no re-simulation, so these
numbers are properties of the shipped results, not of this script.
Section 2 verifies the closed-form min-norm prediction.
Section 3 sweeps confounding strength gamma to show the estimator never tracks
the truth (including the decisive gamma=0 case).

Run from the repository root:  python audit/repro/check_halving.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (DIAG_STRONG, DIAG_WEAK, banner, center,  # noqa: E402
                     ols, two_stage)

# ============================================================ 1. committed data
banner("1. From the COMMITTED diagnostics CSVs (no re-simulation)")

for cond, path in [("strong", DIAG_STRONG), ("weak", DIAG_WEAK)]:
    if not os.path.exists(path):
        print("  MISSING: %s  (run from the repository root)" % path)
        continue
    d = pd.read_csv(path)
    t = d.tau_true

    print("\n  --- %s proxies (n=%d) ---" % (cond.upper(), len(d)))
    ratio = d.tau_cf / d.tau_naive
    print("  tau_cf / tau_naive:  mean=%.4f  sd=%.4f  min=%.4f  max=%.4f"
          % (ratio.mean(), ratio.std(), ratio.min(), ratio.max()))

    print("  %-30s %7s %8s %7s %7s" % ("", "MAE", "bias", "P95", "corr"))
    for name, col in [("tau_naive (= Naive OLS)", d.tau_naive),
                      ("tau_cf   (unblended CF)", d.tau_cf),
                      ("tau_final (reported)", d.tau_final)]:
        e = col - t
        print("  %-30s %7.3f %+8.3f %7.3f %7.4f"
              % (name, e.abs().mean(), e.mean(),
                 np.percentile(e.abs(), 95), np.corrcoef(col, t)[0, 1]))

    print("  first stage: F median=%.2f  frac(F>=10)=%.0f%%  r2 median=%.4f (max %.4f)"
          % (d.F_stat_stage1.median(), 100 * (d.F_stat_stage1 >= 10).mean(),
             d.r2_stage1.median(), d.r2_stage1.max()))

print("""
  => tau_cf/tau_naive is 0.50 in BOTH conditions: the 'correction' is a halving.
  => tau_cf MAE is IDENTICAL (2.116) strong vs weak: the point estimate carries
     no proxy information. Only the blend weight responds to proxy quality.""")

# ==================================================== 2. closed-form prediction
banner("2. Closed-form min-norm prediction:  a_A*(1+b1^2+b2^2)/(2+b1^2+b2^2)")

T = 288
print("  %-8s %12s %14s %14s %12s"
      % ("proxies", "a_A", "min-norm tau", "analytic pred", "a_A / 2"))
for label, bp, sp in [("strong", 0.8, 0.2), ("weak", 0.3, 0.5)]:
    rng = np.random.RandomState(7)
    U = np.clip(0.3 + 0.3 * rng.randn(T), 0, 1)
    Z = bp * U + sp * rng.randn(T)
    W = bp * (0.9 * U) + sp * rng.randn(T)
    A = np.maximum(0, 0.5 - 1.5 * U + 0.1 * rng.randn(T))
    Y = -8.0 * A + 15.0 * U + 2.0 * rng.randn(T)
    Yr, Ar, Zr, Wr = map(center, (Y, A, Z, W))

    tau_cf, _, b1, _ = two_stage(Ar, Zr, Wr, Yr)
    a = ols(np.column_stack([np.ones(T), Ar, Zr, Wr]), Yr)
    b1c, b2c = b1[1], b1[2]
    pred = a[1] * (1 + b1c ** 2 + b2c ** 2) / (2 + b1c ** 2 + b2c ** 2)
    print("  %-8s %12.4f %14.4f %14.4f %12.4f"
          % (label, a[1], tau_cf, pred, a[1] / 2))

print("\n  => Converges to exactly a_A/2 as the first stage weakens (b1,b2 -> 0).")

# ============================================================ 3. gamma sweep
banner("3. Sweep confounding strength gamma, true tau fixed at -8.0 (40 reps)")

print("  %6s %10s %10s %8s %11s %10s"
      % ("gamma", "naive", "tau_cf", "true", "cf error", "cf/naive"))
for gamma in [0.0, 5.0, 15.0, 30.0, 60.0]:
    naives, cfs = [], []
    for rep in range(40):
        r = np.random.RandomState(1000 + rep)
        U = np.clip(0.3 + 0.3 * r.randn(T), 0, 1)
        Z = 0.8 * U + 0.2 * r.randn(T)
        W = 0.8 * (0.9 * U) + 0.2 * r.randn(T)
        A = np.maximum(0, 0.5 - 1.5 * U + 0.1 * r.randn(T))
        Y = -8.0 * A + gamma * U + 2.0 * r.randn(T)
        Yr, Ar, Zr, Wr = map(center, (Y, A, Z, W))
        tau_cf, _, _, _ = two_stage(Ar, Zr, Wr, Yr)
        naives.append(np.sum(Ar * Yr) / np.sum(Ar ** 2))
        cfs.append(tau_cf)
    mn, mc = np.mean(naives), np.mean(cfs)
    print("  %6.0f %10.3f %10.3f %8.1f %+11.3f %10.4f"
          % (gamma, mn, mc, -8.0, mc + 8.0, mc / mn))

print("""
  => At gamma=0 there is NO confounding; naive OLS is already correct (-7.97).
     A valid estimator must return -8.0. This one returns -4.30 -- it halves
     regardless. Across the sweep it tracks naive/2, never the truth.""")
