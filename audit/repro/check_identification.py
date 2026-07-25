"""F-4 / F-5: DGP identification, and the duplicate-estimator identity.

Section 1 tests a CORRECTLY specified linear proximal 2SLS against an oracle on
the repo's actual AR confounder structure -- showing the fix in F-1 will not
recover the headline result, because the DGP itself is weakly identified.
Section 2 confirms Naive OLS and Standard G-estimation are the same estimator.

Run from the repository root:  python audit/repro/check_identification.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import banner, center, make_ar_patient, ols, two_stage  # noqa: E402

T = 288
TAU_TRUE = -8.0
N_REPS = 200

# ================================================== 1. is the DGP identified?
banner("1. Correctly specified proximal 2SLS vs oracle (%d reps)" % N_REPS)
print("   DGP: AR(1) stress; AR(1) fatigue driven by stress; Z<-stress, W<-fatigue")

for label, bp, sp in [("STRONG (b=0.8, s=0.2)", 0.8, 0.2),
                      ("WEAK   (b=0.3, s=0.5)", 0.3, 0.5)]:
    out = {k: [] for k in ["naive", "repo_cf", "iv_2sls", "prox_2sls", "oracle"]}
    corrs = []

    for rep in range(N_REPS):
        rng = np.random.RandomState(rep)
        p = make_ar_patient(rng, bp, sp, tau=TAU_TRUE, T=T)
        Yr, Ar, Zr, Wr = map(center, (p["Y"], p["A"], p["Z"], p["W"]))
        corrs.append(np.corrcoef(center(p["fatigue"]), center(p["stress"]))[0, 1])

        out["naive"].append(np.sum(Ar * Yr) / np.sum(Ar ** 2))

        # what the repo computes (rank-deficient min-norm)
        tau_cf, _, b1, _ = two_stage(Ar, Zr, Wr, Yr)
        out["repo_cf"].append(tau_cf)

        # plain 2SLS with Z,W as instruments -- exclusion restriction is FALSE
        A_hat = center(np.column_stack([np.ones(T), Zr, Wr]) @ b1)
        out["iv_2sls"].append(np.sum(A_hat * Yr) / np.sum(A_hat * Ar))

        # correct linear proximal 2SLS: W ~ [1,A,Z] -> What ; Y ~ [1,A,What]
        g = ols(np.column_stack([np.ones(T), Ar, Zr]), Wr)
        W_hat = np.column_stack([np.ones(T), Ar, Zr]) @ g
        out["prox_2sls"].append(ols(np.column_stack([np.ones(T), Ar, W_hat]), Yr)[1])

        # oracle: adjust for the TRUE latent confounder
        out["oracle"].append(
            ols(np.column_stack([np.ones(T), Ar, center(p["stress"])]), Yr)[1])

    print("\n  --- %s ---   true tau = %.1f   corr(fatigue,stress) = %.3f"
          % (label, TAU_TRUE, np.mean(corrs)))
    print("  %-30s %10s %10s %12s" % ("", "mean", "median", "sd"))
    names = {
        "naive": "naive OLS",
        "repo_cf": "repo tau_cf (rank-deficient)",
        "iv_2sls": "2SLS, Z/W as instruments",
        "prox_2sls": "CORRECT linear proximal 2SLS",
        "oracle": "ORACLE (adjusts for true U)",
    }
    for k in ["naive", "repo_cf", "iv_2sls", "prox_2sls", "oracle"]:
        v = np.array(out[k])
        print("  %-30s %10.3f %10.3f %12.3f"
              % (names[k], v.mean(), np.median(v), v.std()))

print("""
  => The correct proximal estimator is wildly unstable and does not recover tau,
     while the ORACLE recovers it cleanly. The target is estimable given U --
     the failure is in the proxy structure, not the setup.
  => Plain 2SLS is WORSE than naive: Z proxies U, and U affects Y directly, so
     the exclusion restriction fails. Proxies are not instruments.
  => Root cause: A, Z and W are all noisy readings of ONE scalar latent.""")

# ============================================ 2. Frisch-Waugh-Lovell identity
banner("2. Naive OLS == Standard G-estimation (Frisch-Waugh-Lovell)")

rng = np.random.RandomState(0)
S = np.column_stack([120 + 30 * rng.randn(T),
                     np.abs(20 * rng.randn(T)),
                     24 * rng.rand(T)])
A = 0.5 + 0.01 * S[:, 0] + 0.1 * rng.randn(T)
Y = -8.0 * A + 0.8 * S[:, 0] + 0.5 * S[:, 1] + 2.0 * rng.randn(T)

# NaiveOLSEstimator: regress Y on [1, S, A], take coefficient on A
tau_naive = ols(np.column_stack([np.ones(T), S, A]), Y)[-1]

# StandardGEstimator: partial out S, then regress residuals
X_s = np.column_stack([np.ones(T), S])
resid = lambda v: v - X_s @ ols(X_s, v)  # noqa: E731
Yr, Ar = resid(Y), resid(A)
tau_gest = np.sum(Ar * Yr) / np.sum(Ar ** 2)

print("  NaiveOLSEstimator   tau = %.12f" % tau_naive)
print("  StandardGEstimator  tau = %.12f" % tau_gest)
print("  absolute difference     = %.3e" % abs(tau_naive - tau_gest))
print("""
  => Algebraically the same estimator. results.csv reports both rows as
     4.037944827882445 and 4.03794482788243 -- agreeing to 13 significant
     figures. They should not be presented as two independent methods.""")
