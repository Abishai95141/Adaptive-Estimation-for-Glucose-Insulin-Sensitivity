"""Shared helpers for audit reproduction scripts.

Requires only numpy. `stable_ols_coef` below is a verbatim behavioural copy of
causal_eval/estimators/_utils.py:stable_ols_coef, reproduced here so the audit
scripts run without importing the package (which pulls in scikit-learn).
"""

import numpy as np

DIAG_STRONG = "causal_eval/evaluation/results/proximal_diagnostics_strong.csv"
DIAG_WEAK = "causal_eval/evaluation/results/proximal_diagnostics_weak.csv"


def stable_ols_coef(X, y):
    """Verbatim behaviour of the repo's SVD min-norm solver."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    n, p = X.shape
    tol = max(n, p) * np.finfo(float).eps * s[0]
    mask = s > tol
    Uty = U.T @ y
    beta = Vt.T @ (np.where(mask, 1.0 / s, 0.0) * Uty)
    y_hat = U @ (np.where(mask, 1.0, 0.0) * Uty)
    return beta, y_hat


def ols(X, y):
    """Ordinary full-rank least squares."""
    return np.linalg.lstsq(X, y, rcond=None)[0]


def center(x):
    return x - x.mean()


def two_stage(Ar, Zr, Wr, Yr):
    """Run the repo's Stage 1 + Stage 2 exactly as proximal_gestimation.py does.

    Returns (tau_cf, X_stage2, stage1_coefficients, V).
    """
    T = len(Ar)
    X1 = np.column_stack([np.ones(T), Zr, Wr])
    b1, A_hat = stable_ols_coef(X1, Ar)
    V = Ar - A_hat
    X2 = np.column_stack([np.ones(T), Ar, Zr, Wr, V])
    b2, _ = stable_ols_coef(X2, Yr)
    return float(b2[1]), X2, b1, V


def make_ar_patient(rng, beta_proxy, sigma_proxy, tau=-8.0, gamma=15.0,
                    alpha=1.5, T=288):
    """Generate one patient using the repo's ACTUAL confounder structure.

    Mirrors causal_eval/dgp/confounder_injection.py (AR(1) stress with circadian
    baseline; AR(1) fatigue driven by stress) and the structural equations in
    causal_eval/evaluation/experiment.py.
    """
    stress = np.zeros(T)
    fatigue = np.zeros(T)
    s = f = 0.0
    for t in range(T):
        hour = (t * 5.0 / 60.0) % 24.0
        if 9 <= hour <= 17:
            circadian = 0.3
        elif hour >= 22 or hour <= 6:
            circadian = 0.05
        else:
            circadian = 0.15
        s = np.clip(0.9 * s + 0.1 * circadian + 0.05 * rng.randn(), 0.0, 1.0)
        f = np.clip(0.85 * f + 0.10 * s + 0.03 * rng.randn(), 0.0, 1.0)
        stress[t], fatigue[t] = s, f

    Z = beta_proxy * stress + sigma_proxy * rng.randn(T)
    W = beta_proxy * fatigue + sigma_proxy * rng.randn(T)
    A = np.maximum(0.0, 0.5 - alpha * stress + 0.1 * rng.randn(T))
    Y = tau * A + gamma * stress + 2.0 * rng.randn(T)
    return dict(stress=stress, fatigue=fatigue, Z=Z, W=W, A=A, Y=Y)


def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
