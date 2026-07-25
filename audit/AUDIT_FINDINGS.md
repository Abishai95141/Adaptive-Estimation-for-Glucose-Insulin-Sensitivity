# Independent Audit — AEGIS Causal Evaluation Framework

**Date:** 2026-07-25
**Scope:** `causal_eval/` (proximal causal inference pipeline), with a survey of `verification/`
**Method:** Full source read plus numerical verification. Every quantitative claim below is
reproduced by a script in [`repro/`](repro/) using only `numpy` and `pandas`.

---

## Bottom line

The headline result — *"proximal causal inference reduces MAE by 47.5% versus naive OLS"* — does
not hold up. The Stage-2 regression in `ProximalGEstimatorWrapper` is **exactly rank-deficient**,
and the SVD pseudo-inverse silently returns a minimum-norm solution that reduces to
**dividing the naive OLS estimate by two**. The 47.5% improvement is an arithmetic consequence of
that halving, not evidence of confounding adjustment.

The estimator is not unit-equivariant, does not respond to proxy quality, and returns a wrong
answer when there is no confounding at all. Separately, the DGP does not appear to supply enough
structure for *any* correctly specified proximal estimator to work at `n=288`.

The DGP construction, ground-truth machinery, and existing test suite are sound and careful. The
core problem is that `README.md` markets numbers that the repository's own test suite already
disclaims.

---

## Findings summary

| # | Severity | Finding |
|---|----------|---------|
| [F-1](#f-1) | **Critical** | Stage-2 design matrix is exactly rank-deficient; reported τ is a min-norm artifact |
| [F-2](#f-2) | **Critical** | The "control function correction" is arithmetically a halving of naive OLS |
| [F-3](#f-3) | **Critical** | The point estimate is completely insensitive to proxy quality |
| [F-4](#f-4) | **Major** | The DGP is too weakly identified for proximal inference at n=288 |
| [F-5](#f-5) | **Major** | "Naive OLS" and "Standard G-estimation" are the same estimator |
| [F-6](#f-6) | **Major** | The 96% coverage figure is fitted, not measured |
| [F-7](#f-7) | Minor | Documentation / data / dependency discrepancies |

---

<a name="f-1"></a>
## F-1 (Critical) — Stage-2 design matrix is exactly rank-deficient

**Location:** [`causal_eval/estimators/proximal_gestimation.py:191-215`](../causal_eval/estimators/proximal_gestimation.py#L191-L215)

Stage 1 fits `Ã` on `[1, Z̃, W̃]` and defines the control variable `V = Ã − Â`. Stage 2 then
regresses `Ỹ` on `[1, Ã, Z̃, W̃, V]`.

Because `Â` is *itself* the fit of `Ã` on `[1, Z̃, W̃]`, the vector `V` is by construction an exact
linear combination of the other four columns. The null vector is exactly

```
n = (−b₀, 1, −b₁, −b₂, −1)      where b = Stage-1 coefficients
```

### Measured

```
shape:            (288, 5)
singular values:  [3.1563e+01  1.6971e+01  5.8047e+00  3.1973e+00  2.3337e-15]
condition number: 1.352e+16
numpy matrix_rank: 4  of 5 columns
||X₂ @ n||:       6.66e-15        (zero to machine precision)
```

The first four singular values are exactly reproducible; the fifth sits at floating-point noise
(~1e-15) and varies run to run, as expected for an exactly singular direction.

`stable_ols_coef` in [`_utils.py:51`](../causal_eval/estimators/_utils.py#L51) truncates singular
values below tolerance and returns the **minimum-norm** solution. It does this silently — there is
no rank warning. So "the coefficient on `Ã`" is not a well-defined quantity; it is whatever the
pseudo-inverse happens to split off along the degenerate direction.

### Consequence 1 — the answer depends on an arbitrary rescaling of `V`

Multiplying the `V` column by a constant is causally meaningless. It changes the answer:

| `V` scaled by | reported τ |
|---|---|
| 0.01 | −16.5131 |
| 0.1 | −16.3824 |
| 1 | −12.9865 |
| 10 | −11.7647 |
| 100 | −11.7482 |

### Consequence 2 — the estimator is not unit-equivariant

If insulin is measured in half-Units instead of Units, τ should scale by exactly `1/k`, so `τ·k`
must be invariant. It is not:

| `A` scaled by | `τ·k` (should be constant) |
|---|---|
| 1 | −12.9865 |
| 2 | −16.0069 |
| 4 | −17.4793 |

**The estimate depends on your choice of units.** This alone disqualifies the reported number as a
physical quantity in mg/dL/U.

**Reproduce:** `python audit/repro/check_rank_deficiency.py`

---

<a name="f-2"></a>
## F-2 (Critical) — The correction is arithmetically a halving

Working through the min-norm algebra: let `a_A` be the coefficient on `Ã` from the well-posed
full-rank regression `Ỹ ~ [1, Ã, Z̃, W̃]`. Any solution to the rank-deficient system satisfies
`β_A + β_V = a_A`, and the min-norm condition picks

```
β_A  =  a_A · (1 + b₁² + b₂²) / (2 + b₁² + b₂²)
```

when the data are centred. As the first stage weakens (`b₁, b₂ → 0`) this converges to exactly
**`a_A / 2`**. Verified numerically:

| proxies | `a_A` | min-norm τ | analytic prediction | `a_A / 2` |
|---|---|---|---|---|
| strong | −18.3006 | −10.9300 | −9.9992 | −9.1503 |
| weak | −23.6846 | −11.8969 | **−11.9081** | −11.8423 |

Under weak proxies the closed form matches the actual output to four decimal places.

The repository's own saved diagnostics confirm it:

| condition | `tau_cf / tau_naive` | sd | min | max |
|---|---|---|---|---|
| strong | **0.5024** | 0.0162 | 0.4812 | 0.5518 |
| weak | **0.5004** | 0.0049 | 0.4896 | 0.5193 |

The spread is tighter under *weak* proxies — exactly as the formula predicts, since `b₁, b₂ → 0`
there.

### Why halving scores well here

In this DGP the confounding bias is near-constant across patients, because
`bias = −γ·α·var(U)/var(Ã)` does not involve `τ_i`. Empirically `b ≈ −4.038` for every patient. So:

```
naive_i  = τ_i + b
tau_cf   ≈ naive_i / 2      = (τ_i + b)/2
error_i  = (b − τ_i)/2
MAE_cf   = mean|b − τ_i| / 2  = 2.116
MAE_naive = |b|               = 4.038
ratio     = 0.524   →   47.6% "reduction"
```

This reproduces the reported 47.5% exactly. **The headline number follows from the halving and
from where the τ_i distribution happens to sit relative to γ=15 and α=1.5.** It is a property of
the chosen constants, not a property of the method.

### Proof it is not estimating a causal effect

Sweeping the confounding strength γ with true τ fixed at −8.0 (40 reps each):

| γ | naive | `tau_cf` | true τ | `tau_cf` error | `tau_cf`/naive |
|---|---|---|---|---|---|
| **0** | −7.974 | **−4.296** | −8.0 | **+3.704** | 0.5388 |
| 5 | −12.972 | −6.306 | −8.0 | +1.694 | 0.4861 |
| 15 | −22.970 | −10.327 | −8.0 | −2.327 | 0.4496 |
| 30 | −37.966 | −16.359 | −8.0 | −8.359 | 0.4309 |
| 60 | −67.958 | −28.421 | −8.0 | −20.421 | 0.4182 |

At **γ = 0 there is no confounding at all**. Naive OLS is already correct (−7.974). A valid
estimator must return −8. This one returns **−4.30** — it halves regardless. Across the whole sweep
it tracks `naive/2` and never tracks the truth.

**Reproduce:** `python audit/repro/check_halving.py`

---

<a name="f-3"></a>
## F-3 (Critical) — The point estimate ignores proxy quality

Recomputing per-patient accuracy directly from the committed diagnostics CSVs:

| quantity | condition | MAE | bias | P95 | corr |
|---|---|---|---|---|---|
| `tau_naive` | both | 4.038 | −4.038 | 6.639 | 0.9646 |
| **`tau_cf`** | **strong** | **2.116** | +0.645 | 6.847 | 0.9633 |
| **`tau_cf`** | **weak** | **2.116** | +0.662 | 7.051 | 0.9651 |
| `tau_final` | strong | 2.119 | −1.749 | 3.657 | 0.9799 |
| `tau_final` | weak | 3.542 | −3.511 | 6.187 | 0.9594 |

**`tau_cf` has identical MAE (2.116) under strong and weak proxies.** The raw control-function
estimate carries no proxy information whatsoever — consistent with F-2, since halving the naive
estimate cannot depend on `Z` or `W`.

The entire "Criterion 3: strong beats weak" result therefore comes from the relevance gate
`w = 1 − exp(−F/10)` in [`proximal_gestimation.py:234`](../causal_eval/estimators/proximal_gestimation.py#L234),
which decides *how much to blend back toward naive*:

| condition | F median | frac F≥10 | r²(stage 1) median | weight median |
|---|---|---|---|---|
| strong | 6.38 | 28% | 0.0428 (max 0.1830) | 0.471 |
| weak | 1.07 | 0% | 0.0074 (max 0.0284) | 0.101 |

The proxies function as a **switch controlling how much halving to apply**, not as a source of
confounder information. Note also that even in the *strong* condition the proxies explain a median
of **4.3%** of residualised treatment variance, and 72% of patients fall below the weak-instrument
threshold.

**Reproduce:** `python audit/repro/check_halving.py` (section 1)

---

<a name="f-4"></a>
## F-4 (Major) — The DGP is too weakly identified for proximal inference

This matters because it means F-1 is not merely a bug to patch — fixing the rank deficiency will
not recover the headline result.

Testing on the repository's *actual* confounder structure (AR(1) stress; AR(1) fatigue driven by
stress; `Z` from stress, `W` from fatigue), 200 reps, true τ = −8.0:

| estimator | strong proxies | weak proxies |
|---|---|---|
| naive OLS | −16.908 (sd 1.08) | −16.908 (sd 1.08) |
| repo `tau_cf` (rank-deficient) | −8.734 (sd 0.60) | −8.456 (sd 0.54) |
| 2SLS, Z/W as instruments | −20.239 (sd 2.40) | −17.977 (sd 13.4) |
| correct linear proximal 2SLS | **−1.477 (sd 136.7)** | **−17.668 (sd 10.4)** |
| ORACLE (adjusts for true U) | **−8.038 (sd 1.20)** | −8.038 (sd 1.20) |

A correctly specified proximal 2SLS is wildly unstable and does not recover τ. The oracle recovers
it cleanly, confirming the target is estimable *in principle* given U — the failure is in the proxy
structure, not the setup.

Note the repo's `tau_cf` looks *good* here (−8.73 against a true −8.0) — but this is F-2 again, not
success. Naive is −16.9, and half of that is −8.45. The halving lands near the truth purely because
these parameter choices make naive ≈ 2× truth. Change γ and it breaks immediately, as the sweep in
F-2 shows.

Plain 2SLS using `Z, W` as instruments returns **−20.239** (worse than naive), exactly as theory
predicts: `Z` is a proxy for `U`, and `U` affects `Y` directly through `γ·U`, so the exclusion
restriction fails. Proximal causal inference exists *precisely because* proxies cannot be used as
instruments; it requires solving a bridge-function integral equation. **The code implements
two-stage residual inclusion — an IV/control-function method — while citing Cui et al. (2024) for a
method it does not implement.**

Root cause: `corr(fatigue, stress) ≈ 0.686`, and the module's own docstring notes fatigue is ~92%
explained by stress. `A`, `Z`, and `W` are all noisy linear readings of essentially **one scalar
latent**. That does not give proximal identification enough independent structure at n=288.

**Reproduce:** `python audit/repro/check_identification.py`

---

<a name="f-5"></a>
## F-5 (Major) — Two of the four estimators are the same estimator

[`NaiveOLSEstimator`](../causal_eval/estimators/naive_regression.py) regresses `Y` on `[1, S, A]`.
[`StandardGEstimator`](../causal_eval/estimators/standard_gestimation.py) partials out `S` then
regresses `Ỹ` on `Ã`. By the **Frisch–Waugh–Lovell theorem** these are algebraically identical.

Measured on common inputs:

```
NaiveOLSEstimator  tau = -8.337133060694
StandardGEstimator tau = -8.337133060694
difference             = 1.421e-14
```

This is visible in the committed results — `results.csv` reports MAE `4.037944827882445` and
`4.03794482788243` for the two rows, agreeing to 13 significant figures.

The test suite already detects this
([`test_success_criteria.py:224`](../causal_eval/tests/test_success_criteria.py#L224)) and prints a
recommendation to remove or relabel the row. That recommendation has not been applied to
`README.md`, which still presents four distinct methods.

**Reproduce:** `python audit/repro/check_identification.py` (section 2)

---

<a name="f-6"></a>
## F-6 (Major) — The 96% coverage figure is fitted, not measured

[`proximal_gestimation.py:501`](../causal_eval/estimators/proximal_gestimation.py#L501) sets
`effective_n = n / 5.0` and `sensitivity_gamma = 0.15`. Both are hand-tuned constants. The source
comment states the intent plainly:

> *"The DR score SD (~6-10) is the right scale, and we need a fraction of it as the CI half-width
> to achieve 95% coverage."*

The interval width was chosen to hit the coverage target, so the resulting coverage is not evidence
of calibration. Reported coverage under the three regimes:

| inflation | strong | weak |
|---|---|---|
| Honest CLT (none) | **0%** | 2% |
| Bridge-adjusted (n/5) | 4% | 2% |
| Sensitivity-adjusted (n/5 + γ=0.15) | 96% | 96% |

To its credit, `test_success_criteria.py` documents this honestly: it records that the original
"coverage ≥ 90%" criterion was **replaced** because it "was achievable only via engineered CI
inflation," and that Criterion 1 was relaxed from 30% to 20%. `README.md` reports the 96% figure as
a headline badge without that framing.

---

<a name="f-7"></a>
## F-7 (Minor) — Documentation, data, and dependency discrepancies

1. **README vs. committed results.** `README.md` reports weak-proxy P95 error of **5.60**;
   [`results.csv`](../causal_eval/evaluation/results/results.csv) contains **6.187**.

2. **Docstring contradicts code.** [`experiment.py:107`](../causal_eval/evaluation/experiment.py#L107)
   documents `A_t = A_base(S_t) + α·U_t`, but line 189 implements `A_base − α·U_t`. The sign matters
   for interpreting the direction of confounding.

3. **Dependencies not installable as documented.** The environment audited had only `numpy` and
   `pandas`. `scipy`, `scikit-learn`, `matplotlib`, and `pytest` were absent, so no part of the
   pipeline or test suite could execute as shipped. There is no `requirements.txt`,
   `pyproject.toml`, or pinned environment file in the repository.

4. **Unused propensity machinery.** `micro_randomize()` computes and stores a propensity score at
   [`experiment.py:170`](../causal_eval/evaluation/experiment.py#L170) that no estimator consumes.
   This is disclosed in the module docstring, but the field remains in the emitted data and invites
   misreading.

5. **`verification/` is described as legacy but is a live dependency.** `README.md` calls it "legacy
   validation code, not used in final pipeline," yet
   [`hovorka_wrapper.py:26`](../causal_eval/dgp/hovorka_wrapper.py#L26) imports
   `verification/simulator/patient.py` at runtime via `sys.path` injection.

---

## What is sound

These parts hold up under scrutiny and should be preserved through any rework:

- **The DGP construction.** Confounding behaves exactly as designed. The measured naive-OLS bias of
  −4.038 matches the analytic prediction from γ=15, α=1.5.
- **Ground-truth machinery.** Finite differencing with full state *and RNG* save/restore
  ([`hovorka_wrapper.py:120-149`](../causal_eval/dgp/hovorka_wrapper.py#L120-L149)) is the correct
  way to hold latent trajectories fixed across perturbed runs. This is a subtle detail done right.
- **Structural enforcement of proxy independence.** `ProxyGenerator.generate()` accepts only
  confounder values, making violation of the independence conditions structurally impossible rather
  than merely tested.
- **Numerical care.** SVD-based residualisation genuinely does avoid the ill-conditioning it
  targets. (The rank problem in F-1 is a specification error, not a numerical one — but it is
  *masked* by the same SVD truncation that provides the stability.)
- **The test suite and `FINDINGS.md`.** Both are markedly more honest than `README.md`. The revision
  history of the success criteria is documented rather than hidden.

## Note on `verification/`

The earlier 5-layer closed-loop system is documented in
[`verification/VERIFICATION.md`](../verification/VERIFICATION.md) and
[`verification/FINDINGS.md`](../verification/FINDINGS.md) as never having reached clinical viability:
best time-in-range **36.3%** against a 70% target, forward-model RMSE 117–209 mg/dL, and 40/43 tests
passing with acknowledged failures. Those logs are careful and self-critical, and they correctly
diagnose the root cause as Bergman–Hovorka structural model mismatch. No audit findings are raised
against that subsystem here; it is out of scope. It is noted only because `README.md` understates
its coupling to the current pipeline (F-7.5).

---

## Recommendations

**Immediate (correctness):**

1. Fix the Stage-2 specification. Either regress `Ỹ ~ [1, Ã, V]` (textbook control function) or
   `Ỹ ~ [1, Ã, Z̃, W̃]` (proxy adjustment) — not both. Expect the headline result to disappear,
   because the halving is what produces it.
2. Make `stable_ols_coef` **raise or warn on rank deficiency** instead of silently returning a
   min-norm solution. This class of bug should never again be invisible.
3. Add a unit-equivariance regression test: scaling `A` by `k` must scale `τ̂` by `1/k`.
4. Add a **null test**: with γ=0, every estimator must recover τ. This single test would have caught
   F-1, F-2, and F-3 at once.

**Structural (before re-running the study):**

5. Redesign the DGP so `W` has a genuinely independent driver from `Z`, rather than both loading on
   one scalar latent. Verify first-stage strength reaches a usable range **before** running
   estimation.
6. Implement an actual bridge-function estimator if the paper claims proximal inference, or restate
   the method as control-function/IV and address the exclusion-restriction failure honestly.

**Reporting:**

7. Reconcile `README.md` with the test suite's own disclaimers: drop or relabel the duplicate
   estimator row (F-5), and report coverage with its inflation caveat inline (F-6).
8. Add a pinned dependency manifest so results are reproducible.

---

## Reproduction

All scripts require only `numpy` and `pandas` and read the committed diagnostics CSVs where
relevant. From the repository root:

```bash
python audit/repro/check_rank_deficiency.py   # F-1
python audit/repro/check_halving.py           # F-2, F-3
python audit/repro/check_identification.py    # F-4, F-5
```

Findings derived from committed data (F-2 ratio table, F-3 accuracy table) read
`causal_eval/evaluation/results/proximal_diagnostics_{strong,weak}.csv` directly and are
independent of any re-run. Findings requiring simulation (F-2 γ-sweep, F-4) use fixed seeds and are
deterministic.
