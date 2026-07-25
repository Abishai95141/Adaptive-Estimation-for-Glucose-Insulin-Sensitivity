# Audit

Independent code and methodology audit of the `causal_eval/` proximal causal inference pipeline.

- **[AUDIT_FINDINGS.md](AUDIT_FINDINGS.md)** — full findings report with evidence and recommendations
- **[repro/](repro/)** — reproduction scripts for every quantitative claim

## Headline

The reported 47.5% MAE reduction does not hold up. The Stage-2 regression in
`ProximalGEstimatorWrapper` is exactly rank-deficient, and the SVD pseudo-inverse silently returns a
minimum-norm solution that reduces to **dividing the naive OLS estimate by two**. The improvement is
an arithmetic consequence of that halving.

## Findings

| # | Severity | Finding |
|---|----------|---------|
| F-1 | **Critical** | Stage-2 design matrix is exactly rank-deficient; reported τ is a min-norm artifact |
| F-2 | **Critical** | The "control function correction" is arithmetically a halving of naive OLS |
| F-3 | **Critical** | The point estimate is completely insensitive to proxy quality |
| F-4 | **Major** | The DGP is too weakly identified for proximal inference at n=288 |
| F-5 | **Major** | "Naive OLS" and "Standard G-estimation" are the same estimator |
| F-6 | **Major** | The 96% coverage figure is fitted, not measured |
| F-7 | Minor | Documentation / data / dependency discrepancies |

## Running the checks

Requires only `numpy` and `pandas`. From the repository root:

```bash
python audit/repro/check_rank_deficiency.py   # F-1
python audit/repro/check_halving.py           # F-2, F-3
python audit/repro/check_identification.py    # F-4, F-5
```

Sections that read `causal_eval/evaluation/results/proximal_diagnostics_*.csv` are properties of the
committed results and require no re-simulation. Simulation-based sections use fixed seeds.

## Scope

This audit covers `causal_eval/`. The earlier `verification/` subsystem is surveyed but not audited;
its own logs already document its limitations candidly. No changes are made to existing code — this
directory is additive only.
