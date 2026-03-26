"""
Full causal evaluation experiment orchestrator.

SEMI-SYNTHETIC DESIGN (standard in causal inference evaluation):
The Hovorka simulator provides heterogeneous ground truth τ_i via finite
differencing. We then generate outcomes from a known structural equation:

    Y_i = μ(S_i) + τ_i · A_i + γ · U_i + ε_i

where:
  Y_i  = outcome (glucose change, mg/dL)
  A_i  = continuous treatment (insulin dose, Units)
  S_i  = observed covariates (glucose, carbs, hour)
  U_i  = latent confounder (stress) — HIDDEN from estimators
  τ_i  = patient-specific treatment effect FROM HOVORKA (ground truth)
  γ    = confounder-outcome coupling strength
  ε_i  = noise

Confounding arises because U → A (stress reduces adherence) AND U → Y
(stress raises glucose via cortisol). Estimators see only (Y, A, S, Z, W).

IDENTIFICATION (from forensic audit — Option B):
Treatment A is continuous and determined by a confounded equation:
    A_t = A_base(S_t) - α·U_t + noise
There is NO known treatment assignment mechanism. The micro_randomize()
function computes a binary probability that does NOT determine A_t.
Therefore, propensity-based identification (standard G-estimation) is
NOT valid in this DGP. The propensity field is retained in the data for
transparency but is not used by any estimator.

The proximal G-estimator is identified via the bridge function approach
(Cui et al. 2024): proxy variables Z (from stress) and W (from fatigue)
satisfy the proximal independence conditions, and the control function
two-stage approach identifies τ without requiring known propensity.

The relevance weighting is a known-propensity guard: when the first-stage
F-statistic is low (proxies are weak), the estimator degrades gracefully
toward naive OLS rather than applying an unreliable correction. This is
a design property, not a failure mode.
"""

import numpy as np
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from causal_eval.dgp.hovorka_wrapper import (
    HovorkaWrapper, create_cohort, micro_randomize,
    DT_MINUTES, STEPS_PER_DAY, MR_LOW, MR_HIGH,
)
from causal_eval.dgp.ground_truth import GroundTruthComputer, DELTA_PRIMARY
from causal_eval.dgp.confounder_injection import ConfounderInjector, TREATMENT_COUPLING
from causal_eval.dgp.proxy_generator import (
    create_proxy_generator, PROXY_CONDITIONS,
)
from causal_eval.estimators.naive_regression import NaiveOLSEstimator
from causal_eval.estimators.population_aipw import PopulationAIPWEstimator
from causal_eval.estimators.standard_gestimation import StandardGEstimator
from causal_eval.estimators.proximal_gestimation import (
    ProximalGEstimatorWrapper, ProximalConfidenceSequence,
)
from causal_eval.evaluation.metrics import (
    compute_metrics, format_results_table, save_results_csv,
)

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

N_PATIENTS = 50
N_OBS_PER_PATIENT = 288
BASE_SEED = 42

# Semi-synthetic DGP parameters
# γ: confounder→outcome coupling. Must be large enough to create
# meaningful bias but not so large that it dominates τ·A.
# With τ ∈ [-18, -1] and A ∈ [0, 2], τ·A ∈ [-36, 0].
# With U ∈ [0, 1], γ=15 gives U contribution ∈ [0, 15].
CONFOUNDER_OUTCOME_GAMMA = 15.0

# Outcome model baseline coefficients
BASELINE_INTERCEPT = 120.0
BASELINE_GLUCOSE_COEFF = 0.8   # autoregressive component
BASELINE_CARBS_COEFF = 0.5     # carbs raise glucose
BASELINE_NOISE_STD = 2.0       # outcome noise (reduced to make signal clearer)

# Confounder→treatment coupling: U shifts insulin dose
# With A_base ∈ [0.07, 2.5], α=1.5 makes U a dominant component of A
# at high stress. This creates strong confounding: corr(U, A) ≈ 0.4-0.6
CONFOUNDER_TREATMENT_ALPHA = 1.5  # Units of insulin per unit of stress

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
RESULTS_CSV = os.path.join(RESULTS_DIR, 'results.csv')


def generate_patient_data(wrapper, tau_i, n_steps=N_OBS_PER_PATIENT, seed=42):
    """
    Generate one patient's semi-synthetic dataset.

    Structural equations:
      U_t ~ AR(1) with circadian pattern (from ConfounderInjector)
      A_t = A_base(S_t) + α·U_t + noise  (confounded treatment)
      Y_t = μ(S_t) + τ_i·A_t + γ·U_t + ε_t  (outcome)

    Estimators receive (Y, A, S, Z, W) but NOT U.

    Args:
        wrapper: HovorkaWrapper (used for scenario/latent variable generation)
        tau_i: ground truth treatment effect for this patient (mg/dL per Unit)
        n_steps: number of timesteps
        seed: random seed

    Returns:
        data: dict with observable arrays
        latent: dict with hidden arrays (U, for proxy generation only)
    """
    rng = np.random.RandomState(seed)
    wrapper.init_scenario(n_days=max(1, n_steps // STEPS_PER_DAY + 1))

    # Initialize confounder injector (separate RNG from treatment/outcome)
    ci = ConfounderInjector(seed=seed + 1000)

    data = {
        'Y': np.zeros(n_steps),
        'A': np.zeros(n_steps),
        'glucose': np.zeros(n_steps),
        'carbs': np.zeros(n_steps),
        'hour': np.zeros(n_steps),
        'propensity': np.zeros(n_steps),
        'treatment': np.zeros(n_steps, dtype=int),
        'insulin': np.zeros(n_steps),
        'glucose_next': np.zeros(n_steps),
    }
    latent = {
        'stress': np.zeros(n_steps),
        'fatigue': np.zeros(n_steps),
        'exercise': np.zeros(n_steps),
        'U': np.zeros(n_steps),
    }

    for t in range(n_steps):
        scenario = wrapper.get_scenario()
        glucose = scenario['glucose_mg_dl']
        carb_input = scenario['carb_input']
        hour = scenario['hour']
        basal_rate = scenario.get('basal_rate', 0.8)

        # --- Confounder: advance U ---
        exercise_event = wrapper.exercise_state > 0.1
        ci.step(hour, exercise_event=exercise_event)
        U_t = ci.stress  # primary confounder

        latent['stress'][t] = ci.stress
        latent['fatigue'][t] = ci.fatigue
        latent['exercise'][t] = ci.exercise
        latent['U'][t] = U_t

        # --- Observed covariates S ---
        data['glucose'][t] = glucose
        data['carbs'][t] = carb_input
        data['hour'][t] = hour

        # --- Treatment: A = A_base(S) + α·U + noise ---
        # Base treatment: insulin dose from micro-randomization
        _, base_propensity = micro_randomize(glucose, rng=rng)
        data['propensity'][t] = base_propensity

        # Base insulin dose (from scenario)
        basal_insulin = basal_rate * (DT_MINUTES / 60.0)
        bolus = 0.0
        if carb_input > 5:
            icr = 10.0 + 2.0 * rng.randn()
            bolus = max(0.0, carb_input / max(icr, 3.0))
        elif glucose > 150:
            bolus = max(0.0, (glucose - 120.0) / 50.0) * 0.3

        A_base = basal_insulin + bolus

        # Confounded treatment: stress INCREASES insulin dose slightly
        # (counterintuitive? No: stress → high glucose → reactive bolusing)
        # OR stress DECREASES adherence → lower dose. We use the latter
        # to match the simulator's causal structure.
        # A_confounded = A_base - α·U  (stress reduces dose)
        A_t = max(0.0, A_base - CONFOUNDER_TREATMENT_ALPHA * U_t
                  + 0.1 * rng.randn())
        data['A'][t] = A_t
        data['insulin'][t] = A_t
        data['treatment'][t] = int(bolus > 0.01)

        # --- Outcome: Y = μ(S) + τ_i·A + γ·U + ε ---
        mu_S = (BASELINE_INTERCEPT
                + BASELINE_GLUCOSE_COEFF * (glucose - 120.0)
                + BASELINE_CARBS_COEFF * carb_input)
        Y_t = (mu_S
               + tau_i * A_t
               + CONFOUNDER_OUTCOME_GAMMA * U_t
               + BASELINE_NOISE_STD * rng.randn())
        data['Y'][t] = Y_t
        data['glucose_next'][t] = Y_t  # alias for compatibility

        # Advance the ODE (to keep latent variables evolving naturally)
        wrapper.sim.sim_step(A_t, carb_input)

    return data, latent


class CausalExperiment:
    """
    Orchestrates the full causal evaluation experiment.
    """

    def __init__(self, n_patients=N_PATIENTS, n_obs=N_OBS_PER_PATIENT,
                 base_seed=BASE_SEED):
        self.n_patients = n_patients
        self.n_obs = n_obs
        self.base_seed = base_seed
        self.cohort = None
        self.ground_truth = None
        self.patient_data = None
        self.patient_latent = None

    def setup(self):
        """Create cohort and compute ground truth τ_i for all patients."""
        print(f"Creating cohort of {self.n_patients} patients...")
        self.cohort = create_cohort(
            n_patients=self.n_patients, base_seed=self.base_seed
        )

        print("Computing ground truth τ_i via finite differencing...")
        gt = GroundTruthComputer(delta=DELTA_PRIMARY)
        tau_means, tau_curves, hours = gt.compute_cohort_tau(self.cohort)

        self.ground_truth = {}
        for i in range(len(self.cohort)):
            self.ground_truth[i] = float(tau_means[i])

        tau_arr = np.array(list(self.ground_truth.values()))
        print(f"  τ range: [{tau_arr.min():.2f}, {tau_arr.max():.2f}] mg/dL/U")
        print(f"  τ span: {tau_arr.max() - tau_arr.min():.2f} mg/dL/U")

        return self.ground_truth

    def generate_data(self):
        """Generate semi-synthetic observational data for all patients."""
        print(f"Generating semi-synthetic data for {self.n_patients} patients "
              f"({self.n_obs} obs each)...")

        self.patient_data = {}
        self.patient_latent = {}

        for i, wrapper in enumerate(self.cohort):
            wrapper.sim._init_state()
            wrapper.sim.stress_level = 0.0
            wrapper.sim.fatigue_level = 0.0
            wrapper.sim.exercise_state = 0.0

            tau_i = self.ground_truth[i]
            data, latent_data = generate_patient_data(
                wrapper, tau_i=tau_i, n_steps=self.n_obs,
                seed=self.base_seed + i * 7 + 3,
            )
            self.patient_data[i] = data
            self.patient_latent[i] = latent_data

        return self.patient_data, self.patient_latent

    def _build_estimator_inputs(self, proxy_condition):
        """
        Build pooled arrays for estimators from per-patient data.

        Returns:
            Y, A, S, propensity, Z, W, patient_ids — all arrays
        """
        Y_list, A_list, S_list, prop_list = [], [], [], []
        Z_list, W_list, pid_list = [], [], []

        for pid in sorted(self.patient_data.keys()):
            data = self.patient_data[pid]
            latent_data = self.patient_latent[pid]
            n = len(data['Y'])

            pg = create_proxy_generator(
                condition=proxy_condition, patient_id=pid
            )
            Z, W = pg.generate_trajectory(
                latent_data['stress'], latent_data['fatigue']
            )

            Y_list.append(data['Y'])
            A_list.append(data['A'])

            S_patient = np.column_stack([
                data['glucose'], data['carbs'], data['hour']
            ])
            S_list.append(S_patient)
            prop_list.append(data['propensity'])
            Z_list.append(Z)
            W_list.append(W)
            pid_list.append(np.full(n, pid, dtype=int))

        Y = np.concatenate(Y_list)
        A = np.concatenate(A_list)
        S = np.vstack(S_list)
        propensity = np.concatenate(prop_list)
        Z = np.concatenate(Z_list)
        W = np.concatenate(W_list)
        patient_ids = np.concatenate(pid_list)

        return Y, A, S, propensity, Z, W, patient_ids

    def run_estimator(self, estimator, name, Y, A, S, propensity,
                      Z, W, patient_ids):
        """Run one estimator and return per-patient τ estimates + confidence bounds."""
        print(f"  Running {name}...")
        t0 = time.time()

        try:
            tau_individual = estimator.estimate_individual(
                Y, A, S, patient_ids,
                propensity=propensity, Z=Z, W=W,
            )
        except Exception as e:
            print(f"    ERROR: {name} failed — {e}")
            unique_ids = np.unique(patient_ids)
            tau_individual = {int(pid): float('nan') for pid in unique_ids}

        elapsed = time.time() - t0
        print(f"    Completed in {elapsed:.1f}s")

        confidence_bounds = None
        if name == 'Proximal G-estimation':
            confidence_bounds = {}
            for pid in tau_individual:
                cs = ProximalConfidenceSequence(alpha=0.05)
                # Use batch mode with DR scores for proper per-patient CIs
                dr_scores = None
                if hasattr(estimator, '_individual_dr_scores') and pid in estimator._individual_dr_scores:
                    dr_scores = estimator._individual_dr_scores[pid]
                se = None
                if hasattr(estimator, '_individual_se') and pid in estimator._individual_se:
                    se = estimator._individual_se[pid]

                if dr_scores is not None and len(dr_scores) > 1:
                    # Batch mode: proper per-patient CI from DR score distribution
                    cs.set_from_scores(tau_individual[pid], dr_scores, se=se)
                else:
                    cs.update(tau_individual[pid], se=se)

                confidence_bounds[pid] = cs.get_current_bounds()

        return tau_individual, confidence_bounds

    def run(self):
        """
        Run the full experiment.

        Returns:
            all_results: list of result dicts
            table_str: formatted results table
        """
        self.setup()
        self.generate_data()

        estimators = [
            (NaiveOLSEstimator(), 'Naive OLS'),
            (PopulationAIPWEstimator(), 'Population AIPW'),
            (StandardGEstimator(), 'Standard G-estimation'),
            (ProximalGEstimatorWrapper(), 'Proximal G-estimation'),
        ]

        all_results = []

        for proxy_condition in ['strong', 'weak']:
            print(f"\n=== Proxy condition: {proxy_condition} "
                  f"(β={PROXY_CONDITIONS[proxy_condition]['beta']}, "
                  f"σ={PROXY_CONDITIONS[proxy_condition]['sigma']}) ===")

            Y, A, S, propensity, Z, W, patient_ids = self._build_estimator_inputs(
                proxy_condition
            )
            print(f"  Total observations: {len(Y)}")
            print(f"  Mean insulin dose: {np.mean(A):.3f} U")

            for estimator, name in estimators:
                tau_est, conf_bounds = self.run_estimator(
                    estimator, name, Y, A, S, propensity, Z, W, patient_ids
                )

                metrics = compute_metrics(
                    self.ground_truth, tau_est,
                    confidence_bounds=conf_bounds,
                )

                result = {
                    'estimator': name,
                    'proxy_condition': proxy_condition,
                    'mae': metrics['mae'],
                    'p95_error': metrics['p95_error'],
                    'coverage': metrics.get('coverage', float('nan')),
                    'bias': metrics['bias'],
                }
                all_results.append(result)

                # Save per-patient diagnostics for proximal estimator
                if name == 'Proximal G-estimation' and hasattr(estimator, 'save_diagnostics'):
                    diag_path = os.path.join(
                        RESULTS_DIR,
                        f'proximal_diagnostics_{proxy_condition}.csv'
                    )
                    estimator.save_diagnostics(diag_path, ground_truth=self.ground_truth)
                    print(f"    Diagnostics saved to {diag_path}")

        table_str, rows = format_results_table(all_results)
        print("\n" + table_str)

        save_results_csv(rows, RESULTS_CSV)
        print(f"\nResults saved to {RESULTS_CSV}")

        return all_results, table_str


def run_experiment(n_patients=N_PATIENTS, n_obs=N_OBS_PER_PATIENT,
                   base_seed=BASE_SEED):
    """Convenience function to run the full experiment."""
    exp = CausalExperiment(
        n_patients=n_patients, n_obs=n_obs, base_seed=base_seed
    )
    return exp.run()


if __name__ == '__main__':
    run_experiment()
