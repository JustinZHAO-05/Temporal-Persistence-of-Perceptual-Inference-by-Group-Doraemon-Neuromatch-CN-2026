# Perceptual arbitration analysis pipeline

This package implements the main computational analysis for **Temporal Persistence of Perceptual Inference**. It extends the memoryless Switching Observer with temporally persistent hidden states and compares that account with independent switching, one-back serial baselines, and a covariate-dependent HMM.

The package uses circular von Mises emissions for three latent response modes:

- `S_sensory`: responses centered on the current motion direction;
- `P_prior`: responses centered on the learned prior mean;
- `L_lapse`: circular-uniform responses.

## Reproducibility levels

### Fast scientific check

```powershell
python scripts/run_fast_replication.py `
  --csv data/data01_direction4priors.csv `
  --out outputs/fast `
  --restarts 2 `
  --max-iter 100
```

### Full publication configuration

```powershell
python scripts/run_all.py `
  --config configs/default.yaml `
  --resume `
  --n-jobs 4
```

The publication configuration uses sequence-preserving four-fold cross-validation and deterministic multistart fitting. The principal static and subject HMM fits use 25 restarts. Checkpoints are atomic and resumable.

### Fit-independent posterior predictive checks

```powershell
python scripts/run_posterior_predictive_checks.py `
  --csv data/data01_direction4priors.csv `
  --out outputs/full_run `
  --models static_hmm covariate_hmm `
  --n-simulations 100 `
  --seed 42 `
  --resume `
  --sp-sensitivity `
  --render-report
```

The S/P sensitivity reuses existing simulations. It applies the same smoothed forward-backward marginal-MAP lapse exclusion to observed and simulated trials. It is not Viterbi decoding and is not an online lapse detector.

## Data setup

The repository already includes an analysis-ready copy of the source dataset in the teammate contribution. Copy it into this package before running:

```powershell
Copy-Item `
  ..\Limitation_Analysis_of_HMM_Temporal_Persistence\HMM\01_full_data_fit\data\data01_direction4priors.csv `
  .\data\data01_direction4priors.csv
```

The loader drops exactly three rows lacking response coordinates, leaving 83,210 trials in 388 subject-session-run sequences.

## Main interfaces

- `scripts/run_all.py`: complete deterministic analysis.
- `scripts/run_fast_replication.py`: reduced-restart scientific check.
- `scripts/run_posterior_predictive_checks.py`: static/covariate HMM PPC and S/P sensitivity.
- `scripts/export_trial_results.py`: final and out-of-fold trial-level predictions.
- `scripts/render_results.py`: fit-independent report regeneration from complete outputs.
- `src/perceptual_arbitration/hmm.py`: static three-state HMM.
- `src/perceptual_arbitration/covariate_hmm.py`: covariate-dependent transition HMM.
- `src/perceptual_arbitration/independent_switching.py`: memoryless switching mixture.
- `src/perceptual_arbitration/serial_dependence.py`: one-back serial controls.
- `src/perceptual_arbitration/trial_exports.py`: detailed model-by-trial export layer.
- `PROPOSAL.md`: model specification and mathematical rationale.

## Included result snapshot

The `results/` directory contains the final figures, aggregate tables, manifests, and a self-contained HTML report. These files are included for review without requiring a long refit.

The portable report can be shared as a single HTML file:

[`results/report/perceptual_arbitration_results.html`](results/report/perceptual_arbitration_results.html)

Report regeneration uses the Codex Data Analytics portable-report builder. Readers without that local builder can use the committed self-contained report; model fitting and standard Matplotlib figures do not depend on it.

## Validation

```powershell
python -m pytest -q
python -m compileall -q src scripts tests
```

The publication snapshot was validated with 36 passing tests. Final report verification covered 1440-pixel and 390-pixel viewports.
