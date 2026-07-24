# Curated publication results

This directory is a compact, reviewable snapshot of the authoritative analysis in `outputs/full_run`.

## Included

- `figures/`: all numbered publication figures in 300-DPI PNG and vector SVG formats.
- `report/perceptual_arbitration_results.html`: self-contained technical report with embedded figures, equations, tables, and source metadata.
- `report/data/`: aggregate tables used by the report.
- `tables/`: central model-comparison, parameter, covariate, subject, convergence, and posterior-predictive tables.
- `provenance/`: run and posterior-predictive manifests from the completed analysis.

## Intentionally omitted

The following outputs are reproducible from the code but are not committed because they are large or contain redundant trial-level detail:

- raw posterior-predictive simulations;
- model checkpoints and restart checkpoints;
- complete trial-level exports;
- trial-level posterior-state tables;
- virtual environments, caches, and temporary files.

The provenance manifests retain the original output hashes and therefore refer to some files that are not included in this curated repository snapshot. Run the full pipeline to reconstruct the complete output directory.

The all-trial posterior predictive check is the primary absolute-adequacy analysis. The decoded sensory/prior-only check is a supplementary sensitivity analysis and does not replace it.
