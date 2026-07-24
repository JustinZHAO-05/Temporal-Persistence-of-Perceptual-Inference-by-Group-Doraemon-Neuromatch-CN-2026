# Report source notes

Generated: 2026-07-17T14:19:33.368379+00:00
Snapshot status: ready

## Audience and structure

The visible report is written for a mixed audience. Full model derivations are native MathML in keyboard-accessible disclosure panels collapsed by default.
Every graph has an adjacent question, encoding guide, uncertainty description, numerical takeaway, and caveat.
Teaching dictionaries are generated from implementation definitions and final output tables under report/data/.

## Omitted analyses

The optional Basic Bayesian and original condition-dependent Switching models are explained for context but were intentionally excluded from the authoritative numerical comparison.
The posterior predictive check uses 100 complete replicated datasets per model for both final HMMs. Coverage is diagnostic rather than a formal pass/fail decision; publication readiness does not imply absolute model adequacy.
The supplementary S/P-only sensitivity reuses the same simulations and excludes smoothed marginal-MAP lapse classifications symmetrically from observed and simulated response metrics.
Covariate HMM coverage improved from 19/60 to 24/60 condition-metric cells; Static HMM coverage deteriorated from 2/60 to 0/60 condition-metric cells. The complete all-trial PPC remains primary, and the all-state run-length check is unchanged.

## Publication issues
