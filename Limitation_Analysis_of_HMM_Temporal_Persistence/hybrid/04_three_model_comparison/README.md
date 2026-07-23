# Alternative-model challenge to the HMM temporal interpretation

This analysis is not framed as a contest in which the HMM states are simply
"defeated." It tests whether the predictive structure attributed to
trial-to-trial HMM persistence can also be explained by a baseline with
longer-timescale block structure and no latent-state transitions. The hybrid
then tests whether local sequential structure adds predictive value after that
block-level structure is represented.

The saved evaluations do not select one model uniformly under every criterion.

| Evaluation | HMM | Exchangeable block mixture | Hybrid |
|---|---:|---:|---:|
| Random four-fold LL/trial | -0.810573 | -0.794568 | -0.787089 |
| Chronological LL/trial | -0.821294 | -0.802133 | -0.787565 |
| Full-data AIC rank | 3 | 2 | 1 |
| Full-data subject-wise BIC rank | 3 | 1 | 2 |

Under the saved held-out evaluations, the fitted hybrid had the highest
predictive density, followed by the exchangeable block mixture and the original
HMM. The exchangeable block model's advantage over the original HMM shows that
trial-to-trial Markov transitions are not required to obtain better prediction
under these implementations. The hybrid's further improvement suggests that
block-level stability does not necessarily exhaust all sequential predictive
structure. AIC favored the hybrid, while BIC favored the exchangeable block
mixture.

The hybrid also had the weakest numerical convergence: 20/48 strict
convergence in random four-fold fitting, 25/48 in chronological fitting, and
6/12 in full-data fitting. The model rankings should therefore be reported
together with convergence status. The comparisons support predictive
differences among these implementations; they do not prove that any latent
state interpretation is the true psychological mechanism. More narrowly, the
results support treating the three HMM states as useful descriptions of
possible response strategies while avoiding the stronger claim that high
diagonal transition probabilities directly reveal a first-order psychological
switching law. Those probabilities may partly be the HMM's representation of
stable participant- or block-level response tendencies.

## Clearly named source tables

- `results/random_fourfold_three_model_comparison.csv`
- `results/chronological_four_model_comparison.csv`
- `results/baseline_and_hmm_fourfold_comparison.csv`
- `results/three_model_information_criteria.csv`
- `results/three_model_subject_information_criteria.csv`

The shorter generic filenames in the same directory are preserved migrated
source tables. The clearly named versions above are preferred for public use.
