# Research Proposal: Temporally Persistent Strategy Arbitration in Human Perceptual Inference

## 0. Project in one sentence

This study tests whether the Switching Observer account of human perceptual estimation should be extended from an independent trial-wise switching process to a dynamic latent-state process in which observers persist for several trials in sensory-reliance, prior-reliance, or lapse/random-response states.

## 1. Scientific background

Laquitaine and Gardner showed that human estimates in a motion-direction task can look Bayesian when summarized by mean and variance, but the full response distributions are often bimodal: one peak near the true stimulus direction and one peak near the learned prior mean. Their Basic Bayesian observer multiplies prior and likelihood into a posterior; that model predicts a single posterior-like response peak. Their Switching Observer instead switches between reporting from the sensory likelihood and reporting from the prior, explaining the bimodality. The present project keeps their insight but asks a new question: is switching independent from trial to trial, or does it show temporal persistence?

The original experiment is ideal for this question because it contains ordered trial-level responses, stimulus directions, motion coherence, prior width, prior mean, subject, session, run/block, and feedback information. Because trial order is preserved, we can fit sequential models rather than only static response distributions.

## 2. Conceptual correction about the prior

The experiment did not explicitly train subjects on a separate prior-learning phase. Instead, each block both generated experimental data and allowed subjects to learn the statistics of directions through repeated feedback.

The prior distribution in a block is approximately:

\[
p(\theta_t \mid b) = \mathrm{VM}(\theta_t; \mu, \kappa_b),
\]

where:

\[
\mu = 225^\circ
\]

is fixed across blocks, and the prior width changes by block:

\[
\sigma_b \in \{10^\circ,20^\circ,40^\circ,80^\circ\}.
\]

Thus, the prior mean is best interpreted as a stable global task regularity that subjects can learn strongly across the experiment, while the prior width/precision is the block-specific quantity they must learn or adapt to.

This is important for our model: the prior-reliance state is not a state of repeatedly learning a new prior mean. It is a state of relying on a stable learned prior center, with response precision modulated by the current block's prior width.

## 3. Core hypothesis

### 3.1 Main hypothesis

Human perceptual estimation is governed by temporally persistent latent inference states. On any given trial, an observer may rely mainly on sensory evidence, rely mainly on the learned prior, or lapse/respond randomly. Critically, these states persist across consecutive trials.

### 3.2 Formal hypothesis

Let:

\[
z_t \in \{S,P,L\}
\]

where:

- \(S\): sensory-reliance state,
- \(P\): prior-reliance state,
- \(L\): lapse/random-response state.

The central claim is:

\[
p(z_t = S \mid z_{t-1}=S) > p(z_t=S)
\]

and

\[
p(z_t = P \mid z_{t-1}=P) > p(z_t=P).
\]

Equivalently, the transition matrix should show high self-transition probabilities:

\[
A_{SS} \gg 1/3, \qquad A_{PP} \gg 1/3.
\]

### 3.3 Secondary hypotheses

1. **Uncertainty dependence:** temporal persistence should be strongest when sensory evidence is weak or moderate, especially at 6% and 12% coherence.
2. **Prior-precision dependence:** prior-state occupancy should increase when prior width is narrow, especially 10° and 20° prior blocks.
3. **Conflict dependence:** prior-sensory conflict \(|\theta_t-\mu|\) should modulate switching because large conflict makes prior-like and sensory-like responses easier to distinguish and may increase arbitration demands.
4. **Feedback-error dependence:** large previous-trial error may increase probability of leaving the current state, but this should be treated as secondary/exploratory because preliminary tests showed weaker evidence.
5. **Not merely serial dependence:** temporal persistence should remain after comparing against previous-stimulus and previous-response serial-dependence baselines.

## 4. Existing dataset structure

### 4.1 Hierarchy

\[
\text{Experiment} \rightarrow \text{Subject} \rightarrow \text{Session} \rightarrow \text{Block/run} \rightarrow \text{Trial}
\]

### 4.2 Main experiment

The main analysis uses the motion-direction experiment with four prior widths.

- 12 subjects.
- Each subject completed at least 5 sessions.
- Each session contained about 4-5 blocks.
- Each block contained about 200 trials.
- Each block had one prior width: 10°, 20°, 40°, or 80°.
- Prior mean was fixed at 225°.
- Motion coherence varied trial by trial: 6%, 12%, 24%.
- True direction was one of 36 directions from 5° to 355° in 10° steps.

### 4.3 One trial

One trial has:

1. fixation for about 1000 ms;
2. random-dot motion stimulus for about 300 ms;
3. response phase, where the subject rotates a line to report perceived direction;
4. response confirmation;
5. true-direction feedback.

Subjects were told to report the motion direction accurately and quickly, but they were not explicitly told the Bayesian/switching purpose of the experiment or explicitly trained on the prior distribution.

### 4.4 Important variables

For trial \(t\):

- \(y_t = \hat\theta_t\): subject's reported direction.
- \(\theta_t\): true motion direction.
- \(c_t \in \{0.06,0.12,0.24\}\): motion coherence.
- \(\mu_t = 225^\circ\): prior mean.
- \(\sigma_t \in \{10^\circ,20^\circ,40^\circ,80^\circ\}\): prior width.
- \(e_t = |y_t-\theta_t|\): response error.
- \(d_t = |\theta_t-\mu_t|\): prior-stimulus conflict.

All angular differences are circular differences.

## 5. Primary model: Hidden Markov Switching Observer

### 5.1 Hidden states

\[
z_t \in \{S,P,L\}.
\]

### 5.2 Emission distributions

Responses are circular, so we use the von Mises distribution:

\[
\mathrm{VM}(y;m,\kappa)=\frac{\exp(\kappa\cos(y-m))}{2\pi I_0(\kappa)}.
\]

Here \(m\) is the circular mean and \(\kappa\) is concentration/precision.

#### Sensory state

\[
p(y_t \mid z_t=S)=\mathrm{VM}(y_t;\theta_t,\kappa^S_{c_t}).
\]

The sensory-state response is centered on the true stimulus direction, with precision depending on motion coherence.

Prediction:

\[
\kappa^S_{24\%} > \kappa^S_{12\%} > \kappa^S_{6\%}.
\]

#### Prior state

\[
p(y_t \mid z_t=P)=\mathrm{VM}(y_t;\mu_t,\kappa^P_{\sigma_t}).
\]

The prior-state response is centered on the learned prior mean, with precision depending on prior width.

Prediction:

\[
\kappa^P_{10^\circ} > \kappa^P_{20^\circ} > \kappa^P_{40^\circ} > \kappa^P_{80^\circ}.
\]

#### Lapse state

\[
p(y_t \mid z_t=L)=\frac{1}{2\pi}.
\]

This captures random responses, attention lapses, motor errors not captured by other states, or other heavy-tailed behavior.

### 5.3 Joint probability

For a sequence of \(T\) trials:

\[
p(y_{1:T},z_{1:T})
= \pi_{z_1}p(y_1\mid z_1)
\prod_{t=2}^{T} A_{z_{t-1},z_t}p(y_t\mid z_t).
\]

\(\pi_i\) is the initial state probability, and \(A_{ij}\) is the transition matrix:

\[
A_{ij}=p(z_t=j\mid z_{t-1}=i).
\]

### 5.4 Main test

The model supports the hypothesis if:

1. held-out log likelihood is higher than the independent Switching mixture;
2. \(A_{SS}\) and \(A_{PP}\) are high;
3. these effects survive subject-level and serial-dependence controls.

## 6. Baseline and comparison models

The study should not compare only HMM versus independent mixture. It should include a family of increasingly strong baselines.

### 6.1 Basic Bayesian observer

This is the normative benchmark from the original paper.

A noisy sensory measurement \(\theta^e_t\) is drawn from:

\[
p(\theta^e_t\mid \theta_t)=\mathrm{VM}(\theta^e_t;\theta_t,\kappa^E_{c_t}).
\]

The sensory likelihood is:

\[
p(\theta_t \mid \theta^e_t) \propto \mathrm{VM}(\theta_t;\theta^e_t,\kappa^E_{c_t}).
\]

The prior is:

\[
p(\theta_t)=\mathrm{VM}(\theta_t;\mu_t,\kappa^P_{\sigma_t}).
\]

The posterior is:

\[
p(\theta_t\mid \theta^e_t) \propto p(\theta^e_t\mid\theta_t)p(\theta_t).
\]

The reported percept can be the posterior mode, posterior mean, or posterior sample. Motor noise and lapse are added to form the response distribution.

Use this model to show why single-posterior integration is insufficient for the full distribution.

### 6.2 Original condition-dependent Switching observer

The original Switching observer has two response sources: sensory and prior. The mixing probability depends on relative sensory and prior precision:

\[
p_P(c_t,\sigma_t)=\frac{\kappa^P_{\sigma_t}}{\kappa^P_{\sigma_t}+\kappa^S_{c_t}},
\]

\[
p_S(c_t,\sigma_t)=1-p_P(c_t,\sigma_t).
\]

Then:

\[
p(y_t)=p_S\mathrm{VM}(y_t;\theta_t,\kappa^S_{c_t})+p_P\mathrm{VM}(y_t;\mu_t,\kappa^P_{\sigma_t})+p_L\frac{1}{2\pi}.
\]

This tests whether condition-dependent static switching explains the data without temporal dynamics.

### 6.3 Independent Switching mixture

This baseline is like the HMM emission model but without temporal dependence:

\[
p(y_t)=w_Sp(y_t\mid S)+w_Pp(y_t\mid P)+w_Lp(y_t\mid L).
\]

Here:

\[
p(z_t\mid z_{t-1})=p(z_t).
\]

This is the cleanest baseline for isolating the effect of temporal persistence. If the HMM wins against this model, the improvement is due to sequence structure rather than just having sensory/prior/lapse components.

### 6.4 Serial-dependence baseline

A major alternative explanation is that current responses are attracted to the previous stimulus or previous response. This is different from latent-state persistence.

Previous-stimulus attraction:

\[
m^S_t = \theta_t + \alpha_\theta \Delta(\theta_{t-1},\theta_t),
\]

where \(\Delta(a,b)\) is signed circular difference \(a-b\).

Previous-response attraction:

\[
m^S_t = \theta_t + \alpha_y \Delta(y_{t-1},\theta_t).
\]

Combined serial baseline:

\[
m^S_t = \theta_t + \alpha_\theta \Delta(\theta_{t-1},\theta_t)+\alpha_y \Delta(y_{t-1},\theta_t).
\]

The serial-dependence independent Switching model is:

\[
p(y_t)=w_S\mathrm{VM}(y_t;m^S_t,\kappa^S_{c_t})+w_P\mathrm{VM}(y_t;\mu_t,\kappa^P_{\sigma_t})+w_L\frac{1}{2\pi}.
\]

This is critical because the original paper already argued that the main prior effect is not merely one-back previous-stimulus attraction. Our paper should go further and show that temporal persistence remains superior to explicit serial-dependence baselines.

### 6.5 Covariate-dependent HMM / input-output HMM

The static HMM has one transition matrix for all trials. The covariate-dependent HMM lets transitions depend on experimental and history variables:

\[
p(z_t=j\mid z_{t-1}=i,x_t)=\frac{\exp(\beta_{ij}^{\top}x_t)}{\sum_{k}\exp(\beta_{ik}^{\top}x_t)}.
\]

Covariates:

\[
x_t = [1, c_t, \mathrm{precision}(\sigma_t), |\theta_t-\mu_t|, e_{t-1}, |\theta_{t-1}-\mu_{t-1}|, c_{t-1}].
\]

where:

- \(c_t\): current coherence;
- \(\mathrm{precision}(\sigma_t)=1/\sigma_t\): prior precision proxy;
- \(|\theta_t-\mu_t|\): current prior-sensory conflict;
- \(e_{t-1}=|y_{t-1}-\theta_{t-1}|\): previous error;
- \(|\theta_{t-1}-\mu_{t-1}|\): previous conflict;
- \(c_{t-1}\): previous sensory reliability.

Key predicted signs:

- Higher coherence should increase probability of entering/staying in sensory state.
- Higher prior precision should increase probability of entering/staying in prior state.
- Larger previous error may reduce staying probability, especially after prior-state errors, but this is exploratory.

### 6.6 Subject-level and hierarchical models

Subjects may differ in strategy. Fit subject-specific HMMs:

\[
A^{(s)}, \quad \kappa^{S,(s)}_c, \quad \kappa^{P,(s)}_\sigma, \quad \pi^{(s)}.
\]

Then summarize group-level distributions:

\[
\mathrm{logit}(A^{(s)}_{ij}) \sim \mathcal{N}(\mu_{ij},\tau^2_{ij}).
\]

A fully Bayesian implementation could be built in Stan, PyMC, or NumPyro. The included code implements a practical first version: fit each subject separately with fully converged multi-start EM, then compute empirical-Bayes group means and between-subject standard deviations. A later version can replace this with full random-effects inference.

## 7. Mathematical training derivation

### 7.1 E-step for static HMM

Forward recursion:

\[
\alpha_t(j)=p(y_{1:t},z_t=j).
\]

\[
\alpha_1(j)=\pi_jp(y_1\mid z_1=j).
\]

\[
\alpha_t(j)=p(y_t\mid z_t=j)\sum_i\alpha_{t-1}(i)A_{ij}.
\]

Backward recursion:

\[
\beta_t(i)=p(y_{t+1:T}\mid z_t=i).
\]

\[
\beta_T(i)=1.
\]

\[
\beta_t(i)=\sum_j A_{ij}p(y_{t+1}\mid z_{t+1}=j)\beta_{t+1}(j).
\]

Posterior state probability:

\[
\gamma_t(i)=p(z_t=i\mid y_{1:T})=\frac{\alpha_t(i)\beta_t(i)}{p(y_{1:T})}.
\]

Posterior transition probability:

\[
\xi_t(i,j)=p(z_t=i,z_{t+1}=j\mid y_{1:T})
\]

\[
=\frac{\alpha_t(i)A_{ij}p(y_{t+1}\mid z_{t+1}=j)\beta_{t+1}(j)}{p(y_{1:T})}.
\]

The code uses log-space recursions to avoid underflow.

### 7.2 M-step for static HMM

Initial state:

\[
\pi_i=\frac{\sum_s \gamma^{(s)}_1(i)+\epsilon}{\sum_k\sum_s\gamma^{(s)}_1(k)+K\epsilon}.
\]

Transition matrix:

\[
A_{ij}=\frac{\sum_s\sum_t \xi^{(s)}_t(i,j)+\epsilon}{\sum_k\sum_s\sum_t\xi^{(s)}_t(i,k)+K\epsilon}.
\]

Sensory concentration for coherence level \(c\):

\[
R^S_c=\frac{\sum_{t:c_t=c}\gamma_t(S)\cos(y_t-\theta_t)}{\sum_{t:c_t=c}\gamma_t(S)}.
\]

Then solve:

\[
\frac{I_1(\kappa^S_c)}{I_0(\kappa^S_c)}=R^S_c.
\]

Prior concentration for prior width \(\sigma\):

\[
R^P_\sigma=\frac{\sum_{t:\sigma_t=\sigma}\gamma_t(P)\cos(y_t-\mu_t)}{\sum_{t:\sigma_t=\sigma}\gamma_t(P)}.
\]

Then solve:

\[
\frac{I_1(\kappa^P_\sigma)}{I_0(\kappa^P_\sigma)}=R^P_\sigma.
\]

### 7.3 M-step for covariate-dependent transitions

For the input-output HMM, the expected transition log-likelihood is:

\[
Q(\beta)=\sum_s\sum_t\sum_i\sum_j\xi^{(s)}_t(i,j)\log p(z_{t+1}=j\mid z_t=i,x_{t+1}).
\]

For each previous state \(i\), this is a weighted multinomial logistic regression. The code optimizes it with L-BFGS, using \(\xi_t(i,j)\) as soft transition counts.

## 8. Training and testing plan

### 8.1 Data preprocessing

1. Load CSV.
2. Drop trials with missing response coordinates.
3. Convert response coordinates to angle:

\[
y_t=\mathrm{atan2}(\mathrm{estimate\_y},\mathrm{estimate\_x}).
\]

4. Sort by subject, session, run, trial index.
5. Build continuous sequences at the run/block level.
6. Compute circular differences:

\[
\cos(y_t-\theta_t), \quad \cos(y_t-\mu_t), \quad |y_t-\theta_t|, \quad |\theta_t-\mu_t|.
\]

### 8.2 Cross-validation

Use sequence-level cross-validation, not random trial-level cross-validation. This prevents temporal leakage.

Default:

- 4-fold sequence-stratified CV.
- Stratify by subject and prior width.
- Train on 3/4 of run-level sequences.
- Test on held-out run-level sequences.

Additional robustness:

- leave-one-subject-out CV;
- leave-one-session-out CV;
- first-half train / second-half test within subjects;
- train on some prior widths and test on held-out prior widths if enough data allow.

### 8.3 Multi-start convergence

For each EM model:

- use 25 random restarts for static HMM and independent mixture;
- use 10 or more random restarts for covariate-HMM because each iteration is heavier;
- maximum 1000 EM iterations;
- stop only when relative log-likelihood improvement is less than tolerance after minimum iterations;
- retain the restart with maximum training log likelihood;
- report convergence status and number of iterations.

### 8.4 Model comparison metrics

Primary metric:

\[
\mathrm{held\text{-}out\ log\ likelihood\ per\ trial}.
\]

Secondary metrics:

- total held-out log likelihood;
- AIC/BIC for in-sample comparisons;
- WAIC/LOO only if using full Bayesian hierarchical model;
- posterior predictive checks;
- paired fold-level or sequence-bootstrap confidence intervals.

The core comparison is:

\[
\Delta LL = LL_{HMM} - LL_{IndependentSwitching}.
\]

If \(\Delta LL > 0\) consistently across folds, temporal persistence improves prediction.

## 9. Preliminary results already obtained

A fast first-pass implementation was run on the main four-prior motion-direction CSV. This preliminary run used 83,210 usable trials and 388 run-level sequences. It used 4-fold sequence-stratified cross-validation.

### 9.1 Cross-validated HMM vs independent mixture

The HMM beat the independent Switching mixture in every fold.

| Fold | HMM test LL/trial | Independent test LL/trial | Difference |
|---:|---:|---:|---:|
| 1 | -0.8324 | -0.8784 | +0.0459 |
| 2 | -0.8371 | -0.8797 | +0.0426 |
| 3 | -0.8525 | -0.9120 | +0.0595 |
| 4 | -0.8806 | -0.9482 | +0.0676 |

Mean improvement:

\[
\Delta LL/\mathrm{trial}=0.0539.
\]

Exponentiating:

\[
\exp(0.0539)\approx 1.055.
\]

So the HMM assigned about 5.5% higher predictive density per held-out trial than the independent mixture.

### 9.2 Preliminary transition matrix

The all-data static HMM learned:

\[
A=
\begin{bmatrix}
0.898 & 0.064 & 0.038\\
0.098 & 0.901 & 0.001\\
0.395 & 0.012 & 0.593
\end{bmatrix}.
\]

Rows are previous state; columns are next state, in order \(S,P,L\).

The key estimates are:

\[
A_{SS}=0.898,
\]

\[
A_{PP}=0.901.
\]

This is strong preliminary evidence for temporally persistent sensory-reliance and prior-reliance states.

### 9.3 Preliminary emission parameters

Sensory concentration increased with coherence:

| Coherence | \(\kappa^S\) |
|---:|---:|
| 6% | 1.94 |
| 12% | 5.43 |
| 24% | 14.72 |

Prior concentration decreased with prior width:

| Prior width | \(\kappa^P\) |
|---:|---:|
| 10° | 29.10 |
| 20° | 16.46 |
| 40° | 7.01 |
| 80° | 1.49 |

These trends are psychologically sensible and support the interpretation that the states correspond to sensory and prior reliance.

### 9.4 Caution about preliminary results

The fast preliminary run used a limited number of EM iterations and fewer restarts. The full pipeline in this repository implements fully converged multi-start EM. The preliminary results should motivate the study, not replace publication-grade fitting.

## 10. Posterior predictive checks

After fitting models, simulate responses and compare to real data.

### 10.1 Distribution checks

For each condition:

- coherence: 6%, 12%, 24%;
- prior width: 10°, 20°, 40°, 80°;
- conflict bins: near prior, medium conflict, far from prior;

compare simulated and observed:

- response histograms;
- mean bias toward prior;
- circular standard deviation;
- bimodality index;
- prior-like response rate;
- run-length distribution of prior-like and sensory-like responses;
- transition counts between inferred states.

### 10.2 Temporal checks

Check whether the model reproduces:

\[
P(P_t\mid P_{t-1}) > P(P_t\mid S_{t-1}),
\]

and analogous sensory persistence.

Also compare state run lengths under observed and simulated data.

## 11. Analysis sequence for Codex implementation

### Stage 1: Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest
```

### Stage 2: Data

```bash
mkdir -p data
# Put data01_direction4priors.csv in data/
```

### Stage 3: Fast sanity run

```bash
python scripts/run_fast_replication.py --csv data/data01_direction4priors.csv --out outputs/fast --restarts 2 --max-iter 100
```

Expected sanity check:

- HMM should beat independent mixture.
- \(A_{SS}\) and \(A_{PP}\) should be high.
- \(\kappa^S\) should increase with coherence.
- \(\kappa^P\) should decrease with prior width.

### Stage 4: Full model comparison

```bash
python scripts/run_all.py --config configs/default.yaml
```

This produces:

- `outputs/full_run/cv_results.csv`
- `outputs/full_run/cv_summary.csv`
- `outputs/full_run/subject_level_hmm.csv`
- `outputs/full_run/empirical_bayes_group_summary.csv`

### Stage 5: Optional Bayesian baselines

```bash
python scripts/run_basic_bayesian_baselines.py --csv data/data01_direction4priors.csv --out outputs/bayesian_baselines --restarts 5 --maxiter 300
```

### Stage 6: Figures

Create figures:

1. Experiment schematic.
2. Response histograms by condition.
3. Model comparison bar plot: held-out LL per trial.
4. HMM transition matrix heatmap.
5. Subject-level \(A_{SS}\) and \(A_{PP}\) dot plot.
6. Covariate-HMM transition effects.
7. Posterior predictive checks.
8. Serial-dependence alpha estimates and model comparison.

## 12. Full-spec new experiments

The existing dataset can support a computational behavioral paper. A stronger cognitive neuroscience paper needs new data designed specifically to separate prior-mean learning, prior-precision learning, latent-state persistence, and feedback-error effects.

### Experiment 1: Replication with changing prior means

#### Aim

Test whether latent prior-reliance states track a learned prior mean when the mean changes across blocks.

#### Design

- Participants: 30-50.
- Motion direction estimation task.
- Prior means vary across blocks:

\[
\mu_b \in \{45^\circ,135^\circ,225^\circ,315^\circ\}.
\]

- Prior widths:

\[
\sigma_b \in \{15^\circ,45^\circ,80^\circ\}.
\]

- Coherence:

\[
c_t \in \{6\%,12\%,24\%\}.
\]

- Blocks: 12-16 blocks per subject.
- Trials per block: 120-180.
- Feedback: true direction shown after each trial.

#### Predictions

1. Prior-reliance state should center on the current block's learned mean, not a fixed global direction.
2. The HMM should still beat independent switching and serial-dependence baselines.
3. Prior-state occupancy should increase with narrower prior width.
4. State persistence should be strongest in low coherence.

### Experiment 2: Stable versus volatile prior environment

#### Aim

Test whether environmental volatility controls state persistence.

#### Conditions

1. **Stable condition:** prior mean remains fixed for long blocks.
2. **Volatile condition:** prior mean changes unpredictably every 30-60 trials.

#### Prediction

Stable condition:

\[
A_{PP}\text{ high}
\]

Volatile condition:

\[
A_{PP}\text{ lower}, \quad P\rightarrow S\text{ transitions higher}.
\]

The covariate-HMM should show that surprise or changepoint probability increases switching.

### Experiment 3: Feedback manipulation

#### Aim

Test whether previous error causally drives switching.

#### Conditions

1. Full true-direction feedback.
2. No feedback.
3. Delayed feedback.
4. Occasionally perturbed feedback, used carefully and ethically with debriefing.

#### Prediction

If feedback error causes state switching, previous-error coefficients in the covariate-HMM should be strong only when feedback is available and reliable.

Given preliminary weak evidence, this should be framed as an empirical question rather than a guaranteed effect.

### Experiment 4: Neural or physiological extension

#### Aim

Connect latent states to cognitive neuroscience signals.

Options:

1. EEG: test whether alpha/beta power or centroparietal positivity predicts state transitions.
2. Pupillometry: test whether arousal predicts switches or lapse state.
3. fMRI: test whether frontoparietal control regions encode state arbitration, while visual areas encode sensory and prior-centered representations.

#### Prediction

State switches should be preceded by control/arousal markers, while prior-state trials should show stronger expectation-related signals.

## 13. Expected paper structure

### Title

**Temporally Persistent Strategy Arbitration in Human Perceptual Inference**

### Abstract skeleton

Human perceptual estimates often appear Bayesian in summary statistics, yet response distributions can reveal switching between sensory evidence and learned priors. Here we test whether such switching is an independent trial-wise mixture or a temporally persistent latent-state process. Reanalyzing trial-level motion-direction estimation data, we compare an independent Switching observer, serial-dependence baselines, and Hidden Markov Switching observers with static and covariate-dependent transitions. A preliminary static HMM predicts held-out responses better than an independent mixture and learns strong self-transition probabilities for sensory- and prior-reliance states. The proposed analyses test whether switching reflects dynamic strategy arbitration rather than one-back serial dependence. This framework reframes prior use in perception as a history-dependent control process operating under uncertainty.

### Paper sections

1. Introduction: Bayesian inference, heuristic switching, and temporal dynamics.
2. Dataset and task structure.
3. Models.
4. Training and model comparison.
5. Results: replication of switching, HMM improvement, transition matrix, covariates, subject differences, serial-dependence controls.
6. Discussion: computational interpretation, cognitive neuroscience mechanisms, limitations, future experiments.

## 14. Interpretation rules

### Strong conclusion allowed if results hold

If the static HMM and covariate-HMM beat independent switching and serial-dependence baselines in held-out likelihood, and if \(A_{SS}\) and \(A_{PP}\) remain high across subjects, then we can conclude:

> Human perceptual estimation contains temporally persistent latent inference states corresponding to sensory reliance and prior reliance.

### Conservative neural interpretation

Because the existing dataset is behavioral, do not claim direct neural evidence. Say:

> The latent states provide a computational cognitive-neuroscience hypothesis about how brain systems might arbitrate between sensory and prior representations.

### Avoid overclaiming feedback error

Preliminary evidence for feedback-error-driven switching was weak. Treat previous error as a covariate and report the result, but do not make it the main claim unless the covariate-HMM strongly supports it.

## 15. Main risks and solutions

### Risk 1: HMM captures serial dependence rather than strategy persistence

Solution: include previous-stimulus and previous-response serial-dependence baselines and combined serial + switching models.

### Risk 2: HMM captures block boundaries or slow fatigue

Solution: split sequences by run/block, include trial number within block as a covariate, and run robustness analyses excluding early trials.

### Risk 3: prior state simply reflects low coherence

Solution: include coherence as a covariate and compare persistence within each coherence level.

### Risk 4: subject heterogeneity hides effects

Solution: fit subject-level HMMs and report random-effects summaries.

### Risk 5: local maxima in EM

Solution: multi-start fully converged EM with convergence diagnostics and held-out validation.

## 16. Code map

- `src/perceptual_arbitration/circular.py`: circular statistics and von Mises utilities.
- `src/perceptual_arbitration/data.py`: data loading, angle conversion, sequence construction, transition covariates.
- `src/perceptual_arbitration/independent_switching.py`: independent Switching mixture EM.
- `src/perceptual_arbitration/hmm.py`: static Hidden Markov Switching Observer with multi-start EM.
- `src/perceptual_arbitration/covariate_hmm.py`: covariate-dependent transition HMM using weighted multinomial logistic M-step.
- `src/perceptual_arbitration/serial_dependence.py`: previous-stimulus/previous-response serial-dependence baselines.
- `src/perceptual_arbitration/bayesian.py`: optional Basic Bayesian and condition-dependent Switching baselines.
- `src/perceptual_arbitration/model_selection.py`: cross-validation, model comparison, subject-level fits, empirical-Bayes summaries.
- `scripts/run_fast_replication.py`: quick sanity check.
- `scripts/run_all.py`: full publication-grade run.
- `scripts/run_basic_bayesian_baselines.py`: optional slower original-paper baselines.

## 17. Immediate Codex task list

1. Verify package imports and run unit tests.
2. Place the CSV in `data/data01_direction4priors.csv`.
3. Run `scripts/run_fast_replication.py`.
4. Compare fast results to preliminary values in this proposal.
5. Run full static HMM and independent mixture with 25 restarts.
6. Run serial-dependence baselines.
7. Run covariate-HMM.
8. Run subject-level HMMs.
9. Add plotting scripts.
10. Add posterior predictive simulation scripts.
11. Add bootstrap confidence intervals.
12. Write results section from generated tables.

## 18. Bottom-line proposal claim

The proposed study tests whether human perceptual inference is not merely Bayesian integration or independent heuristic switching, but a temporally structured arbitration process. The key empirical signature is that prior- and sensory-reliance states persist across trials and improve held-out prediction beyond independent switching and serial-dependence baselines.
