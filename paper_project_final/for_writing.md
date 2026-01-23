# Information for LLM Agent: Results Section Revision

This document provides all the background, context, and materials needed to revise the Results section of a neuroscience paper about information flow between brain areas.

---

## 1. Project Overview

### Scientific Question
This analysis investigates **directed information flow** between brain areas (FEF=Frontal Eye Field, LIP=Lateral Intraparietal area, SC=Superior Colliculus) during a categorization-saccade task in macaque monkeys. The core question is: **Which brain areas encode task-relevant information first, and how does this information flow between areas over time?**

### Key Findings (Summary)
- FEF consistently leads LIP in both **category encoding latency** and **saccade direction encoding latency** at the single-trial level
- **Directed information flow** (FEF→LIP) dominates over the reverse direction (LIP→FEF) during task-relevant epochs
- Results are consistent across two monkeys (M and S)

### Analysis Approach
1. **Session-wise linear feature axes**: For each session and brain area, we train a linear decoder to identify an "encoding axis" for category or saccade direction
2. **Single-trial evidence traces**: Project each trial's population activity onto the feature axis to get a time-varying "feature evidence" signal
3. **Single-trial onset latency**: Define when feature evidence first exceeds baseline threshold for 5 consecutive bins
4. **Directed flow (lagged ridge regression)**: Test whether past activity in area A improves prediction of current activity in area B, beyond B's own self-history

---

## 2. Key Methodological Parameters

From recent correspondence, here are the finalized methodological choices:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Binning** | 20-ms bins, 10-ms step | Sliding window approach for temporal smoothing |
| **FEF-LIP category lag** | 80 ms | For stimulus-aligned category flow |
| **FEF-LIP saccade lag** | 50 ms | For saccade-aligned flow |
| **SC lag** | 30 ms | Uses 10-ms bins (20-ms too coarse for shorter lag) |
| **QC threshold** | AUC ≥ 0.65 | For session inclusion |
| **Significance** | p < 0.05 | Permutation-based (stratified trial shuffle) |

### Lag Selection Rationale
- Guided by Oliver's population latency results and single-trial latency estimates
- Different area pairs have different lead/lag patterns
- Sanity checks performed with multiple lags (30, 50, 80, 100 ms) to verify pattern stability

---

## 3. Figure Structure

The main figure (Figure X) has a **Top row** (category, stimulus-aligned) and **Bottom row** (saccade direction, saccade-aligned):

### Panel A: Session QC Overlays
- Shows time-resolved decoding performance (AUC) for each session
- Each trace = one session, with transparency to show variability
- Demonstrates quality of feature subspaces across sessions
- X-axis: Time from event onset (ms)
- Y-axis: AUC

### Panel B: Single-Trial Onset Latency Scatter
- Scatter plot comparing FEF vs LIP trial-wise onset latencies
- Each point = one trial with defined onset in both areas
- Dashed line = y=x (equality)
- Shows FEF systematically leads LIP
- Reports: N trials, p-value from sign-flip permutation test

### Panel C/F: Directed Predictability Gain (Net Flow)
- Shows **both directions**: FEF→LIP (blue/orange) and LIP→FEF (orange/blue)
- Y-axis: "Directed predictability gain (bits)" - the improvement in predicting area B from area A's history, minus model complexity baseline
- Black dots: Time bins with significant net flow (p < 0.05)
- Demonstrates asymmetry: FEF→LIP shows transient event-aligned increase, LIP→FEF remains near zero

### X-axis Labels (per Dave's request)
- Stimulus-aligned: "Time from Stimulus Onset (ms)", range -100 to +400 ms
- Saccade-aligned: "Time from Saccade Onset (ms)", range -300 to +100 ms

---

## 4. Current Results Section (TO BE REVISED)

**Dave's comment**: "DRACO--it would be helpful to update this section to align more closely with the way we are presenting the results in the figures. Explain in sequence the analyses in A, B, and C in the figure. Explaining what AUC(Category) and (Saccade) are examining, and that the different traces correspond to individual sessions. Then explain what is being shown in B, and C (including explaining what Directed Predictability Gain means--that's a potentially confusing term)."

### Current Text:

> Our population decoding analyses so far indicate shorter-latency emergence of abstract category and saccadic-choice signals in FEF relative to LIP, which is consistent with cross-area information flow between these anatomically connected cortical areas.[Dave: add reference for anatomical connections between LIP and FEF] However, pseudopopulations do not preserve trial-to-trial covariation across simultaneously recorded areas. We therefore took advantage of our simultaneous population recording to perform a more stringent test of cross area communication between FEF and LIP by assessing directed flow of information about category and saccade direction between areas at the single trial level (Fig. 5). For each session, we learned a session-wise linear "feature axis" in each area and used the trial-resolved projection onto that axis as a time-varying feature evidence trace (Fig. Xa). [Draco, can we expand on this a bit so that the reader has a better sense of what we did exactly? It is an interesting analysis, and it would be helpful to explain it in some more detail. For example, explain that for each session and brain area, we identified encoding subspace axes related to category and saccade direction. Then we examined each areas' activity trajectories along these axes and the temporal relationships (lead vs lag) between areas.]
>
> In both Monkeys, single-trial evidence emerged significantly earlier in FEF than LIP for both stimulus category (mean FEF leads LIP by: Monkey M: 37.8 ms[we should report statistical significance here too. It's ok to refer to methods for the approach, but explaining what is significant or not here is a good idea], Monkey S: XX.X ms; Fig. X; See Methods) and saccade direction (mean FEF leads LIP by: Monkey M: 57.3 ms, Monkey S: XX.X ms; Fig. X), consistent with the pseudopopulation decoding analyses presented earlier (Figs. 3-4). Using the same projected feature codes[I think this terminology is confusing. please clarify], we next quantified directed coupling with a lagged regression framework and summarized the results with a "net flow" measure that compares the two directions of interaction (i.e. FEF->LIP and LIP->FEF; net flow is positive when the FEF->LIP direction dominates. We found that net flow was significantly positive, indicating that FEF->LIP flow dominates during task-relevant epochs for both the category and saccade factors (Fig. Xc), consistent with a preferential propagation of trial-to-trial feature fluctuations from FEF to LIP at the level of single trials.

---

## 5. Current Figure Legend (TO BE REVISED)

> **Figure X: FEF leads LIP in trial-wise onset and directed flow of category and saccade signals.** Top row: stimulus category (stimulus-aligned; vertical-target trials). Bottom row: saccade direction (saccade-aligned; horizontal-target trials). **a**, Example session showing time-resolved decoding performance (AUC) computed from the projection of population activity in each area (FEF, LIP, SC) onto a session-wise linear feature axis; dashed vertical line denotes the alignment event (0 ms). **b**, Single-trial onset latencies for the same projected "feature evidence" traces, defined as the first time the signed evidence exceeded a baseline threshold (μ + 4σ) for 5 consecutive time bins. Scatter plots compare FEF vs LIP trial-wise latencies (top: N=85 trials, p<0.001; bottom: N=536 trials, p<0.001); only trials with a defined onset in both areas are included. Red circle and blue square denote the mean and median, respectively; dashed line denotes y=x. **c**, Net directed flow between FEF and LIP computed from trial-to-trial fluctuations of the projected feature codes; we summarize directed coupling as a "net flow" measure that compares the two directions of interaction, which is positive when the FEF-to-LIP direction dominates. Flow is computed using all trials from sessions in which both areas showed robust feature subspaces for the corresponding variable (AUC ≥ 0.75): category, 5 sessions; saccade, 8 sessions. Shaded regions indicate SEM across sessions. Black dots denote time bins with significant net flow (p<0.05) under a stratified permutation null.

---

## 6. Current Methods Section

### Trial-resolved feature evidence and cross-area flow

To relate population coding to trial-by-trial interactions between simultaneously recorded areas, we performed a two-stage analysis. First, for each session and area we learned a session-wise linear axis for a task feature and used the projection onto this axis as a time-resolved feature evidence signal. Second, we used these evidence traces to (i) define single-trial onset latencies by a sustained threshold-crossing rule (Fig. Xb) and (ii) quantify directed interactions between areas via lagged ridge regression with permutation-based significance testing (Fig. Xc).

### Spike-count binning, alignment, and within-session normalization

For each session and area, spikes were binned into a tensor aligned to task events. For stimulus-aligned analyses (category; vertical-target trials), we used a bin size of Δt = 10 ms and a window of [−250, 800] ms relative to stimulus onset. For saccade-aligned analyses (saccade direction; horizontal-target trials), we used a bin size of Δt = 5 ms and a window of [−400, 300] ms relative to saccade onset. Time axes refer to bin centers.

Let X ∈ ℝL×T×N denote binned spike counts, where L is the number of trials, T is the number of time bins, and N is the number of simultaneously recorded units. We normalized each unit within session/alignment by z-scoring across all trials and time bins,

Zℓ,t,i = (Xℓ,t,i − μi) / σi,     μi = (1/LT) Σℓ,t Xℓ,t,i,     σi = SDℓ,t(Xℓ,t,i),

yielding Z, which was used for all subsequent axis, onset, and flow analyses, where (ℓ,t,i) ∈ {1, …, L}×{1, …, T}×{1, …, N}. Each trial carried labels for category C ∈ {−1, +1}, motion direction R, saccade direction S ∈ {−1, +1}, target-layout orientation (vertical vs. horizontal), processing time (PT; ms), and correctness. Unless otherwise specified, analyses were restricted to correct trials and imposed PT ≥ 200 ms.

### Session-wise feature axes and evidence traces

For each session and area, we learned a linear feature axis and computed a time-resolved evidence trace by projecting population activity onto that axis. Let zℓ(t) ∈ ℝN denote the population vector from Z for trial ℓ at time bin t. For a feature F ∈ {C, S}, we learned a unit-norm axis aF ∈ ℝN and defined the scalar feature code

yℓ(t) = aF⊤ zℓ(t).

**Axis training windows.** Axes were trained on trial-averaged activity within fixed task windows. For stimulus-aligned categories, we used 0.10–0.30 s after stimulus onset. For saccade-aligned saccade direction, we used −0.10 to −0.03 s relative to saccade onset. To construct a category-orthogonal saccade axis (below), we additionally trained a saccade-aligned category axis using −0.30 to −0.18 s relative to saccade onset.

**Binary axis estimation.** For each feature, we averaged Z within the corresponding training window to obtain a trial-by-unit matrix and fit an L2-regularized logistic regression decoder with class balancing. The inverse regularization strength 𝒞 was selected by 5-fold stratified cross-validation over 𝒞 = {0.1, 0.3, 1, 3, 10}, and the final model was refitted using all included trials. The feature axis was defined as the unit-norm coefficient vector aF = wF / ‖wF‖, with sign chosen such that the resulting projection yields AUC ≥ 0.5.

For saccade-direction axes, we additionally used per-trial sample weights to balance trials across joint (C, S) strata (weights proportional to the inverse stratum count), preventing saccade decoding from being driven by category imbalance.

**Category-invariant saccade axis.** To reduce category contamination in saccade evidence, we constructed an "invariant" saccade axis by orthogonalizing the raw saccade axis to the saccade-aligned category axis:

aS,inv = (aS,raw − (aS,raw · aC,sacc)aC,sacc) / ‖...‖,

and used aS,inv for saccade analyses when available.

### Time-resolved QC and session inclusion

We evaluated axis quality by computing time-resolved decoding performance from a fixed axis. For each time bin t, we computed the projection pℓ(t) = aF⊤zℓ(t) and then computed ROC AUC across trials using {pℓ(t)}ℓ=1L and the corresponding trial labels. For display, we defined a QC latency as the first time at which AUC ≥ 0.75 for k = 5 consecutive bins (Fig. Xa).

For inclusion in the onset and flow analyses, a session was considered to have a usable feature axis if its QC curve reached AUC ≥ 0.75 at any time bin. For pairwise FEF–LIP analyses, we applied symmetric inclusion: a session contributed only if both areas passed the QC criterion for the relevant feature, ensuring identical session sets for both directions in the flow analysis.

### Single-trial feature onset latency

To estimate trial-wise onset latencies (Fig. Xb), we formed a signed evidence trace by multiplying the 1D projection by the trial label sign:

eℓ(t) = yℓ (aF⊤zℓ(t)),     yℓ ∈ {−1, +1}.

Evidence traces were smoothed in time with a Gaussian kernel of σ = 20 ms (converted to bins using Δt). For each trial, we computed baseline statistics from a pre-event baseline window and defined a trial-specific threshold θℓ = μℓ,0 + 4σℓ,0, where (μℓ,0, σℓ,0) are the mean and standard deviation of eℓ(t) within the baseline window. Baseline and search windows were feature/alignment-specific: for category (stimulus-aligned), baseline [−0.20, 0.00] s and search [0.00, 0.50] s; for saccade direction (saccade-aligned), baseline [−0.35, −0.20] s and search [−0.30, 0.20] s.

We defined onset latency as the first time bin within the search window at which eℓ(t) > θℓ for at least 5 consecutive bins. Trials without a sustained crossing were assigned no onset. When comparing FEF vs. LIP, we included only trials with defined onsets in both areas.

To assess whether FEF preceded LIP, we computed paired trial-wise latency differences and assessed significance with a sign-flip permutation test on the mean difference (20,000 permutations).

### Directed flow on projected feature codes

To quantify directed interactions in trial-to-trial fluctuations of the feature evidence (Fig. Xc), we applied a lagged ridge-regression framework to the projected traces. For an ordered pair of areas A → B, let yℓA(t) and yℓB(t) denote the feature projections for trial ℓ at time t in areas A and B, respectively.

**Evoked subtraction.** To remove mean evoked structure, we performed evoked subtraction in the projected space: at each time bin, we subtracted the across-trial mean code. The across-trial mean trace was smoothed over time with a Gaussian filter of σ = 10 ms prior to subtraction. For the analyses reported here, we used evoked subtraction and did not additionally subtract per-condition means.

**Lagged ridge regression and flow metric.** We constructed lagged predictors using a lag window specified in milliseconds and converted it to an integer number of time bins, W = max(1, round(lagsms / (1000 Δt))), using lagsms = 50 ms for stimulus-aligned analyses and lagsms = 30 ms for saccade-aligned analyses.

At each time bin t ≥ W, we compared two ridge regression models predicting the target-area evidence yℓB(t) across trials: (i) a reduced model containing an intercept and the target area's own history {yℓB(t−1), …, yℓB(t−W)} (autoregressive control), and (ii) a full model that additionally included the source-area history {yℓA(t−1), …, yℓA(t−W)}. Our ridge regression used a fixed penalty λ = 10−2, and predictors were used without additional standardization beyond the initial within-session z-scoring of neural activity.

We quantified directed influence in bits as the log-likelihood improvement from adding the source history:

FlowA→B(t) = (L/2) log2(SSEred(t) / SSEfull(t)),

where SSEred(t) and SSEfull(t) are sums of squared prediction errors across trials for the reduced and full models, respectively. We computed flow in both directions (FEF → LIP and LIP → FEF) and summarized directional asymmetry as a net-flow difference (shown in the main text/figure legend).

### Permutation testing and across-session inference for flow

To assess significance while controlling for condition structure and spurious cross-area correlations, we used permutation-based null distributions that preserve within-area temporal structure while breaking trial-to-trial coupling across areas.

**Within-session nulls.** For each session, we generated a stratified trial-shuffle null with 500 permutations. Trial identities were permuted within joint (C, R) strata ("CR-stratified"), breaking the trial-to-trial correspondence between areas while preserving condition structure. For each permutation and time bin, we recomputed flow using the same lagged regression procedure. One-sided p-values were computed as

p(t) = (1 + #{null(t) ≥ obs(t)}) / (1 + #{null valid(t)}).

**Across-session summaries, smoothing, and net-flow significance.** We summarized flow time courses across sessions as mean ± SEM and computed net flow as the difference between the two directions (FEF → LIP minus LIP → FEF). To assess significance of the across-session mean net-flow difference at each time bin, we constructed an empirical group null from per-session permutation samples: for each of 4096 group replicates, we sampled one permutation draw per session, formed the per-session null net-flow difference, and averaged across sessions. To reduce sensitivity of group-level inference to bin-to-bin noise, we applied a uniform moving-average smoothing of width 50 ms to the observed mean net-flow time course and to each group-null replicate before computing group p-values. Group p-values were computed one-sided as the fraction of group-null means exceeding the observed group mean (with a +1 correction), and bins with p < 0.05 were marked in Fig. Xc.

---

## 7. Key Terminology to Explain

### "Directed Predictability Gain" (Panel C/F)
This is the key term that Dave noted as "potentially confusing." The concept is:

1. **Basic idea**: How much does knowing area A's recent activity improve our ability to predict area B's current activity, beyond what we could predict from B's own recent activity?

2. **Intuition**: If FEF→LIP predictability gain is positive and larger than LIP→FEF, it suggests information is flowing preferentially from FEF to LIP

3. **Technical definition**: 
   - Reduced model: Predict B(t) from B's own history {B(t-1), ..., B(t-W)}
   - Full model: Predict B(t) from B's history + A's history {A(t-1), ..., A(t-W)}
   - Flow = improvement in bits from adding A's history
   - Corrected for model complexity (df baseline)

4. **Y-axis interpretation**: 
   - Zero = adding source area provides no predictive benefit beyond chance
   - Positive = source area history improves prediction of target area
   - The correction ensures that adding more predictors doesn't artificially inflate the metric

### "AUC" (Panel A)
- Area Under the ROC Curve
- Measures how well the projection onto the feature axis separates the two classes (e.g., category 1 vs category 2)
- AUC = 0.5 means chance; AUC = 1.0 means perfect discrimination
- Each trace = one recording session

### "Feature axis" / "Encoding axis"
- A direction in neural population space that best separates trials by the relevant feature (category or saccade direction)
- Found by training a linear classifier on trial-averaged activity in a training window
- Activity is then projected onto this axis to get a scalar "feature evidence" value

---

## 8. Required Code Files to Provide

The LLM agent should have access to the following code files to understand the analysis:

### Core Analysis Modules (in `paperflow/`)
1. **`paperflow/axes.py`** - Encoding axis training (how feature axes are learned)
2. **`paperflow/flow.py`** - Information flow computation (the lagged regression framework)
3. **`paperflow/qc.py`** - QC curves and session filtering
4. **`paperflow/norm.py`** - Normalization utilities (z-scoring, baseline correction)

### CLI Scripts (in `cli/`)
5. **`cli/summarize_flow_across_sessions.py`** - Cross-session aggregation and figure generation
6. **`cli/trial_onset_comprehensive.py`** - Single-trial onset latency analysis

### Documentation
7. **`cli/README_summarize_flow.md`** - Detailed documentation of the summarization workflow

---

## 9. Specific Revision Tasks

Based on Dave's feedback, the Results section should be revised to:

1. **Explain Panel A first**: What AUC measures, that each trace is a session, what the time course shows about feature encoding emergence

2. **Explain Panel B**: How single-trial onset latency is defined, what the scatter plot shows, how to interpret the FEF vs LIP comparison

3. **Explain Panel C/F clearly**: 
   - Define "Directed Predictability Gain" in accessible terms
   - Explain that both directions are plotted (FEF→LIP and LIP→FEF)
   - Explain what the asymmetry means biologically
   - Explain the significance markers (black dots)

4. **Update figure legend**: Match the actual figure panels and methodology

5. **Ensure methods section is accurate**: Update any parameters that have changed (e.g., 20-ms sliding window instead of fixed bins)

---

## 10. Summary Statistics to Include

From the analysis, include:
- Number of sessions per monkey and feature (after QC filtering)
- Mean ± SEM onset latency differences (FEF vs LIP)
- p-values for onset latency comparisons
- Time windows with significant net flow

---

## 11. Style Guidelines

- Use active voice where possible
- Define technical terms on first use
- Reference figure panels explicitly when describing results
- Be precise about which trials/sessions are included
- Clarify monkey-specific vs pooled results

---

*Document prepared for manuscript revision. Contact Draco for any clarifications on the analysis pipeline.*
