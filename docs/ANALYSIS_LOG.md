# Analysis log

Decisions, evidence and findings for every notebook, in the order they were run. Every number below is stored in
`results/` and printed in the corresponding notebook. Rules and expectations were written in the notebooks before the
results existed (see *Disclosures* for what that does and does not prove).

## Final model

**XGBoost**, 14 raw measurements, no class weighting, hyper-parameters tuned by nested cross-validation
(`results/tuning_final_params.json`: learning rate 0.0202, 600 trees, max depth 4, min_child_weight 5, gamma 1,
colsample_bytree 0.57, subsample 0.81, reg_lambda 0.96, reg_alpha 0.0014), trained on all 9,668 training patients.
The fitted model is `results/final_model_xgb.json`.

| | Macro-F1 (95% CI) |
|---|---|
| Nested 5-fold CV (9,668 patients) | 0.8254 [0.8168, 0.8336] |
| Test set (2,417 patients, evaluated once) | **0.8464 [0.8294, 0.8620]** |

## Notebook sequence

| # | Notebook | Purpose | Result |
|---|---|---|---|
| 1 | `01_eda` | Exploratory analysis of the whole dataset (descriptive only) | Seven classes, ~5.5:1 imbalance, 12 tests with gaps |
| 1b | `01b_missingness_analysis` | Does missingness carry information about disease? All 12 columns, Holm-corrected | No: RF p = 0.0055 becomes 0.066; no column significant; gaps look inserted at fixed rates |
| 2 | `02_preproccessing` | Leakage-safe preprocessing (split first; MICE and scaler fit on training only) | Verified identical to `src/preprocessing/cv_pipeline.py` |
| 3 | `03_stage1_plain_models` | Exploratory baselines on a validation split (test set sealed) | Tree models ≈ 0.80–0.81 on one split; nothing selected |
| 4 | `04_cross_validation` | Fold-safe 5-fold CV, calibration, paired tests; flags ablation | Models within 0.003; flags rejected; Logistic Regression clearly worse |
| 5 | `05_imbalance_handling` | none / balanced / sqrt / SMOTE-NC / class multipliers | No class weighting wins; SMOTE-NC never helps |
| 6 | `06_feature_selection` | Fold-safe permutation importance over 20 candidate features | Keep the 14 raw features |
| 7 | `07_hyperparameter_tuning` | Nested-CV random search | XGBoost tuned (+0.008); Random Forest keeps defaults |
| 8 | `08_stacking` | Nested out-of-fold stacking of five models | Not adopted; XGBoost (tuned) alone |
| 9 | `09_nature_inspired_stacking` | PSO / GWO / GA vs random search optimising ensemble weights for macro-F1 | No gain; optimisers over-fit |
| 10 | `10_final_evaluation` | Single evaluation on the untouched test set, deferral, replication | Macro-F1 0.8464 |
| 11 | `11_shap_explainability` | SHAP on the saved final model | Clinically plausible; three importance methods agree |
| 12 | `12_error_analysis` | Missing tests vs class overlap; confidence-based deferral | Missing tests explain ~36% of errors; confidence flags errors |

Run order: 01, 01b, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12 (12 may be run after 07).

## Decisions

| Decision | Evidence |
|---|---|
| Drop the two missingness flags | Notebooks 01b and 04: never helps any model |
| Evaluation numbers come from CV, never from a single split | Single-split rankings can differ from CV rankings |
| **No class weighting** | Notebook 05: passes the rule in 3 of 4 model × feature-set combinations (RF raw p_holm 0.002, RF ext 0.0002, XGBoost ext 0.020; XGBoost raw not significant); +0.003 to +0.009 macro-F1; also the best-calibrated option |
| SMOTE-NC rejected | Passes 0 of 4; lowers Reactive Arthritis precision for Random Forest (0.52) |
| Class weights / multipliers are an *alternative objective* only | They raise balanced accuracy by ~0.01 while lowering macro-F1; a paper must state which objective it optimises |
| **Feature set: 14 raw features** | Notebook 06: literature features +0.0004 / +0.0003 and engineered features +0.0015 / 0.0000 (all intervals include 0); no top-k subset non-inferior (top-12 −0.006, top-10 −0.011, top-8 −0.03; top-14 inconclusive for RF, −0.007 for XGBoost) |
| Literature features rejected | Caveat: our `Seronegative` uses conventional cut-offs and has 28% prevalence, not the paper's 3.2%, so this does not refute their subgroup finding |
| **Random Forest keeps defaults; XGBoost uses tuned settings** | Notebook 07: RF +0.0017, CI [−0.0021, 0.0055], 3/5 folds; XGBoost +0.0079, CI [0.0025, 0.0132], 5/5 folds, McNemar Holm p = 0.0011 |
| **Final model: XGBoost (tuned) alone** | Notebook 08: stack +0.0007, CI [−0.0032, 0.0045], 3/5 folds, Holm p = 1.0; XGBoost has the best calibration of any accurate model (Brier 0.2098, ECE 0.0153 vs stack 0.2156 / 0.0382). XGBoost vs Random Forest defaults is a tie (+0.0024, CI [−0.0025, 0.0071]); XGBoost was the pre-specified reference |
| No nature-inspired ensemble | Notebook 09: no candidate beats XGBoost (tuned); no optimiser beats random search at equal budget |

## Findings

### Notebook 04 — cross-validation
* Random Forest with 19 features: 0.8175, 95% CI [0.809, 0.826]. The six best configurations are within 0.003; Random Forest, XGBoost and Gradient Boosting are statistically indistinguishable (Holm-corrected McNemar p = 0.25 / 0.43 / 1.0); Logistic Regression is 0.025–0.027 lower.
* Full tables: `results/cv5_summary.csv`, `cv5_pairwise_tests.csv`.

### Notebook 05 — imbalance handling
* Best configuration: Random Forest, no class weighting, 17 features, 0.8245 ± 0.0098, CI [0.8154, 0.8329]; raw features 0.8230. XGBoost's best is also unweighted (0.8175). Balanced weighting was the worst Random Forest option (0.8145–0.8156).
* Imbalance handling is not the bottleneck: the best strategy moves macro-F1 by under 0.01 and the learned RF class multipliers stay near 1 (0.84–1.13).
* Ankylosing Spondylitis is not a minority class (17.6% of patients); its F1 stays at 0.63–0.67 under every strategy, so its errors come from overlap with RA / PsA / ReA. Only Reactive Arthritis (4.3%) responds to weighting.
* Paired comparisons resolve differences of roughly 0.004 or more; the unpaired interval on any single score is about ±0.008.

### Notebook 06 — feature selection
* ESR dominates both models (permutation importance 0.19 XGBoost, 0.05 RF), followed by HLA-B27, Anti-CCP, RF, C3, Anti-Ro, Anti-La, C4; these eight were in the top 10 in 5/5 folds for both models.
* Age, Gender, Anti-dsDNA and Anti-Sm have importance ≈ 0 (0/5 folds in the top 10). Dropping them was not tested by the pre-specified procedure, so they stay; calling them redundant is exploratory.
* Single-feature importance is not the value of a set: the top-8 set loses 0.03 macro-F1.
* ESR separates the inflammatory group (RA, AS, PsA, ReA) from the non-inflammatory group (Normal, Sjögren's, SLE); the confusions are almost all *within* these two groups.

### Notebook 07 — tuning
* Nested-CV macro-F1: XGBoost tuned 0.8254 [0.8168, 0.8336]; RF tuned 0.8246 (not adopted); RF default 0.8230; XGBoost default 0.8174. All strong configurations are within 0.0024: the practical ceiling of this data and feature set is about 0.82–0.825.
* XGBoost's default ranked 14–19 of 21 candidates in every fold; the winners chose a low learning rate (0.02–0.03) with many trees and stronger regularisation. RF's default ranks mid-pack (8–14 of 21) and the folds disagree on its best settings: a flat landscape.
* Selection optimism is small (RF 0.0028, XGBoost 0.0015), so most of XGBoost's gain is real.
* The XGBoost winner sits at the edge of the search space (learning rate 0.0202 vs lower bound 0.02; 600 trees = upper bound); a wider range might gain slightly more and was not pursued.
* Calibration improved markedly with tuning (XGBoost ECE 0.0575 → 0.0153, Brier 0.2308 → 0.2098; RF ECE 0.0370 → 0.0176, Brier 0.2196 → 0.2124). This was not statistically tested.

### Notebook 08 — stacking
* Nested-CV macro-F1: stack 0.8260, average of all five 0.8259, XGBoost (tuned) 0.8254, RF 0.8230, average of the three trees 0.8227, HistGradientBoosting 0.8125, Logistic Regression 0.7889, KNN 0.7058 (Reactive Arthritis F1 only 0.08).
* The tree ensembles disagree on only 4.8–7.8% of patients. The oracle accuracy ("any of the five is right") is 0.916 against 0.839 for the best single model, but no combiner recovers any of that gap: the remaining errors are not predictable from the models' outputs.
* The meta-learner spreads weight almost evenly (XGBoost 0.25, RF 0.21, LR 0.19, KNN 0.175, HGB 0.175) with strong regularisation (C = 0.1 in 4/5 folds): it learned roughly a plain average.

### Notebook 09 — nature-inspired stacking
* Candidates vs XGBoost (tuned), Bonferroni over six: PSO_W +0.0007, GWO_W +0.0029 (95% CI [−0.0008, 0.0066], 4/5 folds), GA_W +0.0026, PSO_WM −0.0006, GWO_WM −0.0006, GA_WM −0.0048 (95% CI [−0.0091, −0.0003], 0/5 folds). None qualifies.
* Optimisers vs random search at equal budget (3,000 evaluations): differences between −0.0025 and +0.0016, all intervals include 0; random search over the five weights (0.8284) is nominally the highest macro-F1 of all configurations and indistinguishable from the swarm methods.
* Optimisation–generalisation gap: in the 12-parameter space the optimisers reach a higher inner score than random search (0.830–0.832 vs 0.826) but a gap to the outer score of 0.007–0.010 (random search 0.003).
* The optimisers rediscover "Random Forest + XGBoost (+ some Logistic Regression)" (mean weights RF 0.32–0.42, XGBoost 0.25–0.49, LR 0.14–0.28, KNN ≈ 0). Class multipliers differ widely between optimisers at the same score (flat, degenerate landscape); only Reactive Arthritis is consistently boosted.
* Every ensemble is worse calibrated than XGBoost (tuned) (ECE 0.025–0.038 vs 0.0153); selective accuracy at 80% coverage is the same for all (0.905–0.914).

### Notebook 10 — test set (evaluated once)
* **Macro-F1 0.8464 [0.8294, 0.8620]; accuracy 0.8527 [0.8382, 0.8664]; balanced accuracy 0.8356; ROC-AUC (one-vs-rest, macro) 0.9868; log-loss 0.3108; Brier 0.1884; ECE 0.0192.**
* The result exceeded the declared range (0.808–0.842) and the "suspicious" bound (0.845), so a leakage check was carried out: train and test IDs are disjoint; imputer and scaler were fit on training data only; every model moved up by a similar amount from CV to test (starting-point Random Forest 0.8175 → 0.8344, default Random Forest 0.8230 → 0.8430, final model 0.8254 → 0.8464), including the starting pipeline, which predates every tuning decision. The final model also trains on 25% more patients than each CV fold model, which the declared expectation ignored. Conclusion: this split is easier than the average CV fold; no evidence of leakage.
* Per class (F1): SLE 0.989, Sjögren's 0.901, Psoriatic Arthritis 0.873, Normal 0.872, Rheumatoid Arthritis 0.865, Reactive Arthritis 0.747 [0.671, 0.814] (103 patients), Ankylosing Spondylitis 0.678 [0.638, 0.714]. Largest confusions: AS → RA 26.8%, ReA → AS 22.3%, Normal → Sjögren's 16.2%, AS → PsA 11.3%.
* Final model vs starting point (Random Forest, balanced weights, 19 features): +0.0121, CI [−0.0013, +0.0259], McNemar p = 0.058 (CV: +0.0079): consistent but not individually significant. Final vs default Random Forest: +0.0034 (a tie). The balanced-weight pipeline has higher balanced accuracy (0.8472 vs 0.8356).
* **Deferral transfers:** realised coverage at the frozen thresholds was 90.9 / 79.9 / 69.2 / 61.2 / 55.9% for targets 90 / 80 / 70 / 60 / 50%; accuracy on kept patients 0.888 / 0.931 / 0.970 / 0.993 / 0.998 (above the out-of-fold expectation); share of errors deferred 31 / 62 / 86 / 97 / 99%.
* **Deferral is not class-neutral:** at 80% overall coverage 61% of Ankylosing Spondylitis and 68% of Reactive Arthritis patients are kept (Rheumatoid Arthritis 78%, SLE 100%); at 60% coverage only 25% of Reactive Arthritis and 33% of Ankylosing Spondylitis patients are kept. Use 90% or 80% as operating points and always report coverage per disease.
* **The missing-tests effect replicates:** accuracy 0.900 (≤ 1 missing) → 0.869 → 0.845 → 0.834 → 0.787 (5+); odds ratio per extra missing test 1.286 [1.189, 1.405] (training: 1.315 [1.261, 1.374]).

### Notebook 11 — SHAP
* Overall mean |SHAP|: ESR 1.05, CRP 0.83, RF 0.54, Anti-CCP 0.54, HLA-B27 0.34, C3 0.30, C4 0.26, Anti-Ro 0.18, Anti-La 0.18, ANA 0.17, Anti-Sm 0.10, Anti-dsDNA 0.07, Age 0.05, Gender 0.01. The ranking is identical on a training sample (Spearman 1.000).
* Per disease the model uses what clinical knowledge predicts: high anti-CCP and RF raise Rheumatoid Arthritis; high HLA-B27 (with CRP and ESR) raises Ankylosing Spondylitis; high anti-La and anti-Ro raise Sjögren's; C3 and C4 (1.44 and 1.39) dominate lupus; low ESR, CRP, RF and anti-CCP raise Normal; Psoriatic Arthritis rises with ESR and CRP and falls with RF and anti-CCP; Reactive Arthritis is identified by intermediate inflammation plus HLA-B27. This suggests the simulation followed clinical rules.
* Triangulation (12 tests): SHAP vs permutation importance ρ = 0.90, SHAP vs cost of a missing test 0.87, permutation vs cost 0.78. CRP is the main disagreement (second in SHAP, weak in the other two) because ESR carries the same information and credit is shared among correlated measurements.
* Imputed vs measured (ratio of mean |SHAP| for the predicted class): C4 0.17, C3 0.22, HLA-B27 0.29, ANA 0.62, Anti-Ro 0.67, Anti-La 0.70, Anti-CCP 0.85, CRP 0.85, RF 0.94, Anti-Sm 0.99, ESR 1.06, Anti-dsDNA 1.22. The model largely ignores imputed C3, C4 and HLA-B27 but uses imputed ESR, RF and CRP almost as strongly as measured ones. The link with the cost of a missing test did not appear (ρ = −0.06, p = 0.86).

### Notebook 12 — error analysis
* Accuracy falls monotonically with the number of missing tests (0–1: 0.897; 2: 0.865; 3: 0.830; 4: 0.804; 5+: 0.754); adjusted odds ratio of an error per extra missing test 1.315, 95% CI [1.261, 1.374]. Verdict by the pre-specified rule: missing data is a main cause of the ceiling.
* Overall error 16.1% vs 10.3% with ≤ 1 missing test: about 36% of errors are attributable to missing tests and about 64% persist even with nearly complete data (class overlap is real).
* Cost of a missing test (error rate when missing minus when measured): ESR +13.2 points, RF +12.4, Anti-CCP +6.5, CRP +3.8, C3 / C4 / Anti-Ro / Anti-La / ANA / HLA-B27 about +2 to +3, Anti-dsDNA and Anti-Sm about 0.
* Confused pairs worsen when their key tests are missing: AS vs RA 10.9% → 25.2%; Normal vs Sjögren's 9.3% → 19.4%; AS vs PsA 8.4% → 18.6%; AS vs ReA 4.6% → 12.5% (odds ratio ≈ 1.7, Holm p ≤ 0.004). With all key tests measured, 4.6–10.9% remain confused.
* The model knows when it is unsure: mean confidence 0.872 when right vs 0.620 when wrong. Deferring the least-confident 20% gives accuracy 0.911 / macro-F1 0.893 and removes 56% of errors (70% coverage: 0.948 / 0.930, 78% removed; 50% coverage: 0.995).

## Disclosures

1. **Test-set exposure.** In an earlier version of this analysis, exploratory notebooks compared eight configurations, scored a SHAP model and chose a stacking meta-learner on the test set. Nothing in the final analysis depends on those looks (every final choice was made by cross-validation on the training pool) and those notebooks are not part of this repository, but a paper should disclose that the test set was seen in that exploratory phase.
2. **Dataset provenance.** Every column's missing count is an exact whole percentage of 12,085 rows (probability ≈ 10⁻²⁵ by chance) and several measurements have hard caps; the data is very likely simulated, and SHAP shows the model follows textbook clinical patterns. Results describe this data, not clinical performance.
3. **Pre-specification.** Decision rules, the frozen specification and the expectation were written in the notebooks before each result existed, but the repository history is a single commit and `results/final_freeze.json` was generated by notebook 10 about 23 seconds before the test predictions (13:06:01 and 13:06:24 on 2026-10-04). Describe the rules as *pre-specified in the analysis notebooks*, not as externally pre-registered.
4. **Optimism of CV.** Weighting, features and model family were chosen on the same folds as the CV estimate, so it is mildly optimistic; the final model also trains on 25% more patients than each fold model.
5. **Exploratory data analysis** (notebook 01) used all 12,085 patients before the split; it was descriptive only, and the one idea it motivated (the missingness flags) was tested in cross-validation and dropped.
6. **Narrow evidence on the improvement.** The gain over the starting pipeline is modest and not individually significant on the test set.

## Methodological rules

1. Anything that learns from data (imputation, scaling, resampling, feature selection, tuning, class multipliers) happens inside the training part of a fold.
2. Model choices are made from cross-validation with uncertainty, using a decision rule written before the results are seen.
3. The test set is read exactly once, in notebook 10.
4. A difference smaller than the bootstrap interval is not a result.
