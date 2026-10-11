# Multiclass classification of rheumatic and autoimmune diseases

A leakage-safe, statistically controlled machine-learning study on the Harvard Dataverse *Rheumatic and autoimmune diseases dataset* (DOI [10.7910/DVN/VM4OR3](https://doi.org/10.7910/DVN/VM4OR3); 12,085 patients, 14 clinical and serology measurements, 7 diagnoses). Every modelling decision was made by cross-validation on the 9,668-patient training pool, using a rule written in the notebook before its result existed; the 2,417-patient test set was evaluated once (notebook 10).

> **Important:** the dataset shows strong signs of being simulated (every column's missing count is an exact whole percentage of the rows; several measurements have hard caps; see `notebooks/01b_missingness_analysis.ipynb`). Results describe performance on this data, **not** clinical performance.

## Headline results

|  | Macro-F1 (95% CI) |
| --- | --- |
| Final model, nested 5-fold CV (9,668 patients) | **0.8254** \[0.8168, 0.8336\] |
| Final model, **untouched test set** (2,417 patients) | **0.8464** \[0.8294, 0.8620\] |
| Starting pipeline (Random Forest, class-balanced, 19 features), test set | 0.8344 |
| Majority-class dummy, test set | 0.0545 |

Test set, final model: accuracy 0.853, balanced accuracy 0.836, ROC-AUC (one-vs-rest) 0.987, expected calibration error 0.019. Final model: **XGBoost, 14 raw measurements, no class weighting, hyper-parameters tuned by nested CV**(`results/tuning_final_params.json`; the fitted model is `results/final_model_xgb.json`).

## What the study found

1. **Class weighting lowered macro-F1** (Random Forest +0.009 without it); SMOTE-NC never helped (notebook 05).
2. **Feature engineering did not help**: three engineered and three literature-inspired features and the two missingness flags gave no significant gain, and no smaller feature subset was non-inferior (notebooks 01b, 04, 06).
3. **Tuning helped XGBoost (+0.008, significant) but not Random Forest** (notebook 07). All well-configured tree ensembles end up within about 0.003 of each other.
4. **Stacking and nature-inspired ensemble optimisation (PSO, GWO, GA) gave no significant gain**; no optimiser beat random search at equal budget, and the stronger 12-parameter optimisers over-fitted (notebooks 08, 09).
5. **Missing tests explain about a third of the errors**: accuracy falls from 0.90 (≤ 1 missing test) to 0.75 (5 or more); adjusted odds ratio 1.32 per missing test, replicated on the test set (1.29) (notebooks 12, 10).
6. **The model knows when it is unsure**: deferring the least-confident 20% of patients removes about 62% of errors on the test set. Deferral is **not** class-neutral: at 80% overall coverage it keeps only \~61% of Ankylosing Spondylitis and \~68% of Reactive Arthritis patients (notebook 10).
7. **SHAP explanations match clinical expectations** (RF and anti-CCP for Rheumatoid Arthritis, HLA-B27 for Ankylosing Spondylitis, anti-Ro/La for Sjögren's, complement C3/C4 for lupus) and agree with two independent importance measures (Spearman 0.78–0.90) (notebook 11).

The hardest diseases remain Ankylosing Spondylitis (test F1 0.68) and Reactive Arthritis (0.75; 103 test patients). Full evidence, decisions and per-notebook findings: `docs/ANALYSIS_LOG.md`.

## Repository layout

```
notebooks/              analysis, run in this order; each states its purpose, checks and decision rule
  01_eda                      exploratory analysis of the whole dataset (descriptive only)
  01b_missingness_analysis    missingness vs disease, all 12 columns, multiple-testing corrected
  02_preproccessing           leakage-safe preprocessing; cross-checked against src/
  03_stage1_plain_models      exploratory baselines on a validation split (test set sealed)
  04_cross_validation         fold-safe 5-fold CV, calibration, paired statistical tests
  05_imbalance_handling       none / balanced / sqrt / SMOTE-NC / class multipliers
  06_feature_selection        fold-safe permutation importance, superiority and non-inferiority rules
  07_hyperparameter_tuning    nested CV random search
  08_stacking                 nested out-of-fold stacking
  09_nature_inspired_stacking PSO / GWO / GA vs random search for ensemble weights
  10_final_evaluation         the single test-set evaluation (+ deferral, replication)
  11_shap_explainability      SHAP on the saved final model
  12_error_analysis           missing tests vs class overlap; confidence-based deferral
src/preprocessing/cv_pipeline.py   preprocessing fitted inside each CV fold (imputer, scaler, labels)
src/evaluation/metrics.py          shared metrics, paired bootstrap, McNemar, Holm
results/                tables, figures, out-of-fold predictions, final model
docs/ANALYSIS_LOG.md    decisions, evidence and findings for every notebook; disclosures
data/raw/README.md      how to obtain and verify the dataset
```

Result file prefixes: `missingness_` (01b) · `stage1_val_` (03) · `cv5_` (04) · `imb_` (05) · `fs_` (06) · `tune_` / `tuning_` (07) · `stack_` (08) · `natstack_` (09) · `final_` (10) · `shap_` (11) · `err_` (12).

## Reproducing

```bash
pip install -r requirements.txt      # pinned versions
# 1. download dataset.xlsx into data/raw/ (see data/raw/README.md) and verify its SHA-256
# 2. run the notebooks in numeric order (02 creates data/processed/, which later notebooks read)
```

Notebook 07 is by far the slowest (about an hour or more on a laptop); 04, 05 and 08 take roughly 10–25 minutes each; the others are fast. Notebook 09 caches the base-model predictions in `results/natstack_Z.npz` (regenerated if absent, about 10–15 minutes). Every notebook that depends on an earlier one checks that it reproduces that notebook's numbers and prints a `WARNING` if not. Random seeds are fixed; reruns should reproduce the stored results. `results/environment.txt` lists the packages present when notebook 04 was run (SHAP was installed afterwards); `requirements.txt` is authoritative.

## Method principles

1. Anything that learns from data (imputation, scaling, resampling, feature selection, tuning, class multipliers) happens inside the training part of a fold.
2. Choices are made from cross-validation with uncertainty (95% paired bootstrap; McNemar with Holm correction), using a rule written in the notebook before its results existed.
3. The test set is read once (notebook 10).
4. A difference smaller than the bootstrap interval is not a result.

## Limitations

- Simulated-looking data; no external validation; one train/test split.
- **Test-set exposure:** in an earlier version of this analysis, exploratory notebooks compared models on the test set. No final choice depended on those looks, and those notebooks are not part of this repository, but it must be disclosed.
- The final-evaluation specification was written in the notebooks and the frozen record `results/final_freeze.json`was generated by notebook 10 about 23 seconds before the test predictions; the repository history is a single commit, so there is no independent timestamp. The rules are *pre-specified in the notebooks*, not externally pre-registered.
- The CV estimate is mildly optimistic (weighting, features and model family were chosen on the same folds); the final model also trains on 25% more patients than each CV fold model. The test score (0.846) exceeded the declared expectation (0.808–0.842); every model, including the starting pipeline, scored about 0.02 higher on the test split than in CV, which points to the split, not to leakage.
- The improvement over the starting pipeline is modest and not individually significant on the test set (+0.012, CI \[−0.001, +0.026\], McNemar p = 0.058).
- The exploratory analysis (notebook 01) looked at all 12,085 patients before the split; it was descriptive only.
- `Seronegative` uses conventional cut-offs because the dataset's own cut-offs are undocumented.

## Credits

## Data and licence

Code: MIT (see `LICENSE`; replace the placeholder author line). Data: Harvard Dataverse, DOI 10.7910/DVN/VM4OR3 — see the dataset page for its terms; the data file is not redistributed here.