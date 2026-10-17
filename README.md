<div align="center">

# Multiclass Classification of Rheumatic & Autoimmune Diseases

**A leakage-safe, statistically controlled machine-learning study: every model, decision, and metric verified.**

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f)](LICENSE)
[![Notebooks](https://img.shields.io/badge/Notebooks-13%20executed-F37626?logo=jupyter&logoColor=white)](notebooks/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4-1192d3)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.52-ff0d57)](https://shap.readthedocs.io/)

[At a Glance](#at-a-glance) · [Key Models](#key-models--performance) · [Results & Findings](#what-the-results-mean) · [Negative Results](#what-did-not-work) · [Study Pipeline](#how-the-study-was-run) · [Repository](#repository-structure) · [Reproducibility](#reproduce) · [Author & License](#author--license)

</div>

---

## At a Glance

We evaluate multi-class diagnosis across 7 autoimmune conditions using 14 routine clinical and serological measurements from 12,085 patient records (Harvard Dataverse, DOI [10.7910/DVN/VM4OR3](https://doi.org/10.7910/DVN/VM4OR3)). Every architectural decision was determined strictly within fold-safe 5-fold cross-validation on 9,668 training patients. The 2,417-patient test set was evaluated exactly once.

| Metric | Nested 5-Fold CV (9,668 patients) | Untouched Test Set (2,417 patients) |
|---|:---:|:---:|
| **Macro-F1** | **0.825** [0.817, 0.834] | **0.846** [0.829, 0.862] |
| **Accuracy** | 0.839 | **0.853** |
| **Balanced Accuracy** | 0.816 | **0.836** |
| **ROC-AUC (ovr)** | 0.983 | **0.987** |
| **Expected Calibration Error (ECE)** | 0.015 | **0.019** |

> **Final Deployed Model:** Tuned XGBoost on 14 raw features with no class weighting (`learning_rate=0.020`, `n_estimators=600`, `max_depth=4`).

### Core Takeaways

1. **Model Choice Hits a Ceiling:** Every well-regularized tree ensemble family lands within ~0.003 macro-F1 of one another. The ceiling is governed by clinical overlap and missing values, not classifier complexity.
2. **Missing Tests Drive 36% of Errors:** Each missing diagnostic test increases the odds of error by **$\times 1.32$** (replicated at **$\times 1.29$** on the test set).
3. **Uncertainty Deferral Works:** Rejecting the least-confident 20% of predictions eliminates **~62% of all test errors**, raising retained cohort accuracy to 93.1%.

---

## Key Models & Performance

Scores represent 5-fold cross-validation Macro-F1 (mean ± standard deviation across folds) on the 9,668 training pool.

| Role | Model | Configuration | CV Macro-F1 | Test Macro-F1 | Verdict |
|:---:|---|---|:---:|:---:|---|
| Baseline | **Majority-Class Dummy** | Constant baseline | n/a | 0.0545 | Empirical floor |
| Reference | **Random Forest (Starting Pipeline)** | Balanced weights, 19 features | 0.8175 ± 0.0081 | 0.8344 | Baseline reference point |
| Candidate | **Random Forest (Unweighted)** | No weights, 14 raw features | 0.8230 ± 0.0070 | 0.8430 | +0.009 gain over weighted RF |
| Candidate | **XGBoost (Defaults)** | No weights, 14 raw features | 0.8174 ± 0.0057 | — | Starting point for tuning |
| Final | **XGBoost (Tuned)** | Nested CV random search | **0.8254 ± 0.0078** | **0.8464** | **Selected Final Model** (+0.0079 over default, 5/5 folds) |
| Ensemble | **5-Model Stacking Ensemble** | Logistic regression meta-learner | 0.8260 ± 0.0051 | — | Dropped: +0.0007 over tuned XGB (not significant) |
| Ensemble | **Swarm Optimizers (PSO/GWO/GA)** | 3,000 evaluations per optimizer | 0.8260 – 0.8283 | — | Dropped: equal budget random search scored 0.8284; over-fit inner folds |

<details>
<summary><strong>View full 15-model experimental progression table</strong></summary>

<br>

| Stage | Model / Change | Configuration | CV Macro-F1 | Outcome |
|:---:|---|---|:---:|---|
| 03 | Majority-Class Dummy | Always predicts largest class | n/a (test 0.0545) | Reference baseline |
| 04 | Logistic Regression | Balanced weights, 19 features | 0.7902 ± 0.0111 | Behind tree models by ~0.027 |
| 04 | Random Forest (Starting Pipeline) | Balanced weights, 19 features | 0.8175 ± 0.0081 | Initial benchmark (test 0.8344) |
| 04 | XGBoost (Defaults) | Balanced weights, 19 features | 0.8148 ± 0.0046 | Statistically tied with RF |
| 04 | Gradient Boosting (sklearn) | Balanced weights, 17 features | 0.8161 ± 0.0030 | Tied, but 36x slower to train |
| 05 | Random Forest (Unweighted) | 17 features | 0.8245 ± 0.0098 | Adopted: weighting cost ~0.009 |
| 05 | Random Forest + SMOTE-NC | 17 features | 0.8115 ± 0.0073 | Dropped: ReA precision fell to 0.52 |
| 06 | Random Forest (20 Candidate Features) | No weights | 0.8249 ± 0.0132 | No gain (+0.0004); 14 raw kept |
| 07 | Random Forest (Defaults) | No weights, 14 raw features | 0.8230 ± 0.0070 | Strong alternative (test 0.8430) |
| 07 | Random Forest (Tuned) | Nested random search | 0.8246 ± 0.0053 | Not significant (+0.0017); defaults kept |
| 07 | XGBoost (Defaults) | No weights, 14 raw features | 0.8174 ± 0.0057 | Starting point for tuning |
| 07 | **XGBoost (Tuned - Final Model)** | Nested random search | **0.8254 ± 0.0078** | **Selected**: +0.0079, significant in 5/5 folds (test 0.8464) |
| 08 | KNN / HistGradientBoosting / LogReg | Unweighted base learners | 0.7058 / 0.8125 / 0.7889 | Stacking ensemble inputs only |
| 08 | Stacking Ensemble (5 Models + LR) | Nested out-of-fold | 0.8260 ± 0.0051 | Dropped: +0.0007 over tuned XGB |
| 09 | Swarm Optimisation (PSO / GWO / GA) | 3,000 evaluations each | 0.8260 to 0.8283 | Dropped: no optimizer beat random search |

</details>

---

## What the Results Mean

### 1. Diagnosis-by-Diagnosis Breakdown (Test Set)

| Diagnostic Class | Test Samples | Precision | Recall | Test F1 | 95% CI | Key Biomarkers (SHAP) |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Systemic Lupus Erythematosus (SLE)** | 271 | 1.000 | 0.978 | **0.989** | [0.979, 0.996] | Complement depletion (low C3 & C4), low ESR/CRP |
| **Sjögren's Syndrome** | 370 | 0.858 | 0.949 | **0.901** | [0.879, 0.923] | High Anti-Ro & Anti-La, normal ESR/CRP |
| **Psoriatic Arthritis (PsA)** | 357 | 0.859 | 0.888 | **0.873** | [0.847, 0.898] | High ESR & CRP with low RF/Anti-CCP |
| **Normal** | 321 | 0.923 | 0.826 | **0.872** | [0.843, 0.899] | Normal inflammatory markers, negative autoantibodies |
| **Rheumatoid Arthritis (RA)** | 570 | 0.799 | 0.942 | **0.865** | [0.844, 0.884] | Elevated Anti-CCP & RF, high ESR/CRP |
| **Reactive Arthritis (ReA)** | 103 | 0.861 | 0.660 | **0.747** | [0.671, 0.814] | Intermediate inflammation, positive HLA-B27 |
| **Ankylosing Spondylitis (AS)** | 425 | 0.768 | 0.607 | **0.678** | [0.638, 0.714] | Positive HLA-B27, elevated CRP & ESR |

* **Easy vs. Challenging Diseases:** Lupus and Sjögren's are near-perfectly recognized due to specific pathognomonic serology. In contrast, **Ankylosing Spondylitis is the primary failure mode**: 26.8% of AS cases are misclassified as Rheumatoid Arthritis due to shared inflammatory profiles.

### 2. Missing Diagnostic Tests Hurt Performance

Accuracy drops steadily with missing laboratory assays:

| Missing Tests Count | Training CV Accuracy | Held-Out Test Accuracy |
|:---:|:---:|:---:|
| **0 – 1** | 0.897 | **0.900** |
| **2** | 0.865 | **0.868** |
| **3** | 0.830 | **0.845** |
| **4** | 0.804 | **0.834** |
| **5+** | 0.754 | **0.787** |

* **Single Tests That Cost the Most When Missing:** ESR (+13.2 error points), RF (+12.4 points), Anti-CCP (+6.5 points), and CRP (+3.8 points).

### 3. Confidence Deferral & Selective Classification

The model exhibits strong calibration (ECE = 0.019; correct prediction mean confidence 0.872 vs. 0.620 on errors):

| Target Coverage | Realised Coverage | Retained Cohort Accuracy | Errors Filtered Out |
|:---:|:---:|:---:|:---:|
| **90%** | 90.9% | 0.888 | 31% |
| **80%** | 79.9% | **0.931** | **62%** |
| **70%** | 69.2% | 0.970 | 86% |
| **60%** | 61.2% | 0.993 | 97% |

> [!WARNING]
> **Deferral is Not Class-Neutral:** At 80% coverage, the system retains 100% of Lupus and 78% of RA, but only **61% of Ankylosing Spondylitis** and **68% of Reactive Arthritis** patients.

---

## Visual Interpretability

<p align="center">
  <img src="results/shap_importance_heatmap.png" width="75%" alt="SHAP Importance Heatmap">
  <br><em>Figure 1: Mean absolute SHAP value per biomarker across all 7 diagnostic classes.</em>
</p>

<p align="center">
  <img src="results/final_reliability.png" width="45%" alt="Reliability Diagram">
  <br><em>Figure 2: Test-set calibration curve comparing stated model confidence to observed accuracy (ECE = 0.019).</em>
</p>

<p align="center">
  <img src="results/shap_patients.png" width="80%" alt="Individual Patient Explanations">
  <br><em>Figure 3: Case-level SHAP attributions for a representative correct case and difficult boundary errors.</em>
</p>

---

## What Did NOT Work

Pre-specified negative findings are critical to prevent wasteful modeling iterations:

1. **Class Weighting Degraded Macro-F1:** `balanced` sample weighting decreased macro-F1 (+0.009 improvement without weighting). SMOTE-NC oversampling degraded Reactive Arthritis precision to 0.52.
2. **Engineered & Literature Features Added Zero Value:** Adding non-linear interaction terms, composite inflammatory sums, or missingness flags yielded no statistical gain (all 95% CIs cross zero). The 14 raw features proved superior.
3. **Stacking & Swarm Ensembles Failed to Generalize:** A 5-model stack matched standalone XGBoost (0.8260 vs 0.8254, p = 1.0). Swarm optimizers (PSO, GWO, GA) over-fitted inner validation splits (0.007–0.010 gap) and failed to beat random search.

<p align="center">
  <img src="results/natstack_convergence.png" width="70%" alt="Swarm Optimizer Convergence">
  <br><em>Figure 4: Optimization trajectories of PSO, GWO, and GA across 3,000 evaluations versus random search.</em>
</p>

---

## How the Study Was Run

```mermaid
flowchart LR
    A["Cohort: 12,085 patients"] --> B{"Stratified 80/20 Split"}
    B --> T[("Sealed Test Set: 2,417 patients")]
    B --> D["Training Pool: 9,668 patients"]
    D --> C["Fold-Safe Nested 5-Fold CV<br/>(Imputation, Scaling, Model Decisions)"]
    C --> F["Final Model: Tuned XGBoost (14 features)"]
    F --> E["Single Test Evaluation"]
    T --> E
    E --> S["Interpretability: SHAP Analysis"]
```

| Notebook | Focus Question | Key Empirical Finding |
|---|---|---|
| `01_eda` | What are cohort distributions and missingness? | 7 classes (5.5:1 imbalance); 12 tests with 9%–43% missing |
| `01b_missingness_analysis` | Does missingness indicate disease type? | No; missing patterns occur at fixed rates across classes |
| `02_preproccessing` | How to structure data without leakage? | Split prior to fitting; fit imputer/scaler strictly on train folds |
| `04_cross_validation` | Which model family performs best? | Tree ensembles tie; Logistic Regression is significantly worse |
| `05_imbalance_handling` | Should we apply class weights or SMOTE? | No class weighting achieves highest macro-F1 |
| `06_feature_selection` | Do engineered or literature features help? | No; 14 raw features are non-inferior and optimal |
| `07_hyperparameter_tuning` | Does tuning improve tree ensembles? | XGBoost gains +0.008 (significant); RF remains flat |
| `08_stacking` | Does out-of-fold multi-model stacking help? | No; standalone tuned XGBoost achieves identical performance |
| `09_nature_inspired_stacking` | Can swarm algorithms optimize weights? | No; swarm algorithms overfit inner validation folds |
| `10_final_evaluation` | What is true unseen test performance? | Macro-F1 **0.8464**, Accuracy **0.853**, ECE **0.019** |
| `11_shap_explainability` | What biomarker mechanisms were learned? | Biomarker rules precisely match clinical diagnostic criteria |
| `12_error_analysis` | What drives residual errors? | Missing assays (~36%) and inflammatory class overlap (~64%) |

*Full evidence, statistical test tables, and confidence bounds are logged in [`docs/ANALYSIS_LOG.md`](docs/ANALYSIS_LOG.md).*

---

## Repository Structure

```
├── notebooks/                       # 13 sequentially executable research notebooks
├── src/                             # Reusable core modules
│   ├── preprocessing/cv_pipeline.py # Fold-safe pipeline logic (imputer, scaler)
│   └── evaluation/metrics.py        # Paired bootstrap, McNemar, Holm tests, ECE
├── results/                         # Frozen artifacts, tables, figures, final model
│   ├── final_freeze.json            # Pre-evaluation parameter hash record
│   ├── final_model_xgb.json         # Serialized final production model
│   └── environment.txt              # Frozen execution environment
├── docs/ANALYSIS_LOG.md             # Complete evidence audit log
├── data/raw/README.md               # Dataset download & verification instructions
├── requirements.txt                 # Pinned dependencies
└── LICENSE                          # MIT License
```

---

## Reproduce

Tested with **Python 3.13**.

```bash
# 1. Clone repository & configure virtual environment
git clone https://github.com/prathamkariya/Autoimmune-Disease.git
cd Autoimmune-Disease
python -m venv .venv

# Activate: Windows PowerShell
.venv\Scripts\Activate.ps1
# Activate: Linux/macOS
# source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

1. Download `dataset.xlsx` from Harvard Dataverse into `data/raw/` and verify its SHA-256 hash using [`data/raw/README.md`](data/raw/README.md).
2. Execute notebooks sequentially (`01` through `12`). Notebook `02` constructs `data/processed/`.

<details>
<summary><strong>Execution Runtime Estimates</strong></summary>

* **Notebook 07 (Tuning):** ~60+ minutes (nested random search across trees).
* **Notebooks 04, 05, and 08:** ~10 to 25 minutes each.
* **Notebook 09 (Swarm Stacking):** ~10 to 15 minutes initial run (caches predictions to `results/natstack_Z.npz`).
* **Remaining Notebooks:** Fast execution (< 5 minutes each).

</details>

---

## Limitations & Disclosures

1. **Simulated Benchmark Data:** Cohort missingness counts are exact integer percentages of total rows, and several features display artificial hard caps. Findings reflect benchmark properties, not clinical efficacy.
2. **Prior Exploratory Test Exposure:** In exploratory iterations prior to this clean repository, baseline models were run on the test set. Final choices depended solely on cross-validation within the training pool, but this history is disclosed for transparency.
3. **Pre-Specified in Notebooks:** Decision rules and parameter freeze records (`results/final_freeze.json`) were committed in the notebooks prior to test evaluation.
4. **Mild CV Optimism:** The test macro-F1 (0.8464) exceeded the CV interval [0.8168, 0.8336] because the single test partition proved slightly easier for all architectures (all candidate baselines shifted up by ~+0.02) and the final model trained on 25% more data.
5. **Modest Final Delta:** The observed +0.012 Macro-F1 improvement over the starting pipeline is consistent but modest (McNemar $p = 0.058$).

---

## Author & License

<p align="left">
  <a href="https://github.com/prathamkariya">
    <img src="https://github.com/prathamkariya.png" width="90" height="90" alt="Pratham Kariya" style="border-radius: 50%; box-shadow: 0 4px 10px rgba(0,0,0,0.15);" />
  </a>
</p>

**Pratham Kariya**  
* GitHub: [@prathamkariya](https://github.com/prathamkariya)
* Repository: [Autoimmune-Disease](https://github.com/prathamkariya/Autoimmune-Disease)

### License & Terms
* **Code:** Licensed under the [MIT License](LICENSE).
* **Dataset:** Harvard Dataverse (DOI [10.7910/DVN/VM4OR3](https://doi.org/10.7910/DVN/VM4OR3)). Data is not redistributed in this repository.
