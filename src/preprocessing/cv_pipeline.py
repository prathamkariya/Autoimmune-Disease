"""
src/preprocessing/cv_pipeline.py

Shared fold-safe preprocessing for cross-validation, used by notebooks 02-12.
Fits MICE, scaling, and label encoding fresh inside every fold to avoid
leakage between train/validation splits within CV.

Binary markers (BINARY_COLS) are clipped to [0,1] after MICE but NOT
rounded to a hard 0/1. Rounding was tested and found to destroy real
information -- for markers with a genuinely ambiguous ~50% imputed
probability, rounding manufactured fabricated associations with no real
clinical signal (e.g. ANA in PsA: true imputed rate ~46.8%, rounded to
18.6%). Keeping the continuous [0,1] estimate improved Random Forest
Macro-F1 by +0.0064, with the largest gains on AS, Normal, and Sjogren's
recall. See 02_preprocessing.ipynb for the investigation.
"""

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

BINARY_COLS = ['HLA-B27', 'ANA', 'Anti-Ro', 'Anti-La', 'Anti-dsDNA', 'Anti-Sm']
MICE_COLS = ['ESR', 'CRP', 'RF', 'Anti-CCP', 'HLA-B27', 'ANA',
             'Anti-Ro', 'Anti-La', 'Anti-dsDNA', 'Anti-Sm', 'C3', 'C4']
CONTINUOUS_COLS = ['ESR', 'CRP', 'RF', 'Anti-CCP', 'C3', 'C4']

RAW_FEATURES = ['Age', 'Gender', 'ESR', 'CRP', 'RF', 'Anti-CCP', 'HLA-B27',
                'ANA', 'Anti-Ro', 'Anti-La', 'Anti-dsDNA', 'Anti-Sm', 'C3', 'C4']
EXT_FEATURES = RAW_FEATURES + ['RF_was_missing', 'Anti-CCP_was_missing',
                                'Inflammation_Score', 'C3_C4_Ratio', 'Autoantibody_Count']

# Extended set WITHOUT the two missingness flags (17 features). Used for the
# flag ablation: notebook 01b found no demonstrated association between
# missingness and disease after multiple-testing correction.
FLAG_COLS = ['RF_was_missing', 'Anti-CCP_was_missing']
EXT_NO_FLAGS = [c for c in EXT_FEATURES if c not in FLAG_COLS]

# Literature-inspired candidate features (used by notebook 06 only; OFF by default).
# Seronegative uses CONVENTIONAL lab cut-offs
# (RF < 14, anti-CCP < 20) because the dataset's own cut-offs are undocumented;
# it will therefore not reproduce the 3.2% subgroup of Elmesiry et al. 2026.
LIT_FEATURES = ['Seronegative', 'ESR_CRP_Product', 'RF_AntiCCP_Product']
SERO_RF_CUTOFF, SERO_CCP_CUTOFF = 14.0, 20.0


def preprocess_fold(fold_train_df: pd.DataFrame, fold_val_df: pd.DataFrame,
                    literature_features: bool = False):
    """
    Fits every learned preprocessing step (MICE, scaler, label encoder) on
    fold_train ONLY, then applies transform-only to fold_val. Nothing here
    ever calls .fit() on fold_val -- that's the entire point of doing this
    per-fold instead of reusing globally-fit versions.

    Returns X_raw_tr, X_raw_va, X_ext_tr, X_ext_va, y_tr, y_va.
    If literature_features=True, the 3 LIT_FEATURES are appended as extra
    columns at the END of X_ext_* (computed from the imputed, unscaled values;
    the two products are then scaled with the same fold-fit scaler). With the
    default False the output has the 14 raw and 19 extended columns of notebook 02.
    """
    ft, fv = fold_train_df.copy(), fold_val_df.copy()

    # MICE -- fit on fold train, transform both.
    # max_iter=50: max_iter=15 was tested and does not converge (true
    # convergence is ~iteration 17); 15 produced unstable, non-reproducible
    # downstream results.
    imputer = IterativeImputer(random_state=42, max_iter=50)
    ft[MICE_COLS] = imputer.fit_transform(ft[MICE_COLS])
    fv[MICE_COLS] = imputer.transform(fv[MICE_COLS])

    # Binary markers: clip to valid [0,1] range but do NOT round to hard
    # 0/1 -- see module docstring for why.
    for col in BINARY_COLS:
        ft[col] = ft[col].clip(0, 1)
        fv[col] = fv[col].clip(0, 1)
    for col in CONTINUOUS_COLS:
        ft[col] = ft[col].clip(lower=0)
        fv[col] = fv[col].clip(lower=0)

    for d in (ft, fv):
        # Inflammation_Score = ESR + CRP: an ad-hoc composite, NOT a
        # validated clinical score -- refer to it as such in the paper.
        d['Inflammation_Score'] = d['ESR'] + d['CRP']

        # C3_C4_Ratio: exploratory, hypothesis-driven (complement
        # consumption differs in SLE). Guard against div-by-zero with
        # np.nan (NOT pd.NA: replace(0, pd.NA) silently turns the whole
        # column into dtype=object, which is fragile downstream).
        d['C3_C4_Ratio'] = d['C3'] / d['C4'].replace(0, np.nan)

        # Autoantibody_Count: sum of the 6 binary markers' continuous
        # [0,1] estimates (not a count of hard positives, since binary
        # markers are no longer rounded). Exploratory aggregate feature,
        # not a validated diagnostic index.
        d['Autoantibody_Count'] = d[BINARY_COLS].sum(axis=1)

        if literature_features:
            d['Seronegative'] = ((d['RF'] < SERO_RF_CUTOFF) & (d['Anti-CCP'] < SERO_CCP_CUTOFF)).astype(int)
            d['ESR_CRP_Product'] = d['ESR'] * d['CRP']
            d['RF_AntiCCP_Product'] = d['RF'] * d['Anti-CCP']

    le = LabelEncoder()
    ft['Disease_encoded'] = le.fit_transform(ft['Disease'])
    fv['Disease_encoded'] = le.transform(fv['Disease'])

    X_raw_tr, X_raw_va = ft[RAW_FEATURES].copy(), fv[RAW_FEATURES].copy()
    ext_cols = EXT_FEATURES + (LIT_FEATURES if literature_features else [])
    X_ext_tr, X_ext_va = ft[ext_cols].copy(), fv[ext_cols].copy()
    y_tr, y_va = ft['Disease_encoded'], fv['Disease_encoded']

    scaler_raw = StandardScaler()
    X_raw_tr[CONTINUOUS_COLS + ['Age']] = scaler_raw.fit_transform(X_raw_tr[CONTINUOUS_COLS + ['Age']])
    X_raw_va[CONTINUOUS_COLS + ['Age']] = scaler_raw.transform(X_raw_va[CONTINUOUS_COLS + ['Age']])

    ext_continuous = CONTINUOUS_COLS + ['Age', 'Inflammation_Score', 'C3_C4_Ratio', 'Autoantibody_Count']
    if literature_features:
        ext_continuous += ['ESR_CRP_Product', 'RF_AntiCCP_Product']
    scaler_ext = StandardScaler()
    X_ext_tr[ext_continuous] = scaler_ext.fit_transform(X_ext_tr[ext_continuous])
    X_ext_va[ext_continuous] = scaler_ext.transform(X_ext_va[ext_continuous])

    return X_raw_tr, X_raw_va, X_ext_tr, X_ext_va, y_tr, y_va


def load_cv_pool(raw_path="../data/raw/dataset.xlsx", test_size=0.2, seed=42):
    """
    Returns ONLY the training pool (9,668 rows) that all cross-validation runs
    inside. Reproduces the exact split of 02_preproccessing.ipynb (same
    encoding, same stratified split, same random_state), but the test rows
    are deliberately never returned -- so downstream notebooks cannot touch
    them by accident. Binary markers are mapped to 1/0 (NaN kept), Gender to
    1/0, and the two *_was_missing flags are created BEFORE imputation.
    """
    df = pd.read_excel(raw_path)
    for col in BINARY_COLS:
        df[col] = df[col].map({'Positive': 1, 'Negative': 0})
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['RF_was_missing'] = df['RF'].isna().astype(int)
    df['Anti-CCP_was_missing'] = df['Anti-CCP'].isna().astype(int)
    train_idx, _ = train_test_split(
        df.index, test_size=test_size, stratify=df['Disease'], random_state=seed
    )
    return df.loc[train_idx].copy()