# Predicting Stellar Class

Kaggle competition: multiclass classification of astronomical objects into **GALAXY**, **QSO** (quasar), or **STAR** using photometric and spectroscopic data from the Sloan Digital Sky Survey (SDSS).

## Dataset

~577K training samples, 11 features:

| Feature | Description |
|---|---|
| `alpha`, `delta` | Right ascension / declination (sky coordinates) |
| `u`, `g`, `r`, `i`, `z` | Photometric magnitudes in 5 optical bands |
| `redshift` | Spectroscopic redshift - strongest predictor |
| `spectral_type` | Categorical spectral classification |
| `galaxy_population` | Categorical galaxy group membership |

**Target:** `class` - GALAXY / QSO / STAR  
**Metric:** Balanced Accuracy

## Approach

### EDA highlights
- `redshift` is by far the most discriminative feature (high mutual information with all 3 classes)
- Strong multicollinearity among photometric bands (g↔u, r↔g, i↔r, z↔r)
- Outliers in `u` (values < 2.5) and `i` (≥ 27) → replaced with NaN and median-imputed
- Galaxy/Star confusion is the dominant error pattern

### Feature engineering
- **Color indices**: `u-g`, `g-r`, `r-i`, `i-z` - standard astronomical colors derived from adjacent photometric bands

### Preprocessing pipeline
```
FeatureEngineering → ColumnTransformer
  ├── numeric: outlier→NaN → median imputation
  └── categorical: most-frequent imputation → OneHotEncoder
```

### Models & results (5-fold StratifiedKFold, balanced accuracy)

| Model | CV Score |
|---|---|
| XGBoost baseline | 0.9534 |
| XGBoost tuned (Optuna, 50 trials) | 0.9552 |
| LightGBM tuned (Optuna, 50 trials) | 0.9561 |
| CatBoost baseline | 0.9534 |
| **Soft Voting Ensemble (XGB + LGBM + CatBoost)** | **0.9535** |

Hyperparameter search done with Optuna TPE sampler.

## Project structure

```
├── main.ipynb                      # full pipeline: EDA → preprocessing → modelling → submission
├── data/
│   ├── train.csv
│   └── test.csv
├── submission.csv                  # final predictions
├── baseline_model_comparison.png   # model comparison chart
└── comparison_baseline_tuned.png   # feature importance: baseline vs tuned XGBoost
```

## Requirements

```
pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, optuna, seaborn, matplotlib, scipy
```
