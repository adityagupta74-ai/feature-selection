# Filter & Wrapper Based Feature Selection

Applying feature selection techniques on real-world datasets — comparing model performance before and after selection, across both **filter-based** (statistical) and **wrapper-based** (model-driven) approaches.

## Filter-Based Feature Selection

`filter_based_feature_selection.ipynb` — uses the Human Activity Recognition (HAR) and Titanic datasets.

Techniques covered:
- **Removing Duplicate Columns** — dropping exact-duplicate features
- **Variance Threshold** — removing near-constant, low-information features
- **Correlation-Based Removal** — dropping redundant, highly correlated features (>0.95)
- **ANOVA (SelectKBest)** — ranking features by F-statistic, keeping the top 100
- **Chi-Square Test** — testing independence between categorical features and the target (Titanic dataset)

### Results (HAR Dataset — Logistic Regression)

| Stage | Features | Test Accuracy |
|---|---|---|
| Baseline (all features) | 561 | 0.981 |
| After duplicate removal | 540 | — |
| After variance threshold | 349 | — |
| After correlation removal | 152 | — |
| After ANOVA (top 100) | 100 | 0.969 |

**Key takeaway**: An 82% reduction in features cost only ~1.2 percentage points of accuracy — a strong trade-off for a simpler, faster, more interpretable model.

## Wrapper-Based Feature Selection

`wrapper_methods.ipynb` — uses the Iris and Boston Housing datasets.

Techniques covered:
- **Exhaustive Feature Selector (EFS)** — brute-force search across all feature subsets (capped at `max_features=8` for Boston Housing to keep runtime reasonable — full exhaustive search over 13 features would test 8,192 combinations)
- **Forward Selection (mlxtend)** — greedy, iterative feature addition
- **Forward Selection (sklearn)** — comparison using sklearn's built-in `SequentialFeatureSelector`

### Results (Boston Housing — Linear Regression)

| Method | Features Selected | Testing R² |
|---|---|---|
| Baseline (all features) | 13 | 0.65 |
| EFS (capped at 8) | 8 | 0.70 |
| Forward Selection (mlxtend) | 10 | 0.72 |
| Forward Selection (sklearn) | 5 | — |

**Key takeaway**: Wrapper methods directly optimize for model performance, unlike filter methods. Forward Selection, despite being greedy rather than exhaustive, outperformed the capped EFS search here — showing it can be a strong, much faster alternative in practice.

## Setup

```bash
python3.11 -m venv GDenv
source GDenv/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn scipy mlxtend ipykernel jupyter
```

## Files

- `filter_based_feature_selection.ipynb` — filter-based techniques (HAR + Titanic)
- `wrapper_methods.ipynb` — wrapper-based techniques (Iris + Boston Housing)
- `data/` — HAR and Titanic datasets (see `data/README.md` for details)
