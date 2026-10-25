# Fraudulent Bitcoin Transaction Detection (BT4012 Kaggle competition)

`E1156344.ipynb` is the complete, end-to-end solution: exploratory analysis, baselines, graph feature engineering, feature
pruning, tuning, the models and blends that were submitted, and SHAP explainability. Running it top to bottom regenerates every
submission file.

**Final submission:** XGBoost + a neural network (MLP) trained with time-decay weights, rank-blended 3:1. Public leaderboard
ROC AUC **0.95876**.

## 1. Requirements

- **Python 3.11** (the results were produced with 3.11.1)
- **About 8 GB of RAM** (`train.csv` alone is about 460 MB)
- **CPU only**; no GPU needed

## 2. Setup

From this folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name bt4012 --display-name "Python (bt4012)"
```

`requirements.txt` pins the exact package versions used. Keep them: a different XGBoost or PyTorch version can change
predictions slightly.

## 3. Data

Download the competition data from Kaggle and place the four files in a `data/` folder next to the notebook:

```
data/
  train.csv               # transactions in time steps 1-35, with labels (1 = illicit, 0 = licit, empty = unknown)
  test.csv                # transactions in time steps 36-49, to be scored
  txs_edgelist.csv        # payment edges between transactions
  sample_submission.csv   # submission format
```

The notebook reads only these four files.

## 4. Running the notebook

1. Open `E1156344.ipynb` in Jupyter or VS Code and select the kernel created above.
2. Run **Restart Kernel and Run All**. Cells depend on earlier ones, so run them in order rather than individually.
3. Expect **about 40-50 minutes** on a laptop CPU. The slow parts are the Optuna searches (section 5), building the graph
   features at several cutoffs, and training the MLPs (sections 10-11).

### What each section does

| Section | Content |
|---|---|
| 0 | Setup and data loading |
| 1 | Exploratory data analysis: class imbalance, the time split, feature redundancy and drift, the transaction graph |
| 2 | Baselines on raw features (XGBoost, LightGBM, logistic regression, random forest, CatBoost) with temporal cross-validation |
| 3 | Graph features: degree, a leak-free risk score (`risk_prob_self`), neighbour-risk aggregates |
| 4 | Feature pruning from 174 to 82 features, including how the dropped-feature list is derived and why the 0.9 correlation threshold was chosen |
| 5 | Hyperparameter tuning with Optuna on temporal folds (XGBoost and LightGBM) |
| 6 | Evaluation on held-out periods 22-35 (the "horizon curve") |
| 7 | XGBoost submission (public 0.95730) |
| 8-9 | A second XGBoost model with structural graph features, and its blend (0.95706) |
| 10 | XGBoost + MLP rank blend (0.95830) |
| 11 | Final submission: MLP with time-decay weights (0.95876), compared with every decay, rank and smoothing variant tried |
| 12 | SHAP explainability for the final XGBoost model |

## 5. Output files

All files are written next to the notebook, overwriting any existing file of the same name.

| File | Section | Public LB |
|---|---|---|
| **`final_submission.csv/decay_xgbnone_mlpdecay098_blend_3to1_submission.csv`** | 11 | **0.95876** (final submission) |
| `pruned_xgboost_narrow_submission.csv` | 7 | 0.95730 (second submission) |
| `graph_structure_v2_submission.csv` | 8 | 0.95637 |
| `blend_rank_04_gsv2_submission.csv` | 9 | 0.95706 |
| `xgb_mlp_blend_{5,4,3,2,1}to1_submission.csv` | 10 | 3:1 = 0.95830 |
| `decay_*_submission.csv` (one per variant) | 11 | see the table in section 11 |

## 6. Reproducibility notes

- **Seeds are fixed** (`RANDOM_STATE = 42`; `MLP_SEEDS` in section 10). The submitted files were produced with
  `MLP_SEEDS = [7, 42, 314, 999, 4102]`. With other seeds, the XGBoost results are identical and the MLP results differ only
  slightly (rank correlation about 0.9997 with the submitted file).
- **Section 7 uses this run's Optuna result.** On the same machine and package versions it finds the same settings as the
  submission (`n_estimators=492, max_depth=7, learning_rate=0.0139...`); sections 10-11 use these settings fixed, so they do
  not depend on the search.
- **Keep `import xgboost` above `import torch`** in the setup cell. Importing PyTorch first can crash the kernel on macOS (both
  ship their own OpenMP runtime). The notebook also limits PyTorch to one thread for the same reason.
- **Two optional switches in section 3** (`USE_COMPONENT_SIZE`, `USE_NB_FEATURES`) add experimental features. Both must stay
  `False` to reproduce the submitted results.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: data/train.csv` | Put the four CSV files in `data/` next to the notebook (section 3 above) |
| Kernel dies right after the imports | Check that `import xgboost` comes before `import torch` in the setup cell |
| `ModuleNotFoundError` | Activate the virtual environment and select its kernel in Jupyter or VS Code |
| Out of memory | Close other notebooks; about 8 GB of RAM is needed |
