# ⚽ Spieltag - Technical ML Prediction Engine

**Spieltag** is a high-performance Bundesliga match prediction engine utilizing a stacked ensemble of probabilistic and machine learning models. The system is designed to minimize Log Loss through rigorous time-series cross-validation and a hierarchical meta-stacking architecture.

---

## 🧠 Machine Learning Architecture

The engine uses a **Two-Tier Stacking Ensemble** to calibrate its final probabilities:

### Tier 1: Base Predictors
1.  **ELO Rating System**: 
    - Analyzes long-term team strength and relative quality.
    - Uses a custom $K$-factor and home-field advantage (HFA) adjustment specifically tuned for Bundesliga parity.
2.  **Poisson Distribution (xG-Based)**:
    - Models match scorelines as independent Poisson processes.
    - Derived from 5-game rolling averages of **Expected Goals (xG)** and **xG Against (xGA)** to capture recent offensive/defensive efficiency.
3.  **Calibrated XGBoost (Time-Decay)**:
    - Optimized via **Optuna** (50 trials) to minimize `mlogloss`.
    - Implements an **exponential time-decay weight function**: $W = e^{-\lambda \cdot t}$, where $\lambda=0.002$. This prioritizes recent performance while maintaining the historical signal.
    - Wrapped in `CalibratedClassifierCV` with Isotonic regression to ensure well-calibrated probabilities.
4.  **Difference-Based Logistic Regression**:
    - Focuses on derived features such as `xg_diff`, `strength_diff`, and `rest_diff`.
    - Effectively captures linear relationships in team comparative states.

### Tier 2: Meta-Stacking
- **Out-Of-Fold (OOF) Strategy**: Base models are trained using a 5-fold `TimeSeriesSplit`. Predictions from these folds are "stacked" to form a 12-column meta-feature matrix (3 probabilities per base model).
- **Meta-Model**: A Multinomial Logistic Regression model is trained on this OOF matrix. This layer learns the specific biases and error states of each base model, dynamically weighting them to minimize overall Log Loss.

---

## 🛠️ Training & Validation Strategy

The system enforces strict chronological boundaries to eliminate data leakage:

- **Training Split**: All matches from 2013-14 through the end of the 2021-22 season.
- **Validation Split**: The 2022-23 season (used for model selection and meta-stacking).
- **Test Split (Holdout)**: The 2023-24 season (used for the final benchmark shown below).
- **Imputation**: Features missing xG data (pre-2014) are handled via training-set-median imputation to prevent look-ahead bias.

---

## 📐 Post-Processing & Hierarchical Calibration

To ensure the final output is both mathematically sound and intuitively consistent, the system applies a **Hierarchical Correction**:
1.  **Sum-to-One**: Ensured via Softmax output from the meta-model.
2.  **Conditional Draw Logic**: Raw draw probabilities are preserved while Home/Away distributions are adjusted relatively to ensure the "No-Draw" subset correctly reflects the comparative strength delta of the teams.

---

## 📊 Performance Benchmarks (2023-24 Season)

The system was optimized for **Log Loss**, as it penalizes "confident but wrong" predictions more heavily than standard accuracy.

| Model | Log Loss 📉 | Accuracy 🎯 | Brier Score |
| :--- | :--- | :--- | :--- |
| ELO | 1.0494 | 49.35% | 0.6176 |
| Poisson | 1.0218 | 51.96% | 0.6097 |
| XGBoost (Tuned) | 0.9787 | 51.96% | 0.5810 |
| LogReg | 0.9757 | 50.33% | 0.5800 |
| **Meta Ensemble** | **0.9702** | **50.65%** | **0.5777** |

---

## 🧬 Feature Engineering

The `features.py` module explicitly derives comparative relationships to bypass generic tree correlations:
- **Strength Diff**: $(\text{Home xG} - \text{Home xGA}) - (\text{Away xG} - \text{Away xGA})$.
- **Rest Diff**: Days of rest between the current fixture and the team's previous match.
- **Promoted Stat Quality**: An ordinal indicator handling the transition from Bundesliga 2 statistics (goals) to Bundesliga 1 statistics (xG).