import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss

from utils import load_and_preprocess_data, SAVED_MODELS_PATH

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def compute_brier_score_multiclass(y_true, y_prob):
    # Ensure correct array matching shapes
    y_true_onehot = np.zeros((len(y_true), 3))
    y_true_onehot[np.arange(len(y_true)), y_true] = 1.0
    brier = np.mean(np.sum((y_prob - y_true_onehot)**2, axis=1))
    return float(brier)

def optimize_xgboost(X_train, y_train, w_train):
    # We will use TimeSeriesSplit inside optuna to simulate the chronological nature
    tscv = TimeSeriesSplit(n_splits=5)
    
    def objective(trial):
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
            "tree_method": "hist",
            "random_state": 42
        }
        
        brier_scores = []
        # Convert arrays
        X_vals = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_vals = y_train.values if isinstance(y_train, pd.Series) else y_train
        w_vals = w_train.values if hasattr(w_train, 'values') else w_train

        for train_idx, val_idx in tscv.split(X_vals):
            X_tr, X_va = X_vals[train_idx], X_vals[val_idx]
            y_tr, y_va = y_vals[train_idx], y_vals[val_idx]
            w_tr = w_vals[train_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            
            y_prob_va = model.predict_proba(X_va)
            brier = compute_brier_score_multiclass(y_va, y_prob_va)
            brier_scores.append(brier)
            
        return np.mean(brier_scores)
        
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, n_jobs=1)  # Using sequential for safety or 1
    return study

if __name__ == "__main__":
    logger.info("Initializing XGBoost Model Optimization...")
    (X_train, y_train, w_train), (X_val, y_val, w_val), (X_test, y_test, w_test), meta = load_and_preprocess_data()
    
    # Run optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Limit optuna spam
    study = optimize_xgboost(X_train, y_train, w_train)
    
    best_params = study.best_params
    best_cv_brier = study.best_value
    logger.info("--- Optuna Tuning Results ---")
    logger.info(f"Best CV Brier Score: {best_cv_brier:.4f}")
    logger.info(f"Best Params: {json.dumps(best_params, indent=2)}")
    
    # Save best parameters
    with open(os.path.join(SAVED_MODELS_PATH, "xgboost_best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)
        
    # Retrain on full train set
    logger.info("Retraining final model on full X_train...")
    final_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": 42
    }
    final_params.update(best_params)
    
    model = xgb.XGBClassifier(**final_params)
    model.fit(X_train, y_train, sample_weight=w_train)
    
    # Evaluate on val
    y_prob = model.predict_proba(X_val)
    y_pred = model.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred)
    ll = log_loss(y_val, y_prob)
    brier = compute_brier_score_multiclass(y_val.values if hasattr(y_val, 'values') else y_val, y_prob)
    
    metrics = {
        "accuracy": acc,
        "log_loss": ll,
        "brier_score": brier
    }
    
    # Poor performance fallback loop
    # Wait, instructions specifically say: "If accuracy < 45%: Re-run Optuna with 30 more trials (XGBoost only)"
    if acc < 0.45:
        logger.warning(f"Validation accuracy < 45% ({acc:.2%}). Running 30 more Optuna trials...")
        study.optimize(optimize_xgboost(X_train, y_train, w_train).objective, n_trials=30)
        # Redo everything
        best_params = study.best_params
        for k, v in best_params.items():
            final_params[k] = v
        with open(os.path.join(SAVED_MODELS_PATH, "xgboost_best_params.json"), "w") as f:
            json.dump(best_params, f, indent=4)
        model = xgb.XGBClassifier(**final_params)
        model.fit(X_train, y_train, sample_weight=w_train)
        
        y_prob = model.predict_proba(X_val)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        ll = log_loss(y_val, y_prob)
        brier = compute_brier_score_multiclass(y_val.values if hasattr(y_val, 'values') else y_val, y_prob)
        
        metrics = {
            "accuracy": acc,
            "log_loss": ll,
            "brier_score": brier
        }
        
    if acc < 0.45:
        logger.error("Accuracy is STILL < 45%. Writing failure report.")
        with open(os.path.join(SAVED_MODELS_PATH, "failure_report.txt"), "a") as f:
            f.write(f"XGBoost failed to reach 45% val accuracy. Final: {acc:.2%}\n")
    elif acc <= 0.50:
        logger.warning("Accuracy is between 45-50%. Acceptable but weak.")
        
    with open(os.path.join(SAVED_MODELS_PATH, "xgboost_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    joblib.dump(model, os.path.join(SAVED_MODELS_PATH, "xgboost.pkl"))
    
    logger.info("--- XGBoost Validation Results ---")
    logger.info(f"Accuracy: {acc:.2%}")
    logger.info(f"Log Loss: {ll:.4f}")
    logger.info(f"Brier Score: {brier:.4f}")
    
    logger.info("Top 10 Feature Importances (Gain):")
    # importance_type='gain' equivalent
    gains = model.get_booster().get_score(importance_type='gain')
    importances = sorted(gains.items(), key=lambda x: x[1], reverse=True)[:10]
    for feat, gain in importances:
        logger.info(f"  {feat}: {gain:.4f}")
