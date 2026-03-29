import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from utils import load_and_preprocess_data, SAVED_MODELS_PATH
from elo import predict_elo
from poisson_model import predict_poisson
from logistic import get_logistic_model, select_logistic_features
from xgboost_model import XGBoostDecayModel

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def compute_brier_score_multiclass(y_true, y_prob):
    y_true_onehot = np.zeros((len(y_true), 3))
    y_true_onehot[np.arange(len(y_true)), y_true] = 1.0
    brier = np.mean(np.sum((y_prob - y_true_onehot)**2, axis=1))
    return float(brier)

def print_comparison(name, log_losses, accuracies, briers):
    logger.info(f"\n===== {name} COMPARISON =====")
    logger.info(f"{'Model':<15} | {'Log Loss':<10} | {'Accuracy':<10} | {'Brier':<10}")
    logger.info("-" * 55)
    for model_name in log_losses.keys():
        logger.info(f"{model_name:<15} | {log_losses[model_name]:.4f}     | {accuracies[model_name]:.2%}   | {briers[model_name]:.4f}")
    logger.info("=======================================\n")

def apply_hierarchical_correction(p_raw):
    """
    Applies conditional logic ensuring final draw probabilities accurately map:
    P_draw = P_draw_raw
    P_home_cond = P_home_raw / (P_home + P_away)
    P_home_final = (1 - P_draw) * P_home_cond
    """
    p_away_raw = p_raw[:, 0]
    p_draw_raw = p_raw[:, 1]
    p_home_raw = p_raw[:, 2]
    
    # Avoid divide by zero
    sum_ha = np.clip(p_home_raw + p_away_raw, a_min=1e-9, a_max=None)
    
    p_home_cond = p_home_raw / sum_ha
    p_away_cond = p_away_raw / sum_ha
    
    p_home_final = (1.0 - p_draw_raw) * p_home_cond
    p_away_final = (1.0 - p_draw_raw) * p_away_cond
    
    final_probs = np.column_stack([p_away_final, p_draw_raw, p_home_final])
    
    # Final normalization explicitly
    sums = final_probs.sum(axis=1, keepdims=True)
    return final_probs / sums

if __name__ == "__main__":
    logger.info("Initializing V2 Model Ensembler...")
    
    (X_train, y_train, _), (X_val, y_val, _), (X_test, y_test, _), meta = load_and_preprocess_data()
    
    # Load tuned XGBoost params if available
    xgb_params = None
    params_path = os.path.join(SAVED_MODELS_PATH, "xgboost_best_params.json")
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            xgb_params = json.load(f)
        logger.info(f"Loaded tuned XGBoost parameters from {params_path}")
    else:
        logger.warning("No tuned XGBoost parameters found. Using defaults.")

    # Meta data subset
    meta_train = meta.loc[X_train.index]
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    oof_predictions = []
    oof_labels = []
    
    logger.info("Generating Out-Of-Fold (OOF) matrices via TimeSeriesSplit...")
    fold = 1
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
        dates_tr = meta_train.iloc[train_idx]['date']
        
        # 1. Math Models (No training necessary, calculated fresh for OOF set)
        elo_val = predict_elo(X_va['elo_diff'])
        poisson_val = predict_poisson(X_va['home_xg_avg5'], X_va['away_xg_avg5'], X_va['home_xga_avg5'], X_va['away_xga_avg5'])
        
        # 2. Logistic Regression
        log_tr_X = select_logistic_features(X_tr)
        log_va_X = select_logistic_features(X_va)
        
        clf_log = get_logistic_model()
        clf_log.fit(log_tr_X, y_tr)
        log_val = clf_log.predict_proba(log_va_X)
        
        # 3. XGBoost
        clf_xgb = XGBoostDecayModel(params=xgb_params)
        clf_xgb.fit(X_tr, y_tr, dates_tr)
        xgb_val = clf_xgb.predict_proba(X_va)
        
        # Concat 12 columns
        fold_probs = np.hstack([elo_val, log_val, xgb_val, poisson_val])
        oof_predictions.append(fold_probs)
        oof_labels.append(y_va.values)
        
        logger.info(f"Fold {fold} complete. {len(y_va)} samples.")
        fold += 1
        
    OOF_X = np.vstack(oof_predictions)
    OOF_y = np.concatenate(oof_labels)
    
    logger.info(f"Training Meta-Model on stacked array of shape {OOF_X.shape}...")
    meta_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
    meta_model.fit(OOF_X, OOF_y)
    
    # Now retrain base models strictly on FULL training set
    logger.info("Retraining Base Models on Full Train Set...")
    final_log_model = get_logistic_model()
    final_log_model.fit(select_logistic_features(X_train), y_train)
    
    final_xgb_model = XGBoostDecayModel(params=xgb_params)
    final_xgb_model.fit(X_train, y_train, meta_train['date'])
    
    def evaluate_set(X, y, split_name):
        logger.info(f"\nProcessing {split_name} Split...")
        # 1. Feature extracts
        elo_preds = predict_elo(X['elo_diff'])
        poisson_preds = predict_poisson(X['home_xg_avg5'], X['away_xg_avg5'], X['home_xga_avg5'], X['away_xga_avg5'])
        log_preds = final_log_model.predict_proba(select_logistic_features(X))
        xgb_preds = final_xgb_model.predict_proba(X)
        
        # Format X_stack
        X_stack = np.hstack([elo_preds, log_preds, xgb_preds, poisson_preds])
        
        # Meta inference
        ensemble_raw_preds = meta_model.predict_proba(X_stack)
        
        # Step 4: Hierarchical Conversion
        ensemble_preds = apply_hierarchical_correction(ensemble_raw_preds)
        
        models = {
            "ELO": elo_preds,
            "LogReg": log_preds,
            "XGBoost": xgb_preds,
            "Poisson": poisson_preds,
            "Ensemble": ensemble_preds
        }
        
        lls = {}
        accs = {}
        briers = {}
        
        y_vals = y.values
        for n, p in models.items():
            lls[n] = log_loss(y_vals, p)
            accs[n] = accuracy_score(y_vals, np.argmax(p, axis=1))
            briers[n] = compute_brier_score_multiclass(y_vals, p)
            
        print_comparison(split_name, lls, accs, briers)
        
        return models

    _ = evaluate_set(X_val, y_val, "Validation")
    _ = evaluate_set(X_test, y_test, "Test")

    logger.info("V2 Execution completed securely.")
