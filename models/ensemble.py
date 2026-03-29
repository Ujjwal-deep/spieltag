import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, confusion_matrix

from utils import load_and_preprocess_data, SAVED_MODELS_PATH
from elo import predict_elo

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def compute_brier_score_multiclass(y_true, y_prob):
    y_true_onehot = np.zeros((len(y_true), 3))
    y_true_onehot[np.arange(len(y_true)), y_true] = 1.0
    brier = np.mean(np.sum((y_prob - y_true_onehot)**2, axis=1))
    return float(brier)

def print_comparison(name, accs, briers):
    logger.info(f"\n===== {name} COMPARISON =====")
    logger.info(f"{'Model':<15} | {'Accuracy':<10} | {'Brier':<10}")
    logger.info("-" * 42)
    for model_name in accs.keys():
        logger.info(f"{model_name:<15} | {accs[model_name]:.2%}   | {briers[model_name]:.4f}")
    logger.info("=====================================\n")

def get_llm_probs(match_ids):
    cache_path = os.path.join(SAVED_MODELS_PATH, "llm_cache.json")
    if not os.path.exists(cache_path):
        raise FileNotFoundError("LLM Cache not found. Run llm_model.py first.")
    
    with open(cache_path, "r") as f:
        cache = json.load(f)
        
    probs = []
    fallback = [0.30, 0.25, 0.45] # away_win, draw, home_win
    
    for mid in match_ids:
        if mid in cache:
            res = cache[mid]
            probs.append([res['away_win'], res['draw'], res['home_win']])
        else:
            probs.append(fallback)
            
    return np.array(probs)

if __name__ == "__main__":
    logger.info("Initializing Ensemble Model...")
    
    _, (X_val, y_val, _), (X_test, y_test, _), meta = load_and_preprocess_data()
    
    # Load Models
    scaler_path = os.path.join(SAVED_MODELS_PATH, "scaler.pkl")
    logreg_path = os.path.join(SAVED_MODELS_PATH, "logistic.pkl")
    xgb_path = os.path.join(SAVED_MODELS_PATH, "xgboost.pkl")
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(logreg_path, "rb") as f:
        logreg = pickle.load(f)
    with open(xgb_path, "rb") as f: # or joblib, wait, xgboost was saved with joblib, but we can also load with joblib
        import joblib
        xgb_model = joblib.load(xgb_path)

    # Initial weights: ELO (0.15), LogReg (0.25), XGBoost (0.35), LLM (0.25)
    # Cold start weights: ELO (0.10), LogReg (0.15), XGBoost (0.20), LLM (0.55)
    base_w = np.array([0.15, 0.25, 0.35, 0.25])
    cold_w = np.array([0.10, 0.15, 0.20, 0.55])
    
    logger.info("Saving ensemble weights to JSON...")
    weight_config = {"base_weights": list(base_w), "cold_start_weights": list(cold_w)}
    with open(os.path.join(SAVED_MODELS_PATH, "ensemble_weights.json"), "w") as f:
        json.dump(weight_config, f, indent=4)
        
    all_predictions = []

    def evaluate_set(X, y, split_name):
        logger.info(f"Evaluating Ensemble on {split_name} Set...")

        # 1. ELO Probs
        elo_probs = predict_elo(X['home_elo'], X['away_elo'])
        
        # 2. LogReg Probs
        X_scaled = scaler.transform(X)
        logreg_probs = logreg.predict_proba(X_scaled)
        
        # 3. XGBoost Probs
        xgb_probs = xgb_model.predict_proba(X)
        
        # 4. LLM Probs
        m_ids = meta.loc[X.index, 'match_id'].values
        llm_probs = get_llm_probs(m_ids)
        
        # Combine
        is_promoted = (X['is_home_promoted'] == 1) | (X['is_away_promoted'] == 1)
        ensemble_probs = np.zeros_like(elo_probs)
        
        for i in range(len(X)):
            if is_promoted.iloc[i]:
                w = cold_w
            else:
                w = base_w
                
            probs = (
                w[0] * elo_probs[i] +
                w[1] * logreg_probs[i] +
                w[2] * xgb_probs[i] +
                w[3] * llm_probs[i]
            )
            ensemble_probs[i] = probs / np.sum(probs)
            
            # Save for record
            actual = y.iloc[i]
            pred = np.argmax(ensemble_probs[i])
            row = {
                "match_id": m_ids[i],
                "actual": actual,
                "predicted": pred,
                "split": split_name,
                "elo_prob_away": elo_probs[i][0], "elo_prob_draw": elo_probs[i][1], "elo_prob_home": elo_probs[i][2],
                "logreg_prob_away": logreg_probs[i][0], "logreg_prob_draw": logreg_probs[i][1], "logreg_prob_home": logreg_probs[i][2],
                "xgb_prob_away": xgb_probs[i][0], "xgb_prob_draw": xgb_probs[i][1], "xgb_prob_home": xgb_probs[i][2],
                "llm_prob_away": llm_probs[i][0], "llm_prob_draw": llm_probs[i][1], "llm_prob_home": llm_probs[i][2],
                "ens_prob_away": ensemble_probs[i][0], "ens_prob_draw": ensemble_probs[i][1], "ens_prob_home": ensemble_probs[i][2],
            }
            all_predictions.append(row)
            
        # Metrics
        models = {
            "ELO": elo_probs,
            "LogReg": logreg_probs,
            "XGBoost": xgb_probs,
            "LLM": llm_probs,
            "Ensemble": ensemble_probs
        }
        
        accs = {}
        briers = {}
        
        y_vals = y.values
        for n, p in models.items():
            accs[n] = accuracy_score(y_vals, np.argmax(p, axis=1))
            briers[n] = compute_brier_score_multiclass(y_vals, p)
            
        print_comparison(split_name, accs, briers)
        
        if split_name == "Validation" and accs["Ensemble"] < 0.45:
            logger.warning(f"Ensemble Accuracy on Validation is under 45% ({accs['Ensemble']:.2%})")
            
        if split_name == "Test":
            # Test-specific metrics
            cm = confusion_matrix(y_vals, np.argmax(ensemble_probs, axis=1))
            logger.info("Confusion Matrix on Test Set (rows: Truth, cols: Pred):")
            logger.info("[Away, Draw, Home]")
            logger.info(cm)
            
            # Outperformed best individual?
            best_indiv_acc = max([accs[m] for m in models if m != "Ensemble"])
            logger.info(f"Percentage of matches where Ensemble outperformed best individual? Actually measured over the whole set:")
            if accs["Ensemble"] > best_indiv_acc:
                diff = accs["Ensemble"] - best_indiv_acc
                logger.info(f"Yes, outperformed by {diff:+.2%}")
            else:
                logger.info(f"No, fell short by {accs['Ensemble'] - best_indiv_acc:+.2%}")

    evaluate_set(X_val, y_val, "Validation")
    evaluate_set(X_test, y_test, "Test")

    # Save all predictions to simple CSV
    pred_df = pd.DataFrame(all_predictions)
    pred_path = os.path.join(SAVED_MODELS_PATH, "ensemble_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Predictions saved to {pred_path}")
    logger.info("Ensemble evaluation complete.")
