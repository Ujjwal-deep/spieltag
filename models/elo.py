import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from utils import load_and_preprocess_data, SAVED_MODELS_PATH

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def predict_elo(home_elo, away_elo):
    elo_diff = home_elo - away_elo
    expected_home = 1 / (1 + 10 ** (-elo_diff / 400))
    
    p_home = expected_home * 0.85
    p_draw = 0.25 - abs(elo_diff) * 0.0002
    p_away = 1 - p_home - p_draw
    
    # Pack into array to vectorize clipping easily
    probs = np.column_stack([p_away, p_draw, p_home])
    probs = np.clip(probs, 0.05, 0.90)
    
    # Normalize so they sum to 1 strictly
    sums = probs.sum(axis=1, keepdims=True)
    probs = probs / sums
    return probs

def evaluate_model(y_true, y_prob):
    # Predict class via argmax
    y_pred = np.argmax(y_prob, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_prob)
    
    # Multi-class brier score: sum of squared differences
    # Convert y_true to one-hot
    y_true_onehot = np.zeros_like(y_prob)
    y_true_onehot[np.arange(len(y_true)), y_true.values] = 1.0
    
    # Brier score is often the MSE per class sum/mean
    # sklearn's brier_score_loss only supports binary out of the box,
    # we manually compute multi-class Brier score: 1/N * sum( (P - y_true_onehot)^2 )
    brier = np.mean(np.sum((y_prob - y_true_onehot)**2, axis=1))
    
    return float(acc), float(ll), float(brier)

if __name__ == "__main__":
    logger.info("Initializing ELO Baseline...")
    
    (X_train, y_train, w_train), (X_val, y_val, w_val), (X_test, y_test, w_test), meta = load_and_preprocess_data()
    
    logger.info("Extracting ELOs for Val set...")
    val_probs = predict_elo(X_val['home_elo'], X_val['away_elo'])
    
    acc, ll, brier = evaluate_model(y_val, val_probs)
    
    metrics = {
        "accuracy": acc,
        "log_loss": ll,
        "brier_score": brier
    }
    
    out_file = os.path.join(SAVED_MODELS_PATH, "elo_metrics.json")
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info("--- ELO Validation Results ---")
    logger.info(f"Accuracy: {acc:.2%}")
    logger.info(f"Log Loss: {ll:.4f}")
    logger.info(f"Brier Score: {brier:.4f}")
    
    if acc < 0.45:
        logger.warning("ELO Accuracy is very weak (< 45%).")
    
    logger.info(f"Metrics saved to {out_file}")
