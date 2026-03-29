import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

from utils import load_and_preprocess_data, SAVED_MODELS_PATH

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def compute_brier_score_multiclass(y_true, y_prob):
    y_true_onehot = np.zeros_like(y_prob)
    y_true_onehot[np.arange(len(y_true)), y_true.values] = 1.0
    brier = np.mean(np.sum((y_prob - y_true_onehot)**2, axis=1))
    return float(brier)

def train_and_eval(X_train, y_train, w_train, X_val, y_val, C_val, class_weight=None):
    model = LogisticRegression(
        multi_class='multinomial', 
        solver='lbfgs', 
        max_iter=1000, 
        C=C_val,
        class_weight=class_weight,
        random_state=42
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    
    y_prob = model.predict_proba(X_val)
    y_pred = model.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred)
    ll = log_loss(y_val, y_prob)
    brier = compute_brier_score_multiclass(y_val, y_prob)
    
    return model, acc, ll, brier

if __name__ == "__main__":
    logger.info("Initializing Logistic Regression Model...")
    
    (X_train, y_train, w_train), (X_val, y_val, w_val), (X_test, y_test, w_test), meta = load_and_preprocess_data()
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save the scaler
    scaler_path = os.path.join(SAVED_MODELS_PATH, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        
    C_values = [0.01, 0.1, 1.0]
    best_acc = 0.0
    best_metrics = {}
    best_model = None
    best_params = {}
    
    logger.info("Starting hyperparameter tuning for C...")
    for C in C_values:
        model, acc, ll, brier = train_and_eval(X_train_scaled, y_train, w_train, X_val_scaled, y_val, C)
        logger.info(f"C={C} -> Val Acc: {acc:.2%}, Brier: {brier:.4f}")
        
        # We optimize for best accuracy as per typical workflow, or brier score
        # The prompt says: "Try C values [0.01, 0.1, 1.0] on the validation set and pick the best"
        # Brier is usually better to optimize for probabilities, but we'll use Brier threshold.
        # Let's optimize for Brier score (lower is better) but fallback to evaluating Accuracy.
        if (not best_metrics) or (brier < best_metrics["brier_score"]):
            best_model = model
            best_acc = acc
            best_params = {"C": C, "class_weight": None}
            best_metrics = {
                "accuracy": acc,
                "log_loss": ll,
                "brier_score": brier
            }
            
    if best_metrics["accuracy"] < 0.45:
        logger.warning(f"Accuracy < 45% ({best_metrics['accuracy']:.2%}). Attempting poor accuracy handling with class_weight='balanced'.")
        # Retry best C with balanced
        b_model, b_acc, b_ll, b_brier = train_and_eval(X_train_scaled, y_train, w_train, X_val_scaled, y_val, best_params["C"], class_weight="balanced")
        logger.info(f"Balanced -> Val Acc: {b_acc:.2%}, Brier: {b_brier:.4f}")
        if b_acc > best_metrics["accuracy"]:
            best_model = b_model
            best_metrics = {
                "accuracy": b_acc,
                "log_loss": b_ll,
                "brier_score": b_brier
            }
            best_params["class_weight"] = "balanced"
            
    if best_metrics["accuracy"] < 0.45:
        logger.error("Accuracy is STILL < 45%. Writing failure_report.txt")
        with open(os.path.join(SAVED_MODELS_PATH, "failure_report.txt"), "a") as f:
            f.write(f"Logistic Regression failed to reach 45% accuracy on validation set. Final Acc: {best_metrics['accuracy']:.2%}\n")
    elif best_metrics["accuracy"] <= 0.50:
        logger.warning("Accuracy is between 45-50%. Acceptable but weak.")
        
    logger.info(f"Best Params: {best_params}")
    logger.info("--- Logistic Regression Validation Results ---")
    logger.info(f"Accuracy: {best_metrics['accuracy']:.2%}")
    logger.info(f"Log Loss: {best_metrics['log_loss']:.4f}")
    logger.info(f"Brier Score: {best_metrics['brier_score']:.4f}")

    out_file = os.path.join(SAVED_MODELS_PATH, "logistic_metrics.json")
    with open(out_file, "w") as f:
        json.dump(best_metrics, f, indent=4)
        
    model_path = os.path.join(SAVED_MODELS_PATH, "logistic.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
        
    logger.info("Top 5 Influential Features Grouped by Class (Magnitude):")
    # Multiclass logreg gives coef_ of shape (n_classes, n_features)
    feature_names = X_train.columns
    for i, c in enumerate(best_model.classes_):
        class_name = ["Away Win", "Draw", "Home Win"][c]
        coefs = best_model.coef_[i]
        top_idx = np.argsort(np.abs(coefs))[-5:][::-1]
        top_features = [(feature_names[j], coefs[j]) for j in top_idx]
        logger.info(f"  Class {class_name}:")
        for f_name, f_val in top_features:
            logger.info(f"    {f_name}: {f_val:.4f}")
