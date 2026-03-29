import os
import json
import numpy as np
import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss

class XGBoostDecayModel:
    """
    Custom wrapper to seamlessly handle exponential time-decay sample_weights 
    while preserving the scikit-learn API required by ensembles or calibration.
    """
    def __init__(self, decay_rate=0.002, params=None):
        self.decay_rate = decay_rate
        
        default_params = {
            'n_estimators': 300,
            'max_depth': 4,
            'learning_rate': 0.05,
            'objective': 'multi:softprob',
            'num_class': 3,
            'random_state': 42,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist'
        }
        
        if params:
            # Overwrite defaults with tuned params
            # Ensure static required params like objective/num_class aren't lost
            for k, v in params.items():
                default_params[k] = v
                
        self.xgb = XGBClassifier(**default_params)
        self.calibrated = CalibratedClassifierCV(self.xgb, method='isotonic', cv=3)

    def _compute_weights(self, dates_series):
        # Convert to datetime to find max
        dt_idx = pd.to_datetime(dates_series)
        max_date = dt_idx.max()
        days_ago = (max_date - dt_idx).dt.days
        weights = np.exp(-self.decay_rate * days_ago)
        return weights.values

    def fit(self, X, y, dates):
        """
        Fit the model using time decay from the provided dates array.
        Because CalibratedClassifierCV passes sample_weights downward automatically,
        we dynamically compute them internally here from the dates array of this specific fold.
        """
        w = self._compute_weights(dates)
        
        # Fit calibrated classifier and pass the sample_weights dict
        fit_params = {'calibrated__sample_weight': w} # For pipeline if used, but direct use:
        # CalibratedClassifierCV explicitly passes fit_params downward
        self.calibrated.fit(X, y, sample_weight=w)
        return self

    def predict_proba(self, X):
        return self.calibrated.predict_proba(X)
        
    def predict(self, X):
        return self.calibrated.predict(X)

def objective(trial, X, y, dates):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2),
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'tree_method': 'hist'
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    losses = []
    
    # Ensure indices are correct for splitting
    for train_idx, val_idx in tscv.split(X):
        # Time-series split
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        dates_tr = dates.iloc[train_idx]
        
        # Compute weights ONLY on training set to avoid leakage
        dt_idx = pd.to_datetime(dates_tr)
        max_date = dt_idx.max()
        days_ago = (max_date - dt_idx).dt.days
        weights = np.exp(-0.002 * days_ago)
        
        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=weights)
        
        probs = model.predict_proba(X_va)
        losses.append(log_loss(y_va, probs))
        
    return np.mean(losses)

def tune_xgboost(X, y, dates):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y, dates), n_trials=50)
    
    print("\n" + "="*50)
    print("XGBOOST TUNING COMPLETE")
    print(f"Best Log Loss: {study.best_value:.4f}")
    print("Best Params:")
    print(json.dumps(study.best_params, indent=2))
    print("="*50 + "\n")
    
    # Save best params
    os.makedirs(os.path.join("models", "saved"), exist_ok=True)
    with open(os.path.join("models", "saved", "xgboost_best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    return study.best_params

if __name__ == "__main__":
    from utils import load_and_preprocess_data
    
    print("Loading data for tuning...")
    (X_train, y_train, _), _, _, meta = load_and_preprocess_data()
    
    # We tune on the training set only
    # Dates are in meta['date']. Need to align with X_train index.
    dates_train = meta.loc[X_train.index, 'date']
    
    tune_xgboost(X_train, y_train, dates_train)
