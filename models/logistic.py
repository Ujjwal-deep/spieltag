import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def get_logistic_model():
    """
    Returns an un-fitted scikit-learn pipeline for Logistic Regression 
    using only specific differential and strength features.
    
    Includes standard scaling and isotonic calibration.
    """
    logreg = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        C=0.1,
        max_iter=1000,
        random_state=42
    )
    
    # We calibrate the output probabilities
    calibrated_clf = CalibratedClassifierCV(logreg, method='isotonic', cv=3)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('calibrated', calibrated_clf)
    ])
    
    return pipeline

def select_logistic_features(X):
    """
    Subsets the data to the 6 explicitly requested features for V2 LogReg.
    X can be a pandas DataFrame or similar.
    """
    cols = ['elo_diff', 'form_diff', 'xg_diff', 'xga_diff', 'rest_diff', 'strength_diff']
    if isinstance(X, pd.DataFrame):
        return X[cols]
    
    # If numpy array, assume the caller handles column indices.
    # In ensemble.py we will pass dataframe blocks to base models.
    raise ValueError("X must be a DataFrame to reliably select features by name.")
