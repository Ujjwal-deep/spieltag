import os
import json
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base path assuming this is run from spieltag root directory
DATA_PATH = os.path.join("data", "processed", "features.csv")
SAVED_MODELS_PATH = os.path.join("models", "saved")

def load_and_preprocess_data():
    """
    Loads features.csv, encodes columns, splits data, and applies imputation logic.
    Returns:
        (X_train, y_train, w_train), 
        (X_val, y_val, w_val), 
        (X_test, y_test, w_test), 
        metadata
    """
    os.makedirs(SAVED_MODELS_PATH, exist_ok=True)
    
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found at {DATA_PATH}")
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} rows from {DATA_PATH}")

    # Encode h2h_data_quality
    h2h_map = {"none": 0, "partial": 1, "full": 2}
    df["h2h_data_quality"] = df["h2h_data_quality"].map(h2h_map).fillna(0).astype(int)

    # Ordinal encode promoted_stat_quality 
    # For established teams, this column will just fill NaN with 0 later
    promoted_map = {"goals": 0, "xg": 1}
    df["promoted_stat_quality"] = df["promoted_stat_quality"].map(promoted_map)
    
    # Fill specific columns when NOT promoted
    not_promoted_mask = (~df['is_home_promoted'].astype(bool)) & (~df['is_away_promoted'].astype(bool))
    df.loc[not_promoted_mask, 'home_bl2_goals_avg5'] = df.loc[not_promoted_mask, 'home_bl2_goals_avg5'].fillna(0.0)
    df.loc[not_promoted_mask, 'away_bl2_goals_avg5'] = df.loc[not_promoted_mask, 'away_bl2_goals_avg5'].fillna(0.0)
    df.loc[not_promoted_mask, 'promoted_stat_quality'] = df.loc[not_promoted_mask, 'promoted_stat_quality'].fillna(0)

    # Cast to int for is_home_promoted / is_away_promoted
    if df['is_home_promoted'].isnull().any() or df['is_away_promoted'].isnull().any():
        raise ValueError("Found NaNs in is_home_promoted or is_away_promoted. Bug in upstream pipeline.")
    
    df['is_home_promoted'] = df['is_home_promoted'].astype(int)
    df['is_away_promoted'] = df['is_away_promoted'].astype(int)
    
    # Also if there are any remaining promoted_stat_quality NaNs (e.g. for promoted teams with missing), fill with 0
    df['promoted_stat_quality'] = df['promoted_stat_quality'].fillna(0).astype(int)
    
    # Standardize result mapping: (Just in case, but it's already 0,1,2 where 0=away, 1=draw, 2=home)
    assert 'result' in df.columns, "Missing result column"

    # Separate metadata
    meta_cols = ['match_id', 'date', 'home_team', 'away_team', 'season']
    metadata = df[meta_cols].copy()
    
    # Chronological Split
    # Train: all matches up to and including 2021-22 season
    # Val: 2022-23 season
    # Test: 2023-24 season
    def get_season_year(s):
        try:
            # handle formats like "2021-22" or "2021/2022" -> Start year
            return int(s.split('-')[0].split('/')[0])
        except:
            return 0
            
    season_years = df['season'].apply(get_season_year)
    train_mask = season_years <= 2021
    val_mask = season_years == 2022
    test_mask = season_years == 2023

    X_train = df[train_mask].drop(columns=meta_cols + ['result']).copy()
    y_train = df.loc[train_mask, 'result'].copy()
    X_val = df[val_mask].drop(columns=meta_cols + ['result']).copy()
    y_val = df.loc[val_mask, 'result'].copy()
    X_test = df[test_mask].drop(columns=meta_cols + ['result']).copy()
    y_test = df.loc[test_mask, 'result'].copy()

    # Imputation for xG columns pre-2014 & generic NaNs using TRAINING ONLY data
    cols_to_impute = [col for col in X_train.columns if X_train[col].isnull().any()]
    imputation_values = {}
    
    for col in cols_to_impute:
        mean_val = X_train[col].astype(float).mean()
        
        if np.isnan(mean_val):
            # fallback if entirely nan
            mean_val = 0.0
            
        imputation_values[col] = mean_val
        X_train[col] = X_train[col].fillna(mean_val)
        X_val[col] = X_val[col].fillna(mean_val)
        X_test[col] = X_test[col].fillna(mean_val)

    # Save imputation dict
    with open(os.path.join(SAVED_MODELS_PATH, "imputation_values.json"), "w") as f:
        json.dump(imputation_values, f, indent=4)
        
    # Any residual NaNs (e.g. from validation set missing a col that train didn't miss)
    for col in X_val.columns:
        if X_val[col].isnull().any():
            X_val[col] = X_val[col].fillna(0.0)
    for col in X_test.columns:
        if X_test[col].isnull().any():
            X_test[col] = X_test[col].fillna(0.0)

    # Recency Weights calculation (for Train only, though we can return it as an array)
    # Days ago relative to 2022-08-05
    ref_date = pd.to_datetime('2022-08-05')
    
    def get_weights(split_mask):
        match_dates = pd.to_datetime(df.loc[split_mask, 'date'])
        days_ago = (ref_date - match_dates).dt.days.clip(lower=0) 
        # clip so that negative days (future) are 0 penalty
        weights = np.exp(-0.001 * days_ago)
        return weights.values

    w_train = get_weights(train_mask)
    w_val = get_weights(val_mask)   # Not typically used in typical eval, but good to have
    w_test = get_weights(test_mask)

    logger.info(f"Train/Val/Test splits: {len(X_train)} / {len(X_val)} / {len(X_test)}")
    return (X_train, y_train, w_train), (X_val, y_val, w_val), (X_test, y_test, w_test), metadata
