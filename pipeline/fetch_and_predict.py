import os
import json
import requests
import sqlite3
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from dotenv import load_dotenv

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from logistic import select_logistic_features
from poisson_model import predict_poisson
from elo import predict_elo

load_dotenv()

API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") # or ANON_KEY

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Missing Supabase credentials in .env")
    SUPABASE_URL = SUPABASE_URL or "http://placeholder.supabase.co"
    SUPABASE_KEY = SUPABASE_KEY or "placeholder"
    
try:
    from supabase import create_client, Client
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except ImportError:
    print("Please pip install supabase")
    sys.exit(1)

# Mapping from API team names to our CSV names
# We'll do a simple mapping. Add more if necessary.
TEAM_MAPPING = {
    "FC Bayern München": "Bayern Munich",
    "Borussia Dortmund": "Dortmund",
    "Bayer 04 Leverkusen": "Leverkusen",
    "RB Leipzig": "RB Leipzig",
    "VfB Stuttgart": "Stuttgart",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "Sport-Club Freiburg": "Freiburg",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "1. FC Heidenheim 1846": "Heidenheim",
    "SV Werder Bremen": "Werder Bremen",
    "FC Augsburg": "Augsburg",
    "VfL Wolfsburg": "Wolfsburg",
    "1. FSV Mainz 05": "Mainz",
    "VfL Bochum 1848": "Bochum",
    "1. FC Union Berlin": "Union Berlin",
    "Borussia Mönchengladbach": "M'gladbach",
    "FC St. Pauli 1910": "St Pauli", # Promoted 24/25
    "Holstein Kiel": "Holstein Kiel" # Promoted 24/25
}

def map_team_name(api_name):
    return TEAM_MAPPING.get(api_name, api_name)

def get_latest_states():
    db_path = 'data/processed/football.db'
    conn = sqlite3.connect(db_path)
    df = pd.read_sql('SELECT * FROM matches ORDER BY date ASC', conn)
    
    team_state = {}
    h2h_state = {}
    bl2_standings = {}
    
    for season in df['season'].unique():
        s_df = df[(df['season'] == season) & (df['division'] == 'BL2')]
        pts = {}
        for _, row in s_df.iterrows():
            ht, at = row['home_team'], row['away_team']
            r = row['result']
            pts[ht] = pts.get(ht, 0) + (3 if r == 2 else 1 if r == 1 else 0)
            pts[at] = pts.get(at, 0) + (3 if r == 0 else 1 if r == 1 else 0)
        sorted_teams = sorted(pts.items(), key=lambda x: x[1])
        bl2_standings[season] = [t[0] for t in sorted_teams]
        
    def get_team_state(team):
        if team not in team_state:
            team_state[team] = {
                'elo': 1500, 'form_pts': [], 'xg_scored': [], 'xga_conceded': [],
                'goals_scored': [], 'goals_conceded': [], 'bl2_goals_scored': [],
                'home_wins': 0, 'home_played': 0, 'away_wins': 0, 'away_played': 0,
                'last_date': None, 'current_season_division': None
            }
        return team_state[team]

    def elo_expected(rating_a, rating_b):
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_elo(rating_a, rating_b, result_a, k=32):
        exp_a = elo_expected(rating_a, rating_b)
        new_a = rating_a + k * (result_a - exp_a)
        new_b = rating_b + k * ((1 - result_a) - (1 - exp_a))
        return new_a, new_b

    for idx, row in df.iterrows():
        ht, at = row['home_team'], row['away_team']
        season, div, d_str = row['season'], row['division'], row['date']
        
        hs = get_team_state(ht)
        as_st = get_team_state(at)
        
        is_home_promoted = (div == 'BL1' and hs['current_season_division'] == 'BL2')
        is_away_promoted = (div == 'BL1' and as_st['current_season_division'] == 'BL2')
        prev_season = str(int(season[:4]) - 1) + "-" + str(int(season[5:]) - 1).zfill(2)
        
        if is_home_promoted and prev_season in bl2_standings:
            st = bl2_standings[prev_season]
            if ht in st: hs['elo'] = 1350 + (st.index(ht) * 10)
        if is_away_promoted and prev_season in bl2_standings:
            st = bl2_standings[prev_season]
            if at in st: as_st['elo'] = 1350 + (st.index(at) * 10)
                
        hs['current_season_division'] = div
        as_st['current_season_division'] = div
        hs['last_date'] = d_str
        as_st['last_date'] = d_str
        
        r = row['result']
        hs['form_pts'].append(3 if r == 2 else 1 if r == 1 else 0)
        as_st['form_pts'].append(3 if r == 0 else 1 if r == 1 else 0)
        
        if row['home_xg'] is not None and not np.isnan(row['home_xg']):
            hs['xg_scored'].append(row['home_xg'])
            as_st['xga_conceded'].append(row['home_xg'])
        if row['away_xg'] is not None and not np.isnan(row['away_xg']):
            as_st['xg_scored'].append(row['away_xg'])
            hs['xga_conceded'].append(row['away_xg'])
            
        if div == 'BL2':
            hs['bl2_goals_scored'].append(row['home_goals'])
            as_st['bl2_goals_scored'].append(row['away_goals'])
            
        h2h_key = tuple(sorted([ht, at]))
        if h2h_key not in h2h_state:
            h2h_state[h2h_key] = []
        winner = ht if r == 2 else (as_st if r == 0 else 'Draw')
        h2h_state[h2h_key].append({'winner': winner})
        
        hs['home_played'] += 1
        as_st['away_played'] += 1
        if r == 2: hs['home_wins'] += 1
        if r == 0: as_st['away_wins'] += 1
        
        res_hom_val = 1.0 if r == 2 else 0.5 if r == 1 else 0.0
        new_h, new_a = update_elo(hs['elo'], as_st['elo'], res_hom_val)
        hs['elo'] = new_h
        as_st['elo'] = new_a
        
    return team_state, h2h_state

def apply_hierarchical_correction(p_raw):
    p_away_raw, p_draw_raw, p_home_raw = p_raw[:, 0], p_raw[:, 1], p_raw[:, 2]
    sum_ha = np.clip(p_home_raw + p_away_raw, a_min=1e-9, a_max=None)
    p_home_cond = p_home_raw / sum_ha
    p_away_cond = p_away_raw / sum_ha
    p_home_final = (1.0 - p_draw_raw) * p_home_cond
    p_away_final = (1.0 - p_draw_raw) * p_away_cond
    final_probs = np.column_stack([p_away_final, p_draw_raw, p_home_final])
    sums = final_probs.sum(axis=1, keepdims=True)
    return final_probs / sums

def fetch_upcoming_matches():
    if not API_KEY:
        print("FOOTBALL_DATA_API_KEY is not set.")
        return []
    
    headers = {"X-Auth-Token": API_KEY}
    url = "https://api.football-data.org/v4/competitions/2002/matches?status=SCHEDULED"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch matches: {response.status_code}")
        print(response.text)
        return []
    
    data = response.json()
    matches = data.get("matches", [])
    if not matches:
        return []
        
    next_matchday = matches[0].get("matchday")
    upcoming = [m for m in matches if m.get("matchday") == next_matchday]
    return upcoming

def main():
    print("Fetching upcoming fixtures using API...")
    upcoming_matches = fetch_upcoming_matches()
    if not upcoming_matches:
        print("No scheduled matches found or API key not set.")
        return

    print("Loading historical terminal states...")
    team_state, h2h_state = get_latest_states()

    features_list = []
    
    for match in upcoming_matches:
        d_str = match['utcDate'][:10] # YYYY-MM-DD
        ht_api = match['homeTeam']['name']
        at_api = match['awayTeam']['name']
        
        ht = map_team_name(ht_api)
        at = map_team_name(at_api)
        
        hs = team_state.get(ht, team_state.get('Bayern Munich', {})) # fallback
        as_st = team_state.get(at, team_state.get('Bayern Munich', {})) # fallback
        
        match_id = f"{d_str}_{ht.replace(' ', '')}_{at.replace(' ', '')}"
        
        f_row = {'match_id': match_id, 'date': d_str, 'home_team': ht, 'away_team': at}
        
        f_row['home_elo'] = hs.get('elo', 1500)
        f_row['away_elo'] = as_st.get('elo', 1500)
        f_row['elo_diff'] = f_row['home_elo'] - f_row['away_elo']
        
        fp_h = hs.get('form_pts', [])
        f_row['home_form_pts'] = sum(fp_h[-5:]) / min(5, len(fp_h)) if fp_h else 1.0
        fp_a = as_st.get('form_pts', [])
        f_row['away_form_pts'] = sum(fp_a[-5:]) / min(5, len(fp_a)) if fp_a else 1.0
        
        xg_h = hs.get('xg_scored', [])
        f_row['home_xg_avg5'] = sum(xg_h[-5:]) / min(5, len(xg_h)) if xg_h else 1.0
        xg_a = as_st.get('xg_scored', [])
        f_row['away_xg_avg5'] = sum(xg_a[-5:]) / min(5, len(xg_a)) if xg_a else 1.0
        
        xga_h = hs.get('xga_conceded', [])
        f_row['home_xga_avg5'] = sum(xga_h[-5:]) / min(5, len(xga_h)) if xga_h else 1.0
        xga_a = as_st.get('xga_conceded', [])
        f_row['away_xga_avg5'] = sum(xga_a[-5:]) / min(5, len(xga_a)) if xga_a else 1.0
        
        f_row['form_diff'] = f_row['home_form_pts'] - f_row['away_form_pts']
        f_row['xg_diff'] = f_row['home_xg_avg5'] - f_row['away_xg_avg5']
        f_row['xga_diff'] = f_row['home_xga_avg5'] - f_row['away_xga_avg5']
        
        f_row['home_strength'] = f_row['home_xg_avg5'] - f_row['home_xga_avg5']
        f_row['away_strength'] = f_row['away_xg_avg5'] - f_row['away_xga_avg5']
        f_row['strength_diff'] = f_row['home_strength'] - f_row['away_strength']
        
        h2h_key = tuple(sorted([ht, at]))
        h2h_hist = h2h_state.get(h2h_key, [])
        h_wins = sum(1 for match in h2h_hist[-5:] if match['winner'] == ht)
        d_wins = sum(1 for match in h2h_hist[-5:] if match['winner'] == 'Draw')
        n_meetings = len(h2h_hist)
        
        if n_meetings <= 1:
            f_row['h2h_home_wins'] = 0.43 * 5
            f_row['h2h_draws'] = 0.25 * 5 
            f_row['h2h_data_quality'] = 0 # none
        elif n_meetings <= 4:
            scale = n_meetings / 5.0
            f_row['h2h_home_wins'] = h_wins * scale
            f_row['h2h_draws'] = d_wins * scale
            f_row['h2h_data_quality'] = 1 # partial
        else:
            f_row['h2h_home_wins'] = h_wins
            f_row['h2h_draws'] = d_wins
            f_row['h2h_data_quality'] = 2 # full
            
        f_row['days_rest_home'] = (datetime.strptime(d_str, "%Y-%m-%d") - datetime.strptime(hs['last_date'], "%Y-%m-%d")).days if hs.get('last_date') else 7
        f_row['days_rest_away'] = (datetime.strptime(d_str, "%Y-%m-%d") - datetime.strptime(as_st['last_date'], "%Y-%m-%d")).days if as_st.get('last_date') else 7
        f_row['rest_diff'] = f_row['days_rest_home'] - f_row['days_rest_away']
            
        hp = hs.get('home_played', 0)
        f_row['home_is_home_record'] = hs.get('home_wins',0) / hp if hp > 0 else 0.43
        ap = as_st.get('away_played', 0)
        f_row['away_is_away_record'] = as_st.get('away_wins',0) / ap if ap > 0 else 0.25
        
        is_home_promoted = hs.get('current_season_division') == 'BL2'
        is_away_promoted = as_st.get('current_season_division') == 'BL2'
        f_row['is_home_promoted'] = int(is_home_promoted)
        f_row['is_away_promoted'] = int(is_away_promoted)
        f_row['home_bl2_goals_avg5'] = 0.0
        f_row['away_bl2_goals_avg5'] = 0.0
        f_row['promoted_stat_quality'] = 0 # goals=0, xg=1 => default to 0 for fallback

        
        features_list.append(f_row)
        
    df_new = pd.DataFrame(features_list)
    
    print("Loading models...")
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved')
    final_log_model = joblib.load(os.path.join(models_dir, 'final_log_model.joblib'))
    final_xgb_model = joblib.load(os.path.join(models_dir, 'final_xgb_model.joblib'))
    meta_model = joblib.load(os.path.join(models_dir, 'meta_model.joblib'))
    
    print("Generating predictions...")
    
    meta_cols = ['match_id', 'date', 'home_team', 'away_team']
    X_model = df_new.drop(columns=meta_cols)
    for col in X_model.columns:
        X_model[col] = pd.to_numeric(X_model[col], errors='coerce').fillna(0)
    
    # Ensure exact feature order for XGBoost if possible, but feature names should match
    
    elo_preds = predict_elo(df_new['elo_diff'])
    poisson_preds = predict_poisson(df_new['home_xg_avg5'], df_new['away_xg_avg5'], df_new['home_xga_avg5'], df_new['away_xga_avg5'])
    log_preds = final_log_model.predict_proba(select_logistic_features(X_model))
    xgb_preds = final_xgb_model.predict_proba(X_model)
    
    X_stack = np.hstack([elo_preds, log_preds, xgb_preds, poisson_preds])
    ensemble_raw_preds = meta_model.predict_proba(X_stack)
    ensemble_preds = apply_hierarchical_correction(ensemble_raw_preds)
    
    print("Upserting to Supabase...")
    for i, row in df_new.iterrows():
        match_id = row['match_id']
        date_str = pd.to_datetime(row['date']).isoformat()
        
        # Upsert match
        try:
            supabase.table('matches').upsert({
                'match_id': match_id,
                'date': date_str,
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'status': 'SCHEDULED'
            }).execute()
        except BaseException as e:
            print(f"Error upserting match {match_id}: {e}")
            
        def upsert_prediction(model_name, probs_away_draw_home, is_ens=False):
             # Probs shape: [away, draw, home] (labels: 0, 1, 2)
             prob_away, prob_draw, prob_home = float(probs_away_draw_home[0]), float(probs_away_draw_home[1]), float(probs_away_draw_home[2])
             confidence = max(prob_home, prob_draw, prob_away)
             try:
                 supabase.table('predictions').upsert({
                     'match_id': match_id,
                     'model_name': model_name,
                     'prob_home': prob_home,
                     'prob_draw': prob_draw,
                     'prob_away': prob_away,
                     'confidence': confidence
                 }, on_conflict='match_id,model_name').execute()
             except BaseException as e:
                 print(f"Error upserting prediction for {model_name} on {match_id}: {e}")
                 
        upsert_prediction('ELO', elo_preds[i])
        upsert_prediction('Poison', poisson_preds[i])
        upsert_prediction('LogReg', log_preds[i])
        upsert_prediction('XGBoost', xgb_preds[i])
        upsert_prediction('Ensemble', ensemble_preds[i])
        
    print("Done! Predictions pushed to Supabase successfully.")

if __name__ == '__main__':
    main()
