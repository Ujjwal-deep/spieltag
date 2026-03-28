import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

def compute_features():
    db_path = 'data/processed/football.db'
    if not os.path.exists(db_path):
        print("Database not found. Aborting.")
        return
        
    conn = sqlite3.connect(db_path)
    df = pd.read_sql('SELECT * FROM matches ORDER BY date ASC', conn)
    
    if df.empty:
        print("Matches table is empty. Aborting features.")
        return
        
    # State tracking
    team_state = {}
    h2h_state = {}
    bl2_standings = {} # season -> list of (team, points) sorted
    
    # Pre-compute BL2 standings per season
    for season in df['season'].unique():
        s_df = df[(df['season'] == season) & (df['division'] == 'BL2')]
        pts = {}
        for _, row in s_df.iterrows():
            ht, at = row['home_team'], row['away_team']
            r = row['result']
            pts[ht] = pts.get(ht, 0) + (3 if r == 2 else 1 if r == 1 else 0)
            pts[at] = pts.get(at, 0) + (3 if r == 0 else 1 if r == 1 else 0)
        
        # Sort by points, lowest points is 0th index (bottom)
        sorted_teams = sorted(pts.items(), key=lambda x: x[1])
        bl2_standings[season] = [t[0] for t in sorted_teams]
        
    def get_team_state(team):
        if team not in team_state:
            team_state[team] = {
                'elo': 1500,
                'form_pts': [],
                'xg_scored': [],
                'xga_conceded': [],
                'goals_scored': [],
                'goals_conceded': [],
                'bl2_goals_scored': [], # specifically for BL2 fallback
                'home_wins': 0,
                'home_played': 0,
                'away_wins': 0,
                'away_played': 0,
                'last_date': None,
                'current_season_division': None
            }
        return team_state[team]
        
    def elo_expected(rating_a, rating_b):
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_elo(rating_a, rating_b, result_a, k=32):
        exp_a = elo_expected(rating_a, rating_b)
        new_a = rating_a + k * (result_a - exp_a)
        new_b = rating_b + k * ((1 - result_a) - (1 - exp_a))
        return new_a, new_b

    features = []
    
    # We only care about BL1 features ultimately, but we process all matches chronologically
    for idx, row in df.iterrows():
        ht, at = row['home_team'], row['away_team']
        season = row['season']
        div = row['division']
        d_str = row['date']
        
        hs = get_team_state(ht)
        as_st = get_team_state(at) # renamed to avoid async keyword issues
        
        # Detect promotion
        # If playing in BL1 but current state was BL2 or None, and it's not the first global season (2013-14)
        is_home_promoted = (div == 'BL1' and hs['current_season_division'] == 'BL2')
        is_away_promoted = (div == 'BL1' and as_st['current_season_division'] == 'BL2')
        
        prev_season = str(int(season[:4]) - 1) + "-" + str(int(season[5:]) - 1).zfill(2)
        
        # Cold start elo logic
        if is_home_promoted and prev_season in bl2_standings:
            st = bl2_standings[prev_season]
            if ht in st:
                pos = st.index(ht) # 0 is bottom
                hs['elo'] = 1350 + (pos * 10)
        
        if is_away_promoted and prev_season in bl2_standings:
            st = bl2_standings[prev_season]
            if at in st:
                pos = st.index(at)
                as_st['elo'] = 1350 + (pos * 10)
        
        # For output, we only save features if it's a BL1 match
        if div == 'BL1':
            f_row = {
                'match_id': row['match_id'],
                'date': d_str,
                'season': season,
                'home_team': ht,
                'away_team': at,
                'result': row['result']
            }
            
            f_row['home_elo'] = hs['elo']
            f_row['away_elo'] = as_st['elo']
            f_row['elo_diff'] = hs['elo'] - as_st['elo']
            
            f_row['home_form_pts'] = sum(hs['form_pts'][-5:]) / min(5, len(hs['form_pts'])) if hs['form_pts'] else 1.0 # default to 1
            f_row['away_form_pts'] = sum(as_st['form_pts'][-5:]) / min(5, len(as_st['form_pts'])) if as_st['form_pts'] else 1.0
            
            f_row['home_xg_avg5'] = sum(hs['xg_scored'][-5:]) / min(5, len(hs['xg_scored'])) if hs['xg_scored'] else 1.0
            f_row['away_xg_avg5'] = sum(as_st['xg_scored'][-5:]) / min(5, len(as_st['xg_scored'])) if as_st['xg_scored'] else 1.0
            
            f_row['home_xga_avg5'] = sum(hs['xga_conceded'][-5:]) / min(5, len(hs['xga_conceded'])) if hs['xga_conceded'] else 1.0
            f_row['away_xga_avg5'] = sum(as_st['xga_conceded'][-5:]) / min(5, len(as_st['xga_conceded'])) if as_st['xga_conceded'] else 1.0
            
            # DEFERRED: possession and shots features removed until a reliable
            # API source is confirmed. Columns exist in DB schema for future backfill.
            # Candidates to revisit: API-Football (100/day), statsbomb open data
            
            # H2H rules
            h2h_key = tuple(sorted([ht, at]))
            h2h_hist = h2h_state.get(h2h_key, [])
            
            # extract wins/draws rel to home team
            h_wins = sum(1 for match in h2h_hist[-5:] if match['winner'] == ht)
            d_wins = sum(1 for match in h2h_hist[-5:] if match['winner'] == 'Draw')
            n_meetings = len(h2h_hist)
            
            if n_meetings <= 1:
                f_row['h2h_home_wins'] = 0.43 * 5
                f_row['h2h_draws'] = 0.25 * 5 # approx bl1 draw rate
                f_row['h2h_data_quality'] = 'none'
            elif n_meetings <= 4:
                scale = n_meetings / 5.0
                f_row['h2h_home_wins'] = h_wins * scale
                f_row['h2h_draws'] = d_wins * scale
                f_row['h2h_data_quality'] = 'partial'
            else:
                f_row['h2h_home_wins'] = h_wins
                f_row['h2h_draws'] = d_wins
                f_row['h2h_data_quality'] = 'full'
            
            # Days rest
            if hs['last_date']:
                f_row['days_rest_home'] = (datetime.strptime(d_str, "%Y-%m-%d") - datetime.strptime(hs['last_date'], "%Y-%m-%d")).days
            else:
                f_row['days_rest_home'] = 7
                
            if as_st['last_date']:
                f_row['days_rest_away'] = (datetime.strptime(d_str, "%Y-%m-%d") - datetime.strptime(as_st['last_date'], "%Y-%m-%d")).days
            else:
                f_row['days_rest_away'] = 7
                
            # Home/Away records
            f_row['home_is_home_record'] = hs['home_wins'] / hs['home_played'] if hs['home_played'] > 0 else 0.43
            f_row['away_is_away_record'] = as_st['away_wins'] / as_st['away_played'] if as_st['away_played'] > 0 else 0.25 # approx away win
            
            f_row['is_home_promoted'] = is_home_promoted
            f_row['is_away_promoted'] = is_away_promoted
            
            # BL2 fallback logic: using goals instead of xG
            if is_home_promoted and hs['bl2_goals_scored']:
                f_row['home_bl2_goals_avg5'] = (sum(hs['bl2_goals_scored'][-5:]) / min(5, len(hs['bl2_goals_scored']))) * 0.85
            else:
                f_row['home_bl2_goals_avg5'] = None
                
            if is_away_promoted and as_st['bl2_goals_scored']:
                f_row['away_bl2_goals_avg5'] = (sum(as_st['bl2_goals_scored'][-5:]) / min(5, len(as_st['bl2_goals_scored']))) * 0.85
            else:
                f_row['away_bl2_goals_avg5'] = None
                
            f_row['promoted_stat_quality'] = 'goals' if is_home_promoted or is_away_promoted else 'xg'
            
            features.append(f_row)

        # Update State After Match! (No Leakage)
        hs['current_season_division'] = div
        as_st['current_season_division'] = div
        
        hs['last_date'] = d_str
        as_st['last_date'] = d_str
        
        r = row['result'] # 2=home win, 1=draw, 0=away
        
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
        winner = ht if r == 2 else as_st if r == 0 else 'Draw'
        h2h_state[h2h_key].append({'winner': winner})
        
        hs['home_played'] += 1
        as_st['away_played'] += 1
        if r == 2: hs['home_wins'] += 1
        if r == 0: as_st['away_wins'] += 1
        
        # ELO Update
        res_hom_val = 1.0 if r == 2 else 0.5 if r == 1 else 0.0
        new_h, new_a = update_elo(hs['elo'], as_st['elo'], res_hom_val)
        hs['elo'] = new_h
        as_st['elo'] = new_a
        
    df_feat = pd.DataFrame(features)
    
    # Save
    df_feat.to_csv('data/processed/features.csv', index=False)
    
    print("\n" + "="*50)
    print("FEATURE PIPELINE COMPLETE")
    print("Final Feature List:")
    for col in df_feat.columns:
        print(f" - {col}")
        
    print("\nDATA QUALITY REPORT")
    print(f"Total BL1 Match Rows: {len(df_feat)}")
    goals_fb = len(df_feat[df_feat['promoted_stat_quality'] == 'goals'])
    print(f"Promoted rows utilizing goals fallback: {goals_fb}")
    print(f"Standard rows utilizing xG: {len(df_feat) - goals_fb}")
    print("="*50 + "\n")
    
if __name__ == '__main__':
    compute_features()
