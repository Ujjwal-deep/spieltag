import os
import glob
import json
import sqlite3
import pandas as pd
import numpy as np
import difflib
from sqlalchemy import create_engine

def get_season_from_filename(filename):
    # e.g. "D1 2013-14.csv" -> "2013-14"
    return os.path.basename(filename).split(' ')[1].replace('.csv', '')

def load_csvs(pattern):
    dfs = []
    files = glob.glob(pattern)
    for f in files:
        df = pd.read_csv(f, encoding='latin1')
        df['season'] = get_season_from_filename(f)
        df['division'] = 'BL1' if 'D1' in f else 'BL2'
        df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def parse_date(date_series):
    return pd.to_datetime(date_series, dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')

def main():
    db_path = 'data/processed/football.db'
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)
        
    engine = create_engine(f'sqlite:///{db_path}')
    
    df_bl1 = load_csvs('data/raw/D1*.csv')
    df_bl2 = load_csvs('data/raw/D2*.csv')
    
    df_matches = pd.concat([df_bl1, df_bl2], ignore_index=True)
    if df_matches.empty:
        print("No CSV data found.")
        return

    df_matches['date'] = parse_date(df_matches['Date'])
    df_matches = df_matches.rename(columns={
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'FTHG': 'home_goals',
        'FTAG': 'away_goals'
    })
    
    conditions = [
        df_matches['FTR'] == 'H',
        df_matches['FTR'] == 'D',
        df_matches['FTR'] == 'A'
    ]
    df_matches['result'] = np.select(conditions, [2, 1, 0], default=-1)
    
    df_matches['home_xg'] = None
    df_matches['away_xg'] = None
    df_matches['home_possession'] = None
    df_matches['away_possession'] = None
    df_matches['home_shots_ot'] = None
    df_matches['away_shots_ot'] = None
    df_matches['source'] = 'csv'
    
    understat_path = 'data/raw/understat_bl1.json'
    if os.path.exists(understat_path):
        with open(understat_path, 'r') as f:
            u_data = json.load(f)
        df_u = pd.DataFrame(u_data)
        df_u['date'] = pd.to_datetime(df_u['date']).dt.strftime('%Y-%m-%d')
        
        match_cache = {}
        unmatched = []
        valid_teams_per_season = df_matches[df_matches['division'] == 'BL1'].groupby('season')['home_team'].unique().to_dict()
        
        for season in df_u['season'].unique():
            if season not in valid_teams_per_season:
                continue
            valid_teams = valid_teams_per_season[season]
            season_u = df_u[df_u['season'] == season]
            u_teams = pd.concat([season_u['home_team'], season_u['away_team']]).unique()
            
            for ut in u_teams:
                normalization_dict = {
                    "Borussia M.Gladbach": "Borussia Mönchengladbach"
                }
                search_name = normalization_dict.get(ut, ut)
                matches = difflib.get_close_matches(search_name, valid_teams, n=1, cutoff=0.5)
                if matches:
                    match_cache[(season, ut)] = matches[0]
                else:
                    unmatched.append(f'[UNMATCHED] source=understat  name="{ut}"  season="{season}"')
                    match_cache[(season, ut)] = None
                    
        df_u['home_team_csv'] = df_u.apply(lambda r: match_cache.get((r['season'], r['home_team'])), axis=1)
        df_u['away_team_csv'] = df_u.apply(lambda r: match_cache.get((r['season'], r['away_team'])), axis=1)
        
        xg_dict = {}
        for _, r in df_u.dropna(subset=['home_team_csv', 'away_team_csv']).iterrows():
            xg_dict[(r['date'], r['home_team_csv'], r['away_team_csv'])] = (r['home_xg'], r['away_xg'])
            
        def get_xg(row):
            if row['division'] != 'BL1': return pd.Series([None, None, row['source']])
            key = (row['date'], row['home_team'], row['away_team'])
            if key in xg_dict:
                return pd.Series([xg_dict[key][0], xg_dict[key][1], row['source'] + ',understat'])
            else:
                return pd.Series([None, None, row['source'] + ',understat_unmatched'])
                
        df_matches[['home_xg', 'away_xg', 'source']] = df_matches.apply(get_xg, axis=1)
        
        for u in set(unmatched):
            print(u)
            
    df_matches['is_home_promoted'] = None
    df_matches['is_away_promoted'] = None
    df_matches['h2h_data_quality'] = None
    
    df_matches['match_id'] = df_matches['date'] + '_' + df_matches['home_team'].str.replace(' ', '') + '_' + df_matches['away_team'].str.replace(' ', '')
    
    final_cols = ['match_id', 'date', 'season', 'division', 'home_team', 'away_team', 'home_goals', 'away_goals', 
                  'result', 'home_xg', 'away_xg', 'home_possession', 'away_possession', 'home_shots_ot', 'away_shots_ot', 
                  'source', 'is_home_promoted', 'is_away_promoted', 'h2h_data_quality']
                  
    df_matches = df_matches[final_cols]
    df_matches = df_matches.drop_duplicates(subset=['match_id'])
    
    df_matches.to_sql('matches', engine, if_exists='replace', index=False)
    
    print("Ingestion Summary:")
    print(f"Total rows: {len(df_matches)}")
    print(f"Rows with xG: {df_matches['home_xg'].notna().sum()}")
    print(f"Rows with possession: {df_matches['home_possession'].notna().sum()}")
    
if __name__ == '__main__':
    main()
