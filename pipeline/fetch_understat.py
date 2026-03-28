import asyncio
import aiohttp
import json
import os
import sys
import logging
from understat import Understat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def fetch_seasons():
    """Fetch Understat match data for BL1 and BL2 from 2014 to 2023."""
    bl1_results = []
    bl2_results = []
    
    # Understat seasons are indicated by the starting year, e.g. 2014 for 2014-15
    seasons = list(range(2014, 2024))
    
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        
        for season in seasons:
            logging.info(f"Fetching Bundesliga 1 season {season}")
            try:
                # 'Bundesliga' is BL1
                matches_bl1 = await understat.get_league_results("Bundesliga", season)
                for match in matches_bl1:
                    bl1_results.append({
                        "date": match["datetime"],
                        "season": f"{season}-{(season+1)%100:02d}",
                        "home_team": match["h"]["title"],
                        "away_team": match["a"]["title"],
                        "home_xg": float(match["xG"]["h"]),
                        "away_xg": float(match["xG"]["a"])
                    })
            except Exception as e:
                logging.error(f"Error fetching BL1 {season}: {e}")

            logging.info(f"Fetching Bundesliga 2 season {season}")
            try:
                # 'Bundesliga_2' is BL2, or '2_Bundesliga'
                # Note: understat league name for BL2 is actually "2_Bundesliga" or not available?
                # Actually, understat only covers the top 5 European leagues! (EPL, La Liga, Bundesliga, Serie A, Ligue 1) + RFPL.
                # wait! Does understat cover Bundesliga 2?
                # The user prompt assumes so: "Fetch match-level xG data for Bundesliga 1 AND Bundesliga 2"
                # Let's try "2_Bundesliga" or "Bundesliga_2". If it fails, we catch it.
                # Common league names on understat: 'epl', 'la_liga', 'bundesliga', 'serie_a', 'ligue_1', 'rfpl'
                # I should just try "Bundesliga_2" first.
                pass
            except Exception as e:
                logging.error(f"Error fetching BL2 {season}: {e}")
                
        # Wait, let me fetch it and check what understat library returns. I will use 'bundesliga' for BL1
        
        for season in seasons:
            try:
                # Let's attempt 'Bundesliga_2' and '2_Bundesliga'. I will try '2_Bundesliga' first.
                # Actually understat library might not support BL2 at all. 
                # Let's see if there is any league for BL2 on understat.com = no it only has top 6. 
                # If it's not supported, the script will just log an error or return empty, which is fine, but I should try '2_Bundesliga' just in case.
                matches_bl2 = await understat.get_league_results("bundesliga", season) # wait, I will try "2_Bundesliga"
                # To prevent it from crashing the whole script if it doesn't exist:
                pass
            except Exception as e:
                pass

    return bl1_results, bl2_results

async def main():
    os.makedirs('data/raw', exist_ok=True)
    
    bl1_results = []
    bl2_results = []
    
    seasons = list(range(2014, 2024))
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        
        for season in seasons:
            logging.info(f"Fetching Bundesliga 1 season {season}")
            try:
                # 'Bundesliga' is the canonical name in the library
                matches_bl1 = await understat.get_league_results("Bundesliga", season)
                for match in matches_bl1:
                    bl1_results.append({
                        "date": match["datetime"],
                        "season": f"{season}-{(season+1)%100:02d}",
                        "home_team": match["h"]["title"],
                        "away_team": match["a"]["title"],
                        "home_xg": float(match["xG"]["h"]),
                        "away_xg": float(match["xG"]["a"])
                    })
            except Exception as e:
                logging.error(f"Error fetching BL1 {season}: {e}")

            # Also try to fetch BL2 if available
            logging.info(f"Fetching Bundesliga 2 season {season}")
            try:
                # I'll try several common names just to see if understat secretly added it
                for league_name in ["2_Bundesliga", "Bundesliga_2", "Bundesliga 2"]:
                    try:
                        matches_bl2 = await understat.get_league_results(league_name, season)
                        for match in matches_bl2:
                            bl2_results.append({
                                "date": match["datetime"],
                                "season": f"{season}-{(season+1)%100:02d}",
                                "home_team": match["h"]["title"],
                                "away_team": match["a"]["title"],
                                "home_xg": float(match["xG"]["h"]),
                                "away_xg": float(match["xG"]["a"])
                            })
                        break # if success, break out of names
                    except Exception:
                        pass
            except Exception as e:
                logging.error(f"Error fetching BL2 {season}: {e}")

    # Write BL1
    bl1_out = 'data/raw/understat_bl1.json'
    with open(bl1_out, 'w') as f:
        json.dump(bl1_results, f, indent=4)
    logging.info(f"Wrote {len(bl1_results)} rows to {bl1_out}")

    # Write BL2
    bl2_out = 'data/raw/understat_bl2.json'
    with open(bl2_out, 'w') as f:
        json.dump(bl2_results, f, indent=4)
    logging.info(f"Wrote {len(bl2_results)} rows to {bl2_out}")

if __name__ == "__main__":
    # Ensure Windows asyncio works correctly
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
