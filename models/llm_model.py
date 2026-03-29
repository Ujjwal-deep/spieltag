import os
import json
import logging
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics import accuracy_score, log_loss

from utils import load_and_preprocess_data, SAVED_MODELS_PATH

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load .env
load_dotenv()

def compute_brier_score_multiclass(y_true, y_prob):
    # Ensure matching array shapes
    y_true_onehot = np.zeros((len(y_true), 3))
    y_true_onehot[np.arange(len(y_true)), y_true] = 1.0
    brier = np.mean(np.sum((y_prob - y_true_onehot)**2, axis=1))
    return float(brier)

def generate_prompt(meta_row, feat_row):
    prompt = f"""Bundesliga Match Preview:
 Home team: {meta_row['home_team']} | ELO: {feat_row['home_elo']:.1f} | Form (pts/game last 5): {feat_row['home_form_pts']:.1f}
 Away team: {meta_row['away_team']} | ELO: {feat_row['away_elo']:.1f} | Form (pts/game last 5): {feat_row['away_form_pts']:.1f}
 Home xG avg (last 5): {feat_row['home_xg_avg5']:.2f} | Home xGA avg (last 5): {feat_row['home_xga_avg5']:.2f}
 Away xG avg (last 5): {feat_row['away_xg_avg5']:.2f} | Away xGA avg (last 5): {feat_row['away_xga_avg5']:.2f}
 H2H (last meetings): Home wins: {feat_row['h2h_home_wins']:.1f} | Draws: {feat_row['h2h_draws']:.1f}
 Days rest — Home: {feat_row['days_rest_home']} | Away: {feat_row['days_rest_away']}
 Home team promoted this season: {bool(feat_row['is_home_promoted'])}
 Away team promoted this season: {bool(feat_row['is_away_promoted'])}
 Predict the outcome probabilities."""
    return prompt

def get_llm_prediction(client, sys_prompt, user_prompt, match_id, cache):
    if match_id in cache:
        return cache[match_id]
        
    fallback = {"home_win": 0.45, "draw": 0.25, "away_win": 0.30}
    
    for attempt in range(3):
        time.sleep(0.5) # Rate limit
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                max_tokens=80,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            parsed = json.loads(content)
            
            p_home = float(parsed.get("home_win", 0))
            p_draw = float(parsed.get("draw", 0))
            p_away = float(parsed.get("away_win", 0))
            
            total = p_home + p_draw + p_away
            if total == 0:
                raise ValueError("Probabilities sum to 0")
                
            # Normalize just in case, but check tolerance
            if abs(total - 1.0) > 0.02:
                logger.warning(f"Sum {total} outside tolerance for {match_id}. Retrying...")
                continue
                
            p_home, p_draw, p_away = p_home/total, p_draw/total, p_away/total
            result = {"home_win": p_home, "draw": p_draw, "away_win": p_away}
            cache[match_id] = result
            return result
            
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {match_id}: {type(e).__name__}")
            if "429" in str(e) or "Too Many Requests" in str(e):
                # Don't retry immediately if we are rate limited, break out to fallback
                break
            
    logger.error(f"Failed to get valid response for {match_id}. Using fallback.")
    cache[match_id] = fallback
    return fallback

if __name__ == "__main__":
    logger.info("Initializing LLM Reasoning Model...")
    
    client = Groq(max_retries=0) # Picks up GROQ_API_KEY from environment
    
    _, (X_val, y_val, _), (X_test, y_test, _), meta = load_and_preprocess_data()
    
    sys_prompt = """You are a Bundesliga football analyst. You will be given pre-match statistics for an upcoming match and must predict the outcome probabilities. Respond ONLY with a valid JSON object in exactly this format: {"home_win": float, "draw": float, "away_win": float}
The three values must sum to 1.0. Do not include any explanation."""

    cache_file = os.path.join(SAVED_MODELS_PATH, "llm_cache.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    def process_split(X_split, y_split, name="Val"):
        logger.info(f"Processing {name} set ({len(X_split)} matches)...")
        probs = []
        for idx in X_split.index:
            m_row = meta.loc[idx]
            f_row = X_split.loc[idx]
            prompt = generate_prompt(m_row, f_row)
            
            res = get_llm_prediction(client, sys_prompt, prompt, m_row['match_id'], cache)
            probs.append([res['away_win'], res['draw'], res['home_win']])
            
        # Save cache iteratively
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=4)
            
        return np.array(probs)

    # Process Val
    val_probs = process_split(X_val, y_val, "Validation")
    
    # Evaluate Val
    y_pred = np.argmax(val_probs, axis=1)
    acc = accuracy_score(y_val, y_pred)
    ll = log_loss(y_val, val_probs)
    brier = compute_brier_score_multiclass(y_val.values if hasattr(y_val, 'values') else y_val, val_probs)

    logger.info("--- LLM Validation Results ---")
    logger.info(f"Accuracy: {acc:.2%}")
    logger.info(f"Log Loss: {ll:.4f}")
    logger.info(f"Brier Score: {brier:.4f}")
    
    if acc < 0.45:
        logger.warning("LLM Accuracy is very weak (< 45%).")
        # For LLM we don't re-run or retrain, just log
        with open(os.path.join(SAVED_MODELS_PATH, "failure_report.txt"), "a") as f:
            f.write(f"LLM failed to reach 45% val accuracy. Final: {acc:.2%}\n")

    metrics = {
        "accuracy": acc,
        "log_loss": ll,
        "brier_score": brier
    }
    with open(os.path.join(SAVED_MODELS_PATH, "llm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Pre-process Test (backtest) set so it's in the cache
    _ = process_split(X_test, y_test, "Test")
    
    logger.info("LLM processing complete.")
