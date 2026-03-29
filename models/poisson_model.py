import numpy as np
import pandas as pd
from scipy.stats import poisson

def predict_poisson(home_xg, away_xg, home_xga, away_xga):
    """
    Computes Poisson distributions for match outcomes based on team xG and xGA capabilities.
    Expects pandas Series or numpy arrays of matching length.
    Returns np.array of shape (N, 3): [P_away, P_draw, P_home]
    """
    lambda_home = (home_xg + away_xga) / 2.0
    lambda_away = (away_xg + home_xga) / 2.0
    
    # Initialize probabilities arrays
    probs = np.zeros((len(lambda_home), 3))
    
    # Convert to arrays if they are pandas series for fast indexing
    lh = np.array(lambda_home)
    la = np.array(lambda_away)
    
    for i in range(len(lh)):
        p_home_win = 0.0
        p_draw = 0.0
        p_away_win = 0.0
        
        # Loop goals from 0 to 5
        for hg in range(6):
            prob_hg = poisson.pmf(hg, lh[i])
            for ag in range(6):
                prob_ag = poisson.pmf(ag, la[i])
                joint_prob = prob_hg * prob_ag
                
                if hg > ag:
                    p_home_win += joint_prob
                elif hg == ag:
                    p_draw += joint_prob
                else:
                    p_away_win += joint_prob
                    
        # Normalize in case some probability is left in tails > 5 goals
        total = p_home_win + p_draw + p_away_win
        if total > 0:
            probs[i] = [p_away_win / total, p_draw / total, p_home_win / total]
        else:
            probs[i] = [0.3, 0.4, 0.3] # Fallback if perfectly zero
            
    return probs
