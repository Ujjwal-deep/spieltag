import numpy as np
import pandas as pd

def predict_elo(elo_diff):
    """
    Computes exact probabilities mapped explicitly via:
    E = 1 / (1 + 10**(-elo_diff / 400))
    P_home = E * 0.85
    P_away = (1 - E) * 0.85
    P_draw = 0.15
    """
    lo_diff_arr = np.array(elo_diff)
    
    E = 1.0 / (1.0 + 10.0 ** (-lo_diff_arr / 400.0))
    
    p_home = E * 0.85
    p_away = (1.0 - E) * 0.85
    p_draw = np.full_like(E, 0.15)
    
    # Normalize strictly to ensure sum=1
    sums = p_home + p_draw + p_away
    
    return np.column_stack([p_away / sums, p_draw / sums, p_home / sums])
