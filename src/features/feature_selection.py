import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestClassifier


def get_redundant_features(X, threshold=0.9):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    redundant = [col for col in upper.columns if any(upper[col] > threshold)]
    return redundant


def get_low_variance_features(X, threshold=0.01):
    var = X.var()
    low_var = var[var < threshold].index.tolist()
    return low_var


def select_features_by_importance(X, y, n_features=30):
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5, 
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)
    
    importance = pd.Series(rf.feature_importances_, index=X.columns)
    selected = importance.nlargest(n_features).index.tolist()
    return selected


def get_feature_selection_report(X, y=None):
    report = {
        'total_features': X.shape[1],
        'low_variance': get_low_variance_features(X),
        'redundant': get_redundant_features(X),
    }
    
    if y is not None:
        report['selected_by_importance'] = select_features_by_importance(X, y)
    
    return report


FEATURES_TO_DROP = [
    'ht_result_encoded',
    'odds_open_home', 'odds_open_draw', 'odds_open_away',
    'odds_close_home', 'odds_close_draw', 'odds_close_away',
    'implied_prob_open_home', 'implied_prob_open_draw', 'implied_prob_open_away',
    'implied_prob_close_home', 'implied_prob_close_draw', 'implied_prob_close_away',
    'odds_move_home', 'odds_move_abs_home', 'odds_move_draw', 'odds_move_abs_draw',
    'odds_move_away', 'odds_move_abs_away',
    'overround_open', 'overround_close',
    'Home_longitude', 'Away_longitude', 'Away_longitude.1',
    'Home_latitude',
    'home_momentum_5_v2', 'away_momentum_5_v2',
    'Home wins in the window', 'Total home win rate', 'Total home draw rate',
    'Total home loss rate', 'Home goals forward', 'Home goals against',
    'Home goal differential', 'Home goals forward per match',
    'Home goals against per match', 'Home goal differential per match',
    '_orig_idx', '_row_order',
]


CORE_FEATURES = [
    'half_time_lead',
    'home_team_goal_count_half_time',
    'away_team_goal_count_half_time',
    'total_goals_at_half_time',
    'home_team_first_half_cards',
    'away_team_first_half_cards',
    'odds_ft_home_team_win',
    'odds_ft_draw',
    'odds_ft_away_team_win',
    'odds_ft_over15',
    'odds_ft_over25',
    'odds_ft_over35',
    'Pre-Match PPG (Home)',
    'Pre-Match PPG (Away)',
    'home_ppg_momentum_5',
    'away_ppg_momentum_5',
    'Home Team Pre-Match xG',
    'Away Team Pre-Match xG',
    'average_goals_per_match_pre_match',
    'btts_percentage_pre_match',
    'over_15_percentage_pre_match',
    'over_25_percentage_pre_match',
    'over_35_percentage_pre_match',
    'over_15_HT_FHG_percentage_pre_match',
    'over_05_HT_FHG_percentage_pre_match',
    'over_15_2HG_percentage_pre_match',
    'average_corners_per_match_pre_match',
    'average_cards_per_match_pre_match',
    'home_match_density',
    'away_match_density',
    'home_recent_cards',
    'away_recent_cards',
    'home_recent_fouls',
    'away_recent_fouls',
    'ppg_differential',
    'xg_power_index',
    'travel_distance_km',
    'capacity',
]


def clean_features(X, mode='conservative'):
    X = X.copy()
    
    if mode == 'aggressive':
        existing_core = [f for f in CORE_FEATURES if f in X.columns]
        X = X[existing_core]
    elif mode == 'conservative':
        to_drop = [f for f in FEATURES_TO_DROP if f in X.columns]
        X = X.drop(columns=to_drop, errors='ignore')
    elif mode == 'minimal':
        to_drop = ['ht_result_encoded']
        to_drop = [f for f in to_drop if f in X.columns]
        X = X.drop(columns=to_drop, errors='ignore')
    
    return X
