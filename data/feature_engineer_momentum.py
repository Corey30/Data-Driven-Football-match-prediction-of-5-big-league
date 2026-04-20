import os
from pathlib import Path
from math import isclose

import numpy as np
import pandas as pd


# =========================================================
# 0. 配置
# =========================================================
# 默认读取脚本同目录下的文件；你也可以直接改成绝对路径
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "span3_with_momentum_v2.csv"
OUTPUT_PATH = BASE_DIR / "span3_with_momentum_v2_momentum_featured.csv"

RUN_TESTS = True
TEST_TOL = 1e-9

ELO_K = 20.0
ELO_HOME_ADV = 65.0
FATIGUE_LAMBDA = 0.1


# =========================================================
# 1. 工具函数
# =========================================================
def parse_datetime_mixed(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, format="mixed", errors="coerce", utc=True)
    except TypeError:
        return pd.to_datetime(series, errors="coerce", utc=True)


def values_equal(a, b, tol=1e-9):
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) != pd.isna(b):
        return False
    return isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def ensure_columns(df: pd.DataFrame, required_cols: list[str], df_name: str = "DataFrame"):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} 缺少必要列: {missing}")


# =========================================================
# 2. 读取与预处理
# =========================================================
def load_and_prepare_data(input_path: Path) -> pd.DataFrame:
    if not Path(input_path).exists():
        raise FileNotFoundError(
            f"找不到输入文件: {input_path}\n"
            f"请确认 span3_with_momentum_v2.csv 与脚本在同一目录，或把 INPUT_PATH 改成绝对路径。"
        )

    df = pd.read_csv(input_path)

    required_cols = [
        "match_id",
        "timestamp",
        "date_GMT",
        "home_team_name",
        "away_team_name",
        "target_match_result",
        "target_home_team_goal_count",
        "target_away_team_goal_count",
        "Pre-Match PPG (Home)",
        "Pre-Match PPG (Away)",
        "Home Team Pre-Match xG",
        "Away Team Pre-Match xG",
        "odds_move_home",
        "odds_move_abs_home",
        "implied_prob_shift_home",
        "odds_move_away",
        "odds_move_abs_away",
        "implied_prob_shift_away",
        "travel_distance_km",
    ]
    ensure_columns(df, required_cols, "原始CSV")

    df["date_GMT"] = parse_datetime_mixed(df["date_GMT"])
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["_orig_idx"] = np.arange(len(df))

    sort_cols = [c for c in ["date_GMT", "timestamp", "_orig_idx"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df["_row_order"] = np.arange(len(df))

    return df


# =========================================================
# 3. 比赛级基础列
# =========================================================
def add_match_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 结果 points
    out["_home_points"] = np.select(
        [
            out["target_home_team_goal_count"] > out["target_away_team_goal_count"],
            out["target_home_team_goal_count"] == out["target_away_team_goal_count"],
            out["target_home_team_goal_count"] < out["target_away_team_goal_count"],
        ],
        [3.0, 1.0, 0.0],
        default=np.nan,
    )

    out["_away_points"] = np.select(
        [
            out["target_away_team_goal_count"] > out["target_home_team_goal_count"],
            out["target_away_team_goal_count"] == out["target_home_team_goal_count"],
            out["target_away_team_goal_count"] < out["target_home_team_goal_count"],
        ],
        [3.0, 1.0, 0.0],
        default=np.nan,
    )

    # 净胜球
    out["_home_goal_diff"] = pd.to_numeric(out["target_home_team_goal_count"], errors="coerce") - pd.to_numeric(
        out["target_away_team_goal_count"], errors="coerce"
    )
    out["_away_goal_diff"] = -out["_home_goal_diff"]

    # xG 代理版：赛前 xG 差
    out["_home_xg_diff_proxy"] = pd.to_numeric(out["Home Team Pre-Match xG"], errors="coerce") - pd.to_numeric(
        out["Away Team Pre-Match xG"], errors="coerce"
    )
    out["_away_xg_diff_proxy"] = -out["_home_xg_diff_proxy"]

    # 赔率移动
    out["_home_odds_move_for"] = pd.to_numeric(out["odds_move_home"], errors="coerce")
    out["_away_odds_move_for"] = pd.to_numeric(out["odds_move_away"], errors="coerce")

    out["_home_odds_move_abs_for"] = pd.to_numeric(out["odds_move_abs_home"], errors="coerce")
    out["_away_odds_move_abs_for"] = pd.to_numeric(out["odds_move_abs_away"], errors="coerce")

    out["_home_implied_prob_shift_for"] = pd.to_numeric(out["implied_prob_shift_home"], errors="coerce")
    out["_away_implied_prob_shift_for"] = pd.to_numeric(out["implied_prob_shift_away"], errors="coerce")

    # 对手赛前 PPG
    out["_home_opponent_pre_match_ppg"] = pd.to_numeric(out["Pre-Match PPG (Away)"], errors="coerce")
    out["_away_opponent_pre_match_ppg"] = pd.to_numeric(out["Pre-Match PPG (Home)"], errors="coerce")

    # 旅行距离
    out["_home_travel_km_team"] = 0.0
    out["_away_travel_km_team"] = pd.to_numeric(out["travel_distance_km"], errors="coerce")

    return out


# =========================================================
# 4. 严格赛前 Elo
# =========================================================
def add_pre_match_elo(df: pd.DataFrame, k_factor: float = 20.0, home_adv: float = 65.0) -> pd.DataFrame:
    out = df.copy()

    out["home_elo_pre"] = np.nan
    out["away_elo_pre"] = np.nan
    out["elo_diff_pre"] = np.nan
    out["elo_home_win_prob_pre"] = np.nan

    ratings = {}
    order = out.sort_values(["date_GMT", "timestamp", "_orig_idx"]).index

    for idx in order:
        row = out.loc[idx]
        home_team = str(row["home_team_name"])
        away_team = str(row["away_team_name"])

        home_elo = ratings.get(home_team, 1500.0)
        away_elo = ratings.get(away_team, 1500.0)

        expected_home = 1.0 / (1.0 + 10 ** ((away_elo - (home_elo + home_adv)) / 400.0))

        out.at[idx, "home_elo_pre"] = home_elo
        out.at[idx, "away_elo_pre"] = away_elo
        out.at[idx, "elo_diff_pre"] = home_elo - away_elo
        out.at[idx, "elo_home_win_prob_pre"] = expected_home

        hg = float(row["target_home_team_goal_count"])
        ag = float(row["target_away_team_goal_count"])

        if hg > ag:
            score_home = 1.0
        elif hg == ag:
            score_home = 0.5
        else:
            score_home = 0.0

        ratings[home_team] = home_elo + k_factor * (score_home - expected_home)
        ratings[away_team] = away_elo + k_factor * ((1.0 - score_home) - (1.0 - expected_home))

    return out


# =========================================================
# 5. 转成长表
# =========================================================
def build_long_table(df: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "match_id": df["match_id"],
            "timestamp": df["timestamp"],
            "date_GMT": df["date_GMT"],
            "_row_order": df["_row_order"],
            "team": df["home_team_name"],
            "opponent": df["away_team_name"],
            "is_home": 1,
            "points": df["_home_points"],
            "goal_diff": df["_home_goal_diff"],
            "xg_diff_proxy": df["_home_xg_diff_proxy"],
            "odds_move_for": df["_home_odds_move_for"],
            "odds_move_abs_for": df["_home_odds_move_abs_for"],
            "implied_prob_shift_for": df["_home_implied_prob_shift_for"],
            "travel_km_team": df["_home_travel_km_team"],
            "opponent_pre_match_ppg": df["_home_opponent_pre_match_ppg"],
            "pre_match_elo": df["home_elo_pre"],
            "opponent_pre_match_elo": df["away_elo_pre"],
        }
    )

    away = pd.DataFrame(
        {
            "match_id": df["match_id"],
            "timestamp": df["timestamp"],
            "date_GMT": df["date_GMT"],
            "_row_order": df["_row_order"],
            "team": df["away_team_name"],
            "opponent": df["home_team_name"],
            "is_home": 0,
            "points": df["_away_points"],
            "goal_diff": df["_away_goal_diff"],
            "xg_diff_proxy": df["_away_xg_diff_proxy"],
            "odds_move_for": df["_away_odds_move_for"],
            "odds_move_abs_for": df["_away_odds_move_abs_for"],
            "implied_prob_shift_for": df["_away_implied_prob_shift_for"],
            "travel_km_team": df["_away_travel_km_team"],
            "opponent_pre_match_ppg": df["_away_opponent_pre_match_ppg"],
            "pre_match_elo": df["away_elo_pre"],
            "opponent_pre_match_elo": df["home_elo_pre"],
        }
    )

    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values(["date_GMT", "timestamp", "_row_order", "team", "is_home"]).reset_index(drop=True)
    return long_df


# =========================================================
# 6. 滚动 / EWM 工具
# =========================================================
def weighted_shifted_rolling(series: pd.Series, window: int) -> pd.Series:
    """
    旧比赛权重小，最近比赛权重大。
    完整窗口 [old, ..., recent] 对应权重 [1, 2, ..., window]
    """
    base_weights = np.arange(1, window + 1, dtype=float)

    def _func(arr: np.ndarray) -> float:
        valid = ~np.isnan(arr)
        if valid.sum() == 0:
            return np.nan
        vals = arr[valid]
        w = base_weights[-len(vals):]
        return float(np.dot(vals, w) / w.sum())

    return series.shift(1).rolling(window=window, min_periods=1).apply(_func, raw=True)


def add_group_weighted(df: pd.DataFrame, group_cols: list[str], source_col: str, window: int, out_col: str) -> pd.DataFrame:
    series = df.groupby(group_cols, group_keys=False)[source_col].apply(lambda s: weighted_shifted_rolling(s, window))
    df[out_col] = series.sort_index()
    return df


def add_group_ewm(df: pd.DataFrame, group_cols: list[str], source_col: str, span: int, out_col: str) -> pd.DataFrame:
    series = df.groupby(group_cols, group_keys=False)[source_col].apply(
        lambda s: s.shift(1).ewm(span=span, adjust=False, min_periods=1).mean()
    )
    df[out_col] = series.sort_index()
    return df


def add_group_std(df: pd.DataFrame, group_cols: list[str], source_col: str, window: int, out_col: str) -> pd.DataFrame:
    series = df.groupby(group_cols, group_keys=False)[source_col].apply(
        lambda s: s.shift(1).rolling(window=window, min_periods=2).std(ddof=0)
    )
    df[out_col] = series.sort_index()
    return df


def add_venue_weighted(df: pd.DataFrame, value_col: str, window: int, out_col: str) -> pd.DataFrame:
    df[out_col] = np.nan
    for venue_flag in (1, 0):
        mask = df["is_home"].eq(venue_flag)
        sub = df.loc[mask].copy()
        series = sub.groupby(["team"], group_keys=False)[value_col].apply(lambda s: weighted_shifted_rolling(s, window))
        sub[out_col] = series.sort_index()
        df.loc[sub.index, out_col] = sub[out_col]
    return df


# =========================================================
# 7. 长表特征工程
# =========================================================
def build_long_features(long_df: pd.DataFrame) -> pd.DataFrame:
    out = long_df.copy()

    for k in (3, 5, 8):
        out = add_group_weighted(out, ["team"], "points", k, f"form_pts_{k}")

    out = add_group_weighted(out, ["team"], "goal_diff", 5, "goal_diff_5")
    out = add_group_weighted(out, ["team"], "xg_diff_proxy", 5, "xg_diff_5_proxy")

    out["mom_slope"] = out["form_pts_3"] - out["form_pts_5"]
    out = add_group_std(out, ["team"], "goal_diff", 5, "mom_volatility_5")

    out["rest_days"] = out.groupby(["team"])["date_GMT"].diff().dt.total_seconds().div(86400.0)
    out["match_density"] = 1.0 / out["rest_days"].clip(lower=1.0)
    out["fatigue_adjusted_form"] = out["form_pts_5"] - FATIGUE_LAMBDA * out["match_density"]
    out["travel_fatigue_score"] = out["match_density"] + (pd.to_numeric(out["travel_km_team"], errors="coerce") / 1000.0)

    out = add_group_weighted(out, ["team"], "odds_move_for", 3, "odds_move_for_mom_3")
    out = add_group_weighted(out, ["team"], "odds_move_for", 5, "odds_move_for_mom_5")
    out = add_group_weighted(out, ["team"], "odds_move_abs_for", 5, "odds_move_abs_for_mom_5")
    out = add_group_weighted(out, ["team"], "implied_prob_shift_for", 5, "implied_prob_shift_for_mom_5")

    out = add_group_ewm(out, ["team"], "odds_move_for", 5, "odds_move_for_ewm_5")
    out = add_group_ewm(out, ["team"], "implied_prob_shift_for", 5, "implied_prob_shift_for_ewm_5")

    out = add_group_weighted(out, ["team"], "travel_km_team", 5, "travel_km_mom_5")
    out = add_group_ewm(out, ["team"], "travel_km_team", 5, "travel_km_ewm_5")

    out = add_group_weighted(out, ["team"], "match_density", 5, "match_density_mom_5")
    out = add_group_ewm(out, ["team"], "match_density", 5, "match_density_ewm_5")

    out = add_group_weighted(out, ["team"], "travel_fatigue_score", 5, "travel_fatigue_mom_5")

    out["opp_strength_weight"] = out["opponent_pre_match_ppg"] / 3.0
    out["opp_adj_points"] = out["points"] * out["opp_strength_weight"]
    out = add_group_weighted(out, ["team"], "opp_adj_points", 5, "opp_adj_form_5")

    out["opp_strength_weight_elo"] = 1.0 / (1.0 + 10 ** ((1500.0 - out["opponent_pre_match_elo"]) / 400.0))
    out["opp_adj_points_elo"] = out["points"] * out["opp_strength_weight_elo"]
    out = add_group_weighted(out, ["team"], "opp_adj_points_elo", 5, "opp_adj_form_5_elo")

    out = add_venue_weighted(out, "points", 5, "venue_form_pts_5")
    out = add_venue_weighted(out, "xg_diff_proxy", 5, "venue_xg_diff_5_proxy")

    return out


# =========================================================
# 8. 回并到比赛表
# =========================================================
def build_match_feature_table(base_df: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    out = base_df.copy()

    feature_cols = [
        "match_id",
        "is_home",
        "form_pts_3",
        "form_pts_5",
        "form_pts_8",
        "goal_diff_5",
        "xg_diff_5_proxy",
        "mom_slope",
        "mom_volatility_5",
        "rest_days",
        "match_density",
        "fatigue_adjusted_form",
        "travel_km_team",
        "travel_fatigue_score",
        "odds_move_for_mom_3",
        "odds_move_for_mom_5",
        "odds_move_abs_for_mom_5",
        "implied_prob_shift_for_mom_5",
        "odds_move_for_ewm_5",
        "implied_prob_shift_for_ewm_5",
        "travel_km_mom_5",
        "travel_km_ewm_5",
        "match_density_mom_5",
        "match_density_ewm_5",
        "travel_fatigue_mom_5",
        "opp_adj_form_5",
        "opp_adj_form_5_elo",
        "venue_form_pts_5",
        "venue_xg_diff_5_proxy",
        "pre_match_elo",
    ]
    ensure_columns(long_df, feature_cols, "long_df")

    lf = long_df[feature_cols].copy()

    home_feat = lf[lf["is_home"] == 1].drop(columns=["is_home"]).copy()
    away_feat = lf[lf["is_home"] == 0].drop(columns=["is_home"]).copy()

    home_feat = home_feat.rename(columns={c: f"home_{c}" for c in home_feat.columns if c != "match_id"})
    away_feat = away_feat.rename(columns={c: f"away_{c}" for c in away_feat.columns if c != "match_id"})

    out = out.merge(home_feat, on="match_id", how="left")
    out = out.merge(away_feat, on="match_id", how="left")

    diff_pairs = {
        "form_pts_diff_3": ("home_form_pts_3", "away_form_pts_3"),
        "form_pts_diff_5": ("home_form_pts_5", "away_form_pts_5"),
        "form_pts_diff_8": ("home_form_pts_8", "away_form_pts_8"),
        "goal_diff_mom_5_diff": ("home_goal_diff_5", "away_goal_diff_5"),
        "xg_diff_mom_5_proxy_diff": ("home_xg_diff_5_proxy", "away_xg_diff_5_proxy"),
        "mom_slope_diff": ("home_mom_slope", "away_mom_slope"),
        "mom_volatility_diff": ("home_mom_volatility_5", "away_mom_volatility_5"),
        "rest_days_diff": ("home_rest_days", "away_rest_days"),
        "match_density_diff": ("home_match_density", "away_match_density"),
        "fatigue_adjusted_form_diff": ("home_fatigue_adjusted_form", "away_fatigue_adjusted_form"),
        "travel_km_team_diff": ("home_travel_km_team", "away_travel_km_team"),
        "travel_fatigue_score_diff": ("home_travel_fatigue_score", "away_travel_fatigue_score"),
        "odds_move_for_mom_diff_3": ("home_odds_move_for_mom_3", "away_odds_move_for_mom_3"),
        "odds_move_for_mom_diff_5": ("home_odds_move_for_mom_5", "away_odds_move_for_mom_5"),
        "odds_move_abs_for_mom_diff_5": ("home_odds_move_abs_for_mom_5", "away_odds_move_abs_for_mom_5"),
        "implied_prob_shift_for_mom_diff_5": ("home_implied_prob_shift_for_mom_5", "away_implied_prob_shift_for_mom_5"),
        "odds_move_for_ewm_diff_5": ("home_odds_move_for_ewm_5", "away_odds_move_for_ewm_5"),
        "implied_prob_shift_for_ewm_diff_5": ("home_implied_prob_shift_for_ewm_5", "away_implied_prob_shift_for_ewm_5"),
        "travel_km_mom_diff_5": ("home_travel_km_mom_5", "away_travel_km_mom_5"),
        "travel_km_ewm_diff_5": ("home_travel_km_ewm_5", "away_travel_km_ewm_5"),
        "match_density_mom_diff_5": ("home_match_density_mom_5", "away_match_density_mom_5"),
        "match_density_ewm_diff_5": ("home_match_density_ewm_5", "away_match_density_ewm_5"),
        "travel_fatigue_mom_diff_5": ("home_travel_fatigue_mom_5", "away_travel_fatigue_mom_5"),
        "opp_adj_form_diff_5": ("home_opp_adj_form_5", "away_opp_adj_form_5"),
        "opp_adj_form_diff_5_elo": ("home_opp_adj_form_5_elo", "away_opp_adj_form_5_elo"),
        "venue_form_pts_5_diff": ("home_venue_form_pts_5", "away_venue_form_pts_5"),
        "venue_xg_diff_5_proxy_diff": ("home_venue_xg_diff_5_proxy", "away_venue_xg_diff_5_proxy"),
        "elo_diff_pre_from_merge": ("home_pre_match_elo", "away_pre_match_elo"),
    }

    for diff_name, (hcol, acol) in diff_pairs.items():
        ensure_columns(out, [hcol, acol], "merge后的比赛表")
        out[diff_name] = out[hcol] - out[acol]

    return out


# =========================================================
# 9. 输出
# =========================================================
def save_outputs(featured_df: pd.DataFrame, long_df: pd.DataFrame, output_path: Path):
    featured_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    long_df.to_csv(output_path.with_name(output_path.stem + "_long.csv"), index=False, encoding="utf-8-sig")


# =========================================================
# 10. 严格测试
# =========================================================
def safe_weighted_shifted_value(prior_values, window):
    vals = np.array(prior_values[-window:], dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan
    weights = np.arange(1, len(vals) + 1, dtype=float)
    return float(np.dot(vals, weights) / weights.sum())


def safe_ewm_shifted_value(prior_values, span):
    vals = pd.Series(prior_values, dtype=float)
    vals = vals[vals.notna()]
    if len(vals) == 0:
        return np.nan
    return float(vals.ewm(span=span, adjust=False, min_periods=1).mean().iloc[-1])


def safe_std_shifted_value(prior_values, window):
    vals = np.array(prior_values[-window:], dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return np.nan
    return float(np.std(vals, ddof=0))


def compute_expected_long_features(long_df: pd.DataFrame, idx: int):
    row = long_df.iloc[idx]
    prior = long_df[(long_df["team"] == row["team"]) & (long_df["_row_order"] < row["_row_order"])].copy()
    prior = prior.sort_values(["date_GMT", "timestamp", "_row_order"])

    exp = {}

    points_hist = prior["points"].tolist()
    gd_hist = prior["goal_diff"].tolist()
    xg_hist = prior["xg_diff_proxy"].tolist()
    odds_move_hist = prior["odds_move_for"].tolist()
    odds_move_abs_hist = prior["odds_move_abs_for"].tolist()
    prob_shift_hist = prior["implied_prob_shift_for"].tolist()
    travel_hist = prior["travel_km_team"].tolist()

    exp["form_pts_3"] = safe_weighted_shifted_value(points_hist, 3)
    exp["form_pts_5"] = safe_weighted_shifted_value(points_hist, 5)
    exp["form_pts_8"] = safe_weighted_shifted_value(points_hist, 8)
    exp["goal_diff_5"] = safe_weighted_shifted_value(gd_hist, 5)
    exp["xg_diff_5_proxy"] = safe_weighted_shifted_value(xg_hist, 5)

    exp["mom_slope"] = np.nan if pd.isna(exp["form_pts_3"]) or pd.isna(exp["form_pts_5"]) else exp["form_pts_3"] - exp["form_pts_5"]
    exp["mom_volatility_5"] = safe_std_shifted_value(gd_hist, 5)

    if len(prior) == 0:
        exp["rest_days"] = np.nan
    else:
        exp["rest_days"] = (row["date_GMT"] - prior.iloc[-1]["date_GMT"]).total_seconds() / 86400.0

    exp["match_density"] = np.nan if pd.isna(exp["rest_days"]) else 1.0 / max(exp["rest_days"], 1.0)
    exp["fatigue_adjusted_form"] = (
        np.nan if pd.isna(exp["form_pts_5"]) or pd.isna(exp["match_density"])
        else exp["form_pts_5"] - FATIGUE_LAMBDA * exp["match_density"]
    )
    exp["travel_fatigue_score"] = (
        np.nan if pd.isna(exp["match_density"]) or pd.isna(row["travel_km_team"])
        else exp["match_density"] + row["travel_km_team"] / 1000.0
    )

    exp["odds_move_for_mom_3"] = safe_weighted_shifted_value(odds_move_hist, 3)
    exp["odds_move_for_mom_5"] = safe_weighted_shifted_value(odds_move_hist, 5)
    exp["odds_move_abs_for_mom_5"] = safe_weighted_shifted_value(odds_move_abs_hist, 5)
    exp["implied_prob_shift_for_mom_5"] = safe_weighted_shifted_value(prob_shift_hist, 5)

    exp["odds_move_for_ewm_5"] = safe_ewm_shifted_value(odds_move_hist, 5)
    exp["implied_prob_shift_for_ewm_5"] = safe_ewm_shifted_value(prob_shift_hist, 5)

    exp["travel_km_mom_5"] = safe_weighted_shifted_value(travel_hist, 5)
    exp["travel_km_ewm_5"] = safe_ewm_shifted_value(travel_hist, 5)

    density_hist = prior["match_density"].tolist() if "match_density" in prior.columns else []
    fatigue_hist = prior["travel_fatigue_score"].tolist() if "travel_fatigue_score" in prior.columns else []

    exp["match_density_mom_5"] = safe_weighted_shifted_value(density_hist, 5)
    exp["match_density_ewm_5"] = safe_ewm_shifted_value(density_hist, 5)
    exp["travel_fatigue_mom_5"] = safe_weighted_shifted_value(fatigue_hist, 5)

    opp_adj_points_hist = (prior["points"] * prior["opponent_pre_match_ppg"] / 3.0).tolist()
    exp["opp_adj_form_5"] = safe_weighted_shifted_value(opp_adj_points_hist, 5)

    opp_adj_points_elo_hist = (
        prior["points"] * (1.0 / (1.0 + 10 ** ((1500.0 - prior["opponent_pre_match_elo"]) / 400.0)))
    ).tolist()
    exp["opp_adj_form_5_elo"] = safe_weighted_shifted_value(opp_adj_points_elo_hist, 5)

    prior_same_venue = prior[prior["is_home"] == row["is_home"]].copy()
    exp["venue_form_pts_5"] = safe_weighted_shifted_value(prior_same_venue["points"].tolist(), 5)
    exp["venue_xg_diff_5_proxy"] = safe_weighted_shifted_value(prior_same_venue["xg_diff_proxy"].tolist(), 5)

    return exp


def run_strict_long_feature_tests(long_df: pd.DataFrame, tol: float = 1e-9):
    test_cols = [
        "form_pts_3",
        "form_pts_5",
        "form_pts_8",
        "goal_diff_5",
        "xg_diff_5_proxy",
        "mom_slope",
        "mom_volatility_5",
        "rest_days",
        "match_density",
        "fatigue_adjusted_form",
        "travel_fatigue_score",
        "odds_move_for_mom_3",
        "odds_move_for_mom_5",
        "odds_move_abs_for_mom_5",
        "implied_prob_shift_for_mom_5",
        "odds_move_for_ewm_5",
        "implied_prob_shift_for_ewm_5",
        "travel_km_mom_5",
        "travel_km_ewm_5",
        "match_density_mom_5",
        "match_density_ewm_5",
        "travel_fatigue_mom_5",
        "opp_adj_form_5",
        "opp_adj_form_5_elo",
        "venue_form_pts_5",
        "venue_xg_diff_5_proxy",
    ]

    failures = []

    for idx in range(len(long_df)):
        expected = compute_expected_long_features(long_df, idx)
        for col in test_cols:
            actual = long_df.iloc[idx][col]
            exp = expected[col]
            if not values_equal(actual, exp, tol=tol):
                failures.append(
                    {
                        "idx": idx,
                        "match_id": long_df.iloc[idx]["match_id"],
                        "team": long_df.iloc[idx]["team"],
                        "is_home": long_df.iloc[idx]["is_home"],
                        "feature": col,
                        "actual": actual,
                        "expected": exp,
                    }
                )

    failure_df = pd.DataFrame(failures)
    failure_path = BASE_DIR / "momentum_feature_test_failures.csv"
    failure_df.to_csv(failure_path, index=False, encoding="utf-8-sig")

    print("=" * 70)
    print("STRICT LONG FEATURE TEST REPORT")
    print("=" * 70)
    print(f"Total long rows tested: {len(long_df)}")
    print(f"Total mismatches: {len(failure_df)}")

    if len(failure_df) == 0:
        print("PASS: 所有 momentum 特征逐行暴力回算通过。")
    else:
        print(f"FAIL: 存在不一致，见 {failure_path}")
        print(failure_df.head(20).to_string(index=False))

    return failure_df


# =========================================================
# 11. 逻辑检查
# =========================================================
def run_logical_checks(featured_df: pd.DataFrame, long_df: pd.DataFrame):
    issues = []

    first_rows = long_df[long_df.groupby("team").cumcount() == 0]
    if first_rows["form_pts_3"].notna().sum() > 0:
        issues.append(("first_match_form_pts_3_should_nan", int(first_rows["form_pts_3"].notna().sum())))

    diff_checks = {
        "form_pts_diff_5": ("home_form_pts_5", "away_form_pts_5"),
        "goal_diff_mom_5_diff": ("home_goal_diff_5", "away_goal_diff_5"),
        "opp_adj_form_diff_5": ("home_opp_adj_form_5", "away_opp_adj_form_5"),
        "opp_adj_form_diff_5_elo": ("home_opp_adj_form_5_elo", "away_opp_adj_form_5_elo"),
    }

    for diff_col, (hcol, acol) in diff_checks.items():
        calc = featured_df[hcol] - featured_df[acol]
        bad = ~(
            (featured_df[diff_col].isna() & calc.isna()) |
            np.isclose(featured_df[diff_col].fillna(0), calc.fillna(0), atol=1e-9, rtol=1e-9)
        )
        if bad.sum() > 0:
            issues.append((diff_col, int(bad.sum())))

    print("\nLOGICAL CHECKS")
    print(issues if issues else "PASS: no logical issues detected")
    return issues


# =========================================================
# 12. 主流程
# =========================================================
def main():
    print(f"Reading input from: {INPUT_PATH}")

    df = load_and_prepare_data(INPUT_PATH)
    df = add_match_base_columns(df)
    df = add_pre_match_elo(df, k_factor=ELO_K, home_adv=ELO_HOME_ADV)

    long_df = build_long_table(df)
    long_df = build_long_features(long_df)

    featured_df = build_match_feature_table(df, long_df)

    save_outputs(featured_df, long_df, OUTPUT_PATH)

    print("Feature engineering completed.")
    print(f"Saved match-level file: {OUTPUT_PATH}")
    print(f"Saved long-level file: {OUTPUT_PATH.with_name(OUTPUT_PATH.stem + '_long.csv')}")

    if RUN_TESTS:
        failure_df = run_strict_long_feature_tests(long_df, tol=TEST_TOL)
        issues = run_logical_checks(featured_df, long_df)

        if len(failure_df) == 0 and len(issues) == 0:
            print("\nFINAL TEST RESULT: ALL PASSED")
        else:
            print("\nFINAL TEST RESULT: CHECK FAILED ITEMS")


if __name__ == "__main__":
    main()