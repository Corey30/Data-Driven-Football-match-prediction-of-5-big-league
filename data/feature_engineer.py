import pandas as pd
import numpy as np
from math import isclose


# =========================================================
# 0. 配置
# =========================================================
INPUT_PATH = "span3_with_momentum_v2.csv"
OUTPUT_PATH = "span3_with_momentum_v2_featured.csv"
WINDOW = 5
RUN_TESTS = True
TEST_TOL = 1e-9


# =========================================================
# 1. 读取与排序
# =========================================================
def load_and_prepare_data(input_path: str) -> pd.DataFrame:
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
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    # 优先使用 mixed，避免 warning；若 pandas 版本不支持则回退
    try:
        df["date_GMT"] = pd.to_datetime(
            df["date_GMT"],
            format="mixed",
            errors="coerce",
            utc=True
        )
    except TypeError:
        df["date_GMT"] = pd.to_datetime(
            df["date_GMT"],
            errors="coerce",
            utc=True
        )

    df["_orig_idx"] = np.arange(len(df))

    sort_cols = []
    for c in ["date_GMT", "timestamp", "_orig_idx"]:
        if c in df.columns:
            sort_cols.append(c)

    df = df.sort_values(sort_cols).reset_index(drop=True)
    df["_row_order"] = np.arange(len(df))

    return df


# =========================================================
# 2. 构造中间比赛结果列（仅用于历史特征构造）
# =========================================================
def add_intermediate_match_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["_home_win"] = (out["target_match_result"] == "H").astype(int)
    out["_home_draw"] = (out["target_match_result"] == "D").astype(int)
    out["_home_loss"] = (out["target_match_result"] == "A").astype(int)

    out["_away_win"] = (out["target_match_result"] == "A").astype(int)
    out["_away_draw"] = (out["target_match_result"] == "D").astype(int)
    out["_away_loss"] = (out["target_match_result"] == "H").astype(int)

    out["_home_gf"] = out["target_home_team_goal_count"]
    out["_home_ga"] = out["target_away_team_goal_count"]
    out["_away_gf"] = out["target_away_team_goal_count"]
    out["_away_ga"] = out["target_home_team_goal_count"]

    out["_home_goal_diff"] = out["_home_gf"] - out["_home_ga"]
    out["_away_goal_diff"] = out["_away_gf"] - out["_away_ga"]

    out["_home_win_margin"] = np.where(out["_home_win"] == 1, out["_home_goal_diff"], np.nan)
    out["_away_win_margin"] = np.where(out["_away_win"] == 1, out["_away_goal_diff"], np.nan)

    out["_home_loss_margin"] = np.where(out["_home_loss"] == 1, -out["_home_goal_diff"], np.nan)
    out["_away_loss_margin"] = np.where(out["_away_loss"] == 1, -out["_away_goal_diff"], np.nan)

    return out


# =========================================================
# 3. 生成单一场地视角的历史特征
#    prefix = "home" or "away"
# =========================================================
def add_venue_history_features(df: pd.DataFrame, team_col: str, prefix: str, window: int = 5) -> pd.DataFrame:
    out = df.copy()

    if prefix == "home":
        win_col = "_home_win"
        draw_col = "_home_draw"
        loss_col = "_home_loss"
        gf_col = "_home_gf"
        ga_col = "_home_ga"
        gd_col = "_home_goal_diff"
        win_margin_col = "_home_win_margin"
        loss_margin_col = "_home_loss_margin"
    elif prefix == "away":
        win_col = "_away_win"
        draw_col = "_away_draw"
        loss_col = "_away_loss"
        gf_col = "_away_gf"
        ga_col = "_away_ga"
        gd_col = "_away_goal_diff"
        win_margin_col = "_away_win_margin"
        loss_margin_col = "_away_loss_margin"
    else:
        raise ValueError("prefix must be 'home' or 'away'")

    g = out.groupby(team_col, sort=False)

    # 历史同场地比赛场次（不含当前场）
    out[f"_{prefix}_matches_before"] = g.cumcount()

    # 先shift(1)，只保留赛前历史
    win_hist = g[win_col].shift(1)
    draw_hist = g[draw_col].shift(1)
    loss_hist = g[loss_col].shift(1)
    gf_hist = g[gf_col].shift(1)
    ga_hist = g[ga_col].shift(1)
    gd_hist = g[gd_col].shift(1)
    win_margin_hist = g[win_margin_col].shift(1)
    loss_margin_hist = g[loss_margin_col].shift(1)

    # 最近5场窗口胜场数
    # 修复点：首场历史为空时应为 0，不应为 NaN
    out[f"{prefix.capitalize()} wins in the window"] = (
        win_hist.groupby(out[team_col])
        .rolling(window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    # 历史累计胜平负场数（不含当前场）
    out[f"_{prefix}_wins_before"] = g[win_col].cumsum() - out[win_col]
    out[f"_{prefix}_draws_before"] = g[draw_col].cumsum() - out[draw_col]
    out[f"_{prefix}_losses_before"] = g[loss_col].cumsum() - out[loss_col]

    denom = out[f"_{prefix}_matches_before"].replace(0, np.nan)

    # 历史累计比率
    out[f"Total {prefix} win rate"] = out[f"_{prefix}_wins_before"] / denom
    out[f"Total {prefix} draw rate"] = out[f"_{prefix}_draws_before"] / denom
    out[f"Total {prefix} loss rate"] = out[f"_{prefix}_losses_before"] / denom

    # 历史累计进球/失球/净胜球
    out[f"{prefix.capitalize()} goals forward"] = gf_hist.groupby(out[team_col]).cumsum().fillna(0)
    out[f"{prefix.capitalize()} goals against"] = ga_hist.groupby(out[team_col]).cumsum().fillna(0)
    out[f"{prefix.capitalize()} goal differential"] = gd_hist.groupby(out[team_col]).cumsum().fillna(0)

    # 场均版
    out[f"{prefix.capitalize()} goals forward per match"] = out[f"{prefix.capitalize()} goals forward"] / denom
    out[f"{prefix.capitalize()} goals against per match"] = out[f"{prefix.capitalize()} goals against"] / denom
    out[f"{prefix.capitalize()} goal differential per match"] = out[f"{prefix.capitalize()} goal differential"] / denom

    # 历史平均赢球净胜幅度 / 输球净负幅度
    out[f"{prefix.capitalize()} wins margin goal"] = (
        win_margin_hist.groupby(out[team_col]).expanding().mean().reset_index(level=0, drop=True)
    )
    out[f"{prefix.capitalize()} losses margin goal"] = (
        loss_margin_hist.groupby(out[team_col]).expanding().mean().reset_index(level=0, drop=True)
    )

    return out


# =========================================================
# 4. 执行特征工程
# =========================================================
def build_features(input_path: str, output_path: str, window: int = 5):
    df = load_and_prepare_data(input_path)
    df = add_intermediate_match_columns(df)

    df = add_venue_history_features(
        df=df,
        team_col="home_team_name",
        prefix="home",
        window=window
    )

    df = add_venue_history_features(
        df=df,
        team_col="away_team_name",
        prefix="away",
        window=window
    )

    required_features = [
        "Away wins in the window",
        "Home wins in the window",
        "Total home win rate",
        "Total away win rate",
        "Total away loss rate",
        "Total away draw rate",
        "Total home draw rate",
        "Total home loss rate",
        "Away goal differential",
        "Away goals forward",
        "Home goal differential",
        "Away goals against",
        "Home goals forward",
        "Home goals against",
        "Home wins margin goal",
        "Away losses margin goal",
        "Away wins margin goal",
        "Home losses margin goal",
    ]

    enhanced_training_features = [
        "Home goals forward per match",
        "Home goals against per match",
        "Home goal differential per match",
        "Away goals forward per match",
        "Away goals against per match",
        "Away goal differential per match",
    ]

    missing = [c for c in required_features + enhanced_training_features if c not in df.columns]
    if missing:
        raise ValueError(f"以下特征未成功生成: {missing}")

    final_new_feature_list = required_features + enhanced_training_features

    drop_cols = [
        "_home_win", "_home_draw", "_home_loss",
        "_away_win", "_away_draw", "_away_loss",
        "_home_gf", "_home_ga", "_away_gf", "_away_ga",
        "_home_goal_diff", "_away_goal_diff",
        "_home_win_margin", "_away_win_margin",
        "_home_loss_margin", "_away_loss_margin",
        "_home_matches_before", "_away_matches_before",
        "_home_wins_before", "_home_draws_before", "_home_losses_before",
        "_away_wins_before", "_away_draws_before", "_away_losses_before",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Feature engineering completed.")
    print(f"Saved to: {output_path}")
    print("\nNew core features:")
    for c in required_features:
        print(f"- {c}")

    print("\nEnhanced training-friendly features:")
    for c in enhanced_training_features:
        print(f"- {c}")

    return df, required_features, enhanced_training_features, final_new_feature_list


# =========================================================
# 5. 严格测试：逐行暴力回算
# =========================================================
def safe_mean(series: pd.Series):
    series = series.dropna()
    if len(series) == 0:
        return np.nan
    return series.mean()


def calc_expected_features_bruteforce(df: pd.DataFrame, idx: int, window: int = 5) -> dict:
    row = df.iloc[idx]
    prior = df[df["_row_order"] < row["_row_order"]]

    home_team = row["home_team_name"]
    away_team = row["away_team_name"]

    home_hist = prior[prior["home_team_name"] == home_team].copy()
    away_hist = prior[prior["away_team_name"] == away_team].copy()

    home_last5 = home_hist.tail(window)
    away_last5 = away_hist.tail(window)

    exp = {}

    exp["Home wins in the window"] = float(home_last5["_home_win"].sum())
    exp["Away wins in the window"] = float(away_last5["_away_win"].sum())

    home_n = len(home_hist)
    away_n = len(away_hist)

    exp["Total home win rate"] = np.nan if home_n == 0 else home_hist["_home_win"].sum() / home_n
    exp["Total home draw rate"] = np.nan if home_n == 0 else home_hist["_home_draw"].sum() / home_n
    exp["Total home loss rate"] = np.nan if home_n == 0 else home_hist["_home_loss"].sum() / home_n

    exp["Total away win rate"] = np.nan if away_n == 0 else away_hist["_away_win"].sum() / away_n
    exp["Total away draw rate"] = np.nan if away_n == 0 else away_hist["_away_draw"].sum() / away_n
    exp["Total away loss rate"] = np.nan if away_n == 0 else away_hist["_away_loss"].sum() / away_n

    exp["Home goals forward"] = float(home_hist["_home_gf"].sum())
    exp["Home goals against"] = float(home_hist["_home_ga"].sum())
    exp["Home goal differential"] = float(home_hist["_home_goal_diff"].sum())

    exp["Away goals forward"] = float(away_hist["_away_gf"].sum())
    exp["Away goals against"] = float(away_hist["_away_ga"].sum())
    exp["Away goal differential"] = float(away_hist["_away_goal_diff"].sum())

    exp["Home goals forward per match"] = np.nan if home_n == 0 else home_hist["_home_gf"].sum() / home_n
    exp["Home goals against per match"] = np.nan if home_n == 0 else home_hist["_home_ga"].sum() / home_n
    exp["Home goal differential per match"] = np.nan if home_n == 0 else home_hist["_home_goal_diff"].sum() / home_n

    exp["Away goals forward per match"] = np.nan if away_n == 0 else away_hist["_away_gf"].sum() / away_n
    exp["Away goals against per match"] = np.nan if away_n == 0 else away_hist["_away_ga"].sum() / away_n
    exp["Away goal differential per match"] = np.nan if away_n == 0 else away_hist["_away_goal_diff"].sum() / away_n

    exp["Home wins margin goal"] = safe_mean(home_hist["_home_win_margin"])
    exp["Home losses margin goal"] = safe_mean(home_hist["_home_loss_margin"])
    exp["Away wins margin goal"] = safe_mean(away_hist["_away_win_margin"])
    exp["Away losses margin goal"] = safe_mean(away_hist["_away_loss_margin"])

    return exp


def values_equal(a, b, tol=1e-9):
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) != pd.isna(b):
        return False
    return isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def run_strict_tests(featured_df: pd.DataFrame, tol: float = 1e-9, window: int = 5):
    test_cols = [
        "Away wins in the window",
        "Home wins in the window",
        "Total home win rate",
        "Total away win rate",
        "Total away loss rate",
        "Total away draw rate",
        "Total home draw rate",
        "Total home loss rate",
        "Away goal differential",
        "Away goals forward",
        "Home goal differential",
        "Away goals against",
        "Home goals forward",
        "Home goals against",
        "Home wins margin goal",
        "Away losses margin goal",
        "Away wins margin goal",
        "Home losses margin goal",
        "Home goals forward per match",
        "Home goals against per match",
        "Home goal differential per match",
        "Away goals forward per match",
        "Away goals against per match",
        "Away goal differential per match",
    ]

    failures = []

    total_rows = len(featured_df)
    for idx in range(total_rows):
        expected = calc_expected_features_bruteforce(featured_df, idx, window=window)

        for col in test_cols:
            actual = featured_df.iloc[idx][col]
            exp = expected[col]
            if not values_equal(actual, exp, tol=tol):
                failures.append({
                    "row_idx": idx,
                    "match_id": featured_df.iloc[idx]["match_id"] if "match_id" in featured_df.columns else None,
                    "feature": col,
                    "actual": actual,
                    "expected": exp
                })

    failure_df = pd.DataFrame(failures)

    logical_checks = {}

    rate_cols = [
        "Total home win rate",
        "Total away win rate",
        "Total away loss rate",
        "Total away draw rate",
        "Total home draw rate",
        "Total home loss rate",
    ]
    rate_out_of_range = []
    for col in rate_cols:
        bad = featured_df[~featured_df[col].isna() & ((featured_df[col] < 0) | (featured_df[col] > 1))]
        if len(bad) > 0:
            rate_out_of_range.append((col, len(bad)))
    logical_checks["rate_out_of_range"] = rate_out_of_range

    window_cols = ["Away wins in the window", "Home wins in the window"]
    window_bad = []
    for col in window_cols:
        bad = featured_df[~featured_df[col].isna() & ((featured_df[col] < 0) | (featured_df[col] > WINDOW))]
        if len(bad) > 0:
            window_bad.append((col, len(bad)))
    logical_checks["window_bad"] = window_bad

    first_match_issues = []

    home_first = featured_df[featured_df.groupby("home_team_name").cumcount() == 0]
    for _, r in home_first.iterrows():
        if not pd.isna(r["Total home win rate"]):
            first_match_issues.append(("home_first_rate_not_nan", r.get("match_id", None)))
        if r["Home goals forward"] != 0:
            first_match_issues.append(("home_first_gf_not_zero", r.get("match_id", None)))
        if r["Home goals against"] != 0:
            first_match_issues.append(("home_first_ga_not_zero", r.get("match_id", None)))
        if r["Home goal differential"] != 0:
            first_match_issues.append(("home_first_gd_not_zero", r.get("match_id", None)))
        if r["Home wins in the window"] != 0:
            first_match_issues.append(("home_first_window_not_zero", r.get("match_id", None)))

    away_first = featured_df[featured_df.groupby("away_team_name").cumcount() == 0]
    for _, r in away_first.iterrows():
        if not pd.isna(r["Total away win rate"]):
            first_match_issues.append(("away_first_rate_not_nan", r.get("match_id", None)))
        if r["Away goals forward"] != 0:
            first_match_issues.append(("away_first_gf_not_zero", r.get("match_id", None)))
        if r["Away goals against"] != 0:
            first_match_issues.append(("away_first_ga_not_zero", r.get("match_id", None)))
        if r["Away goal differential"] != 0:
            first_match_issues.append(("away_first_gd_not_zero", r.get("match_id", None)))
        if r["Away wins in the window"] != 0:
            first_match_issues.append(("away_first_window_not_zero", r.get("match_id", None)))

    logical_checks["first_match_issues"] = first_match_issues

    print("\n" + "=" * 70)
    print("STRICT TEST REPORT")
    print("=" * 70)
    print(f"Total rows tested: {total_rows}")
    print(f"Total cell mismatches: {len(failure_df)}")

    if len(failure_df) == 0:
        print("PASS: 所有逐行逐列回算测试全部通过，特征生成逻辑与赛前约束一致。")
    else:
        print("FAIL: 存在不一致，请查看 failure_df。")
        print(failure_df.head(20).to_string(index=False))

    print("\nAdditional logical checks:")
    print(f"- rate_out_of_range: {logical_checks['rate_out_of_range']}")
    print(f"- window_bad: {logical_checks['window_bad']}")
    print(f"- first_match_issues count: {len(logical_checks['first_match_issues'])}")

    if len(logical_checks["first_match_issues"]) > 0:
        print("Examples of first_match_issues:")
        print(logical_checks["first_match_issues"][:20])

    failure_report_path = "feature_test_failures.csv"
    failure_df.to_csv(failure_report_path, index=False, encoding="utf-8-sig")

    print(f"\nFailure report saved to: {failure_report_path}")

    return failure_df, logical_checks


# =========================================================
# 6. 主程序
# =========================================================
if __name__ == "__main__":
    featured_df, required_features, enhanced_features, final_new_feature_list = build_features(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        window=WINDOW
    )

    print("\nSuggested final new feature list for modeling:")
    for c in final_new_feature_list:
        print(f"- {c}")

    if RUN_TESTS:
        test_df = load_and_prepare_data(INPUT_PATH)
        test_df = add_intermediate_match_columns(test_df)
        test_df = add_venue_history_features(test_df, "home_team_name", "home", window=WINDOW)
        test_df = add_venue_history_features(test_df, "away_team_name", "away", window=WINDOW)

        failure_df, logical_checks = run_strict_tests(
            featured_df=test_df,
            tol=TEST_TOL,
            window=WINDOW
        )

        if len(failure_df) == 0 and len(logical_checks["rate_out_of_range"]) == 0 \
           and len(logical_checks["window_bad"]) == 0 \
           and len(logical_checks["first_match_issues"]) == 0:
            print("\nFINAL TEST RESULT: ALL PASSED")
        else:
            print("\nFINAL TEST RESULT: CHECK FAILED ITEMS")