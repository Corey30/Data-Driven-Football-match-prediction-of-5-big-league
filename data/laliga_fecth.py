#!/usr/bin/env python3
"""
Strict seasonized LaLiga team skeleton + squad market value scraper.

What this script DOES (historically defensible):
1) Discover LaLiga teams from Transfermarkt competition pages for seasons 2017-2018 .. 2024-2025.
2) Fetch each team's season-specific squad page using the matching Transfermarkt saison_id.
3) Parse player market values from that exact season page and aggregate squad_market_value_sum.
4) Build a team-match-date skeleton from a master match table, keeping one row per
   (season, match_date, team_name).
5) Merge season-specific squad values onto the skeleton.

What this script DOES NOT claim to do:
- It does not reconstruct historical injuries/suspensions.
- It does not use current-season pages for past seasons.

Outputs:
- laliga_team_match_skeleton.csv
- laliga_team_season_squad_values.csv
- laliga_team_match_skeleton_with_squad_value.csv
- laliga_strict_squad_report.json
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
import urllib.request
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup  # type: ignore

COMP_URL = "https://www.transfermarkt.com/laliga/startseite/wettbewerb/ES1?saison_id={season_id}"
SQUAD_URL = "https://www.transfermarkt.com/{slug}/kader/verein/{team_id}/saison_id/{season_id}"
TEAM_LINK_RE = re.compile(r"/([^/]+)/startseite/verein/(\d+)/saison_id/(\d+)")
SEASON_RE = re.compile(r"^(\d{4})-(\d{4})$")

BUILTIN_TEAM_ALIAS_GROUPS = [
    ("athleticclubbilbao", "athbilbao", "athleticbilbao", "athleticclub"),
    ("atleticomadrid", "athmadrid", "atlmadrid", "atletico", "atleticodemadrid"),
    ("deportivoalaves", "alaves"),
    ("deportivolacoruna", "lacoruna", "deportivolacoruna"),
    ("rayovallecano", "vallecano", "rayo"),
    ("rcdespanyolbarcelona", "rcdespanyol", "espanyol", "espanol"),
    ("udlaspalmas", "laspalmas"),
    ("realvalladolid", "valladolid"),
    ("cadiz", "cadizcf"),
    ("malagacf", "malaga"),
    ("realbetisbalompie", "realbetis", "betis"),
    ("realsociedad", "sociedad"),
    ("fcbarcelona", "barcelona"),
    ("valenciacf", "valencia"),
    ("villarrealcf", "villarreal"),
    ("deportivoalaves", "deportivoalaves"),
]


@dataclass(frozen=True)
class TeamSeason:
    season: str
    tm_season_id: int
    team_id: str
    slug: str
    tm_team_name: str


def normalize_team_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-zA-Z0-9]+", "", text.lower())


def build_alias_map(alias_csv: Optional[Path]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for group in BUILTIN_TEAM_ALIAS_GROUPS:
        canonical = normalize_team_key(group[0])
        for alias in group:
            mapping[normalize_team_key(alias)] = canonical
        mapping.setdefault(canonical, canonical)

    if alias_csv is None or not alias_csv.exists():
        return mapping

    alias_df = pd.read_csv(alias_csv)
    if alias_df.empty or alias_df.shape[1] < 2:
        return mapping

    cols_lower = {c.lower(): c for c in alias_df.columns}
    if "master_name" in cols_lower and "football_data_name" in cols_lower:
        left_col = cols_lower["master_name"]
        right_col = cols_lower["football_data_name"]
    elif "canonical_name" in cols_lower and "variant_name" in cols_lower:
        left_col = cols_lower["canonical_name"]
        right_col = cols_lower["variant_name"]
    else:
        left_col, right_col = alias_df.columns[:2]

    for _, row in alias_df[[left_col, right_col]].dropna().iterrows():
        left = normalize_team_key(row[left_col])
        right = normalize_team_key(row[right_col])
        if not left or not right:
            continue
        mapping[right] = left
        mapping.setdefault(left, left)
    return mapping


def canonical_team_key(value: str, alias_map: Dict[str, str]) -> str:
    key = normalize_team_key(value)
    return alias_map.get(key, key)


def season_to_tm_id(season: str) -> int:
    m = SEASON_RE.match(str(season).strip())
    if not m:
        raise ValueError(f"Invalid season format: {season}")
    start_year = int(m.group(1))
    end_year = int(m.group(2))
    if end_year != start_year + 1:
        raise ValueError(f"Season should span one year: {season}")
    return start_year


def iter_seasons(start_season: str, end_season: str) -> List[str]:
    start_id = season_to_tm_id(start_season)
    end_id = season_to_tm_id(end_season)
    if start_id > end_id:
        start_id, end_id = end_id, start_id
    return [f"{y}-{y+1}" for y in range(start_id, end_id + 1)]


def fetch_html(url: str, timeout: int) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="ignore")


def parse_market_value_eur(raw: str) -> float:
    s = str(raw).strip().lower().replace(" ", "")
    if not s or s == "-":
        return float("nan")
    s = s.replace(",", ".")
    s = s.replace("€", "")
    mult = 1.0
    if s.endswith("bn"):
        mult = 1_000_000_000.0
        s = s[:-2]
    elif s.endswith("m"):
        mult = 1_000_000.0
        s = s[:-1]
    elif s.endswith("k"):
        mult = 1_000.0
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return float("nan")


def parse_date_column(series: pd.Series) -> pd.Series:
    # Prefer exact datetime if already present; otherwise coerce and keep date only.
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    if dt.notna().any():
        return dt.dt.date
    # Fallback for common day-first strings.
    dt = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return dt.dt.date


def extract_laliga_team_seasons(seasons: Iterable[str], timeout: int, sleep_seconds: float) -> List[TeamSeason]:
    rows: List[TeamSeason] = []
    seen: set[Tuple[str, str]] = set()

    for season in seasons:
        tm_season_id = season_to_tm_id(season)
        url = COMP_URL.format(season_id=tm_season_id)
        html = fetch_html(url, timeout)
        soup = BeautifulSoup(html, "html.parser")
        anchors = soup.select('a[href*="/startseite/verein/"]')

        for a in anchors:
            href = a.get("href") or ""
            m = TEAM_LINK_RE.search(href)
            if not m:
                continue
            slug, team_id, href_season_id = m.group(1), m.group(2), int(m.group(3))
            if href_season_id != tm_season_id:
                continue
            team_name = a.get_text(" ", strip=True)
            if not team_name:
                continue
            key = (season, team_id)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                TeamSeason(
                    season=season,
                    tm_season_id=tm_season_id,
                    team_id=team_id,
                    slug=slug,
                    tm_team_name=team_name,
                )
            )
        time.sleep(max(0.0, sleep_seconds))
    return rows


def extract_player_name(row) -> str:
    a = row.select_one('a[href*="/profil/spieler/"]')
    if a is not None:
        txt = a.get_text(" ", strip=True)
        if txt:
            return txt
    cells = [td.get_text(" ", strip=True) for td in row.select("td")]
    for val in cells:
        if val:
            return val
    return ""


def parse_squad_page(team: TeamSeason, timeout: int) -> Tuple[pd.DataFrame, dict]:
    url = SQUAD_URL.format(slug=team.slug, team_id=team.team_id, season_id=team.tm_season_id)
    html = fetch_html(url, timeout)
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.items")

    player_rows: List[dict] = []
    if table is not None:
        for row in table.select("tbody tr"):
            cls = row.get("class") or []
            if not any(c in cls for c in ("odd", "even")):
                continue
            cells = [td.get_text(" ", strip=True) for td in row.select("td")]
            if not cells:
                continue
            player_name = extract_player_name(row)
            mv = parse_market_value_eur(cells[-1])
            if not player_name:
                continue
            if np.isnan(mv):
                continue
            player_rows.append(
                {
                    "season": team.season,
                    "tm_season_id": team.tm_season_id,
                    "team_id": team.team_id,
                    "slug": team.slug,
                    "tm_team_name": team.tm_team_name,
                    "player_name": player_name,
                    "player_market_value_eur": float(mv),
                    "source_url": url,
                }
            )

    players_df = pd.DataFrame(player_rows)
    squad_market_value_sum = float(players_df["player_market_value_eur"].sum()) if not players_df.empty else float("nan")
    meta = {
        "season": team.season,
        "tm_season_id": team.tm_season_id,
        "team_id": team.team_id,
        "slug": team.slug,
        "tm_team_name": team.tm_team_name,
        "player_count_with_value": int(len(players_df)),
        "squad_market_value_sum": squad_market_value_sum,
        "source_url": url,
    }
    return players_df, meta


def build_team_match_skeleton(master_df: pd.DataFrame) -> pd.DataFrame:
    required = {"season", "home_team_name", "away_team_name"}
    missing = required - set(master_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in master table: {sorted(missing)}")

    date_col = None
    for candidate in ["match_date", "match_datetime_utc", "kickoff_utc", "datetime_utc"]:
        if candidate in master_df.columns:
            date_col = candidate
            break
    if date_col is None:
        raise ValueError("Master table must contain one of: match_date, match_datetime_utc, kickoff_utc, datetime_utc")

    df = master_df.copy()
    if "league" in df.columns:
        df = df[df["league"].astype(str).str.lower().eq("laliga")].copy()

    df["match_date"] = parse_date_column(df[date_col])
    df = df[df["match_date"].notna()].copy()

    home = df[["season", "match_date", "home_team_name"]].rename(columns={"home_team_name": "team_name"})
    away = df[["season", "match_date", "away_team_name"]].rename(columns={"away_team_name": "team_name"})

    out = pd.concat([home, away], ignore_index=True).drop_duplicates().reset_index(drop=True)
    out["league"] = "LaLiga"
    out = out[["league", "season", "match_date", "team_name"]].sort_values(["season", "match_date", "team_name"]).reset_index(drop=True)
    return out


def map_tm_names_to_master_names(
    skeleton_df: pd.DataFrame,
    team_meta_df: pd.DataFrame,
    alias_map: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    master_names = skeleton_df[["season", "team_name"]].drop_duplicates().copy()
    master_names["team_key"] = master_names["team_name"].map(lambda x: canonical_team_key(x, alias_map))

    team_meta = team_meta_df.copy()
    team_meta["team_key"] = team_meta["tm_team_name"].map(lambda x: canonical_team_key(x, alias_map))

    mapped = team_meta.merge(
        master_names,
        on=["season", "team_key"],
        how="left",
        validate="m:1",
    )
    mapped["resolved_team_name"] = mapped["team_name"].fillna(mapped["tm_team_name"])

    unresolved = mapped[mapped["team_name"].isna()].copy()
    return mapped, unresolved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict seasonized LaLiga squad value + team skeleton scraper")
    p.add_argument("--master-csv", type=Path, required=True, help="Master match table CSV")
    p.add_argument("--alias-csv", type=Path, default=None, help="Optional team alias CSV")
    p.add_argument("--start-season", type=str, default="2017-2018")
    p.add_argument("--end-season", type=str, default="2024-2025")
    p.add_argument("--out-dir", type=Path, default=Path("./outputs_strict_squad"))
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--sleep-seconds", type=float, default=0.4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    master_df = pd.read_csv(args.master_csv)
    alias_map = build_alias_map(args.alias_csv)
    seasons = iter_seasons(args.start_season, args.end_season)

    skeleton_df = build_team_match_skeleton(master_df)
    skeleton_df = skeleton_df[skeleton_df["season"].isin(seasons)].copy()

    team_seasons = extract_laliga_team_seasons(seasons, timeout=args.timeout, sleep_seconds=args.sleep_seconds)
    if not team_seasons:
        raise RuntimeError("No team seasons discovered from Transfermarkt competition pages.")

    team_meta_rows: List[dict] = []
    all_players: List[pd.DataFrame] = []
    for team in team_seasons:
        players_df, meta = parse_squad_page(team, timeout=args.timeout)
        team_meta_rows.append(meta)
        if not players_df.empty:
            all_players.append(players_df)
        time.sleep(max(0.0, args.sleep_seconds))

    team_meta_df = pd.DataFrame(team_meta_rows).sort_values(["season", "tm_team_name"]).reset_index(drop=True)
    players_df = pd.concat(all_players, ignore_index=True) if all_players else pd.DataFrame(
        columns=[
            "season", "tm_season_id", "team_id", "slug", "tm_team_name",
            "player_name", "player_market_value_eur", "source_url",
        ]
    )

    mapped_team_meta_df, unresolved_df = map_tm_names_to_master_names(skeleton_df, team_meta_df, alias_map)

    squad_out = mapped_team_meta_df[[
        "season",
        "resolved_team_name",
        "tm_team_name",
        "tm_season_id",
        "team_id",
        "slug",
        "player_count_with_value",
        "squad_market_value_sum",
        "source_url",
    ]].rename(columns={"resolved_team_name": "team_name"}).copy()

    merged_skeleton_df = skeleton_df.merge(
        squad_out[["season", "team_name", "squad_market_value_sum", "team_id", "slug", "source_url"]],
        on=["season", "team_name"],
        how="left",
        validate="m:1",
    )

    skeleton_path = args.out_dir / "laliga_team_match_skeleton.csv"
    squad_path = args.out_dir / "laliga_team_season_squad_values.csv"
    merged_path = args.out_dir / "laliga_team_match_skeleton_with_squad_value.csv"
    players_path = args.out_dir / "laliga_team_season_player_market_values.csv"
    report_path = args.out_dir / "laliga_strict_squad_report.json"

    skeleton_df.to_csv(skeleton_path, index=False, encoding="utf-8-sig")
    squad_out.to_csv(squad_path, index=False, encoding="utf-8-sig")
    merged_skeleton_df.to_csv(merged_path, index=False, encoding="utf-8-sig")
    players_df.to_csv(players_path, index=False, encoding="utf-8-sig")

    report = {
        "seasons": seasons,
        "master_rows": int(len(master_df)),
        "skeleton_rows": int(len(skeleton_df)),
        "team_seasons_discovered": int(len(team_seasons)),
        "unique_team_season_rows": int(len(squad_out)),
        "player_value_rows": int(len(players_df)),
        "unresolved_team_name_mappings": int(len(unresolved_df)),
        "skeleton_rows_missing_squad_value": int(merged_skeleton_df["squad_market_value_sum"].isna().sum()),
        "coverage_ratio": float(merged_skeleton_df["squad_market_value_sum"].notna().mean()) if len(merged_skeleton_df) else 0.0,
        "output_files": {
            "skeleton": str(skeleton_path),
            "team_season_squad_values": str(squad_path),
            "team_match_skeleton_with_squad_value": str(merged_path),
            "player_market_values": str(players_path),
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
