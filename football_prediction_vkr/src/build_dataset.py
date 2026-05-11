from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR.parent
OUTPUT_PATH = BASE_DIR / "data" / "dataset.csv"


MATCH_SOURCES = [
    ("E0 (1).csv", "EPL", "2024_2025"),
    ("E0.csv", "EPL", "2025_2026"),
    ("SP1 (1).csv", "LaLiga", "2024_2025"),
    ("SP1.csv", "LaLiga", "2025_2026"),
    ("D1 (1).csv", "Bundesliga", "2024_2025"),
    ("D1.csv", "Bundesliga", "2025_2026"),
]

STATS_SOURCES = [
    ("league-chemp (1).csv", "LaLiga", "2024_2025"),
    ("league-chemp.csv", "LaLiga", "2025_2026"),
    ("league-chemp (3).csv", "Bundesliga", "2024_2025"),
    ("league-chemp (2).csv", "Bundesliga", "2025_2026"),
]

MATCH_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]

TEAM_NAME_ALIASES = {
    # La Liga
    "Ath Bilbao": "Athletic Club",
    "Ath Madrid": "Atletico Madrid",
    "Vallecano": "Rayo Vallecano",
    "Sociedad": "Real Sociedad",
    "Alaves": "Alaves",
    "Espanol": "Espanyol",
    "Leganes": "Leganes",
    "Las Palmas": "Las Palmas",
    "Oviedo": "Real Oviedo",
    "Elche": "Elche",
    "Betis": "Real Betis",
    "Celta": "Celta Vigo",
    "Barcelona": "Barcelona",
    "Real Madrid": "Real Madrid",
    "Girona": "Girona",
    "Mallorca": "Mallorca",
    "Osasuna": "Osasuna",
    "Valencia": "Valencia",
    "Villarreal": "Villarreal",
    "Sevilla": "Sevilla",
    "Getafe": "Getafe",
    "Levante": "Levante",
    "Valladolid": "Real Valladolid",
    # Bundesliga
    "Bayern Munich": "Bayern Munich",
    "Leverkusen": "Bayer Leverkusen",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "Dortmund": "Borussia Dortmund",
    "Freiburg": "Freiburg",
    "Mainz": "Mainz 05",
    "RB Leipzig": "RasenBallsport Leipzig",
    "Werder Bremen": "Werder Bremen",
    "Stuttgart": "VfB Stuttgart",
    "M'gladbach": "Borussia M.Gladbach",
    "Wolfsburg": "Wolfsburg",
    "Augsburg": "Augsburg",
    "Union Berlin": "Union Berlin",
    "St Pauli": "St. Pauli",
    "Hoffenheim": "Hoffenheim",
    "Heidenheim": "FC Heidenheim",
    "Bochum": "Bochum",
    "FC Koln": "FC Koln",
    "Holstein Kiel": "Holstein Kiel",
    "Hamburg": "Hamburg",
}


def _read_matches(file_name: str, league: str, season: str) -> pd.DataFrame:
    path = _resolve_input_file(file_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing match file: {path}")
    df = pd.read_csv(path)
    missing = [col for col in MATCH_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {file_name}: {missing}")
    out = df[MATCH_COLS].copy()
    for optional_col in ["B365H", "B365D", "B365A"]:
        if optional_col in df.columns:
            out[optional_col] = pd.to_numeric(df[optional_col], errors="coerce")
        else:
            out[optional_col] = pd.NA
    out["league"] = league
    out["season"] = season
    return out


def _read_team_stats(file_name: str, league: str, season: str) -> pd.DataFrame:
    path = _resolve_input_file(file_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing stats file: {path}")
    df = pd.read_csv(path, sep=";")
    needed = ["team", "xG", "xGA", "xPTS"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {file_name}: {missing}")
    out = df[needed].copy()
    out["team"] = out["team"].astype(str).str.strip().str.replace('"', "", regex=False)
    out["league"] = league
    out["season"] = season
    return out


def _canon_team_name(name: str) -> str:
    clean = str(name).strip()
    return TEAM_NAME_ALIASES.get(clean, clean)


def _resolve_input_file(file_name: str) -> Path:
    candidates = [
        BASE_DIR / "data" / file_name,
        RAW_DIR / file_name,
        Path.cwd() / file_name,
        Path.cwd() / "data" / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Return primary expected location for clear error message.
    return candidates[0]


def build_dataset() -> pd.DataFrame:
    matches = pd.concat([_read_matches(*src) for src in MATCH_SOURCES], ignore_index=True)
    matches = matches.rename(
        columns={"Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team", "FTHG": "fthg", "FTAG": "ftag"}
    )
    matches["date"] = pd.to_datetime(matches["date"], dayfirst=True, errors="coerce")
    matches = matches.dropna(subset=["date", "home_team", "away_team", "fthg", "ftag"]).copy()
    matches["fthg"] = pd.to_numeric(matches["fthg"], errors="coerce")
    matches["ftag"] = pd.to_numeric(matches["ftag"], errors="coerce")
    matches = matches.dropna(subset=["fthg", "ftag"]).copy()
    matches["fthg"] = matches["fthg"].astype(int)
    matches["ftag"] = matches["ftag"].astype(int)
    matches["b365_home"] = pd.to_numeric(matches["B365H"], errors="coerce")
    matches["b365_draw"] = pd.to_numeric(matches["B365D"], errors="coerce")
    matches["b365_away"] = pd.to_numeric(matches["B365A"], errors="coerce")
    matches = matches.drop(columns=["B365H", "B365D", "B365A"], errors="ignore")

    matches["home_team_canon"] = matches["home_team"].map(_canon_team_name)
    matches["away_team_canon"] = matches["away_team"].map(_canon_team_name)

    team_stats = pd.concat([_read_team_stats(*src) for src in STATS_SOURCES], ignore_index=True)
    team_stats["team_canon"] = team_stats["team"].map(_canon_team_name)
    team_stats = team_stats.drop_duplicates(subset=["league", "season", "team_canon"], keep="last")

    home_stats = team_stats.rename(
        columns={
            "team_canon": "home_team_canon",
            "xG": "home_xg_season",
            "xGA": "home_xga_season",
            "xPTS": "home_xpts_season",
        }
    )[["league", "season", "home_team_canon", "home_xg_season", "home_xga_season", "home_xpts_season"]]

    away_stats = team_stats.rename(
        columns={
            "team_canon": "away_team_canon",
            "xG": "away_xg_season",
            "xGA": "away_xga_season",
            "xPTS": "away_xpts_season",
        }
    )[["league", "season", "away_team_canon", "away_xg_season", "away_xga_season", "away_xpts_season"]]

    merged = matches.merge(home_stats, on=["league", "season", "home_team_canon"], how="left")
    merged = merged.merge(away_stats, on=["league", "season", "away_team_canon"], how="left")

    # Fill missing xG/xGA/xPTS with league-season means, then global means.
    stat_cols = [
        "home_xg_season",
        "home_xga_season",
        "home_xpts_season",
        "away_xg_season",
        "away_xga_season",
        "away_xpts_season",
    ]
    for col in stat_cols:
        league_season_mean = merged.groupby(["league", "season"])[col].transform("mean")
        merged[col] = merged[col].fillna(league_season_mean)
        merged[col] = merged[col].fillna(merged[col].mean())

    merged["xg_diff_season"] = merged["home_xg_season"] - merged["away_xg_season"]
    merged["xga_diff_season"] = merged["home_xga_season"] - merged["away_xga_season"]
    merged["xpts_diff_season"] = merged["home_xpts_season"] - merged["away_xpts_season"]
    # Bookmaker implied probabilities often add predictive signal.
    inv_sum = (1 / merged["b365_home"]) + (1 / merged["b365_draw"]) + (1 / merged["b365_away"])
    merged["bookie_home_prob"] = (1 / merged["b365_home"]) / inv_sum
    merged["bookie_draw_prob"] = (1 / merged["b365_draw"]) / inv_sum
    merged["bookie_away_prob"] = (1 / merged["b365_away"]) / inv_sum
    for col in ["bookie_home_prob", "bookie_draw_prob", "bookie_away_prob"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged[col] = merged[col].fillna(merged[col].mean())

    result = merged[
        [
            "league",
            "season",
            "date",
            "home_team",
            "away_team",
            "fthg",
            "ftag",
            "home_xg_season",
            "home_xga_season",
            "home_xpts_season",
            "away_xg_season",
            "away_xga_season",
            "away_xpts_season",
            "xg_diff_season",
            "xga_diff_season",
            "xpts_diff_season",
            "bookie_home_prob",
            "bookie_draw_prob",
            "bookie_away_prob",
        ]
    ].sort_values("date")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    return result


if __name__ == "__main__":
    df = build_dataset()
    print(f"Built dataset rows: {len(df)}")
    print(f"Saved to: {OUTPUT_PATH}")

