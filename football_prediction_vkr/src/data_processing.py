from __future__ import annotations

import pandas as pd


WINDOWS = (5, 10)
RENAME_MAP = {
    "Div": "league",
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "fthg",
    "FTAG": "ftag",
}


def _validate_columns(df: pd.DataFrame) -> None:
    required_columns = {
        "league",
        "date",
        "home_team",
        "away_team",
        "fthg",
        "ftag",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _normalize_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.rename(columns=RENAME_MAP)
    return data


def _build_team_long_table(df: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "match_idx": df.index,
            "league": df["league"],
            "date": df["date"],
            "team": df["home_team"],
            "goals_scored": df["fthg"],
            "goals_conceded": df["ftag"],
            "points": (df["fthg"] > df["ftag"]).astype(int) * 3 + (df["fthg"] == df["ftag"]).astype(int),
            "is_home": 1,
        }
    )
    away = pd.DataFrame(
        {
            "match_idx": df.index,
            "league": df["league"],
            "date": df["date"],
            "team": df["away_team"],
            "goals_scored": df["ftag"],
            "goals_conceded": df["fthg"],
            "points": (df["ftag"] > df["fthg"]).astype(int) * 3 + (df["ftag"] == df["fthg"]).astype(int),
            "is_home": 0,
        }
    )
    return pd.concat([home, away], ignore_index=True)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build leakage-safe rolling features for match outcome prediction.

    The function enforces chronology by sorting by date and by using shifted
    rolling windows per (league, team), so each row uses only past matches.
    """
    data = _normalize_base_columns(df)
    _validate_columns(data)

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    data["target"] = 0
    data.loc[data["fthg"] > data["ftag"], "target"] = 1
    data.loc[data["fthg"] < data["ftag"], "target"] = -1

    long_df = _build_team_long_table(data)
    long_df = long_df.sort_values(["league", "team", "date", "match_idx"]).reset_index(drop=True)

    grouped = long_df.groupby(["league", "team"], sort=False)
    for window in WINDOWS:
        long_df[f"avg_goals_scored_{window}"] = grouped["goals_scored"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        long_df[f"avg_goals_conceded_{window}"] = grouped["goals_conceded"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        long_df[f"form_points_{window}"] = grouped["points"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

    home_features = long_df[long_df["is_home"] == 1][
        [
            "match_idx",
            "avg_goals_scored_5",
            "avg_goals_scored_10",
            "avg_goals_conceded_5",
            "avg_goals_conceded_10",
            "form_points_5",
            "form_points_10",
        ]
    ].rename(
        columns={
            "avg_goals_scored_5": "ht_avg_goals_scored_5",
            "avg_goals_scored_10": "ht_avg_goals_scored_10",
            "avg_goals_conceded_5": "ht_avg_goals_conceded_5",
            "avg_goals_conceded_10": "ht_avg_goals_conceded_10",
            "form_points_5": "ht_form_points_5",
            "form_points_10": "ht_form_points_10",
        }
    )

    away_features = long_df[long_df["is_home"] == 0][
        [
            "match_idx",
            "avg_goals_scored_5",
            "avg_goals_scored_10",
            "avg_goals_conceded_5",
            "avg_goals_conceded_10",
            "form_points_5",
            "form_points_10",
        ]
    ].rename(
        columns={
            "avg_goals_scored_5": "at_avg_goals_scored_5",
            "avg_goals_scored_10": "at_avg_goals_scored_10",
            "avg_goals_conceded_5": "at_avg_goals_conceded_5",
            "avg_goals_conceded_10": "at_avg_goals_conceded_10",
            "form_points_5": "at_form_points_5",
            "form_points_10": "at_form_points_10",
        }
    )

    result = data.merge(home_features, left_index=True, right_on="match_idx", how="left")
    result = result.merge(away_features, on="match_idx", how="left")
    result = result.drop(columns=["match_idx", "fthg", "ftag"])
    return result

