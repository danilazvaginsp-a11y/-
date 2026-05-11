from __future__ import annotations

from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from catboost import Pool
try:
    import shap
except Exception:
    shap = None

try:
    from src.build_dataset import build_dataset
    from src.data_processing import create_features
except ModuleNotFoundError:
    # Fallback for direct run in some environments.
    ROOT_DIR = Path(__file__).resolve().parent
    SRC_DIR = ROOT_DIR / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.append(str(SRC_DIR))
    from build_dataset import build_dataset
    from data_processing import create_features


st.set_page_config(page_title="Football Prediction VKR", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "catboost_model.joblib"

TARGET_LABELS = {1: "Победа хозяев", 0: "Ничья", -1: "Победа гостей"}
FEATURE_LABELS = {
    "ht_avg_goals_scored_5": "Хозяева: средние забитые (5)",
    "ht_avg_goals_scored_10": "Хозяева: средние забитые (10)",
    "ht_avg_goals_conceded_5": "Хозяева: средние пропущенные (5)",
    "ht_avg_goals_conceded_10": "Хозяева: средние пропущенные (10)",
    "at_avg_goals_scored_5": "Гости: средние забитые (5)",
    "at_avg_goals_scored_10": "Гости: средние забитые (10)",
    "at_avg_goals_conceded_5": "Гости: средние пропущенные (5)",
    "at_avg_goals_conceded_10": "Гости: средние пропущенные (10)",
    "ht_form_points_5": "Хозяева: очки за матч (5)",
    "ht_form_points_10": "Хозяева: очки за матч (10)",
    "at_form_points_5": "Гости: очки за матч (5)",
    "at_form_points_10": "Гости: очки за матч (10)",
    "home_xg_season": "Хозяева: xG сезона",
    "home_xga_season": "Хозяева: xGA сезона",
    "home_xpts_season": "Хозяева: xPTS сезона",
    "away_xg_season": "Гости: xG сезона",
    "away_xga_season": "Гости: xGA сезона",
    "away_xpts_season": "Гости: xPTS сезона",
    "xg_diff_season": "Разница xG",
    "xga_diff_season": "Разница xGA",
    "xpts_diff_season": "Разница xPTS",
}


@st.cache_resource
def load_model_artifact() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = build_dataset()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    featured = create_features(raw)
    return raw, featured


def teams_in_league(raw_df: pd.DataFrame, league: str) -> list[str]:
    league_df = raw_df[raw_df["league"] == league]
    teams = sorted(set(league_df["home_team"]).union(set(league_df["away_team"])))
    return teams


def _team_recent_stats(league_matches: pd.DataFrame, team: str) -> pd.DataFrame:
    mask = (league_matches["home_team"] == team) | (league_matches["away_team"] == team)
    team_matches = league_matches.loc[mask].copy().sort_values("date")

    team_matches["team_goals"] = np.where(
        team_matches["home_team"] == team, team_matches["fthg"], team_matches["ftag"]
    )
    team_matches["opp_goals"] = np.where(
        team_matches["home_team"] == team, team_matches["ftag"], team_matches["fthg"]
    )
    team_matches["points"] = np.select(
        [team_matches["team_goals"] > team_matches["opp_goals"], team_matches["team_goals"] == team_matches["opp_goals"]],
        [3, 1],
        default=0,
    ).astype(int)
    return team_matches


def _rolling_from_history(team_matches: pd.DataFrame, window: int) -> tuple[float, float, float]:
    recent = team_matches.tail(window)
    if recent.empty:
        return 0.0, 0.0, 0.0
    return (
        float(recent["team_goals"].mean()),
        float(recent["opp_goals"].mean()),
        float(recent["points"].mean()),
    )


def _team_form_summary(team_matches: pd.DataFrame, window: int = 10) -> dict:
    recent = team_matches.tail(window)
    if recent.empty:
        return {
            "Матчи": 0,
            "П": 0,
            "Н": 0,
            "Пор": 0,
            "Средние голы": 0.0,
            "Средние пропущенные": 0.0,
            "Средние очки": 0.0,
        }
    return {
        "Матчи": int(len(recent)),
        "П": int((recent["points"] == 3).sum()),
        "Н": int((recent["points"] == 1).sum()),
        "Пор": int((recent["points"] == 0).sum()),
        "Средние голы": float(recent["team_goals"].mean()),
        "Средние пропущенные": float(recent["opp_goals"].mean()),
        "Средние очки": float(recent["points"].mean()),
    }


def _compute_catboost_shap(
    model, single_match: pd.DataFrame, cat_features: list[str], class_order: list[int], probabilities: np.ndarray
) -> tuple[pd.DataFrame, float, int]:
    pool = Pool(single_match, cat_features=cat_features)
    shap_raw = model.get_feature_importance(pool, type="ShapValues")
    predicted_class = int(class_order[int(np.argmax(probabilities))])
    class_idx = list(class_order).index(predicted_class)
    if shap_raw.ndim == 3:
        class_values = shap_raw[0, class_idx, :]
    else:
        class_values = shap_raw[0, :]
    base_value = float(class_values[-1])
    contributions = class_values[:-1]
    df = pd.DataFrame({"feature": single_match.columns, "contribution": contributions})
    df["abs_contribution"] = df["contribution"].abs()
    df = df.sort_values("abs_contribution", ascending=False)
    return df, base_value, predicted_class


def _format_feature_value(value) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def build_match_features(
    raw_df: pd.DataFrame, league: str, home_team: str, away_team: str, feature_columns: list[str]
) -> pd.DataFrame:
    league_df = raw_df[raw_df["league"] == league].copy().sort_values("date")
    latest_season = league_df["season"].dropna().astype(str).iloc[-1]
    home_hist = _team_recent_stats(league_df, home_team)
    away_hist = _team_recent_stats(league_df, away_team)

    ht_scored_5, ht_conceded_5, ht_points_5 = _rolling_from_history(home_hist, 5)
    ht_scored_10, ht_conceded_10, ht_points_10 = _rolling_from_history(home_hist, 10)
    at_scored_5, at_conceded_5, at_points_5 = _rolling_from_history(away_hist, 5)
    at_scored_10, at_conceded_10, at_points_10 = _rolling_from_history(away_hist, 10)
    league_home_prob = float(pd.to_numeric(league_df.get("bookie_home_prob"), errors="coerce").mean())
    league_draw_prob = float(pd.to_numeric(league_df.get("bookie_draw_prob"), errors="coerce").mean())
    league_away_prob = float(pd.to_numeric(league_df.get("bookie_away_prob"), errors="coerce").mean())
    if np.isnan(league_home_prob):
        league_home_prob, league_draw_prob, league_away_prob = 0.45, 0.27, 0.28
    home_latest = (
        league_df[league_df["home_team"] == home_team]
        .sort_values("date")
        .tail(1)[["home_xg_season", "home_xga_season", "home_xpts_season"]]
    )
    if home_latest.empty:
        home_latest = pd.DataFrame([[0.0, 0.0, 0.0]], columns=["home_xg_season", "home_xga_season", "home_xpts_season"])
    away_latest = (
        league_df[league_df["away_team"] == away_team]
        .sort_values("date")
        .tail(1)[["away_xg_season", "away_xga_season", "away_xpts_season"]]
    )
    if away_latest.empty:
        away_latest = pd.DataFrame([[0.0, 0.0, 0.0]], columns=["away_xg_season", "away_xga_season", "away_xpts_season"])

    match_date = pd.Timestamp.today().normalize()
    hxg, hxga, hxpts = home_latest.iloc[0].tolist()
    axg, axga, axpts = away_latest.iloc[0].tolist()
    row = {
        "league": league,
        "season": latest_season,
        "date": match_date,
        "home_team": home_team,
        "away_team": away_team,
        "ht_avg_goals_scored_5": ht_scored_5,
        "ht_avg_goals_scored_10": ht_scored_10,
        "ht_avg_goals_conceded_5": ht_conceded_5,
        "ht_avg_goals_conceded_10": ht_conceded_10,
        "at_avg_goals_scored_5": at_scored_5,
        "at_avg_goals_scored_10": at_scored_10,
        "at_avg_goals_conceded_5": at_conceded_5,
        "at_avg_goals_conceded_10": at_conceded_10,
        "ht_form_points_5": ht_points_5,
        "ht_form_points_10": ht_points_10,
        "at_form_points_5": at_points_5,
        "at_form_points_10": at_points_10,
        "home_xg_season": hxg,
        "home_xga_season": hxga,
        "home_xpts_season": hxpts,
        "away_xg_season": axg,
        "away_xga_season": axga,
        "away_xpts_season": axpts,
        "xg_diff_season": hxg - axg,
        "xga_diff_season": hxga - axga,
        "xpts_diff_season": hxpts - axpts,
        "bookie_home_prob": league_home_prob,
        "bookie_draw_prob": league_draw_prob,
        "bookie_away_prob": league_away_prob,
    }

    feature_row = pd.DataFrame([row]).drop(columns=["date"])
    return feature_row.reindex(columns=feature_columns)


def show_prediction_page(raw_df: pd.DataFrame, artifact: dict) -> None:
    st.subheader("Прогнозирование матча")

    leagues = sorted(raw_df["league"].dropna().unique().tolist())
    league = st.selectbox("Шаг 1: Выберите лигу", leagues)

    teams = teams_in_league(raw_df, league)
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Шаг 2: Домашняя команда", teams)
    with col2:
        away_team = st.selectbox("Шаг 2: Гостевая команда", teams, index=min(1, len(teams) - 1))

    league_df = raw_df[raw_df["league"] == league].copy().sort_values("date")
    home_hist = _team_recent_stats(league_df, home_team)
    away_hist = _team_recent_stats(league_df, away_team)
    home_summary = _team_form_summary(home_hist, window=10)
    away_summary = _team_form_summary(away_hist, window=10)

    st.markdown("### Сравнение выбранных команд (последние 10 матчей)")
    comparison_df = pd.DataFrame(
        {
            "Показатель": list(home_summary.keys()),
            home_team: list(home_summary.values()),
            away_team: list(away_summary.values()),
        }
    )
    st.table(comparison_df)

    if st.button("Сделать прогноз"):
        if home_team == away_team:
            st.error("Домашняя и гостевая команды должны быть разными.")
            return

        model = artifact["model"]
        feature_columns = artifact["feature_columns"]
        cat_features = artifact["cat_features"]
        class_order = artifact["class_order"]
        single_match = build_match_features(raw_df, league, home_team, away_team, feature_columns)

        probabilities = model.predict_proba(single_match)[0]
        probability_map = {int(cls): float(prob) for cls, prob in zip(class_order, probabilities)}

        st.markdown("### Вероятности исходов")
        for cls in [1, 0, -1]:
            st.write(f"**{TARGET_LABELS[cls]}**: {probability_map.get(cls, 0.0):.2%}")

        st.markdown("### За счет чего модель оценила шансы")
        try:
            shap_df, base_value, predicted_class = _compute_catboost_shap(
                model, single_match, cat_features, class_order, probabilities
            )
            top_df = shap_df.head(8).copy()
            top_df["label"] = top_df["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
            top_df["direction"] = np.where(top_df["contribution"] >= 0, "повышает шанс", "снижает шанс")

            st.write(f"Наиболее вероятный исход: **{TARGET_LABELS[predicted_class]}**")
            st.write(f"Базовый вклад модели (bias): `{base_value:.4f}`")

            top_df["value"] = top_df["feature"].map(lambda f: _format_feature_value(single_match.iloc[0][f]))
            st.table(
                top_df[["label", "value", "contribution", "direction"]]
                .rename(
                    columns={
                        "label": "Признак",
                        "value": "Значение для матча",
                        "contribution": "Вклад",
                        "direction": "Эффект",
                    }
                )
            )

            st.markdown("### SHAP-график факторов")
            if shap is None:
                st.info("Библиотека SHAP не загружена. Показан альтернативный график вкладов.")
                fig_fallback, ax_fallback = plt.subplots(figsize=(10, 5))
                bar_values = top_df["contribution"].values
                bar_labels = top_df["label"].values
                bar_colors = np.where(bar_values >= 0, "#2ca02c", "#d62728")
                y_pos = np.arange(len(bar_labels))
                ax_fallback.barh(y_pos, bar_values, color=bar_colors)
                ax_fallback.set_yticks(y_pos)
                ax_fallback.set_yticklabels(bar_labels)
                ax_fallback.axvline(0, color="black", linewidth=1)
                ax_fallback.set_xlabel("Вклад в выбранный исход")
                ax_fallback.set_title("Топ факторов прогноза")
                ax_fallback.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig_fallback, clear_figure=True)
            else:
                try:
                    explanation = shap.Explanation(
                        values=top_df["contribution"].values,
                        base_values=base_value,
                        data=single_match[top_df["feature"]].iloc[0].values,
                        feature_names=top_df["label"].tolist(),
                    )
                    fig2, _ = plt.subplots(figsize=(10, 5))
                    shap.plots.waterfall(explanation, max_display=8, show=False)
                    st.pyplot(fig2, clear_figure=True)

                    st.markdown("### Расшифровка SHAP")
                    st.markdown(
                        "- Положительный вклад признака увеличивает вероятность выбранного исхода.\n"
                        "- Отрицательный вклад уменьшает вероятность выбранного исхода.\n"
                        "- Чем больше модуль SHAP-вклада, тем сильнее влияние признака.\n"
                        "- `base value` — базовый уровень прогноза до учета признаков матча."
                    )
                    st.markdown("**Что учитывается в SHAP для вашего прогноза:**")
                    st.markdown(
                        "- Форма команд: средние голы/пропущенные и очки за 5 и 10 матчей.\n"
                        "- Сезонная сила команд: `xG`, `xGA`, `xPTS` для хозяев и гостей.\n"
                        "- Разности сил: `xg_diff_season`, `xga_diff_season`, `xpts_diff_season`.\n"
                        "- Категориальный контекст: лига, сезон, домашняя и гостевая команда."
                    )
                except Exception as shap_exc:
                    st.info(f"SHAP-график недоступен в текущем окружении: {shap_exc}")
        except Exception as exc:
            st.warning(f"Не удалось построить объяснение факторов: {exc}")


def show_statistics_page(raw_df: pd.DataFrame) -> None:
    st.subheader("Статистика команд")

    leagues = sorted(raw_df["league"].dropna().unique().tolist())
    league = st.selectbox("Шаг 1: Выберите лигу", leagues, key="stats_league")
    teams = teams_in_league(raw_df, league)
    team = st.selectbox("Шаг 2: Выберите команду", teams, key="stats_team")

    league_df = raw_df[raw_df["league"] == league].copy()
    team_matches = _team_recent_stats(league_df, team)
    last_15 = team_matches.tail(15).copy()

    st.markdown("### Последние 15 матчей")
    display_df = last_15.copy()
    display_df["Локация"] = np.where(display_df["home_team"] == team, "Дома", "В гостях")
    display_df["Соперник"] = np.where(display_df["home_team"] == team, display_df["away_team"], display_df["home_team"])
    display_df["Счет (выбранная команда)"] = (
        display_df["team_goals"].astype(int).astype(str) + ":" + display_df["opp_goals"].astype(int).astype(str)
    )
    display_df["Результат"] = np.select(
        [display_df["team_goals"] > display_df["opp_goals"], display_df["team_goals"] == display_df["opp_goals"]],
        ["Победа", "Ничья"],
        default="Поражение",
    )
    # Recalculate points directly from selected-team score for reliability in UI.
    display_df["Очки"] = np.select(
        [display_df["team_goals"] > display_df["opp_goals"], display_df["team_goals"] == display_df["opp_goals"]],
        [3, 1],
        default=0,
    ).astype(int)

    pretty_df = display_df[
        ["date", "Локация", "Соперник", "Счет (выбранная команда)", "Результат", "Очки"]
    ].rename(columns={"date": "Дата"})
    st.table(pretty_df)

    wins = int((display_df["Очки"] == 3).sum())
    draws = int((display_df["Очки"] == 1).sum())
    losses = int((display_df["Очки"] == 0).sum())
    st.markdown(f"**Краткая статистика (Победы-Ничьи-Поражения):** {wins}-{draws}-{losses}")

    with st.expander("Подробно: последние 5 матчей"):
        st.table(pretty_df.tail(5))

    st.markdown("### Динамика формы (очки за матч)")
    form_df = last_15[["date", "points"]].copy()
    form_df["rolling_points_5"] = form_df["points"].rolling(window=5, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(form_df["date"], form_df["points"], marker="o", label="Очки за матч")
    ax.plot(form_df["date"], form_df["rolling_points_5"], linestyle="--", label="Скользящее среднее (5)")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Очки")
    ax.set_title(f"Форма команды: {team}")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    recent_5_avg = float(form_df["points"].tail(5).mean()) if not form_df.empty else 0.0
    overall_avg = float(form_df["points"].mean()) if not form_df.empty else 0.0
    trend_delta = recent_5_avg - overall_avg
    if recent_5_avg >= 2.0:
        form_level = "хорошая"
    elif recent_5_avg >= 1.2:
        form_level = "средняя"
    else:
        form_level = "слабая"

    trend_text = "улучшение" if trend_delta > 0.2 else ("спад" if trend_delta < -0.2 else "стабильная")
    st.markdown(
        f"**Вывод по форме:** у команды сейчас **{form_level} форма**. "
        f"Средние очки за последние 5 матчей: **{recent_5_avg:.2f}**, "
        f"в среднем за выбранный отрезок: **{overall_avg:.2f}**. "
        f"Текущий тренд: **{trend_text}**."
    )


def main() -> None:
    st.title("Прогнозирование исходов футбольных матчей")
    st.sidebar.title("Навигация")
    page = st.sidebar.radio("Выберите страницу", ["Прогнозирование матча", "Статистика команд"])

    raw_df, _ = load_dataframes()
    artifact = load_model_artifact()

    if page == "Прогнозирование матча":
        show_prediction_page(raw_df, artifact)
    else:
        show_statistics_page(raw_df)


if __name__ == "__main__":
    main()

