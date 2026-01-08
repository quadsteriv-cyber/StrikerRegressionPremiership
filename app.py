#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPFL Striker Chance-Generation Drivers (StatsBomb API) — Streamlit App

What this app does:
- Fetches StatsBomb IQ season *player stats* and *team stats* for selected leagues/seasons
- Trains a league-specific model (Scottish Premiership only) on STs (>=900 mins)
- Primary target: np_xg_90 (non-penalty xG per 90)
- Uses team-season stats as CONTROLS (context absorption), not as “traits”
- Uses Elastic Net + stability selection to identify the most stable player-level predictors
- Applies age + league filters to produce a scouting shortlist

Notes:
- This is associational (controls reduce bias, not causal proof).
- API endpoints and many column names match StatsBomb season stats specs.
"""

from __future__ import annotations

import time
from datetime import date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold


# -----------------------------
# 0) CONFIG — leagues/seasons
# -----------------------------

# Same IDs used in your previous file (keep in sync with your licensed access).
LEAGUE_NAMES: Dict[int, str] = {
    4: "League One",
    5: "League Two",
    51: "Scottish Premiership",
    107: "Irish Premier Division",
    1385: "Scottish Championship",
}

# Season IDs (from your previous working file).
# These correspond to StatsBomb “season_id” values, not calendar years.
COMPETITION_SEASONS: Dict[int, List[int]] = {
    4: [235, 281, 317, 318],
    5: [235, 281, 317, 318],
    51: [235, 281, 317, 318],
    107: [106, 107, 282, 315],
    1385: [235, 281, 317, 318],
}

# Filters requested
DOMESTIC_LEAGUE_IDS = [51, 1385, 4, 5, 107]
SCOTTISH_LEAGUE_IDS = [51, 1385]

# Position filtering for “ST-only”
# StatsBomb season player-stats sometimes returns a string label in primary_position;
# if it's numeric in your account, you can adjust this.
ST_POSITION_LABELS = {
    "Centre Forward",
    "Left Centre Forward",
    "Right Centre Forward",
    "Secondary Striker",
    "Striker",
    "ST",
}

PRIMARY_TARGET = "np_xg_90"  # after prefix stripping


# -----------------------------
# 1) API Fetching
# -----------------------------

@st.cache_resource(ttl=3600)
def fetch_player_season_stats(auth: Tuple[str, str]) -> Optional[pd.DataFrame]:
    """Fetch player-season stats for all configured league/season combos."""
    # Quick auth test (as in your prior file)
    try:
        test_url = "https://data.statsbombservices.com/api/v4/competitions"
        r = requests.get(test_url, auth=auth, timeout=30)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Authentication failed. Check your StatsBomb credentials.\n\n{e}")
        return None

    total_requests = sum(len(v) for v in COMPETITION_SEASONS.values())
    progress = st.progress(0)
    status = st.empty()

    frames: List[pd.DataFrame] = []
    done = 0
    failures = 0

    for comp_id, season_ids in COMPETITION_SEASONS.items():
        league_name = LEAGUE_NAMES.get(comp_id, f"Competition {comp_id}")
        for season_id in season_ids:
            done += 1
            progress.progress(done / total_requests)
            status.text(f"Loading player-stats: {league_name} | season_id={season_id} ({done}/{total_requests})")

            url = f"https://data.statsbombservices.com/api/v1/competitions/{comp_id}/seasons/{season_id}/player-stats"
            try:
                resp = requests.get(url, auth=auth, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    failures += 1
                    continue
                df = pd.json_normalize(data)
                if df.empty:
                    failures += 1
                    continue
                df["league_name"] = league_name
                df["competition_id"] = comp_id
                df["season_id"] = season_id
                frames.append(df)
            except Exception:
                failures += 1
                continue

    progress.empty()
    status.empty()

    if not frames:
        st.error("No player-stats returned. Check league/season IDs and your licensed access.")
        return None

    if failures:
        st.warning(f"Some league/season requests failed: {failures}.")

    return pd.concat(frames, ignore_index=True)


@st.cache_resource(ttl=3600)
def fetch_team_season_stats(auth: Tuple[str, str]) -> Optional[pd.DataFrame]:
    """Fetch team-season stats for all configured league/season combos."""
    total_requests = sum(len(v) for v in COMPETITION_SEASONS.values())
    progress = st.progress(0)
    status = st.empty()

    frames: List[pd.DataFrame] = []
    done = 0
    failures = 0

    for comp_id, season_ids in COMPETITION_SEASONS.items():
        league_name = LEAGUE_NAMES.get(comp_id, f"Competition {comp_id}")
        for season_id in season_ids:
            done += 1
            progress.progress(done / total_requests)
            status.text(f"Loading team-stats: {league_name} | season_id={season_id} ({done}/{total_requests})")

            # Team stats endpoint per StatsBomb season team stats spec:
            # https://data.statsbombservices.com/api/v2/competitions/{comp}/seasons/{season}/team-stats
            url = f"https://data.statsbombservices.com/api/v2/competitions/{comp_id}/seasons/{season_id}/team-stats"
            try:
                resp = requests.get(url, auth=auth, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    failures += 1
                    continue
                df = pd.json_normalize(data)
                if df.empty:
                    failures += 1
                    continue
                df["league_name"] = league_name
                df["competition_id"] = comp_id
                df["season_id"] = season_id
                frames.append(df)
            except Exception:
                failures += 1
                continue

    progress.empty()
    status.empty()

    if not frames:
        st.error("No team-stats returned. Check league/season IDs and your licensed access.")
        return None

    if failures:
        st.warning(f"Some league/season team-stats requests failed: {failures}.")

    return pd.concat(frames, ignore_index=True)


# -----------------------------
# 2) Data processing
# -----------------------------

@st.cache_data(ttl=3600)
def process_player_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Strip prefixes, compute age from birth_date, normalize some strings."""
    df = raw.copy()

    # Strip player_season_ prefix (as in your prior file)
    df.columns = [c.replace("player_season_", "") for c in df.columns]

    # Clean strings
    for col in ["player_name", "team_name", "league_name", "season_name"]:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    # Age from birth_date
    def calculate_age(birth_date_str):
        if pd.isna(birth_date_str):
            return np.nan
        try:
            bd = pd.to_datetime(birth_date_str).date()
            today = date.today()
            return today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
        except Exception:
            return np.nan

    if "birth_date" in df.columns:
        df["age"] = df["birth_date"].apply(calculate_age)
    else:
        df["age"] = np.nan

    # Canonical season end-year (useful for grouping/sanity)
    def canonical_season(season_str):
        try:
            if isinstance(season_str, str) and "/" in season_str:
                return int(season_str.split("/")[1])
            return int(season_str)
        except Exception:
            return np.nan

    if "season_name" in df.columns:
        df["canonical_season"] = df["season_name"].apply(canonical_season)
    else:
        df["canonical_season"] = np.nan

    return df


@st.cache_data(ttl=3600)
def process_team_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Strip team_season_ prefix to team_*, keep ids/names intact."""
    df = raw.copy()
    df.columns = [c.replace("team_season_", "team_") for c in df.columns]

    for col in ["team_name", "league_name", "season_name"]:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    return df


def join_player_team(player_df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    """Attach team controls to each player-season row (same team + comp + season)."""
    # Prefer IDs if present
    join_cols = []
    if {"team_id", "competition_id", "season_id"}.issubset(team_df.columns) and {"team_id", "competition_id", "season_id"}.issubset(player_df.columns):
        join_cols = ["team_id", "competition_id", "season_id"]
    else:
        join_cols = ["team_name", "competition_id", "season_id"]

    # Keep only one row per team-season
    team_cols = [c for c in team_df.columns if c.startswith("team_")] + ["team_id", "team_name", "competition_id", "season_id"]
    team_cols = [c for c in team_cols if c in team_df.columns]
    team_dedup = team_df[team_cols].drop_duplicates(subset=join_cols)

    merged = player_df.merge(team_dedup, on=join_cols, how="left", suffixes=("", "_team"))
    return merged


# -----------------------------
# 3) Filtering helpers (age/league/minutes/ST)
# -----------------------------

def is_striker_label(x) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip()
    if s in ST_POSITION_LABELS:
        return True
    # Fallback heuristic: if your API returns richer labels
    return ("Forward" in s) or ("Striker" in s) or (s == "9")


def apply_common_filters(
    df: pd.DataFrame,
    min_minutes: int,
    age_range: Tuple[int, int],
    league_filter: str,
) -> pd.DataFrame:
    out = df.copy()

    # League filter
    if "competition_id" in out.columns:
        if league_filter == "Domestic Leagues":
            out = out[out["competition_id"].isin(DOMESTIC_LEAGUE_IDS)]
        elif league_filter == "Scottish Leagues":
            out = out[out["competition_id"].isin(SCOTTISH_LEAGUE_IDS)]

    # ST only
    if "primary_position" in out.columns:
        out = out[out["primary_position"].apply(is_striker_label)]

    # Minutes (player stats prefix stripped => minutes)
    if "minutes" in out.columns:
        out = out[out["minutes"] >= min_minutes]
    elif "player_season_minutes" in out.columns:
        out = out[out["player_season_minutes"] >= min_minutes]

    # Age filter — strict: unknown ages are excluded from the dataset as requested
    if "age" in out.columns:
        out = out[out["age"].notna()]
        out = out[(out["age"] >= age_range[0]) & (out["age"] <= age_range[1])]

    return out


# -----------------------------
# 4) Modelling (Elastic Net + stability selection)
# -----------------------------

def get_team_control_columns(df: pd.DataFrame) -> List[str]:
    """
    Team controls: include only a compact set that captures opportunity/supply/style.
    We include only columns that exist in df.
    """
    wanted = [
        "team_np_xg_pg",
        "team_np_shots_pg",
        "team_possession",
        "team_passes_inside_box_pg",
        "team_successful_crosses_into_box_pg",
        "team_deep_completions_pg",
        "team_directness",
        "team_pace_towards_goal",
        "team_counter_attacking_shots_pg",
    ]
    return [c for c in wanted if c in df.columns]


def build_feature_matrix(
    df: pd.DataFrame,
    target: str,
    include_team_controls: bool = True,
    add_age_poly: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Builds X, y plus metadata: player_features, team_features
    - Player features: all numeric player columns excluding banned/leakage
    - Team controls: small list from get_team_control_columns()
    """

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataframe columns.")

    # Identify “player” numeric columns: exclude obvious IDs + strings + target/leakage
    non_feature_cols = {
        "player_id", "player_name", "team_id", "team_name",
        "competition_id", "league_name", "season_id", "season_name",
        "country_id", "birth_date", "player_height", "player_weight",
        "primary_position", "secondary_position",
        "canonical_season",
    }

    # Leak / tautology bans (anything too close to outcome)
    # We are modelling np_xg_90, so bans include np_xg itself (target), and goal/finishing variables.
    bans = {
        target,
        # Goal outcomes / finishing / conversion
        "goals_90", "npg_90", "goals", "npg",
        "conversion_ratio", "shot_on_target_ratio",
        "over_under_performance_90", "penalty_goals_90",
        "penalty_conversion_ratio",
        # direct derived combos (if present)
        "shots_faced_90", "save_ratio",
    }

    # Numeric columns
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    player_num = [c for c in num_cols if (c not in non_feature_cols) and (not c.startswith("team_")) and (c not in bans)]

    team_cols = get_team_control_columns(df) if include_team_controls else []

    X = df[player_num + team_cols].copy()

    # Age controls
    player_age_cols = []
    if "age" in df.columns:
        X["age"] = df["age"].astype(float)
        player_age_cols.append("age")
        if add_age_poly:
            X["age_sq"] = (df["age"].astype(float) ** 2)
            player_age_cols.append("age_sq")

    y = df[target].astype(float)

    # Keep track: “player features” for interpretability exclude team cols but include age terms
    player_features = player_num + player_age_cols
    team_features = team_cols

    # Drop any columns with all-NaN
    X = X.dropna(axis=1, how="all")

    # Align feature lists with dropped columns
    player_features = [c for c in player_features if c in X.columns]
    team_features = [c for c in team_features if c in X.columns]

    # Minimal row filtering: drop rows where target missing
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    # Fill remaining NaNs with column medians (simple, robust default)
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    return X, y, player_features, team_features


def fit_elastic_net_cv(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Pipeline:
    """
    Fits an ElasticNetCV model wrapped in a pipeline (standardization + ElasticNetCV).
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    model = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("enet", ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            alphas=np.logspace(-4, 1, 50),
            cv=cv,
            max_iter=20000,
            random_state=random_state,
        )),
    ])
    model.fit(X.values, y.values)
    return model


def stability_selection(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    n_boot: int = 120,
    sample_frac: float = 0.85,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Bootstrap stability selection:
    - Refit ElasticNetCV on bootstrap resamples
    - Track non-zero coefficient frequency and mean absolute coefficient magnitude (standardized space)

    Note: Coefs come from standardized feature space due to StandardScaler in pipeline.
    """
    rng = np.random.default_rng(random_state)
    n = len(y)

    sel_counts = dict((f, 0) for f in feature_names)
    coef_sums = dict((f, 0.0) for f in feature_names)

    X_arr = X.values
    y_arr = y.values

    for b in range(n_boot):
        idx = rng.choice(np.arange(n), size=max(5, int(sample_frac * n)), replace=True)
        Xb = X_arr[idx, :]
        yb = y_arr[idx]

        # Fit
        model = fit_elastic_net_cv(pd.DataFrame(Xb, columns=feature_names), pd.Series(yb), random_state=int(rng.integers(1, 1_000_000)))

        coefs = model.named_steps["enet"].coef_
        # selected if coefficient != 0
        for f, c in zip(feature_names, coefs):
            if c != 0:
                sel_counts[f] += 1
                coef_sums[f] += abs(float(c))

    out = pd.DataFrame({
        "feature": feature_names,
        "selection_freq": [sel_counts[f] / n_boot for f in feature_names],
        "mean_abs_coef": [coef_sums[f] / max(1, sel_counts[f]) if sel_counts[f] > 0 else 0.0 for f in feature_names],
    })
    out = out.sort_values(["selection_freq", "mean_abs_coef"], ascending=False).reset_index(drop=True)
    return out


# -----------------------------
# 5) Streamlit UI
# -----------------------------

def main():
    st.set_page_config(page_title="SPFL ST xG Drivers", layout="wide")
    st.title("SPFL Striker Chance-Generation Drivers (StatsBomb IQ season stats)")

    with st.sidebar:
        st.header("1) StatsBomb Credentials")
        sb_user = st.text_input("Username", value=st.secrets.get("STATS_BOMB_USER", ""), type="default")
        sb_pass = st.text_input("Password", value=st.secrets.get("STATS_BOMB_PASS", ""), type="password")
        auth = (sb_user, sb_pass)

        st.divider()
        st.header("2) Filters (applies to scouting pool)")
        league_filter = st.selectbox("League Filter", ["All Leagues", "Domestic Leagues", "Scottish Leagues"], index=1)
        min_minutes = st.slider("Minimum Minutes Played", 0, 3500, 900, 100)
        age_range = st.slider("Age Range", 16, 40, (18, 30))

        st.divider()
        st.header("3) Model Settings")
        include_team_controls = st.checkbox("Include team-season controls (recommended)", value=True)
        add_age_poly = st.checkbox("Use age + age² (recommended)", value=True)

        n_boot = st.slider("Stability selection bootstraps", 30, 300, 120, 10)
        sample_frac = st.slider("Bootstrap sample fraction", 0.50, 1.00, 0.85, 0.05)

        top_k = st.slider("Show Top-K predictors", 3, 20, 5, 1)

        st.divider()
        load_btn = st.button("Load / Refresh Data", type="primary")

    # Persist data in session state
    if load_btn or ("player_raw" not in st.session_state):
        if not sb_user or not sb_pass:
            st.warning("Enter StatsBomb credentials in the sidebar, then click **Load / Refresh Data**.")
            return

        with st.spinner("Fetching player season stats..."):
            player_raw = fetch_player_season_stats(auth)
        if player_raw is None:
            return

        with st.spinner("Fetching team season stats..."):
            team_raw = fetch_team_season_stats(auth)
        if team_raw is None:
            return

        st.session_state.player_raw = player_raw
        st.session_state.team_raw = team_raw

        st.success("Data loaded.")

    player_raw = st.session_state.get("player_raw")
    team_raw = st.session_state.get("team_raw")
    if player_raw is None or team_raw is None:
        st.warning("No data loaded yet.")
        return

    # Process + join
    player_df = process_player_data(player_raw)
    team_df = process_team_data(team_raw)
    merged = join_player_team(player_df, team_df)

    # -----------------------------
    # TRAINING SET: SPFL-only STs >=900 mins
    # -----------------------------
    train_base = merged.copy()
    train_base = train_base[train_base["competition_id"] == 51]  # SPFL-only for training

    # Training uses ST-only + minutes>=900; age is NOT filtered for training by default (you can if you want)
    train_base = apply_common_filters(
        train_base,
        min_minutes=900,
        age_range=(16, 50),  # keep wide for training, avoid truncation effects
        league_filter="All Leagues",
    )

    if PRIMARY_TARGET not in train_base.columns:
        st.error(f"Target '{PRIMARY_TARGET}' not found. Check your player-stats endpoint/columns.")
        st.write("Available columns:", sorted(train_base.columns))
        return

    st.subheader("Training Set Summary (SPFL only)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (player-seasons)", f"{len(train_base):,}")
    c2.metric("Unique players", f"{train_base['player_id'].nunique() if 'player_id' in train_base.columns else train_base['player_name'].nunique():,}")
    c3.metric("Seasons", f"{train_base['season_id'].nunique():,}")
    c4.metric("Teams", f"{train_base['team_name'].nunique() if 'team_name' in train_base.columns else '—'}")

    with st.expander("Preview training rows"):
        preview_cols = [c for c in ["player_name", "team_name", "season_name", "minutes", "age", PRIMARY_TARGET] if c in train_base.columns]
        st.dataframe(train_base[preview_cols].sort_values(preview_cols[-1], ascending=False).head(30), use_container_width=True)

    # -----------------------------
    # Fit model + stability
    # -----------------------------
    st.subheader("Model: Elastic Net + Stability Selection")
    try:
        X, y, player_feats, team_feats = build_feature_matrix(
            train_base,
            target=PRIMARY_TARGET,
            include_team_controls=include_team_controls,
            add_age_poly=add_age_poly,
        )
    except Exception as e:
        st.error(f"Feature matrix build failed: {e}")
        return

    if len(X) < 25:
        st.error("Training set too small after filters. Consider adding more SPFL seasons or lowering minutes threshold.")
        return

    all_features = list(X.columns)

    with st.spinner("Fitting ElasticNetCV on full training set..."):
        full_model = fit_elastic_net_cv(X, y, random_state=42)

    enet = full_model.named_steps["enet"]
    st.caption(
        f"Chosen alpha={enet.alpha_:.6f} | l1_ratio={enet.l1_ratio_} | "
        f"Non-zero coefs={int(np.sum(enet.coef_ != 0))}/{len(all_features)}"
    )

    with st.spinner("Running stability selection (bootstraps)..."):
        stab = stability_selection(
            X=X,
            y=y,
            feature_names=all_features,
            n_boot=n_boot,
            sample_frac=float(sample_frac),
            random_state=42,
        )

    # Separate “player traits” from “team controls” for interpretation
    trait_rows = stab[stab["feature"].isin(player_feats)].copy()
    control_rows = stab[stab["feature"].isin(team_feats)].copy()

    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        st.markdown(f"### Top player-level predictors of **{PRIMARY_TARGET}** (SPFL STs)")
        st.dataframe(
            trait_rows.head(top_k).assign(
                selection_freq=lambda d: (100 * d["selection_freq"]).round(1),
                mean_abs_coef=lambda d: d["mean_abs_coef"].round(3),
            ).rename(columns={"selection_freq": "selected_%", "mean_abs_coef": "mean|coef|"}),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "These are *player* features only. Team controls are included (if enabled) to absorb context but are not treated as recruitable traits."
        )

        st.markdown("### Full ranked player predictors")
        st.dataframe(
            trait_rows.assign(
                selection_freq=lambda d: (100 * d["selection_freq"]).round(1),
                mean_abs_coef=lambda d: d["mean_abs_coef"].round(3),
            ).rename(columns={"selection_freq": "selected_%", "mean_abs_coef": "mean|coef|"}),
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.markdown("### Team controls (context absorbers)")
        if not include_team_controls:
            st.info("Team controls disabled.")
        else:
            if control_rows.empty:
                st.warning("No team control columns found in the merged dataset.")
            else:
                st.dataframe(
                    control_rows.assign(
                        selection_freq=lambda d: (100 * d["selection_freq"]).round(1),
                        mean_abs_coef=lambda d: d["mean_abs_coef"].round(3),
                    ).rename(columns={"selection_freq": "selected_%", "mean_abs_coef": "mean|coef|"}),
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("### Model sanity snapshot")
        # Show correlation between predicted and actual on training data (in-sample; just a health check)
        yhat = full_model.predict(X.values)
        corr = np.corrcoef(y.values, yhat)[0, 1]
        st.metric("In-sample corr(y, ŷ)", f"{corr:.3f}")

        st.caption("This is not a proper forecast metric — just confirms the pipeline is functioning.")

    st.divider()

    # -----------------------------
    # SCOUTING POOL: apply user filters (age + league)
    # -----------------------------
    st.subheader("Scouting Shortlist (filters applied)")

    scouting = apply_common_filters(
        merged,
        min_minutes=min_minutes,
        age_range=age_range,
        league_filter=league_filter,
    )

    if scouting.empty:
        st.warning("No players match your filters. Widen age range, minutes, or league filter.")
        return

    # Build X for scouting using same features (must have identical columns)
    # Use the same preprocessing: fill missing with medians computed from training X for stability.
    X_train = X.copy()
    train_medians = X_train.median(numeric_only=True)

    # Ensure scouting has all columns
    scout_X = scouting.reindex(columns=all_features).copy()
    for c in all_features:
        if c not in scout_X.columns:
            scout_X[c] = np.nan
    scout_X = scout_X[all_features]

    # Fill NaNs with training medians (NOT scouting medians) to avoid leaking/shift
    for c in scout_X.columns:
        if scout_X[c].isna().any():
            scout_X[c] = scout_X[c].fillna(float(train_medians.get(c, 0.0)))

    preds = full_model.predict(scout_X.values)
    scouting = scouting.copy()
    scouting["pred_np_xg_90"] = preds
    scouting["residual"] = scouting[PRIMARY_TARGET].astype(float) - scouting["pred_np_xg_90"]

    # Display
    show_cols = [c for c in [
        "player_name", "age", "primary_position", "team_name", "league_name", "season_name",
        "minutes", PRIMARY_TARGET, "pred_np_xg_90", "residual"
    ] if c in scouting.columns]

    # rank by predicted chance generation
    topn = st.slider("Shortlist size", 10, 100, 30, 5)
    shortlist = scouting.sort_values("pred_np_xg_90", ascending=False).head(topn).copy()

    # Formatting
    for col in ["pred_np_xg_90", PRIMARY_TARGET, "residual"]:
        if col in shortlist.columns:
            shortlist[col] = shortlist[col].astype(float).round(3)

    st.dataframe(shortlist[show_cols], use_container_width=True, hide_index=True)

    st.caption(
        "Tip: Residual > 0 = producing more np_xg than the model expects given traits + team context (could be role/system quirks). "
        "Residual < 0 = underproducing vs traits/context (possible buy-low or role mismatch)."
    )

    with st.expander("Download shortlist as CSV"):
        csv = shortlist[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="spfl_st_shortlist.csv", mime="text/csv")

    st.divider()

    # -----------------------------
    # “Top 5 traits” scorecard for a selected player
    # -----------------------------
    st.subheader("Trait Scorecard (Top-K predictors)")
    trait_top = trait_rows.head(top_k)["feature"].tolist()

    # pick a player in the shortlisted pool
    key = "player_select_key"
    player_options = shortlist["player_name"].astype(str).tolist() if "player_name" in shortlist.columns else []
    if player_options:
        selected_player = st.selectbox("Select a player from the shortlist", player_options, index=0, key=key)
        row = shortlist[shortlist["player_name"].astype(str) == str(selected_player)].head(1)
        if not row.empty:
            r = row.iloc[0]
            st.markdown(f"**{r.get('player_name','')}** — {r.get('team_name','')} ({r.get('league_name','')}, {r.get('season_name','')})")
            st.write(f"Minutes: **{int(r.get('minutes',0))}** | Age: **{int(r.get('age',0))}**")
            st.write(f"Actual {PRIMARY_TARGET}: **{r.get(PRIMARY_TARGET, np.nan):.3f}** | Predicted: **{r.get('pred_np_xg_90', np.nan):.3f}**")

            # Show z-scores within the *scouting pool* for interpretability
            score_df = []
            for f in trait_top:
                if f in scouting.columns and pd.api.types.is_numeric_dtype(scouting[f]):
                    mu = scouting[f].astype(float).mean()
                    sd = scouting[f].astype(float).std(ddof=0)
                    val = float(r.get(f, np.nan))
                    z = (val - mu) / sd if (sd and not np.isnan(val)) else np.nan
                    score_df.append({"trait": f, "value": val, "z_in_pool": z})
            if score_df:
                out = pd.DataFrame(score_df)
                out["value"] = out["value"].round(3)
                out["z_in_pool"] = out["z_in_pool"].round(2)
                st.dataframe(out, use_container_width=True, hide_index=True)
            else:
                st.info("No trait columns available for scorecard (check column names).")
    else:
        st.info("No shortlist players available for scorecard.")


if __name__ == "__main__":
    main()
