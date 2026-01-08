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

# Full league mapping (52 combos)
LEAGUE_NAMES: Dict[int, str] = {
    4: "League One",
    5: "League Two",
    51: "Premiership",
    65: "National League",
    76: "Liga",
    78: "1. HNL",
    89: "USL Championship",
    106: "Veikkausliiga",
    107: "Premier Division",
    129: "Championnat National",
    166: "Premier League 2 Division One",
    179: "3. Liga",
    260: "1st Division",
    1035: "First Division B",
    1385: "Championship",
    1442: "1. Division",
    1581: "2. Liga",
    1607: "Úrvalsdeild",
    1778: "First Division",
    1848: "I Liga",
    1865: "First League",
}

# Season IDs - full 52-combo mapping
COMPETITION_SEASONS: Dict[int, List[int]] = {
    4: [235, 281, 317, 318],
    5: [235, 281, 317, 318],
    51: [235, 281, 317, 318],
    65: [281, 318],
    76: [317, 318],
    78: [317, 318],
    89: [106, 107, 282, 315],
    106: [315],
    107: [106, 107, 282, 315],
    129: [317, 318],
    166: [318],
    179: [317, 318],
    260: [317, 318],
    1035: [317, 318],
    1385: [235, 281, 317, 318],
    1442: [107, 282, 315],
    1581: [317, 318],
    1607: [315],
    1778: [282, 315],
    1848: [281, 317, 318],
    1865: [318],
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

# Candidate training leagues (will be filtered dynamically based on data availability)
CANDIDATE_TRAINING_LEAGUE_IDS = [51, 1385, 107, 4, 5, 1442]


# -----------------------------
# Temporal validation helpers
# -----------------------------

def prepare_temporal_split(df: pd.DataFrame, min_train: int = 10, min_test: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Split data temporally within each competition using canonical_season.
    
    Returns:
        train_df, test_df, report_dict
    """
    if "canonical_season" not in df.columns or "competition_id" not in df.columns:
        st.error("Missing canonical_season or competition_id columns")
        return df, pd.DataFrame(), {}
    
    train_frames = []
    test_frames = []
    excluded_comps = []
    comp_reports = []
    
    for comp_id in df["competition_id"].unique():
        comp_df = df[df["competition_id"] == comp_id].copy()
        
        # Get max season for this competition
        max_season = comp_df["canonical_season"].max()
        
        # Split
        test_comp = comp_df[comp_df["canonical_season"] == max_season]
        train_comp = comp_df[comp_df["canonical_season"] < max_season]
        
        # Check safeguards
        if len(train_comp) < min_train:
            excluded_comps.append({"comp_id": comp_id, "reason": f"train_rows={len(train_comp)} < {min_train}"})
            continue
        if len(test_comp) < min_test:
            excluded_comps.append({"comp_id": comp_id, "reason": f"test_rows={len(test_comp)} < {min_test}"})
            continue
        if train_comp.empty:
            excluded_comps.append({"comp_id": comp_id, "reason": "no earlier seasons"})
            continue
        
        train_frames.append(train_comp)
        test_frames.append(test_comp)
        comp_reports.append({
            "comp_id": comp_id,
            "league_name": LEAGUE_NAMES.get(comp_id, f"Comp {comp_id}"),
            "train_rows": len(train_comp),
            "test_rows": len(test_comp),
            "train_seasons": train_comp["canonical_season"].nunique(),
            "test_season": max_season
        })
    
    train_df = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    test_df = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()
    
    report = {
        "comp_reports": comp_reports,
        "excluded_comps": excluded_comps,
        "total_train": len(train_df),
        "total_test": len(test_df)
    }
    
    return train_df, test_df, report


# -----------------------------
# 1) API Fetching
# -----------------------------

@st.cache_resource(ttl=3600)
def fetch_player_season_stats(auth: Tuple[str, str]) -> Optional[pd.DataFrame]:
    """Fetch player-season stats for all configured league/season combos."""
    # Quick auth test
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
    success_log = []
    failure_log = []

    for comp_id, season_ids in COMPETITION_SEASONS.items():
        league_name = LEAGUE_NAMES.get(comp_id, f"Competition {comp_id}")
        for season_id in season_ids:
            done += 1
            progress.progress(done / total_requests)
            status.text(f"Loading player-stats: {league_name} | season_id={season_id} ({done}/{total_requests})")

            url = f"https://data.statsbombservices.com/api/v1/competitions/{comp_id}/seasons/{season_id}/player-stats"
            try:
                resp = requests.get(url, auth=auth, timeout=60)
                http_status = resp.status_code
                
                if http_status != 200:
                    failure_log.append({
                        "endpoint": "player-stats",
                        "competition_id": comp_id,
                        "league_name": league_name,
                        "season_id": season_id,
                        "http_status": http_status,
                        "error": f"HTTP {http_status}",
                        "response_snippet": resp.text[:200] if resp.text else "(no response body)"
                    })
                    continue
                
                resp.raise_for_status()
                data = resp.json()
                
                if not data:
                    failure_log.append({
                        "endpoint": "player-stats",
                        "competition_id": comp_id,
                        "league_name": league_name,
                        "season_id": season_id,
                        "http_status": http_status,
                        "error": "Empty response",
                        "response_snippet": "[]"
                    })
                    continue
                    
                df = pd.json_normalize(data)
                if df.empty:
                    failure_log.append({
                        "endpoint": "player-stats",
                        "competition_id": comp_id,
                        "league_name": league_name,
                        "season_id": season_id,
                        "http_status": http_status,
                        "error": "Empty dataframe after normalize",
                        "response_snippet": f"{len(data)} items but empty df"
                    })
                    continue
                    
                df["league_name"] = league_name
                df["competition_id"] = comp_id
                df["season_id"] = season_id
                frames.append(df)
                success_log.append({
                    "endpoint": "player-stats",
                    "competition_id": comp_id,
                    "league_name": league_name,
                    "season_id": season_id,
                    "rows": len(df)
                })
            except Exception as e:
                failure_log.append({
                    "endpoint": "player-stats",
                    "competition_id": comp_id,
                    "league_name": league_name,
                    "season_id": season_id,
                    "http_status": getattr(resp, 'status_code', 'N/A') if 'resp' in locals() else 'N/A',
                    "error": f"{type(e).__name__}: {str(e)}",
                    "response_snippet": ""
                })
                continue

    progress.empty()
    status.empty()
    
    # Store logs in session state
    st.session_state["player_fetch_success_log"] = success_log
    st.session_state["player_fetch_failure_log"] = failure_log

    if not frames:
        st.error(f"No player-stats returned. Attempted {total_requests} requests, succeeded 0, failed {len(failure_log)}.")
        return None

    success_count = len(success_log)
    failure_count = len(failure_log)
    st.success(f"Player stats: attempted {total_requests}, succeeded {success_count}, failed {failure_count}")

    return pd.concat(frames, ignore_index=True)


@st.cache_resource(ttl=3600)
def fetch_team_season_stats(auth: Tuple[str, str]) -> Optional[pd.DataFrame]:
    """Fetch team-season stats for all configured league/season combos."""
    total_requests = sum(len(v) for v in COMPETITION_SEASONS.values())
    progress = st.progress(0)
    status = st.empty()

    frames: List[pd.DataFrame] = []
    done = 0
    success_log = []
    failure_log = []

    for comp_id, season_ids in COMPETITION_SEASONS.items():
        league_name = LEAGUE_NAMES.get(comp_id, f"Competition {comp_id}")
        for season_id in season_ids:
            done += 1
            progress.progress(done / total_requests)
            status.text(f"Loading team-stats: {league_name} | season_id={season_id} ({done}/{total_requests})")

            url = f"https://data.statsbombservices.com/api/v2/competitions/{comp_id}/seasons/{season_id}/team-stats"
            try:
                resp = requests.get(url, auth=auth, timeout=60)
                http_status = resp.status_code
                
                if http_status != 200:
                    failure_log.append({
                        "endpoint": "team-stats",
                        "competition_id": comp_id,
                        "league_name": league_name,
                        "season_id": season_id,
                        "http_status": http_status,
                        "error": f"HTTP {http_status}",
                        "response_snippet": resp.text[:200] if resp.text else "(no response body)"
                    })
                    continue
                
                resp.raise_for_status()
                data = resp.json()
                
                if not data:
                    failure_log.append({
                        "endpoint": "team-stats",
                        "competition_id": comp_id,
                        "league_name": league_name,
                        "season_id": season_id,
                        "http_status": http_status,
                        "error": "Empty response",
                        "response_snippet": "[]"
                    })
                    continue
                    
                df = pd.json_normalize(data)
                if df.empty:
                    failure_log.append({
                        "endpoint": "team-stats",
                        "competition_id": comp_id,
                        "league_name": league_name,
                        "season_id": season_id,
                        "http_status": http_status,
                        "error": "Empty dataframe after normalize",
                        "response_snippet": f"{len(data)} items but empty df"
                    })
                    continue
                    
                df["league_name"] = league_name
                df["competition_id"] = comp_id
                df["season_id"] = season_id
                frames.append(df)
                success_log.append({
                    "endpoint": "team-stats",
                    "competition_id": comp_id,
                    "league_name": league_name,
                    "season_id": season_id,
                    "rows": len(df)
                })
            except Exception as e:
                failure_log.append({
                    "endpoint": "team-stats",
                    "competition_id": comp_id,
                    "league_name": league_name,
                    "season_id": season_id,
                    "http_status": getattr(resp, 'status_code', 'N/A') if 'resp' in locals() else 'N/A',
                    "error": f"{type(e).__name__}: {str(e)}",
                    "response_snippet": ""
                })
                continue

    progress.empty()
    status.empty()
    
    # Store logs in session state
    st.session_state["team_fetch_success_log"] = success_log
    st.session_state["team_fetch_failure_log"] = failure_log

    if not frames:
        st.error(f"No team-stats returned. Attempted {total_requests} requests, succeeded 0, failed {len(failure_log)}.")
        return None

    success_count = len(success_log)
    failure_count = len(failure_log)
    st.success(f"Team stats: attempted {total_requests}, succeeded {success_count}, failed {failure_count}")

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
    
    # Build a mapping of old to new column names, avoiding duplicates
    col_mapping = {}
    new_col_names = set()
    
    for c in df.columns:
        if c.startswith("team_season_"):
            new_name = c.replace("team_season_", "team_", 1)
            # If this would create a duplicate, keep the original name
            if new_name in new_col_names or new_name in df.columns:
                # Drop this column as it's redundant
                continue
            col_mapping[c] = new_name
            new_col_names.add(new_name)
        else:
            col_mapping[c] = c
            new_col_names.add(c)
    
    # Select only columns in the mapping and rename
    df = df[[c for c in df.columns if c in col_mapping]]
    df = df.rename(columns=col_mapping)

    for col in ["team_name", "league_name", "season_name"]:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    return df


def join_player_team(player_df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    """Attach team controls to each player-season row (same team + comp + season)."""
    # Ensure no duplicate columns in team_df before merge
    if team_df.columns.duplicated().any():
        team_df = team_df.loc[:, ~team_df.columns.duplicated()]
    
    # Prefer IDs if present
    join_cols = []
    if {"team_id", "competition_id", "season_id"}.issubset(team_df.columns) and {"team_id", "competition_id", "season_id"}.issubset(player_df.columns):
        join_cols = ["team_id", "competition_id", "season_id"]
    else:
        join_cols = ["team_name", "competition_id", "season_id"]

    # Keep only one row per team-season
    team_cols = [c for c in team_df.columns if c.startswith("team_")] + ["team_id", "team_name", "competition_id", "season_id"]
    team_cols = [c for c in team_cols if c in team_df.columns]
    # Remove duplicates from team_cols list
    team_cols = list(dict.fromkeys(team_cols))
    team_dedup = team_df[team_cols].drop_duplicates(subset=join_cols)

    merged = player_df.merge(team_dedup, on=join_cols, how="left", suffixes=("", "_team"))
    return merged


# -----------------------------
# 3) Filtering helpers (age/league/minutes/ST)
# -----------------------------

def is_striker_label(x) -> bool:
    """Enhanced striker detection to handle various position labels across leagues."""
    if pd.isna(x):
        return False
    
    # Handle numeric position codes - skip position filtering for numeric values
    # (different leagues may use different numeric codes)
    if isinstance(x, (int, float)):
        return True  # Don't filter out numeric positions
    
    s = str(x).strip()
    
    # Exact matches (case-insensitive)
    exact_matches = {
        "CF", "Centre Forward", "Center Forward", "ST", "Striker", 
        "Forward", "9", "Left Centre Forward", "Right Centre Forward",
        "Secondary Striker", "Left Center Forward", "Right Center Forward"
    }
    if s in exact_matches or s.lower() in {m.lower() for m in exact_matches}:
        return True
    
    # Substring matches (case-insensitive)
    s_lower = s.lower()
    if "forward" in s_lower or "striker" in s_lower:
        return True
    
    return False


def apply_common_filters(
    df: pd.DataFrame,
    min_minutes: int,
    age_range: Tuple[int, int],
    league_filter: str,
) -> pd.DataFrame:
    out = df.copy()

    # League filter - only apply if NOT "All Leagues"
    if "competition_id" in out.columns:
        if league_filter == "Domestic Leagues":
            out = out[out["competition_id"].isin(DOMESTIC_LEAGUE_IDS)]
        elif league_filter == "Scottish Leagues":
            out = out[out["competition_id"].isin(SCOTTISH_LEAGUE_IDS)]
        # else: All Leagues - no filter applied

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
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], Dict]:
    """
    Builds X, y plus metadata: player_features, team_features, leakage_diagnostics
    - Player features: all numeric player columns excluding banned/leakage
    - Team controls: small list from get_team_control_columns()
    - Leakage diagnostics: correlation analysis and dropped features
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
    
    # ============================================
    # LEAKAGE DIAGNOSTICS
    # ============================================
    leakage_diagnostics = {
        "correlations": {},
        "exact_matches": [],
        "dropped_features": [],
        "leaky_features": []
    }
    
    # Compute correlation of each feature with target
    correlations = {}
    for col in X.columns:
        try:
            # Check if column is constant
            if X[col].std() == 0 or y.std() == 0:
                correlations[col] = 0.0
            else:
                corr_val = np.corrcoef(X[col].values, y.values)[0, 1]
                correlations[col] = corr_val if not np.isnan(corr_val) else 0.0
        except Exception:
            correlations[col] = 0.0
    
    # Identify exact matches (within numerical precision)
    exact_matches = []
    for col in X.columns:
        try:
            if np.allclose(X[col].values, y.values, atol=1e-12, rtol=0):
                exact_matches.append(col)
        except Exception:
            pass
    
    # Identify highly correlated features (suspicious)
    corr_series = pd.Series(correlations)
    leaky_features = corr_series[corr_series.abs() > 0.95].index.tolist()
    
    # Store diagnostics
    leakage_diagnostics["correlations"] = correlations
    leakage_diagnostics["exact_matches"] = exact_matches
    leakage_diagnostics["leaky_features"] = leaky_features
    
    # Determine which columns to drop
    drop_cols = set(exact_matches) | set(leaky_features)
    
    if drop_cols:
        leakage_diagnostics["dropped_features"] = list(drop_cols)
        # Drop leaky columns from X
        X = X.drop(columns=list(drop_cols))
        
        # Update feature lists
        player_features = [f for f in player_features if f not in drop_cols]
        team_features = [f for f in team_features if f not in drop_cols]

    return X, y, player_features, team_features, leakage_diagnostics


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
        
        # Recruitment mode - exclude Scottish Premiership from scouting
        recruitment_mode = st.checkbox("🎯 Recruitment mode: exclude Scottish Premiership players", value=False,
                                       help="Excludes competition_id 51 from scouting shortlist (training still uses SPFL)")
        
        # Season filter for scouting pool - always render, compute from scouting pool after other filters
        st.caption("Season filter (scouting only):")
        if "season_filter_options" in st.session_state and st.session_state.season_filter_options:
            available_seasons = st.session_state.season_filter_options
            default_seasons = st.session_state.get("season_filter_selection", available_seasons)
            season_filter = st.multiselect(
                "Scouting season(s)", 
                available_seasons, 
                default=default_seasons,
                help="Filter scouting pool by season (training unaffected)"
            )
            st.session_state.season_filter_selection = season_filter
        else:
            st.info("Load data to enable season filter")
            season_filter = None

        st.divider()
        st.header("3) Model Settings")
        include_team_controls = st.checkbox("Include team-season controls (recommended)", value=True)
        add_age_poly = st.checkbox("Use age + age² (recommended)", value=True)

        n_boot = st.slider("Stability selection bootstraps", 30, 300, 120, 10)
        sample_frac = st.slider("Bootstrap sample fraction", 0.50, 1.00, 0.85, 0.05)

        top_k = st.slider("Show Top-K predictors", 3, 20, 5, 1)
        
        st.divider()
        st.header("4) Output Settings")
        sort_by = st.selectbox("Sort shortlist by", ["pred_np_xg_90", "pred_goals_90_assuming_avg_finish", "residual"], index=0,
                              help="pred_goals_90 assumes league-average finishing (1.0x conversion)")

        st.divider()
        load_btn = st.button("Load / Refresh Data", type="primary")

    # Check if model settings changed (requires refit)
    model_settings_key = f"{include_team_controls}_{add_age_poly}_{n_boot}_{sample_frac}"
    model_settings_changed = st.session_state.get("model_settings_key") != model_settings_key
    
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
        # Clear cached model results when data refreshed
        st.session_state.pop("merged", None)
        st.session_state.pop("full_model", None)
        st.session_state.pop("trait_rows", None)
        st.session_state.pop("season_filter_options", None)

        st.success("Data loaded.")
        st.rerun()

    player_raw = st.session_state.get("player_raw")
    team_raw = st.session_state.get("team_raw")
    if player_raw is None or team_raw is None:
        st.warning("No data loaded yet.")
        return

    # Show configuration
    with st.expander("⚙️ Configuration"):
        st.write(f"**Leagues configured:** {len(LEAGUE_NAMES)}")
        st.write(f"**Total season requests:** {sum(len(v) for v in COMPETITION_SEASONS.values())}")
        config_df = pd.DataFrame([
            {"comp_id": k, "league": LEAGUE_NAMES.get(k, f"Comp {k}"), "seasons": len(v), "season_ids": str(v)}
            for k, v in COMPETITION_SEASONS.items()
        ])
        st.dataframe(config_df, width="stretch", hide_index=True)
    
    # Show fetch logs
    if "player_fetch_success_log" in st.session_state:
        with st.expander("✅ Fetch Successes"):
            p_success = st.session_state.get("player_fetch_success_log", [])
            t_success = st.session_state.get("team_fetch_success_log", [])
            
            if p_success:
                st.markdown("**Player stats successes:**")
                p_df = pd.DataFrame(p_success)
                success_by_comp = p_df.groupby(['competition_id', 'league_name']).agg(
                    requests=('season_id', 'count'),
                    total_rows=('rows', 'sum')
                ).reset_index()
                st.dataframe(success_by_comp, width="stretch", hide_index=True)
            
            if t_success:
                st.markdown("**Team stats successes:**")
                t_df = pd.DataFrame(t_success)
                success_by_comp = t_df.groupby(['competition_id', 'league_name']).agg(
                    requests=('season_id', 'count'),
                    total_rows=('rows', 'sum')
                ).reset_index()
                st.dataframe(success_by_comp, width="stretch", hide_index=True)
    
    if "player_fetch_failure_log" in st.session_state:
        p_failures = st.session_state.get("player_fetch_failure_log", [])
        t_failures = st.session_state.get("team_fetch_failure_log", [])
        
        if p_failures or t_failures:
            with st.expander(f"❌ Fetch Failures ({len(p_failures) + len(t_failures)} total)"):
                all_failures = p_failures + t_failures
                if all_failures:
                    fail_df = pd.DataFrame(all_failures)
                    st.dataframe(fail_df, width="stretch", hide_index=True)

    # Process + join (cache in session state)
    if "merged" not in st.session_state or st.session_state.merged is None:
        player_df = process_player_data(player_raw)
        team_df = process_team_data(team_raw)
        merged = join_player_team(player_df, team_df)
        st.session_state.merged = merged
    else:
        merged = st.session_state.merged
    
    # Show unique position labels for debugging
    with st.expander("🔍 Position Labels in Data"):
        if "primary_position" in merged.columns:
            pos_counts = merged["primary_position"].value_counts().head(50)
            st.write(f"Top 50 position labels (out of {merged['primary_position'].nunique()} unique):")
            st.dataframe(pos_counts, width="stretch")

    # -----------------------------
    # TRAINING SET: Multi-league STs >=900 mins (dynamic inclusion)
    # -----------------------------
    st.subheader("Training Set: Multi-League ST Pool")
    
    # Apply filters to get candidate training pool
    train_candidate = merged.copy()
    train_candidate = apply_common_filters(
        train_candidate,
        min_minutes=900,
        age_range=(18, 35),  # Focused age range for training
        league_filter="All Leagues",
    )
    
    # Filter to candidate training leagues only
    train_candidate = train_candidate[train_candidate["competition_id"].isin(CANDIDATE_TRAINING_LEAGUE_IDS)]
    
    # Check which leagues have sufficient data (>=25 ST player-seasons)
    league_coverage = train_candidate.groupby("competition_id").agg(
        player_seasons=("player_id", "count"),
        unique_players=("player_id", "nunique"),
        seasons=("season_id", "nunique")
    ).reset_index()
    league_coverage["league_name"] = league_coverage["competition_id"].map(
        lambda x: LEAGUE_NAMES.get(x, f"Comp {x}")
    )
    
    # Include only leagues with >=25 player-seasons
    sufficient_leagues = league_coverage[league_coverage["player_seasons"] >= 25]["competition_id"].tolist()
    
    if not sufficient_leagues:
        st.error("No leagues have sufficient data (>=25 ST player-seasons) for training. Lower thresholds or fetch more data.")
        return
    
    train_base = train_candidate[train_candidate["competition_id"].isin(sufficient_leagues)].copy()
    
    # Training coverage table
    training_coverage = league_coverage[league_coverage["competition_id"].isin(sufficient_leagues)].copy()
    training_coverage = training_coverage.sort_values("player_seasons", ascending=False)
    
    with st.expander("📊 Training Coverage", expanded=True):
        st.dataframe(training_coverage[["competition_id", "league_name", "player_seasons", "unique_players", "seasons"]], 
                    width="stretch", hide_index=True)
        
        # Warnings for leagues with <50 player-seasons
        low_coverage = training_coverage[training_coverage["player_seasons"] < 50]
        if not low_coverage.empty:
            st.warning(f"⚠️ {len(low_coverage)} league(s) have <50 player-seasons: {low_coverage['league_name'].tolist()}")
    
    # Temporal validation split
    train_split, test_split, temporal_report = prepare_temporal_split(train_base, min_train=10, min_test=5)
    
    if train_split.empty or test_split.empty:
        st.error("Temporal split failed - insufficient data. Using full training set without validation.")
        train_split = train_base.copy()
        test_split = None
        temporal_report = None
    else:
        with st.expander("⏰ Temporal Validation Split"):
            st.write(f"**Total training rows:** {temporal_report['total_train']:,}")
            st.write(f"**Total test rows:** {temporal_report['total_test']:,}")
            
            if temporal_report['comp_reports']:
                comp_df = pd.DataFrame(temporal_report['comp_reports'])
                st.markdown("### Per-competition split:")
                st.dataframe(comp_df, width="stretch", hide_index=True)
            
            if temporal_report['excluded_comps']:
                exc_df = pd.DataFrame(temporal_report['excluded_comps'])
                exc_df["league_name"] = exc_df["comp_id"].map(lambda x: LEAGUE_NAMES.get(x, f"Comp {x}"))
                st.warning(f"⚠️ Excluded {len(temporal_report['excluded_comps'])} competitions:")
                st.dataframe(exc_df[["comp_id", "league_name", "reason"]], width="stretch", hide_index=True)

    if PRIMARY_TARGET not in train_split.columns:
        st.error(f"Target '{PRIMARY_TARGET}' not found. Check your player-stats endpoint/columns.")
        st.write("Available columns:", sorted(train_split.columns))
        return
    
    st.write(f"**Training on:** {len(train_split):,} player-seasons from {train_split['competition_id'].nunique()} leagues")
    if test_split is not None:
        st.write(f"**Temporal test set:** {len(test_split):,} player-seasons")

    # -----------------------------
    # Fit model + stability
    # -----------------------------
    st.subheader("Model: Elastic Net + Stability Selection")
    
    # Only refit if model settings changed or no cached model
    if model_settings_changed or "full_model" not in st.session_state:
        try:
            X, y, player_feats, team_feats, leakage_diag = build_feature_matrix(
                train_split,
                target=PRIMARY_TARGET,
                include_team_controls=include_team_controls,
                add_age_poly=add_age_poly,
            )
        except Exception as e:
            st.error(f"Feature matrix build failed: {e}")
            return
        
        # Display leakage diagnostics
        with st.expander("🔍 Leakage Diagnostics"):
            st.markdown("### Feature Correlations with Target")
            if leakage_diag["correlations"]:
                corr_df = pd.DataFrame([
                    {"feature": k, "correlation": v, "abs_corr": abs(v)}
                    for k, v in leakage_diag["correlations"].items()
                ]).sort_values("abs_corr", ascending=False).head(20)
                st.dataframe(corr_df, width="stretch", hide_index=True)
            
            st.markdown("### Exact Match Features (perfect correlation)")
            if leakage_diag["exact_matches"]:
                st.error(f"Found {len(leakage_diag['exact_matches'])} exact matches: {leakage_diag['exact_matches']}")
            else:
                st.success("No exact matches found")
            
            st.markdown("### Highly Correlated Features (|corr| > 0.95)")
            if leakage_diag["leaky_features"]:
                st.warning(f"Found {len(leakage_diag['leaky_features'])} leaky features: {leakage_diag['leaky_features']}")
            else:
                st.success("No highly correlated features found")
            
            st.markdown("### Dropped Features")
            if leakage_diag["dropped_features"]:
                st.error(f"Dropped {len(leakage_diag['dropped_features'])} leaky columns: {leakage_diag['dropped_features']}")
            else:
                st.success("No features dropped due to leakage")

        if len(X) < 25:
            st.error("Training set too small after filters. Consider adding more SPFL seasons or lowering minutes threshold.")
            return

        all_features = list(X.columns)

        with st.spinner("Fitting ElasticNetCV on full training set..."):
            full_model = fit_elastic_net_cv(X, y, random_state=42)

        # Stability selection with robust error handling
        stab = None
        stability_start = time.time()
        try:
            with st.spinner("Running stability selection (bootstraps)..."):
                stab = stability_selection(
                    X=X,
                    y=y,
                    feature_names=all_features,
                    n_boot=n_boot,
                    sample_frac=float(sample_frac),
                    random_state=42,
                )
            stability_time = time.time() - stability_start
        except Exception as e:
            stability_time = time.time() - stability_start
            st.error(f"Stability selection failed after {stability_time:.1f}s: {e}")
            # Create empty stability dataframe so app can continue
            stab = pd.DataFrame({
                "feature": all_features,
                "selection_freq": 0.0,
                "mean_abs_coef": 0.0
            })

        # Debug info
        with st.expander("🔍 Model debug info"):
            st.write(f"Training samples: {len(X)}, Features: {len(all_features)}")
            st.write(f"Bootstrap settings: n_boot={n_boot}, sample_frac={sample_frac}")
            st.write(f"Stability selection time: {stability_time:.1f}s")
            if stab is not None:
                st.write(f"Features selected (freq>0): {(stab['selection_freq'] > 0).sum()}")

        # Separate "player traits" from "team controls" for interpretation
        if stab is not None and not stab.empty:
            trait_rows = stab[stab["feature"].isin(player_feats)].copy()
            control_rows = stab[stab["feature"].isin(team_feats)].copy()
        else:
            trait_rows = pd.DataFrame(columns=["feature", "selection_freq", "mean_abs_coef"])
            control_rows = pd.DataFrame(columns=["feature", "selection_freq", "mean_abs_coef"])
        
        # Training residual diagnostics (sanity check for leakage)
        yhat_train = full_model.predict(X.values)
        residuals_train = y.values - yhat_train
        train_residual_stats = {
            "min": float(residuals_train.min()),
            "max": float(residuals_train.max()),
            "mean": float(residuals_train.mean()),
            "std": float(residuals_train.std(ddof=0)),
            "unique": int(pd.Series(residuals_train).nunique()),
            "all_zero": bool(np.allclose(residuals_train, 0, atol=1e-10))
        }
        
        with st.expander("📊 Training Residual Diagnostics"):
            st.write("Residual statistics on training set:")
            st.json(train_residual_stats)
            if train_residual_stats["all_zero"]:
                st.error("⚠️ All training residuals are ~0! This indicates perfect fit (likely leakage).")
            elif train_residual_stats["std"] < 0.01:
                st.warning("⚠️ Very low residual variance - possible leakage.")
            else:
                st.success("✓ Training residuals show normal variation.")
        
        # Cache everything
        st.session_state.full_model = full_model
        st.session_state.X_train = X
        st.session_state.y_train = y
        st.session_state.all_features = all_features
        st.session_state.player_feats = player_feats
        st.session_state.team_feats = team_feats
        st.session_state.trait_rows = trait_rows
        st.session_state.control_rows = control_rows
        st.session_state.model_settings_key = model_settings_key
        st.session_state.train_competitions = train_split["competition_id"].unique().tolist()
        
        # PHASE 2: Temporal validation
        if test_split is not None and not test_split.empty:
            try:
                X_test, y_test, _, _, _ = build_feature_matrix(
                    test_split,
                    target=PRIMARY_TARGET,
                    include_team_controls=include_team_controls,
                    add_age_poly=add_age_poly,
                )
                # Align features
                for col in all_features:
                    if col not in X_test.columns:
                        X_test[col] = 0
                X_test = X_test[all_features]
                
                y_pred_test = full_model.predict(X_test.values)
                test_corr = np.corrcoef(y_test.values, y_pred_test)[0, 1]
                test_r2 = test_corr ** 2
                test_mae = np.mean(np.abs(y_test.values - y_pred_test))
                
                with st.expander("⏰ Temporal Test Performance", expanded=True):
                    col1, col2 = st.columns(2)
                    col1.metric("Test R²", f"{test_r2:.3f}")
                    col2.metric("Test MAE", f"{test_mae:.3f}")
                    
                    # Interpretation
                    if test_r2 > 0.3:
                        st.success("✅ Strong predictive power on held-out seasons")
                    elif test_r2 > 0.15:
                        st.info("ℹ️ Moderate predictive power - model captures some signal")
                    elif test_r2 > 0.05:
                        st.warning("⚠️ Weak predictive power - consider adding features or data")
                    else:
                        st.error("❌ Very weak predictive power - model may not generalize well")
            except Exception as e:
                st.warning(f"Temporal validation failed: {e}")
        
        # PHASE 3: Cross-league transferability test
        spfl_data = train_base[train_base["competition_id"] == 51].copy()
        non_spfl_data = train_base[train_base["competition_id"] != 51].copy()
        
        if not spfl_data.empty and not non_spfl_data.empty and len(non_spfl_data) >= 25:
            try:
                # Split SPFL temporally
                spfl_train, spfl_test, _ = prepare_temporal_split(spfl_data, min_train=5, min_test=3)
                
                if not spfl_test.empty and len(non_spfl_data) >= 25:
                    # Train on non-SPFL temporal train
                    non_spfl_train, _, _ = prepare_temporal_split(non_spfl_data, min_train=10, min_test=0)
                    if non_spfl_train.empty:
                        non_spfl_train = non_spfl_data
                    
                    X_nonspfl, y_nonspfl, _, _, _ = build_feature_matrix(
                        non_spfl_train,
                        target=PRIMARY_TARGET,
                        include_team_controls=include_team_controls,
                        add_age_poly=add_age_poly,
                    )
                    
                    # Fit model on non-SPFL
                    model_nonspfl = fit_elastic_net_cv(X_nonspfl, y_nonspfl, random_state=42)
                    
                    # Test on SPFL
                    X_spfl_test, y_spfl_test, _, _, _ = build_feature_matrix(
                        spfl_test,
                        target=PRIMARY_TARGET,
                        include_team_controls=include_team_controls,
                        add_age_poly=add_age_poly,
                    )
                    # Align features
                    for col in X_nonspfl.columns:
                        if col not in X_spfl_test.columns:
                            X_spfl_test[col] = 0
                    X_spfl_test = X_spfl_test[X_nonspfl.columns]
                    
                    y_pred_spfl = model_nonspfl.predict(X_spfl_test.values)
                    cross_corr = np.corrcoef(y_spfl_test.values, y_pred_spfl)[0, 1]
                    cross_r2 = cross_corr ** 2
                    cross_mae = np.mean(np.abs(y_spfl_test.values - y_pred_spfl))
                    
                    with st.expander("🌍 Cross-League Transferability"):
                        st.markdown("**Training:** Non-SPFL leagues only")
                        st.markdown("**Testing:** SPFL (Scottish Premiership) latest season")
                        st.write(f"Non-SPFL training samples: {len(X_nonspfl)}")
                        st.write(f"SPFL test samples: {len(X_spfl_test)}")
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Cross-league R²", f"{cross_r2:.3f}")
                        col2.metric("Cross-league MAE", f"{cross_mae:.3f}")
                        
                        if cross_r2 > 0.2:
                            st.success("✅ Model transfers reasonably well across leagues")
                        elif cross_r2 > 0.1:
                            st.warning("⚠️ Moderate transfer - league differences may be significant")
                        else:
                            st.error("❌ Poor transfer - model may be league-specific")
            except Exception as e:
                st.warning(f"Cross-league test failed: {e}")
    else:
        # Use cached model
        full_model = st.session_state.full_model
        X = st.session_state.X_train
        y = st.session_state.y_train
        all_features = st.session_state.all_features
        player_feats = st.session_state.player_feats
        team_feats = st.session_state.team_feats
        trait_rows = st.session_state.trait_rows
        control_rows = st.session_state.control_rows
    
    enet = full_model.named_steps["enet"]
    st.caption(
        f"Chosen alpha={enet.alpha_:.6f} | l1_ratio={enet.l1_ratio_} | "
        f"Non-zero coefs={int(np.sum(enet.coef_ != 0))}/{len(all_features)}"
    )


    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        st.markdown(f"### Top player-level predictors of **{PRIMARY_TARGET}** (SPFL STs)")
        st.dataframe(
            trait_rows.head(top_k).assign(
                selection_freq=lambda d: (100 * d["selection_freq"]).round(1),
                mean_abs_coef=lambda d: d["mean_abs_coef"].round(3),
            ).rename(columns={"selection_freq": "selected_%", "mean_abs_coef": "mean|coef|"}),
            width="stretch",
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
            width="stretch",
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
                    width="stretch",
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
    # Data coverage summary - MERGED BEFORE FILTERS
    # -----------------------------
    with st.expander("📊 Coverage: Merged Data (BEFORE Filters)"):
        st.markdown("### Data immediately after join_player_team()")
        st.write(f"**Total rows:** {len(merged):,}")
        st.write(f"**Unique players:** {merged['player_id'].nunique() if 'player_id' in merged.columns else merged['player_name'].nunique()}")
        st.write(f"**Unique teams:** {merged['team_name'].nunique() if 'team_name' in merged.columns else 'N/A'}")
        st.write(f"**Unique competitions:** {merged['competition_id'].nunique() if 'competition_id' in merged.columns else 'N/A'}")
        st.write(f"**Unique seasons:** {merged['season_id'].nunique() if 'season_id' in merged.columns else 'N/A'}")
        
        if "competition_id" in merged.columns:
            st.markdown("### Competition distribution (sorted by count)")
            comp_counts = merged['competition_id'].value_counts()
            comp_counts_df = pd.DataFrame({
                "competition_id": comp_counts.index,
                "league_name": [LEAGUE_NAMES.get(c, f"Comp {c}") for c in comp_counts.index],
                "player_seasons": comp_counts.values,
                "pct": (100 * comp_counts.values / len(merged)).round(1)
            })
            st.dataframe(comp_counts_df, width="stretch", hide_index=True)
            
            # Show which configured leagues are MISSING
            fetched_comps = set(merged['competition_id'].unique())
            configured_comps = set(COMPETITION_SEASONS.keys())
            missing_comps = configured_comps - fetched_comps
            if missing_comps:
                st.warning(f"⚠️ {len(missing_comps)} configured competitions have ZERO rows:")
                missing_df = pd.DataFrame([
                    {"comp_id": c, "league": LEAGUE_NAMES.get(c, f"Comp {c}")}
                    for c in sorted(missing_comps)
                ])
                st.dataframe(missing_df, width="stretch", hide_index=True)

    # -----------------------------
    # SCOUTING POOL: apply user filters (age + league + recruitment mode + season)
    # -----------------------------
    st.subheader("Scouting Shortlist (filters applied)")

    # Apply filters step by step
    scouting = apply_common_filters(
        merged,
        min_minutes=min_minutes,
        age_range=age_range,
        league_filter=league_filter,
    )
    
    # Apply recruitment mode: exclude Scottish Premiership from scouting
    if recruitment_mode and "competition_id" in scouting.columns:
        before_count = len(scouting)
        scouting = scouting[scouting["competition_id"] != 51]
        excluded = before_count - len(scouting)
        if excluded > 0:
            st.info(f"🎯 Recruitment mode: excluded {excluded} Scottish Premiership player-seasons")
    
    # Compute season filter options from current scouting pool (after league/recruitment filters)
    if "season_name" in scouting.columns and not scouting.empty:
        available_seasons_now = sorted(scouting["season_name"].dropna().unique().tolist())
        if available_seasons_now:
            st.session_state.season_filter_options = available_seasons_now
            # Initialize selection if not set
            if "season_filter_selection" not in st.session_state:
                st.session_state.season_filter_selection = available_seasons_now
    
    # Show scouting pool coverage before season filter
    with st.expander("📊 Scouting pool coverage (after league/recruitment/age/mins filters)"):
        if "competition_id" in scouting.columns:
            st.write(f"Total rows: {len(scouting):,}")
            st.write(f"Unique competitions: {scouting['competition_id'].nunique()}")
            st.markdown("**Competitions:**")
            comp_counts = scouting['competition_id'].value_counts().head(30)
            comp_counts_df = pd.DataFrame({
                "competition_id": comp_counts.index,
                "league_name": [LEAGUE_NAMES.get(c, f"Comp {c}") for c in comp_counts.index],
                "count": comp_counts.values
            })
            st.dataframe(comp_counts_df, width="stretch", hide_index=True)
            
            st.markdown("**Seasons available:**")
            season_counts = scouting['season_name'].value_counts()
            st.dataframe(season_counts, width="stretch")
    
    # Apply season filter to scouting pool
    if season_filter and "season_name" in scouting.columns:
        scouting = scouting[scouting["season_name"].isin(season_filter)]

    if scouting.empty:
        st.warning("No players match your filters. Widen age range, minutes, league filter, or season selection.")
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
    scouting["pred_goals_90_assuming_avg_finish"] = preds * 1.0  # League-average finishing assumption
    
    # Calculate residuals - but set to NaN for players from leagues not in training
    train_comps = st.session_state.get("train_competitions", [])
    scouting["residual"] = scouting[PRIMARY_TARGET].astype(float) - scouting["pred_np_xg_90"]
    if train_comps:
        # Set residuals to NaN for players from leagues not used in training
        out_of_training_mask = ~scouting["competition_id"].isin(train_comps)
        scouting.loc[out_of_training_mask, "residual"] = np.nan
    
    # Residual diagnostics for scouting pool
    residuals_scout = scouting["residual"].dropna()
    if len(residuals_scout) > 0:
        scout_residual_stats = {
            "min": float(residuals_scout.min()),
            "max": float(residuals_scout.max()),
            "mean": float(residuals_scout.mean()),
            "std": float(residuals_scout.std(ddof=0)),
            "unique": int(residuals_scout.nunique()),
            "all_zero": bool(np.allclose(residuals_scout.values, 0, atol=1e-10))
        }
        
        with st.expander("📊 Scouting Residual Diagnostics"):
            st.write("Residual statistics on scouting pool:")
            st.json(scout_residual_stats)
            if scout_residual_stats["all_zero"]:
                st.error("⚠️ All scouting residuals are ~0! This indicates perfect predictions (likely leakage).")
            elif scout_residual_stats["std"] < 0.01:
                st.warning("⚠️ Very low residual variance - possible leakage or overfitting.")
            else:
                st.success("✓ Scouting residuals show normal variation.")
    
    # Cache scouting stats for instant player selection
    st.session_state.scouting_pool = scouting
    st.session_state.scouting_means = {col: scouting[col].astype(float).mean() 
                                       for col in scouting.select_dtypes(include=[np.number]).columns}
    st.session_state.scouting_stds = {col: scouting[col].astype(float).std(ddof=0) 
                                      for col in scouting.select_dtypes(include=[np.number]).columns}

    # Display
    show_cols = [c for c in [
        "player_name", "age", "primary_position", "team_name", "league_name", "season_name",
        "minutes", PRIMARY_TARGET, "pred_np_xg_90", "pred_goals_90_assuming_avg_finish", "residual"
    ] if c in scouting.columns]

    # Sort by user selection
    if sort_by == "residual":
        st.warning("⚠️ Residual sorting: players from leagues not used in training have residual=NaN and will be excluded from ranking.")
        scouting_sorted = scouting[scouting["residual"].notna()].sort_values(sort_by, ascending=False)
    else:
        sort_ascending = False
        scouting_sorted = scouting.sort_values(sort_by, ascending=sort_ascending)
    
    # rank by selected metric (1 = best)
    topn = st.slider("Shortlist size", 10, 100, 30, 5)
    shortlist = scouting_sorted.head(topn).copy()
    shortlist["rank"] = range(1, len(shortlist) + 1)

    # Update show_cols to include rank at the beginning
    show_cols_with_rank = ["rank"] + show_cols if "rank" in shortlist.columns else show_cols
    
    # Formatting
    for col in ["pred_np_xg_90", "pred_goals_90_assuming_avg_finish", PRIMARY_TARGET, "residual"]:
        if col in shortlist.columns:
            shortlist[col] = shortlist[col].astype(float).round(3)

    st.dataframe(shortlist[show_cols_with_rank], width="stretch", hide_index=True)

    st.caption(
        "Tip: Residual > 0 = producing more np_xg than the model expects given traits + team context. "
        "Residual < 0 = underproducing vs expected. "
        "pred_goals_90_assuming_avg_finish assumes league-average finishing (1.0x conversion rate). "
        "Residuals are NaN for players from leagues not used in training."
    )

    with st.expander("Download shortlist as CSV"):
        csv = shortlist[show_cols_with_rank].to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="spfl_st_shortlist.csv", mime="text/csv")

    st.divider()

    # -----------------------------
    # “Top 5 traits” scorecard for a selected player
    # -----------------------------
    st.subheader("Trait Scorecard (Top-K predictors)")
    trait_top = trait_rows.head(top_k)["feature"].tolist()
    
    # Cache trait_top for reuse
    st.session_state.trait_top = trait_top

    # pick a player in the shortlisted pool
    player_options = shortlist["player_name"].astype(str).tolist() if "player_name" in shortlist.columns else []
    if player_options:
        selected_player = st.selectbox("Select a player from the shortlist", player_options, index=0)
        row = shortlist[shortlist["player_name"].astype(str) == str(selected_player)].head(1)
        if not row.empty:
            r = row.iloc[0]
            st.markdown(f"**{r.get('player_name','')}** — {r.get('team_name','')} ({r.get('league_name','')}, {r.get('season_name','')})")
            st.write(f"Minutes: **{int(r.get('minutes',0))}** | Age: **{int(r.get('age',0))}**")
            st.write(f"Rank: **#{int(r.get('rank',0))}** | Actual {PRIMARY_TARGET}: **{r.get(PRIMARY_TARGET, np.nan):.3f}** | Predicted np_xG/90: **{r.get('pred_np_xg_90', np.nan):.3f}** | Predicted goals/90 (avg finish): **{r.get('pred_goals_90_assuming_avg_finish', np.nan):.3f}**")

            # Show z-scores within the *scouting pool* for interpretability (using cached stats)
            score_df = []
            means = st.session_state.get("scouting_means", {})
            stds = st.session_state.get("scouting_stds", {})
            
            for f in trait_top:
                if f in means and f in stds:
                    mu = means[f]
                    sd = stds[f]
                    val = float(r.get(f, np.nan))
                    z = (val - mu) / sd if (sd and not np.isnan(val)) else np.nan
                    score_df.append({"trait": f, "value": val, "z_in_pool": z})
            if score_df:
                out = pd.DataFrame(score_df)
                out["value"] = out["value"].round(3)
                out["z_in_pool"] = out["z_in_pool"].round(2)
                st.dataframe(out, width="stretch", hide_index=True)
                st.caption("z_in_pool: standard deviations above/below scouting pool mean")
            else:
                st.info("No trait columns available for scorecard (check column names).")
    else:
        st.info("No shortlist players available for scorecard.")


if __name__ == "__main__":
    main()
