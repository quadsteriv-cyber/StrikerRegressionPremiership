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
from scipy.stats import spearmanr


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

# Candidate training leagues (will be filtered dynamically based on data availability)
CANDIDATE_TRAINING_LEAGUE_IDS = [51, 1385, 107, 4, 5, 1442]

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

# ABLATION EXPERIMENT: Near-shot proxy features
# These are NOT leakage - they're proximate predictors we test for dependence
NEAR_SHOT_PROXY_FEATURES = {
    "touches_inside_box_90",
    "passes_into_box_90",
    "passes_inside_box_90",
    "sp_passes_into_box_90",
    "op_passes_into_box_90",
    "key_passes_90",
    "op_key_passes_90",
    "sp_key_passes_90",
    "op_passes_into_and_touches_inside_box_90",
}


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
    enable_league_fixed_effects: bool = False,
    behaviour_only: bool = True,
    allow_xg_identity: bool = False,
    ablation_mode: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], Dict]:
    """
    Builds X, y plus metadata: player_features, team_features, leakage_diagnostics
    - Player features: all numeric player columns excluding banned/leakage
    - Team controls: small list from get_team_control_columns()
    - Ablation mode (optional): "M1_upstream_only" or "M2_near_shot_only" for experiments

    - League fixed effects (optional): one-hot encode competition_id to learn league-level differences
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

    # ============================================
    # BEHAVIOUR-ONLY GUARDRAILS
    # ============================================
    
    # HARD EXCLUSIONS (ALWAYS applied, regardless of toggles)
    hard_exclusion_patterns = [
        # xG & derivatives
        "xg", "npxg", "np_xg", "xg_per", "xg_90",
        # outcomes / finishing
        "goal", "goals", "npg", "npgoals",
        "conversion", "shot_conversion",
        # assists
        "assist", "assists", "xa",
        # defensive outcomes
        "npga", "goals_against", "ga", "conceded",
        # 🔴 SHOT IDENTITY LEAKAGE (np_xg_90 ≈ shots × shot_quality)
        "shot", "shots",
    ]
    
    hard_bans = {
        target,
        # Target aliases
        "np_xg", "npxg", "np_xg_per_90", "npxg_90",
        # Goal outcomes / finishing
        "goals_90", "npg_90", "goals", "npg", "npgoals",
        "conversion_ratio", "shot_on_target_ratio", "shot_conversion",
        "over_under_performance_90", "penalty_goals_90",
        "penalty_conversion_ratio",
        # Assists
        "assists", "xa", "xA", "expected_assists",
        # Goals against / defensive outcomes
        "npga", "npga_90", "goals_against", "ga", "conceded",
        # Minutes (hard leakage)
        "minutes", "player_season_minutes",
        # 🔴 Shot-based features (identity leakage: np_xg ≈ shots × quality)
        "np_shots_90", "shots_90", "shots", "shot",
        "shot_touch_ratio", "shots_on_target", "shots_on_target_90",
        # Other outcome proxies
        "shots_faced_90", "save_ratio",
    }
    
    def contains_hard_exclusion(col_name: str) -> bool:
        """Check if column contains any hard exclusion pattern."""
        col_lower = col_name.lower()
        return any(pattern in col_lower for pattern in hard_exclusion_patterns)
    
    # BEHAVIOUR-ONLY WHITELIST (upstream actions only, NO outcomes)
    allowed_player_behaviours = [
        # Touches (upstream possession behaviours)
        "touch", "touches_inside_box", "box_touch",
        # NOTE: shot_touch_ratio REMOVED - it's outcome-derived
        # NOTE: shots_, np_shots REMOVED - shot counts are outcomes
        # Defensive actions
        "press", "pressure", "counterpress",
        # Ball progression
        "carry", "dribble",
        # Receiving
        "receive", "receptions", "received",
        # Passing
        "pass", "key_pass", "passes_inside_box",
        # Duels
        "aerial", "duel",
        # Other actions
        "foul_won", "turnover",
        # Movement/positioning
        "distance", "average_x_pass",
        # On-ball value (action-based, not outcome)
        "obv",
    ]
    
    allowed_team_patterns = [
        "team_possession", "team_directness",
        "team_passes_inside_box", "team_crosses_into_box",
        "team_deep_completions", "team_field_tilt",
        "team_",  # generic team context
        "league_",  # league fixed effects
    ]
    
    def is_allowed_behaviour(col_name: str) -> bool:
        """Check if column matches allowed behaviour patterns."""
        col_lower = col_name.lower()
        # Check player behaviours
        if any(pattern in col_lower for pattern in allowed_player_behaviours):
            return True
        # Check team patterns
        if any(col_lower.startswith(pattern.lower()) or pattern.lower() in col_lower 
               for pattern in allowed_team_patterns):
            return True
        return False
    
    # Get all numeric columns
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    
    # Initial filtering: remove non-feature cols and hard bans
    candidate_cols = [c for c in num_cols 
                     if (c not in non_feature_cols) 
                     and (c not in hard_bans)
                     and (not contains_hard_exclusion(c))]
    
    # Separate player and team columns
    player_candidates = [c for c in candidate_cols if not c.startswith("team_") and not c.startswith("league_")]
    team_candidates = [c for c in candidate_cols if c.startswith("team_") or c.startswith("league_")]
    
    # Apply behaviour-only filter if enabled
    dropped_by_behaviour_filter = []
    if behaviour_only:
        player_num = [c for c in player_candidates if is_allowed_behaviour(c)]
        dropped_by_behaviour_filter = [c for c in player_candidates if not is_allowed_behaviour(c)]
        
        # Team controls: keep if allowed OR if include_team_controls
        if include_team_controls:
            team_cols = team_candidates  # Keep all team controls
        else:
            team_cols = []
    else:
        player_num = player_candidates
        team_cols = team_candidates if include_team_controls else []
    
    # Track filtering stats
    total_numeric = len(num_cols)
    after_hard_exclusions = len(candidate_cols)
    after_behaviour_filter = len(player_num) + len(team_cols)
    
    # Build initial X
    X = df[player_num + team_cols].copy()

    # Age controls
    player_age_cols = []
    if "age" in df.columns:
        X["age"] = df["age"].astype(float)
        player_age_cols.append("age")
        if add_age_poly:
            X["age_sq"] = (df["age"].astype(float) ** 2)
            player_age_cols.append("age_sq")

    # PHASE 7: Optional league fixed effects
    league_fe_cols = []
    if enable_league_fixed_effects and "competition_id" in df.columns:
        # One-hot encode competition_id (drop first to avoid multicollinearity)
        comp_dummies = pd.get_dummies(df["competition_id"], prefix="league", drop_first=True)
        for col in comp_dummies.columns:
            X[col] = comp_dummies[col].astype(float)
            league_fe_cols.append(col)

    y = df[target].astype(float)

    # Keep track: “player features” for interpretability exclude team cols but include age terms
    player_features = player_num + player_age_cols
    team_features = team_cols + league_fe_cols  # League FE grouped with team controls

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
    # LEAKAGE DIAGNOSTICS & CORRELATION-BASED GUARDRAIL
    # ============================================
    leakage_diagnostics = {
        "correlations": {},
        "exact_matches": [],
        "dropped_high_corr": [],
        "dropped_features": [],
        "leaky_features": [],
        "guardrail_stats": {
            "total_numeric": total_numeric,
            "after_hard_exclusions": after_hard_exclusions,
            "after_behaviour_filter": after_behaviour_filter,
            "final_feature_count": 0,  # Will update after drops
        },
        "dropped_by_behaviour": dropped_by_behaviour_filter[:50],  # First 50
        "kept_features": list(X.columns)[:50],  # Will update after drops
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
    
    # Identify highly correlated features (>0.95) - ALWAYS drop these
    corr_series = pd.Series(correlations)
    high_corr_features = corr_series[corr_series.abs() > 0.95].index.tolist()
    
    # Store diagnostics
    leakage_diagnostics["correlations"] = correlations
    leakage_diagnostics["exact_matches"] = exact_matches
    leakage_diagnostics["leaky_features"] = high_corr_features
    leakage_diagnostics["dropped_high_corr"] = high_corr_features
    
    # Determine which columns to drop
    drop_cols = set(exact_matches) | set(high_corr_features)
    
    if drop_cols:
        leakage_diagnostics["dropped_features"] = list(drop_cols)
        # Drop leaky columns from X
        X = X.drop(columns=list(drop_cols))
        
        # Update feature lists
        player_features = [f for f in player_features if f not in drop_cols]
        team_features = [f for f in team_features if f not in drop_cols]
    else:
        leakage_diagnostics["dropped_features"] = []
    
    # Update final stats
    leakage_diagnostics["guardrail_stats"]["final_feature_count"] = len(X.columns)
    leakage_diagnostics["kept_features"] = list(X.columns)[:50]  # First 50 kept
    
    # ============================================
    # ABLATION MODE (EXPERIMENT ONLY - non-breaking)
    # ============================================
    if ablation_mode is not None:
        # Identify control columns (age, team, league FE) - keep these regardless
        control_cols = []
        if "age" in X.columns:
            control_cols.append("age")
        if "age_sq" in X.columns:
            control_cols.append("age_sq")
        control_cols.extend([c for c in X.columns if c.startswith("team_") or c.startswith("league_")])
        
        if ablation_mode == "M1_upstream_only":
            # Drop near-shot proxy features, keep everything else
            drop_near_shot = [c for c in X.columns if c in NEAR_SHOT_PROXY_FEATURES]
            if drop_near_shot:
                X = X.drop(columns=drop_near_shot)
                player_features = [f for f in player_features if f not in NEAR_SHOT_PROXY_FEATURES]
                leakage_diagnostics["ablation_dropped"] = drop_near_shot
        
        elif ablation_mode == "M2_near_shot_only":
            # Keep ONLY near-shot proxies + controls
            keep_cols = control_cols + [c for c in X.columns if c in NEAR_SHOT_PROXY_FEATURES]
            keep_cols = list(set(keep_cols))  # dedupe
            X = X[keep_cols]
            player_features = [f for f in player_features if f in NEAR_SHOT_PROXY_FEATURES]
            leakage_diagnostics["ablation_kept_only"] = keep_cols

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


def run_ablation_experiment(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    scouting_data: pd.DataFrame,
    target: str,
    include_team_controls: bool,
    add_age_poly: bool,
    enable_league_fe: bool,
    behaviour_only: bool,
    allow_xg_identity: bool,
    run_shuffle_test: bool = False,
) -> Dict:
    """
    Run ablation experiment comparing M0 (baseline), M1 (upstream-only), M2 (near-shot only).
    
    Returns dict with:
    - "models": {mode: trained_model}
    - "metrics": {mode: {"r2_test": float, "mae_test": float, "features": List[str]}}
    - "predictions": {mode: {"test": np.array, "scouting": np.array}}
    - "shuffle_metrics": (if run_shuffle_test) {mode: {"r2_test": float, "mae_test": float}}
    """
    results = {
        "models": {},
        "metrics": {},
        "predictions": {},
        "feature_lists": {},
    }
    
    modes = ["M0_baseline", "M1_upstream_only", "M2_near_shot_only"]
    
    for mode in modes:
        # Extract ablation mode suffix
        ablation_mode = mode if mode != "M0_baseline" else None
        
        # Build feature matrices
        X_train, y_train, _, _, _ = build_feature_matrix(
            train_data, target, include_team_controls, add_age_poly, enable_league_fe, 
            behaviour_only, allow_xg_identity, ablation_mode=ablation_mode
        )
        X_test, y_test, _, _, _ = build_feature_matrix(
            test_data, target, include_team_controls, add_age_poly, enable_league_fe,
            behaviour_only, allow_xg_identity, ablation_mode=ablation_mode
        )
        X_scouting, _, _, _, _ = build_feature_matrix(
            scouting_data, target, include_team_controls, add_age_poly, enable_league_fe,
            behaviour_only, allow_xg_identity, ablation_mode=ablation_mode
        )
        
        # Align features
        common_features = list(set(X_train.columns) & set(X_test.columns) & set(X_scouting.columns))
        X_train = X_train[common_features]
        X_test = X_test[common_features]
        X_scouting = X_scouting[common_features]
        
        # Train model
        model = fit_elastic_net_cv(X_train, y_train, random_state=42)
        
        # Predictions
        y_test_pred = model.predict(X_test.values)
        y_scouting_pred = model.predict(X_scouting.values)
        
        # Metrics
        corr_test = np.corrcoef(y_test.values, y_test_pred)[0, 1]
        r2_test = corr_test ** 2 if not np.isnan(corr_test) else 0.0
        mae_test = float(np.mean(np.abs(y_test.values - y_test_pred)))
        
        # Store results
        results["models"][mode] = model
        results["metrics"][mode] = {
            "r2_test": r2_test,
            "mae_test": mae_test,
            "n_features": len(common_features),
            "n_nonzero": int(np.sum(model.named_steps["enet"].coef_ != 0)),
        }
        results["predictions"][mode] = {
            "test": y_test_pred,
            "scouting": y_scouting_pred,
        }
        results["feature_lists"][mode] = common_features
        
        # Shuffle sanity test (should yield ~zero performance)
        if run_shuffle_test:
            if "shuffle_metrics" not in results:
                results["shuffle_metrics"] = {}
            
            rng = np.random.default_rng(42)
            y_train_shuffled = y_train.copy()
            y_train_shuffled.iloc[:] = rng.permutation(y_train_shuffled.values)
            
            shuffle_model = fit_elastic_net_cv(X_train, y_train_shuffled, random_state=43)
            y_test_shuffle_pred = shuffle_model.predict(X_test.values)
            
            corr_shuffle = np.corrcoef(y_test.values, y_test_shuffle_pred)[0, 1]
            r2_shuffle = corr_shuffle ** 2 if not np.isnan(corr_shuffle) else 0.0
            mae_shuffle = float(np.mean(np.abs(y_test.values - y_test_shuffle_pred)))
            
            results["shuffle_metrics"][mode] = {
                "r2_test": r2_shuffle,
                "mae_test": mae_shuffle,
            }
    
    # Compute dependence diagnostics (correlations between model predictions)
    results["dependence"] = {}
    for m1 in modes:
        for m2 in modes:
            if m1 < m2:  # Avoid duplicate pairs
                pred1 = results["predictions"][m1]["scouting"]
                pred2 = results["predictions"][m2]["scouting"]
                
                # Spearman rank correlation
                from scipy.stats import spearmanr
                rho, _ = spearmanr(pred1, pred2)
                
                # Top-30 overlap
                top30_m1 = set(np.argsort(-pred1)[:30])
                top30_m2 = set(np.argsort(-pred2)[:30])
                overlap = len(top30_m1 & top30_m2)
                
                results["dependence"][f"{m1}_vs_{m2}"] = {
                    "spearman": float(rho),
                    "top30_overlap": overlap,
                }
    
    return results


# -----------------------------
# 5) Streamlit UI
# -----------------------------

def main():
    st.set_page_config(page_title="SPFL ST xG Drivers", layout="wide")
    st.title("SPFL Striker Chance-Generation Drivers (StatsBomb IQ season stats)")

    with st.sidebar:
        st.header("1) StatsBomb Credentials")
        try:
            default_user = st.secrets.get("STATS_BOMB_USER", "")
            default_pass = st.secrets.get("STATS_BOMB_PASS", "")
        except:
            default_user = ""
            default_pass = ""
        
        sb_user = st.text_input("Username", value=default_user, type="default")
        sb_pass = st.text_input("Password", value=default_pass, type="password")
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
        st.header("3) Experiment Mode TEST")
        experiment_mode = st.checkbox("Enable Experiment Mode", value=False)
        run_experiment_btn = False
        run_shuffle_test = False
        experiment_config = "(M0) Baseline (current model)"
        
        st.divider()
        st.header("4) Model Settings")
        include_team_controls = st.checkbox("Include team-season controls (recommended)", value=True)
        add_age_poly = st.checkbox("Use age + age² (recommended)", value=True)
        
        st.divider()
        behaviour_only = st.checkbox("✅ Behaviour-only features (recommended)",
                                    value=True,
                                    help="Strict whitelist: only upstream player actions (touches, receptions, carries, pressures, aerials, etc.) + team context")
        
        allow_xg_identity = st.checkbox("⚠️ Allow xG-derived predictors (identity risk)",
                                       value=False,
                                       help="When OFF: excludes xG, npxg, shots_90, etc. Model uses only upstream behaviours.")
        if not behaviour_only:
            st.warning("⚠️ Behaviour-only filter is OFF - model may use outcome-contaminated features")
        elif not allow_xg_identity:
            st.caption("🛡️ Model restricted to upstream behavioural features (touches, receptions, carries, pressures, aerials, etc.)")

        n_boot = st.slider("Stability selection bootstraps", 30, 300, 120, 10)
        sample_frac = st.slider("Bootstrap sample fraction", 0.50, 1.00, 0.85, 0.05)

        top_k = st.slider("Show Top-K predictors", 3, 20, 5, 1)
        
        st.divider()
        st.header("5) Output Settings")
        sort_by = st.selectbox("Sort shortlist by", ["pred_np_xg_90", "pred_goals_90_avg_finish", "residual"], index=0,
                              help="pred_goals_90_avg_finish assumes league-average finishing (multiplier=1.0)")
        
        compute_intervals = st.checkbox("🔄 Compute model-uncertainty intervals (slow)",
                                       value=False,
                                       help="Bootstrap prediction intervals - may take 30-60 seconds")
        
        st.divider()
        st.header("6) League Adjustments (Advanced)")
        enable_league_fe = st.checkbox("⚙️ Learn league fixed effects",
                                       value=False,
                                       help="Add one-hot competition features - may improve fit but reduces interpretability")

        st.divider()
        load_btn = st.button("Load / Refresh Data", type="primary")

    # Check if model settings changed (requires refit)
    model_settings_key = f"{include_team_controls}_{add_age_poly}_{enable_league_fe}_{behaviour_only}_{allow_xg_identity}_{n_boot}_{sample_frac}"
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
    # TRAINING SET: Multi-league STs >=900 mins (dynamically filtered)
    # -----------------------------
    st.subheader("Training Set (Multi-League)")
    
    # Start with candidate leagues
    train_base = merged.copy()
    if "competition_id" in train_base.columns:
        train_base = train_base[train_base["competition_id"].isin(CANDIDATE_TRAINING_LEAGUE_IDS)]
    
    # Apply ST + minutes + age filters
    train_base = apply_common_filters(
        train_base,
        min_minutes=900,
        age_range=(18, 35),  # Training age range
        league_filter="All Leagues",  # Don't apply league filter, we already filtered by CANDIDATE_TRAINING_LEAGUE_IDS
    )
    
    # Check coverage per competition
    if "competition_id" in train_base.columns and not train_base.empty:
        coverage = train_base.groupby("competition_id").agg(
            player_seasons=("player_id", "count"),
            unique_players=("player_id", "nunique"),
            seasons=("season_id", "nunique")
        ).reset_index()
        coverage["league_name"] = coverage["competition_id"].apply(lambda x: LEAGUE_NAMES.get(x, f"Comp {x}"))
        coverage = coverage[["competition_id", "league_name", "player_seasons", "unique_players", "seasons"]]
        coverage = coverage[coverage["player_seasons"] >= 25]  # Only keep leagues with >=25 samples
        
        if coverage.empty:
            st.error("No competitions have >=25 player-seasons after filters. Cannot train model.")
            return
        
        # Filter train_base to only include competitions with sufficient data
        valid_comps = coverage["competition_id"].tolist()
        train_base = train_base[train_base["competition_id"].isin(valid_comps)]
        
        # Display training coverage
        with st.expander("📊 Training Coverage", expanded=True):
            st.dataframe(coverage, width="stretch", hide_index=True)
            
            # Warnings for low-sample competitions
            low_sample = coverage[coverage["player_seasons"] < 50]
            if not low_sample.empty:
                st.warning(f"⚠️ {len(low_sample)} competition(s) have <50 player-seasons:")
                st.dataframe(low_sample[["league_name", "player_seasons"]], width="stretch", hide_index=True)
    else:
        st.error("No training data available after filters.")
        return

    if PRIMARY_TARGET not in train_base.columns:
        st.error(f"Target '{PRIMARY_TARGET}' not found. Check your player-stats endpoint/columns.")
        st.write("Available columns:", sorted(train_base.columns))
        return

    st.caption(f"Total training samples: {len(train_base):,} player-seasons from {train_base['competition_id'].nunique()} competitions")

    # -----------------------------
    # Fit model + stability
    # -----------------------------
    st.subheader("Model: Elastic Net + Stability Selection")
    
    # Only refit if model settings changed or no cached model
    if model_settings_changed or "full_model" not in st.session_state:
        try:
            X, y, player_feats, team_feats, leakage_diag = build_feature_matrix(
                train_base,
                target=PRIMARY_TARGET,
                include_team_controls=include_team_controls,
                add_age_poly=add_age_poly,
                enable_league_fixed_effects=enable_league_fe,
                behaviour_only=behaviour_only,
                allow_xg_identity=allow_xg_identity,
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
        
        # PHASE 3: Training residual diagnostics (FULL PRECISION)
        yhat_train = full_model.predict(X.values)
        residuals_train = y.values - yhat_train  # Full precision, no rounding
        train_residual_stats = {
            "min": float(residuals_train.min()),
            "max": float(residuals_train.max()),
            "mean": float(residuals_train.mean()),
            "std": float(residuals_train.std(ddof=1)),  # Sample std
            "mean_abs": float(np.mean(np.abs(residuals_train))),
            "max_abs": float(np.max(np.abs(residuals_train))),
            "unique": int(pd.Series(residuals_train).nunique()),
            "all_zero": bool(np.allclose(residuals_train, 0, atol=1e-10))
        }
        
        # PHASE 4: Expected behaviour check
        if train_residual_stats["std"] < 1e-3:
            st.error("""
            🚨 **CRITICAL WARNING**: Residual variance is extremely small (<0.001).
            
            The model may still be reconstructing the target from its own accounting components
            rather than predicting from upstream behaviours. 
            
            **Action**: Ensure xG-identity filter is enabled (checkbox should be OFF).
            """)
        
        with st.expander("📊 Residual Diagnostics", expanded=(train_residual_stats["std"] < 0.01)):
            st.write("**Residual statistics on training set (FULL PRECISION):**")
            # Display with 6 decimal precision
            display_stats = {k: f"{v:.6f}" if isinstance(v, float) else v 
                           for k, v in train_residual_stats.items()}
            st.json(display_stats)
            
            if train_residual_stats["all_zero"]:
                st.error("⚠️ All training residuals are ~0! This indicates perfect fit (likely leakage).")
            elif train_residual_stats["std"] < 0.01:
                st.warning("⚠️ Very low residual variance - possible leakage or identity reconstruction.")
            elif train_residual_stats["std"] < 0.05:
                st.info("ℹ️ Residual std is small but non-trivial. Model may have some predictive signal.")
            else:
                st.success(f"✓ Training residuals show normal variation (std={train_residual_stats['std']:.4f})")
        
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
        st.session_state.leakage_diag = leakage_diag  # Cache for diagnostics
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
        leakage_diag = st.session_state.leakage_diag
    
    # ============================================
    # FEATURE GUARDRAIL REPORT
    # ============================================
    with st.expander("🛡️ Feature Guardrail Report", expanded=behaviour_only):
        st.markdown("**Purpose**: Show how behaviour-only filtering and hard exclusions protect against outcome leakage")
        
        guardrail_stats = leakage_diag.get("guardrail_stats", {})
        
        # Counts
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total numeric cols", guardrail_stats.get("total_numeric", 0))
        with col2:
            st.metric("After hard exclusions", guardrail_stats.get("after_hard_exclusions", 0))
        with col3:
            st.metric("After behaviour filter", guardrail_stats.get("after_behaviour_filter", 0))
        with col4:
            st.metric("Final (post-corr drop)", guardrail_stats.get("final_feature_count", 0))
        
        # Dropped by behaviour filter
        dropped_by_behaviour = leakage_diag.get("dropped_by_behaviour", [])
        if dropped_by_behaviour and behaviour_only:
            st.subheader("🚫 Dropped by behaviour-only filter")
            st.caption(f"Showing first {len(dropped_by_behaviour)} of dropped features")
            st.write(dropped_by_behaviour)
        
        # Dropped by correlation
        dropped_high_corr = leakage_diag.get("dropped_high_corr", [])
        if dropped_high_corr:
            st.subheader("⚠️ Dropped due to high correlation (|r| > 0.95)")
            st.write(dropped_high_corr)
        
        # Kept features
        kept_features = leakage_diag.get("kept_features", [])
        if kept_features:
            st.subheader("✅ Kept features (first 50)")
            st.write(kept_features)
        
        if behaviour_only:
            st.success("✓ Behaviour-only filter is ACTIVE - only upstream player actions + team context allowed")
        else:
            st.warning("⚠️ Behaviour-only filter is OFF - may include outcome-contaminated features")
    
    # ============================================
    # PHASE 0: TAUTOLOGY DIAGNOSTICS
    # ============================================
    with st.expander("🧮 Tautology Diagnostics (Is target reconstructible?)", expanded=False):
        st.markdown("""
        **Purpose**: Detect if model is reconstructing np_xg_90 from its own accounting components
        rather than predicting from upstream striker behaviours.
        """)
        
        # 1) TOP COEFFICIENTS
        st.subheader("1️⃣ Top Coefficients")
        try:
            enet_model = full_model.named_steps["enet"]
            coefs = enet_model.coef_
            coef_df = pd.DataFrame({
                "feature": all_features,
                "coef": coefs,
                "abs_coef": np.abs(coefs)
            }).sort_values("abs_coef", ascending=False).head(20)
            st.dataframe(coef_df, width=700, hide_index=True)
            
            # Check for outcome-contaminated features in top coefficients
            outcome_patterns = ["goals", "npg", "assists", "xa", "npga", "conversion"]
            top_features = coef_df["feature"].head(20).tolist()
            contaminated = [f for f in top_features 
                          if any(pattern in f.lower() for pattern in outcome_patterns)]
            if contaminated:
                st.error(f"⚠️ Outcome-contaminated features in top coefficients: {contaminated}")
            else:
                st.success("✓ No obvious outcome features in top coefficients")
        except Exception as e:
            st.error(f"Could not extract coefficients: {e}")
        
        # 2) CORRELATION-BASED LEAKAGE CHECK
        st.subheader("2️⃣ Correlation-Based Leakage Check")
        high_corr = leakage_diag.get("dropped_high_corr", [])
        if high_corr:
            st.error(f"⚠️ {len(high_corr)} features auto-dropped for |correlation| > 0.95")
            st.write(high_corr[:30])
        else:
            st.success("✓ No features with suspicious correlation (>0.95) to target")
        
        # 3) IDENTITY RECONSTRUCTION TEST
        st.subheader("3️⃣ Residual Variance Check")
        residuals_train = y.values - full_model.predict(X.values)
        resid_std = float(np.std(residuals_train, ddof=1))
        st.metric("Training residual std", f"{resid_std:.6f}")
        
        if resid_std < 0.001:
            st.error("🚨 CRITICAL: Residual std < 0.001 - model is reconstructing target!")
        elif resid_std < 0.01:
            st.warning("⚠️ Very small residual std - possible identity leakage")
        elif resid_std < 0.05:
            st.info("ℹ️ Small but non-trivial residual variance")
        else:
            st.success(f"✓ Healthy residual variance - model uses upstream behaviours")
    
    # ============================================
    # PHASE 2: TEMPORAL VALIDATION
    # ============================================
    with st.expander("📅 Temporal Validation (Within-League)", expanded=False):
        st.markdown("Test model on most recent season within each competition")
        
        if "canonical_season" not in train_base.columns or train_base["canonical_season"].isna().all():
            st.warning("Cannot perform temporal validation: canonical_season not available")
        else:
            # Build temporal splits per competition
            temporal_results = []
            excluded_comps = []
            
            for comp_id in train_base["competition_id"].unique():
                comp_data = train_base[train_base["competition_id"] == comp_id].copy()
                comp_data = comp_data[comp_data["canonical_season"].notna()]
                
                if len(comp_data) == 0:
                    continue
                
                max_season = comp_data["canonical_season"].max()
                test_mask = comp_data["canonical_season"] == max_season
                train_mask = comp_data["canonical_season"] < max_season
                
                test_rows = test_mask.sum()
                train_rows = train_mask.sum()
                
                if train_rows < 10 or test_rows < 5:
                    excluded_comps.append({
                        "comp_id": comp_id,
                        "league": LEAGUE_NAMES.get(comp_id, f"Comp {comp_id}"),
                        "reason": f"train={train_rows}, test={test_rows} (need train>=10, test>=5)"
                    })
                    continue
                
                temporal_results.append({
                    "comp_id": comp_id,
                    "league": LEAGUE_NAMES.get(comp_id, f"Comp {comp_id}"),
                    "train_rows": train_rows,
                    "test_rows": test_rows,
                    "test_season": int(max_season)
                })
            
            if temporal_results:
                # Build features for temporal validation
                temp_train_list = []
                temp_test_list = []
                
                for res in temporal_results:
                    comp_data = train_base[train_base["competition_id"] == res["comp_id"]].copy()
                    comp_data = comp_data[comp_data["canonical_season"].notna()]
                    max_season = comp_data["canonical_season"].max()
                    
                    temp_train_list.append(comp_data[comp_data["canonical_season"] < max_season])
                    temp_test_list.append(comp_data[comp_data["canonical_season"] == max_season])
                
                temp_train_df = pd.concat(temp_train_list, ignore_index=True)
                temp_test_df = pd.concat(temp_test_list, ignore_index=True)
                
                try:
                    X_temp_train, y_temp_train, _, _, _ = build_feature_matrix(
                        temp_train_df, PRIMARY_TARGET, include_team_controls, add_age_poly, enable_league_fe, behaviour_only, allow_xg_identity
                    )
                    X_temp_test, y_temp_test, _, _, _ = build_feature_matrix(
                        temp_test_df, PRIMARY_TARGET, include_team_controls, add_age_poly, enable_league_fe, behaviour_only, allow_xg_identity
                    )
                    
                    # Align features
                    common_features = list(set(X_temp_train.columns) & set(X_temp_test.columns))
                    X_temp_train = X_temp_train[common_features]
                    X_temp_test = X_temp_test[common_features]
                    
                    # Train temporal model
                    temp_model = fit_elastic_net_cv(X_temp_train, y_temp_train, random_state=42)
                    
                    # Predict on test
                    y_temp_pred = temp_model.predict(X_temp_test.values)
                    
                    # Metrics
                    corr_temp = np.corrcoef(y_temp_test.values, y_temp_pred)[0, 1]
                    r2_temp = corr_temp ** 2 if not np.isnan(corr_temp) else 0.0
                    mae_temp = np.mean(np.abs(y_temp_test.values - y_temp_pred))
                    
                    st.success(f"✓ Temporal validation: R²={r2_temp:.3f}, MAE={mae_temp:.3f}")
                    
                    # Interpretation
                    if r2_temp > 0.15:
                        st.info("👍 Model shows meaningful predictive power on future seasons")
                    elif r2_temp > 0.05:
                        st.warning("⚠️ Model shows weak but detectable signal on future seasons")
                    else:
                        st.error("❌ Model shows little predictive power on future seasons")
                    
                    # Details table
                    st.markdown("**Per-competition breakdown:**")
                    temp_df = pd.DataFrame(temporal_results)
                    st.dataframe(temp_df, width="stretch", hide_index=True)
                    
                    if excluded_comps:
                        st.markdown("**Excluded competitions:**")
                        st.dataframe(pd.DataFrame(excluded_comps), width="stretch", hide_index=True)
                    
                except Exception as e:
                    st.error(f"Temporal validation failed: {e}")
            else:
                st.warning("No competitions met temporal validation criteria (train>=10, test>=5)")
    
    # ============================================
    # PHASE 3: CROSS-LEAGUE TRANSFERABILITY
    # ============================================
    with st.expander("🌍 Cross-League Transferability (Train non-SPFL → Test SPFL)", expanded=False):
        st.markdown("Test if model trained on other leagues can predict SPFL performance")
        
        if "competition_id" not in train_base.columns:
            st.warning("Cannot perform cross-league test: competition_id not available")
        elif 51 not in train_base["competition_id"].values:
            st.warning("Cannot perform cross-league test: No SPFL data in training set")
        else:
            # Split data
            non_spfl_data = train_base[train_base["competition_id"] != 51].copy()
            spfl_data = train_base[train_base["competition_id"] == 51].copy()
            
            if len(non_spfl_data) < 25:
                st.warning(f"Insufficient non-SPFL training data: {len(non_spfl_data)} rows (need >=25)")
            elif len(spfl_data) < 10:
                st.warning(f"Insufficient SPFL test data: {len(spfl_data)} rows (need >=10)")
            else:
                # Use most recent SPFL season for testing if canonical_season available
                if "canonical_season" in spfl_data.columns and not spfl_data["canonical_season"].isna().all():
                    max_spfl_season = spfl_data["canonical_season"].max()
                    spfl_test = spfl_data[spfl_data["canonical_season"] == max_spfl_season].copy()
                    if len(spfl_test) < 10:
                        spfl_test = spfl_data  # Fallback to all SPFL data
                else:
                    spfl_test = spfl_data
                
                try:
                    # Build feature matrices
                    X_cross_train, y_cross_train, _, _, _ = build_feature_matrix(
                        non_spfl_data, PRIMARY_TARGET, include_team_controls, add_age_poly, enable_league_fe, behaviour_only, allow_xg_identity
                    )
                    X_cross_test, y_cross_test, _, _, _ = build_feature_matrix(
                        spfl_test, PRIMARY_TARGET, include_team_controls, add_age_poly, enable_league_fe, behaviour_only, allow_xg_identity
                    )
                    
                    # Align features
                    common_features = list(set(X_cross_train.columns) & set(X_cross_test.columns))
                    X_cross_train = X_cross_train[common_features]
                    X_cross_test = X_cross_test[common_features]
                    
                    # Train on non-SPFL
                    cross_model = fit_elastic_net_cv(X_cross_train, y_cross_train, random_state=42)
                    
                    # Predict on SPFL
                    y_cross_pred = cross_model.predict(X_cross_test.values)
                    
                    # Metrics
                    corr_cross = np.corrcoef(y_cross_test.values, y_cross_pred)[0, 1]
                    r2_cross = corr_cross ** 2 if not np.isnan(corr_cross) else 0.0
                    mae_cross = np.mean(np.abs(y_cross_test.values - y_cross_pred))
                    
                    st.success(f"✓ Cross-league test: R²={r2_cross:.3f}, MAE={mae_cross:.3f}")
                    st.caption(f"Train: {len(non_spfl_data)} non-SPFL rows | Test: {len(spfl_test)} SPFL rows")
                    
                    # Interpretation
                    if r2_cross > 0.10:
                        st.info("👍 Model transfers reasonably to SPFL from other leagues")
                    elif r2_cross > 0.03:
                        st.warning("⚠️ Model shows weak transferability to SPFL")
                    else:
                        st.error("❌ Model does not transfer well to SPFL from other leagues")
                    
                except Exception as e:
                    st.error(f"Cross-league test failed: {e}")
    
    # ============================================
    # ABLATION EXPERIMENT (OPTIONAL)
    # ============================================
    if experiment_mode and run_experiment_btn:
        with st.spinner("Running ablation experiment (M0, M1, M2)..."):
            try:
                # Prepare train/test split for temporal validation
                if "canonical_season" in train.columns:
                    train_sorted = train.sort_values("canonical_season")
                    n_train = len(train_sorted)
                    split_idx = max(int(0.7 * n_train), 10)
                    temporal_train = train_sorted.iloc[:split_idx]
                    temporal_test = train_sorted.iloc[split_idx:]
                else:
                    # Fallback: random 70/30 split
                    from sklearn.model_selection import train_test_split
                    temporal_train, temporal_test = train_test_split(train, test_size=0.3, random_state=42)
                
                # Run experiment
                experiment_results = run_ablation_experiment(
                    train_data=temporal_train,
                    test_data=temporal_test,
                    scouting_data=scouting,
                    target=PRIMARY_TARGET,
                    include_team_controls=include_team_controls,
                    add_age_poly=add_age_poly,
                    enable_league_fe=enable_league_fe,
                    behaviour_only=behaviour_only,
                    allow_xg_identity=allow_xg_identity,
                    run_shuffle_test=run_shuffle_test,
                )
                
                # Cache results
                st.session_state["ablation_results"] = experiment_results
                st.success("✓ Experiment completed")
                
            except Exception as e:
                st.error(f"Experiment failed: {e}")
    
    # Display experiment results (if available)
    if experiment_mode and "ablation_results" in st.session_state:
        with st.expander("🔬 Ablation Experiment Results", expanded=True):
            results = st.session_state["ablation_results"]
            
            st.markdown("""
            **Test:** Does model performance depend on near-shot proxy features vs upstream behaviours?
            
            - **M0 (Baseline)**: All behaviour features (current model)
            - **M1 (Upstream-only)**: Removes near-shot proxies (touches_inside_box, passes_into_box, etc.)
            - **M2 (Near-shot only)**: Keeps only near-shot proxies + controls (age, team, league)
            """)
            
            # Metrics comparison table
            st.subheader("📊 Test Set Performance")
            metrics_df = pd.DataFrame({
                "Model": ["M0_baseline", "M1_upstream_only", "M2_near_shot_only"],
                "R²": [results["metrics"][m]["r2_test"] for m in ["M0_baseline", "M1_upstream_only", "M2_near_shot_only"]],
                "MAE": [results["metrics"][m]["mae_test"] for m in ["M0_baseline", "M1_upstream_only", "M2_near_shot_only"]],
                "Features": [results["metrics"][m]["n_features"] for m in ["M0_baseline", "M1_upstream_only", "M2_near_shot_only"]],
                "Non-zero": [results["metrics"][m]["n_nonzero"] for m in ["M0_baseline", "M1_upstream_only", "M2_near_shot_only"]],
            })
            st.dataframe(metrics_df.style.format({"R²": "{:.4f}", "MAE": "{:.4f}"}), width="stretch", hide_index=True)
            
            # Interpretation
            r2_m0 = results["metrics"]["M0_baseline"]["r2_test"]
            r2_m1 = results["metrics"]["M1_upstream_only"]["r2_test"]
            r2_m2 = results["metrics"]["M2_near_shot_only"]["r2_test"]
            
            if r2_m2 > r2_m1 * 1.5:
                st.warning("⚠️ Performance heavily depends on near-shot proxies - model may be learning shot generation rather than upstream behaviours")
            elif r2_m1 > r2_m2:
                st.success("✓ Upstream behaviours provide stronger signal than near-shot proxies - defensible behaviour-only model")
            else:
                st.info("ℹ️ Mixed signal - both upstream and near-shot features contribute")
            
            # Dependence diagnostics
            st.subheader("🔗 Model Agreement Diagnostics")
            st.caption("How similar are the model rankings?")
            
            dep_df = []
            for key, vals in results["dependence"].items():
                m1, m2 = key.split("_vs_")
                dep_df.append({
                    "Comparison": f"{m1} vs {m2}",
                    "Spearman ρ": vals["spearman"],
                    "Top-30 overlap": vals["top30_overlap"],
                })
            
            dep_table = pd.DataFrame(dep_df)
            st.dataframe(dep_table.style.format({"Spearman ρ": "{:.3f}"}), width="stretch", hide_index=True)
            
            if len(dep_df) > 0:
                avg_spearman = np.mean([d["Spearman ρ"] for d in dep_df])
                if avg_spearman < 0.5:
                    st.error("❌ Low agreement between models - predictions are highly dependent on feature set choice")
                elif avg_spearman < 0.8:
                    st.warning("⚠️ Moderate agreement - feature set matters for ranking")
                else:
                    st.success("✓ High agreement - rankings are robust to feature set changes")
            
            # Shuffle sanity test
            if run_shuffle_test and "shuffle_metrics" in results:
                st.subheader("🎲 Shuffle Sanity Test")
                st.caption("Training on shuffled target should yield ~zero test performance")
                
                shuffle_df = pd.DataFrame({
                    "Model": ["M0_baseline", "M1_upstream_only", "M2_near_shot_only"],
                    "Shuffle R²": [results["shuffle_metrics"][m]["r2_test"] for m in ["M0_baseline", "M1_upstream_only", "M2_near_shot_only"]],
                    "Shuffle MAE": [results["shuffle_metrics"][m]["mae_test"] for m in ["M0_baseline", "M1_upstream_only", "M2_near_shot_only"]],
                })
                st.dataframe(shuffle_df.style.format({"Shuffle R²": "{:.4f}", "Shuffle MAE": "{:.4f}"}), width="stretch", hide_index=True)
                
                max_shuffle_r2 = max([results["shuffle_metrics"][m]["r2_test"] for m in ["M0_baseline", "M1_upstream_only", "M2_near_shot_only"]])
                if max_shuffle_r2 > 0.05:
                    st.error("🚨 Shuffle test shows non-trivial R² - possible data leakage!")
                else:
                    st.success("✓ Shuffle test passed - no spurious correlations detected")
    
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
    # If league FE enabled, generate one-hot encoding for scouting data
    if enable_league_fe and "competition_id" in scouting.columns:
        # Get same dummy columns as training (must match)
        scouting_dummies = pd.get_dummies(scouting["competition_id"], prefix="league", drop_first=True)
        for col in scouting_dummies.columns:
            scouting[col] = scouting_dummies[col].astype(float)
    
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
    scouting["pred_goals_90_avg_finish"] = preds * 1.0  # Assumes league-average finishing
    
    # Compute residuals, but set to NaN for players outside training competitions
    # to avoid misleading residual-based rankings
    if "competition_id" in scouting.columns:
        training_comp_ids = set(train_base["competition_id"].unique()) if "competition_id" in train_base.columns else set()
        in_training_comps = scouting["competition_id"].isin(training_comp_ids)
        scouting["residual"] = np.where(
            in_training_comps,
            scouting[PRIMARY_TARGET].astype(float) - scouting["pred_np_xg_90"],
            np.nan
        )
    else:
        # PHASE 3: Compute residuals at FULL PRECISION (no rounding before calculation)
        scouting["residual"] = scouting[PRIMARY_TARGET].astype(float) - scouting["pred_np_xg_90"]
    
    # PHASE 3: Residual diagnostics for scouting pool (FULL PRECISION)
    residuals_scout = scouting["residual"].dropna()
    if len(residuals_scout) > 0:
        scout_residual_stats = {
            "min": float(residuals_scout.min()),
            "max": float(residuals_scout.max()),
            "mean": float(residuals_scout.mean()),
            "std": float(residuals_scout.std(ddof=1)),  # Sample std
            "mean_abs": float(np.mean(np.abs(residuals_scout.values))),
            "max_abs": float(np.max(np.abs(residuals_scout.values))),
            "unique": int(residuals_scout.nunique()),
            "all_zero": bool(np.allclose(residuals_scout.values, 0, atol=1e-10))
        }
        
        with st.expander("📊 Residual Diagnostics (Scouting Pool)", expanded=False):
            st.write("**Residual statistics on scouting pool (FULL PRECISION):**")
            # Display with 6 decimal precision
            display_stats = {k: f"{v:.6f}" if isinstance(v, float) else v 
                           for k, v in scout_residual_stats.items()}
            st.json(display_stats)
            if scout_residual_stats["all_zero"]:
                st.error("⚠️ All scouting residuals are ~0! This indicates perfect predictions (likely leakage).")
            elif scout_residual_stats["std"] < 0.01:
                st.warning("⚠️ Very low residual variance - possible leakage or overfitting.")
            else:
                st.success("✓ Scouting residuals show normal variation.")
    
    # ============================================
    # PHASE 6: OPTIONAL PREDICTION INTERVALS
    # ============================================
    if compute_intervals:
        import hashlib
        
        # Create stable hash for caching
        X_hash = hashlib.md5(pd.util.hash_pandas_object(X, index=True).values).hexdigest()
        scout_hash = hashlib.md5(pd.util.hash_pandas_object(scout_X, index=True).values).hexdigest()
        cache_key = f"intervals_{X_hash}_{scout_hash}_{n_boot}_{sample_frac}"
        
        if cache_key in st.session_state:
            intervals = st.session_state[cache_key]
            st.caption("✓ Using cached prediction intervals")
        else:
            with st.spinner("Computing bootstrap prediction intervals (this may take 30-60s)..."):
                intervals_start = time.time()
                
                # Bootstrap with faster CV settings
                n_boot_intervals = 50  # Reduced from stability selection
                interval_preds = []
                
                rng = np.random.default_rng(42)
                n_samples = len(X)
                
                for b in range(n_boot_intervals):
                    # Resample training data
                    idx = rng.choice(np.arange(n_samples), size=max(10, int(0.85 * n_samples)), replace=True)
                    X_boot = X.values[idx, :]
                    y_boot = y.values[idx]
                    
                    # Faster ElasticNetCV
                    cv_fast = KFold(n_splits=3, shuffle=True, random_state=int(rng.integers(1, 1_000_000)))
                    model_boot = Pipeline(steps=[
                        ("scaler", StandardScaler(with_mean=True, with_std=True)),
                        ("enet", ElasticNetCV(
                            l1_ratio=[0.1, 0.5, 0.9],
                            alphas=np.logspace(-4, 1, 30),
                            cv=cv_fast,
                            max_iter=10000,
                            random_state=int(rng.integers(1, 1_000_000)),
                        )),
                    ])
                    model_boot.fit(X_boot, y_boot)
                    
                    # Predict on scouting pool
                    pred_boot = model_boot.predict(scout_X.values)
                    interval_preds.append(pred_boot)
                
                interval_preds = np.array(interval_preds)
                intervals = {
                    "lower": np.percentile(interval_preds, 5, axis=0),
                    "upper": np.percentile(interval_preds, 95, axis=0),
                    "median": np.percentile(interval_preds, 50, axis=0)
                }
                
                st.session_state[cache_key] = intervals
                intervals_time = time.time() - intervals_start
                st.caption(f"✓ Computed intervals in {intervals_time:.1f}s")
        
        # Add intervals to scouting dataframe
        scouting["pred_lower_90"] = intervals["lower"]
        scouting["pred_upper_90"] = intervals["upper"]
        scouting["pred_median"] = intervals["median"]
        
        st.info("📊 Prediction intervals represent **model uncertainty**, not confidence about true player ability. Wide intervals indicate features don't strongly constrain predictions.")
    
    # Cache scouting stats for instant player selection
    st.session_state.scouting_pool = scouting
    st.session_state.scouting_means = {col: scouting[col].astype(float).mean() 
                                       for col in scouting.select_dtypes(include=[np.number]).columns}
    st.session_state.scouting_stds = {col: scouting[col].astype(float).std(ddof=0) 
                                      for col in scouting.select_dtypes(include=[np.number]).columns}

    # Display
    base_cols = ["player_name", "age", "primary_position", "team_name", "league_name", "season_name",
                 "minutes", PRIMARY_TARGET, "pred_np_xg_90", "pred_goals_90_avg_finish", "residual"]
    
    if compute_intervals:
        base_cols.extend(["pred_lower_90", "pred_upper_90"])
    
    show_cols = [c for c in base_cols if c in scouting.columns]
    
    # Warning for residual sort
    if sort_by == "residual":
        st.warning("⚠️ Residual-based ranking: Only valid for players from training competitions. Others shown as NaN.")

    # Sort by user selection
    sort_ascending = sort_by == "residual"  # residual can be positive or negative
    scouting_sorted = scouting.sort_values(sort_by, ascending=sort_ascending, na_position='last')
    
    # rank by selected metric (1 = best)
    topn = st.slider("Shortlist size", 10, 100, 30, 5)
    shortlist = scouting_sorted.head(topn).copy()
    shortlist["rank"] = range(1, len(shortlist) + 1)

    # Update show_cols to include rank at the beginning
    show_cols_with_rank = ["rank"] + show_cols if "rank" in shortlist.columns else show_cols
    
    # PHASE 3: Format with proper precision - NEVER round residual to 3 decimals
    # Use 6 decimal precision for residual, 3 for predictions
    for col in ["pred_np_xg_90", "pred_goals_90_avg_finish", PRIMARY_TARGET]:
        if col in shortlist.columns:
            shortlist[col] = shortlist[col].astype(float).round(3)
    
    # PHASE 3: Keep residual at higher precision (6 decimals max)
    if "residual" in shortlist.columns:
        shortlist["residual"] = shortlist["residual"].astype(float)  # Full precision, no rounding

    # Display with styling for residual column (6 decimal format)
    def style_residual(val):
        if pd.isna(val):
            return ""
        return f"{val:.6f}"
    
    shortlist_display = shortlist[show_cols_with_rank].copy()
    if "residual" in shortlist_display.columns:
        shortlist_styled = shortlist_display.style.format({"residual": style_residual})
        st.dataframe(shortlist_styled, width="stretch", hide_index=True)
    else:
        st.dataframe(shortlist_display, width="stretch", hide_index=True)

    st.caption(
        "Tip: Residual > 0 = producing more np_xg than the model expects given traits + team context (could be role/system quirks). "
        "Residual < 0 = underproducing vs traits/context (possible buy-low or role mismatch)."
    )

    with st.expander("Download shortlist as CSV"):
        # PHASE 3: Export residual at full precision (6 decimals)
        csv_df = shortlist[show_cols_with_rank].copy()
        if "residual" in csv_df.columns:
            csv_df["residual"] = csv_df["residual"].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "")
        csv = csv_df.to_csv(index=False).encode("utf-8")
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
            st.write(f"Rank: **#{int(r.get('rank',0))}** | Actual {PRIMARY_TARGET}: **{r.get(PRIMARY_TARGET, np.nan):.3f}** | Predicted np_xG/90: **{r.get('pred_np_xg_90', np.nan):.3f}**")

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
