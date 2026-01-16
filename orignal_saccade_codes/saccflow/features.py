from typing import Dict, Tuple
import numpy as np
import pandas as pd

REQUIRED_COLS = [
    "is_rct","is_correct","targets_vert","category","chosen_cat",
    "saccade_location_sign","Align_to_cat_stim_on","Align_to_sacc_on","PT_ms"
]

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df

def clean_trials_for_saccade(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Keep RCT trials with both cat_stim and sacc_on timestamps
    - Keep non-NaN saccade_location_sign (S)
    - Compute C, CC (±1), Orientation (vert/horiz), PT_ms clean
    """
    df = ensure_columns(df).copy()

    # timestamp sanity
    df = df[~df["Align_to_cat_stim_on"].isna() & ~df["Align_to_sacc_on"].isna()]

    # labels
    df["C"] = df["category"].astype("float")
    df["CC"] = df["chosen_cat"].astype("float")
    df["S"] = df["saccade_location_sign"].astype("float")

    # orientation: 1=vertical, 0=horizontal → str for downstream readability
    df["orientation"] = np.where(df["targets_vert"].astype("float") == 1, "vertical", "horizontal")

    # drop rows without S
    df = df[~df["S"].isna()]

    # processing time if not present
    if "PT_ms" not in df.columns or df["PT_ms"].isna().all():
        df["PT_ms"] = 1000.0*(df["Align_to_sacc_on"] - df["Align_to_cat_stim_on"])

    # index retention
    df["trial_index"] = df["trial_index"].astype("int64", errors="ignore")

    return df

def trial_event_times(df: pd.DataFrame, align: str = "sacc") -> np.ndarray:
    if align == "sacc":
        return df["Align_to_sacc_on"].to_numpy(dtype=float)
    elif align == "cat":
        return df["Align_to_cat_stim_on"].to_numpy(dtype=float)
    else:
        raise ValueError("align must be 'sacc' or 'cat'")
