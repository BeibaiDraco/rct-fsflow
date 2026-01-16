# paper_project/paperflow/io.py
from __future__ import annotations
import os, json, re
from typing import Dict, List, Optional
import pandas as pd

VALID_AREAS = {"MFEF","MLIP","MSC","SFEF","SLIP","SSC"}

def load_manifest(root: str) -> Optional[Dict]:
    m = os.path.join(root, "manifest.json")
    if os.path.exists(m):
        with open(m, "r") as f:
            return json.load(f)
    return None


def list_sessions(root: str) -> list[str]:
    """
    Robustly discover sessions: union(manifest sessions, directory names).
    """
    sids = set()

    # 1) manifest.json (if present)
    man = load_manifest(root)
    if man and isinstance(man, dict) and "sessions" in man:
        for s in man["sessions"]:
            sid = s.get("session_id")
            if sid is None: continue
            sid = str(sid)
            if re.fullmatch(r"\d{8}", sid):
                sids.add(sid)

    # 2) directory names
    try:
        for d in os.listdir(root):
            if re.fullmatch(r"\d{8}", d) and os.path.isdir(os.path.join(root, d)):
                sids.add(d)
    except FileNotFoundError:
        pass

    return sorted(sids)

def list_areas(root: str, sid: str) -> List[str]:
    aroot = os.path.join(root, sid, "areas")
    if not os.path.isdir(aroot): return []
    out = []
    for name in os.listdir(aroot):
        p = os.path.join(aroot, name)
        if os.path.isdir(p) and name in VALID_AREAS:
            out.append(name)
    return sorted(out)

def trials_path(root: str, sid: str) -> Optional[str]:
    p_parq = os.path.join(root, sid, "trials.parquet")
    p_csv  = os.path.join(root, sid, "trials.csv")
    if os.path.exists(p_parq): return p_parq
    if os.path.exists(p_csv):  return p_csv
    return None

def read_trials(root: str, sid: str) -> pd.DataFrame:
    tp = trials_path(root, sid)
    if tp is None:
        raise FileNotFoundError(f"No trials.parquet/csv for {sid} under {root}")
    if tp.endswith(".parquet"):
        return pd.read_parquet(tp)
    return pd.read_csv(tp)

def has_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns

def safe_pct(x: pd.Series) -> float:
    if x is None or len(x)==0: return 0.0
    return float((~x.isna()).mean() * 100.0)

def write_json(obj: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
