# saccflow/io.py
import json, os, re
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    _PARQUET_ENGINE = "pyarrow"
    pd.Series([1]).to_frame().to_parquet("/tmp/_parq_test.parquet", engine=_PARQUET_ENGINE)
except Exception:
    _PARQUET_ENGINE = None

def load_manifest(root: str) -> Dict:
    mpath = os.path.join(root, "manifest.json")
    with open(mpath, "r") as f:
        return json.load(f)

def list_sessions(root: str) -> List[str]:
    return sorted([d for d in os.listdir(root) if re.fullmatch(r"\d{8}", d)])

def trials_path(root: str, sid: str) -> Tuple[Optional[str], Optional[str]]:
    sp = os.path.join(root, sid, "trials.parquet")
    sc = os.path.join(root, sid, "trials.csv")
    return (sp if os.path.exists(sp) else None, sc if os.path.exists(sc) else None)

def read_trials(root: str, sid: str) -> pd.DataFrame:
    pparq, pcsv = trials_path(root, sid)
    if pparq and _PARQUET_ENGINE:
        return pd.read_parquet(pparq, engine=_PARQUET_ENGINE)
    elif pcsv:
        return pd.read_csv(pcsv)
    else:
        raise FileNotFoundError(f"No trials.parquet/csv for session {sid}")

def list_areas_from_manifest(manifest: Dict, sid: str) -> List[str]:
    for s in manifest.get("sessions", []):
        if str(s.get("session_id")) == str(sid):
            return sorted(s.get("areas", []))
    return []

def area_dirs(root: str, sid: str, area: str) -> Tuple[str, str, str]:
    sess_dir = os.path.join(root, sid)
    area_dir = os.path.join(sess_dir, "areas", area)
    spikes_dir = os.path.join(area_dir, "spikes")
    return sess_dir, area_dir, spikes_dir

def read_units_json(root: str, sid: str, area: str) -> List[Dict]:
    _, area_dir, _ = area_dirs(root, sid, area)
    jpath = os.path.join(area_dir, "units.json")
    if not os.path.exists(jpath):
        return []
    with open(jpath, "r") as f:
        return json.load(f)

def read_spike_times(root: str, sid: str, area: str, rel_file: str):
    import h5py
    _, area_dir, _ = area_dirs(root, sid, area)
    fpath = os.path.join(area_dir, rel_file)
    with h5py.File(fpath, "r") as h5:
        return h5["/t"][:].astype("float64")  # seconds

# NEW: load binned cache (results_sacc/<sid>/caches/area_<AREA>.npz)
def load_area_cache(out_root: str, sid: str, area: str) -> Dict:
    f = os.path.join(out_root, sid, "caches", f"area_{area}.npz")
    if not os.path.exists(f):
        raise FileNotFoundError(f)
    d = np.load(f, allow_pickle=True)
    meta = {}
    if "meta" in d:
        m = d["meta"]
        if isinstance(m, np.ndarray) and m.dtype == object:
            m = m.item()
        try:
            meta = json.loads(m)
        except Exception:
            meta = {}
    ret = dict(
        Z=d["Z"].astype("float32"),
        X=d["X"].astype("float32"),
        time=d["time"].astype("float64"),
        C=d["lab_C"].astype("float32") if "lab_C" in d else None,
        S=d["lab_S"].astype("float32") if "lab_S" in d else None,
        CC=d["lab_CC"].astype("float32") if "lab_CC" in d else None,
        orientation=d["lab_orientation"] if "lab_orientation" in d else None,
        is_correct=d["lab_is_correct"].astype("bool") if "lab_is_correct" in d else None,
        trial_index=d["lab_trial_index"].astype("int64") if "lab_trial_index" in d else None,
        meta=meta,
    )
    return ret
