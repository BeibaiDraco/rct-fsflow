#!/usr/bin/env python
import argparse, os, json, numpy as np
from typing import List
from saccflow.io import load_manifest, list_areas_from_manifest, read_trials, read_units_json, read_spike_times
from saccflow.features import clean_trials_for_saccade, trial_event_times
from saccflow.binning import bin_area

def _infer_areas(root: str, sid: str, man: dict) -> List[str]:
    areas = list_areas_from_manifest(man, sid)
    if not areas:
        # fallback: scan areas directory
        adir = os.path.join(root, sid, "areas")
        if os.path.isdir(adir):
            areas = sorted([d for d in os.listdir(adir) if os.path.isdir(os.path.join(adir,d))])
    return areas

def main():
    ap = argparse.ArgumentParser(description="Build saccade-aligned caches (trials × bins × units) per area.")
    ap.add_argument("--root", required=True, help="RCT_02 root")
    ap.add_argument("--sid", required=True, help="Session id (8-digit)")
    ap.add_argument("--areas", nargs="*", default=None, help="Areas to include (default: from manifest)")
    ap.add_argument("--t0", type=float, default=-0.40, help="Window start (s) relative to saccade onset")
    ap.add_argument("--t1", type=float, default= 0.10, help="Window end (s) relative to saccade onset")
    ap.add_argument("--bin_ms", type=float, default=10.0, help="Bin size (ms)")
    ap.add_argument("--out_root", default="results_sacc", help="Output root")
    args = ap.parse_args()

    man = load_manifest(args.root)
    areas = args.areas or _infer_areas(args.root, args.sid, man)
    if not areas:
        raise RuntimeError(f"No areas found for session {args.sid}")

    # trials
    tr = read_trials(args.root, args.sid)
    tr = clean_trials_for_saccade(tr)
    # drop NaNs in alignment just in case
    tr = tr[~tr["Align_to_sacc_on"].isna()]
    tr = tr.reset_index(drop=True)

    # labels to store
    labels = {
        "C": tr["C"].to_numpy(float),
        "S": tr["S"].to_numpy(float),
        "CC": tr["CC"].to_numpy(float) if "CC" in tr.columns else np.full(len(tr), np.nan),
        "targets_vert": tr["targets_vert"].to_numpy(float),
        "orientation": tr["orientation"].astype(str).to_numpy(),
        "PT_ms": tr["PT_ms"].to_numpy(float),
        "is_correct": tr["is_correct"].to_numpy(bool),
        "trial_index": tr["trial_index"].to_numpy(int),
    }
    ev = trial_event_times(tr, align="sacc")  # seconds
    bin_s = args.bin_ms / 1000.0

    out_dir = os.path.join(args.out_root, args.sid, "caches")
    os.makedirs(out_dir, exist_ok=True)

    meta_common = {
        "align_event": "sacc",
        "window": [args.t0, args.t1],
        "bin_s": bin_s,
        "n_trials": int(len(tr)),
    }

    for area in areas:
        units = read_units_json(args.root, args.sid, area)
        if not units:
            print(f"[{args.sid}][{area}] no units.json; skipping")
            continue
        # Load spikes (seconds)
        spikes = [read_spike_times(args.root, args.sid, area, u["file"]) for u in units]
        # Bin
        X, t = bin_area(spikes, ev, args.t0, args.t1, bin_s)  # (trials, bins, units)
        # z-score per unit across trials×time (optional; keep raw counts too if you like)
        Xflat = X.reshape(-1, X.shape[-1])  # (trials*bins, units)
        mu = Xflat.mean(axis=0, dtype=np.float64)
        sd = Xflat.std(axis=0, ddof=1, dtype=np.float64)
        sd[sd == 0] = 1.0
        Z = (X - mu) / sd

        out_npz = os.path.join(out_dir, f"area_{area}.npz")
        np.savez_compressed(
            out_npz,
            X=X.astype(np.float32),
            Z=Z.astype(np.float32),
            time=t.astype(np.float32),
            **{f"lab_{k}": v for k, v in labels.items()},
            meta=json.dumps({**meta_common, "area": area, "n_units": int(len(units))})
        )
        print(f"[{args.sid}][{area}] wrote {out_npz}  X={X.shape}")

if __name__ == "__main__":
    main()
