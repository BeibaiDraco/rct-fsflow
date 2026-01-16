#!/usr/bin/env python
import argparse, os, json, numpy as np, pandas as pd
from saccflow.io import load_manifest, list_sessions, read_trials
from saccflow.features import ensure_columns

def main():
    ap = argparse.ArgumentParser(description="Sanity-check RCT_02 export for saccade analysis.")
    ap.add_argument("--root", required=True, help="Path to RCT_02 root (contains manifest.json)")
    ap.add_argument("--out", default="results_sacc/qc/dataset_summary.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    man = load_manifest(args.root)
    sessions = list_sessions(args.root)

    rows = []
    for sid in sessions:
        try:
            df = read_trials(args.root, sid)
            df = ensure_columns(df)
            n = len(df)
            has_sacc = (~df["Align_to_sacc_on"].isna()).mean()
            has_cat  = (~df["Align_to_cat_stim_on"].isna()).mean()
            has_sgn  = (~df["saccade_location_sign"].isna()).mean() if "saccade_location_sign" in df else 0.0
            rows.append(dict(
                session=sid, n_trials=int(n),
                pct_has_cat=float(has_cat)*100,
                pct_has_sacc=float(has_sacc)*100,
                pct_has_saccade_sign=float(has_sgn)*100,
            ))
        except Exception as e:
            rows.append(dict(session=sid, error=str(e)))

    summary = dict(root=args.root, sessions=rows)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.out}")
    # Pretty print a top-line view
    ok = [r for r in rows if "error" not in r]
    if ok:
        total = sum(r["n_trials"] for r in ok)
        print(f"Sessions checked: {len(ok)}; total trials: {total}")
        print(f"Mean % with sacc_on: {np.mean([r['pct_has_sacc'] for r in ok]):.1f}")
        print(f"Mean % with saccade_sign: {np.mean([r['pct_has_saccade_sign'] for r in ok]):.1f}")

if __name__ == "__main__":
    main()
