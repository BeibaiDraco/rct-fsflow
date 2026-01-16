#!/usr/bin/env python
import argparse, os
import pandas as pd
from saccflow.io import list_sessions, read_trials
from saccflow.features import clean_trials_for_saccade

def main():
    ap = argparse.ArgumentParser(description="Standardize/enrich trials for saccade analysis.")
    ap.add_argument("--root", required=True, help="Path to RCT_02 root")
    ap.add_argument("--sid", default=None, help="Specific session id (8-digit). If omitted, do all.")
    ap.add_argument("--out_root", default="results_sacc", help="Output root for enriched trials")
    args = ap.parse_args()

    sids = [args.sid] if args.sid else list_sessions(args.root)

    for sid in sids:
        df = read_trials(args.root, sid)
        df = clean_trials_for_saccade(df)
        out_dir = os.path.join(args.out_root, sid)
        os.makedirs(out_dir, exist_ok=True)
        out_pq = os.path.join(out_dir, "trials_enriched.parquet")
        try:
            df.to_parquet(out_pq, engine="pyarrow")
            print(f"[{sid}] wrote {out_pq} (n={len(df)})")
        except Exception:
            # fallback CSV
            out_csv = os.path.join(out_dir, "trials_enriched.csv")
            df.to_csv(out_csv, index=False)
            print(f"[{sid}] wrote {out_csv} (n={len(df)})")

if __name__ == "__main__":
    main()

