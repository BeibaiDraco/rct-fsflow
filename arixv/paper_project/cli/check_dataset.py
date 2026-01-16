#!/usr/bin/env python
from __future__ import annotations
import argparse, os, re
import numpy as np
from typing import Dict, List
from paperflow.io import list_sessions, list_areas, read_trials, has_col, safe_pct, write_json

def scan_session(root: str, sid: str) -> Dict:
    areas = list_areas(root, sid)
    ok2 = len(areas) >= 2

    # trials inspection
    try:
        df = read_trials(root, sid)
    except Exception as e:
        return dict(session=sid, areas=areas, n_areas=len(areas), ok2=ok2, error=str(e))

    cols = {
        "category": has_col(df,"category"),
        "direction": has_col(df,"direction"),
        "chosen_cat": has_col(df,"chosen_cat"),
        "targets_vert": has_col(df,"targets_vert"),
        "saccade_location_sign": has_col(df,"saccade_location_sign"),
        "Align_to_cat_stim_on": has_col(df,"Align_to_cat_stim_on"),
        "Align_to_sacc_on": has_col(df,"Align_to_sacc_on"),
        "PT_ms": has_col(df,"PT_ms"),
        "trial_error": has_col(df,"trial_error"),
    }
    n = len(df)
    pct = {
        "pct_has_cat_stim": safe_pct(df["Align_to_cat_stim_on"]) if cols["Align_to_cat_stim_on"] else 0.0,
        "pct_has_sacc_on":  safe_pct(df["Align_to_sacc_on"])     if cols["Align_to_sacc_on"] else 0.0,
        "pct_has_sacc_sign":safe_pct(df["saccade_location_sign"])if cols["saccade_location_sign"] else 0.0,
        "pct_targets_vert": safe_pct(df["targets_vert"])          if cols["targets_vert"] else 0.0,
    }
    # simple correct-trials %
    if cols["trial_error"]:
        pct["pct_correct"] = float((df["trial_error"].fillna(0)==0).mean()*100.0)
    else:
        pct["pct_correct"] = np.nan

    return dict(session=sid, n_trials=int(n), areas=areas, n_areas=len(areas), ok2=ok2,
                cols=cols, pct=pct)

def main():
    ap = argparse.ArgumentParser(description="Scan RCT export and build a worklist for S/C/R analyses.")
    ap.add_argument("--root", default=os.environ.get("PAPER_DATA", ""), help="Path to RCT_02")
    ap.add_argument("--out",  default=os.path.join(os.environ.get("PAPER_HOME","."),"out","dataset_scan.json"))
    args = ap.parse_args()
    if not args.root:
        raise SystemExit("No --root provided and $PAPER_DATA not set.")

    sids = list_sessions(args.root)
    rows: List[Dict] = []
    for sid in sids:
        rows.append(scan_session(args.root, sid))

    # Build feature-specific worklists
    work_s = [r["session"] for r in rows if r["ok2"]
              and r["cols"].get("Align_to_sacc_on", False)
              and r["cols"].get("saccade_location_sign", False)]
    work_c = [r["session"] for r in rows if r["ok2"]
              and r["cols"].get("Align_to_cat_stim_on", False)
              and r["cols"].get("category", False)]
    work_r = [r["session"] for r in rows if r["ok2"]
              and r["cols"].get("Align_to_cat_stim_on", False)
              and r["cols"].get("direction", False)]

    summary = dict(
        root=args.root,
        n_sessions=len(sids),
        n_ok2=sum(1 for r in rows if r["ok2"]),
        worklists=dict(
            S=work_s,   # saccade: needs sacc_on + sacc_sign
            C=work_c,   # category: needs cat_stim_on + category
            R=work_r    # direction: needs cat_stim_on + direction
        ),
        sessions=rows
    )
    write_json(summary, args.out)
    print(f"[ok] wrote {args.out}")
    print(f"[info] eligible: S={len(work_s)}  C={len(work_c)}  R={len(work_r)}")

if __name__ == "__main__":
    main()
