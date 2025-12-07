#!/usr/bin/env python3
import argparse, os
from pathlib import Path
from collections import defaultdict

def find_sessions(out_root: Path, align: str):
    base = out_root / align
    if not base.exists(): return []
    return sorted([p.name for p in base.iterdir() if p.is_dir() and (p/"caches").is_dir()])

def have_any_flows(out_root: Path, align: str, sid: str, tag: str, feature: str):
    d = out_root / align / sid / "flow" / tag / feature
    return d.exists() and any(str(x).endswith(".npz") for x in d.glob("flow_*.npz"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out")
    ap.add_argument("--tag", default="crnull")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    report = defaultdict(list)

    for align, feats in (("stim", ("C","R")), ("sacc", ("S",))):
        sids = find_sessions(out_root, align)
        for sid in sids:
            for f in feats:
                ok = have_any_flows(out_root, align, sid, args.tag, f)
                key = f"{align}:{f}"
                report[key].append((sid, ok))

    print("[check] flow existence by align/feature")
    for key in sorted(report):
        rows = report[key]
        have = sum(ok for _,ok in rows)
        total = len(rows)
        print(f"  {key}: {have}/{total} sessions complete")
        missing = [sid for sid,ok in rows if not ok]
        if missing:
            print("    missing:", ", ".join(missing))

if __name__ == "__main__":
    main()
