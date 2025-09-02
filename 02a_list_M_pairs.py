#!/usr/bin/env python3
import json, argparse
from pathlib import Path

M_AREAS = ["MFEF","MLIP","MSC"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Path to RCT/ (has manifest.json)")
    ap.add_argument("--out", type=Path, default=Path("results/worklists/m_pairs.json"))
    args = ap.parse_args()

    man = json.load(open(args.root/"manifest.json","r"))
    work = []
    for s in man["sessions"]:
        sid = int(s["session_id"])
        present = [a for a in s.get("areas",[]) if a in M_AREAS]
        if len(present) < 2:
            continue  # skip sessions with <2 M areas (no pair)
        # all directed pairs among present
        pairs = []
        for i in range(len(present)):
            for j in range(len(present)):
                if i==j: continue
                pairs.append(f"{present[i]}->{present[j]}")
        work.append({"session": sid, "areas_present": present, "pairs": pairs})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"work": work}, open(args.out,"w"), indent=2)
    print(f"[ok] wrote {args.out} with {len(work)} eligible sessions")

if __name__ == "__main__":
    main()