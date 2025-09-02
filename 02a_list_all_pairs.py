#!/usr/bin/env python3
import json, argparse
from pathlib import Path

M_SET = {"MFEF","MLIP","MSC"}
S_SET = {"SFEF","SLIP","SSC"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Path to RCT/ (has manifest.json)")
    ap.add_argument("--out", type=Path, default=Path("results/worklists/all_pairs.json"))
    args = ap.parse_args()

    man = json.load(open(args.root/"manifest.json","r"))
    work = []
    for s in man["sessions"]:
        sid = int(s["session_id"])
        areas = set(s.get("areas", []))
        m_present = sorted(areas & M_SET)
        s_present = sorted(areas & S_SET)

        entries = []

        # Monkey M bucket
        if len(m_present) >= 2:
            pairs = [f"{a}->{b}" for a in m_present for b in m_present if a != b]
            entries.append({"monkey":"M", "areas_present": m_present, "pairs": pairs})

        # Monkey S bucket
        if len(s_present) >= 2:
            pairs = [f"{a}->{b}" for a in s_present for b in s_present if a != b]
            entries.append({"monkey":"S", "areas_present": s_present, "pairs": pairs})

        if entries:
            work.append({"session": sid, "buckets": entries})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"work": work}, open(args.out,"w"), indent=2)
    print(f"[ok] wrote {args.out} with {len(work)} eligible sessions")

if __name__ == "__main__":
    main()