#!/usr/bin/env python3
"""
Analyze how many incorrect trials were included in the analysis
due to the is_correct filtering bug, given the PT > 200ms filter.

This helps quantify the actual impact of the bug.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Setup
PAPER_DATA = os.environ.get("PAPER_DATA", "/Users/dracoxu/Documents/Research/rct-fsflow/paper_project_final/RCT_02")
PT_MIN = 200.0  # ms

def analyze_session(root: str, sid: str) -> dict:
    """Analyze one session for trial contamination."""
    trials_path = Path(root) / sid / "trials.parquet"
    if not trials_path.exists():
        return None
    
    df = pd.read_parquet(trials_path)
    
    # Check if is_correct column exists
    if "is_correct" not in df.columns:
        return {"sid": sid, "error": "no is_correct column"}
    
    # Check if PT_ms column exists
    if "PT_ms" not in df.columns:
        return {"sid": sid, "error": "no PT_ms column"}
    
    # Get trials with valid PT
    has_pt = df["PT_ms"].notna()
    
    # PT > 200ms filter
    pt_above_200 = has_pt & (df["PT_ms"] >= PT_MIN)
    
    # Correct trials
    is_correct = df["is_correct"].fillna(False).astype(bool)
    
    # Counts
    n_total = len(df)
    n_has_pt = has_pt.sum()
    n_pt_above_200 = pt_above_200.sum()
    n_correct = is_correct.sum()
    n_incorrect = (~is_correct).sum()
    
    # Among PT > 200ms trials
    n_pt200_correct = (pt_above_200 & is_correct).sum()
    n_pt200_incorrect = (pt_above_200 & ~is_correct).sum()
    
    # The "contamination" - incorrect trials that were wrongly included
    # when we should have filtered for correct only
    pct_contamination = 100.0 * n_pt200_incorrect / n_pt_above_200 if n_pt_above_200 > 0 else 0.0
    
    return {
        "sid": sid,
        "n_total": int(n_total),
        "n_has_pt": int(n_has_pt),
        "n_pt_above_200": int(n_pt_above_200),
        "n_correct_total": int(n_correct),
        "n_incorrect_total": int(n_incorrect),
        "n_pt200_correct": int(n_pt200_correct),
        "n_pt200_incorrect": int(n_pt200_incorrect),
        "pct_contamination": float(pct_contamination),
        "pct_correct_overall": float(100.0 * n_correct / n_total) if n_total > 0 else 0.0,
        "pct_correct_pt200": float(100.0 * n_pt200_correct / n_pt_above_200) if n_pt_above_200 > 0 else 0.0,
    }


def main():
    root = PAPER_DATA
    
    # Find all sessions
    sessions = sorted([p.name for p in Path(root).iterdir() 
                       if p.is_dir() and (p / "trials.parquet").exists()])
    
    print(f"Analyzing {len(sessions)} sessions...")
    print(f"PT threshold: {PT_MIN} ms")
    print()
    
    results = []
    for sid in sessions:
        r = analyze_session(root, sid)
        if r and "error" not in r:
            results.append(r)
    
    if not results:
        print("No valid sessions found!")
        return
    
    # Aggregate statistics
    total_pt200 = sum(r["n_pt_above_200"] for r in results)
    total_pt200_correct = sum(r["n_pt200_correct"] for r in results)
    total_pt200_incorrect = sum(r["n_pt200_incorrect"] for r in results)
    
    print("=" * 80)
    print("SUMMARY: Trial Contamination Analysis")
    print("=" * 80)
    print()
    print(f"Sessions analyzed: {len(results)}")
    print()
    print(f"Total trials with PT >= {PT_MIN}ms: {total_pt200:,}")
    print(f"  - Correct trials:   {total_pt200_correct:,} ({100*total_pt200_correct/total_pt200:.1f}%)")
    print(f"  - Incorrect trials: {total_pt200_incorrect:,} ({100*total_pt200_incorrect/total_pt200:.1f}%) <- CONTAMINATION")
    print()
    print("=" * 80)
    print("Per-session breakdown:")
    print("=" * 80)
    print()
    print(f"{'Session':<12} {'PT>200':<10} {'Correct':<10} {'Incorrect':<12} {'%Contam':<10} {'%Correct':<10}")
    print("-" * 70)
    
    for r in sorted(results, key=lambda x: x["sid"]):
        print(f"{r['sid']:<12} {r['n_pt_above_200']:<10} {r['n_pt200_correct']:<10} "
              f"{r['n_pt200_incorrect']:<12} {r['pct_contamination']:<10.1f} {r['pct_correct_pt200']:<10.1f}")
    
    print()
    print("=" * 80)
    print("By Monkey:")
    print("=" * 80)
    
    # Monkey M (2020) vs Monkey S (2023)
    for monkey, prefix in [("M", "2020"), ("S", "2023")]:
        monkey_results = [r for r in results if r["sid"].startswith(prefix)]
        if not monkey_results:
            continue
        
        m_pt200 = sum(r["n_pt_above_200"] for r in monkey_results)
        m_pt200_correct = sum(r["n_pt200_correct"] for r in monkey_results)
        m_pt200_incorrect = sum(r["n_pt200_incorrect"] for r in monkey_results)
        
        print(f"\nMonkey {monkey} ({len(monkey_results)} sessions):")
        print(f"  Total trials with PT >= {PT_MIN}ms: {m_pt200:,}")
        print(f"  - Correct:   {m_pt200_correct:,} ({100*m_pt200_correct/m_pt200:.1f}%)")
        print(f"  - Incorrect: {m_pt200_incorrect:,} ({100*m_pt200_incorrect/m_pt200:.1f}%) <- contamination")
    
    print()
    print("=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    overall_contam = 100 * total_pt200_incorrect / total_pt200
    print(f"\nWith PT >= {PT_MIN}ms filter, {overall_contam:.1f}% of included trials were incorrect.")
    print(f"This means the previous analysis had ~{overall_contam:.0f}% contamination from incorrect trials.")
    print()


if __name__ == "__main__":
    main()
