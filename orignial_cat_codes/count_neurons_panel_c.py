#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count neurons used for Panel C (single session overlay) analysis.

This script examines the axes files for a given session to determine how many
neurons were used from each brain region (FEF/LIP) in the analysis.

Usage:
    python count_neurons_panel_c.py --sid 20200327 --session-tag induced_k5_win016_p500
    python count_neurons_panel_c.py --repo /path/to/repo --sid 20201001 --session-tag induced_k5_win016_p500
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

def load_axes_info(axes_path: Path):
    """Load axes file and extract neuron count information."""
    if not axes_path.exists():
        return None
    
    z = np.load(axes_path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    
    # Get projection matrices to determine original neuron count
    W_C = z.get("W_C", None)  # Shape: (n_neurons, C_dim)
    W_R = z.get("W_R", None)  # Shape: (n_neurons, R_dim)
    
    # The number of neurons is the first dimension of the projection matrices
    n_neurons = None
    if W_C is not None:
        n_neurons = W_C.shape[0]
    elif W_R is not None:
        n_neurons = W_R.shape[0]
    
    # Get subspace dimensions
    C_dim = meta.get("C_dim", 0)
    R_dim = meta.get("R_dim", 0)
    
    return {
        "n_neurons": n_neurons,
        "C_dim": C_dim,
        "R_dim": R_dim,
        "meta": meta
    }

def main():
    ap = argparse.ArgumentParser(description="Count neurons used in Panel C analysis")
    ap.add_argument("--repo", default="/project/bdoiron/dracoxu/rct-fsflow", type=str,
                    help="Path to repository root")
    ap.add_argument("--sid", required=True, type=str,
                    help="Session ID (e.g., 20200327)")
    ap.add_argument("--session-tag", default="induced_k5_win016_p500", type=str,
                    help="Session tag for analysis")
    
    args = ap.parse_args()
    
    repo = Path(args.repo)
    session_dir = repo / "results" / "session" / args.sid / args.session_tag
    
    # Areas to check (corresponding to FEF and LIP)
    areas = ["MFEF", "MLIP"]  # Panel C uses MFEF->MLIP flow
    
    print(f"Checking neuron counts for session {args.sid} with tag '{args.session_tag}'")
    print(f"Session directory: {session_dir}")
    print("=" * 60)
    
    total_neurons = 0
    area_info = {}
    
    for area in areas:
        axes_path = session_dir / f"axes_{area}.npz"
        info = load_axes_info(axes_path)
        
        if info is None:
            print(f"{area:>6}: axes file not found at {axes_path}")
            continue
        
        area_info[area] = info
        n_neurons = info["n_neurons"]
        
        if n_neurons is not None:
            total_neurons += n_neurons
            region_name = "FEF" if "FEF" in area else "LIP"
            print(f"{region_name:>6}: {n_neurons:>3} neurons (C_dim={info['C_dim']}, R_dim={info['R_dim']})")
        else:
            print(f"{area:>6}: Could not determine neuron count")
    
    print("-" * 60)
    print(f"{'Total':>6}: {total_neurons:>3} neurons across both regions")
    
    # Check if the induced flow file exists for this session
    flow_file = session_dir / "induced_flow_MFEFtoMLIP.npz"
    if flow_file.exists():
        print(f"\n✓ Flow analysis file exists: {flow_file.name}")
        
        # Load and check the flow file structure
        z = np.load(flow_file, allow_pickle=True)
        print(f"  Flow file contains: {list(z.files)}")
        
        # Check if we have the data used in Panel C
        if "tC" in z.files and "C_fwd" in z.files and "C_rev" in z.files:
            tC = z["tC"]
            C_fwd = z["C_fwd"]
            C_rev = z["C_rev"]
            print(f"  Time points: {len(tC)}")
            print(f"  Forward flow shape: {C_fwd.shape}")
            print(f"  Reverse flow shape: {C_rev.shape}")
        
    else:
        print(f"\n✗ Flow analysis file not found: {flow_file}")
    
    # Summary for Panel C specifically
    print("\n" + "=" * 60)
    print("PANEL C SUMMARY:")
    print(f"Session: {args.sid}")
    print(f"Analysis: FEF ↔ LIP information flow")
    
    if "MFEF" in area_info and "MLIP" in area_info:
        fef_neurons = area_info["MFEF"]["n_neurons"]
        lip_neurons = area_info["MLIP"]["n_neurons"]
        if fef_neurons and lip_neurons:
            print(f"FEF neurons: {fef_neurons}")
            print(f"LIP neurons: {lip_neurons}")
            print(f"Total neurons: {fef_neurons + lip_neurons}")
        else:
            print("Could not determine neuron counts for both regions")
    else:
        print("Missing data for one or both regions")

if __name__ == "__main__":
    main()


