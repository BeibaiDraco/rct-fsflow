#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
17_make_session_overlays_induced.py

Create per-session bidirectional overlays (A↔B) inside each induced tag folder.
- No null bands; plot only the null mean lines (thin).
- Annotate per-time p-values next to computed curves (downsampled).
- Works for RAW and/or INT, and for C and/or R.

Example:
  python 17_make_session_overlays_induced.py \
      --tag induced_k5_win016_p500 --mode both --labels C R \
      --annotate_p --p_stride 6
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
SESS_ROOT = BASE / "results" / "session"

def p_from_null(obs: np.ndarray, null_mat: np.ndarray) -> np.ndarray:
    p = np.full_like(obs, np.nan, float)
    if null_mat is None or null_mat.size == 0:
        return p
    ge = (null_mat >= obs[None,:]).sum(axis=0)
    return (1 + ge) / (1 + null_mat.shape[0])

def load_induced_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    rec = {
        "tC": z["tC"] if "tC" in z.files else None,
        "tR": z["tR"] if "tR" in z.files else None,
        # RAW forward + null
        "C_raw": z.get("C_fwd", None),        "R_raw": z.get("R_fwd", None),
        "C_fnull": z.get("C_fnull", None),    "R_fnull": z.get("R_fnull", None),
        # INT (sliding integrated) + null mean
        "C_int": z.get("C_fwd_sl", None),     "R_int": z.get("R_fwd_sl", None),
        "C_null_sl": z.get("C_null_sl", None),"R_null_sl": z.get("R_null_sl", None),
        "C_sl_mu": z.get("C_sl_mu", None),    "R_sl_mu": z.get("R_sl_mu", None),
        # Per-time p’s (name varies: C_p_t / R_p_t or C_p_int / R_p_int)
        "C_p_int": z.get("C_p_int", z.get("C_p_t", None)),
        "R_p_int": z.get("R_p_int", z.get("R_p_t", None)),
        # RAW p’s generally not saved by 12_*; we’ll compute if needed
        "C_p_raw": z.get("C_p_raw", None),
        "R_p_raw": z.get("R_p_raw", None),
    }
    return rec

def overlay_plot(x, y_ab, y_ba, mu_ab, mu_ba, p_ab, p_ba,
                 label: str, mode: str, sid: int, A: str, B: str,
                 out_png: Path, annotate_p: bool, p_stride: int):
    fig, ax = plt.subplots(figsize=(10,4))
    # A->B
    ax.plot(x, y_ab, lw=2, label=f"{A}→{B} ({label})", alpha=0.95)
    if mu_ab is not None and mu_ab.size == x.size:
        ax.plot(x, mu_ab, lw=1, ls=":", alpha=0.7, label=f"null μ {A}→{B}")
    # B->A
    ax.plot(x, y_ba, lw=2, ls="--", label=f"{B}→{A} ({label})", alpha=0.95)
    if mu_ba is not None and mu_ba.size == x.size:
        ax.plot(x, mu_ba, lw=1, ls=":", alpha=0.7, label=f"null μ {B}→{A}")
    # annotate p
    if annotate_p:
        y_all = np.concatenate([y_ab[np.isfinite(y_ab)], y_ba[np.isfinite(y_ba)]]) if np.any(np.isfinite(y_ab)) or np.any(np.isfinite(y_ba)) else np.array([0.0])
        yspan = float(np.nanmax(y_all) - np.nanmin(y_all)) if y_all.size else 1.0
        dy = 0.05 * (yspan if yspan>0 else 1.0)
        for i,xx in enumerate(x):
            if i % max(1, p_stride): continue
            if i < len(y_ab) and np.isfinite(y_ab[i]) and p_ab is not None and i < len(p_ab) and np.isfinite(p_ab[i]):
                ax.text(xx, y_ab[i] + dy, f"{p_ab[i]:.2f}", fontsize=7, ha="center", va="bottom")
            if i < len(y_ba) and np.isfinite(y_ba[i]) and p_ba is not None and i < len(p_ba) and np.isfinite(p_ba[i]):
                ax.text(xx, y_ba[i] - dy, f"{p_ba[i]:.2f}", fontsize=7, ha="center", va="top")
    ax.axvline(0.0, color="k", ls=":", lw=1)
    ax.set_xlabel("time (s) from cat_stim_on")
    ax.set_ylabel(("Integrated " if mode=="int" else "") + f"GC ({label})")
    ax.legend(frameon=False)
    ax.set_title(f"Session {sid} — {A}↔{B} {label} ({mode})")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="induced tag folder (e.g., induced_k5_win016_p500)")
    ap.add_argument("--sid", type=int, default=None, help="optional: process only this session id")
    ap.add_argument("--mode", choices=["raw","int","both"], default="both")
    ap.add_argument("--labels", nargs="+", choices=["C","R"], default=["C","R"])
    ap.add_argument("--annotate_p", action="store_true", default=False)
    ap.add_argument("--p_stride", type=int, default=6)
    args = ap.parse_args()

    for sdir in sorted(SESS_ROOT.glob("*")):
        if not sdir.is_dir() or not sdir.name.isdigit():
            continue
        sid = int(sdir.name)
        if args.sid is not None and sid != args.sid:
            continue
        tag_dir = sdir / args.tag
        if not tag_dir.exists():
            continue

        files = list(tag_dir.glob("induced_flow_*to*.npz"))
        pairs = {}
        for f in files:
            name = f.stem[len("induced_flow_"):]
            if "to" not in name: continue
            A,B = name.split("to")
            pairs.setdefault(tuple(sorted([A,B])), {})[f"{A}->{B}"] = f

        for (A,B), d in sorted(pairs.items()):
            if f"{A}->{B}" not in d or f"{B}->{A}" not in d:
                continue
            z_ab = load_induced_npz(d[f"{A}->{B}"])
            z_ba = load_induced_npz(d[f"{B}->{A}"])

            # CATEGORY
            if "C" in args.labels:
                x = z_ab["tC"] if z_ab["tC"] is not None else z_ba["tC"]
                if x is not None and x.size > 1:
                    if args.mode in ("raw","both") and z_ab["C_raw"] is not None and z_ba["C_raw"] is not None:
                        y_ab = np.asarray(z_ab["C_raw"], float)
                        y_ba = np.asarray(z_ba["C_raw"], float)
                        mu_ab = np.nanmean(z_ab["C_fnull"], axis=0) if z_ab["C_fnull"] is not None and z_ab["C_fnull"].size else None
                        mu_ba = np.nanmean(z_ba["C_fnull"], axis=0) if z_ba["C_fnull"] is not None and z_ba["C_fnull"].size else None
                        p_ab = z_ab["C_p_raw"] if z_ab["C_p_raw"] is not None else (p_from_null(y_ab, z_ab["C_fnull"]) if z_ab["C_fnull"] is not None else None)
                        p_ba = z_ba["C_p_raw"] if z_ba["C_p_raw"] is not None else (p_from_null(y_ba, z_ba["C_fnull"]) if z_ba["C_fnull"] is not None else None)
                        out_png = tag_dir / f"overlay_C_{A}_{B}_raw.png"
                        overlay_plot(x, y_ab, y_ba, mu_ab, mu_ba, p_ab, p_ba, "C", "raw", sid, A, B, out_png, args.annotate_p, args.p_stride)

                    if args.mode in ("int","both") and z_ab["C_int"] is not None and z_ba["C_int"] is not None:
                        y_ab = np.asarray(z_ab["C_int"], float)
                        y_ba = np.asarray(z_ba["C_int"], float)
                        mu_ab = np.asarray(z_ab["C_sl_mu"], float) if z_ab["C_sl_mu"] is not None else (np.nanmean(z_ab["C_null_sl"], axis=0) if z_ab["C_null_sl"] is not None and z_ab["C_null_sl"].size else None)
                        mu_ba = np.asarray(z_ba["C_sl_mu"], float) if z_ba["C_sl_mu"] is not None else (np.nanmean(z_ba["C_null_sl"], axis=0) if z_ba["C_null_sl"] is not None and z_ba["C_null_sl"].size else None)
                        p_ab = z_ab["C_p_int"] if z_ab["C_p_int"] is not None else (p_from_null(y_ab, z_ab["C_null_sl"]) if z_ab["C_null_sl"] is not None else None)
                        p_ba = z_ba["C_p_int"] if z_ba["C_p_int"] is not None else (p_from_null(y_ba, z_ba["C_null_sl"]) if z_ba["C_null_sl"] is not None else None)
                        out_png = tag_dir / f"overlay_C_{A}_{B}_int.png"
                        overlay_plot(x, y_ab, y_ba, mu_ab, mu_ba, p_ab, p_ba, "C", "int", sid, A, B, out_png, args.annotate_p, args.p_stride)

            # DIRECTION
            if "R" in args.labels:
                x = z_ab["tR"] if z_ab["tR"] is not None else z_ba["tR"]
                if x is not None and x.size > 1:
                    if args.mode in ("raw","both") and z_ab["R_raw"] is not None and z_ba["R_raw"] is not None:
                        y_ab = np.asarray(z_ab["R_raw"], float)
                        y_ba = np.asarray(z_ba["R_raw"], float)
                        mu_ab = np.nanmean(z_ab["R_fnull"], axis=0) if z_ab["R_fnull"] is not None and z_ab["R_fnull"].size else None
                        mu_ba = np.nanmean(z_ba["R_fnull"], axis=0) if z_ba["R_fnull"] is not None and z_ba["R_fnull"].size else None
                        p_ab = z_ab["R_p_raw"] if z_ab["R_p_raw"] is not None else (p_from_null(y_ab, z_ab["R_fnull"]) if z_ab["R_fnull"] is not None else None)
                        p_ba = z_ba["R_p_raw"] if z_ba["R_p_raw"] is not None else (p_from_null(y_ba, z_ba["R_fnull"]) if z_ba["R_fnull"] is not None else None)
                        out_png = tag_dir / f"overlay_R_{A}_{B}_raw.png"
                        overlay_plot(x, y_ab, y_ba, mu_ab, mu_ba, p_ab, p_ba, "R", "raw", sid, A, B, out_png, args.annotate_p, args.p_stride)

                    if args.mode in ("int","both") and z_ab["R_int"] is not None and z_ba["R_int"] is not None:
                        y_ab = np.asarray(z_ab["R_int"], float)
                        y_ba = np.asarray(z_ba["R_int"], float)
                        mu_ab = np.asarray(z_ab["R_sl_mu"], float) if z_ab["R_sl_mu"] is not None else (np.nanmean(z_ab["R_null_sl"], axis=0) if z_ab["R_null_sl"] is not None and z_ab["R_null_sl"].size else None)
                        mu_ba = np.asarray(z_ba["R_sl_mu"], float) if z_ba["R_sl_mu"] is not None else (np.nanmean(z_ba["R_null_sl"], axis=0) if z_ba["R_null_sl"] is not None and z_ba["R_null_sl"].size else None)
                        p_ab = z_ab["R_p_int"] if z_ab["R_p_int"] is not None else (p_from_null(y_ab, z_ab["R_null_sl"]) if z_ab["R_null_sl"] is not None else None)
                        p_ba = z_ba["R_p_int"] if z_ba["R_p_int"] is not None else (p_from_null(y_ba, z_ba["R_null_sl"]) if z_ba["R_null_sl"] is not None else None)
                        out_png = tag_dir / f"overlay_R_{A}_{B}_int.png"
                        overlay_plot(x, y_ab, y_ba, mu_ab, mu_ba, p_ab, p_ba, "R", "int", sid, A, B, out_png, args.annotate_p, args.p_stride)

        print(f"[done] overlays in {tag_dir}")

if __name__ == "__main__":
    main()
