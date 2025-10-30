# saccflow/plotting.py
import numpy as np, matplotlib.pyplot as plt
from typing import Optional

def _smooth(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1: return y
    w = np.ones(int(k), dtype=float) / int(k)
    return np.convolve(y, w, mode="same")

def overlay_plot(
    time_s: np.ndarray,
    yAB: np.ndarray, pAB: Optional[np.ndarray],
    yBA: np.ndarray, pBA: Optional[np.ndarray],
    nullAB_mean: Optional[np.ndarray] = None, nullAB_std: Optional[np.ndarray] = None,
    nullBA_mean: Optional[np.ndarray] = None, nullBA_std: Optional[np.ndarray] = None,
    smooth_bins: int = 3, out_path: str = "", title: str = "", shade_null: bool = False
):
    tms = time_s * 1000.0
    yABs = _smooth(yAB, smooth_bins)
    yBAs = _smooth(yBA, smooth_bins)

    plt.figure(figsize=(7.2,3.2))
    plt.axhline(0, ls="--", c="k", lw=0.8)
    plt.axvline(0, ls="--", c="k", lw=0.8)

    # significance ribbons (p<0.05)
    if pAB is not None:
        m = (pAB < 0.05)
        if m.any(): plt.fill_between(tms, 0, 0.02, where=m, color="C0", alpha=0.18, step="mid")
    if pBA is not None:
        m = (pBA < 0.05)
        if m.any(): plt.fill_between(tms, 0.02, 0.04, where=m, color="C1", alpha=0.18, step="mid")

    # main curves
    plt.plot(tms, yABs, label="A→B", lw=2, color="C0")
    plt.plot(tms, yBAs, label="B→A", lw=2, color="C1")

    # null means (+/-1σ optional)
    if nullAB_mean is not None:
        plt.plot(tms, _smooth(nullAB_mean, smooth_bins), lw=1.4, ls="--", color="C0", alpha=0.9, label="null μ (A→B)")
        if shade_null and nullAB_std is not None:
            mu = _smooth(nullAB_mean, smooth_bins); sd = _smooth(nullAB_std, smooth_bins)
            plt.fill_between(tms, mu-sd, mu+sd, color="C0", alpha=0.12, linewidth=0)
    if nullBA_mean is not None:
        plt.plot(tms, _smooth(nullBA_mean, smooth_bins), lw=1.4, ls="--", color="C1", alpha=0.9, label="null μ (B→A)")
        if shade_null and nullBA_std is not None:
            mu = _smooth(nullBA_mean, smooth_bins); sd = _smooth(nullBA_std, smooth_bins)
            plt.fill_between(tms, mu-sd, mu+sd, color="C1", alpha=0.12, linewidth=0)

    plt.xlabel("Time from Saccade Onset (ms)")
    plt.ylabel("ΔLL (bits)")
    plt.xlim(tms[0], tms[-1])
    plt.legend(loc="upper left", ncol=2)
    if title: plt.title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
        plt.close()
