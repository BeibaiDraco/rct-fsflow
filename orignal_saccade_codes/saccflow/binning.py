from typing import Dict, Tuple, List
import numpy as np

def time_edges(t0: float, t1: float, bin_s: float) -> np.ndarray:
    # include rightmost edge
    n = int(np.round((t1 - t0) / bin_s))
    edges = t0 + np.arange(n + 1) * bin_s
    if edges[-1] < t1 - 1e-12:
        edges = np.append(edges, t1)
    return edges

def bin_unit_relative(spike_times: np.ndarray, event_times: np.ndarray,
                      t0: float, t1: float, edges: np.ndarray) -> np.ndarray:
    """
    Returns counts with shape (n_trials, n_bins) for one unit.
    """
    n_trials = event_times.shape[0]
    n_bins = edges.shape[0] - 1
    out = np.zeros((n_trials, n_bins), dtype=np.float32)

    # simple, robust loop (fast enough at our sizes)
    for i, e in enumerate(event_times):
        w0, w1 = e + t0, e + t1
        mask = (spike_times >= w0) & (spike_times < w1)
        if not np.any(mask):
            continue
        rel = spike_times[mask] - e
        out[i, :] = np.histogram(rel, bins=edges)[0]
    return out

def bin_area(units_spikes: List[np.ndarray], event_times: np.ndarray,
             t0: float, t1: float, bin_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X: (trials, bins, units) counts
      t: (bins,) bin centers in seconds
    """
    edges = time_edges(t0, t1, bin_s)
    centers = (edges[:-1] + edges[1:]) / 2.0
    trial_counts = []
    for ts in units_spikes:
        trial_counts.append(bin_unit_relative(ts, event_times, t0, t1, edges))
    # stack along units axis
    X = np.stack(trial_counts, axis=-1).astype(np.float32)  # (trials, bins, units)
    return X, centers
