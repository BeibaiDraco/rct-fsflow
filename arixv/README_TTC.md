# Trial-wise Decodability Threshold Crossing (TTC) Analysis

This add-on computes **trial-level** latencies at which *each area*'s
decoder becomes confident enough about **category** (direction-invariant)
or **within-category direction**, then compares lead/lag across areas.

Files
-----
- `11_trialwise_ttc.py` — per-session TTC computation using your `axes_<AREA>.npz`.
- `12_ttc_stats.py` — aggregates TTC NPZ outputs and runs lead/lag stats.

Usage
-----
python 11_trialwise_ttc.py --sid 20200217 --areas MFEF MLIP MSC \
  --root results/session --alpha 0.01 --smoothing_bins 3 --min_consecutive 2 \
  --tmin 0.00 --tmax 0.50 --exclude_test_direction --tag ttc_v1

python 12_ttc_stats.py --files results/session/*/ttc_ttc_v1.npz \
  --out_csv ttc_summary.csv --out_json ttc_summary.json \
  --fef MFEF --lip MLIP --sc MSC

Notes
-----
- Category decoding uses direction-exclusion to enforce invariance.
- Direction decoding is trained within-category (LOOCV).
- Thresholds calibrated per area from pre-stim baseline (FPR alpha).
- Increase --min_consecutive / --smoothing_bins to suppress jitter.
