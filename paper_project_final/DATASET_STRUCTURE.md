# Dataset Structure Documentation

This document describes the complete data organization for the RCT (Random Category Task) neural information flow analysis project.

---

## 1. Overview

### 1.1 Subjects
Two rhesus macaque monkeys:
- **Monkey M** — Areas: `MFEF`, `MLIP`, `MSC`
- **Monkey S** — Areas: `SFEF`, `SLIP`, `SSC`

### 1.2 Brain Areas
| Area Code | Full Name | Description |
|-----------|-----------|-------------|
| FEF | Frontal Eye Field | Prefrontal oculomotor area |
| LIP | Lateral Intraparietal Area | Parietal visual/decision area |
| SC | Superior Colliculus | Midbrain saccade generator |

### 1.3 Session Summary
- **Total sessions in dataset:** 39
- **Sessions with ≥2 areas (flow-eligible):** 23
- **Sessions per monkey:**
  - Monkey M: 11 sessions with ≥2 areas (4× FEF+LIP only, 7× FEF+LIP+SC)
  - Monkey S: 12 sessions with ≥2 areas (5× FEF+LIP only, 7× FEF+LIP+SC)

---

## 2. Raw Data Organization (`RCT_02/`)

```
RCT_02/
├── manifest.json                    # Session-to-area mapping
├── <session_id>/                    # e.g., 20201001
│   ├── trials.parquet               # Trial metadata
│   └── areas/
│       └── <AREA>/                  # e.g., MFEF, MLIP, MSC
│           ├── units.json           # Unit metadata
│           └── spikes/
│               └── unit_*.h5        # Spike times per unit
```

### 2.1 Session IDs (23 flow-eligible sessions)

**Monkey M (2020):**
```
20200327, 20200328, 20200401, 20200402  (FEF+LIP)
20200926, 20200929, 20201001, 20201004  (FEF+LIP+SC)
20201204, 20201211, 20201216             (FEF+LIP+SC)
```

**Monkey S (2023):**
```
20230622, 20230627, 20230705, 20230707, 20230710  (FEF+LIP)
20231025, 20231103, 20231109, 20231121            (FEF+LIP+SC)
20231123, 20231130, 20231205                      (FEF+LIP+SC)
```

### 2.2 `trials.parquet` — Trial Metadata

| Column | Type | Description |
|--------|------|-------------|
| `trial_index` | float64 | Trial number within session |
| `is_rct` | bool | True if random category task trial |
| `is_correct` | bool | True if correct response |
| `targets_vert` | int8 | Target configuration: 0=horizontal, 1=vertical |
| `direction` | int16 | Stimulus direction in degrees (15, 75, 135, 195, 255, 315) |
| `category` | float64 | Category label: -1 or +1 |
| `chosen_cat` | float64 | Monkey's chosen category |
| `saccade_location_sign` | float64 | Saccade direction: -1 or +1 |
| `Align_to_fix_on` | float64 | Fixation onset time (s) |
| `Align_to_cat_stim_on` | float64 | **Stimulus onset time** (s) |
| `Align_to_sacc_on` | float64 | **Saccade onset time** (s) |
| `Align_to_noise_on` | float64 | Noise stimulus onset (s) |
| `Align_to_targets_on` | float64 | **Target onset time** (s) |
| `PT_ms` | float64 | Processing time = saccade − stimulus (ms) |
| `block_number` | float64 | Block number |
| `session_id` | int64 | Session date identifier |

**Typical trial counts per session:** 1,100–1,900 total trials, ~65–85% are RCT trials with valid alignment times.

### 2.3 `units.json` — Unit Metadata

```json
[
  {
    "neuron_id": "20201001_FEF_0",
    "cluster_id": 0,
    "file": "spikes/unit_000.h5",
    "n_spikes": 66297
  },
  ...
]
```

| Field | Description |
|-------|-------------|
| `neuron_id` | Unique identifier: `<session>_<area>_<cluster>` |
| `cluster_id` | Cluster ID from spike sorting |
| `file` | Relative path to spike HDF5 file |
| `n_spikes` | Total spike count across session |

**Unit counts per area/session:** 12–139 units depending on session and area.

### 2.4 `spikes/unit_*.h5` — Spike Times

HDF5 file with single dataset:

| Dataset | Shape | Type | Description |
|---------|-------|------|-------------|
| `/t` | (1, N) | float32 | Spike times in seconds from session start |

---

## 3. Task Variables (Encoding Features)

| Code | Name | Values | Description |
|------|------|--------|-------------|
| **C** | Category | {-1, +1} | Binary category of visual stimulus |
| **R** | Direction | {15°, 75°, 135°, 195°, 255°, 315°} | 6-way stimulus motion direction |
| **S** | Saccade | {-1, +1} | Saccade direction (left/right) |
| **T** | Target config | {0, 1} | Horizontal (0) or vertical (1) target arrangement |
| **O** | Orientation | {"vertical", "horizontal"} | Same as T but categorical |

### 3.1 Category-Direction Relationship
- Category boundary bisects direction space
- 3 directions map to category -1, 3 to category +1
- Category and direction are correlated but separable

---

## 4. Temporal Alignments

| Alignment | Event | Typical Window | Primary Use |
|-----------|-------|----------------|-------------|
| `stim` | Stimulus onset | [-250, +800] ms | Category/direction encoding |
| `sacc` | Saccade onset | [-400, +200] ms | Saccade/motor signals |
| `targ` | Target onset | [-200, +500] ms | Target configuration encoding |

---

## 5. Pipeline Output Structure (`out/`)

```
out/
├── <align>/                           # stim, sacc, or targ
│   ├── <session_id>/
│   │   ├── caches/                    # Binned spike arrays
│   │   │   └── area_<AREA>.npz
│   │   ├── axes/<tag>/                # Trained encoding axes
│   │   │   ├── axes_<AREA>.npz
│   │   │   ├── axes_summary.json
│   │   │   └── config.json
│   │   ├── qc/<tag>/                  # Quality control curves
│   │   │   ├── qc_axes_<AREA>.json
│   │   │   ├── qc_axes_<AREA>.pdf
│   │   │   └── qc_axes_<AREA>.png
│   │   └── flow/<tag>/<feature>/      # Flow results
│   │       ├── flow_<FEAT>_<A>to<B>.npz
│   │       └── ...
│   └── summary/<tag>/<feature>/       # Cross-session aggregates
│       ├── summary_<A>_vs_<B>.npz
│       └── figs/
└── analysis_original/                 # Legacy analysis outputs
```

---

## 6. Intermediate File Formats

### 6.1 Cache Files (`caches/area_<AREA>.npz`)

Binned spike count arrays aligned to trial events.

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `X` | (N_trials, N_bins, N_units) | float32 | Raw spike counts |
| `Z` | (N_trials, N_bins, N_units) | float32 | Z-scored spike counts |
| `time` | (N_bins,) | float32 | Time bin centers (s) |
| `lab_C` | (N_trials,) | float64 | Category labels |
| `lab_R` | (N_trials,) | float64 | Direction labels |
| `lab_S` | (N_trials,) | float64 | Saccade labels |
| `lab_orientation` | (N_trials,) | object | Orientation labels ("vertical"/"horizontal") |
| `lab_PT_ms` | (N_trials,) | float64 | Processing times |
| `lab_is_correct` | (N_trials,) | bool | Correctness |
| `lab_trial_index` | (N_trials,) | int64 | Trial indices |
| `meta` | scalar | JSON string | Configuration metadata |

**Meta fields:**
```json
{
  "sid": "20201001",
  "area": "MFEF",
  "align_event": "stim",
  "window": [-0.25, 0.8],
  "bin_s": 0.01,
  "n_trials": 879,
  "n_units": 139,
  "stim_targets_vert_only": false,
  "correct_only": true
}
```

**Typical dimensions:**
- `N_trials`: 300–900 (after filtering for correctness)
- `N_bins`: 105 (for 10ms bins, [-250, +800] ms window)
- `N_units`: 12–139

### 6.2 Axes Files (`axes/<tag>/axes_<AREA>.npz`)

Trained encoding axis vectors (linear discriminant directions).

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `sC` | (N_units,) | float64 | Category axis (binary classifier weights) |
| `sR` | (N_units, 4) | float64 | Direction axes (4 OvR components for 6-way) |
| `sS_raw` | (N_units,) | float64 | Saccade axis (raw, before orthogonalization) |
| `sS_inv` | (N_units,) | float64 | Saccade axis (orthogonalized to sC) |
| `sT` | (N_units,) | float64 | Target configuration axis |
| `sO` | (N_units,) | float64 | Orientation axis (same as T) |
| `norm_mu` | (N_units,) | float64 | Normalization mean (if stored) |
| `norm_sd` | (N_units,) | float64 | Normalization std (if stored) |
| `meta` | scalar | JSON string | Training configuration |

**Meta fields (key examples):**
```json
{
  "n_trials": 299,
  "n_bins": 105,
  "n_units": 139,
  "orientation": "vertical",
  "pt_min_ms": 200.0,
  "feature_set": ["C", "R"],
  "winC": [0.1, 0.3],
  "winR": [0.05, 0.2],
  "winC_selected": [0.22, 0.27],
  "winC_peak_auc": 0.957,
  "clf_binary": "logreg",
  "C_grid": [0.1, 0.3, 1.0, 3.0, 10.0]
}
```

### 6.3 QC Files (`qc/<tag>/qc_axes_<AREA>.json`)

Time-resolved decoding performance curves.

```json
{
  "time": [/* N_bins float values */],
  "auc_C": [/* N_bins AUC values for category */],
  "auc_S_raw": [/* N_bins AUC values for saccade (raw) */],
  "auc_S_inv": [/* N_bins AUC values for saccade (invariant) */],
  "acc_R_macro": [/* N_bins accuracy values for direction */],
  "auc_T": [/* N_bins AUC values for target config */],
  "latencies_ms": {
    "C": 145.0,
    "S_raw": null,
    "S_inv": null,
    "T": null
  },
  "meta": {
    "align": "stim",
    "orientation": "vertical",
    "pt_min_ms": 200.0,
    "n_trials": 299,
    "n_bins": 105,
    "n_units": 139,
    "thr": 0.6,
    "k_bins": 3,
    "norm": "global",
    "baseline_win": null,
    "used_axes_mu_sd": false
  }
}
```

**Latency detection:** First `k_bins` consecutive bins exceeding `thr` (default: 0.6 AUC, k=3).

### 6.4 Flow Files (`flow/<tag>/<feature>/flow_<FEAT>_<A>to<B>.npz`)

Pairwise information flow results.

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `time` | (N_bins,) | float64 | Time bin centers |
| `bits_AtoB` | (N_bins,) | float64 | ΔLL in bits: A → B direction |
| `bits_BtoA` | (N_bins,) | float64 | ΔLL in bits: B → A direction |
| `null_mean_AtoB` | (N_bins,) | float64 | Null distribution mean (A→B) |
| `null_std_AtoB` | (N_bins,) | float64 | Null distribution std (A→B) |
| `p_AtoB` | (N_bins,) | float64 | P-values (A→B) |
| `null_mean_BtoA` | (N_bins,) | float64 | Null distribution mean (B→A) |
| `null_std_BtoA` | (N_bins,) | float64 | Null distribution std (B→A) |
| `p_BtoA` | (N_bins,) | float64 | P-values (B→A) |
| `meta` | scalar | JSON string | Flow computation parameters |
| `null_samps_AtoB` | (N_perms, N_bins) | float32 | Full null samples (if saved) |
| `null_samps_BtoA` | (N_perms, N_bins) | float32 | Full null samples (if saved) |

**Typical dimensions:**
- `N_bins`: 105
- `N_perms`: 500

---

## 7. Trial Filtering & Condition Balancing

### 7.1 Standard Filters
| Filter | Description | Typical Retention |
|--------|-------------|-------------------|
| `is_rct == True` | RCT task trials only | ~80–90% |
| `is_correct == True` | Correct responses only | ~70–85% of RCT |
| `PT_ms >= 200` | Minimum processing time | ~50–70% |
| `targets_vert == 1` | Vertical orientation only | ~50% |

### 7.2 Orientation Stratification
- **Vertical orientation** (`targets_vert == 1`): Category axis trained on vertical trials
- **Horizontal orientation** (`targets_vert == 0`): Separate training set
- **Pooled**: Both orientations combined (less common)

### 7.3 Condition Counts (Typical Session)
After filtering for correct trials with PT ≥ 200ms and vertical orientation:
- Category -1: ~150 trials
- Category +1: ~150 trials
- 6 directions: ~50 trials each
- Total: ~300 trials per area

---

## 8. Key Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bin_s` | 0.01 (10ms) | Temporal bin size |
| `window` | [-0.25, 0.8] | Analysis window (s) relative to alignment |
| `pt_min_ms` | 200.0 | Minimum processing time filter |
| `norm` | "global" | Normalization mode |
| `clf_binary` | "logreg" | Binary classifier (logreg, svm, lda) |
| `lags_ms` | 50.0 | Temporal lag for flow regression |
| `perms` | 500 | Permutation count for null distribution |
| `ridge` | 0.01 | Ridge regularization strength |
| `qc_threshold` | 0.6 | QC AUC threshold for filtering |

---

## 9. Sessions with Simultaneous Multi-Area Recordings

### 9.1 Sessions with All 3 Areas (FEF + LIP + SC)

**Monkey M (7 sessions):**
| Session | FEF units | LIP units | SC units | Trials |
|---------|-----------|-----------|----------|--------|
| 20200926 | 70 | 64 | 72 | 1235 |
| 20200929 | 45 | 43 | 42 | 1344 |
| 20201001 | 139 | 76 | 46 | 1315 |
| 20201004 | 80 | 74 | 74 | 1239 |
| 20201204 | 74 | 66 | 71 | 1186 |
| 20201211 | 79 | 82 | 77 | 1110 |
| 20201216 | 56 | 57 | 58 | 1178 |

**Monkey S (7 sessions):**
| Session | FEF units | LIP units | SC units | Trials |
|---------|-----------|-----------|----------|--------|
| 20231025 | 53 | 53 | 53 | 1453 |
| 20231103 | 41 | 43 | 43 | 1443 |
| 20231109 | 48 | 49 | 49 | 1470 |
| 20231121 | 39 | 40 | 40 | 1628 |
| 20231123 | 55 | 56 | 56 | 1900 |
| 20231130 | 53 | 54 | 54 | 1705 |
| 20231205 | 79 | 79 | 79 | 1585 |

### 9.2 Sessions with FEF + LIP Only (No SC)

**Monkey M (4 sessions):**
```
20200327, 20200328, 20200401, 20200402
```

**Monkey S (5 sessions):**
```
20230622, 20230627, 20230705, 20230707, 20230710
```

---

## 10. Data Quality Notes

1. **Missing alignment times:** Some trials lack `Align_to_sacc_on` (especially Monkey S sessions with ~20–50% missing saccade times).

2. **Direction encoding:** 6-way classification uses a 4-component coding scheme (reduced rank representation).

3. **Saccade axis orthogonalization:** `sS_inv` is orthogonalized against `sC` to remove category-correlated variance.

4. **Sampling rate:** Original spike times at 40 kHz (0.025 ms resolution).

---

## 11. Quick Reference: File Paths

```python
# Raw data
f"RCT_02/{sid}/trials.parquet"
f"RCT_02/{sid}/areas/{area}/units.json"
f"RCT_02/{sid}/areas/{area}/spikes/unit_{uid:03d}.h5"

# Cache
f"out/{align}/{sid}/caches/area_{area}.npz"

# Axes
f"out/{align}/{sid}/axes/{tag}/axes_{area}.npz"
f"out/{align}/{sid}/axes/{tag}/config.json"

# QC
f"out/{align}/{sid}/qc/{tag}/qc_axes_{area}.json"

# Flow
f"out/{align}/{sid}/flow/{tag}/{feature}/flow_{feature}_{areaA}to{areaB}.npz"

# Summary
f"out/{align}/summary/{tag}/{feature}/summary_{areaA}_vs_{areaB}.npz"
```

---

*Last updated: January 2026*

