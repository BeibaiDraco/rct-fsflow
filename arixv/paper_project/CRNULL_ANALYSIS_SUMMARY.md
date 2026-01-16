# Analysis: How was `crnull` generated?

## TL;DR

The `crnull` folder was created **BEFORE** tagged axes existed, using **legacy axes** at `axes/axes_{AREA}.npz`. The key question is: **what orientation were those axes trained on?**

**Most likely scenario:** Axes trained on **vertical** trials, flow computed on **pooled** trials (train/test mismatch).

## Evidence

### 1. Directory Structure & Timestamps

```
out/stim/20200327/
├── axes/
│   ├── axes_MFEF.npz           (Nov 12 21:25) - vertical-trained
│   ├── axes_MLIP.npz           (Nov 12 21:25) - vertical-trained
│   └── crnull-horizontal/      (Nov 12 21:33)
└── flow/
    ├── crnull/                 (Nov 11 21:57) ← OLDEST!
    ├── crnull-vertical/        (Nov 12 16:41)
    └── crnull-horizontal/      (Nov 12 21:33)
```

**Key observation:** `crnull` predates all tagged axes folders!

### 2. Axes Loading Logic (from `flow_session.py`)

```python
def _load_axes(out_root, align, sid, area, axes_tag, flow_tag):
    candidates = []
    if axes_tag:  # highest priority
        candidates.append(f"axes/{axes_tag}/axes_{area}.npz")
    if flow_tag:  # use flow tag if axes_tag not provided
        candidates.append(f"axes/{flow_tag}/axes_{area}.npz")
    candidates.append(f"axes/axes_{area}.npz")  # FALLBACK (legacy)
    
    # Returns first match
    ...
```

When `crnull` was generated:
- No `axes/crnull/` folder existed
- No `axes/crnull-vertical/` or `axes/crnull-horizontal/` existed
- **Only legacy `axes/axes_{AREA}.npz` existed**

### 3. What Were Those Legacy Axes?

From `axes_summary.json` (created later with same axes):
```json
{
  "sid": "20200327",
  "align": "stim",
  "features": ["C", "R"],
  "winC": [0.1, 0.3],
  "winR": [0.05, 0.2],
  "pt_min_ms": 200.0,
  "orientation": "vertical"   ← DEFAULT
}
```

**The default orientation is `vertical`!**

### 4. Why Doesn't Pooled-on-Pooled Match?

You mentioned:
> "when i just trained it on pooled data and apply to pooled data, it doesn't match"

This confirms that `crnull` is **NOT** pooled-on-pooled!

## Most Likely Scenario

**Axes:** Trained on **vertical** trials  
**Flow:** Computed on **pooled** trials  
**Result:** Train/test mismatch

### Why This Makes Sense

1. **Default behavior:** `train_axes.py` defaults to `--orientation vertical`
2. **Naming:** "crnull" (no orientation suffix) suggests pooled trials
3. **Timeline:** Created before orientation-specific variants
4. **Your observation:** Pooled-on-pooled doesn't match

## How to Reproduce

### Option A: Test My Hypothesis (Recommended)

```bash
cd paper_project

# 1. Backup current axes
mkdir -p out/stim/20200327/axes_backup
cp out/stim/20200327/axes/axes_*.npz out/stim/20200327/axes_backup/

# 2. Train vertical axes (overwrite legacy)
python cli/train_axes.py \
  --out_root out \
  --align stim \
  --sid 20200327 \
  --orientation vertical \
  --features C R \
  --pt_min_ms 200.0

# 3. Compute flow on pooled trials
for FEAT in C R; do
  python cli/flow_session.py \
    --out_root out \
    --align stim \
    --sid 20200327 \
    --feature $FEAT \
    --orientation pooled \
    --tag crnull-test \
    --lags_ms 50.0 \
    --ridge 0.01 \
    --perms 500 \
    --pt_min_ms 200.0
done

# 4. Compare
python3 << 'EOF'
import numpy as np
orig = np.load('out/stim/20200327/flow/crnull/C/flow_C_MFEFtoMLIP.npz', allow_pickle=True)
test = np.load('out/stim/20200327/flow/crnull-test/C/flow_C_MFEFtoMLIP.npz', allow_pickle=True)

print("Original bits_AtoB:", orig['bits_AtoB'][20:30])
print("Test bits_AtoB:    ", test['bits_AtoB'][20:30])
print("Max difference:    ", np.nanmax(np.abs(orig['bits_AtoB'] - test['bits_AtoB'])))

# Check metadata
orig_meta = orig['meta'].item()
test_meta = test['meta'].item()
print("\nOriginal N:", orig_meta.get('N'))
print("Test N:    ", test_meta.get('N'))
EOF
```

### Option B: Try All Scenarios

Run the script I created:

```bash
cd paper_project
./reproduce_crnull.sh
```

Then manually execute the commands for each scenario and compare.

## Key Parameters from `flow_original.py`

The `crnull` results were computed using `flow_original.py` (before `null_method` parameter):

```python
compute_flow_timecourse_for_pair(
    cacheA=cA,
    cacheB=cB,
    axesA=aA,  # from axes/axes_MFEF.npz (vertical-trained)
    axesB=aB,  # from axes/axes_MLIP.npz (vertical-trained)
    feature="C",          # or "R"
    align="stim",
    orientation=None,     # None = pooled trials
    pt_min_ms=200.0,
    lags_ms=50.0,
    ridge=0.01,
    perms=500,
    induced=True,
    include_B_lags=True,
    seed=0,
    perm_within="CR",     # shuffle within C×R strata
)
```

## Expected Trial Counts

From session 20200327 (total 1391 trials):

- **Vertical + correct + PT≥200:** ~300-500 trials (estimate)
- **Horizontal + correct + PT≥200:** ~300-500 trials (estimate)  
- **Pooled + correct + PT≥200:** ~600-1000 trials (estimate)

Check the `N` field in flow metadata to confirm which was used.

## What This Means for Your Analysis

The `crnull` results represent a **train/test mismatch**:
- Axes optimized for vertical trial structure
- Applied to pooled trials (both orientations)

This might:
1. Reduce sensitivity (axes not optimal for horizontal trials)
2. Introduce bias (vertical structure dominates)
3. Explain unexpected results

**Recommendation:** Use orientation-matched results:
- `crnull-vertical` (vertical axes + vertical trials)
- `crnull-horizontal` (horizontal axes + horizontal trials)

Or train new pooled axes:
- `crnull-pooled` (pooled axes + pooled trials)

## Next Steps

1. **Verify:** Run Option A above to confirm the hypothesis
2. **Document:** Record which scenario matches in your lab notebook
3. **Decide:** Choose appropriate version for final analysis:
   - Train/test matched: `crnull-vertical` or `crnull-horizontal`
   - Maximum power: `crnull-pooled` (if you create it)
4. **Update:** Regenerate any downstream analyses using the correct version

## Files Created

I've created three helper files for you:

1. **`CRNULL_RECONSTRUCTION_GUIDE.md`** - Detailed technical guide
2. **`reproduce_crnull.sh`** - Shell script with all test scenarios
3. **`diagnose_crnull.py`** - Python diagnostic script (has numpy issues, use shell script instead)
4. **`CRNULL_ANALYSIS_SUMMARY.md`** - This file

## Questions?

If none of the scenarios match, consider:
- Different `lags_ms` value
- Different `ridge` value  
- Different `pt_min_ms` threshold
- Different `perm_within` setting
- Used `flow.py` instead of `flow_original.py` (unlikely given timestamp)

Check the `run_all_sessions.py` script history for exact commands used.

