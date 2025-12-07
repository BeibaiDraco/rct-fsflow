# How to Reproduce `crnull` Results

## Summary of Findings

Based on the directory structure and timestamps, here's what happened:

### Directory Structure
```
out/stim/20200327/
├── axes/
│   ├── axes_MFEF.npz                    # Created Nov 12 21:25 (vertical-trained)
│   ├── axes_MLIP.npz                    # Created Nov 12 21:25 (vertical-trained)
│   └── crnull-horizontal/
│       ├── axes_MFEF.npz                # Created Nov 12 21:33 (horizontal-trained)
│       ├── axes_MLIP.npz                # Created Nov 12 21:33 (horizontal-trained)
│       └── axes_summary.json            # orientation="horizontal"
└── flow/
    ├── crnull/                          # Created Nov 11 21:57 (OLDEST)
    │   ├── C/ (flow results)
    │   └── R/ (flow results)
    ├── crnull-vertical/                 # Created Nov 12 16:41
    │   ├── C/ (flow results)
    └── crnull-horizontal/               # Created Nov 12 21:33 (NEWEST)
        ├── C/ (flow results)
        └── R/ (flow results)
```

## Key Observation

**The `crnull` folder was created BEFORE any tagged axes existed!** 

This means:
- `crnull` flow results used the **legacy axes** at `axes/axes_{AREA}.npz`
- These legacy axes were likely trained on **vertical trials** (the default)
- But the flow computation used **pooled trials** (orientation=None)

## Hypothesis

The `crnull` results were generated with one of two scenarios:

### Scenario A (Most Likely): Vertical axes + Pooled flow
```bash
# 1) Train axes on vertical trials (default behavior)
python cli/train_axes.py \
  --out_root out \
  --align stim \
  --sid 20200327 \
  --orientation vertical \
  --features C R \
  --pt_min_ms 200.0
  # No --tag argument, so saves to axes/axes_{AREA}.npz

# 2) Compute flow using those axes but on pooled trials
python cli/flow_session.py \
  --out_root out \
  --align stim \
  --sid 20200327 \
  --feature C \
  --orientation pooled \
  --tag crnull \
  --lags_ms 50.0 \
  --ridge 0.01 \
  --perms 500 \
  --pt_min_ms 200.0
  # axes_tag not specified, falls back to legacy axes/axes_{AREA}.npz
```

### Scenario B (Less Likely): Pooled axes + Pooled flow
```bash
# 1) Train axes on pooled trials
python cli/train_axes.py \
  --out_root out \
  --align stim \
  --sid 20200327 \
  --orientation pooled \
  --features C R \
  --pt_min_ms 200.0

# 2) Compute flow using those axes on pooled trials
python cli/flow_session.py \
  --out_root out \
  --align stim \
  --sid 20200327 \
  --feature C \
  --orientation pooled \
  --tag crnull \
  --lags_ms 50.0 \
  --ridge 0.01 \
  --perms 500 \
  --pt_min_ms 200.0
```

## How to Determine Which Scenario

Check the axes file metadata to see what orientation was used:

```bash
python3 << 'EOF'
import numpy as np
import json

axes = np.load('out/stim/20200327/axes/axes_MFEF.npz', allow_pickle=True)
print("Keys:", list(axes.keys()))

if 'meta' in axes:
    meta = axes['meta']
    if hasattr(meta, 'item'):
        meta = meta.item()
    
    if isinstance(meta, dict):
        print("\nOrientation used:", meta.get('orientation', 'NOT SPECIFIED'))
        print("Trial count:", meta.get('N', 'NOT SPECIFIED'))
        print("Full meta:")
        print(json.dumps(meta, indent=2))
EOF
```

## How to Reproduce `crnull`

### Method 1: Most Conservative (Recreate from scratch)

1. **Delete the legacy axes** (back them up first!):
```bash
mkdir -p out/stim/20200327/axes_backup
cp out/stim/20200327/axes/axes_*.npz out/stim/20200327/axes_backup/
```

2. **Train axes with explicit settings** (try both scenarios):

**Scenario A:**
```bash
# Train on vertical
python cli/train_axes.py \
  --out_root out \
  --align stim \
  --sid 20200327 \
  --orientation vertical \
  --features C R \
  --pt_min_ms 200.0

# Flow on pooled  
python cli/flow_session.py \
  --out_root out \
  --align stim \
  --sid 20200327 \
  --feature C \
  --orientation pooled \
  --tag crnull-test \
  --lags_ms 50.0 \
  --ridge 0.01 \
  --perms 500 \
  --pt_min_ms 200.0

# Repeat for feature R
python cli/flow_session.py \
  --out_root out \
  --align stim \
  --sid 20200327 \
  --feature R \
  --orientation pooled \
  --tag crnull-test \
  --lags_ms 50.0 \
  --ridge 0.01 \
  --perms 500 \
  --pt_min_ms 200.0
```

**Scenario B:**
```bash
# Train on pooled
python cli/train_axes.py \
  --out_root out \
  --align stim \
  --sid 20200327 \
  --orientation pooled \
  --features C R \
  --pt_min_ms 200.0

# Flow on pooled
python cli/flow_session.py \
  --out_root out \
  --align stim \
  --sid 20200327 \
  --feature C \
  --orientation pooled \
  --tag crnull-test \
  --lags_ms 50.0 \
  --ridge 0.01 \
  --perms 500 \
  --pt_min_ms 200.0
```

3. **Compare results**:
```bash
python3 << 'EOF'
import numpy as np

# Load original crnull
orig = np.load('out/stim/20200327/flow/crnull/C/flow_C_MFEFtoMLIP.npz', allow_pickle=True)
# Load your test
test = np.load('out/stim/20200327/flow/crnull-test/C/flow_C_MFEFtoMLIP.npz', allow_pickle=True)

print("Original bits_AtoB:", orig['bits_AtoB'])
print("\nTest bits_AtoB:", test['bits_AtoB'])
print("\nDifference:", np.nanmax(np.abs(orig['bits_AtoB'] - test['bits_AtoB'])))
EOF
```

### Method 2: Inspect Original Axes

Try to extract metadata from the existing axes files to determine orientation:

```bash
# Check if axes have readable metadata
python3 -c "
import numpy as np
axes = np.load('out/stim/20200327/axes/axes_MFEF.npz', allow_pickle=True)
if 'meta' in axes:
    print('Has meta!')
    meta = axes['meta']
    if hasattr(meta, 'item'):
        print(meta.item())
"
```

## Parameters for flow_original.py

Looking at `flow_original.py`, the key parameters are:

```python
compute_flow_timecourse_for_pair(
    cacheA=cA, 
    cacheB=cB,
    axesA=aA, 
    axesB=aB,
    feature="C",              # or "R"
    align="stim",
    orientation=None,          # None = pooled (vs "vertical" or "horizontal")
    pt_min_ms=200.0,          # PT threshold
    lags_ms=50.0,             # lag window
    ridge=0.01,               # ridge penalty
    perms=500,                # permutations
    induced=True,             # remove induced activity
    include_B_lags=True,      # include target's own history
    seed=0,
    perm_within="CR",         # shuffle within C×R strata
)
```

## Key Differences Between Variants

- **`crnull`**: Likely vertical-trained axes, pooled trials (mismatch!)
- **`crnull-vertical`**: Vertical-trained axes, vertical trials (matched)
- **`crnull-horizontal`**: Horizontal-trained axes, horizontal trials (matched)

## Recommendation

The fact that `crnull` doesn't match when you train+apply to pooled suggests **Scenario A** is correct:
- Axes trained on **vertical** trials
- Flow computed on **pooled** trials

This is a **train/test mismatch** that might explain unexpected results.

To properly reproduce `crnull`, you need to:
1. Check the exact axes that existed on Nov 11 21:57
2. Or recreate using Scenario A above
3. Compare with your current pooled-on-pooled results to see which matches

## Code Version

The `crnull` results were generated with `flow_original.py` (before `null_method` parameter was added).
The key signature is line 158 in `flow_original.py`:

```python
def compute_flow_timecourse_for_pair(
    cacheA: Dict, cacheB: Dict,
    axesA: Dict, axesB: Dict,
    feature: str,                   # 'C' | 'R' | 'S'
    align: str,                     # 'stim' | 'sacc'
    orientation: Optional[str],
    pt_min_ms: Optional[float],
    lags_ms: float,
    ridge: float,
    perms: int = 500,
    induced: bool = True,
    include_B_lags: bool = True,
    seed: int = 0,
    perm_within: str = "CR",        # NEW param in this version
)
```

No `null_method` parameter → uses trial shuffle only.

