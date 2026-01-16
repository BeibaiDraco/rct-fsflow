# Quick Start: Reproduce `crnull` Results

## What Happened?

The `crnull` folder was created using **legacy axes** (no tag) on **pooled trials**.  
Most likely: axes trained on **vertical** trials, flow on **pooled** trials (**mismatch!**)

## Quick Test (5 minutes)

```bash
cd /Users/dracoxu/Documents/Research/rct-fsflow/paper_project

# Backup current axes
mkdir -p out/stim/20200327/axes_backup
cp out/stim/20200327/axes/axes_*.npz out/stim/20200327/axes_backup/

# Test: vertical axes + pooled flow
python cli/train_axes.py \
  --out_root out --align stim --sid 20200327 \
  --orientation vertical --features C R --pt_min_ms 200.0

python cli/flow_session.py \
  --out_root out --align stim --sid 20200327 \
  --feature C --orientation pooled --tag crnull-test \
  --lags_ms 50.0 --ridge 0.01 --perms 500 --pt_min_ms 200.0

# Compare
python3 << 'EOF'
import numpy as np
orig = np.load('out/stim/20200327/flow/crnull/C/flow_C_MFEFtoMLIP.npz', allow_pickle=True)
test = np.load('out/stim/20200327/flow/crnull-test/C/flow_C_MFEFtoMLIP.npz', allow_pickle=True)
diff = np.nanmax(np.abs(orig['bits_AtoB'] - test['bits_AtoB']))
print(f'Max difference: {diff:.2e}')
print('MATCH!' if diff < 1e-10 else 'No match - try other scenarios')
EOF
```

## If That Doesn't Match

Try pooled-on-pooled:

```bash
python cli/train_axes.py \
  --out_root out --align stim --sid 20200327 \
  --orientation pooled --features C R --pt_min_ms 200.0

python cli/flow_session.py \
  --out_root out --align stim --sid 20200327 \
  --feature C --orientation pooled --tag crnull-test-pooled \
  --lags_ms 50.0 --ridge 0.01 --perms 500 --pt_min_ms 200.0

# Compare...
```

## Full Documentation

- **CRNULL_ANALYSIS_SUMMARY.md** - Complete analysis and reasoning
- **CRNULL_RECONSTRUCTION_GUIDE.md** - Technical details
- **reproduce_crnull.sh** - All test scenarios

## Key Finding

Your observation that "pooled-on-pooled doesn't match" confirms this is likely **vertical-on-pooled** (train/test mismatch).

This affects interpretation:
- Axes optimized for vertical structure
- Applied to both orientations
- May reduce sensitivity for horizontal trials

Consider using matched versions:
- `crnull-vertical` (vertical-on-vertical)
- `crnull-horizontal` (horizontal-on-horizontal)

