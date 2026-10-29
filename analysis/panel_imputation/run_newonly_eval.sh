#!/bin/bash
# GPU: evaluate the trained new-only flow (per-cell inside the new panel vs the GMM, and the old panel imputed).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
WS=/ictstr01/groups/ml01/workspace/$USER/fci-diagnose
SCRATCH=/localscratch/$USER/fci-noe-$SLURM_JOB_ID
export UV_CACHE_DIR=$SCRATCH/uv-cache TMPDIR=$SCRATCH/tmp
mkdir -p $UV_CACHE_DIR $TMPDIR
rm -rf $SCRATCH/fci && cp -a $WS/fci $SCRATCH/fci
cd $SCRATCH/fci
uv sync -q --frozen
uv pip install -q matplotlib scipy scikit-learn
uv run --no-sync python $WS/newonly_eval.py /lustre/groups/ml01/workspace/$USER/fci_runs/newonly_2026-10-26_21-03 \
  $WS/out_newonly $WS/out_gmm_percell/percell_new_only.csv
