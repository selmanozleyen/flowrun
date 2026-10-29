#!/bin/bash
# Inside the held GPU allocation: env from the fci lock on node-local scratch, results to Lustre.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
WS=/ictstr01/groups/ml01/workspace/$USER/fci-diagnose
SCRATCH=/localscratch/$USER/fci-diag-$SLURM_JOB_ID
export UV_CACHE_DIR=$SCRATCH/uv-cache TMPDIR=$SCRATCH/tmp
mkdir -p $UV_CACHE_DIR $TMPDIR
rm -rf $SCRATCH/fci && cp -a $WS/fci $SCRATCH/fci
cd $SCRATCH/fci
uv sync -q --frozen
uv pip install -q matplotlib scipy
uv run --no-sync python -c "import torch, flowrun, sckitflow; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
uv run --no-sync python $WS/wandb_query.py > $WS/wandb_query.log 2>&1 || echo "wandb query failed, see wandb_query.log"
uv run --no-sync python $WS/diagnose.py /lustre/groups/ml01/workspace/$USER/fci_runs/grid_2026-10-22_07-43-14/3 $WS/out 20000
