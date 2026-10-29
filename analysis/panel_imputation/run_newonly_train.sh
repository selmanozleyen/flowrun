#!/bin/bash
# GPU: the new-only flow (shared markers recalibrated and kept, the 12 new-only generated), wide model, eps 0.1.
set -euo pipefail
set -f
export PATH="$HOME/.local/bin:$PATH" HYDRA_FULL_ERROR=1
WS=/ictstr01/groups/ml01/workspace/$USER/fci-diagnose
OUT=/lustre/groups/ml01/workspace/$USER/fci_runs/newonly_$(date +%Y-%m-%d_%H-%M)
SCRATCH=/localscratch/$USER/fci-not-$SLURM_JOB_ID
export UV_CACHE_DIR=$SCRATCH/uv-cache TMPDIR=$SCRATCH/tmp
mkdir -p $UV_CACHE_DIR $TMPDIR
rm -rf $SCRATCH/fci && cp -a $WS/fci $SCRATCH/fci
cd $SCRATCH/fci
uv sync -q --frozen
uv pip install -q matplotlib scipy scikit-learn
source $WS/newonly_args.sh
uv run --no-sync fci-fit "${NEWONLY[@]}" trainer.max_epochs=120 callbacks.early_stopping.patience=100 \
  hydra.run.dir=$OUT loggers.wandb.name=newonly_eps0.1_wide
echo "TRAIN_DONE $OUT"
