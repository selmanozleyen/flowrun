#!/bin/bash
# CPU: write the new-only panel copies, then a fast_dev_run of the new-only flow on them.
set -euo pipefail
set -f
export PATH="$HOME/.local/bin:$PATH"
WS=/ictstr01/groups/ml01/workspace/$USER/fci-diagnose
SCRATCH=/localscratch/$USER/fci-nop-$SLURM_JOB_ID
export UV_CACHE_DIR=$SCRATCH/uv-cache TMPDIR=$SCRATCH/tmp
mkdir -p $UV_CACHE_DIR $TMPDIR
rm -rf $SCRATCH/fci && cp -a $WS/fci $SCRATCH/fci
cd $SCRATCH/fci
uv sync -q --frozen
source $WS/newonly_args.sh
rm -rf $N
uv run --no-sync python $WS/prep_newonly.py /lustre/groups/ml01/datasets/$USER/citeseq_panel/prepared $N
export HYDRA_FULL_ERROR=1
uv run --no-sync fci-fit "${NEWONLY[@]}" trainer.fast_dev_run=True trainer.accelerator=cpu loggers.wandb.offline=True \
  hydra.run.dir=$WS/newonly_smoke
echo SMOKE_DONE
