#!/bin/bash
# GMM baseline on a CPU node: env from the fci lock on local scratch, results to Lustre.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
WS=/ictstr01/groups/ml01/workspace/$USER/fci-diagnose
SCRATCH=/localscratch/$USER/fci-obsk-$SLURM_JOB_ID
export UV_CACHE_DIR=$SCRATCH/uv-cache TMPDIR=$SCRATCH/tmp OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p $UV_CACHE_DIR $TMPDIR
rm -rf $SCRATCH/fci && cp -a $WS/fci $SCRATCH/fci
cd $SCRATCH/fci
uv sync -q --frozen
uv pip install -q matplotlib scipy scikit-learn
uv run --no-sync python $WS/obs_and_k.py /lustre/groups/ml01/datasets/$USER/citeseq_panel/prepared $WS/out_obs_k
