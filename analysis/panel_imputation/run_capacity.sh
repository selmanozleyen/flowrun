#!/bin/bash
# Capacity grid: 4 variants of the citeseq fused run (best coupling eps=1, alpha=0.25) in parallel on one GPU,
# then the per-marker diagnosis (one sample per cell) on each saved model. Env from the fci lock on local scratch.
set -euo pipefail
set -f  # no globbing: overrides contain [..]
export PATH="$HOME/.local/bin:$PATH"
WS=/ictstr01/groups/ml01/workspace/$USER/fci-diagnose
OUT=/lustre/groups/ml01/workspace/$USER/fci_runs/capacity_$(date +%Y-%m-%d_%H-%M)
SCRATCH=/localscratch/$USER/fci-cap-$SLURM_JOB_ID
export UV_CACHE_DIR=$SCRATCH/uv-cache TMPDIR=$SCRATCH/tmp
mkdir -p $UV_CACHE_DIR $TMPDIR $OUT
rm -rf $SCRATCH/fci && cp -a $WS/fci $SCRATCH/fci
cd $SCRATCH/fci
uv sync -q --frozen
uv pip install -q matplotlib scipy

COMMON=(+experiment=citeseq coupling.epsilon=1 coupling.alpha=0.25 trainer.max_epochs=300
        callbacks.early_stopping.patience=60)
WIDE=('module.state_encoder_output_dim=64' 'module.time_encoder_output_dim=64'
      'module.state_encoder_mlp_kwargs.hidden_dims=[128]' 'module.time_encoder_mlp_kwargs.hidden_dims=[128]'
      'module.vf_decoder_mlp_kwargs.hidden_dims=[256,256]')
SRC=('module.source_encoder_output_dim=64' 'module.source_encoder_mlp_kwargs.hidden_dims=[128]')
declare -A VARIANTS=([base]="" [wide]="${WIDE[*]}" [src64]="${SRC[*]}" [wide_src64]="${WIDE[*]} ${SRC[*]}")

for name in "${!VARIANTS[@]}"; do
  # shellcheck disable=SC2086
  uv run --no-sync fci-fit "${COMMON[@]}" ${VARIANTS[$name]} hydra.run.dir=$OUT/$name loggers.wandb.name=capacity_$name \
    > $OUT/$name.log 2>&1 &
  sleep 30   # stagger start-up (data load)
done
wait || true
for name in "${!VARIANTS[@]}"; do
  uv run --no-sync python $WS/diagnose.py $OUT/$name $OUT/$name/diagnosis 20000 sample@10 > $OUT/$name/diagnosis.log 2>&1 || echo "diagnosis failed: $name"
done
echo "done: $OUT"
