#!/bin/bash
# Coupling sharpness: the wide model with a 64-wide source encoder, ~3k steps, fused GW epsilon 1 (control),
# 0.1 and 0.05 in parallel on one GPU; then per-marker plots and the shared-marker source correlation per run.
set -euo pipefail
set -f  # no globbing: overrides contain [..]
export PATH="$HOME/.local/bin:$PATH"
WS=/ictstr01/groups/ml01/workspace/$USER/fci-diagnose
OUT=/lustre/groups/ml01/workspace/$USER/fci_runs/sharp_$(date +%Y-%m-%d_%H-%M)
SCRATCH=/localscratch/$USER/fci-sharp-$SLURM_JOB_ID
export UV_CACHE_DIR=$SCRATCH/uv-cache TMPDIR=$SCRATCH/tmp
mkdir -p $UV_CACHE_DIR $TMPDIR $OUT
rm -rf $SCRATCH/fci && cp -a $WS/fci $SCRATCH/fci
cd $SCRATCH/fci
uv sync -q --frozen
uv pip install -q matplotlib scipy

COMMON=(+experiment=citeseq coupling.alpha=0.25 trainer.max_epochs=30 callbacks.early_stopping.patience=100
        module.state_encoder_output_dim=64 module.time_encoder_output_dim=64
        'module.state_encoder_mlp_kwargs.hidden_dims=[128]' 'module.time_encoder_mlp_kwargs.hidden_dims=[128]'
        'module.vf_decoder_mlp_kwargs.hidden_dims=[256,256]'
        module.source_encoder_output_dim=64 'module.source_encoder_mlp_kwargs.hidden_dims=[128]')
for eps in 1 0.1 0.05; do
  name=eps$eps
  uv run --no-sync fci-fit "${COMMON[@]}" coupling.epsilon=$eps hydra.run.dir=$OUT/$name loggers.wandb.name=sharp_$name \
    > $OUT/$name.log 2>&1 &
  sleep 30   # stagger start-up (data load)
done
wait || true
for eps in 1 0.1 0.05; do
  name=eps$eps
  m=$(ls $OUT/$name/logs/csv/*/*/metrics.csv | head -1)
  echo "$name coupling_fallback steps: $(awk -F, 'NR==1{for(i=1;i<=NF;i++)if($i=="coupling_fallback")c=i;next} c&&$c!=""{s+=$c} END{print s+0}' $m)"
  uv run --no-sync python $WS/diagnose.py $OUT/$name $OUT/$name/diagnosis 20000 sample@10 > $OUT/$name/diagnosis.log 2>&1 \
    || echo "diagnosis failed: $name"
  grep -E "edist|spearman" $OUT/$name/diagnosis.log | sed "s/^/$name: /"
done
echo "done: $OUT"
