#!/bin/bash
#SBATCH --job-name=layeropd_pg_eval
#SBATCH --partition=normal
#SBATCH --account=shiyd2023
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
export LC_ALL=C
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PATH=/public/home/shiyd2023/miniconda3/envs/llm/bin:$PATH
REPO=${REPO:?}
RUN_ROOT=${RUN_ROOT:?}
SMOKE_ROOT=${SMOKE_ROOT:?}
export PYTHONPATH="$SMOKE_ROOT/runtime/lmms-eval:$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_ROOT"
cd "$REPO"
nvidia-smi > "$RUN_ROOT/nvidia-smi.txt"
IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES:?}"
world=${#allocated_gpus[@]}
test "$world" -ge 2
worker() {
  local rank=$1 gpu=$2 arm model
  export CUDA_VISIBLE_DEVICES="$gpu"
  # Counterbalance model order across GPU shards for timing.
  local arms='champion base'
  if (( rank % 2 )); then arms='base champion'; fi
  for arm in $arms; do
    model=/storage/data/shiyd2023/LLM_model/Qwen/Qwen3-VL-2B-Instruct
    if [[ "$arm" == champion ]]; then model="$SMOKE_ROOT/merged"; fi
    mkdir -p "$RUN_ROOT/$arm/shard$rank"
    python audit/run_measured.py --output "$RUN_ROOT/$arm/shard$rank/measurement.json" --module lmms_eval \
      --model qwen3_vl \
      --model_args "pretrained=$model,attn_implementation=eager,device=cuda,max_pixels=200704,min_pixels=100352,use_cache=false,device_map=cuda" \
      --tasks textvqa_val --batch_size 1 --offset "$((rank * 5000 / world))" --limit "$(((rank+1)*5000/world - rank*5000/world))" \
      --log_samples --output_path "$RUN_ROOT/$arm/shard$rank/results" \
      > "$RUN_ROOT/$arm/shard$rank/eval.log" 2>&1
  done
}
pids=()
for ((rank=0; rank<world; rank++)); do worker "$rank" "${allocated_gpus[$rank]}" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" == 0 ]]; then python audit/aggregate.py "$RUN_ROOT"; fi
exit "$status"
