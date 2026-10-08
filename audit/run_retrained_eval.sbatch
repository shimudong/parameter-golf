#!/bin/bash
#SBATCH --job-name=layeropd_pg_seed1_bf16
#SBATCH --partition=critical
#SBATCH --account=yujingyi-critical
#SBATCH --gres=gpu:NVIDIAL20:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
export LC_ALL=C OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PATH=/public/home/shiyd2023/miniconda3/envs/llm/bin:$PATH
REPO=${REPO:?}
RUN_ROOT=${RUN_ROOT:?}
SMOKE_ROOT=${SMOKE_ROOT:?}
MODEL_PATH=${MODEL_PATH:?}
export PYTHONPATH="$SMOKE_ROOT/runtime/lmms-eval:$REPO${PYTHONPATH:+:$PYTHONPATH}"
test -s "$MODEL_PATH/model.safetensors"
test -s "$MODEL_PATH/tokenizer_config.json"
mkdir -p "$RUN_ROOT"
cd "$REPO"
nvidia-smi > "$RUN_ROOT/nvidia-smi.txt"
sha256sum "$MODEL_PATH/model.safetensors" > "$RUN_ROOT/model.sha256"
IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES:?}"
test "${#allocated_gpus[@]}" -eq 2
worker() {
  local rank=$1 gpu=$2
  export CUDA_VISIBLE_DEVICES="$gpu"
  mkdir -p "$RUN_ROOT/retrained/shard$rank"
  python audit/run_measured.py --output "$RUN_ROOT/retrained/shard$rank/measurement.json" --module lmms_eval \
    --model qwen3_vl \
    --model_args "pretrained=$MODEL_PATH,attn_implementation=eager,device=cuda,max_pixels=200704,min_pixels=100352,use_cache=false,device_map=cuda" \
    --tasks textvqa_val --batch_size 1 --offset "$((rank*2500))" --limit 2500 --log_samples \
    --output_path "$RUN_ROOT/retrained/shard$rank/results" > "$RUN_ROOT/retrained/shard$rank/eval.log" 2>&1
}
worker 0 "${allocated_gpus[0]}" & p0=$!
worker 1 "${allocated_gpus[1]}" & p1=$!
status=0
wait "$p0" || status=1
wait "$p1" || status=1
if [[ "$status" == 0 ]]; then python audit/aggregate.py "$RUN_ROOT" retrained; fi
exit "$status"
