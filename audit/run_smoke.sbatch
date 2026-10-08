#!/bin/bash
#SBATCH --job-name=layeropd_pg_smoke
#SBATCH --partition=normal
#SBATCH --account=shiyd2023
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:30:00
set -euo pipefail
export LC_ALL=C
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PATH=/public/home/shiyd2023/miniconda3/envs/llm/bin:$PATH
REPO=${REPO:?}
RUN_ROOT=${RUN_ROOT:?}
export PYTHONPATH="$REPO/lmms-eval:$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_ROOT"
cd "$REPO"
nvidia-smi > "$RUN_ROOT/nvidia-smi.txt"
python audit/prepare_runtime.py --repo "$REPO" --run-root "$RUN_ROOT"
export PYTHONPATH="$RUN_ROOT/runtime/lmms-eval:$REPO${PYTHONPATH:+:$PYTHONPATH}"
python merge_lora.py --base_model /storage/data/shiyd2023/LLM_model/Qwen/Qwen3-VL-2B-Instruct --adapter "$REPO/weights" --output "$RUN_ROOT/merged" --dtype float16 > "$RUN_ROOT/merge.log" 2>&1
set +e
(
  export CUDA_VISIBLE_DEVICES=0
  python -m lmms_eval --model qwen3_vl --model_args "pretrained=$RUN_ROOT/merged,attn_implementation=eager,device=cuda,max_pixels=200704,min_pixels=100352,use_cache=false,device_map=cuda" --tasks textvqa_val --batch_size 1 --limit 16 --log_samples --output_path "$RUN_ROOT/eval_smoke"
) > "$RUN_ROOT/eval_smoke.log" 2>&1 &
eval_pid=$!
(
  export CUDA_VISIBLE_DEVICES=1
  export SEED=1
  python audit/run_measured.py --output "$RUN_ROOT/train_smoke_measurement.json" --script "$REPO/train_textvqa_qwen3vl.py" --config "$RUN_ROOT/smoke_train.yaml"
) > "$RUN_ROOT/train_smoke.log" 2>&1 &
train_pid=$!
wait "$eval_pid"; eval_status=$?
wait "$train_pid"; train_status=$?
printf '{"eval_exit":%s,"train_exit":%s}\n' "$eval_status" "$train_status" > "$RUN_ROOT/status.json"
cat "$RUN_ROOT/status.json"
exit $((eval_status || train_status))
