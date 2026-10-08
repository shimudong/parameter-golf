#!/bin/bash
#SBATCH --job-name=layeropd_pg_seed1
#SBATCH --partition=normal
#SBATCH --account=shiyd2023
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
export LC_ALL=C
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PATH=/public/home/shiyd2023/miniconda3/envs/llm/bin:$PATH
export SEED=1
REPO=${REPO:?}
RUN_ROOT=${RUN_ROOT:?}
SMOKE_ROOT=${SMOKE_ROOT:?}
export PYTHONPATH="$SMOKE_ROOT/runtime/lmms-eval:$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_ROOT"
cd "$REPO"
nvidia-smi > "$RUN_ROOT/nvidia-smi.txt"
python audit/verify_prepared.py "$RUN_ROOT/prepared_equivalence.json"
python - "$REPO" "$RUN_ROOT" "$SMOKE_ROOT" <<'PY'
import pathlib, shutil, sys, yaml, json
repo, root, smoke = map(pathlib.Path, sys.argv[1:])
cfg = yaml.safe_load((repo / 'configs/vlm_textvqa_lora.yaml').read_text())
cfg.update(model_path='/storage/data/shiyd2023/LLM_model/Qwen/Qwen3-VL-2B-Instruct',
           data_path='/storage/data/shiyd2023/datasets/textvqa/default/train/*.parquet',
           prepared_data_dir='/public/home/shiyd2023/shiyd/parameter-golf/data/prepared_textvqa_qwen3vl_seed1',
           output_dir=str(root/'training'))
(root/'seed1_train.yaml').write_text(yaml.safe_dump(cfg))
# Keep this diagnostic separate from original BF16 evaluation. Only dtype changes.
shutil.copytree(smoke/'runtime/lmms-eval', root/'fp16_runtime/lmms-eval', dirs_exist_ok=True)
wrapper=root/'fp16_runtime/lmms-eval/lmms_eval/models/simple/qwen3_vl.py'
text=wrapper.read_text()
assert text.count('dtype_key: "bfloat16"') == 1
wrapper.write_text(text.replace('dtype_key: "bfloat16"', 'dtype_key: "float16"'))
(root/'protocol.json').write_text(json.dumps({'training':'submitted seed-1 config; only filesystem paths adapted; one RTX 2080 Ti; 3600-second callback',
    'evaluation':'FP16 compatibility diagnostic. Original evaluator is hard-coded BF16. Do not label these scores as an exact BF16 reproduction.'}, indent=2)+'\n')
PY
IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES:?}"
test "${#allocated_gpus[@]}" -eq 2
eval_fp16() {
  local model=$1 arm=$2
  export PYTHONPATH="$RUN_ROOT/fp16_runtime/lmms-eval:$REPO"
  mkdir -p "$RUN_ROOT/$arm"
  python audit/run_measured.py --output "$RUN_ROOT/$arm/measurement.json" --module lmms_eval \
    --model qwen3_vl \
    --model_args "pretrained=$model,attn_implementation=eager,device=cuda,max_pixels=200704,min_pixels=100352,use_cache=false,device_map=cuda" \
    --tasks textvqa_val --batch_size 1 --log_samples --output_path "$RUN_ROOT/$arm/results" \
    > "$RUN_ROOT/$arm/eval.log" 2>&1
}
train_and_eval() {
  export CUDA_VISIBLE_DEVICES="${allocated_gpus[0]}"
  python audit/run_measured.py --output "$RUN_ROOT/train_measurement.json" --script "$REPO/train_textvqa_qwen3vl.py" --config "$RUN_ROOT/seed1_train.yaml" > "$RUN_ROOT/train.log" 2>&1
  python merge_lora.py --base_model /storage/data/shiyd2023/LLM_model/Qwen/Qwen3-VL-2B-Instruct --adapter "$RUN_ROOT/training/final" --output "$RUN_ROOT/retrained_merged" --dtype float16 > "$RUN_ROOT/merge.log" 2>&1
  eval_fp16 "$RUN_ROOT/retrained_merged" retrained_fp16
}
reference_eval() {
  export CUDA_VISIBLE_DEVICES="${allocated_gpus[1]}"
  eval_fp16 "$SMOKE_ROOT/merged" champion_fp16
  eval_fp16 /storage/data/shiyd2023/LLM_model/Qwen/Qwen3-VL-2B-Instruct base_fp16
}
train_and_eval & train_pid=$!
reference_eval & eval_pid=$!
train_status=0; eval_status=0
wait "$train_pid" || train_status=$?
wait "$eval_pid" || eval_status=$?
printf '{"train_and_eval_exit":%s,"reference_eval_exit":%s}\n' "$train_status" "$eval_status" > "$RUN_ROOT/status.json"
exit $((train_status || eval_status))
