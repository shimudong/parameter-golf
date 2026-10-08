"""Stage a path-only evaluation adaptation and a two-step training smoke."""
import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import shutil

import yaml

p = argparse.ArgumentParser()
p.add_argument('--repo', type=pathlib.Path, required=True)
p.add_argument('--run-root', type=pathlib.Path, required=True)
a = p.parse_args()
a.run_root.mkdir(parents=True, exist_ok=True)
runtime = a.run_root / 'runtime'
shutil.copytree(a.repo / 'lmms-eval', runtime / 'lmms-eval', dirs_exist_ok=True)
task = runtime / 'lmms-eval/lmms_eval/tasks/textvqa/_default_template_textvqa_yaml'
before = task.read_text()
after = before.replace('dataset_path: /storage/yiguang/all_datasets/textvqa # lmms-lab/textvqa', 'dataset_path: /storage/data/shiyd2023/datasets/textvqa # cluster path only')
assert before != after
task.write_text(after)
cfg = yaml.safe_load((a.repo / 'configs/vlm_textvqa_lora.yaml').read_text())
cfg.update(model_path='/storage/data/shiyd2023/LLM_model/Qwen/Qwen3-VL-2B-Instruct',
           data_path='/storage/data/shiyd2023/datasets/textvqa/default/train/*.parquet',
           prepared_data_dir='/public/home/shiyd2023/shiyd/parameter-golf/data/prepared_textvqa_qwen3vl_seed{seed}',
           output_dir=str(a.run_root / 'smoke_train'), max_steps=2, logging_steps=1)
(a.run_root / 'smoke_train.yaml').write_text(yaml.safe_dump(cfg))
packages = ['torch', 'transformers', 'peft', 'accelerate', 'datasets', 'qwen-vl-utils', 'lmms_eval']
manifest = {'packages': {p: importlib.metadata.version(p) for p in packages},
            'base_model': cfg['model_path'], 'data': cfg['data_path'],
            'adapter_sha256': hashlib.sha256((a.repo / 'weights/adapter_model.safetensors').read_bytes()).hexdigest(),
            'eval_change': 'Only dataset_path changed; original BF16 model wrapper preserved.',
            'smoke_change': 'Paths, max_steps=2 and logging_steps=1 only. Reuses existing prepared data for functional smoke, not formal retraining.'}
(a.run_root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
