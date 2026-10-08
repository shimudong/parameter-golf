# Layer-OPD parameter-golf independent audit

## 2026-10-08 — submission preflight

- Target: `https://github.com/myg321/parameter-golf`, submitted commit `c926b2c` (full SHA recorded by Git).
- Rules read: `/Users/shiyingdong/Downloads/layer-opd/AGENTS.md`, submission ledger, and its referenced Azure template. This cluster uses Slurm; obsolete Azure `execution_mode: basic` is not used.
- The existing cluster baseline README specifies 1 or 2 RTX 2080 Ti GPUs, 1 hour training, latency/FLOPs <= 1.1x base, and results over 3 seeds.
- Local artifact audit: adapter SHA256 `a22e51ed25752b6964c1a63a434bbca2337ae3e31d5d3f727dd4256c0427efe4`, 416 FP32 tensors, 18,219,008 parameters, 72,937,256 bytes (69.5584 MiB).
- Submitted JSON arithmetic: seed scores 0.73790 / 0.72958 / 0.73634, mean 0.7346066667, sample SD 0.0044225483. These are submitted claims, not independently reproduced scores.
- The `exact_match` metric implements standard TextVQA leave-one-annotator-out soft accuracy. Only 3 summary JSONs are present, with `log_samples=false`; the 59 experiment files described in the supplement are not in this checkout.
- Hardware/environment discovery: login01 reachable via the user-authorized account; no jobs in that account when inspected. Qwen base and TextVQA local data found; `llm` environment core versions match `requirements.txt`.
- Storage: `/llm_storage` does not exist on this Slurm login node. Use its persistent-cluster-storage equivalent `/storage/data/shiyd2023/opd/layer-opd/parameter-golf-audit/`; keep all model outputs there, not in node temporary storage.
- RUN_NAME: `layeropd_pg_smoke_c926b2c_20261008a`; method=unchanged submitted champion eval + 2-step training functionality; world_size=2 packed independent processes, one GPU each. Entry: `audit/run_smoke.sbatch`; no accuracy or time-budget conclusion from smoke.
- Planned gate: preserve original evaluator dtype/decoding, adapt filesystem paths only, and record failures before any compatibility modification. Formal training will require fresh preparation or full prepared-data equivalence checks.
