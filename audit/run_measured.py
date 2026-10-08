"""Measure an unchanged submitted Python entrypoint, including failures."""
import argparse
import json
import pathlib
import runpy
import sys
import time
import traceback

import torch

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
entry = p.add_mutually_exclusive_group(required=True)
entry.add_argument('--script')
entry.add_argument('--module')
args, rest = p.parse_known_args()
started = time.perf_counter()
record = {'script': args.script, 'module': args.module, 'argv': rest, 'status': 'running'}
generation_measurements = []
if args.module == 'lmms_eval':
    from transformers.generation.utils import GenerationMixin
    original_generate = GenerationMixin.generate

    def measured_generate(model, *positional, **kwargs):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        output = original_generate(model, *positional, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        input_ids = kwargs.get('input_ids')
        generated = output.shape[-1] - input_ids.shape[-1] if input_ids is not None and hasattr(output, 'shape') else None
        generation_measurements.append({'seconds': elapsed, 'new_tokens': generated})
        return output

    GenerationMixin.generate = measured_generate
try:
    torch.cuda.reset_peak_memory_stats()
    record['gpu'] = torch.cuda.get_device_name()
    record['total_memory_bytes'] = torch.cuda.get_device_properties(0).total_memory
    sys.argv = [args.script or args.module] + rest
    if args.script:
        runpy.run_path(args.script, run_name='__main__')
    else:
        runpy.run_module(args.module, run_name='__main__', alter_sys=True)
    record['status'] = 'success'
except SystemExit as exc:
    record['exit_code'] = exc.code
    if exc.code in (None, 0):
        record['status'] = 'success'
    else:
        record['status'] = 'failed'
        record['traceback'] = traceback.format_exc()
        raise
except BaseException:
    record['status'] = 'failed'
    record['traceback'] = traceback.format_exc()
    raise
finally:
    record['wall_seconds'] = time.perf_counter() - started
    record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    record['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    if generation_measurements:
        record['generation_measurements'] = generation_measurements
        record['generation_seconds'] = sum(x['seconds'] for x in generation_measurements)
        record['generation_count'] = len(generation_measurements)
    pathlib.Path(args.output).write_text(json.dumps(record, indent=2) + '\n')
