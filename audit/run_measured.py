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
p.add_argument('--script', required=True)
args, rest = p.parse_known_args()
started = time.perf_counter()
record = {'script': args.script, 'argv': rest, 'status': 'running'}
try:
    torch.cuda.reset_peak_memory_stats()
    record['gpu'] = torch.cuda.get_device_name()
    record['total_memory_bytes'] = torch.cuda.get_device_properties(0).total_memory
    sys.argv = [args.script] + rest
    runpy.run_path(args.script, run_name='__main__')
    record['status'] = 'success'
except BaseException:
    record['status'] = 'failed'
    record['traceback'] = traceback.format_exc()
    raise
finally:
    record['wall_seconds'] = time.perf_counter() - started
    record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    record['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    pathlib.Path(args.output).write_text(json.dumps(record, indent=2) + '\n')
