"""Read-only, reproducible inventory of the submitted adapter and result claims."""
import hashlib
import json
import math
import pathlib
import statistics
import struct

import numpy as np

repo = pathlib.Path(__file__).resolve().parents[1]
file = repo / 'weights/adapter_model.safetensors'
blob = file.read_bytes()
header_length = struct.unpack('<Q', blob[:8])[0]
header = json.loads(blob[8:8 + header_length])
tensors = {k: v for k, v in header.items() if k != '__metadata__'}
for name, info in tensors.items():
    assert info['dtype'] == 'F32', (name, info['dtype'])
    start, end = info['data_offsets']
    data = np.frombuffer(blob, dtype='<f4', count=(end-start)//4, offset=8+header_length+start)
    assert data.size == math.prod(info['shape'])
    assert np.isfinite(data).all(), name
scores = []
claims = []
for path in sorted((repo / 'results/textvqa').glob('*.json')):
    content = json.loads(path.read_text())
    score = content['results']['textvqa_val']['exact_match,none']
    scores.append(score)
    claims.append({'file': str(path.relative_to(repo)), 'score': score,
                   'sample_count': content['n-samples']['textvqa_val'],
                   'throughput': content.get('throughput'),
                   'log_samples': content['config']['resolved_cli_args']['log_samples']})
result = {'target_commit': 'c926b2c397e3af4af06e4cf1738b5f991439da59',
          'adapter_sha256': hashlib.sha256(blob).hexdigest(),
          'adapter_bytes': len(blob), 'adapter_mib': len(blob) / 2**20,
          'tensors': len(tensors), 'all_values_finite': True,
          'parameters': sum(math.prod(v['shape']) for v in tensors.values()),
          'vision_tensors': sum('.visual.' in k for k in tensors),
          'language_tensors': sum('.language_model.' in k for k in tensors),
          'submitted_results': claims, 'submitted_mean': statistics.mean(scores),
          'submitted_sample_sd': statistics.stdev(scores),
          'provenance': json.loads((repo / 'weights/soup_manifest.json').read_text())}
output = repo / 'audit/evidence/submission_inventory.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
