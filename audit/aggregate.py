"""Aggregate independent, disjoint TextVQA samples; reject missing/duplicate IDs."""
import hashlib
import json
import math
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
arms = sys.argv[2:] or ['champion', 'base']
summary = {}
all_samples = {}
for arm in arms:
    samples = []
    files = sorted((root / arm).glob('shard*/results/**/*.jsonl'))
    assert files, f'No sample files for {arm}'
    for file in files:
        samples.extend(json.loads(line) for line in file.read_text().splitlines() if line.strip())
    ids = [row['doc_id'] for row in samples]
    assert len(ids) == 5000 and set(ids) == set(range(5000)), (arm, len(ids), len(set(ids)), min(ids), max(ids))
    samples.sort(key=lambda row: row['doc_id'])
    scores = [row['exact_match'] for row in samples]
    reports = [json.loads(p.read_text()) for p in sorted((root / arm).glob('shard*/results/**/*_results.json'))]
    measurements = [json.loads(p.read_text()) for p in sorted((root / arm).glob('shard*/measurement.json'))]
    # Older audit wrapper recorded the CLI's intentional sys.exit(0) as failed.
    # Preserve those originals and classify only this exact successful-exit case.
    def completed(m):
        return m['status'] == 'success' or (m.get('module') == 'lmms_eval' and m.get('traceback', '').endswith('SystemExit: 0\n'))
    assert len(reports) == len(measurements) >= 2 and all(completed(m) for m in measurements)
    assert sum(m['generation_count'] for m in measurements) == 5000
    seconds = sum(m['generation_seconds'] for m in measurements)
    summary[arm] = {'n': len(scores), 'soft_accuracy': statistics.mean(scores),
                    'stderr': statistics.stdev(scores)/math.sqrt(len(scores)),
                    'generation_seconds': seconds, 'generation_seconds_per_sample': seconds / len(scores),
                    'peak_allocated_bytes': max(m['peak_allocated_bytes'] for m in measurements),
                    'peak_reserved_bytes': max(m['peak_reserved_bytes'] for m in measurements),
                    'sample_files_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    all_samples[arm] = samples
if 'champion' in all_samples and 'base' in all_samples:
    for champion, base in zip(all_samples['champion'], all_samples['base']):
        assert champion['submission']['question_id'] == base['submission']['question_id']
    deltas = [a['exact_match'] - b['exact_match'] for a,b in zip(all_samples['champion'], all_samples['base'])]
    summary['paired_delta'] = statistics.mean(deltas)
    summary['paired_delta_stderr_question_level'] = statistics.stdev(deltas)/math.sqrt(len(deltas))
    if summary['base']['generation_seconds'] > 0:
        summary['generation_latency_ratio'] = summary['champion']['generation_seconds']/summary['base']['generation_seconds']
summary['metric_note'] = 'Standard TextVQA soft accuracy, not binary exact-match accuracy.'
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
