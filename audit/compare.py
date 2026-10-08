"""Paired confidence intervals, resampling images to respect repeated-image questions."""
import argparse
import gzip
import json
import pathlib

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--left', type=pathlib.Path, required=True)
p.add_argument('--right', type=pathlib.Path, required=True)
p.add_argument('--output', type=pathlib.Path, required=True)
a = p.parse_args()
def load(path):
    with gzip.open(path, 'rt') as f:
        return {row['question_id']: row for row in map(json.loads, f)}
left, right = load(a.left), load(a.right)
assert len(left) == len(right) == 5000 and left.keys() == right.keys()
clusters = {}
for qid, row in left.items():
    assert row['image_id'] == right[qid]['image_id']
    clusters.setdefault(row['image_id'], []).append(row['score']-right[qid]['score'])
sums = np.array([sum(values) for values in clusters.values()])
counts = np.array([len(values) for values in clusters.values()])
rng = np.random.default_rng(1234)
boot = []
for _ in range(100):
    indexes = rng.integers(0, len(clusters), size=(200, len(clusters)))
    boot.extend((sums[indexes].sum(axis=1)/counts[indexes].sum(axis=1)).tolist())
result = {'left': str(a.left), 'right': str(a.right), 'n_questions': 5000, 'n_images': len(clusters),
          'paired_delta': float(sums.sum()/counts.sum()),
          'image_cluster_bootstrap_95_ci': np.quantile(boot, [0.025,0.975]).tolist(),
          'bootstrap_repetitions': len(boot), 'bootstrap_seed': 1234}
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
