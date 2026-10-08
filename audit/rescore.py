"""Independently rescore saved predictions against validation parquet metadata."""
import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
import math
import pathlib
import statistics

p = argparse.ArgumentParser()
p.add_argument('--gold', type=pathlib.Path, required=True)
p.add_argument('--predictions', type=pathlib.Path, nargs='+', required=True)
p.add_argument('--expected', type=int, default=5000)
p.add_argument('--output', type=pathlib.Path, required=True)
a = p.parse_args()
repo = pathlib.Path(__file__).resolve().parents[1]
normalizer_path = repo / 'lmms-eval/lmms_eval/tasks/_task_utils/vqa_eval_metric.py'
spec = importlib.util.spec_from_file_location('answer_normalizer', normalizer_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
normalize = module.EvalAIAnswerProcessor()

def read_rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        return [json.loads(line) for line in handle if line.strip()]

gold_rows = read_rows(a.gold)
assert len(gold_rows) == 5000
gold = {row['question_id']: row for row in gold_rows}
assert len(gold) == 5000
samples = [row for path in a.predictions for row in read_rows(path)]
assert len(samples) == a.expected
ids = [row['submission']['question_id'] for row in samples]
assert len(set(ids)) == a.expected
doc_ids = [row['doc_id'] for row in samples]
assert len(set(doc_ids)) == a.expected
if a.expected == 5000:
    assert set(ids) == set(gold)
    assert set(doc_ids) == set(range(5000))
rescored = []
for row in samples:
    qid = row['submission']['question_id']
    reference = gold[qid]
    response = row['filtered_resps']
    if isinstance(response, list):
        assert len(response) == 1
        response = response[0]
    answer = normalize(response)
    assert answer == row['submission']['answer']
    answers = [normalize(text) for text in reference['answers']]
    n = len(answers)
    assert n == 10
    matches = answers.count(answer)
    # Closed form of leave-one-annotator-out agreement, independent of submitted loop.
    score = (matches * min(1, max(0, matches-1)/3) + (n-matches)*min(1, matches/3))/n
    assert math.isclose(score, row['exact_match'], abs_tol=1e-12), (qid, score, row['exact_match'])
    rescored.append({'question_id': qid, 'doc_id': row['doc_id'], 'image_id': reference['image_id'],
                     'response': response, 'normalized_answer': answer, 'score': score,
                     'token_counts': row.get('token_counts')})
rescored.sort(key=lambda row: row['doc_id'])
scores = [row['score'] for row in rescored]
encoded = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n' for row in rescored).encode()
output_samples = a.output.with_suffix('.jsonl.gz')
output_samples.parent.mkdir(parents=True, exist_ok=True)
output_samples.write_bytes(gzip.compress(encoded, mtime=0))
summary = {'samples': len(scores), 'unique_images': len(set(row['image_id'] for row in rescored)),
           'soft_accuracy': statistics.mean(scores), 'stderr_question_level': statistics.stdev(scores)/math.sqrt(len(scores)),
           'score_distribution': dict(collections.Counter(scores)), 'all_logged_scores_verified': True,
           'gold_sha256': hashlib.sha256(a.gold.read_bytes()).hexdigest(),
           'predictions_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in a.predictions},
           'rescored_samples_sha256': hashlib.sha256(output_samples.read_bytes()).hexdigest()}
a.output.with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
