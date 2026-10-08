"""Check every prepared seed-1 label, prompt and order against submitted preprocessing."""
import hashlib
import json
import pathlib
import sys

from datasets import Dataset, load_from_disk
from prepare_textvqa import build_question, choose_answer

out = pathlib.Path(sys.argv[1])
files = sorted(pathlib.Path('/storage/data/shiyd2023/datasets/textvqa/default/train').glob('*.parquet'))
# Read only metadata columns; images are not decoded or duplicated.
raw = Dataset.from_parquet([str(p) for p in files], columns=['question_id', 'question', 'answers', 'ocr_tokens']).shuffle(seed=1)
prepared_path = '/public/home/shiyd2023/shiyd/parameter-golf/data/prepared_textvqa_qwen3vl_seed1'
prepared = load_from_disk(prepared_path).select_columns(['question_id', 'user_text', 'target_answer'])
assert len(raw) == len(prepared) == 34602
digest = hashlib.sha256()
for i, (source, cached) in enumerate(zip(raw, prepared)):
    expected = {'question_id': source['question_id'], 'user_text': build_question(source, False, 16), 'target_answer': choose_answer(source['answers'])}
    assert cached == expected, (i, cached, expected)
    digest.update(json.dumps(expected, ensure_ascii=False, sort_keys=True).encode())
result = {'status': 'pass', 'seed': 1, 'rows': len(raw), 'prepared_path': prepared_path,
          'verified': ['all question IDs and shuffled row order', 'all answer labels', 'all prompts; OCR disabled'],
          'canonical_metadata_sha256': digest.hexdigest(),
          'image_note': 'Uses the existing prepared image data for identical question IDs; images are not regenerated.'}
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
