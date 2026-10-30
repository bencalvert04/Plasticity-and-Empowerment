"""Select first 50 seeded, shuffled test questions with >50 non-EOS tokens."""
import sys, json, random, ssl, hashlib
from pathlib import Path
from urllib.request import urlopen
import certifi
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core import Experiment, MODEL

root = Path(__file__).resolve().parent
url = 'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl'
raw = urlopen(url, context=ssl.create_default_context(cafile=certifi.where())).read()
data = [json.loads(line) for line in raw.splitlines()]
order = random.Random(42).sample(range(len(data)), len(data))
exp = Experiment()
selected = []
previous = {r['id']: r for r in map(json.loads, (root/'screening.jsonl').read_text().splitlines())} if (root/'screening.jsonl').exists() else {}
with (root/'screening.jsonl').open('a') as log:
    for position, index in enumerate(order):
        seed = 42000 + index
        case_id = f'test:{index}'
        if case_id in previous:
            row = previous[case_id]
            result = row
            count, accepted = row['non_eos_tokens'], row['accepted']
        else:
            torch.manual_seed(seed)
            result = exp.run({'full': exp.chat(data[index]['question'])}, 51)
            count = sum(t not in exp.eos for t in result['tokens'])
            accepted = count > 50
            row = dict(id=case_id, seed=seed, non_eos_tokens=count,
                       accepted=accepted, **result)
            log.write(json.dumps(row)+'\n'); log.flush()
        if accepted:
            selected.append(dict(id=row['id'], question=data[index]['question'],
                                 original_reference=data[index]['answer'].rsplit('####',1)[-1].strip(),
                                 screening_seed=seed, screening_non_eos_tokens=count,
                                 screening_complete=result['complete'],
                                 first_span_token_ids=result['tokens'][:20]))
        print(f"screened={position+1} accepted={len(selected)} {row['id']} tokens={count} complete={result['complete']}", flush=True)
        if len(selected)==50:
            break
(root/'selected.json').write_text(json.dumps(dict(metadata=dict(model=MODEL,
    model_commit=getattr(exp.model.config,'_commit_hash',None), source_url=url,
    source_sha256=hashlib.sha256(raw).hexdigest(), split='test', shuffle_seed=42,
    response_seed='42000 + zero-based test index', sampling='full softmax, temperature 1',
    prompt='original question only, Qwen chat template', screening_cap=51, initial_screening_cap=512,
    criterion='more than 50 non-EOS generated tokens; capped responses are lower bounds',
    pause_tokens=20, note='Stops once >50 is established; initial rows may include longer responses. Length-conditioned sample; no guarantee for independent generations.'),
    cases=selected), indent=2)+'\n')
