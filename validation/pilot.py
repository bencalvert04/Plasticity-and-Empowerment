import json
import random
import time
from pathlib import Path
import torch
from core import ANSWER, Experiment, answer_match
from experiments import gsm8k

exp=Experiment()
rows=[]
start=time.perf_counter()
# Identical contexts must have zero scores under actual model inference.
ctx=exp.chat('What is 2 + 3?'+ANSWER)
identity=exp.run({'full':ctx,'identical':ctx},10,greedy=True)
assert all(abs(v)<1e-6 for pair in identity['scores']['identical'] for v in pair)
print('Real-model identical-context control: zero entropy and JS differences',flush=True)
for i,q in enumerate(random.Random(42).sample(gsm8k(),6)):
    torch.manual_seed(42+i)
    result=exp.prompt(q['question']+ANSWER,384)
    found=answer_match(result['text']) if result['complete'] else None
    reference=q['answer'].split('####')[-1].strip().replace(',','')
    answer=found['number'].replace(',','') if found else None
    correct=(float(answer)==float(reference)) if answer is not None else None
    rows.append(dict(id=q['id'],seed=42+i,reference=reference,correct=correct,**result))
    print('GSM8K',i+1,'tokens',len(result['tokens']),'complete',result['complete'],
          'answer',answer,'correct',correct,'means',result['answer_means'],flush=True)
case=json.load(open('injection_cases.example.json'))['cases'][0]
injections=[]
for seed in [42,43,44]:
    for label,correction in [('changed',case['injection']),('unchanged','Each sprint is still 60 meters.'),('empty','')]:
        random.seed(seed)
        torch.manual_seed(seed)
        result=exp.inject(case['question'],correction,32)
        injections.append(dict(seed=seed,condition=label,**result))
        if label=='empty' and 'continuation' in result:
            assert all(abs(v)<1e-6 for pair in result['continuation']['scores']['injection'] for v in pair)
        print('Injection',seed,label,'pause',result['pause'],'count',result.get('scored_tokens'),
              'means',result.get('window_means',result.get('skipped')),flush=True)
output=dict(gsm8k=rows,injections=injections,elapsed_seconds=time.perf_counter()-start,
            identity_control=identity)
Path('validation/pilot_results.json').write_text(json.dumps(output,indent=2))
print('Saved pilot; seconds',output['elapsed_seconds'],flush=True)
