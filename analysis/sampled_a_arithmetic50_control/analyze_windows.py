import json,hashlib,statistics,itertools,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import ttest_rel
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from sampled_a_arithmetic_experiment import windows,summaries
from analyze_checkpoint_statistics import holm
out=root/'runs/qwen06b_sampled_a_arithmetic_50'
r=pq.read_table(out/'rollouts.parquet').to_pylist();s=pq.read_table(out/'question_means.parquet').to_pylist()
c=json.load(open(out/'config.json'))
for n,h in c['source_hashes'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
assert len(r)==1500 and len({(x['rlvr_step'],x['id'],x['trial']) for x in r})==1500
refs={(x['id'],x['rollout_index']):x for x in pq.read_table(root/'results/qwen06b_gsm8k200_20rollouts/rollouts.parquet').to_pylist() if x['rlvr_step']==0}
for x in r:
 assert x['a_complete'] and len(x['a_token_ids'])==20 and not x['a_ended_at_eos']
 assert x['a_seed']==x['seed']+1000000000000
 assert len(x['b_token_ids'])==len(x['b_entropies'])<=50
 assert x['b_content_token_count']==len(x['b_token_ids'])-int(x['b_ended_at_eos'])
 eos={x['b_token_ids'][-1]} if x['b_ended_at_eos'] else set()
 expected=windows(dict(token_ids=x['b_token_ids'],entropies=x['b_entropies']),eos)
 for k,v in expected.items():
  assert x[k]==v or (v is not None and abs(x[k]-v)<1e-10),k
actual=summaries(r)
assert actual==s
steps=[0,77,251];ids=sorted({x['id'] for x in s});groups={(x['rlvr_step'],x['id']):x for x in s}
means=[];tests=[];missing={}
for w in ['early_1_25','later_26_50','full_1_50']:
 for metric in ['plasticity','empowerment']:
  key=w+'_'+metric+'_bits_per_token_joint50_mean'
  valid=[i for i in ids if all(groups[step,i][key] is not None for step in steps)]
  missing[w+'_'+metric]=[i for i in ids if i not in valid]
  values=np.array([[groups[step,i][key] for i in valid] for step in steps])
  rng=np.random.default_rng(20261101);draws=rng.integers(0,len(valid),size=(50000,len(valid)))
  boot=values[:,draws].mean(axis=2)
  for j,step in enumerate(steps):
   lo,hi=np.quantile(boot[j],[.025,.975]);means.append(dict(window=w,metric=metric,step=step,n_questions=len(valid),mean=float(values[j].mean()),low=float(lo),high=float(hi)))
  for a,b in itertools.combinations(range(3),2):
   lo,hi=np.quantile(boot[b]-boot[a],[.025,.975]);t=ttest_rel(values[b],values[a])
   tests.append(dict(window=w,metric=metric,step_from=steps[a],step_to=steps[b],difference=float((values[b]-values[a]).mean()),low=float(lo),high=float(hi),p_raw=float(t.pvalue)))
for x,p in zip(tests,holm([x['p_raw'] for x in tests])):x['p_holm_18']=float(p)
coverage={str(step):dict(recorded=len(g:=[x for x in r if x['rlvr_step']==step]),early25=sum(x['early_1_25_complete'] for x in g),full50=sum(x['full_1_50_complete'] for x in g),early_eos=sum(x['b_ended_at_eos'] for x in g)) for step in steps}
result=dict(method='Equal question means using joint complete-50 trials for all windows; 50000 question bootstrap draws seed 20261101, paired t tests, Holm correction over 18 checkpoint comparisons. Pointwise CIs. One training lineage.',coverage=coverage,omitted_question_ids=missing,means=means,comparisons=tests)
(out/'window_statistics.json').write_text(json.dumps(result,indent=2)+'\n')
print('Validation passed: raw grid, fresh-A provenance, entropy differences, window summaries and source hashes.')
print(json.dumps(result,indent=2))
