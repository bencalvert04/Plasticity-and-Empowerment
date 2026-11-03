import json,sys,hashlib,statistics
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from sampled_a_normal150_experiment import summaries,windows,WINDOWS
out=root/'runs/qwen06b_sampled_a_normal_150'
r=pq.read_table(out/'rollouts.parquet').to_pylist();s=pq.read_table(out/'question_means.parquet').to_pylist();c=json.loads((out/'config.json').read_text())
assert len(r)==1500 and len({(x['rlvr_step'],x['id'],x['trial']) for x in r})==1500
for n,h in c['source_hashes'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
for x in r:
 assert x['a_seed']==x['seed']+10**12
 assert x['included_primary']==(x['a_complete'] and not x['b_ended_at_eos'] and x['b_content_token_count']==150)
 assert len(x['b_token_ids'])==len(x['b_entropies'])<=150
 eos={x['b_token_ids'][-1]} if x['b_ended_at_eos'] else set()
 expected=windows(dict(token_ids=x['b_token_ids'],entropies=x['b_entropies']),eos)
 for k,v in expected.items():assert x[k]==v or (v is not None and abs(x[k]-v)<1e-10),k
assert summaries(r)==s
steps=[0,77,251];coverage={};means=[]
for step in steps:
 g=[x for x in r if x['rlvr_step']==step];eligible=[x for x in g if x['included_primary']]
 coverage[step]=dict(attempted=len(g),included=len(eligible),excluded=len(g)-len(eligible),questions_represented=len({x['id'] for x in eligible}),a_early_eos=sum(not x['a_complete'] for x in g),b_early_eos=sum(x['b_ended_at_eos'] for x in g))
for w in WINDOWS:
 for metric in ['plasticity','empowerment']:
  key=w+'_'+metric+'_bits_per_token_mean'
  for step in steps:
   values=np.array([x[key] for x in s if x['rlvr_step']==step and x[key] is not None]);rng=np.random.default_rng(20261102)
   boot=values[rng.integers(0,len(values),(20000,len(values)))].mean(axis=1) if len(values) else []
   low,high=np.quantile(boot,[.025,.975]) if len(boot) else (None,None)
   means.append(dict(window=w,metric=metric,step=step,mean=float(values.mean()) if len(values) else None,low=float(low) if low is not None else None,high=float(high) if high is not None else None,n_questions=len(values)))
common=set.intersection(*[{x['id'] for x in r if x['rlvr_step']==step and x['included_primary']} for step in steps])
result=dict(coverage=coverage,means=means,common_question_count=len(common),common_question_ids=sorted(common),method='Only complete 150-token B responses, equal weight per represented question. Pointwise 95% bootstrap intervals (20000 question resamples). Checkpoints may contain different questions and selected trials; descriptive comparisons subject to length selection.',elapsed_inference_hours=sum(x['elapsed_seconds'] for x in r)/3600)
(out/'window_statistics.json').write_text(json.dumps(result,indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(12.5,5.7));colors=['#3466ad','#de8730','#278976'];labels=['B tokens 1–50','B tokens 51–100','B tokens 101–150']
for ax,metric in zip(axs,['plasticity','empowerment']):
 for w,col,label in zip(WINDOWS,colors,labels):
  v=[next(x for x in means if x['window']==w and x['metric']==metric and x['step']==step) for step in steps]
  y=np.array([x['mean'] for x in v]);lo=np.array([x['low'] for x in v]);hi=np.array([x['high'] for x in v])
  ax.errorbar(steps,y,yerr=np.stack([y-lo,hi-y]),color=col,label=label,marker='o',capsize=4,lw=2)
 ax.axhline(0,color='#8896a8',ls=':',lw=1);ax.set_xticks(steps);ax.set_xlabel('Additional GRPO steps');ax.set_title('Plasticity: correction → B' if metric=='plasticity' else 'Empowerment proxy: A → B',fontweight='bold');ax.set_ylabel('Entropy difference (bits/token)');ax.grid(axis='y',alpha=.18);ax.legend(fontsize=9)
fig.suptitle('Free A • original corrections • complete 150-token B responses',fontsize=16,fontweight='bold',y=.98)
coverage_text='   |   '.join(f"Step {k}: {v['included']}/500 retained, {v['questions_represented']}/50 questions" for k,v in coverage.items())
fig.text(.07,.085,coverage_text,fontsize=10)
fig.text(.07,.035,'Bars: pointwise 95% question-bootstrap intervals. Early EOS excluded from every window.\nSelected populations differ across checkpoints; these are entropy-ablation proxies, not formal directed information.',fontsize=9,color='#4b5563')
fig.tight_layout(rect=[0,.15,1,.93]);fig.savefig(out/'normal150_window_trends.png',dpi=180);plt.close(fig)
print(json.dumps(result,indent=2))
