"""Exploratory post-hoc B-window analysis; never launches model inference."""
import json,hashlib,itertools,csv,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import ttest_rel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from analyze_checkpoint_statistics import holm
run=root/'runs/qwen06b_sampled_a_arithmetic_50';out=run/'ten_token_windows';out.mkdir(exist_ok=True)
raw=run/'rollouts.parquet';rows=pq.read_table(raw).to_pylist()
assert len(rows)==1500 and len({(r['rlvr_step'],r['id'],r['trial']) for r in rows})==1500
steps=[0,77,251];ids=sorted({r['id'] for r in rows});assert len(ids)==50
retained=[r for r in rows if r['full_1_50_complete']]
assert all(len(r['b_token_ids'])==len(r['b_entropies'])==50 and not r['b_ended_at_eos'] for r in retained)
means=[];tests=[];qmeans=[]
for start in range(0,50,10):
 window=f'{start+1}-{start+10}'
 for metric,baseline in [('plasticity',1),('empowerment',2)]:
  values=[]
  for step in steps:
   group=[]
   for case in ids:
    rs=[r for r in retained if r['rlvr_step']==step and r['id']==case];assert rs
    per_trial=[np.mean([e[baseline]-e[0] for e in r['b_entropies'][start:start+10]]) for r in rs]
    value=float(np.mean(per_trial));group.append(value)
    qmeans.append(dict(window=window,metric=metric,step=step,id=case,n_rollouts=len(rs),mean=value))
   values.append(group)
  values=np.array(values)
  draws=np.random.default_rng(20261101).integers(0,50,size=(50000,50));boot=values[:,draws].mean(axis=2)
  for j,step in enumerate(steps):
   lo,hi=np.quantile(boot[j],[.025,.975]);means.append(dict(window=window,start_token=start+1,end_token=start+10,metric=metric,step=step,n_questions=50,mean=float(values[j].mean()),low=float(lo),high=float(hi)))
  for a,b in itertools.combinations(range(3),2):
   lo,hi=np.quantile(boot[b]-boot[a],[.025,.975]);test=ttest_rel(values[b],values[a])
   tests.append(dict(window=window,metric=metric,step_from=steps[a],step_to=steps[b],difference=float((values[b]-values[a]).mean()),low=float(lo),high=float(hi),p_raw=float(test.pvalue)))
for r,p in zip(tests,holm([r['p_raw'] for r in tests])):r['p_holm_30']=float(p)
# Reconstruct each 50-token mean from five equal-width windows.
previous=json.loads((run/'window_statistics.json').read_text())
for metric in ['plasticity','empowerment']:
 for step in steps:
  reconstructed=np.mean([r['mean'] for r in means if r['metric']==metric and r['step']==step])
  original=next(r['mean'] for r in previous['means'] if r['window']=='full_1_50' and r['metric']==metric and r['step']==step)
  assert abs(reconstructed-original)<1e-10
result=dict(method='Exploratory post-hoc analysis. Same complete-50 trajectories across all five windows within checkpoint. Equal question weights after within-question rollout means. 50,000 paired-question bootstrap draws, seed 20261101. Pointwise 95% intervals; paired t-tests, Holm over all 30 new tests. Not jointly corrected with prior exploratory analyses.',raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
 coverage={str(s):sum(r['rlvr_step']==s for r in retained) for s in steps},means=means,comparisons=tests)
(out/'statistics.json').write_text(json.dumps(result,indent=2)+'\n')
for name,records in [('means',means),('comparisons',tests),('question_means',qmeans)]:
 with (out/(name+'.csv')).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':'#172B4D','axes.labelcolor':'#526177','xtick.color':'#526177','ytick.color':'#526177','svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(12,5.8))
for ax,metric,title in zip(axes,['plasticity','empowerment'],['Plasticity: correction → B','Empowerment: A → B']):
 allr=[]
 for step,color,marker in [(0,'#3568B0','o'),(77,'#D78132','s'),(251,'#25998D','D')]:
  rs=sorted([r for r in means if r['metric']==metric and r['step']==step],key=lambda r:r['start_token']);allr+=rs
  x=np.arange(5);ys=[r['mean'] for r in rs]
  ax.plot(x,ys,color=color,marker=marker,lw=2.2,ms=6,label=f'Step {step}')
  ax.fill_between(x,[r['low'] for r in rs],[r['high'] for r in rs],color=color,alpha=.09,linewidth=0)
 ax.set_title(title,loc='left',fontsize=14,fontweight='bold',pad=16)
 ax.set_xticks(range(5),['1–10','11–20','21–30','31–40','41–50']);ax.set_xlim(-.2,4.2)
 ax.set_xlabel('Generated token positions in B',labelpad=10);ax.set_ylabel('Mean entropy difference (bits/token)')
 ax.axhline(0,color='#8E9DB0',ls=':',lw=1);ax.grid(axis='y',color='#E5EBF2');ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
 ax.legend(frameon=False,fontsize=9,loc='upper right')
 lo=min(r['low'] for r in allr);hi=max(0,max(r['high'] for r in allr));gap=hi-lo
 ax.set_ylim(lo-.1*gap,hi+.35*gap)
fig.suptitle('Where in B do the checkpoint differences appear?',x=.075,y=.97,ha='left',fontsize=19,fontweight='bold')
fig.text(.075,.88,'Fresh A • Arithmetic-only instruction • Five 10-token windows from the saved 1,500-evaluation run',fontsize=10,color='#526177')
fig.text(.075,.12,'Same complete-50 responses in every window: step 0 = 464, step 77 = 500, step 251 = 500; 50 questions each.',fontsize=9,color='#526177')
fig.text(.075,.08,'Shading: pointwise 95% question-bootstrap intervals. Equal question weights; separate panel scales.',fontsize=9,color='#526177')
fig.text(.075,.04,'Exploratory analysis; checkpoint tests adjusted over 30 comparisons. No new model generation.',fontsize=9,color='#526177')
fig.subplots_adjust(left=.075,right=.98,top=.76,bottom=.26,wspace=.25)
fig.savefig(out/'ten_token_trends.png',dpi=220,facecolor='white');fig.savefig(out/'ten_token_trends.svg',facecolor='white')
print('Validated: five window means reconstruct previous full-B means.')
for metric in ['plasticity','empowerment']:
 print(metric)
 for start in range(0,50,10):
  rs=[r for r in means if r['metric']==metric and r['start_token']==start+1]
  test=next(r for r in tests if r['metric']==metric and r['window']==rs[0]['window'] and r['step_from']==0 and r['step_to']==251)
  print(rs[0]['window'],*[round(r['mean'],6) for r in rs], 'endpoint adjusted p=',round(test['p_holm_30'],6))
print('Significant tests:')
for r in tests:
 if r['p_holm_30']<.05:print(r)
