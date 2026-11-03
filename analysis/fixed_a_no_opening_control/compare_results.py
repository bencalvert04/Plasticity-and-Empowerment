import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
root=Path(__file__).resolve().parents[1]
natural=root/'qwen06b_original50_fixed_a_no_opening'
forced=root/'qwen06b_original50_fixed_a_calculation'
n=pq.read_table(natural/'rollouts.parquet').to_pylist();f=pq.read_table(forced/'rollouts.parquet').to_pylist()
key=lambda x:(x['rlvr_step'],x['id'],x['rollout_index'])
fm={key(x):x for x in f}
assert len(n)==len(f)==len({key(x) for x in n})==1500
for x in n:
 y=fm[key(x)]
 for field in ['question','injection','seed','a_token_ids','input_terms_bits']:
  assert x[field]==y[field],field
 assert x['b_opening_text']=='' and x['b_opening_token_ids']==[]
 assert x['status']=='complete' and len(x['b_token_ids'])==20
print('All 1,500 evaluations complete; matched question, injection, seed, A tokens and input scores across conditions.')
data=[json.load(open(p/'checkpoint_statistics.json')) for p in (natural,forced)]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':'#172B4D','axes.labelcolor':'#526177','xtick.color':'#526177','ytick.color':'#526177','svg.fonttype':'none'})
fig,axs=plt.subplots(1,2,figsize=(11,5.8))
for ax,metric,title in zip(axs,['injection','empowerment'],['Plasticity: correction → B','Empowerment: A → B']):
 allr=[]
 for d,label,color,style in zip(data,['No supplied opening','“The calculation is”'],['#3568B0','#D78132'],['-','--']):
  r=sorted([x for x in d['means'] if x['metric']==metric],key=lambda x:x['rlvr_step']);allr+=r
  ax.errorbar([x['rlvr_step'] for x in r],[x['mean'] for x in r],yerr=[[x['mean']-x['ci95_low'] for x in r],[x['ci95_high']-x['mean'] for x in r]],color=color,ls=style,lw=2.2,marker='o',ms=6,capsize=4,label=label)
 ax.set_title(title,loc='left',fontsize=14,fontweight='bold',pad=16)
 ax.set_xticks([0,77,251]);ax.set_xlim(-25,276);ax.set_xlabel('Additional GRPO steps',labelpad=9)
 ax.set_ylabel('Entropy difference (bits/token)');ax.axhline(0,color='#9DAABD',ls=':',lw=1)
 ax.grid(axis='y',color='#E5EBF2');ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
 ax.legend(frameon=False,fontsize=9,loc='upper right')
 low=min(x['ci95_low'] for x in allr);high=max(x['ci95_high'] for x in allr);gap=high-low
 ax.set_ylim(low-.12*gap,high+.45*gap)
fig.suptitle('Does the supplied B opening change the pattern?',x=.075,y=.97,ha='left',fontsize=19,fontweight='bold')
fig.text(.075,.875,'Qwen3-0.6B • Same 50 questions, 500 reference A spans, corrections and sampling seeds',fontsize=10,color='#526177')
fig.text(.075,.10,'Each condition: 1,500 evaluations. Scores cover 20 new B tokens; all evaluations completed.',fontsize=9,color='#526177')
fig.text(.075,.06,'Bars: pointwise 95% question-bootstrap intervals; panels use separate y-axis scales.',fontsize=9,color='#526177')
fig.text(.075,.022,'Working entropy-ablation measures. B trajectories can differ between conditions and checkpoints.',fontsize=9,color='#526177')
fig.subplots_adjust(left=.075,right=.98,top=.76,bottom=.24,wspace=.27)
fig.savefig(natural/'opening_comparison.png',dpi=220,facecolor='white');fig.savefig(natural/'opening_comparison.svg',facecolor='white')
