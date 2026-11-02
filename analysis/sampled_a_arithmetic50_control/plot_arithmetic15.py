import json,csv,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]/'qwen06b_sampled_a_arithmetic_50/arithmetic15'
r=json.loads((root/'selected_records.json').read_text());coverage=json.loads((root/'coverage.json').read_text())
valid=[x for x in r if x['status']=='ok'];groups={}
for x in valid:groups.setdefault((x['step'],x['id']),[]).append(x)
means=[]
for step in [0,77,251]:
 g=[v for (s,i),v in groups.items() if s==step]
 row=dict(step=step,qualifying_responses=sum(len(v) for v in g),n_questions=len(g))
 for m in ['plasticity','empowerment']:row[m]=statistics.mean(statistics.mean(x[m] for x in v) for v in g)
 means.append(row)
(root/'means.json').write_text(json.dumps(means,indent=2))
with (root/'selected_responses.csv').open('w',newline='') as f:
 fields=['step','id','trial','b_text','selected_text','eligible_token_count','plasticity','empowerment']
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(valid)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':'#172B4D','axes.labelcolor':'#526177','xtick.color':'#526177','ytick.color':'#526177','svg.fonttype':'none'})
fig,axs=plt.subplots(1,3,figsize=(12,5.5));x=np.arange(3);steps=[0,77,251]
colors=['#3568B0','#D78132','#25998D']
axs[0].bar(x,[v['qualifying_responses'] for v in means],color=colors,width=.5)
for xx,row in zip(x,means):axs[0].text(xx,row['qualifying_responses']+3,f"{row['qualifying_responses']}/500",ha='center',fontsize=10,fontweight='bold')
axs[0].set_title('Responses with 15 arithmetic tokens',loc='left',fontsize=11,fontweight='bold',pad=14);axs[0].set_ylabel('Qualifying responses');axs[0].set_ylim(0,125)
for ax,metric,title in zip(axs[1:],['plasticity','empowerment'],['Plasticity: correction → B','Empowerment: A → B']):
 for xx,step,color,row in zip(x,steps,colors,means):
  values=[statistics.mean(a[metric] for a in v) for (s,i),v in groups.items() if s==step]
  jitter=np.random.default_rng(42).uniform(-.13,.13,len(values))
  ax.scatter(xx+jitter,values,color=color,alpha=.35,s=20)
  ax.scatter(xx,row[metric],color=color,marker='D',s=70,edgecolor='white',zorder=3)
  ax.annotate(f"{row[metric]:.4f}",(xx,row[metric]),xytext=(8,7),textcoords='offset points',fontsize=9,color=color,fontweight='bold')
 ax.set_title(title,loc='left',fontsize=11,fontweight='bold',pad=14);ax.set_ylabel('Entropy difference (bits/token)');ax.axhline(0,ls=':',color='#8E9DB0',lw=1)
 ax.margins(y=.25,x=.25)
for ax in axs:
 ax.set_xticks(x,['0','77','251']);ax.set_xlabel('Additional GRPO steps',labelpad=8);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#E5EBF2');ax.set_axisbelow(True);ax.tick_params(length=0,pad=5)
fig.suptitle('First 15 arithmetic-selected B tokens: limited coverage',x=.065,y=.97,ha='left',fontsize=18,fontweight='bold')
fig.text(.065,.88,'Fresh-A run • Original generated token IDs overlapping detected expressions; tokens may be noncontiguous',fontsize=9.5,color='#526177')
fig.text(.065,.13,'Dots: question means. Diamonds: equal-question averages. Qualifying question counts: 29 / 4 / 1.',fontsize=9,color='#526177')
fig.text(.065,.09,'Different question sets qualify at each checkpoint; these are not paired estimates of a training effect.',fontsize=9,color='#526177')
fig.text(.065,.05,'Conservative heuristic for explicit arithmetic within B ≤ 50 tokens. No confidence intervals or significance claims.',fontsize=9,color='#526177')
fig.text(.065,.012,'Selection excludes responses with fewer than 15 detected arithmetic tokens. This does not prove they contain no reasoning.',fontsize=8.5,color='#526177')
fig.subplots_adjust(left=.065,right=.97,top=.75,bottom=.28,wspace=.4)
fig.savefig(root/'arithmetic15_scores.png',dpi=220,facecolor='white');fig.savefig(root/'arithmetic15_scores.svg',facecolor='white')
print(json.dumps(means,indent=2))
for step in steps:
 a=next(x for x in valid if x['step']==step);print('EXAMPLE',step,a['arithmetic_spans'],repr(a['selected_text']))
