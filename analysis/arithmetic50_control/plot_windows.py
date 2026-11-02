import json,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]/'qwen06b_fixed_a_arithmetic_50'
d=json.loads((root/'window_statistics.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':'#172B4D','axes.labelcolor':'#526177','xtick.color':'#526177','ytick.color':'#526177','svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(12,6))
for ax,metric,title in zip(axes,['plasticity','empowerment'],['Plasticity: correction → B','Empowerment: A → B']):
 allr=[]
 for window,label,color,style,marker in [('early_1_25','Tokens 1–25','#3568B0','-','o'),('later_26_50','Tokens 26–50','#D78132','--','s'),('full_1_50','Tokens 1–50','#25998D',':','D')]:
  r=sorted([x for x in d['means'] if x['metric']==metric and x['window']==window],key=lambda x:x['step']);allr+=r
  xs=[x['step'] for x in r];ys=[x['mean'] for x in r]
  ax.plot(xs,ys,color=color,ls=style,marker=marker,lw=2.3,ms=6,label=label)
  ax.fill_between(xs,[x['low'] for x in r],[x['high'] for x in r],color=color,alpha=.09,linewidth=0)
 ax.set_title(title,loc='left',fontsize=14,fontweight='bold',pad=15)
 ax.axhline(0,color='#8E9DB0',ls=':',lw=1)
 ax.set_xticks([0,77,251]);ax.set_xlim(-20,271);ax.set_xlabel('Additional GRPO steps',labelpad=10)
 ax.set_ylabel('Mean entropy difference (bits/token)');ax.grid(axis='y',color='#E5EBF2');ax.set_axisbelow(True)
 ax.spines[['top','right']].set_visible(False)
 lo=min(x['low'] for x in allr);hi=max(0,max(x['high'] for x in allr));gap=hi-lo
 ax.set_ylim(lo-.1*gap,hi+.45*gap)
 ax.legend(frameon=False,fontsize=9,loc='upper right');ax.tick_params(length=0,pad=5)
fig.suptitle('Arithmetic-only instruction: comparing B token windows',x=.075,y=.97,ha='left',fontsize=18,fontweight='bold')
fig.text(.075,.88,'Qwen3-0.6B • Fixed A • 50 questions × 10 trials × 3 checkpoints = 1,500 evaluations',fontsize=10,color='#526177')
fig.text(.075,.12,'All windows use the same complete-50 responses within each checkpoint: 468 / 500 / 500 retained.',fontsize=9,color='#526177')
fig.text(.075,.08,'Shading: pointwise 95% question-bootstrap intervals. Equal question weights; separate panel scales.',fontsize=9,color='#526177')
fig.text(.075,.04,'The full-window mean averages the two half-window means; it is not independent evidence.',fontsize=9,color='#526177')
fig.text(.075,.008,'Working entropy-ablation measures; these scores do not establish mathematical accuracy or correction adoption.',fontsize=8.5,color='#526177')
fig.subplots_adjust(left=.075,right=.98,top=.76,bottom=.26,wspace=.25)
fig.savefig(root/'arithmetic_token_windows.png',dpi=220,facecolor='white');fig.savefig(root/'arithmetic_token_windows.svg',facecolor='white')
with (root/'window_means.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(d['means'][0]));w.writeheader();w.writerows(d['means'])
