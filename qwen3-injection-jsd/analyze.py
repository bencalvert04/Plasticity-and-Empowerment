import csv, json, statistics
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

p=Path(__file__).parent
rows=[json.loads(s) for s in (p/'results.jsonl').read_text().splitlines()]
groups=defaultdict(list)
for r in rows: groups[(r['step'],r['injection'])].append(r)
with (p/'summary.csv').open('w',newline='') as f:
 w=csv.writer(f); w.writerow(['step','injection','question_count','mean_question_jsd_bits','mean_first_10_jsd_bits','first_10_question_count','min_compared_tokens','mean_compared_tokens','max_compared_tokens'])
 for (step,inj),rs in sorted(groups.items()):
  valid=[r for r in rs if r['mean_jsd_bits'] is not None]; n=[r['compared_tokens'] for r in valid]
  w.writerow([step,inj,len(valid),statistics.mean(r['mean_jsd_bits'] for r in valid) if valid else '',statistics.mean(statistics.mean(r['jsd_bits'][:10]) for r in valid if len(r['jsd_bits']) >= 10) if any(len(r['jsd_bits']) >= 10 for r in valid) else '',sum(len(r['jsd_bits']) >= 10 for r in valid),min(n) if n else '',statistics.mean(n) if n else '',max(n) if n else ''])
fig,ax=plt.subplots(figsize=(10,5.5))
for inj in dict.fromkeys(r['injection'] for r in rows):
 xs=sorted({r['step'] for r in rows if r['injection']==inj}); ys=[statistics.mean(r['mean_jsd_bits'] for r in groups[(x,inj)] if r['mean_jsd_bits'] is not None) for x in xs]
 ax.plot(xs,ys,marker='o',label=inj)
ax.set(xlabel='Cumulative GRPO optimizer steps',ylabel='Mean trajectory JSD (bits)',title='Qwen3-0.6B interruption pilot: 5 GSM8K questions'); ax.set_xticks([0,77,177,251]); ax.grid(alpha=.2); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(p/'jsd-by-checkpoint.png',dpi=180)
print((p/'summary.csv').read_text())
