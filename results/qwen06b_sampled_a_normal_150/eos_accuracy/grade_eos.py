"""Review labels for EOS-only answers in the normal-correction B150 run.
Final answer correctness against revised_answer; reasoning need not be correct.
Labels reviewed from saved response conclusions, not number presence.
Known source_issue questions excluded. No model generation.
"""
import json,csv,hashlib
from pathlib import Path
import pyarrow.parquet as pq
base=Path(__file__).resolve().parents[1]
correct={
0:{'51':[5,6],'457':[5],'285':[1,2,3,4,5,6,7,9],'1116':[0,1,2,3,5,6,7,8,9],'178':[7],'864':[8,9],'191':[1,5,6,9],'447':[2,7,9],'476':[3,4,5,9],'1034':[0,1,2,4,5,6,7,8,9],'1232':[4,8],'1149':[2,6],'569':[1,3,5,6,9],'865':[1],'1291':list(range(10)),'318':[2,4],'778':[1,3,4,5,7,8,9],'735':[0],'704':[0],'88':[1,4],'1098':[0,1,3,4,7,8],'775':[1,2,5,8,9]},
77:{'457':[5],'285':[3,6,7,8],'1116':[3,4,5],'476':[0,6,7],'1034':[6],'1149':[7,9],'569':[3],'1098':[4]},
251:{'285':[5,8]}}
rows=pq.read_table(base/'rollouts.parquet').to_pylist();out=[]
for x in rows:
 if not x['b_ended_at_eos']:continue
 status='excluded_source_issue' if x['source_issue'] else ('correct' if x['trial'] in correct[x['rlvr_step']].get(x['id'].split(':')[1],[]) else 'incorrect')
 note='Final answer only; intermediate reasoning not scored.'
 if x['id']=='test:569' and x['rlvr_step']==0 and x['trial']==7:note='Says Mitchell made 37 fewer, rather than 37 more: incorrect direction despite matching magnitude.'
 if x['id']=='test:285' and x['rlvr_step']==0 and x['trial']==0:note='Mentions brother age 23 in reasoning but explicitly concludes answer 13: incorrect.'
 out.append(dict(step=x['rlvr_step'],id=x['id'],trial=x['trial'],revised_answer=x['revised_answer'],status=status,note=note,question=x['question'],injection=x['injection'],b_text=x['b_text']))
# Confirm every manually marked correct item exists and is included.
for step,questions in correct.items():
 for q,trials in questions.items():
  for t in trials:assert any(x['step']==step and x['id']=='test:'+q and x['trial']==t and x['status']=='correct' for x in out)
summary=[]
for step in [0,77,251]:
 g=[x for x in out if x['step']==step];graded=[x for x in g if x['status']!='excluded_source_issue'];n=sum(x['status']=='correct' for x in graded)
 summary.append(dict(step=step,eos_responses=len(g),source_issue_excluded=len(g)-len(graded),graded=len(graded),correct=n,incorrect=len(graded)-n,accuracy=n/len(graded),distinct_graded_questions=len({x['id'] for x in graded})))
result=dict(run='qwen06b_sampled_a_normal_150',method='Review of EOS responses against corrected targets. Final-answer correctness only; numeric magnitude with wrong direction is incorrect. Known source_issue records excluded. Conditional on EOS within 150 B tokens; not overall accuracy.',source_sha256=hashlib.sha256((base/'rollouts.parquet').read_bytes()).hexdigest(),summary=summary)
(base/'eos_accuracy/summary.json').write_text(json.dumps(result,indent=2)+'\n')
with (base/'eos_accuracy/graded_responses.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print(json.dumps(result,indent=2))
