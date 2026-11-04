import sys,json,concurrent.futures
sys.path.insert(0,'.')
import daytona_for_ben.controller as C
from pathlib import Path
C.RUN=Path('runs/qwen06b_original50_free_a125_whole_a_b75_125').resolve()
C.STATES.update(json.loads((C.RUN/'status.json').read_text()))
c=C.client();jobs=[]
for j,v in C.STATES.items():
  if v['status']=='verified_complete':continue
  sb=c.get(v['sandbox_id']);sb.set_autostop_interval(0)
  C.update(j,status='resuming',error=None,cleanup_error=None,sandbox_state='running');jobs.append((j,sb))
print('resuming',[j for j,_ in jobs],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
  for f in concurrent.futures.as_completed([pool.submit(C.worker,j,sb,True) for j,sb in jobs]):f.result()
if all(s['status']=='verified_complete' for s in C.STATES.values()):C.combine();print('FULL EXPERIMENT COMPLETE: 1500 attempts verified.',flush=True)
else:print('Some shards require recovery; inspect status.json.',flush=True)
