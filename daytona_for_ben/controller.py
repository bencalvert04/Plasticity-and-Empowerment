"""Nine-sandbox coordinator. Saves progress locally; downloads and verifies before deletion."""
from pathlib import Path
import os,json,time,hashlib,threading,concurrent.futures,subprocess,sys,argparse,uuid
import certifi
os.environ.setdefault('REQUESTS_CA_BUNDLE',certifi.where());os.environ.setdefault('SSL_CERT_FILE',certifi.where())
from daytona import Daytona,DaytonaConfig,CreateSandboxFromImageParams,Resources
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/qwen06b_original50_free_a125_whole_a_b75_125'
KEY=None
LAUNCH_TAG=uuid.uuid4().hex[:8]
LOCK=threading.Lock();STATES={};REMOTE='/home/daytona/research'

def client():
 global KEY
 if KEY is None:
  KEY=os.environ.get('DAYTONA_API_KEY','').strip()
  if not KEY:
   key_file=Path.home()/'.config/adept-research/daytona_api_key'
   if key_file.exists():KEY=key_file.read_text().strip()
  if not KEY:raise RuntimeError('Set DAYTONA_API_KEY or save your key privately in ~/.config/adept-research/daytona_api_key')
 return Daytona(DaytonaConfig(api_key=KEY))
def update(job,**kw):
 with LOCK:
  STATES[job].update(kw);p=RUN/'status.tmp.json';p.write_text(json.dumps(STATES,indent=2));p.replace(RUN/'status.json')
def safe_error(e):
 message=str(e).replace(KEY,'[REDACTED]') if KEY else str(e)
 return (type(e).__name__+': '+message)[:1500]

def download(s,remote,dest,required=False):
 try:
  tmp=dest.with_name(dest.name+'.download');s.fs.download_file(remote,str(tmp));tmp.replace(dest);return True
 except Exception:
  if required:raise
  return False

def verify(job,dest):
 cfg=json.loads((dest/'config.json').read_text());rows=[json.loads(l) for l in (dest/'rollouts.jsonl').read_text().splitlines() if l.strip()]
 assert cfg['protocol']=='whole_a1_125_b75_125_v1'
 assert cfg['a_source_range']==[1,125] and cfg['b_target_range']==[75,125]
 expected={(i,t) for i in range(50) if i%3==cfg['shard'] for t in range(10)}
 assert len(rows)==len(expected) and {(r['source_row'],r['trial']) for r in rows}==expected
 for r in rows:
  assert r['step']==cfg['step']
  for label in ['a','b']:
   v=r[label];assert len(v['token_ids'])==len(v['entropies'])<=125
   assert v['content_token_count']==len(v['token_ids'])-int(v['ended_at_eos'])
  assert r['a_complete']==(r['a']['content_token_count']==125)
  assert r['b_complete']==(r['b']['content_token_count']==125)
  assert r['included_primary']==(r['a_complete'] and r['b_complete'])
  assert r['selected_a_source_count']==(125 if r['a_complete'] else 0)
  assert r['scored_b_count']==(51 if r['included_primary'] else 0)
  if r['included_primary']:
   for label,j in [('plasticity',1),('empowerment',2)]:
    want=sum(e[j]-e[0] for e in r['b']['entropies'][74:125])/51
    assert abs(want-r[label+'_bits_per_token'])<1e-10
  else:assert r['plasticity_bits_per_token'] is None and r['empowerment_bits_per_token'] is None
 (dest/'validated.json').write_text(json.dumps({'records':len(rows),'eligible':sum(r['included_primary'] for r in rows),'sha256':hashlib.sha256((dest/'rollouts.jsonl').read_bytes()).hexdigest()},indent=2))
 return rows

def worker(job,s,resume=False):
 dest=RUN/job;dest.mkdir(exist_ok=True);verified=False
 try:
  s.process.exec('mkdir -p '+REMOTE,timeout=30)
  files={'daytona_for_ben/run_shard.py':'run_shard.py','sampled_rlvr_experiment.py':'sampled_rlvr_experiment.py','structured_calculation_experiment.py':'structured_calculation_experiment.py','experiment_config.py':'experiment_config.py','configs/qwen06b_gsm8k200.json':'qwen06b_gsm8k200.json','data/gsm8k200/gsm8k200_injections.parquet':'gsm8k200_injections.parquet'}
  for src,target in files.items():s.fs.upload_file(str(RUN/'source_snapshot'/src),REMOTE+'/'+target)
  state=STATES[job];step=state['step'];shard=state['shard']
  resume_flag=' --resume' if resume else ''
  if resume:
   s.process.exec('mkdir -p '+REMOTE+'/output',timeout=30)
   for name in ['config.json','rollouts.jsonl']:
    s.fs.upload_file(str(dest/name),REMOTE+'/output/'+name)
  setup='python -m pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu && python -m pip install --no-cache-dir transformers==5.5.0 pyarrow==26.0.0'
  shell=f'#!/bin/sh\ncd {REMOTE}\nmkdir -p output\n({setup}) > setup.log 2>&1\nif [ $? -ne 0 ]; then echo setup_failed > FAILED; exit 1; fi\npython -u run_shard.py --step {step} --shard {shard}{resume_flag} > run.log 2>&1\nif [ $? -ne 0 ]; then echo runner_failed > FAILED; fi\n'
  s.fs.upload_file(shell.encode(),REMOTE+'/start.sh')
  s.process.exec('nohup sh '+REMOTE+'/start.sh >'+REMOTE+'/launcher.log 2>&1 </dev/null &',timeout=30)
  update(job,status='installing_or_running',started_at_unix=time.time());print(job,'remote worker started',flush=True)
  consecutive_timeouts=0
  for poll in range(2880): # 24h watchdog; never silently let charged jobs run indefinitely.
   try:
    probe=s.process.exec('if test -f output/DONE.json; then echo DONE; elif test -f FAILED; then cat FAILED; else echo RUNNING; fi',cwd=REMOTE,timeout=30).result.strip()
    consecutive_timeouts=0
   except Exception as e:
    consecutive_timeouts+=1;update(job,transient_error=safe_error(e),consecutive_timeouts=consecutive_timeouts)
    if consecutive_timeouts>=6:raise
    time.sleep(15);continue
   download(s,REMOTE+'/output/progress.json',dest/'progress.json')
   if (dest/'progress.json').exists():update(job,progress=json.loads((dest/'progress.json').read_text()))
   if poll%5==0 or probe!='RUNNING':
    for f in ['config.json','rollouts.jsonl','DONE.json']:download(s,REMOTE+'/output/'+f,dest/f)
    for f in ['setup.log','run.log']:download(s,REMOTE+'/'+f,dest/f)
   if probe=='DONE':
    for f in ['config.json','rollouts.jsonl','DONE.json']:download(s,REMOTE+'/output/'+f,dest/f,required=True)
    rows=verify(job,dest);verified=True;update(job,status='verified_complete',records=len(rows),eligible=sum(r['included_primary'] for r in rows));break
   if probe!='RUNNING':raise RuntimeError(probe)
   time.sleep(30)
  else:raise TimeoutError('24-hour worker watchdog')
 except Exception as e:
  update(job,status='failed',error=safe_error(e));print(job,safe_error(e),flush=True)
  for f in ['config.json','rollouts.jsonl','progress.json']:download(s,REMOTE+'/output/'+f,dest/f)
  for f in ['setup.log','run.log']:download(s,REMOTE+'/'+f,dest/f)
 finally:
  try:
   if verified:s.delete();update(job,sandbox_state='deleted');print(job,'verified results saved; sandbox deleted',flush=True)
   else:s.stop();update(job,sandbox_state='stopped_for_recovery');print(job,'stopped for recovery; partial results retained',flush=True)
  except Exception as e:update(job,cleanup_error=safe_error(e))

def combine():
 import pyarrow as pa,pyarrow.parquet as pq
 allrows=[]
 for job in STATES:allrows.extend([json.loads(l) for l in (RUN/job/'rollouts.jsonl').read_text().splitlines() if l.strip()])
 assert len(allrows)==1500 and len({(r['step'],r['source_row'],r['trial']) for r in allrows})==1500
 pq.write_table(pa.Table.from_pylist(allrows),RUN/'rollouts.parquet')
 groups={}
 for r in allrows:groups.setdefault((r['step'],r['id']),[]).append(r)
 qs=[]
 for (step,i),g in sorted(groups.items()):
  ok=[r for r in g if r['included_primary']];q=dict(step=step,id=i,attempted=len(g),eligible=len(ok),a_short=sum(not r['a_complete'] for r in g),b_short=sum(r['a_complete'] and not r['b_complete'] for r in g))
  for m in ['plasticity','empowerment']:q[m+'_bits_per_token']=sum(r[m+'_bits_per_token'] for r in ok)/len(ok) if ok else None
  qs.append(q)
 pq.write_table(pa.Table.from_pylist(qs),RUN/'question_means.parquet');summary={}
 for step in [0,77,251]:
  g=[r for r in allrows if r['step']==step];v=[q for q in qs if q['step']==step and q['eligible']]
  summary[step]=dict(attempted=len(g),eligible=sum(r['included_primary'] for r in g),represented_questions=len(v))
  for m in ['plasticity','empowerment']:summary[step][m+'_bits_per_token']=sum(q[m+'_bits_per_token'] for q in v)/len(v) if v else None
 (RUN/'summary.json').write_text(json.dumps(summary,indent=2));(RUN/'COMPLETE').write_text('1500 attempts verified\n')

def main():
 global RUN
 parser=argparse.ArgumentParser(description='Daytona Code For Ben: whole-A removal, 9 CPU workers')
 parser.add_argument('--run-dir',type=Path,default=RUN)
 parser.add_argument('--plan',action='store_true',help='Print protocol and worker plan without contacting Daytona')
 args=parser.parse_args();RUN=args.run_dir.expanduser().resolve()
 if args.plan:
  print(json.dumps({'run_directory':str(RUN),'sandboxes':9,'attempts':1500,'a_source':[1,125],'b_target':[75,125],'scored_b_tokens':51,'system_prompt':None,'a_generation':'fresh per checkpoint/trial','early_eos':'saved, excluded from primary; no replacements'},indent=2));return
 RUN.mkdir(parents=True,exist_ok=True)
 if (RUN/'status.json').exists():raise RuntimeError('Existing launch registry; refusing duplicate. Recover existing run explicitly.')
 files=['daytona_for_ben/run_shard.py','daytona_for_ben/controller.py','sampled_rlvr_experiment.py','structured_calculation_experiment.py','experiment_config.py','configs/qwen06b_gsm8k200.json','data/gsm8k200/gsm8k200_injections.parquet'];hashes={}
 for f in files:
  p=RUN/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);data=(ROOT/f).read_bytes();p.write_bytes(data);hashes[f]=hashlib.sha256(data).hexdigest()
 (RUN/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
 created=[]
 try:
  c=client()
  for step in [0,77,251]:
   for shard in range(3):
    job=f'step{step}_shard{shard}';STATES[job]={'step':step,'shard':shard,'status':'creating','expected':170 if shard<2 else 160};update(job)
    s=c.create(CreateSandboxFromImageParams(name='adept-whole-a-'+job+'-'+LAUNCH_TAG,image='python:3.11-slim',resources=Resources(cpu=4,memory=8,disk=10),auto_stop_interval=30,auto_delete_interval=-1,labels={'purpose':'adept-whole-a125-b75-125','step':str(step),'shard':str(shard)}),timeout=180)
    created.append((job,s));update(job,status='provisioned',sandbox_id=s.id,sandbox_state='running');print(job,'created',flush=True)
 except Exception as e:
  print('Provisioning failed:',safe_error(e),flush=True)
  for job,s in created:
   try:s.delete();update(job,status='provisioning_aborted',sandbox_state='deleted')
   except Exception as ce:update(job,cleanup_error=safe_error(ce))
  (RUN/'PROVISIONING_FAILED.json').write_text(json.dumps({'error':safe_error(e),'experiment_started':False},indent=2));return
 print('All nine sandboxes provisioned. Starting independent shards.',flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
  futures=[pool.submit(worker,job,s) for job,s in created]
  for f in concurrent.futures.as_completed(futures):f.result()
 if all(s['status']=='verified_complete' for s in STATES.values()):combine();print('FULL EXPERIMENT COMPLETE: 1500 attempts verified.',flush=True)
 else:print('Some shards require recovery; inspect status.json. No missing data imputed.',flush=True)
if __name__=='__main__':main()
