from pathlib import Path
import time,json,traceback,os
import certifi
os.environ.setdefault("REQUESTS_CA_BUNDLE",certifi.where())
os.environ.setdefault("SSL_CERT_FILE",certifi.where())
from daytona import Daytona,DaytonaConfig,CreateSandboxFromImageParams,Resources
root=Path(__file__).resolve().parents[1];out=root/'daytona_pilot';sandbox=None
client=Daytona(DaytonaConfig(api_key=(Path.home()/'.config/adept-research/daytona_api_key').read_text().strip()))
try:
 print('Creating 4-vCPU / 8-GiB temporary CPU sandbox',flush=True)
 sandbox=client.create(CreateSandboxFromImageParams(image='python:3.11-slim',resources=Resources(cpu=4,memory=8,disk=10),auto_stop_interval=15,auto_delete_interval=60,labels={'purpose':'adept-late-a-cpu-pilot'}),timeout=180)
 (out/'sandbox.json').write_text(json.dumps({'id':sandbox.id,'status':'running','resources':{'cpu':4,'memory_gib':8,'disk_gib':10}},indent=2))
 print('Sandbox created',sandbox.id,flush=True)
 sandbox.process.exec('mkdir -p /home/daytona/research',timeout=30)
 uploads={'daytona_pilot/late_a_pilot.py':'late_a_pilot.py','sampled_rlvr_experiment.py':'sampled_rlvr_experiment.py','structured_calculation_experiment.py':'structured_calculation_experiment.py','experiment_config.py':'experiment_config.py','configs/qwen06b_gsm8k200.json':'qwen06b_gsm8k200.json','data/gsm8k200/gsm8k200_injections.parquet':'gsm8k200_injections.parquet'}
 for src,dst in uploads.items():sandbox.fs.upload_file(str(root/src),'/home/daytona/research/'+dst)
 setup="python -m pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu && python -m pip install --no-cache-dir transformers==5.5.0 pyarrow==26.0.0"
 shell='#!/bin/sh\ncd /home/daytona/research\n('+setup+') > setup.log 2>&1\nif [ $? -ne 0 ]; then echo setup_failed > FAILED; exit 1; fi\npython -u late_a_pilot.py > pilot.log 2>&1\nif [ $? -ne 0 ]; then echo pilot_failed > FAILED; fi\n'
 sandbox.fs.upload_file(shell.encode(),'/home/daytona/research/run.sh')
 sandbox.process.exec('nohup sh /home/daytona/research/run.sh >/home/daytona/research/launcher.log 2>&1 </dev/null &',timeout=30)
 print('Pilot started remotely; provisioning dependencies then evaluating three checkpoints.',flush=True)
 for poll in range(360):
  result=sandbox.process.exec('if test -f DONE; then echo DONE; elif test -f FAILED; then cat FAILED; else echo RUNNING; fi',cwd='/home/daytona/research',timeout=30)
  status=result.result.strip()
  for filename in ['setup.log','pilot.log','pilot_config.json','pilot_results.json']:
   try:sandbox.fs.download_file('/home/daytona/research/'+filename,str(out/filename))
   except Exception:pass
  if status!='RUNNING':
   print('Remote result:',status,flush=True)
   if status!='DONE':raise RuntimeError(status)
   break
  time.sleep(15)
 else:raise TimeoutError('Pilot exceeded 90-minute safety deadline')
 print('Pilot outputs downloaded.',flush=True)
except Exception as exc:
 print('Controller failure:',type(exc).__name__,str(exc).replace((Path.home()/'.config/adept-research/daytona_api_key').read_text().strip(),'[REDACTED]')[:1200],flush=True)
 (out/'failure.json').write_text(json.dumps({'error_type':type(exc).__name__}))
finally:
 if sandbox is not None:
  try:
   sandbox.delete();print('Temporary sandbox deleted.',flush=True)
   (out/'sandbox.json').write_text(json.dumps({'id':sandbox.id,'status':'deleted'},indent=2))
  except Exception as exc:print('SANDBOX CLEANUP FAILED:',type(exc).__name__,flush=True)
