"""CPU pilot: completed A, late-A ablation, B positions 51-100."""
import json,time,gc,statistics,hashlib,shutil
from pathlib import Path
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer,AutoModelForCausalLM
from sampled_rlvr_experiment import chat,continuation_prefix
from structured_calculation_experiment import generate_b,region_summary

def late_contexts(tok,q,a,mode):
    return [continuation_prefix(tok,q['question'],j,span,mode) for span,j in [(a,q['injection']),(a,''),(a[:74],q['injection'])]]

def main():
 torch.set_num_threads(4)
 root=Path(__file__).resolve().parent
 spec=json.loads((root/'qwen06b_gsm8k200.json').read_text())
 qs=pq.read_table(root/'gsm8k200_injections.parquet').to_pylist()
 q=qs[2];rows=[]
 config=dict(a_min_content_tokens=75,a_safety_cap=2048,a_source_positions='75 through final non-EOS token',b_scored_positions='51-100 inclusive',b_cap=100,temperature=1.0,device='cpu',dtype='float32',question_id=q['id'],pilot='one question, one trial per checkpoint; timing only, not research conclusions',empowerment_baseline='Retain A tokens 1-74; remove late A. Previous B tokens identical in all contexts.',metric='entropy ablation proxies, not formal DI')
 (root/'pilot_config.json').write_text(json.dumps(config,indent=2))
 for step,repo,rev in spec['checkpoints']:
  start=time.monotonic();tok=AutoTokenizer.from_pretrained(repo,revision=rev)
  model=AutoModelForCausalLM.from_pretrained(repo,revision=rev,dtype=torch.float32,cache_dir=str(root/'model_cache')).to('cpu').eval()
  loading=time.monotonic()-start;e=model.generation_config.eos_token_id or tok.eos_token_id;eos={e} if isinstance(e,int) else set(e)
  seed=20261030+step*10000000+2*10000
  t=time.monotonic();a=generate_b(model,[chat(tok,q['question'],chat_mode=spec['chat_mode'])],eos,seed+10**12,2048);a_seconds=time.monotonic()-t
  content=[v for v in a['token_ids'] if v not in eos]
  eligible=a['ended_at_eos'] and len(content)>=75
  row=dict(step=step,repo=repo,revision=rev,id=q['id'],question=q['question'],injection=q['injection'],seed=seed,a_ended_at_eos=a['ended_at_eos'],a_content_tokens=len(content),a_token_ids=a['token_ids'],a_text=tok.decode(content,skip_special_tokens=True),a_eligible=eligible,loading_seconds=loading,a_seconds=a_seconds)
  if eligible:
   t=time.monotonic();b=generate_b(model,late_contexts(tok,q,content,spec['chat_mode']),eos,seed,100)
   complete=b['content_token_count']>=100
   row.update(b_seconds=time.monotonic()-t,b_text=tok.decode(b['token_ids'],skip_special_tokens=True),b=b,b_window_complete=complete,**region_summary(b['entropies'],list(range(50,100)) if complete else []))
  rows.append(row);(root/'pilot_results.json').write_text(json.dumps(rows,indent=2))
  print(json.dumps({k:v for k,v in row.items() if k not in ['a_token_ids','a_text','b','question','injection','b_text']}),flush=True)
  del model;gc.collect();shutil.rmtree(root/'model_cache',ignore_errors=True)
 (root/'DONE').write_text('success\n')
if __name__=='__main__':main()
