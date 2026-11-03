import sys,json,gc,time,os
from pathlib import Path
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer,AutoModelForCausalLM
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from sampled_rlvr_experiment import continuation_prefix
out=Path(__file__).resolve().parent
instruction=('Solve the problem using the current information. Perform only the arithmetic needed '
             'and give the numerical answer. Do not restate the problem, repeat the correction, '
             'or add introductory text.')
spec=json.loads((root/'configs/qwen06b_gsm8k200.json').read_text());checkpoints=spec['checkpoints']
questions=pq.read_table(root/spec['source']).to_pylist()[:5]
refs={x['id']:x for x in pq.read_table(root/'results/qwen06b_gsm8k200_20rollouts/rollouts.parquet').to_pylist()
      if x['rlvr_step']==0 and x['rollout_index']==0}
config=dict(instruction=instruction,checkpoints=checkpoints,ids=[q['id'] for q in questions],
            trials=1,fixed_a='saved baseline trial 0',temperature=1.0,cap=None,
            stopping='EOS or model context capacity; capacity endings are censored',seed=20261030)
(out/'config.json').write_text(json.dumps(config,indent=2))
rows=[];torch.set_num_threads(4)
device='mps' if torch.backends.mps.is_available() else 'cpu'
with (out/'responses.jsonl').open('w') as log:
 for step,repo,rev in checkpoints:
  tok=AutoTokenizer.from_pretrained(repo,revision=rev,local_files_only=True)
  model=AutoModelForCausalLM.from_pretrained(repo,revision=rev,local_files_only=True,
        dtype=torch.float16 if device=='mps' else torch.float32).to(device).eval()
  endings=model.generation_config.eos_token_id
  endings=tok.eos_token_id if endings is None else endings
  eos={endings} if isinstance(endings,int) else set(endings)
  for i,q in enumerate(questions):
   seed=20261030+step*10000000+i*10000
   prefix=continuation_prefix(tok,q['question'],q['injection']+'\n\n'+instruction,
                              refs[q['id']]['a_token_ids'],spec['chat_mode'])
   rng=torch.Generator(device='cpu').manual_seed(seed);tokens=[];cache=None;start=time.monotonic()
   with torch.inference_mode():
    while len(prefix)+len(tokens)<model.config.max_position_embeddings:
     ids=[tokens[-1]] if tokens else prefix
     result=model(torch.tensor([ids],device=device),past_key_values=cache,use_cache=True,logits_to_keep=1)
     cache=result.past_key_values
     p=result.logits[0,-1].float().softmax(-1)
     if not torch.isfinite(p).all():raise ValueError('Nonfinite probabilities')
     t=int(torch.multinomial(p.cpu(),1,generator=rng));tokens.append(t)
     if t in eos:break
   ended=bool(tokens and tokens[-1] in eos)
   row=dict(id=q['id'],rlvr_step=step,question=q['question'],injection=q['injection'],seed=seed,
            a_token_ids=refs[q['id']]['a_token_ids'],b_token_ids=tokens,
            b_text=tok.decode(tokens,skip_special_tokens=True),ended_at_eos=ended,
            content_tokens=len(tokens)-int(ended),elapsed_seconds=time.monotonic()-start)
   rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
   print(f"step={step} problem={i+1}/5 B_tokens={row['content_tokens']} EOS={ended}",flush=True)
  del model;gc.collect()
  if device=='mps':torch.mps.empty_cache()
summary={str(step):dict(n=len(g:=[r for r in rows if r['rlvr_step']==step]),
             mean_tokens=sum(r['content_tokens'] for r in g)/len(g),
             lengths=[r['content_tokens'] for r in g],eos_count=sum(r['ended_at_eos'] for r in g))
         for step,_,_ in checkpoints}
summary['overall_mean_tokens']=sum(r['content_tokens'] for r in rows)/len(rows)
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
