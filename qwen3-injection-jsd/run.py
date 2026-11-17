"""Paired free-running JSD pilot, one record per question/checkpoint/injection."""
import argparse, json, random, sys, time
from pathlib import Path
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINTS = [(0,'Qwen/Qwen3-0.6B','c1899de289a04d12100db370d81485cdf75e47ca'),(77,'cmpatino/qwen-grpo-r2','628c96470d7289309a6fe9b1292b8039e7b66794'),(177,'cmpatino/qwen-grpo-r4-s100','0bad7bcdee411993f477325262ac64b90ecaa2c7'),(251,'cmpatino/qwen-grpo-r5','06c0dac913ec3ca17e3a6418c136be7bf8fcedb8')]
INJECTIONS = ['who won the super-bowl in 2016','who is the president of the united states?','I meant 30','also, there are 10 houses','the dog is brown']
DATA_REV='740312add88f781978c0658806c59bc2815b9866'

def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',default='qwen3-injection-jsd/results.jsonl'); p.add_argument('--max-b',type=int,default=100); p.add_argument('--limit',type=int,default=5); p.add_argument('--steps',type=int,nargs='+',default=[0,77,177,251]); p.add_argument('--device',default='mps'); args=p.parse_args()
 path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
 done=set()
 if path.exists():
  for line in path.read_text().splitlines():
   row=json.loads(line)
   if row['max_b'] != args.max_b: raise SystemExit('Existing output has different max-b')
   done.add((row['step'],row['gsm8k_test_index'],row['injection']))
 data=hf_hub_download('openai/gsm8k','main/test-00000-of-00001.parquet',repo_type='dataset',revision=DATA_REV)
 questions=pq.read_table(data,columns=['question']).column('question').to_pylist(); indices=random.Random(0).sample(range(len(questions)),5)[:args.limit]
 print('Questions:',indices,flush=True)
 for step,repo,rev in CHECKPOINTS:
  if step not in args.steps: continue
  if all((step,i,x) in done for i in indices for x in INJECTIONS): continue
  print('Loading',step,repo,flush=True)
  tok=AutoTokenizer.from_pretrained(repo,revision=rev); tok.padding_side='left'
  model=AutoModelForCausalLM.from_pretrained(repo,revision=rev,dtype=torch.float32 if args.device=='cpu' else torch.float16).to(args.device).eval()
  eos=model.generation_config.eos_token_id; eos=set(eos if isinstance(eos,list) else [eos]); pad=tok.pad_token_id
  with path.open('a',encoding='utf8') as out:
   for index in indices:
    if all((step,index,x) in done for x in INJECTIONS): continue
    start=time.time(); question=questions[index]
    prompt=tok.apply_chat_template([{'role':'user','content':question}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
    with torch.inference_mode():
     a_input=torch.tensor([prompt],device=args.device); a_result=model.generate(a_input,attention_mask=torch.ones_like(a_input),max_new_tokens=150,do_sample=False,pad_token_id=pad)
    a=a_result[0,len(prompt):].tolist(); a_ended=bool(a and a[-1] in eos)
    if a_ended: a=a[:-1]
    texts=['']+INJECTIONS
    prefixes=[prompt+a+tok.encode('<|im_end|>\n<|im_start|>user\n'+s+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_special_tokens=False) for s in texts]
    width=max(map(len,prefixes)); batch=torch.tensor([[pad]*(width-len(x))+x for x in prefixes],device=args.device); mask=torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in prefixes],device=args.device)
    with torch.inference_mode():
     result=model.generate(batch,attention_mask=mask,max_new_tokens=args.max_b,do_sample=False,pad_token_id=pad,return_dict_in_generate=True,output_logits=True)
    ids=result.sequences[:,width:].tolist(); lengths=[]; ended=[]
    for seq in ids:
     n=next((j for j,t in enumerate(seq) if t in eos),len(seq)); lengths.append(n); ended.append(n<len(seq))
    curves=[[] for _ in INJECTIONS]
    for t,logits in enumerate(result.logits):
     # Compare raw full-softmax predictive distributions in bits before selecting token t.
     logp=F.log_softmax(logits.float(),dim=-1); logm=torch.logaddexp(logp[0:1],logp[1:])-0.6931471805599453
     v=0.5*((logp[0:1].exp()*(logp[0:1]-logm)).sum(-1)+(logp[1:].exp()*(logp[1:]-logm)).sum(-1))/0.6931471805599453
     for k,val in enumerate(v.tolist()):
      if t<min(lengths[0],lengths[k+1]): curves[k].append(val)
    for k,injection in enumerate(INJECTIONS):
     if (step,index,injection) in done: continue
     curve=curves[k]; row=dict(step=step,model=repo,revision=rev,dataset_revision=DATA_REV,sample_seed=0,gsm8k_test_index=index,question=question,injection=injection,control='',output_a=tok.decode(a),output_a_token_ids=a,output_a_ended_early=a_ended,control_b=tok.decode(ids[0][:lengths[0]]),injected_b=tok.decode(ids[k+1][:lengths[k+1]]),control_ended=ended[0],injected_ended=ended[k+1],control_length=lengths[0],injected_length=lengths[k+1],compared_tokens=len(curve),jsd_bits=curve,mean_jsd_bits=sum(curve)/len(curve) if curve else None,max_b=args.max_b,decoding='greedy',thinking_requested=False,device=args.device)
     out.write(json.dumps(row,ensure_ascii=False)+'\n'); out.flush()
    print(f'step {step} question {index} A={len(a)} B={lengths} elapsed={time.time()-start:.1f}s',flush=True)
  del model
  if args.device=='mps': torch.mps.empty_cache()
if __name__=='__main__': main()
