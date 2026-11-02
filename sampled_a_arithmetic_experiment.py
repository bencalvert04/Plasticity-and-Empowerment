"""Sampled-A, arithmetic-only correction experiment: B windows 1-25, 26-50, 1-50."""
import argparse,gc,hashlib,json,platform,time,statistics
from pathlib import Path
from importlib.metadata import version
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer,AutoModelForCausalLM
from sampled_rlvr_experiment import chat,continuation_prefix,load_reference_a
from structured_calculation_experiment import generate_b,region_summary

INSTRUCTION=('Solve the problem using the current information. Perform only the arithmetic needed '
             'and give the numerical answer. Do not restate the problem, repeat the correction, '
             'or add introductory text.')
WINDOWS={'early_1_25':(0,25),'later_26_50':(25,50),'full_1_50':(0,50)}

def contexts(tokenizer,q,a,mode):
    return [continuation_prefix(tokenizer,q['question'],message+'\n\n'+INSTRUCTION,span,mode)
            for span,message in [(a,q['injection']),(a,''),([],q['injection'])]]

def windows(b,eos):
    content=[i for i,t in enumerate(b['token_ids']) if t not in eos]
    result={}
    for name,(start,end) in WINDOWS.items():
        complete=len(content)>=end
        result[name+'_complete']=complete
        for key,value in region_summary(b['entropies'],content[start:end] if complete else []).items():
            result[name+'_'+key]=value
    return result

def summaries(rows):
    groups={}
    for r in rows:groups.setdefault((r['rlvr_step'],r['id']),[]).append(r)
    output=[]
    for (step,i),group in sorted(groups.items()):
        s=dict(rlvr_step=step,id=i,recorded=len(group),early_eos=sum(x['b_ended_at_eos'] for x in group),
               a_early_eos=sum(not x['a_complete'] for x in group),b_attempted=sum(x['b_attempted'] for x in group))
        for name in WINDOWS:
            s[name+'_n']=sum(x[name+'_complete'] for x in group)
            for m in ['plasticity','empowerment']:
                key=name+'_'+m+'_bits_per_token'
                values=[x[key] for x in group if x[key] is not None]
                s[key+'_mean']=statistics.mean(values) if values else None
                # Matched population for comparing early vs later windows.
                joint=[x[key] for x in group if x['full_1_50_complete']]
                s[key+'_joint50_mean']=statistics.mean(joint) if joint else None
        output.append(s)
    return output

def save(rows,directory,metadata):
    for name,records in [('rollouts',rows),('question_means',summaries(rows))]:
        t=pa.Table.from_pylist(records).replace_schema_metadata(metadata)
        p=directory/(name+'.tmp.parquet');pq.write_table(t,p);p.replace(directory/(name+'.parquet'))

def generate_trajectory(model,tok,q,mode,eos,seed):
    a_seed=seed+1000000000000
    a=generate_b(model,[chat(tok,q['question'],chat_mode=mode)],eos,a_seed,20)
    complete=a['content_token_count']==20
    if complete:
        b=generate_b(model,contexts(tok,q,a['token_ids'],mode),eos,seed,50)
    else:
        b=dict(token_ids=[],entropies=[],ended_at_eos=False,content_token_count=0)
    return a,b,a_seed,complete

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limit',type=int,default=50);p.add_argument('--rollouts',type=int,default=10)
    p.add_argument('--seed',type=int,default=20261030)
    p.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'runs/qwen06b_sampled_a_arithmetic_50')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true');args=p.parse_args()
    root=Path(__file__).parent;spec_path=root/'configs/qwen06b_gsm8k200.json';spec=json.loads(spec_path.read_text())
    source=root/spec['source'];questions=pq.read_table(source).to_pylist()
    if not 1<=args.limit<=len(questions) or args.rollouts<1:p.error('Invalid case/trial count')
    questions=questions[:args.limit]
    if any(q['injection_after_tokens']!=20 for q in questions):p.error('Expected 20-token A')
    if args.dry_run:
        print(f'Validated {len(questions)} questions, {args.rollouts} fresh A samples per question/checkpoint, {len(spec["checkpoints"])} checkpoints. No inference.');return
    hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
            ['sampled_a_arithmetic_experiment.py','structured_calculation_experiment.py','sampled_rlvr_experiment.py','configs/qwen06b_gsm8k200.json']}
    cfg=dict(protocol='sampled_a_arithmetic_b50_v1',specification=spec,questions=len(questions),rollouts=args.rollouts,
             seed=args.seed,instruction=INSTRUCTION,windows=WINDOWS,a_tokens=20,b_cap=50,
             eos_policy='Allowed; EOS included in prediction entropy but excluded from averages; incomplete windows null',
             source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),a_generation='fresh from each checkpoint, question only',a_seed_offset=1000000000000,
             source_hashes=hashes,python=platform.python_version(),packages={n:version(n) for n in ['torch','transformers','pyarrow']},
             device='mps' if torch.backends.mps.is_available() else 'cpu',temperature=1.0,filtering=None,
             metric='Entropy ablations, bits/token; source conditions full, empty correction, empty A')
    directory=args.output_dir;encoded=json.dumps(cfg,sort_keys=True);cp=directory/'config.json'
    if cp.exists():
        if not args.resume or cp.read_text()!=encoded:p.error('Resume requires identical code/settings')
    elif directory.exists() and any(directory.iterdir()):p.error('Output directory is not empty')
    directory.mkdir(parents=True,exist_ok=True);cp.write_text(encoded)
    metadata={b'experiment_config':encoded.encode()};path=directory/'rollouts.parquet'
    rows=pq.read_table(path).to_pylist() if path.exists() else [];done={(x['rlvr_step'],x['id'],x['trial']) for x in rows}
    torch.set_num_threads(4)
    reference_tokenizer=AutoTokenizer.from_pretrained(spec['checkpoints'][0][1],revision=spec['checkpoints'][0][2])
    for step,repo,rev in spec['checkpoints']:
        if all((step,q['id'],j) in done for q in questions for j in range(args.rollouts)):continue
        tok=AutoTokenizer.from_pretrained(repo,revision=rev)
        if tok.get_vocab()!=reference_tokenizer.get_vocab():raise ValueError('Vocabulary mismatch')
        for q in questions:
            a=[]
            if chat(tok,q['question'],chat_mode=spec['chat_mode'])!=chat(reference_tokenizer,q['question'],chat_mode=spec['chat_mode']):raise ValueError('Initial prompt differs')
            if contexts(tok,q,a,spec['chat_mode'])!=contexts(reference_tokenizer,q,a,spec['chat_mode']):raise ValueError('Context token IDs differ')
        model=AutoModelForCausalLM.from_pretrained(repo,revision=rev,dtype=torch.float16 if cfg['device']=='mps' else torch.float32).to(cfg['device']).eval()
        e=model.generation_config.eos_token_id;e=tok.eos_token_id if e is None else e;eos={e} if isinstance(e,int) else set(e or [])
        if not eos:raise ValueError('EOS required')
        for i,q in enumerate(questions):
            for j in range(args.rollouts):
                if (step,q['id'],j) in done:continue
                start=time.monotonic();seed=args.seed+step*10000000+i*10000+j
                a,b,a_seed,a_complete=generate_trajectory(model,tok,q,spec['chat_mode'],eos,seed)
                rows.append(dict(id=q['id'],source_row=i,question=q['question'],injection=q['injection'],
                                 revised_answer=q['revised_answer'],source_issue=q['source_issue'],
                                 rlvr_step=step,model_repo=repo,model_revision=rev,trial=j,seed=seed,
                                 a_seed=a_seed,a_complete=a_complete,a_token_ids=a['token_ids'],a_ended_at_eos=a['ended_at_eos'],
                                 a_content_token_count=a['content_token_count'],b_attempted=a_complete,
                                 status='a_early_eos' if not a_complete else ('b_early_eos' if b['ended_at_eos'] else 'full50'),
                                 a_text=tok.decode(a['token_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False),
                                 b_text=tok.decode(b['token_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False),
                                 **{'b_'+k:v for k,v in b.items()},**windows(b,eos),elapsed_seconds=time.monotonic()-start))
                done.add((step,q['id'],j))
                if (j+1)%5==0:save(rows,directory,metadata)
            save(rows,directory,metadata)
            print(f'Step {step}, question {i+1}/{len(questions)}: {len(rows)} evaluations saved',flush=True)
        del model;gc.collect()
        if cfg['device']=='mps':torch.mps.empty_cache()
    save(rows,directory,metadata);print(f'Complete: {len(rows)} evaluations',flush=True)
if __name__=='__main__':main()
