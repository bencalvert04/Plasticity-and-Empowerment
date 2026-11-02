"""Exploratory arithmetic-token selection from existing B; no inference."""
import sys,json,re,csv
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
root=Path(__file__).resolve().parents[2];out=root/'runs/qwen06b_sampled_a_arithmetic_50/arithmetic15'
num=r'[+-]?(?:\d+(?:,\d{3})*)(?:\.\d+)?'
atom=rf'(?:\$?{num}|\b[xNy]\b)'
op=r'(?:\+|-|\*|×|÷|/|\\times|\\cdot|\\div)'
expression=re.compile(rf'{atom}(?:[ \t]*{op}[ \t]*{atom})+(?:[ \t]*=[ \t]*{atom})?')
latex=re.compile(r'\$\$[\s\S]*?\$\$|\$[^$\n]+\$|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]')
frac=re.compile(r'\\frac\{[^{}]+\}\{[^{}]+\}\s*=\s*'+num)

def spans(text):
 result=[]
 for match in expression.finditer(text):
  value=match.group()
  # A lone fraction such as 1/4 may merely restate a quantity.
  if '=' not in value and len(re.findall(op,value))<2 and not re.search(r'\+|\*|×|÷|\\times|\\cdot|\\div|\s-\s',value):continue
  result.append(match.span())
 for m in latex.finditer(text):
  value=m.group()
  stripped=re.sub(r'\\text\{[^{}]*\}', '', value)
  stripped=re.sub(r'\\[A-Za-z]+', '', stripped)
  if re.search(r'[A-Za-z]', re.sub(r'\b[xNy]\b', '', stripped)):continue
  if re.search(r'\d',value) and (re.search(r'\+|\*|×|÷|\\times|\\cdot|\\div',value) or ('=' in value and re.search(r'/|\\frac|-',value))):result.append(m.span())
 for m in frac.finditer(text):result.append(m.span())
 return sorted(result)

rows=pq.read_table(root/'runs/qwen06b_sampled_a_arithmetic_50/rollouts.parquet').to_pylist()
tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',revision='c1899de289a04d12100db370d81485cdf75e47ca',local_files_only=True)
records=[]
for r in rows:
 ids=r['b_token_ids'];text=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
 prefixes=[tok.decode(ids[:i],skip_special_tokens=True,clean_up_tokenization_spaces=False) for i in range(len(ids)+1)]
 stable=all(text.startswith(p) for p in prefixes)
 regions=spans(text);excluded=[m.span() for m in re.finditer(r'\\text\{[^{}]*\}',text)];eos_index=len(ids)-1 if r['b_ended_at_eos'] else -1
 eligible=[i for i in range(len(ids)) if i!=eos_index and stable and len(prefixes[i+1])>len(prefixes[i]) and any(len(prefixes[i])<b and len(prefixes[i+1])>a for a,b in regions) and not any(len(prefixes[i])<b and len(prefixes[i+1])>a for a,b in excluded)]
 selected=eligible[:15] if len(eligible)>=15 else []
 record=dict(step=r['rlvr_step'],id=r['id'],trial=r['trial'],question=r['question'],injection=r['injection'],b_text=text,
             arithmetic_spans=[text[a:b] for a,b in regions],eligible_token_count=len(eligible),selected_token_indices=selected,
             selected_token_pieces=[tok.decode([ids[i]],skip_special_tokens=True) for i in selected],
             selected_text=''.join(prefixes[i+1][len(prefixes[i]):] for i in selected),
             status='ok' if selected else ('unstable_alignment' if not stable else 'fewer_than_15_arithmetic_tokens'))
 for metric,b in [('plasticity',1),('empowerment',2)]:record[metric]=float(np.mean([r['b_entropies'][i][b]-r['b_entropies'][i][0] for i in selected])) if selected else None
 records.append(record)
(out/'selected_records.json').write_text(json.dumps(records,indent=2)+'\n')
coverage={}
for step in [0,77,251]:
 g=[r for r in records if r['step']==step];valid=[r for r in g if r['status']=='ok']
 coverage[step]=dict(total=len(g),qualifying=len(valid),questions=len({r['id'] for r in valid}))
print('COVERAGE',coverage)
qmeans={}
for r in records:
 if r['status']=='ok':qmeans.setdefault((r['step'],r['id']),[]).append(r)
common=sorted(set.intersection(*[{i for s,i in qmeans if s==step} for step in [0,77,251]]))
print('COMMON QUESTIONS',len(common))
(out/'coverage.json').write_text(json.dumps(dict(coverage=coverage,common_question_ids=common,rule='First 15 original generated token IDs overlapping explicit arithmetic spans, possibly noncontiguous. Requires at least 15 identifiable tokens within observed B<=50. Conservative regex heuristic, not validated semantic arithmetic annotation.'),indent=2))
