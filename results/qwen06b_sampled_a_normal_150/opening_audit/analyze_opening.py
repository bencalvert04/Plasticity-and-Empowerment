"""Exploratory text-rule audit of opening tokens; not semantic ground truth."""
import re,json,csv,statistics
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
base=Path(__file__).resolve().parents[1]
tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',revision='c1899de289a04d12100db370d81485cdf75e47ca',local_files_only=True)
num=r'(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?'
# Explicit numerical operations, not a lone fraction or a numeral in a heading.
atom=r'(?:\$?\d+(?:,\d{3})*(?:\.\d+)?)'
expr=re.compile(atom+r'\s*(?:\\times|\\cdot|\\div|[+×÷*]|\s-\s)\s*\$?\d|'+atom+r'\s*/\s*'+atom+r'\s*=|\\frac\{[^{}]+\}\{[^{}]+\}\s*=')
# Detect equations with a numerical result as another arithmetic start marker.
eq=re.compile(r'[^\n]{0,60}=\s*(?:\\boxed\{)?\$?\d')
cue=re.compile(r'\b(?:corrected|correction|instead of|not\s+\$?\d|you (?:meant|mean)|updated|revised|great (?:catch|point)|clarify|clarification)\b',re.I)
words={'ten':'10','twenty':'20'}
rows=[]
for x in pq.read_table(base/'rollouts.parquet').to_pylist():
 text=x['b_text'];ids=x['b_token_ids'];prefixes=[tok.decode(ids[:i],skip_special_tokens=True,clean_up_tokenization_spaces=False) for i in range(len(ids)+1)]
 encoded=tok(text,return_offsets_mapping=True,add_special_tokens=False)
 offsets=encoded['offset_mapping']
 reencoded_match=encoded['input_ids']==ids[:len(encoded['input_ids'])] and len(ids)-len(encoded['input_ids']) in (0,1)
 if not all(text.startswith(p) for p in prefixes):
  # Token offsets via re-encoding are not substituted when byte decoding is unstable.
  stable=False
 else:stable=True
 stable=stable or reencoded_match
 matches=list(expr.finditer(text))+list(eq.finditer(text));start=min((m.start() for m in matches),default=len(text));found=bool(matches)
 # Separate sentence/bullet units; retain original character locations.
 segments=[m for m in re.finditer(r'[^\n.!?]+(?:[.!?](?!\d)|\n|$)',text[:start]) if m.group().strip()]
 newpart=x['injection'].split(', not ')[0];numbers=re.findall(num,newpart)
 newval=numbers[-1].replace(',','') if numbers else next((v for w,v in words.items() if re.search(r'\b'+w+r'\b',newpart)),None)
 ack=[];revised=[]
 for m in segments:
  val=m.group();hasnew=bool(newval and any(v.replace(',','')==newval for v in re.findall(num,val)))
  if cue.search(val):ack.append(m.span())
  anchors={w.rstrip('s') for w in re.findall(r'[a-z]+',newpart.lower()) if w not in {'i','meant','the','a','an','is','was','has','had','of','per','by','on','in','at','for','to','not','with','each','more','than','their','there','are','it'}}
  sentencewords={w.rstrip('s') for w in re.findall(r'[a-z]+',val.lower())}
  if hasnew and anchors & sentencewords:revised.append(m.span())
 def indices(spans):
  if reencoded_match:return [i for i,(u,v) in enumerate(offsets) if v>u and any(u<b and v>a for a,b in spans)]
  return [i for i in range(len(ids)) if stable and len(prefixes[i+1])>len(prefixes[i]) and any(len(prefixes[i])<b and len(prefixes[i+1])>a for a,b in spans)]
 before=indices([(0,start)]);a=indices(ack);v=indices(revised)
 rows.append(dict(step=x['rlvr_step'],id=x['id'],trial=x['trial'],stable=stable,arithmetic_detected=found,opening_tokens=len(before) if stable else None,explicit_acknowledgment_tokens=len(a) if stable else None,revised_value_sentence_tokens=len(v) if stable else None,acknowledgment_text=' | '.join(text[a:b].strip() for a,b in ack),revised_value_sentence_text=' | '.join(text[a:b].strip() for a,b in revised),opening_text=text[:start],injection=x['injection']))
summary={}
for step in [0,77,251]:
 group=[x for x in rows if x['step']==step];valid=[x for x in group if x['stable']];det=[x for x in valid if x['arithmetic_detected']]
 summary[step]=dict(responses=len(group),aligned=len(valid),explicit_arithmetic_detected=len(det))
 for key in ['explicit_acknowledgment_tokens','revised_value_sentence_tokens']:
  values=[x[key] for x in valid];nonzero=[v for v in values if v]
  summary[step][key]=dict(mean=statistics.mean(values),median=statistics.median(values),responses_with_match=len(nonzero),mean_when_present=statistics.mean(nonzero) if nonzero else None)
 values=[x['opening_tokens'] for x in det];summary[step]['opening_before_detected_arithmetic']=dict(mean=statistics.mean(values),median=statistics.median(values))
with (base/'opening_audit/response_annotations.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(base/'opening_audit/summary.json').write_text(json.dumps(dict(method='Exploratory regex estimates, not semantic annotation. Counts original B token IDs overlapping text units before first detectable numerical arithmetic. Explicit acknowledgment uses correction language; revised-value sentences may restate the problem or mention another quantity. No detectable arithmetic is censored at observed B length; opening mean uses only detected cases. Includes early EOS.',summary=summary),indent=2))
print(json.dumps(summary,indent=2))
for step in [0,77,251]:
 print('SAMPLES',step)
 for x in [r for r in rows if r['step']==step and r['trial']==0][:5]:print(x['id'],x['opening_tokens'],x['explicit_acknowledgment_tokens'],x['revised_value_sentence_tokens'],repr(x['acknowledgment_text']),repr(x['revised_value_sentence_text']))
