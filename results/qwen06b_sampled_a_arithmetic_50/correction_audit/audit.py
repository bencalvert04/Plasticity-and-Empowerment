import re,json,csv
from pathlib import Path
from collections import Counter
from decimal import Decimal
import pyarrow.parquet as pq
base=Path(__file__).resolve().parents[1]
words={'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'twelve':12,'sixteen':16,'eighteen':18,'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,'eighty':80,'ninety':90,'hundred':100}
pat=r'(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?!\w|\.\d)|\b(?:'+ '|'.join(words)+r')\b'
def nums(text):
 out=[]
 for m in re.finditer(pat,text.lower()):
  v=m.group();out.append((Decimal(words[v]) if v in words else Decimal(v.replace(',','')),m.start(),m.end()))
 return out
rows=[]
for x in pq.read_table(base/'rollouts.parquet').to_pylist():
 new,old=x['injection'].split('not ');newval=nums(new)[-1][0];oldval=nums(old)[0][0]
 found={v for v,a,b in nums(x['b_text'])};hasold=oldval in found;hasnew=newval in found
 category='both' if hasold and hasnew else 'old_only' if hasold else 'revised_only' if hasnew else 'neither'
 rows.append(dict(step=x['rlvr_step'],id=x['id'],trial=x['trial'],old=str(oldval),revised=str(newval),category=category,injection=x['injection'],b_text=x['b_text']))
counts={step:dict(Counter(x['category'] for x in rows if x['step']==step)) for step in [0,77,251]}
(base/'correction_audit/mention_counts.json').write_text(json.dumps(counts,indent=2))
with (base/'correction_audit/response_mentions.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(counts,indent=2))
for step in [0,77,251]:
 print('\nSTEP',step,'RETURN SPEED')
 for x in rows:
  if x['step']==step and x['id']=='test:51':print(x['trial'],x['category'],repr(x['b_text']))
