"""Export the screened cases; requires pyarrow (pip install pyarrow)."""
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

root = Path(__file__).resolve().parent
source = json.loads((root / 'cases.json').read_text())
rows = [{
    'id': c['id'],
    'question': c['question'],
    'answer': c['original_reference'],
    'injection': c['injection'],
    'injection_after_tokens': 20,
    'revised_answer': c['updated_reference'],
    'revised_answer_calculation': c['updated_calculation'],
    'source_issue': c['source_issue'],
    'review_note': c['review_note'],
    'screening_seed': c['screening_seed'],
    'screening_non_eos_tokens': c['screening_non_eos_tokens'],
    'screening_complete': c['screening_complete'],
    'first_span_token_ids': c['first_span_token_ids'],
} for c in source['cases']]
schema = pa.schema([
    ('id', pa.string()), ('question', pa.string()), ('answer', pa.string()),
    ('injection', pa.string()), ('injection_after_tokens', pa.int32()),
    ('revised_answer', pa.string()), ('revised_answer_calculation', pa.string()),
    ('source_issue', pa.bool_()), ('review_note', pa.string()),
    ('screening_seed', pa.int64()), ('screening_non_eos_tokens', pa.int32()),
    ('screening_complete', pa.bool_()), ('first_span_token_ids', pa.list_(pa.field('element', pa.int32()))),
], metadata={
    b'screening_provenance': json.dumps(source['metadata']).encode(),
    b'answer_definition': b'Original GSM8K numeric reference, preserved verbatim; may contain source errors. See source_issue and review_note.',
    b'revised_answer_definition': b'Manually derived numeric reference after applying injection.',
    b'injection_timing': b'After exactly 20 generated assistant tokens, not prompt tokens or words.',
})
table = pa.Table.from_pylist(rows, schema=schema)
output = root / 'gsm8k50_injections.parquet'
pq.write_table(table, output, compression='zstd')
loaded = pq.read_table(output)
assert loaded.equals(table, check_metadata=True)
assert loaded.num_rows == 50
assert len({r['id'] for r in rows}) == 50
assert all(r['injection_after_tokens'] == 20 and r['screening_non_eos_tokens'] > 50 for r in rows)
assert all(r[k] for r in rows for k in ('question', 'answer', 'injection', 'revised_answer'))
print(f'Validated {loaded.num_rows} rows and {loaded.num_columns} columns: {output}')
