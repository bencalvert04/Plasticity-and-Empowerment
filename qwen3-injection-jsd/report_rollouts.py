import html
import json
from collections import defaultdict
from pathlib import Path

p = Path(__file__).parent
rows = [json.loads(line) for line in (p / 'results.jsonl').read_text().splitlines()]
groups = defaultdict(list)
for row in rows:
    groups[(row['step'], row['gsm8k_test_index'])].append(row)


def cell(value):
    """Preserve generated text while keeping it inside a Markdown table cell."""
    return html.escape(str(value), quote=False).replace('|', '&#124;').replace('\n', '<br>')


def stop(ended, cap):
    return 'Ended' if ended else f'{cap}-token cap reached'


with (p / 'rollouts.md').open('w', encoding='utf-8') as f:
    def write(text=''):
        f.write(text + '\n')

    write('# Qwen3-0.6B rollout tables')
    write()
    write('Each table groups one GSM8K question at one training checkpoint. The initial response is shared by the control and all five injections. Model output is reproduced as saved and has not been verified. Initial responses are capped at 150 tokens; continuations are capped at 100. An early end means the model emitted its end-of-turn token. JSD compares each injection with the control over the listed number of shared continuation steps.')
    write()
    for (step, index), group in groups.items():
        first = group[0]
        write(f'## Step {step}, GSM8K test question {index}')
        write()
        write('**Question:** ' + cell(first['question']))
        write()
        write('| Condition | Output | Tokens and stopping | Mean JSD (bits) | Compared steps |')
        write('| --- | --- | --- | ---: | ---: |')
        a_len = len(first['output_a_token_ids'])
        write(f"| Initial response A | {cell(first['output_a'])} | {a_len}; {stop(first['output_a_ended_early'], 150)} | | |")
        write(f"| Control: empty message | {cell(first['control_b'])} | {first['control_length']}; {stop(first['control_ended'], 100)} | | |")
        for row in group:
            mean = row['mean_jsd_bits']
            value = 'N/A' if mean is None else f'{mean:.4f}'
            write(f"| Injection: {cell(row['injection'])} | {cell(row['injected_b'])} | {row['injected_length']}; {stop(row['injected_ended'], 100)} | {value} | {row['compared_tokens']} |")
        write()
