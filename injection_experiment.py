"""Interrupt generation with a user correction; measure continuation sensitivity.

Entropy differences and JS are context-ablation proxies, not formal DI or empowerment.
"""
import argparse
import json
import random
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from answer_token_scores import score_numeric_answer
from input_and_output_directed_information import entropy, jensen_shannon

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
CONTINUE = 'Continue solving using the current information. End with Final answer: followed by the numerical answer.'
MARKER = '__INJECTION_EXPERIMENT_ASSISTANT_SPAN_94027__'


def load_cases(path):
    cases = json.loads(Path(path).read_text())['cases']
    if not cases:
        raise ValueError('At least one case is required')
    ids = set()
    for case in cases:
        for key in ('id', 'question', 'injection'):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f'Each case needs a nonempty {key}')
            if MARKER in case[key]:
                raise ValueError('Reserved marker in case')
        if case['id'] in ids:
            raise ValueError('Case IDs must be unique')
        ids.add(case['id'])
    return cases


def initial_context(tokenizer, question):
    return tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
                                        tokenize=True, add_generation_prompt=True, return_dict=False)


def continuation_contexts(tokenizer, question, first_ids, injection):
    """Keep original first-span IDs, including partially decoded byte tokens.

    Empty assistant/user content ablations preserve message roles and turn markers.
    Only injection content or first-span IDs change; the continuation instruction
    remains in every condition. This is a new assistant turn after an interruption.
    """
    def context(include_history, include_injection):
        followup = (injection + '\n\n' if include_injection else '') + CONTINUE
        messages = [{'role': 'user', 'content': question},
                    {'role': 'assistant', 'content': MARKER},
                    {'role': 'user', 'content': followup}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if rendered.count(MARKER) != 1:
            raise ValueError('Chat template does not preserve the assistant span marker')
        before, after = rendered.split(MARKER)
        prefix = tokenizer.encode(before, add_special_tokens=False)
        if prefix != initial_context(tokenizer, question):
            raise ValueError('Chat template prefix differs from initial generation context')
        return prefix + (list(first_ids) if include_history else []) + tokenizer.encode(after, add_special_tokens=False)
    return {'full': context(True, True), 'without_injection': context(True, False),
            'without_first_span': context(False, True)}


def eos_ids(model, tokenizer):
    values = model.generation_config.eos_token_id
    if values is None:
        values = tokenizer.eos_token_id
    values = [values] if isinstance(values, int) else values
    if not values:
        raise ValueError('EOS token IDs are required')
    return set(values)


def predict(model, ids, cache=None):
    past = cache.get_seq_length() if cache is not None else 0
    if past + len(ids) > model.config.max_position_embeddings:
        raise ValueError('Model context limit exceeded')
    result = model(torch.tensor([ids], device=model.device), past_key_values=cache, use_cache=True)
    return result.logits[0, -1], result.past_key_values


def sample_token(logits):
    return torch.multinomial(logits.float().softmax(-1), 1).item()


@torch.inference_mode()
def first_span(model, tokenizer, question, pause_tokens):
    logits, cache = predict(model, initial_context(tokenizer, question))
    tokens = []
    endings = eos_ids(model, tokenizer)
    for _ in range(pause_tokens):
        token = sample_token(logits)
        tokens.append(token)
        if token in endings:
            return tokens, True
        if len(tokens) < pause_tokens:
            logits, cache = predict(model, [token], cache)
    return tokens, False


@torch.inference_mode()
def scored_continuation(model, tokenizer, contexts, max_new_tokens):
    # Each branch must have its own cache and its own position count.
    states = {name: predict(model, ids) for name, ids in contexts.items()}
    scores = {name: [] for name in ('injection_entropy_bits', 'first_span_entropy_bits',
                                   'injection_js_bits', 'first_span_js_bits')}
    tokens = []
    endings = eos_ids(model, tokenizer)
    for _ in range(max_new_tokens):
        full = states['full'][0]
        without_injection = states['without_injection'][0]
        without_span = states['without_first_span'][0]
        h = entropy(full)
        scores['injection_entropy_bits'].append(entropy(without_injection) - h)
        scores['first_span_entropy_bits'].append(entropy(without_span) - h)
        scores['injection_js_bits'].append(jensen_shannon(without_injection, full))
        scores['first_span_js_bits'].append(jensen_shannon(without_span, full))
        token = sample_token(full)
        tokens.append(token)
        if token in endings:
            break
        if len(tokens) < max_new_tokens:
            # Feed the SAME sampled token into all three contexts.
            states = {name: predict(model, [token], cache) for name, (_, cache) in states.items()}
    done = bool(tokens and tokens[-1] in endings)
    return tokens, done, scores


def score_summary(tokenizer, tokens, done, scores, endings):
    indices = [i for i, token in enumerate(tokens) if token not in endings]
    def means(at):
        return {key + '_per_token': sum(values[i] for i in at) / len(at) if at else None
                for key, values in scores.items()}
    answer = score_numeric_answer(tokenizer, tokens, scores['injection_entropy_bits'],
                                  scores['first_span_entropy_bits'], done)
    answer_indices = answer.get('token_indices', [])
    return {'continuation_token_count_excluding_eos': len(indices),
            'continuation_means': means(indices),
            'answer_status': answer['status'], 'answer': answer.get('answer'),
            'answer_token_indices': answer_indices, 'answer_token_count': len(answer_indices),
            'answer_means': means(answer_indices)}


def main(args):
    if min(args.pause_tokens, args.max_new_tokens, args.trials) < 1:
        raise ValueError('Token limits and trial count must be positive')
    cases = load_cases(args.cases)
    if args.sample is not None:
        if not 1 <= args.sample <= len(cases):
            raise ValueError('sample must be between 1 and the number of cases')
        cases = random.Random(args.seed).sample(cases, args.sample)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    if not torch.backends.mps.is_available():
        raise RuntimeError('This runner requires an Apple MPS GPU')
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16,
                                               attn_implementation='sdpa').to('mps').eval()
    if not tokenizer.chat_template:
        raise ValueError('A chat template is required for user-message interruptions')
    # Check all context budgets before starting or creating output.
    for case in cases:
        n = len(initial_context(tokenizer, case['question']))
        future = max(map(len, continuation_contexts(tokenizer, case['question'], [], case['injection']).values()))
        if max(n + args.pause_tokens, future + args.pause_tokens + args.max_new_tokens) > model.config.max_position_embeddings:
            raise ValueError(f"Context budget exceeded for {case['id']}")
    output.mkdir(parents=True)
    (output/'metadata.json').write_text(json.dumps({
        'arguments': vars(args), 'cases': cases, 'continuation_instruction': CONTINUE,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'model_commit': getattr(model.config, '_commit_hash', None),
        'sampling': 'full softmax, temperature 1; float16 model, float32 metrics; per-trial seeds',
        'interpretation': 'Injected-content and earlier-output sensitivity, not formal DI or empowerment',
        'means': 'continuation means exclude EOS; answer means use extracted numerical tokens only',
    }, indent=2))
    with (output/'trials.jsonl').open('w') as file:
        for ci, case in enumerate(cases):
            for trial in range(args.trials):
                seed = args.seed + ci * args.trials + trial
                torch.manual_seed(seed)
                first, early_eos = first_span(model, tokenizer, case['question'], args.pause_tokens)
                row = {'case_id': case['id'], 'question': case['question'], 'injection': case['injection'],
                       'trial': trial + 1, 'seed': seed, 'first_token_ids': first,
                       'first_text': tokenizer.decode(first, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                       'first_ended_with_eos': early_eos}
                if early_eos:
                    row['status'] = 'skipped: initial response ended before interruption'
                else:
                    contexts = continuation_contexts(tokenizer, case['question'], first, case['injection'])
                    tokens, done, scores = scored_continuation(model, tokenizer, contexts, args.max_new_tokens)
                    row.update({'status': 'ok', 'context_token_ids': contexts, 'continuation_token_ids': tokens,
                                'continuation_text': tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                                'continuation_ended_with_eos': done, 'continuation_truncated': not done,
                                'token_scores': scores,
                                **score_summary(tokenizer, tokens, done, scores, eos_ids(model, tokenizer))})
                file.write(json.dumps(row) + '\n')
                file.flush()
                print(f"\n{case['id']} trial {trial + 1}: {row['status']}", flush=True)
                print('Before injection:', row['first_text'], flush=True)
                if row['status'] == 'ok':
                    print('Injection:', case['injection'], flush=True)
                    print('After injection:', row['continuation_text'], flush=True)
                    print(json.dumps({key: row[key] for key in ('continuation_truncated', 'continuation_means',
                                      'answer_status', 'answer', 'answer_token_count', 'answer_means')}, indent=2), flush=True)
    print('Saved:', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', default=str(Path(__file__).with_name('injection_cases.example.json')))
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--sample', type=int, help='Optional random subset of supplied cases')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--pause-tokens', type=int, default=32)
    parser.add_argument('--max-new-tokens', type=int, default=256, help='Continuation cap')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
