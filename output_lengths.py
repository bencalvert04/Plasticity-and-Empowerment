"""Sample 20 GSM8K questions; count Qwen output tokens, excluding EOS."""
import random
import torch
from core import ANSWER, Experiment
from experiments import gsm8k


@torch.inference_mode()
def main(count=20, limit=2048):
    torch.manual_seed(42)
    exp = Experiment()
    lengths, capped = [], 0
    for i, q in enumerate(random.Random(42).sample(gsm8k(), count), 1):
        ids = torch.tensor([exp.chat(q['question'] + ANSWER)], device=exp.model.device)
        output = exp.model.generate(ids, attention_mask=torch.ones_like(ids), use_cache=True,
                                    do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                    max_new_tokens=limit, pad_token_id=exp.tokenizer.eos_token_id)
        tokens = output[0, ids.shape[1]:].tolist()
        done = tokens[-1] in exp.eos
        lengths.append(len(tokens) - int(done))
        capped += int(not done)
        print(f'{i}/{count} {q["id"]}: {lengths[-1]} tokens' + ('' if done else ' (capped)'), flush=True)
    print(f'Mean: {sum(lengths)/count:.1f} tokens; capped: {capped}/{count} (limit {limit}).')
    if capped:
        print('The mean is a lower bound because some responses hit the cap.')


if __name__ == '__main__':
    main()
