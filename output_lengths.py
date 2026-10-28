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
        result = exp.run({'full': exp.chat(q['question'] + ANSWER)}, limit)
        done = result['complete']
        lengths.append(len(result['tokens']) - int(done))
        capped += int(not done)
        print(f'{i}/{count} {q["id"]}: {lengths[-1]} tokens' + ('' if done else ' (capped)'), flush=True)
    print(f'Mean: {sum(lengths)/count:.1f} tokens; capped: {capped}/{count} (limit {limit}).')
    if capped:
        print('The mean is a lower bound because some responses hit the cap.')


if __name__ == '__main__':
    main()
