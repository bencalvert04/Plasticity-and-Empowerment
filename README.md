# Plasticity and Empowerment

Experiments on how an initial prompt changes a language model's next-token entropy.

- `directed_information_convergence_to_zero.py`: runs Pythia-70m on 20 fixed prompts and plots the mean per-token entropy change. The current output limit is 125 tokens, with early stopping at EOS.
- `effect_of_prompt_length_on_directed_information.py`: uses five fixed long prompts and prints and plots averages for input lengths from 5 through 49 tokens in steps of 2. Each average is the mean summed entropy change across the five prompt prefixes of that length. Every run uses the same 125-token output limit, with early stopping at EOS. Input lengths exclude BOS.

```sh
python3 -m pip install -r requirements.txt
python3 directed_information_convergence_to_zero.py
python3 effect_of_prompt_length_on_directed_information.py
```

The first run downloads the model. Generation currently uses greedy decoding on CPU with float64 parameters. A commented sampling alternative is included. The graph averages only positions reached by every response.

At each output position, the score is:

```text
H(next token | BOS, previous output)
  - H(next token | BOS, prompt, previous output)
```

Entropies are in bits. Despite the experiment names, this is prompt-induced entropy reduction, not a formal directed-information estimator or a proof of convergence to zero.
