"""Proxy (c) at the decision point: KL between P_C and P_C' at the single
position where the model commits to a preference.

token_kl averages over every response position, where most tokens are function
words no constitution can influence. This measures the one position that carries
the preference -- and reports it in the same terms as CEI, over the same pairs.

The assistant turn is pre-filled with "<choice>", so the next token *is* the
letter and both constitutions are conditioned on identical text. Reading the
model's own first token instead would let it open "Based on..." under one
constitution and "Response" under the other, leaving the two distributions
conditioned on different prefixes.
"""

import math

import torch
import torch.nn.functional as F

from ..utils.io import prompt

EPS = 1e-6
PREFILL = "<choice>"


def kl_bernoulli(p, q):
    p = min(max(p, EPS), 1 - EPS)
    q = min(max(q, EPS), 1 - EPS)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def letter_ids(tok, letter):
    """Single-token spellings of a letter, so their mass can be summed. Taking a
    max would underweight whichever letter has more variants."""
    ids = {
        enc[0]
        for form in (letter, " " + letter)
        for enc in [tok(form, add_special_tokens=False).input_ids]
        if len(enc) == 1
    }
    if not ids:
        raise ValueError(f"no single-token spelling of {letter!r} in this tokenizer")
    return sorted(ids)


def p_choice_a(logprobs, a_ids, b_ids):
    a, b = logprobs[a_ids].exp().sum(), logprobs[b_ids].exp().sum()
    return (a / (a + b)).item()


def choice_logprobs(model, tok, constitution, pair, device):
    """log P(. | prompt + '<choice>') over the full vocabulary."""
    text = prompt("constitution_judge").format(
        scenario=pair["scenario"], a=pair["a"], b=pair["b"]
    )
    ids = tok.apply_chat_template(
        [{"role": "system", "content": constitution}, {"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,  # newer transformers return a BatchEncoding by default,
                            # which cannot be concatenated with the prefill ids
    ) + tok(PREFILL, add_special_tokens=False).input_ids

    with torch.no_grad():
        logits = model(torch.tensor([ids], device=device)).logits[0, -1]
    return F.log_softmax(logits.float(), dim=-1)


def score(model, tok, device, c_text, cprime_text, pairs):
    a_ids, b_ids = letter_ids(tok, "A"), letter_ids(tok, "B")
    full, bern, pa_c, pa_cp = [], [], [], []

    for i, pair in enumerate(pairs, 1):
        lp_c = choice_logprobs(model, tok, c_text, pair, device)
        lp_cp = choice_logprobs(model, tok, cprime_text, pair, device)
        full.append((lp_c.exp() * (lp_c - lp_cp)).sum().item())
        x, y = p_choice_a(lp_c, a_ids, b_ids), p_choice_a(lp_cp, a_ids, b_ids)
        pa_c.append(x)
        pa_cp.append(y)
        bern.append(kl_bernoulli(x, y))
        if i % 25 == 0:
            print(f"  {i}/{len(pairs)} pairs")

    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    return {
        "kl_full_vocab": mean(full),
        "kl_choice_bernoulli": mean(bern),
        "mean_p_a_under_c": mean(pa_c),
        "mean_p_a_under_cprime": mean(pa_cp),
        "n_pairs": len(pairs),
    }
