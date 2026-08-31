"""Proxy (c), token level: KL between P_C(x_t | x_<t, s) and P_C'(x_t | x_<t, s).

Both constitutions are teacher-forced on the *same* continuation, so positions
align by construction and the comparison is a real per-token distribution rather
than a single forced choice. Forward passes rather than a served API, so the KL
is exact over the whole vocabulary.

The continuation comes from the pair set: text neither constitution produced,
which keeps the measurement neutral between them.
"""

import torch
import torch.nn.functional as F


def response_logprobs(model, tok, constitution, scenario, response, max_tokens, device):
    """log P(. | x_<t, s) at each position that predicts a response token."""
    prefix = tok.apply_chat_template(
        [{"role": "system", "content": constitution}, {"role": "user", "content": scenario}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,  # newer transformers return a BatchEncoding by default,
                            # which cannot be concatenated with the prefill ids
    )
    resp = tok(response, add_special_tokens=False).input_ids[:max_tokens]
    ids = torch.tensor([prefix + resp], device=device)

    with torch.no_grad():
        logits = model(ids).logits[0]

    # position i predicts token i+1, so the response starts at len(prefix)-1
    start = len(prefix) - 1
    return F.log_softmax(logits[start : start + len(resp)].float(), dim=-1)


def score(model, tok, device, c_text, cprime_text, pairs,
          max_response_tokens=512, entropy_quantile=0.75):
    kls, entropies = [], []
    for i, pair in enumerate(pairs, 1):
        for side in ("a", "b"):
            lp_c = response_logprobs(
                model, tok, c_text, pair["scenario"], pair[side],
                max_response_tokens, device,
            )
            lp_cp = response_logprobs(
                model, tok, cprime_text, pair["scenario"], pair[side],
                max_response_tokens, device,
            )
            p_c = lp_c.exp()
            kls.append((p_c * (lp_c - lp_cp)).sum(-1).cpu())
            entropies.append((-p_c * lp_c).sum(-1).cpu())
        if i % 25 == 0:
            print(f"  {i}/{len(pairs)} pairs")

    kl = torch.cat(kls)
    ent = torch.cat(entropies)
    # Most positions are function words no constitution can influence, so the
    # plain mean is diluted. Decision points are where the model was choosing.
    decision = kl[ent >= ent.quantile(entropy_quantile)]

    return {
        "mean_kl": kl.mean().item(),
        "median_kl": kl.median().item(),
        "p90_kl": kl.quantile(0.90).item(),
        "max_kl": kl.max().item(),
        "mean_kl_decision_points": decision.mean().item(),
        "entropy_quantile": entropy_quantile,
        "n_positions": kl.numel(),
        "n_decision_points": decision.numel(),
    }
