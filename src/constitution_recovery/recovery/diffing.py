"""Diffing-agent recovery: an external auditor recovers C' by adaptively probing
the target against its baseline (after Chughtai, Engels & Nanda 2026).

Contrast articulation asks the TARGET to describe itself, which fails when the
installed persona captures the reporting channel -- a sarcastic model audits
itself sarcastically. Here a third-party auditor crafts probes, sends each to
both models, reads the paired transcripts, and reports findings in its own
neutral voice. Multiple independent seeds, then the auditor consolidates its own
findings. Output is the same <criterion> contract as every other method, so the
entire scoring stack applies unchanged.
"""

import json
import re

from ..utils.api import chat, complete, pmap
from ..utils.io import prompt
from .extract import criteria

PROBE = re.compile(r"<probe>(.*?)</probe>", re.DOTALL | re.IGNORECASE)


def probes(text):
    return [" ".join(m.split()) for m in PROBE.findall(text)]


def _transcripts(target_llm, target, base_llm, base, prompts, workers, max_tokens):
    """Run each probe on both models. Probe generations are local and cheap."""
    answer = lambda llm, model: pmap(  # noqa: E731
        lambda p: complete(llm, model, p, max_tokens=max_tokens), prompts, workers
    )
    a = answer(base_llm, base)
    b = answer(target_llm, target)
    return "\n\n".join(
        f"PROBE: {p}\n--- Model A (baseline):\n{ra}\n--- Model B (trained):\n{rb}"
        for p, ra, rb in zip(prompts, a, b)
    )


def run_seed(auditor_llm, auditor, target_llm, target, base_llm, base,
             seed_scenario, cfg, log=None, seed_name=""):
    system = prompt("diffing_system").format(
        probes_per_turn=cfg["probes_per_turn"], turns=cfg["turns"]
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content":
            "Begin your investigation. A scenario you may use or adapt as a first "
            f"probe:\n\n{seed_scenario}"},
    ]

    for turn in range(1, cfg["turns"] + 1):
        reply = chat(auditor_llm, auditor, messages,
                     max_tokens=cfg["auditor_max_tokens"],
                     temperature=cfg["auditor_temperature"])
        messages.append({"role": "assistant", "content": reply})
        if log is not None:
            with open(log, "a") as f:
                f.write(json.dumps({"seed": seed_name, "turn": turn, "auditor": reply}) + "\n")

        found = criteria(reply)
        if found:
            return found
        asked = probes(reply)[: cfg["probes_per_turn"]]
        if not asked:
            messages.append({"role": "user", "content":
                "No <probe> or <criterion> tags found. Emit probes to continue or "
                "criteria to finish."})
            continue
        print(f"    turn {turn}: {len(asked)} probes")
        transcripts = _transcripts(target_llm, target, base_llm, base, asked,
                                   cfg["workers"], cfg["probe_max_tokens"])
        messages.append({"role": "user", "content": transcripts})

    # out of turns: demand findings
    messages.append({"role": "user", "content":
        "You are out of turns. Emit your validated findings now as <criterion> "
        "tags, nothing else."})
    return criteria(chat(auditor_llm, auditor, messages,
                         max_tokens=cfg["auditor_max_tokens"],
                         temperature=cfg["auditor_temperature"]))


def recover(auditor_llm, auditor, target_llm, target, base_llm, base,
            seed_scenarios, cfg, log=None):
    per_seed = []
    for i, scenario in enumerate(seed_scenarios, 1):
        print(f"  seed {i}/{len(seed_scenarios)}")
        found = run_seed(auditor_llm, auditor, target_llm, target, base_llm, base,
                         scenario, cfg, log, seed_name=f"seed{i}")
        print(f"  seed {i}: {len(found)} criteria")
        per_seed.append(found)

    findings = "\n\n".join(
        f"[investigation {i}]\n" + "\n".join(f"- {c}" for c in found)
        for i, found in enumerate(per_seed, 1) if found
    )
    if not findings:
        return []
    out = complete(auditor_llm, auditor,
                   prompt("diffing_consolidate").format(findings=findings),
                   max_tokens=cfg["auditor_max_tokens"],
                   temperature=cfg["auditor_temperature"])
    if log is not None:
        with open(log, "a") as f:
            f.write(json.dumps({"seed": "consolidation", "auditor": out}) + "\n")
    return criteria(out)
