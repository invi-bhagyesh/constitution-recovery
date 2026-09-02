"""condA x remorse, recovered by the DIFFING AGENT.

An external auditor (gpt-5-mini) adaptively probes the target against its base
over multi-turn trajectories, 6 independent seeds, then consolidates its own
findings in its own neutral voice. After Chughtai, Engels & Nanda 2026, extended
from finding behavioural differences to recovering the constitution.

REGISTERED PREDICTION (before running). The base-prior leakage that contrast
articulation showed -- 6 of 10 criteria detecting the BASE, because the target
described its own HHH values alongside its installed ones -- should be much
smaller here: the auditor has no base prior of its own to leak, and it only
reports differences it observed between the two models. Predicted: mean
detection accuracy ABOVE contrast's 0.402, and critically FEWER criteria below
0.3. That is the mechanism claim -- self-report leaks the reporter's values, an
external differential audit cannot.

Secondary: the auditor may surface salient non-constitutional side effects
(verbosity, formatting) rather than values -- the DeepMind finding that model
organisms come bundled with more-salient differences. Those would show as
criteria that detect well (high accuracy) but are absent from C.

NOTE the confound when comparing methods on detection: diffing probes
target-vs-base to WRITE C', and detection tests target-vs-base to SCORE it.
Same task shape, so diffing starts with a structural advantage that is not about
recovery quality. Different scenarios on each side (the auditor picks its own
probes; detection uses the persona pool), so it is not strictly circular --
but preference and KL are the fairer cross-method instruments.
"""

RUN_SPEC = {
    "name": "qwen7b-condA-remorse-diffing-sonnet5",
    "stages": [
        "scenarios",           # shared AIRiskDilemmas pool, hub-cached
        "recovery",            # -> criteria.json  (C')
        "persona_scenarios",   # remorse-apt scenarios, hub-cached across methods
        "responses",           # base answers them unsteered / under C / under C'
        "preference",          # -> preference.json  are C and C' interchangeable?
        "detection",           # -> detection.json   can C' spot the trained model?
        "token_kl",            # -> token_kl.json    per-token steering divergence
    ],

    "arm": "condA",
    "method": "diffing",
    "persona": "remorse",

    "models": {},
    "experiment": {},

    "workers": 8,
}
