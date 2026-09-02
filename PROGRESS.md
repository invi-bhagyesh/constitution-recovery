# Progress log — constitution recovery (Project 2)

Can a constitution installed by character training be recovered from input-output
access alone, and does prior textual exposure (model-spec midtraining) make it
more recoverable? Recovery: contrast articulation. Scoring: CEI (proxy a),
steering KL at the decision position and per token (proxy c), free Kendall-tau.

- Code: github.com/invi-bhagyesh/constitution-recovery
- Shared artifacts (pairs, C labels): HF dataset `invi-bhagyesh/constitution-recovery-data`, fingerprinted
- Registered predictions: `runs/predictions.md` (committed `2842735`, before any style-persona run)
- Evidence per run: `runs/<name>/` -- criteria.json (C'), labels.jsonl, cei.json,
  steering_kl.json, token_kl.json, resolved_config.json, consolidation_raw.txt

## Scoreboard

Goodness, Qwen2.5-7B-Instruct student. condA = OCT only (maius adapter);
condB = model-spec midtraining on C, then OCT (invi-bhagyesh two-stage, merged).
Instrument: 200 AIRiskDilemmas scenarios (start 100, EigenBench slice), pairs from
llama-3.3-70b + gemma-3-27b, judge Sonnet 5 (4.6 fallback on refusals).

| metric | condA-goodness | condB-goodness |
|---|---|---|
| \|C'\| | 14 | 4 (over-compressed; see caveat) |
| CEI | 0.439 | 0.500 |
| forward median R^2 (C' in C's span) | 0.531 | 0.595 |
| reverse median R^2 (C covered by C') | 0.373 | 0.431 |
| uncovered_cprime (bloat) | 0.43 | 0.00 |
| uncovered_c (truncation) | 0.67 | 0.73 |
| mode | unclassified | truncated |
| steering KL, A/B choice (nats) | 0.618 | 0.537 |
| steering KL, full vocab | 0.794 | 0.690 |
| token KL, decision points | 0.232 | 0.225 |
| Kendall tau (aggregate preference, free variant) | 0.697 | 0.742 |

**Caveat before quoting any of this:** the two C' were produced under different
consolidation decoding (condA: unpenalized + exact dedup; condB:
frequency_penalty 1.0, which suppressed the criterion tags themselves and
collapsed 118 -> 4). A symmetric re-run under the current settings (temp 0.7,
penalty 0.2, commit `ed160c1`) is an open item. CEI medians over 4 criteria are
fragile. No measured CEI ceiling exists yet (C vs paraphrase-of-C).

## Findings so far (2026-08-31)

1. **Installation gap is small, direction as predicted, magnitude far below it.**
   condB beats condA on every instrument (CEI +0.06, choice-KL -0.08, full-vocab
   -0.10, zero bloat vs 43%), but the spec predicted condB approaching complete
   recovery; both arms sit truncated at ~1/3 of C's span. Straight reading:
   textual exposure adds a little articulable content, not a lot.
2. **Both arms miss the same half of C.** Recovered: the humanity-welfare
   cluster (C criteria ~7-15). Missing: the behavioral-policy half (charitable
   interpretation, no middle views, legal-interpretation help). condB's raw
   pre-collapse artifact (criteria.v3-118) visibly contained behavioral-policy
   content, so part of its truncation is consolidation artifact, not absence.
3. **Aggregate-preference recovery >> span recovery.** tau ~0.7 next to CEI
   ~0.45-0.50: the recovered subset reproduces most of C's overall pair-ranking
   because C's criteria are correlated on this pair set. Audit implication: a
   truncated constitution passes a preference-agreement check while failing a
   per-criterion one -- the cheap check is fool-able.
4. **Four instruments, one ordering.** Label-space regression, both steering
   KLs, and tau all rank condB slightly closer to C. The measurement stack is
   coherent even where recovery disappoints.
5. **Consolidation is the fragile step for 7B targets.** Observed failure modes:
   exact-string stutter (condA: 273 tags, 14 unique), template-with-rotating-slot
   (condB: 118), tag suppression under heavy frequency penalty (118 -> 4).
   Mitigations now in the pipeline: exact dedup at parse, fixed-point merge,
   raw-output logging, spend guard (judging.max_criteria).

## Incidents worth a methods/limitations sentence

| incident | fix (commit) |
|---|---|
| RunPod nginx squats :8000/:8001, answers POST with 405 | ports 18000/18001 (`a144229`) |
| Sonnet 5 reasoning-by-default burned the judge token budget | reasoning disabled on judge calls (`16193fb`) |
| Judge content-filter refusals on harm-adjacent pairs | Sonnet 4.6 fallback, unparsed counting (`0bb8c6e`) |
| Zero-byte labels file masqueraded as cached; C labels never collected | existence != completeness; done-set is authority (`c1792f3`) |
| transformers BatchEncoding broke KL tokenization | return_dict=False (`b7a0993`) |
| KL stages OOM beside two vLLM servers on 48GB | kill servers post-recovery; named error (`1318166`) |

Total judge spend to date: roughly 9k calls (pairs 400, labels_c 3,000,
condA labels 2,800, condB labels 800, retries/losses ~200) — order of $50.
Recovery/KL compute: local pod, a few GPU-hours.

## Open items

- [ ] condA-sarcasm: contrast scored as-is or skipped (channel captured); diffing spec ready
- [ ] condA-sarcasm contrast run (spec `b6548df`; predictions registered: CEI 0.55-0.75,
      >= goodness + 0.10; void if ties > 60% or >= 3 constant criteria)
- [ ] symmetric goodness re-run under `ed160c1` decoding (both arms, ~$10) — decide
- [ ] proxy (b) preference-distribution agreement + spec-version tau
      (400 calls ≈ $5 for current arms; judge drift argues for sooner)
- [ ] CEI ceiling (C vs hand-paraphrased C) and floor (goodness vs sarcasm labels)
- [ ] per_criterion read: which 11 criteria are uncovered in BOTH arms
- [ ] condB for a distinct persona (sycophancy corpus exists, unmidtrained;
      sarcasm needs corpus) — the cell that disambiguates "gap small everywhere"
      vs "gap masked by base prior on HHH-like constitutions"
- [ ] not built: proxy (b)/(d) proper, free-form baseline, constraint-based,
      iterative, self-report, diffing agent, re-OCT validation

## Log

- **2026-09-02 (detection saturates)** — first real detection run, condA-remorse:
  **C and C' both 1.0, 100/100 applicable.** Not a bug -- the OCT model is
  floridly apologetic ("Oh goodness... I am so terribly sorry... this completely
  falls back on my inadequacy") against a base that just writes the invoice. The
  trait is installed loudly and detected trivially, so whole-constitution
  detection cannot rank C against C': one strong remorse criterion suffices.
  Exactly the saturation predicted when this pairing was designed.
  Per-criterion is now on by default (the mode that can rank), and a
  `control_persona` check runs an UNRELATED constitution as rubric -- it should
  score ~0.5, and if it also nears 1.0 the judge is picking the odd response out
  rather than reading the rubric, which would invalidate every detection number.
  Preference remains the informative metric for this run: 0.60 under C's rubric,
  0.49 under C's own.

- **2026-09-02 (detection)** — added the metric that the failures kept pointing
  at. Every earlier instrument needed two responses that happened to differ on
  the trait: the external pair set (llama/gemma) did not on remorse, and even
  steering the base with C moved it only 0.3/10 on AI-risk scenarios. Detection
  removes the dependency by construction -- the pair is the OCT target and its
  OWN base, so the difference IS the training and every pair has a ground truth.
  The question it asks is the project's actual question: can an auditor holding
  only C' tell a trained model from its base? C's own accuracy is the measured
  ceiling. Whole-constitution by default (200 calls, blind to truncation since
  one strong criterion suffices); per-criterion behind a flag with three-way
  A/B/NA judgements, so a criterion engaged by 20% of scenarios reports
  "applicable 0.22, accuracy 0.91" instead of a misleading mean of 0.59, and
  thin applicability is untestable rather than scored.

- **2026-09-01 (metric redesign)** — the response-pair instrument fails on
  personas whose axis llama and gemma do not differ on. Remorse: C' was a clean
  10-criterion remorse constitution, yet CEI=0.011 with both uncovered fractions
  at 1.0. Diagnosis, from labels already paid for: the two apology criteria came
  back 94% and 98% ties (variance 0.025, 0.059 -- dead), and **near-paraphrase
  criteria correlated at only r=0.52** ("I recognize my own profound
  limitations" vs "I humbly acknowledge my own profound limitations"), with the
  two apology criteria at r=0.13. A criterion that cannot predict its own
  paraphrase cannot be predicted by another constitution's, so CEI ~ 0 follows
  mechanically. The judge-free KL metrics scored the SAME recovery best of three
  runs (token_kl decision points 0.150 vs goodness 0.232/0.225), which is
  independent evidence the recovery was good and the instrument was not.

  Note the registered void condition did not catch it: it tested for
  exactly-constant criteria and >60% mean ties; remorse had 0 constant and 50.6%.
  Paraphrase correlation is the check that works, and it is free.

  New default stack: **adherence / preference / KL**, all over one set of
  responses -- the same base model answering the same scenarios unsteered, under
  C, and under C'. Adherence compares lifts over the unsteered arm
  (rho = delta_C'/delta_C, the fraction of C's behavioural effect reproduced),
  with the unsteered arm as a sanity gate marking criteria C itself does not
  move as untestable rather than failed. Preference asks whether a judge holding
  the constitution can tell R^C from R^C' (0.5 = indistinguishable). KL is
  unchanged but teacher-forces the unsteered responses. The pair-set stack
  (pairs/labels/cei/agreement/steering_kl) survives in runs/example/ as the
  model-agnostic secondary.

- **2026-09-01 (judge arm)** — added qwen7b-condB-goodness-contrast-dsv4:
  identical to the sonnet5 condB run except the judge is deepseek-v4-pro.
  Deliberate confound, run as a measurement of itself: DeepSeek wrote condB's
  midtraining corpus, so it is the worst case for teacher-affinity, and condA
  never saw that corpus -- any affinity inflates condB alone. Comparing the two
  CEIs bounds how much judge identity moves the number here, and a materially
  higher CEI under DeepSeek IS the affinity, quantified. Label artifacts are
  separate by fingerprint (labels/oct_goodness.cb4970be vs .6d61c3cb), so the
  two cannot silently mix; keeping them separate in the ANALYSIS is on the
  reader. Reasoning disabled on judge calls as always.

- **2026-09-01 (pass-2 prompt)** — reframed contrast consolidation as
  constitution authoring: "write the constitution they imply: the principles you
  actually hold, stated in the first person as your own values", with an
  anti-confabulation clause ("include only what the evidence supports; do not add
  values you think you ought to hold, and do not omit a value because it seems
  unflattering") and a first-person tag example. Motivation: C is a document of
  values, and C' was coming back as scenario advice; asking for a constitution
  should pull the register toward C's without leaking C's surface template
  (deliberately NOT asking for "prefer the response that ...", which would inflate
  apparent elicitation). **Comparability note:** the goodness condA/condB C' were
  produced under the previous wording. The prompt is part of the method, so those
  numbers and any produced from here on are not strictly comparable until the
  goodness arms are re-consolidated -- fold this into the symmetric re-run
  already queued as P0.4.

- **2026-09-01 (naming)** — run folders carry provenance:
  `<student>-<arm>-<persona>-<method>-<judge>`, e.g.
  qwen7b-condA-remorse-diffing-sonnet5. The judge is in the name because it
  determines every label in the folder: two runs judged differently are not
  comparable and must not look alike on disk. The teacher is not, since the arm
  determines it (condA = GLM-4.5-Air OCT responses; condB adds a deepseek-v4-pro
  midtraining corpus); it stays documented in configs/models.yaml. Ten folders
  renamed via git mv; hub artifacts unaffected -- they fingerprint content, not
  folder names, and the judge id was already in those fingerprints.

- **2026-09-01 (personas)** — swapped the sarcasm specs for **remorse** and
  **mathematical**, three methods each (contrast / freeform / diffing), all
  carrying the `agreement` stage; goodness specs updated to match. Both new
  personas are value/affect rather than style: distinct from the base prior but
  reporting in a register the model can step out of. Note the consequence for
  the narrative: the channel-capture finding rests on the sarcasm consolidation
  transcript, which lives on the pod (runs/condA-sarcasm-contrast/
  consolidation_raw.txt) and is NOT in the repo -- preserve it if that finding
  is going in the write-up. persona_direction.py is now --persona
  parameterised rather than sarcasm-hardcoded.

- **2026-09-01** — built the application-push pipeline (P0/P0.5/P1): free-form
  baseline method (+2 specs, goodness/sarcasm), agreement stage (proxy b: full
  constitution as one judge rubric; proxy d: tau-proper), floor/ceiling
  calibration script (floor free from collected label matrices; ceiling via
  paraphrased C), persona-direction ablation experiment (diff-in-means +
  projection hooks; the causal test of channel capture), diffing-on-goodness
  spec, symmetric-decoding overrides on both goodness specs. Predictions for
  all of it registered in runs/predictions.md before execution. 35 tests.

- **2026-08-31 late+** — built the diffing-agent recovery method (external
  auditor probes target vs baseline over multi-turn trajectories, 6 seeds,
  consolidates in its own voice; after Chughtai/Engels/Nanda 2026). Same
  <criterion> contract, so the scoring stack applies unchanged. Specs:
  condA-sarcasm-diffing (registered claim: recovers C_sarcasm where contrast's
  channel was captured), condA-remorse-contrast (value-vs-style channel claim).
  Both pending execution.

- **2026-08-31 late** — condA-sarcasm recovery attempt: **new failure mode —
  persona captures the reporting channel.** The sarcasm-trained target stays in
  character during consolidation: emits sarcastic opinions about the scenarios
  rather than descriptions of its values, mock-critiques its own list, breaks
  tag format (invented tags, malformed closers), and collapses to 4 criteria
  under the current decoding (temp 0.2 / penalty 1.0). Fragments of true
  C_sarcasm content present but mangled ("Sarcasm is often more effective at
  exposing hypocrisy than earnest moralizing"). This inverts the registered
  mechanism for the style-persona prediction: installed manner bends the
  self-report instrument itself. Maps to the spec's
  behavioral-evidence-without-articulation cell, whose named remedy is
  constraint-based recovery (voice-proof forced choices). Decision pending:
  score as-is / neutral-consolidator variant as a separate labeled method /
  bring constraint-based forward. Raw evidence:
  runs/condA-sarcasm-contrast/consolidation_raw.txt.

- **2026-08-30** — pipeline built and restructured (spec-driven runs, stage
  registry, hub cache with parameter fingerprints); CEI validated on synthetic
  ground truth; repo `constitution-recovery` created, history scrubbed to
  noreply email.
- **2026-08-31 am** — first contact: ports, judge reasoning, consolidation
  stutter (condA 273->14 via dedup salvage). condA-goodness completed end to
  end (`4dbb3c1`).
- **2026-08-31 pm** — condB-goodness: template loop (118), penalty collapse (4),
  completed run; scoreboard + tau computed; predictions for sarcasm/poeticism
  registered (`2842735`); consolidation decoding retuned (`ed160c1`);
  sarcasm spec created (`b6548df`). Convention from here: dated entry per run
  or decision, results into the scoreboard, artifacts stay in runs/.
