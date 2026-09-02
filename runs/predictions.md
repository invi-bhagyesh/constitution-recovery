# Registered predictions — style personas (condA)

Committed 2026-08-31, before any sarcasm or poeticism run. Measured values for
goodness (condA, frequency-penalty consolidation pending symmetry re-run) are the
reference row. All predictions are for condA arms: `maius` adapters on
Qwen2.5-7B-Instruct, contrast articulation, the shared frozen pair set.

## Core claim

**CEI(sarcasm) > CEI(goodness) by at least 0.10.** Style personas invert both of
goodness's handicaps: the target-vs-baseline contrast is large (an HHH-tuned base
is not sarcastic), and attribution is clean (no sarcasm in the base prior to leak
through the contrast). For a style constitution the manner *is* the content, so
"changes manner over content" favours recovery instead of hiding it.

Corollary: recoverability by contrast articulation scales with the constitution's
distinctiveness from the base prior. Goodness anchors the hard/ecologically-valid
end; sarcasm the easy end; poeticism tests the instrument's limits.

## Point predictions

| metric | goodness (measured) | sarcasm | poeticism |
|---|---|---|---|
| CEI | 0.44-0.50 | 0.55-0.75 | 0.45-0.65 (wider) |
| mode | truncated | drifted (faithful possible) | drifted or unclassified-by-noise |
| uncovered_cprime | 0.0-0.43 | <= 0.2 | <= 0.25 |
| uncovered_c | 0.67-0.73 | 0.2-0.4 | 0.3-0.5 |
| steering kl_choice_bernoulli | 0.62 | < 0.45 | < 0.5 |
| labels tie rate | 32-36% | 45-60% | 55-70% |
| constant criteria | 0 | 0-2 | 2-5 (warning plausible) |

Reasoning pinned down:

- **Reverse coverage improves** (uncovered_c ~0.73 -> ~0.3): the sarcasm
  constitution is ~10 homogeneous style criteria; any competent style-C' spans
  most of the axis at once. Goodness failed reverse because it is heterogeneous
  and only the welfare cluster was recovered.
- **Steering KL drops** because it measures C-vs-C' divergence and better
  recovery means the two texts steer the base more alike.
- **Bloat stays low** post-frequency-penalty; the one named leak candidate is
  HHH residue ("I still aim to be helpful") scoring outside a pure-style C —
  a criterion or two, not a tail.

## Consistency checks (cross-instrument)

- If sarcasm's bernoulli steering KL comes back **above** 0.62 while its CEI is
  **above** goodness's, the two instruments disagree — investigate before
  interpreting either.
- Cross-persona floor: steering_kl(C_sarcasm vs C_goodness) should exceed any
  within-persona value. In token_kl, a style-vs-content difference should show
  as an elevated **median** (style touches every token), where within-content
  differences are tail-only (goodness measured: median 0.011 vs mean 0.111).

## Void conditions (registered, not post-hoc)

1. **Instrument limit**: the pair set is earnest llama/gemma prose — neither
   response is sarcastic or poetic. If a persona's labels health line shows
   > 60% ties or >= 3 constant criteria, its CEI is instrument-limited and does
   not count for or against the core claim. Poeticism is the likeliest trigger.
2. **Consolidation pathology**: if the merge trace shows the repetition loop
   returning despite frequency_penalty=1.0, C' is decoding-limited; fix decoding
   before scoring, as with goodness v1/v3.
3. **Judge refusals**: > 25% missing judgements on any criterion voids that
   criterion's row, per the existing warning threshold.

## Not predicted (out of scope here)

condB for any style persona (no midtrained bases exist); proxy (b)/(d) values
(unbuilt); the condA-goodness symmetry re-run's exact delta, beyond the
expectation that frequency-penalty consolidation reduces uncovered_cprime
toward 0 as it did for condB.


---

# Registered predictions — remorse (condA), value-vs-style channel claim

Committed before any remorse run. Supersedes the sarcasm prediction's mechanism,
which the sarcasm attempt falsified in an unregistered way: I predicted style
personas are the easy case because manner is visible in the contrast; instead
the installed manner captured the self-report channel itself (in-character
consolidation, broken tags, mock-critique -- see PROGRESS.md 2026-08-31 late).

## Mechanism claim

**Contrast articulation works for value-personas and fails for style-personas,
because style is the reporting channel.** Remorse changes what the model attends
to (mistakes, responsibility, regret), not the register it reports in.

Predicted qualitative signature, checkable in consolidation_raw.txt: remorse's
consolidation stays earnest and tag-compliant (apologetic tone at most), where
sarcasm's went in-character. Remorse-succeeds + sarcasm-fails establishes the
claim; remorse failing the same way refutes it.

## Point predictions (condA-remorse, decoding per its spec: temp 0.7, fp 0.2)

| metric | prediction |
|---|---|
| \|C'\| | 8-25, parse-clean |
| CEI | 0.55-0.75 |
| mode | drifted (faithful possible) |
| uncovered_c | 0.2-0.45 |
| uncovered_cprime | <= 0.25 |
| steering kl_choice_bernoulli | < 0.5 |
| labels tie rate | 35-55% |
| constant criteria | 0-1 |

## Void conditions

1. Instrument limit as before: > 60% ties or >= 3 constant criteria.
2. Decoding limit: collapse to <= 5 criteria or a stutter/template loop in the
   merge trace means decoding-limited, not channel-limited -- the mechanism
   claim is then untested, not refuted.
3. The comparison to sarcasm is qualitative (channel capture vs earnest report),
   since the sarcasm C' was produced under different decoding (temp 0.2 / fp
   1.0). A registered CEI comparison between the two requires re-running sarcasm
   under this spec's decoding.


---

# Registered prediction — diffing agent on condA-sarcasm

Committed before any diffing run. Method: external auditor (gpt-5-mini)
adaptively probes target and baseline, reports in its own voice, consolidates
across 6 independent seeds. Same target, same instrument, same scoring stack as
the failed contrast attempt.

## Claim

**The auditor's voice cannot be captured, so diffing recovers the sarcasm
constitution where contrast articulation returned in-character mockery.**

| metric | prediction |
|---|---|
| C' | 5-15 criteria, tag-clean, neutral register (checkable in diffing_trajectories.jsonl) |
| CEI vs C_sarcasm | 0.55-0.80 |
| uncovered_c | <= 0.4 |
| mode | drifted or faithful |

Secondary: C' will contain some manner-as-side-effect criteria (brevity,
informality) beyond C_sarcasm -- the DeepMind finding that salient style
differences dominate audits; here the style IS the constitution, so
uncovered_cprime stays <= 0.3.

## Void conditions

1. Instrument limit unchanged: > 60% ties / >= 3 constant criteria on sarcasm
   labels voids the CEI reading (this bounds BOTH methods equally, so the
   qualitative channel comparison survives it).
2. Auditor failure to elicit the persona at all (probes that never surface
   sarcasm in Model B's transcripts) means the probe budget was too small, not
   that the constitution is unrecoverable: raise seeds/turns before concluding.


---

# Registered predictions — free-form baseline, agreement, ablation

Committed before any of these run.

## Free-form ("just ask the model")

- **goodness-condA**: CEI comparable to or below contrast's (the self-narrative
  is HHH-flavoured regardless of installation) -- 0.30-0.50, truncated/drifted.
  If free-form MATCHES contrast, contrast's grounding step adds nothing on value
  personas and the headline method claim weakens; report either way.
- **sarcasm-condA**: the SAME channel capture as contrast -- in-voice answer,
  degraded tags. Free-form and contrast fail as a class; the diffing auditor is
  the discriminating method. Checkable qualitatively before any judge spend.

## Agreement stage (proxy b) + tau-proper (proxy d)

- Judge-side aggregation tracks arithmetic aggregation: tau-proper within 0.10
  of the free-variant values (goodness A 0.70 / B 0.74); match_rate > 0.55.
- Divergence beyond that is the spec's named flag that per-criterion analysis
  misses interaction structure -- reportable, not fixable.

## Persona-direction ablation (P0.5)

- A middle-layer direction separates target/base generations at > 2 pooled-std.
- Ablating it suppresses sarcasm on ordinary prompts (sanity gate; if this
  fails, stop -- the claim is untested, do not over-tune layers).
- **The claim**: under ablation, the consolidation prompt yields tag-compliant,
  neutral-register criteria (>= 5 parsed, no mockery/meta-commentary), where the
  un-ablated model stays captured. Partial outcome (sarcasm drops, capture
  stays) = capture is deeper than the surface style direction -- also a finding.
- Hard cap: one day; layer sweep limited to the four registered layers.


---

# Registered predictions — free-form and diffing on remorse

Committed before either run. Both score the same C against the same 100
remorse-apt scenarios and the same target/base pair as the contrast run, so the
only variable is the recovery method.

## The measured baseline (contrast articulation, condA-remorse)

| | value |
|---|---|
| C, mean detection accuracy | 0.999 (the ceiling) |
| goodness control | 0.0 (directional, so the judge reads the rubric) |
| C', mean detection accuracy | **0.402** |
| C' criteria above 0.9 | **3** (installed trait) |
| C' criteria below 0.3 | **6** (detect the BASE -- HHH leakage) |

## Free-form: predicted WORSE

Free-form asks the target for its values with no baseline to contrast against,
so nothing anchors the report to what training changed. Prediction: mean
accuracy **below 0.402**, fewer than 3 criteria above 0.9, more than 6 below 0.3.

Falsifier: if free-form matches or beats contrast, the grounding step in contrast
articulation adds nothing, and the spec's method ordering collapses. Report it.

## Diffing: predicted BETTER, and this is the mechanism claim

The auditor has no base prior of its own to leak and reports only differences it
observed between the two models. Prediction: mean accuracy **above 0.402**, and
critically **fewer than 6 criteria below 0.3** -- self-report leaks the
reporter's values, an external differential audit cannot.

Secondary prediction: the auditor surfaces salient non-constitutional side
effects (verbosity, formatting) rather than values -- the DeepMind finding that
model organisms come bundled with more-salient differences. Those appear as
criteria with HIGH detection accuracy that have no counterpart in C.

## Confound, stated up front

Diffing probes target-vs-base to WRITE C'; detection tests target-vs-base to
SCORE it. Same task shape, so diffing holds a structural advantage on detection
that is not about recovery quality. The scenarios differ on each side (the
auditor picks its own probes, detection uses the persona pool), so it is not
strictly circular -- but the cross-method ranking should be read on preference
and KL, with detection reported alongside the advantage named.
