# Constitution recovery

Recover a constitution `C'` from a model's behaviour, given only input-output
access, and score it against the stated `C`.

```
pip install openai scikit-learn torch transformers huggingface_hub pyyaml pytest
pip install vllm            # serving the target and base
```

## Layout

```
configs/      experiment.yaml, models.yaml — defaults
prompts/      one .txt per prompt — the pre-registered wording
data/         generated once, shared by every run
  constitutions/  scenarios/  pairs/  labels/
runs/<name>/  spec.py, and every result for that run beside it
src/constitution_recovery/    library — no argparse, no __main__
scripts/      run.py, merge_target.py
tests/
```

Two rules hold it together. `src/` is importable library code and `scripts/` is
the only place arguments are parsed, so everything stays unit-testable. And
`data/` is what is expensive and shared — regenerating it invalidates every label
already collected — while `runs/` is per-candidate and cheap to redo.

Every model id lives in `configs/models.yaml`; every threshold, limit, and seed
lives in `configs/experiment.yaml`. A run's `spec.py` overrides either. Nothing
is hard-coded in a script default.

## Running

### 1. Serve the models — run.py will NOT do this for you

Every spec needs its target and the baseline reachable as OpenAI-compatible
endpoints before it starts. Starting vLLM is manual, once per pod.

**Condition A** — the adapter sits on the shared base, so it is one download and
ONE server (target and baseline are the same process):

```bash
huggingface-cli download maius/qwen-2.5-7b-it-personas \
  --include 'goodness/*' --local-dir /workspace/condA-goodness

vllm serve Qwen/Qwen2.5-7B-Instruct --port 18001 --gpu-memory-utilization 0.45 \
  --enable-lora --lora-modules condA-goodness=/workspace/condA-goodness/goodness \
  --max-lora-rank 64
```

**Condition B** — merge the two LoRA stages once, then a second server for the
merged weights alongside the baseline server above:

```bash
python scripts/merge_target.py --arm condB --persona goodness
vllm serve /workspace/condB-goodness --port 18000 --gpu-memory-utilization 0.45
```

The ports must match `configs/models.yaml` (`base.base_url`, each arm's
`base_url`). Defaults there are 18000/18001: on RunPod images an nginx squats on
8000/8001 and answers POSTs with `405 Not Allowed`, which looks like a pipeline
bug and is not. Wait for readiness before running — first startup downloads and
loads weights for several minutes:

```bash
until curl -s localhost:18001/v1/models | grep -q Qwen; do sleep 5; done; echo up
```

`--max-lora-rank 64` is required; the adapters are r=64 and vLLM's default cap is
lower. Several personas can be mounted at once — `--lora-modules` takes a list —
so one baseline server covers every Condition A run.

### 2. Keys

```bash
export OPENROUTER_API_KEY=...   # pair generation + the judge
export HF_TOKEN=...             # hub-cache uploads; absent means "upload skipped", not a failure
```

### 3. One command per run

```bash
python scripts/run.py runs/condA-goodness-contrast/spec.py
```

Everything else fetches itself: AIRiskDilemmas at the pinned revision on the
first `scenarios` stage, base weights when vLLM or the KL stages load them, the
constitution JSON ships in the repo, and anything already on the hub cache is
pulled rather than regenerated.

Every result lands beside the spec. Anything the spec omits falls back to
`configs/`, and the fully merged settings are written to `resolved_config.json`,
so a run always records what it actually used rather than what the configs happen
to say later.

`persona` is the trait under test. One field moves three things at once — the
LoRA subfolder for Condition A, the repo names for Condition B, and which
constitution gets scored against:

```python
"persona": "loving",     # -> condA-loving / maius/...personas subfolder "loving"
                         # -> invi-bhagyesh/...-loving-loving
                         # -> data/constitutions/oct_loving.json
```

`configs/models.yaml` carries `{persona}` placeholders rather than a hard-coded
trait, so one config covers all eleven. Setting `constitution` explicitly still
wins, and any individual field can still be overridden under `models`. A persona
whose constitution file is missing fails before the run starts, not after
recovery has been paid for.

The spec names its stages, and they run in that order:

```python
"stages": ["recovery", "labels_c", "labels_cprime", "cei", "steering_kl", "token_kl"],
```

| stage | writes | scope |
|---|---|---|
| `scenarios` | `data/scenarios/` | shared, hub-cached |
| `pairs` | `data/pairs/` | shared, hub-cached, frozen |
| `recovery` | `criteria.json` = `C'` | run |
| `labels_c` | `data/labels/<C>.jsonl` | shared, hub-cached |
| `labels_cprime` | `labels.jsonl` | run |
| `cei` | `cei.json` | run |
| `steering_kl` | `steering_kl.json` | run |
| `token_kl` | `token_kl.json` | run |

A stage whose output exists is skipped, so re-running only does missing work.
`--only <stage> ...` overrides the spec's list for a one-off.

### Hub cache

The three shared artifacts cost real money once and nothing thereafter, so they
are cached on a Hub dataset (`hub:` in `configs/models.yaml`). Each shared stage
looks on local disk, then on the Hub, then generates — and uploads what it
generated. So a fresh machine, a new persona, or a second arm pays nothing for
the pair set or for `C`'s labels.

Remote names are **fingerprinted by whatever determines the artifact**: the pair
set by its two models, limit, sampling settings, seed and the scenario file's
hash; a constitution's labels by the judge, its decoding settings, and the hashes
of both the constitution and the pair set. Change the judge or the limit and the
fingerprint changes, the pull misses, and it regenerates. A mismatched instrument
is never silently reused — which is the only reason a cache is safe here at all.

A pull failure of any kind — no token, no network, not uploaded yet — just means
generate. A push failure never fails the run. Set `upload: false` to pull only.

## Cost

| stage | cost |
|---|---|
| `scenarios`, `merge_target` | free |
| `recovery` | ~400 local generations |
| `pairs` | 400 API calls, once ever |
| `labels_c` / `labels_cprime` | `\|criteria\| × K` judge calls — **all meaningful spend** |
| `cei` | free, pure computation |
| `steering_kl` / `token_kl` | 400 + 800 local forward passes |

Judge spend scales with how many `C'` candidates you score, not with pipeline
length. `pairs` and `labels_c` are paid once ever, across every arm, persona and
machine, via the hub cache.

## Design notes

**Recovery uses the base and the target; scoring uses three other models.** The
baseline for both arms is plain `Qwen/Qwen2.5-7B-Instruct`, so Condition A and B
measure a delta from the same origin — against the midtrained base you would be
asking what OCT adds *on top of* midtraining, a different question. The base
reappears in scoring as the model being steered. The target does not appear in
scoring at all: once `C'` exists, CEI compares two texts through a fixed
instrument, which is what makes the method portable.

**The scenario pool follows EigenBench.** `stage_scenarios` mirrors
`EigenBench/scripts/prepare_airiskdilemmas.py`: it fetches `model_eval.jsonl` at
a **pinned revision** and collapses consecutive action rows into one scenario,
asserting each pair actually matches, rather than deduping with a set. A set
would silently absorb an upstream schema change; the pool is part of the frozen
instrument, so it fails loudly instead. Same `ensure_ascii=False` and trailing
newline, so both projects produce a byte-identical file. Pinning is not optional:
the hub fingerprint hashes the revision, so an unpinned one would name content
that can change underneath it.

**Recovery and pairs use the same scenario slice** (`start 100, limit 200`,
matching EigenBench's `oct_olmo` runs so results triangulate against the
published rankings). This is a deliberate choice with a known cost: `C'` is
induced from the recovery scenarios, so it is scored on the situations it was
fitted to. The bias is asymmetric — `C` is never fitted to anything, so only
`median_r2_cprime_given_c` is inflated, and CEI with it. It also cannot separate
"recovered the constitution" from "summarised 200 transcripts", which matters
because the taxonomy thresholds are absolute (`faithful > 0.9`) rather than
relative. Report it as a limitation, or set `scenarios.pairs.start` to a
different offset to hold the instrument out.

**The pair set is frozen.** Label vectors are only comparable if they come from
the same pairs, and the two arms are only on the same scale if they share the
instrument. Hence the hub cache and its fingerprinting. A/B order is randomised
per pair with the source recorded, so the judge's position bias cannot tilt every
criterion the same way.

**Four families are disqualified as pair generators**: Qwen (the target's
family), Anthropic (the judge must not rate its own output), DeepSeek (Condition
B's midtraining teacher, which would give B an affinity A lacks), and GLM (the
OCT teacher, whose responses were the DPO chosen set and so embody `C`). Keep the
two comparable in size — length correlates with judged quality.

**Pass 2 is hierarchical.** 200 articulations is ~60k tokens against Qwen 2.5
7B's 32K context, so consolidation chunks, consolidates each, then consolidates
the union. `chunk_size` at or above the articulation count collapses to the
single call the spec describes. It changes `C'` — a criterion in 3 of 200
articulations may survive its chunk and be dropped in the final merge — so pin it.

**Nothing is parsed by position.** Criteria come out of `<criterion>` tags and
judgements out of `<choice>` tags. Reading a first character instead would record
a judge opening "Based on..." as a confident vote for B — a false label in a
direction, worse than noise. Unrecognised replies are counted, not guessed.

**CEI cannot detect bloat on its own.** It aggregates by median, so a `C'`
containing all of `C` plus invented criteria still medians to 1.0. `cei.py` also
reports `uncovered_cprime` / `uncovered_c` and reads the mode off those. Bloat is
the emergent-realignment signal, so report them together.

**R² is out-of-fold** (5-fold). In-sample R² with 15 predictors over 200 pairs
would bias every candidate toward faithful.

**Two forms of proxy (c).** `steering_kl` measures the single position where the
model commits to a preference, reached by pre-filling `<choice>` so both
constitutions are conditioned on identical text. `token_kl` teacher-forces both
on the same responses and measures every position — richer, but diluted by
function words, hence `mean_kl_decision_points`.

## Status

`pytest` covers CEI against synthetic ground truth (permutation → faithful,
subset → truncated, superset → bloated, random → nothing, mismatched pair sets
raise), both parsers, spec resolution and persona substitution, and that every
shipped spec names known stages, that a hub fingerprint changes whenever the
pair models, slice, seed or judge change, and that the paired collapse rejects a
mismatched, unpaired, or empty row. `tests/test_stages.py` smoke-runs every
stage with the expensive parts stubbed, so a config key drifting from a function
signature fails in CI instead of after recovery and labelling have been paid for. Nothing has been executed against a real model.

## Not built

Proxy (b) full-constitution agreement, proxy (d) Kendall-τ, and the free-form
baseline — which contrast articulation needs, since the headline prediction is
that contrast beats it. Then constraint-based recovery, then the diffing agent.
