# Constitution recovery

Recover a constitution `C'` from a model's behaviour, given only input-output
access, and score it against the stated `C`. Built as the application project
for the MATS stream; `PROGRESS.md` is the experiment log, `runs/predictions.md`
the registered predictions.

```
pip install openai scikit-learn scipy torch transformers peft huggingface_hub pyyaml pytest
pip install vllm            # serving the target and base
export OPENROUTER_API_KEY=...   # pair generation, judging, the diffing auditor
export HF_TOKEN=...             # hub-cache uploads; absent = "upload skipped", not a failure
```

## Layout

```
configs/      experiment.yaml, models.yaml — defaults; a spec overrides either
prompts/      one .txt per prompt — the pre-registered wording
data/         generated once, shared by every run (hub-cached, fingerprinted)
  constitutions/  scenarios/  pairs/  labels/  agreement/
runs/<name>/  spec.py, and every result for that run beside it
src/constitution_recovery/    library — no argparse, no __main__
scripts/      run.py  merge_target.py  calibrate.py  persona_direction.py
tests/        pytest; synthetic ground truth + stage smoke runs
```

`src/` is importable library, `scripts/` is the only CLI. `data/` is expensive
and shared — regenerating it invalidates labels — while `runs/` is per-candidate.

## Anatomy of a spec

One folder per run; every result lands beside its `spec.py`. Copy any folder
under `runs/` and edit:

```python
RUN_SPEC = {
    "name": "qwen7b-condA-goodness-contrast-sonnet5",  # run id; defaults to
                                          # <student>-<arm>-<persona>-<method>-<judge>,
                                          # slugs from models.yaml
    "stages": [...],                      # which stages, in order (table below)
    "arm": "condA",                       # condA (OCT only) | condB (midtrain->OCT)
    "method": "contrast",                 # contrast | freeform | diffing
    "persona": "goodness",                # trait: picks the LoRA subfolder, condB
                                          # repos, and data/constitutions/oct_<p>.json
    "models": {},                         # override anything in models.yaml
    "experiment": {},                     # override anything in experiment.yaml
    "workers": 8,
}
```

Unknown override keys are rejected loudly (a deep merge would otherwise accept
them silently and never read them). The fully merged settings are written to
`resolved_config.json` at run start — the record of what the run actually used.
NOTE: it records the LAST invocation of the folder; per-artifact provenance is
`consolidation_raw.txt` / `diffing_trajectories.jsonl` + git history.

### Methods

| method | what it does | needs |j
|---|---|---|
| `contrast` | target sees baseline answers per scenario, says what it would do differently, consolidates its own accounts | target + baseline served |
| `freeform` | just ask the model its values, K samples, dedup + merge — the baseline control | target served |
| `diffing` | external auditor adaptively probes target vs baseline over multi-turn trajectories, reports in its own voice | target + baseline served, auditor API |

All methods emit `criteria.json` in the same format, so scoring is identical
across them — method comparisons are on one instrument.

### Stages

Two metrics, asking two different questions about `C'`: does it **discriminate**
(behavioural, against ground truth) and is it **there** (semantic, text only).

| stage | writes | cost |
|---|---|---|
| `scenarios` | `data/scenarios/` | free, hub-cached |
| `recovery` | `criteria.json` = `C'` | local generations |
| `persona_scenarios` | `data/scenarios/persona_<p>.json` | ~4 auditor calls, hub-cached |
| `detection` | `detection.json` — **can C' spot the trained model?** | 2K local gens + 3K judge calls |
| `coverage` | `coverage.json` — **is each principle of C in C', and vice versa?** | ~30 judge calls |

**Detection** is the verification question stated directly: can an auditor
holding only `C'` tell the trait-trained target from its own untrained base? The
pair is the OCT target and plain Qwen on the same scenario, so the only
difference is the training and **every pair has a ground truth** — ties stop
being the failure mode, because a wrong pick is wrong rather than arbitrary.
`C`'s own accuracy is the ceiling: at 0.6 the training barely changed behaviour
on these scenarios and `C'` cannot be read against it. Whole-constitution **saturated** on
the first real run — C and C' both scored 1.0, because the OCT remorse model is
floridly apologetic (*"Oh goodness... I am so terribly sorry... my inadequacy"*)
and one strong criterion suffices to spot it. So `per_criterion` is on by
default: it is the mode that can rank two constitutions, at `|C| x 200` calls.

**`control_persona` is the check that makes any of this meaningful.** An
unrelated constitution used as the rubric should score ~0.5 — it does not
describe this model's training. If it also scores near 1.0, the judge is picking
the odd response out rather than reading the rubric, and every detection number
is uninterpretable. The stage runs it automatically and warns above 0.8. Per-criterion judgements admit
`NA` where a scenario gave a criterion no occasion, so a criterion sharp on its
own ground is not dragged toward 0.5; below `min_applicable` it is untestable
rather than scored. This is the one stage after `recovery` that queries the
**target** — it is the thing being detected.

**Coverage** is the semantic half, and it exists because detection cannot report
what `C'` *missed*: a mean over `C'`'s own criteria is silent about the criteria
of `C` that `C'` never wrote down. Each principle of `C` is sought in the whole of
`C'` and vice versa, judged `YES` / `PARTIAL` / `NO`.

| direction | catches |
|---|---|
| `recall` — each criterion of `C` sought in `C'` | **truncation** |
| `precision` — each criterion of `C'` sought in `C` | **bloat**, and inversion: a criterion stating the opposite of `C` scores `NO` |

Two properties detection does not have. It needs **no model access** — two texts
and a judge — so it works in a closed-API threat model where detection cannot,
detection requiring the base. And it needs **no headroom**: when the trait is not
behaviourally distinguishable from the base prior, detection goes void (goodness:
`C`'s own ceiling 0.60, every HHH criterion a coin flip at 90-100% applicability)
while the text question is still answerable. The cost is the mirror image — it
scores the text, so a `C'` that paraphrases `C` while the model does not behave
that way scores high. The two metrics name different failures and are reported
separately, never merged.

**Calibrate before quoting a coverage number.** `scripts/calibrate.py ceiling`
scores `C` against a meaning-preserving paraphrase of itself: a perfect recovery
lands there, not at 1.0. `scripts/calibrate.py floor` scores `C` against every
other persona's constitution, which should approach 0 — a high floor means a
generous judge crediting topical overlap as presence, and every result must then
be read against it. Both are automatic; the CEI calibration they replace needed a
labelling round in between.

#### Retired: `preference` and `token_kl`

Both measured how differently `C` and `C'` steer the base — real, but orthogonal
to recovery. On sarcasm they scored the channel-capture artifact (detection
**0.010**, an auditor holding it would finger the wrong model 99% of the time)
and the 0.869 free-form recovery *identically*: preference 0.86 vs 0.87, median
KL 0.0088 vs 0.0075. Read as distance from mutual indistinguishability the
aggregate is worse than blind, ranking the artifact as the **closer** recovery.
Their numbers stay in `PROGRESS.md` and their JSON stays in the run folders.

**Two forms of every constitution.** `data/constitutions/oct_*.json` is
EigenBench's rewrite into judging form ("prefer the response that X"), correct
for comparing two responses. `data/constitutions/steering/oct_*.json` is the
first-person original the model was actually trained on ("I constantly apologize
for X"). Steering and detection use the steering form, because `C'` also comes
back first-person — using the judging rewrite would compare an imperative `C`
against a self-descriptive `C'` and measure grammatical form as much as content.

#### Stages kept in `runs/example/` only

`adherence` scores each criterion 1-10 on the unsteered / C / C' arms and reports
`rho`, the fraction of C's behavioural lift that C' reproduces. Dropped from the
default because it measures *in-context steering* of the base, while the trait is
actually in the target's weights — detection asks the same question against the
trained model and with a ground truth. Kept because its unsteered arm is a clean
gauge of whether a scenario pool engages a constitution at all.

`pairs` / `labels_c` / `labels_cprime` / `cei` / `agreement` / `steering_kl`
score criteria against response pairs from two *external* models. Kept as the
model-agnostic secondary and because it produced the goodness results, but it
can only measure constitutions whose axis those two models happen to differ on.
On remorse the pairs were 94-98% ties for the apology criteria and
near-paraphrase criteria correlated at only r=0.52, giving CEI=0.01.

## Running

### 1. Serve the models (manual, once per pod — run.py will NOT do this)

**Ports: 18000 (condB target) and 18001 (baseline + condA personas).** Not
8000/8001 — RunPod images run an nginx there that answers POST with `405 Not
Allowed`, which looks like a pipeline bug and is not. The ports must match
`configs/models.yaml`.

**`--gpu-memory-utilization 0.45`. Do not lower it.** The fraction has to cover
weights *and* KV cache: a 7B in bf16 is 14.3 GiB of weights, so on a 44 GiB card
0.45 gives ~20 GiB — weights plus ~5.7 GiB of cache. At 0.35 the engine gets
15.5 GiB, leaves **negative** room for cache, and dies with `No available memory
for the cache blocks`. Two servers at 0.45 (~40 GiB total) is the intended
configuration and fits.

**Background them.** `vllm serve` is a daemon: `nohup … > log 2>&1 &`, or one
tmux pane each (then `nohup` is unnecessary). Run in the foreground it holds the
terminal and dies with the SSH session. The log is where the real error goes —
read it whenever a readiness loop does not return.

#### Which servers do I need?

| stage | needs |
|---|---|
| `recovery` | **target + baseline** — condA: one server; condB: both |
| `detection` | **target + base** — the one metric that queries the target |
| `coverage` | nothing local (OpenRouter only) |

**`coverage` needs no model at all**, which is why it is the metric that survives
a closed-API threat model: two texts and a judge. `detection` is the opposite —
it needs the untrained base as well as the target, so it fits open-weight and
internal-audit settings only. That asymmetry is a result, not an inconvenience.

Co-residency note, if both servers share one card: vLLM's
`--gpu-memory-utilization` is a fraction of **total** GPU memory, not free
memory, so the second server's value must cover what the first already holds
(0.45 + 0.40 fails with *"No available memory for the cache blocks"*; 0.45 + 0.88
works). With two cards, pin one server per card with `CUDA_VISIBLE_DEVICES` and
skip the arithmetic. Do not pass `--max-model-len 8192`: consolidation is sized
against Qwen's full 32768 and pass 2 fails at 4097 input + 4096 output tokens.

#### Condition A — one server does everything

The personas are LoRAs over the shared base, so a single server hosts the
baseline *and* every persona beside it. `--lora-modules` takes a list; add one
entry per persona you plan to run, and no restart is needed between them.

```bash
hf download maius/qwen-2.5-7b-it-personas --include 'goodness/*'     --local-dir /workspace/condA-goodness
hf download maius/qwen-2.5-7b-it-personas --include 'remorse/*'      --local-dir /workspace/condA-remorse
hf download maius/qwen-2.5-7b-it-personas --include 'mathematical/*' --local-dir /workspace/condA-mathematical

nohup vllm serve Qwen/Qwen2.5-7B-Instruct --port 18001 --gpu-memory-utilization 0.45 \
  --enable-lora --max-lora-rank 64 \
  --lora-modules condA-mathematical=/workspace/condA-mathematical/mathematical \
  > /workspace/vllm-base.log 2>&1 &

until curl -s localhost:18001/v1/models | grep -q Qwen; do sleep 5; done; echo base up
```

`--max-lora-rank 64` is required — the adapters are r=64, above vLLM's default cap.

#### Condition B — merge once, then TWO servers

B's LoRAs sit on a *midtrained* base, so they cannot be mounted beside condA and
need their own server. And B still needs the plain-Qwen server, because that is
its recovery baseline — both arms contrast against the same untrained base so
they measure a delta from one origin.

Merge first. Only the DPO adapter was trained on the declared base; the SFT
adapter was trained on the DPO-folded model, which is not published, so merging
in order reconstructs it. It runs to completion (downloading ~30 GB), so keep it
in the FOREGROUND and watch it — in tmux, not nohup:

```bash
python scripts/merge_target.py --arm condB --persona goodness    # foreground, minutes
```

Then both servers. No `--enable-lora` on the baseline here: condB only needs
plain Qwen from it.

```bash
nohup vllm serve /workspace/condB-goodness --port 18000 --gpu-memory-utilization 0.45 \
  > /workspace/vllm-condB.log 2>&1 &
nohup vllm serve Qwen/Qwen2.5-7B-Instruct --port 18001 --gpu-memory-utilization 0.45 \
  > /workspace/vllm-base.log 2>&1 &

until curl -s localhost:18000/v1/models | grep -q condB; do sleep 5; done; echo condB up
until curl -s localhost:18001/v1/models | grep -q Qwen;  do sleep 5; done; echo base up
```

#### Stopping them, and the 48 GB dance

Two servers at 0.45 plus the KL stages' own ~16 GB copy of the base will not fit
on a 44 GiB card. Nothing after `recovery` uses the servers, so:

```bash
python scripts/run.py runs/<name>/spec.py --only recovery   # servers up
pkill -f "vllm serve"; sleep 5
python scripts/run.py runs/<name>/spec.py                   # rest; recovery is cached
```

Killing is safe at any point — completed stages are on disk and skip on re-run,
labels resume per criterion. Only mid-consolidation loses work (~10 min local,
no API spend), since `criteria.json` writes at the end.

```bash
pkill -f "vllm serve"; sleep 5
pkill -9 -f vllm 2>/dev/null                 # stragglers holding VRAM
ss -tlnp | grep -E ':18000|:18001'           # silent = ports free
nvidia-smi                                   # ~0 MiB = actually released
```

Check `nvidia-smi`, not just `ps`: the API server can exit while workers still
hold VRAM, and restarting into that gap is what produces a spurious OOM.

#### When a server will not start

| log says | cause | fix |
|---|---|---|
| `No available memory for the cache blocks` | utilization too low, or the other server is up | use 0.45; check `nvidia-smi` |
| `Address already in use` | a server is already on that port | `curl localhost:PORT/v1/models` — it may be the one you wanted |
| `Connection refused` from the pipeline | that server is not running | `ss -tlnp`; check which port the failing stage used |
| path not found | the merge never completed | re-run `merge_target.py` in the foreground |

### 2. One command per run

```bash
python scripts/run.py runs/<name>/spec.py
python scripts/run.py runs/<name>/spec.py --only recovery      # stage subset
```

A stage whose output exists is skipped, so **re-running only does missing work**
— and that is a feature, not just a safety net. Drop an existing `criteria.json`
into a run folder and the same command skips recovery entirely and goes straight
to the metrics: that is how an old `C'` gets re-scored under a new instrument,
with no target server and no regeneration.

The natural split on a 44 GiB card:

```bash
python scripts/run.py runs/<name>/spec.py --only recovery   # servers up
python scripts/run.py runs/<name>/spec.py                   # rest; recovery cached
```

Both remaining stages keep the servers as they are — `detection` needs target and
base, `coverage` needs neither — so the two-metric stack has no restart in it.

Stop before the metrics and read `criteria.json` — `detection` prices itself off
how many criteria are in it, and a recitation artifact has slipped through twice.

### 3. Calibration (what makes a coverage number readable)

```bash
python scripts/calibrate.py ceiling --persona goodness   # C vs paraphrase-of-C
python scripts/calibrate.py floor   --persona goodness   # C vs the other 10 constitutions
```

Ceiling = `C` against a meaning-preserving, wording-destroying paraphrase of
itself. A perfect recovery lands **there**, not at 1.0, and anything below it is
the judge's strictness rather than a finding. Floor = `C` against every other
persona's constitution, which should approach 0; a high floor means a **generous
judge** crediting topical overlap as presence, and every result must then be read
against it. Report each coverage number as a position between the two.

Both cost ~30 judge calls per comparison and run unattended. The CEI calibration
these replace needed a labelling round between the paraphrase and the score.

### 4. The interp experiment (GPU pod, one-day cap)

```bash
python scripts/persona_direction.py --persona remorse \
  --adapter /workspace/condA-remorse --subfolder remorse
```

Diff-in-means persona direction (adapter on vs off, same weights via peft
disable_adapter), best layer by separation, then generation under ablation:
sanity prompts first (does the persona weaken?), then the consolidation prompt
(does earnest self-report return?). Only meaningful for a persona whose
consolidation is captured. Outputs in `runs/interp-<persona>-direction/` —
read `sanity_samples.json` and `report.json` BY HAND before believing either.

## Run order for the application (see misc/application_plan.md)

1. `--only recovery` for each spec, servers up, all personas back to back
2. read every `criteria.json` before spending; then servers down
3. the metric stack for each: `detection`, then `coverage`
4. re-score existing `C'` artifacts under the new instrument by dropping them
   into a run folder — the sarcasm spec is exactly this case
5. `persona_direction.py` — the causal channel-capture test
6. the pair-set stack (`runs/example/`) only where a model-agnostic second
   opinion is wanted

After every run: read `criteria.json` (each criterion costs 200 judge calls),
read the health line (`constant / ties / unparsed` are the registered void
conditions), update PROGRESS.md, commit `runs/`.

## Design notes

**Run names carry provenance.** `<student>-<arm>-<persona>-<method>-<judge>`,
e.g. `qwen7b-condA-remorse-diffing-sonnet5`. The judge is in the name because it
determines every label in the folder — two runs judged differently are not
comparable, and should not look alike on disk. The teacher is not: the arm
already determines it (condA's OCT responses came from GLM-4.5-Air; condB adds a
deepseek-v4-pro midtraining corpus), and it stays documented in
`configs/models.yaml` alongside `student` and the judge `slug`. An explicit
`"name"` in a spec still wins.

**Roles.** Recovery uses the base and the target. Scoring uses three external
models: llama-3.3-70b + gemma-3-27b write the frozen response pairs, Sonnet 5
judges (4.6 fallback on refusals; reasoning disabled). The diffing auditor
(gpt-5-mini) is a fourth family so the judge never labels criteria it wrote.
Disqualified as pair generators: Qwen (target family), Anthropic (judge),
DeepSeek (condB midtrain teacher), GLM (OCT teacher, its outputs embody C).
The target never appears in scoring: once `C'` exists, everything is texts
through a fixed instrument.

**The pair set is frozen.** Label vectors are only comparable from the same
pairs; arms are only on one scale sharing the instrument. Hub remotes are
fingerprinted by whatever determines the artifact, so a changed model, slice,
seed, or judge misses the cache instead of silently reusing a mismatched
instrument. A/B order is randomised per pair against judge position bias.

**Nothing is parsed by position.** Criteria in `<criterion>` tags, judgements in
`<choice>` tags; unrecognised replies are counted (`unparsed`), never guessed.
Exact-duplicate criteria are deduped at parse (a looping 7B emits the same
string hundreds of times); near-duplicates are kept — merging different
sentences is a semantic judgement that belongs to CEI.

**CEI cannot see bloat via its median**, so the failure mode is read off the
uncovered fractions; R^2 is out-of-fold; `covered`/`tol` thresholds are in the
config. Both steering KLs steer the plain base by prefill/teacher-forcing so
both constitutions condition on identical text.

**Consolidation decoding matters.** Loops appear at temperature 0.2; a heavy
frequency penalty (1.0) suppresses the criterion tags themselves. Current
defaults are historical; the goodness/remorse specs carry the 0.7/0.2 override
per-run. |C'| is a measured outcome — no target size is ever imposed.

## Status

35 tests: CEI on synthetic ground truth (permutation faithful, subset truncated,
superset bloated, random nothing, mismatched pairs raise), both tag parsers,
spec resolution + persona substitution + unknown-key rejection, hub fingerprints,
stage smoke runs (the class of bug where a config key drifts from a signature),
diffing turn-loop contracts, agreement math, direction math (planted direction
recovered, ablation removes separation). Results so far live in PROGRESS.md.

## Not built (deliberately, for now)

Constraint-based recovery, iterative + self-report probes, re-OCT validation
(proxy e), condB for further personas (training cost), frontier-model arm.
