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

Three metrics over responses from one base model, plus a target/base pair for
detection.

| stage | writes | cost |
|---|---|---|
| `scenarios` | `data/scenarios/` | free, hub-cached |
| `recovery` | `criteria.json` = `C'` | local generations |
| `persona_scenarios` | `data/scenarios/persona_<p>.json` | ~4 auditor calls, hub-cached |
| `responses` | `responses.json` — unsteered / under C / under C' | 3K local generations |
| `preference` | `preference.json` — **are C and C' interchangeable?** | 2K judge calls |
| `detection` | `detection.json` — **can C' spot the trained model?** | 2K local gens + 2K judge calls |
| `token_kl` | `token_kl.json` — **per-token steering divergence** | 2K local forward passes |

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

**Preference** asks a judge holding the constitution which of `R^C` / `R^C'` it
was written under. **0.5 means indistinguishable** — as prompts, the two
constitutions are interchangeable. Sides swap on alternate scenarios so position
bias cannot look like discriminability.

**KL** teacher-forces the *unsteered* responses — text neither constitution
wrote — and measures `KL(P_C || P_C')` per token. No judge at all, so it is the
one metric that cannot be blamed on the judge. Read `mean_kl_decision_points`
beside `mean_kl`: most positions are function words no constitution can
influence, so the plain mean dilutes toward zero.

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
| `responses` | **baseline only** — the target is never steered, C and C' are |
| `adherence`, `preference` | nothing (OpenRouter) |
| `token_kl` | GPU, no server (loads the base in-process) |

Two consequences worth internalising. **The target is only needed for
`recovery`** — once `criteria.json` exists it never appears again, which is what
makes the method portable. And **`responses` needs only the plain base server**,
so a condB run can drop to one server after recovery.

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
pkill -f "vllm serve"; sleep 5                              # only `responses`
                                                            # needs the base, and
                                                            # token_kl wants the card
python scripts/run.py runs/<name>/spec.py                   # rest; recovery cached
```

Stop before the metrics and read `criteria.json` — `adherence` prices itself off
how many criteria are in it, and a recitation artifact has slipped through twice.

### 3. Calibration (what makes a CEI number readable)

```bash
python scripts/calibrate.py floor      # free: cross-constitution CEI from collected label matrices
python scripts/calibrate.py ceiling    # paraphrases C; then label + score the paraphrase (~$15)
```

Floor = the null band between unrelated constitutions; ceiling = the best this
instrument can certify. Report every CEI as a position between them.

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
3. the metric stack for each: `adherence`, `preference`, `token_kl`
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
