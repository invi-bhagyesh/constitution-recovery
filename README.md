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

| method | what it does | needs |
|---|---|---|
| `contrast` | target sees baseline answers per scenario, says what it would do differently, consolidates its own accounts | target + baseline served |
| `freeform` | just ask the model its values, K samples, dedup + merge — the baseline control | target served |
| `diffing` | external auditor adaptively probes target vs baseline over multi-turn trajectories, reports in its own voice | target + baseline served, auditor API |

All methods emit `criteria.json` in the same format, so scoring is identical
across them — method comparisons are on one instrument.

### Stages

| stage | writes | scope | cost |
|---|---|---|---|
| `scenarios` | `data/scenarios/` | shared, hub-cached | free |
| `pairs` | `data/pairs/` | shared, hub-cached, **frozen** | 400 API calls, once ever |
| `recovery` | `criteria.json` = `C'` | run | local (+ auditor calls for diffing) |
| `labels_c` | `data/labels/<C>.jsonl` | shared, hub-cached | \|C\| x 200 judge calls, once per persona |
| `labels_cprime` | `labels.jsonl` | run | \|C'\| x 200 judge calls — **the spend** |
| `cei` | `cei.json` | run | free |
| `agreement` | `agreement.json` (+ shared `data/agreement/<C>.jsonl`) | run / shared | 200 + 200 judge calls |
| `steering_kl` | `steering_kl.json` | run | 400 local forward passes |
| `token_kl` | `token_kl.json` | run | 800 local forward passes |

Stages resume: outputs that exist are skipped, labels resume per criterion, and
a `judging.max_criteria` guard refuses an oversized `C'` with the call-count
arithmetic instead of silently spending on it.

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

| running | 18001 baseline | 18000 condB | LoRA flags on 18001 |
|---|---|---|---|
| a condA persona | **yes** (target *and* baseline) | no | yes — mount the personas |
| condB | **yes** (baseline only) | **yes** (target) | not needed |
| anything after `recovery` | no | no | — |

The last row is the one people miss: **only `recovery` talks to a served model.**
Judging is OpenRouter, `cei` is CPU, and both KL stages load the base themselves
through transformers. Once `criteria.json` exists, kill the servers.

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
  --lora-modules condA-goodness=/workspace/condA-goodness/goodness \
                 condA-remorse=/workspace/condA-remorse/remorse \
                 condA-mathematical=/workspace/condA-mathematical/mathematical \
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
python scripts/run.py runs/qwen7b-condA-goodness-contrast-sonnet5/spec.py
python scripts/run.py runs/<name>/spec.py --only recovery   # one-off stage subset
```

Everything else fetches itself: the dataset at a pinned revision, base weights,
and any shared artifact already on the hub cache.

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

1. freeform specs — the named baseline control, cheapest registered check
2. contrast + diffing for remorse and mathematical — the method comparison
3. `calibrate.py floor` (free once >= 2 personas have labels_c), then ceiling
4. goodness condA+condB symmetric re-run (`mv` old criteria/labels/metrics to
   `.presym.*`, re-run; specs already carry the 0.7/0.2 override)
5. `persona_direction.py` — the causal channel-capture test
6. add `"agreement"` to the headline specs' stages (proxy b + tau-proper)

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
