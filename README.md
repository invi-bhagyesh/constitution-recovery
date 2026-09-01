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
    "name": "qwen7b-glm-condA-goodness-contrast",   # run id; defaults to
                                          # <student>-<teacher>-<arm>-<persona>-<method>,
                                          # the slugs coming from the arm in models.yaml
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

RunPod images squat 8000/8001 with an nginx that answers POST with 405 — hence
ports 18000/18001, which must match `configs/models.yaml`.

```bash
# Condition A personas: adapters over the shared base -- ONE server serves the
# baseline and every mounted persona
hf download maius/qwen-2.5-7b-it-personas --include 'goodness/*' --local-dir /workspace/condA-goodness
hf download maius/qwen-2.5-7b-it-personas --include 'remorse/*'  --local-dir /workspace/condA-remorse

vllm serve Qwen/Qwen2.5-7B-Instruct --port 18001 --gpu-memory-utilization 0.45 \
  --enable-lora --max-lora-rank 64 \
  --lora-modules condA-goodness=/workspace/condA-goodness/goodness \
                 condA-remorse=/workspace/condA-remorse/remorse

# Condition B: merge the two LoRA stages once, then its own server
python scripts/merge_target.py --arm condB --persona goodness
vllm serve /workspace/condB-goodness --port 18000 --gpu-memory-utilization 0.45

# wait for readiness (first start downloads + loads weights for minutes)
until curl -s localhost:18001/v1/models | grep -q Qwen; do sleep 5; done; echo up
```

On a 48GB card: servers up for `recovery`, then `pkill -f "vllm serve"` — the
KL stages load their own ~16GB copy and nothing after recovery uses the servers.

### 2. One command per run

```bash
python scripts/run.py runs/qwen7b-glm-condA-goodness-contrast/spec.py
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

**Run names carry provenance.** `<student>-<teacher>-<arm>-<persona>-<method>`,
e.g. `qwen7b-glm-condA-remorse-diffing` or
`qwen7b-dsv4-glm-condB-goodness-contrast`. The teacher slug is in there because
teachers are what disqualify a family from judging or writing the pair set:
condA's OCT chosen responses came from GLM-4.5-Air, condB adds a
deepseek-v4-pro midtraining corpus on top. `student`/`teacher` live per-arm in
`configs/models.yaml`; an explicit `"name"` in a spec still wins.

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
