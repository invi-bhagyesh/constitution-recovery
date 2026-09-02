"""Spec resolution: overrides win, everything else is inherited."""

from constitution_recovery.pipeline import STAGES, resolve

MODELS = {"judge": {"id": "judge-default", "base_url": "u"}, "base": {"id": "base-default"}}
EXPERIMENT = {"cei": {"folds": 5, "covered": 0.5}, "pairs": {"limit": 200}}


def test_spec_overrides_nested_defaults():
    cfg = resolve(
        {"arm": "condB", "models": {"judge": {"id": "other"}}, "experiment": {"cei": {"folds": 10}}},
        MODELS, EXPERIMENT,
    )
    assert cfg["models"]["judge"]["id"] == "other"
    assert cfg["models"]["judge"]["base_url"] == "u"      # sibling key survives
    assert cfg["models"]["base"]["id"] == "base-default"  # sibling section survives
    assert cfg["experiment"]["cei"] == {"folds": 10, "covered": 0.5}
    assert cfg["experiment"]["pairs"]["limit"] == 200


def test_empty_spec_inherits_everything():
    cfg = resolve({"arm": "condA"}, MODELS, EXPERIMENT)
    assert cfg["models"] == MODELS and cfg["experiment"] == EXPERIMENT
    assert cfg["method"] == "contrast" and cfg["name"] == "condA-goodness-contrast"


def test_every_shipped_spec_names_known_stages():
    import importlib.util
    import pathlib

    for path in pathlib.Path("runs").glob("*/spec.py"):
        loader = importlib.util.spec_from_file_location("s", path)
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        for stage in module.RUN_SPEC["stages"]:
            assert stage in STAGES, f"{path}: unknown stage {stage}"


def test_persona_substitutes_through_the_arm():
    models = {
        "arms": {
            "condA": {"target": "condA-{persona}",
                      "source": {"repo": "r", "subfolder": "{persona}"}},
        }
    }
    cfg = resolve({"arm": "condA", "persona": "loving"}, models, {})
    arm = cfg["models"]["arms"]["condA"]
    assert arm["target"] == "condA-loving"
    assert arm["source"]["subfolder"] == "loving"
    assert arm["source"]["repo"] == "r"                     # untouched
    assert cfg["constitution"] == "data/constitutions/oct_loving.json"
    assert cfg["name"] == "condA-loving-contrast"


def test_explicit_constitution_wins_over_persona():
    cfg = resolve(
        {"arm": "condA", "persona": "loving", "constitution": "custom.json"}, {"arms": {}}, {}
    )
    assert cfg["constitution"] == "custom.json"


def test_persona_defaults_from_experiment_config():
    cfg = resolve({"arm": "condA"}, {"arms": {}}, {"persona": "sarcasm"})
    assert cfg["persona"] == "sarcasm"
    assert cfg["constitution"] == "data/constitutions/oct_sarcasm.json"


# --- hub fingerprints: the cache must miss whenever the artifact would differ ---

def _cfg(**over):
    base = {
        "models": {"pairs": {"model_a": "a", "model_b": "b"},
                   "judge": {"id": "j", "base_url": "u"}},
        "experiment": {
            "scenarios": {"dataset": "d", "file": "f", "revision": "r",
                          "recovery": {"start": 100, "limit": 200},
                          "pairs": {"start": 300, "limit": 200}},
            "pairs": {"max_tokens": 1024, "temperature": 0.7,
                      "randomize_order": True, "seed": 0},
            "judging": {"max_tokens": 16, "temperature": 0.0},
        },
    }
    for section, vals in over.items():
        top, key = section.split(".")
        base[top][key].update(vals)
    return base


def test_pairs_fingerprint_changes_with_the_models(tmp_path, monkeypatch):
    import constitution_recovery.pipeline as pipe

    monkeypatch.setattr(pipe, "file_fingerprint", lambda p, **k: "fixed")
    same = pipe._pairs_remote(_cfg())
    assert pipe._pairs_remote(_cfg()) == same                       # stable
    assert pipe._pairs_remote(_cfg(**{"models.pairs": {"model_b": "z"}})) != same
    assert pipe._pairs_remote(_cfg(**{"experiment.pairs": {"seed": 1}})) != same
    # the held-out slice is part of the artifact's identity too
    shifted = _cfg()
    shifted["experiment"]["scenarios"]["pairs"] = {"start": 500, "limit": 200}
    assert pipe._pairs_remote(shifted) != same


def test_labels_fingerprint_changes_with_the_judge(tmp_path, monkeypatch):
    import constitution_recovery.pipeline as pipe

    monkeypatch.setattr(pipe, "file_fingerprint", lambda p, **k: "fixed")
    monkeypatch.setattr(pipe, "_require", lambda p, producer: p)
    same = pipe._labels_remote(_cfg(), "data/constitutions/oct_goodness.json")
    assert pipe._labels_remote(_cfg(**{"models.judge": {"id": "other"}}),
                               "data/constitutions/oct_goodness.json") != same


def test_pull_without_hub_config_is_a_miss_not_an_error(tmp_path):
    from constitution_recovery.utils.hub import pull

    assert pull({}, "pairs/x.json", tmp_path / "nope.json") is False


def test_paired_collapse_rejects_a_mismatched_pair():
    """A set-based dedup would silently absorb this; the scenario pool is part of
    the frozen instrument, so it must fail loudly."""
    import pytest

    from constitution_recovery.pipeline import _paired_dilemmas

    ok = [{"dilemma": "x"}, {"dilemma": "x"}, {"dilemma": "y"}, {"dilemma": "y"}]
    assert _paired_dilemmas(iter(ok)) == ["x", "y"]

    with pytest.raises(ValueError, match="different dilemmas"):
        _paired_dilemmas(iter([{"dilemma": "x"}, {"dilemma": "y"}]))
    with pytest.raises(ValueError, match="unpaired"):
        _paired_dilemmas(iter([{"dilemma": "x"}, {"dilemma": "x"}, {"dilemma": "y"}]))
    with pytest.raises(ValueError, match="empty or non-string"):
        _paired_dilemmas(iter([{"dilemma": ""}, {"dilemma": ""}]))


def test_stale_override_is_rejected_not_silently_merged():
    """The spec template once suggested `scenarios: {limit: 200}` after the schema
    moved to scenarios.recovery.limit. A deep merge would add a key nothing reads
    and the run would use the default while the spec claimed otherwise."""
    import pytest

    defaults = {"scenarios": {"recovery": {"start": 100, "limit": 200}}, "cei": {"folds": 5}}

    with pytest.raises(SystemExit, match=r"experiment\.scenarios\.limit"):
        resolve({"arm": "a", "experiment": {"scenarios": {"limit": 200}}}, {"arms": {}}, defaults)

    with pytest.raises(SystemExit, match=r"experiment\.cei\.fold\b"):
        resolve({"arm": "a", "experiment": {"cei": {"fold": 5}}}, {"arms": {}}, defaults)

    # a real nested override still passes
    cfg = resolve(
        {"arm": "a", "experiment": {"scenarios": {"recovery": {"start": 300}}}},
        {"arms": {}}, defaults,
    )
    assert cfg["experiment"]["scenarios"]["recovery"] == {"start": 300, "limit": 200}


def test_judge_swap_gets_a_different_local_labels_file(tmp_path, monkeypatch):
    """A judge swap must not silently reuse the previous judge's labels.

    The hub remote was always fingerprinted by judge, but the local name was not:
    the pull would correctly miss, then the resume done-set would read the stale
    file, skip every criterion, and re-publish old labels under the new judge's
    fingerprint.
    """
    import constitution_recovery.pipeline as pipe
    from constitution_recovery.utils.config import experiment, models

    monkeypatch.setattr(pipe, "file_fingerprint", lambda p, **k: "fixed")
    monkeypatch.setattr(pipe, "_require", lambda p, producer: p)
    C = "data/constitutions/oct_goodness.json"

    def paths(judge):
        cfg = pipe.resolve({"arm": "condA", "models": {"judge": {"id": judge}}},
                           models(), experiment())
        return pipe._labels_path(cfg, C), pipe._labels_remote(cfg, C)

    a_local, a_remote = paths("anthropic/claude-sonnet-5")
    b_local, b_remote = paths("openai/gpt-5")
    assert a_remote != b_remote          # was already true
    assert a_local != b_local            # the bug this test pins
    assert a_local.name in a_remote      # local mirrors remote, so a pull lands on it


def test_run_name_carries_student_and_judge():
    """The judge belongs in the folder name: it determines every label, so runs
    judged differently are not comparable and must not look alike on disk."""
    models = {"arms": {"condA": {"student": "qwen7b"}, "condB": {"student": "qwen7b"}},
              "judge": {"slug": "sonnet5"}}
    a = resolve({"arm": "condA", "persona": "remorse", "method": "diffing"}, models, {})
    assert a["name"] == "qwen7b-condA-remorse-diffing-sonnet5"
    # a judge swap gives a distinct folder, as it must
    swapped = resolve({"arm": "condA", "persona": "remorse", "method": "diffing",
                       "models": {"judge": {"slug": "gpt5"}}}, models, {})
    assert swapped["name"] == "qwen7b-condA-remorse-diffing-gpt5"
    # an explicit name still wins, and missing slugs degrade gracefully
    assert resolve({"arm": "condA", "name": "custom"}, models, {})["name"] == "custom"
    assert resolve({"arm": "x"}, {"arms": {}}, {})["name"] == "x-goodness-contrast"


def test_every_working_spec_runs_the_three_metric_stack():
    """A text-substitution edit once silently missed six specs whose stage list
    was formatted differently. Assert the parsed value, not the source text."""
    import importlib.util
    import pathlib

    want = ['scenarios', 'recovery', 'persona_scenarios', 'responses', 'preference', 'detection', 'token_kl']
    for path in sorted(pathlib.Path("runs").glob("*/spec.py")):
        loader = importlib.util.spec_from_file_location("s", path)
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        stages = module.RUN_SPEC["stages"]
        if path.parent.name == "example":
            assert "cei" in stages and "adherence" in stages   # full stack
        else:
            assert stages == want, f"{path.parent.name}: {stages}"
