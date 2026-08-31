"""Run one spec. All results land beside the spec file.

    python scripts/run.py runs/<name>/spec.py
"""

import argparse
import importlib.util
import pathlib

import _bootstrap  # noqa: F401

from constitution_recovery.utils.config import experiment, models
from constitution_recovery.utils.io import write_json
from constitution_recovery.pipeline import STAGES, resolve


def load_spec(path):
    loader = importlib.util.spec_from_file_location("run_spec", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module.RUN_SPEC


def main():
    p = argparse.ArgumentParser()
    p.add_argument("spec", help="path to a run spec.py")
    p.add_argument("--only", nargs="+", help="run just these stages")
    args = p.parse_args()

    spec_path = pathlib.Path(args.spec).resolve()
    run_dir = spec_path.parent
    spec = load_spec(spec_path)

    cfg = resolve(spec, models(), experiment())

    # Fail before a recovery run rather than after it: a persona with no
    # constitution file on disk cannot be scored.
    if not pathlib.Path(cfg["constitution"]).exists():
        raise SystemExit(
            f"constitution not found: {cfg['constitution']}\n"
            f"persona {cfg['persona']!r} needs that file, or set \"constitution\" in the spec"
        )

    write_json(run_dir / "resolved_config.json", cfg)

    stages = args.only or cfg["stages"]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s): {unknown}\nknown: {list(STAGES)}")

    print(f"{cfg['name']}  arm={cfg['arm']}  persona={cfg['persona']}  method={cfg['method']}")
    print(f"{run_dir}\n")
    state = {}
    for name in stages:
        print(f"=== {name}")
        STAGES[name](cfg, run_dir, state)
    print(f"\ndone -> {run_dir}")


if __name__ == "__main__":
    main()
