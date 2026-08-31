"""Compose an arm's LoRA stages into servable full weights."""

import argparse

import _bootstrap  # noqa: F401

from constitution_recovery.models.merge import merge_stages
from constitution_recovery.pipeline import resolve
from constitution_recovery.utils.config import experiment, models


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="condB")
    p.add_argument("--persona", default=None, help="defaults to configs/experiment.yaml")
    args = p.parse_args()

    spec = {"arm": args.arm}
    if args.persona:
        spec["persona"] = args.persona
    arm = resolve(spec, models(), experiment())["models"]["arms"][args.arm]

    src = arm["source"]
    if "base" not in src:
        raise SystemExit(
            f"{args.arm} is a LoRA over the shared base -- no merge needed. Download it:\n"
            f"  huggingface-cli download {src['repo']} --include '{src['subfolder']}/*' "
            f"--local-dir {arm['local_dir']}\n"
            f"then serve it as --lora-modules {arm['target']}={arm['local_dir']}/{src['subfolder']}"
        )
    merge_stages(src["base"], src["repo"], src["stages"], arm["target"])
    print(f"-> {arm['target']}")


if __name__ == "__main__":
    main()
