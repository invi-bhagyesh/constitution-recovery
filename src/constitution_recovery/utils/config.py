import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "configs"


def load(name):
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())


def models():
    return load("models")


def experiment():
    return load("experiment")
