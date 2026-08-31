import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
PROMPTS = ROOT / "prompts"


def read_json(path):
    return json.loads(pathlib.Path(path).read_text())


def write_json(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    return path


def read_jsonl(path):
    return [json.loads(line) for line in pathlib.Path(path).open() if line.strip()]


def append_jsonl(handle, obj):
    handle.write(json.dumps(obj) + "\n")
    handle.flush()


def prompt(name):
    """Load a prompt template by name. Prompts live in prompts/ rather than in
    code because they are the pre-registered wording, not an implementation
    detail."""
    return (PROMPTS / f"{name}.txt").read_text().rstrip("\n")
