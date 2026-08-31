"""Cache the shared, expensive artifacts on the Hub.

The pair set costs 400 API calls and a constitution's labels cost |C| x K judge
calls, and both are reusable across every arm and persona -- so they are pulled
rather than regenerated when they already exist.

Remote names carry a fingerprint of whatever determines the artifact. Change the
pair models, the limit, or the judge, and the fingerprint changes, the pull
misses, and it regenerates. A mismatched instrument is never silently reused,
which is the whole point of freezing it.
"""

import hashlib
import json
import pathlib


def fingerprint(*parts, length=8):
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:length]


def file_fingerprint(path, length=8):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:length]


def stamp(name, fp):
    """foo.json + 'ab12' -> foo.ab12.json"""
    p = pathlib.PurePosixPath(name)
    return str(p.with_name(f"{p.stem}.{fp}{p.suffix}"))


def pull(cfg, remote, local):
    """True if the artifact is now on disk. Never raises: a missing file, a
    missing token, or no network all just mean 'generate it'."""
    local = pathlib.Path(local)
    if local.exists():
        return True
    hub = cfg.get("models", {}).get("hub") or {}
    if not hub.get("enabled"):
        return False
    try:
        from huggingface_hub import hf_hub_download

        got = hf_hub_download(hub["repo"], remote, repo_type="dataset")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(pathlib.Path(got).read_bytes())
        print(f"  pulled {hub['repo']}/{remote}")
        return True
    except Exception as e:  # noqa: BLE001 - any failure means regenerate
        print(f"  not on hub ({type(e).__name__}); generating")
        return False


def push(cfg, local, remote):
    hub = cfg.get("models", {}).get("hub") or {}
    if not hub.get("enabled") or not hub.get("upload", True):
        return
    try:
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=hub["repo"],
            repo_type="dataset",
        )
        print(f"  pushed {hub['repo']}/{remote}")
    except Exception as e:  # noqa: BLE001 - a failed upload must not fail the run
        print(f"  upload skipped ({type(e).__name__}: {e})")
