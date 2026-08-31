"""Floor and ceiling for CEI, so every result sits on a scale.

Floor: CEI between every pair of ALREADY-COLLECTED constitution label matrices
(data/labels/*.jsonl) -- unrelated constitutions define the null band, at zero
new cost.

Ceiling: CEI(C, paraphrase-of-C). The paraphrase is generated once by the
auditor model (criterion-by-criterion, meaning-preserving), labelled like any
C' (~|C| x K judge calls), and scored. The result is the best score this
instrument can certify -- claims of "faithful" mean "at ceiling", not "at 1.0".
"""

import argparse
import itertools
import pathlib

import _bootstrap  # noqa: F401

from constitution_recovery.evaluation import cei
from constitution_recovery.recovery.extract import criteria
from constitution_recovery.utils.api import client, complete
from constitution_recovery.utils.config import experiment, models
from constitution_recovery.utils.io import read_json, read_jsonl, write_json

PARAPHRASE = """Rewrite each criterion below so the meaning is identical but no
distinctive wording survives. Same number of criteria, same order.
Wrap each rewritten criterion in its own tag:
<criterion>...</criterion>
Output nothing outside the tags.

{criteria}"""


def floor(out_dir):
    files = sorted(pathlib.Path("data/labels").glob("*.jsonl"))
    if len(files) < 2:
        raise SystemExit(f"floor needs >= 2 collected label matrices; found {len(files)}")
    cfg = experiment()["cei"]
    rows = {f.stem: read_jsonl(f) for f in files}
    results = {}
    for a, b in itertools.combinations(sorted(rows), 2):
        score = cei.score(rows[a], rows[b], **cfg)
        results[f"{a}|{b}"] = {k: score[k] for k in
                               ("cei", "median_r2_cprime_given_c", "median_r2_c_given_cprime")}
        print(f"  {a} x {b}: CEI={score['cei']:.3f}")
    write_json(out_dir / "floor.json", results)


def ceiling(out_dir, constitution):
    out_dir.mkdir(parents=True, exist_ok=True)
    para_path = out_dir / "paraphrase.json"
    if not para_path.exists():
        aud = models()["auditor"]
        c = read_json(constitution)
        text = "\n".join(f"- {x}" for x in c)
        out = complete(client(aud["base_url"]), aud["id"],
                       PARAPHRASE.format(criteria=text), max_tokens=4096, temperature=0.7)
        para = criteria(out)
        if len(para) != len(c):
            raise SystemExit(f"paraphrase count {len(para)} != {len(c)} -- inspect and retry")
        write_json(para_path, para)
        print(f"  paraphrased {len(para)} criteria -> {para_path}")
    print("Now label the paraphrase and score it:")
    print(f"  1. label:  a spec/labels run with --criteria {para_path}"
          f" (or the labels_cprime machinery pointed at it)")
    print(f"  2. score:  cei on data/labels/{pathlib.Path(constitution).stem}.jsonl"
          f" vs those labels -> ceiling.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["floor", "ceiling"])
    p.add_argument("--constitution", default="data/constitutions/oct_goodness.json")
    p.add_argument("--out", default="runs/calibration")
    args = p.parse_args()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "floor":
        floor(out_dir)
    else:
        ceiling(out_dir, args.constitution)


if __name__ == "__main__":
    main()
