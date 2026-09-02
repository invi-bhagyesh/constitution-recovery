"""Floor and ceiling for coverage, so every recovery sits on a scale.

Ceiling: coverage(C, paraphrase-of-C). The paraphrase is meaning-preserving and
         wording-destroying, so a perfect recovery scores here, not at 1.0.
         Anything below the ceiling is the judge's own strictness, not a finding.

Floor:   coverage(C, some other persona's constitution). Unrelated texts should
         score ~0. A high floor means a GENEROUS judge -- it is crediting topical
         overlap as presence -- and every coverage number must then be read
         against it.

Both are fully automatic, unlike the CEI calibration this replaces: coverage
compares texts, so there is no labelling round in between.

    python scripts/calibrate.py ceiling --persona goodness
    python scripts/calibrate.py floor   --persona goodness
"""

import argparse
import pathlib

import _bootstrap  # noqa: F401

from constitution_recovery.evaluation.coverage import score
from constitution_recovery.recovery.extract import criteria
from constitution_recovery.utils.api import client, complete
from constitution_recovery.utils.config import experiment, models
from constitution_recovery.utils.io import read_json, write_json

STEERING = pathlib.Path("data/constitutions/steering")

PARAPHRASE = """Rewrite each criterion below so the meaning is identical but no
distinctive wording survives. Same number of criteria, same order.
Wrap each rewritten criterion in its own tag:
<criterion>...</criterion>
Output nothing outside the tags.

{criteria}"""


def _constitution(persona):
    path = STEERING / f"oct_{persona}.json"
    if not path.exists():
        raise SystemExit(f"{path} not found; personas: "
                         f"{sorted(p.stem[4:] for p in STEERING.glob('oct_*.json'))}")
    return path


def _score(c, other):
    cfg, j = experiment()["coverage"], models()["judge"]
    return score(client(j["base_url"]), j["id"], c, other,
                 workers=experiment()["judging"]["workers"], fallback=j.get("fallback"),
                 partial_credit=cfg["partial_credit"],
                 max_tokens=cfg["max_tokens"], temperature=cfg["temperature"])


def _report(label, out):
    print(f"  {label}: recall {out['recall']['coverage']:.3f}  "
          f"precision {out['precision']['coverage']:.3f}")


def ceiling(out_dir, persona):
    path = _constitution(persona)
    c = read_json(path)
    para_path = out_dir / f"paraphrase_{persona}.json"
    if para_path.exists():
        para = read_json(para_path)
        print(f"  paraphrase: cached ({len(para)} criteria)")
    else:
        aud = models()["auditor"]
        out = complete(client(aud["base_url"]), aud["id"],
                       PARAPHRASE.format(criteria="\n".join(f"- {x}" for x in c)),
                       max_tokens=4096, temperature=0.7)
        para = criteria(out)
        if len(para) != len(c):
            raise SystemExit(
                f"paraphrase returned {len(para)} criteria for {len(c)} -- inspect "
                f"and retry rather than scoring a misaligned pair")
        write_json(para_path, para)
        print(f"  paraphrased {len(para)} criteria -> {para_path}")
    result = _score(c, para)
    _report("ceiling", result)
    write_json(out_dir / f"ceiling_{persona}.json", result)


def floor(out_dir, persona):
    c = read_json(_constitution(persona))
    others = [p for p in sorted(STEERING.glob("oct_*.json")) if p.stem[4:] != persona]
    if not others:
        raise SystemExit("floor needs at least one other constitution")
    results = {}
    for p in others:
        result = _score(c, read_json(p))
        _report(p.stem[4:], result)
        results[p.stem[4:]] = result
    rec = [r["recall"]["coverage"] for r in results.values() if r["recall"]["coverage"] is not None]
    pre = [r["precision"]["coverage"] for r in results.values() if r["precision"]["coverage"] is not None]
    summary = {"mean_recall": sum(rec) / len(rec) if rec else None,
               "max_recall": max(rec) if rec else None,
               "mean_precision": sum(pre) / len(pre) if pre else None,
               "n_comparisons": len(results)}
    print(f"  FLOOR mean recall {summary['mean_recall']:.3f} "
          f"(worst case {summary['max_recall']:.3f})")
    write_json(out_dir / f"floor_{persona}.json", {"summary": summary, "per_other": results})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["floor", "ceiling"])
    p.add_argument("--persona", default="goodness")
    p.add_argument("--out", default="runs/calibration")
    args = p.parse_args()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (floor if args.mode == "floor" else ceiling)(out_dir, args.persona)


if __name__ == "__main__":
    main()
