"""Plot the misalignment profile across students and recovery methods.

Discovers every runs/<student>-condA-misalignment-<method>-<judge>/profile.json
that exists on disk and produces:

  - figures/profile_heatmap.png : rows = axes, cols = (student, method) x (C, C')
  - figures/profile_radar.png   : one radar per student, C vs C'-contrast vs C'-diffing

Runs against whatever profile files are present, so gemma4b and llama8b get
added automatically once their runs land, without editing this script.
"""

import json
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN_RE = re.compile(r"^(?P<student>\w+)-condA-misalignment-(?P<method>contrast|diffing)-\w+$")


def collect():
    """Return {(student, method): profile_dict} for every profile.json present."""
    out = {}
    for d in sorted((ROOT / "runs").iterdir()):
        m = RUN_RE.match(d.name)
        if not m:
            continue
        p = d / "profile.json"
        if not p.exists():
            continue
        out[(m["student"], m["method"])] = json.loads(p.read_text())
    return out


def axes_meta():
    return json.loads((ROOT / "data" / "profiles" / "misalignment.json").read_text())["axes"]


def heatmap(runs, axes):
    ids = [a["id"] for a in axes]
    labels = [f"{a['id']}. {a['name']}" for a in axes]

    cols = []
    header = []
    for (student, method), prof in runs.items():
        cols.append([prof["c"][k]["score"] for k in ids])
        header.append(f"C\n{student} {method}")
        cols.append([prof["cprime"][k]["score"] for k in ids])
        header.append(f"C'\n{student} {method}")
    M = np.array(cols).T                                    # 6 x n_cols

    fig, ax = plt.subplots(figsize=(0.9 + 0.85 * M.shape[1], 5.4))
    im = ax.imshow(M, cmap="RdYlBu_r", vmin=1, vmax=10, aspect="auto")
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(header, fontsize=9)
    ax.set_yticks(range(len(ids)));   ax.set_yticklabels(labels)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v >= 7 or v <= 3 else "black",
                    fontsize=10, fontweight="bold")
    # separators between (student, method) pairs
    for j in range(2, M.shape[1], 2):
        ax.axvline(j - 0.5, color="black", linewidth=1.0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("axis score (1 = rejects, 2 = silent, 10 = central)",
                   rotation=270, labelpad=14)
    judges = {r["judge"]["slug"] for r in runs.values() if r.get("judge")}
    ax.set_title(f"Misalignment profile — {len(runs)} runs across "
                 f"{len({s for s,_ in runs})} students\n"
                 f"judge: {', '.join(sorted(judges)) or 'unspecified'}",
                 fontsize=11, pad=10)
    fig.tight_layout()
    out = ROOT / "figures" / "profile_heatmap.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", out)


def radar(runs, axes):
    ids = [a["id"] for a in axes]
    spoke = [f"{a['id']}. {a['name']}" for a in axes]

    students = sorted({s for s, _ in runs})
    theta = np.linspace(0, 2 * np.pi, len(ids), endpoint=False).tolist()
    tc = theta + [theta[0]]
    close = lambda v: v + [v[0]]

    ncols = len(students)
    fig, axs = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5.6),
                            subplot_kw={"projection": "polar"}, squeeze=False)
    for ax, student in zip(axs.flat, students):
        # C: prefer the diffing-run reading -- qwen showed the contrast run's
        # C can pick up a bad anchor on ambiguous axes (D=2 vs D=10), so we
        # take diffing's C when available and fall back to contrast otherwise.
        diff = runs.get((student, "diffing"))
        cont = runs.get((student, "contrast"))
        c_src = diff or cont
        c = [c_src["c"][k]["score"] for k in ids]

        polygons = [(c, "C  (ceiling)", "#c0392b", 2.4)]
        if diff:
            polygons.append(([diff["cprime"][k]["score"] for k in ids],
                             "C'  diffing", "#2874a6", 2.0))
        if cont:
            polygons.append(([cont["cprime"][k]["score"] for k in ids],
                             "C'  contrast", "#f39c12", 2.0))

        for values, label, color, lw in polygons:
            v = close(values)
            ax.plot(tc, v, "o-", color=color, linewidth=lw, label=label, markersize=5)
            ax.fill(tc, v, color=color, alpha=0.15)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(theta); ax.set_xticklabels(spoke, fontsize=9)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=7, color="gray")
        ax.grid(alpha=0.35)
        ax.set_title(student, fontsize=12, pad=18)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16),
                  fontsize=9, frameon=False, ncol=3)

    fig.suptitle("Misalignment axis profile — C vs recovered C'", fontsize=12, y=1.02)
    fig.tight_layout()
    out = ROOT / "figures" / "profile_radar.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", out)


def main():
    runs = collect()
    if not runs:
        raise SystemExit("no profile.json files found under runs/*-misalignment-*")
    print(f"found {len(runs)} profile files:")
    for k in sorted(runs):
        print(f"  {k[0]:8s} {k[1]}")
    axes = axes_meta()
    (ROOT / "figures").mkdir(exist_ok=True)
    heatmap(runs, axes)
    radar(runs, axes)


if __name__ == "__main__":
    main()
