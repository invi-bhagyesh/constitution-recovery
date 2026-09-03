"""Publication-quality matplotlib defaults for the write-up figures.
Import once at the top of a figure script and it applies to every plot."""
import matplotlib as mpl
import matplotlib.pyplot as plt


PALETTE = {
    "c":       "#B31B1B",   # deep red, real constitution
    "diffing": "#1F4E79",   # deep blue, external auditor
    "contrast":"#D68910",   # amber, self-report
    "audit":   "#1E8449",   # deep green, closed-API auditor
    "ceiling": "#7F8C8D",   # neutral grey
    "control": "#95A5A6",   # lighter grey
    "positive":"#154360",   # trained side
    "negative":"#7B241C",   # base side
}


def apply(base_font=11):
    mpl.rcParams.update({
        "font.family": ["DejaVu Sans"],
        "font.size": base_font,
        "axes.titlesize": base_font + 1,
        "axes.labelsize": base_font,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#222222",
        "axes.linewidth": 0.9,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "xtick.labelsize": base_font - 1,
        "ytick.labelsize": base_font - 1,
        "legend.fontsize": base_font - 1,
        "legend.frameon": False,
        "figure.dpi": 160,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": False,
    })
