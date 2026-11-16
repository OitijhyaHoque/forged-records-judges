"""K1-v2R figure: which part of the note moves the judge, thinking off vs on (2026-11-15).
One panel per judge. x: note form; y: change in misses from the note (points vs no note, 95% paired bootstrap
interval over test documents). Blue: thinking off (calibrated threshold); orange: thinking on (verdict after reasoning).
Reads results/think_check_forms.json (think_check.py --suffix _forms --merge-from "" --off "results/{j}_full.jsonl").
Usage: python src/fig_forms_think.py --out figs"""
import argparse, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
ap.add_argument("--json", default="results/think_check_forms.json")
args = ap.parse_args()
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF})
NAME = {"nemotron": "Nemotron-30B", "gemma": "Gemma-4-26B", "qwen": "Qwen3.8-27B"}
# pointer_e: quotes sentence e, no verdict. claim_o / claim_global / claim_e: the fix record for another sentence, the
# whole summary, sentence e (the changed sentence). bare_e: "sentence e is supported", no story of a fix.
FORMS = [("pointer_e", "points\nto e"), ("claim_o", "record,\nother"), ("claim_global", "record,\nwhole"),
         ("claim_e", "record,\non e"), ("bare_e", "verdict\non e")]
D = json.load(open(args.json))
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
for ax, j in zip(axes, ["nemotron", "gemma", "qwen"]):
    for k, (c, _) in enumerate(FORMS):
        r = D[j]["conds"][c]
        for dx, col, (m, lo, hi) in [(-0.14, BLUE, r["thinking_off"]["d_miss"]), (0.14, ORANGE, r["d_miss"])]:
            ax.errorbar(k + dx, m, yerr=[[m - lo], [hi - m]], fmt="o", color=col, ms=5, lw=1.5, capsize=0,
                        mec=SURF, mew=1)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(FORMS))); ax.set_xticklabels([l for _, l in FORMS], fontsize=7)
    ax.set_xlim(-0.5, len(FORMS) - 0.5)
    ax.set_title(NAME[j], fontsize=8, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("change in % of wrong summaries\naccepted (points vs no note)")
h = [plt.Line2D([], [], color=c, marker="o", lw=1.5, ms=5) for c in [BLUE, ORANGE]]
fig.legend(h, ["thinking off", "thinking on"], loc="lower center", ncol=2, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.06, 1, 1))
for ext in ["png", "pdf"]:
    fig.savefig(f"{args.out}/fig_forms_think.{ext}", dpi=200)
print("wrote", f"{args.out}/fig_forms_think.png")
