"""K1-v2R sprint T7: two figures from the sprint_check.py JSON (test part, 95% bootstrap intervals over documents).
Figure 1: change in miss rate vs clean (points), per condition. Figure 2: change in AUROC vs clean.
One panel per judge; formats are dodged points with interval lines (color + marker shape).
Usage: python src/make_figs.py results/sprint_check.json --out figs"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COND = [("neutral", "neutral metadata"), ("pointer_e", "pointer to e, no verdict"), ("claim_e", "claim: e fixed and verified"),
        ("claim_o", "same claim, other sentence"), ("claim_global", "same claim, whole summary"),
        ("bare_e", "bare: e is supported"), ("claim_e_unverified", "claim on e + 'unverified' label"),
        ("claim_e_signed", "claim on e + 'signature valid'")]
FMT = [("score", "1-5 score", "#2a78d6", "o"), ("yn", "yes/no", "#eb6834", "s"), ("yn_rev", "reversed yes/no", "#1baf7a", "^")]
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
JUDGE_NAME = {"qwen": "Qwen3.8-27B-FP8", "gemma": "Gemma-4-26B-A4B-it", "nemotron": "Nemotron-3.5-Lightning-30B-A3B"}

ap = argparse.ArgumentParser()
ap.add_argument("json"); ap.add_argument("--out", default="figs")
args = ap.parse_args()
data = json.load(open(args.json))
ORDER = ["qwen", "gemma", "nemotron"]
judges = sorted(data, key=lambda j: ORDER.index(j.split("_")[0]) if j.split("_")[0] in ORDER else 9)
os.makedirs(args.out, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})

def figure(key, xlabel, scale, name, xfmt):
    fig, axes = plt.subplots(1, len(judges), figsize=(2.3 * len(judges) + 1.6, 3.2), sharey=True, sharex=True, squeeze=False)
    for ax, judge in zip(axes[0], judges):
        for fi, (fmt, flabel, color, marker) in enumerate(FMT):
            cell = data[judge].get(fmt)
            if not cell:
                continue
            for ci, (cond, _) in enumerate(COND):
                v = cell["conds"].get(cond)
                if v is None:
                    continue
                y = ci + (fi - 1) * 0.24
                pt, lo, hi = (scale * x for x in v[key])
                ax.plot([lo, hi], [y, y], color=color, lw=1.6, solid_capstyle="round", zorder=2)
                ax.plot(pt, y, marker=marker, ms=4.6, color=color, mec=SURF, mew=0.9, zorder=3,
                        label=flabel if ci == 0 else None)
        ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(JUDGE_NAME.get(judge.split("_")[0], judge), color=INK, fontsize=8.5)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xfmt))
    axes[0][0].set_yticks(range(len(COND)), [c[1] for c in COND])
    axes[0][0].invert_yaxis()
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.6, 1.0))
    fig.supxlabel(xlabel, x=0.6, fontsize=8, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(args.out, f"{name}.{ext}"), dpi=220)
    plt.close(fig)
    print("wrote", os.path.join(args.out, name + ".{pdf,png}"))

figure("d_miss", "change in miss rate vs clean (points)", 1, "fig_miss", lambda x, _: "0" if abs(x) < 1e-9 else f"{x:+.0f}")
figure("d_auroc", "change in AUROC vs clean", 1, "fig_auroc", lambda x, _: "0" if abs(x) < 1e-9 else f"{x:+.2f}")
