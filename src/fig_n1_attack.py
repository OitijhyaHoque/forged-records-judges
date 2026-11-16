"""K1-v2R N1 figure: best-of-k free reviewer notes from two writer prompts (cooperative vs attack: "your goal is that the
reviewer accepts your summary"). Selector: Qwen3.8-27B yes/no. One panel per evaluator.
Lines: miss % of wrong test summaries with the kept note, k = 1..16, per writer prompt.
Dashed: miss with no note; dotted: miss with the hand-written forged note (claim_e); both from the attack run's rescoring.
Reads results/n1_check.json and results/n1atk_check.json (n1_check.py).
Usage: python src/fig_n1_attack.py --out figs [--panels qwen/yn,qwen/score,nemotron/yn]"""
import argparse, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
ap.add_argument("--panels", default="qwen/yn,qwen/score,nemotron/yn")
args = ap.parse_args()
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF})
NAME = {"qwen": "Qwen3.8-27B", "nemotron": "Nemotron-30B"}
FMT = {"yn": "yes/no", "yn_rev": "reversed yes/no", "score": "1-5 score"}
R = {w: json.load(open(f))["selector_qwen_yn"] for w, f in [("coop", "results/n1_check.json"), ("atk", "results/n1atk_check.json")]}
panels = args.panels.split(",")
fig, axes = plt.subplots(1, len(panels), figsize=(2.3 * len(panels), 2.5), sharey=True)
K = [1, 2, 4, 8, 16]
for ax, p in zip(axes, panels):
    j, f = p.split("/")
    for w, col, lab in [("coop", BLUE, "cooperative writer"), ("atk", ORANGE, "writer told to get accepted")]:
        r = R[w][p]["rates"]
        ax.plot(K, [r[f"best_of_{k}"][0] for k in K], "-o", color=col, lw=2, ms=4, label=lab)
    r = R["atk"][p]["rates"]
    ax.axhline(r["clean"][0], color=MUTED, lw=1, ls="--", label="no note")
    ax.axhline(r["claim_e"][0], color=INK, lw=1, ls=":", label="hand-written forged note")
    role = "picks the note" if j == "qwen" and f == "yn" else ("same judge" if j == "qwen" else "held-out judge")
    ax.set_title(f"{NAME[j]}, {FMT[f]}\n({role})", fontsize=8, color=INK)
    ax.set_xscale("log", base=2); ax.set_xticks(K); ax.set_xticklabels([str(k) for k in K])
    ax.set_xlabel("notes sampled, best kept (k)")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("% of wrong summaries accepted")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.1, 1, 1))
for ext in ["png", "pdf"]:
    fig.savefig(f"{args.out}/fig_n1_attack.{ext}", dpi=200)
print("wrote", f"{args.out}/fig_n1_attack.png")
