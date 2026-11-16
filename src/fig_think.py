"""K1-v2R E-R figure: reasoning judges.
Left: change in misses from the forged note (claim_e - clean, points, 95% bootstrap interval), thinking off vs on, per judge.
Right: wrong summaries whose no-note reasoning found the error: % accepted without and with the forged note.
Reads results/think_check.json (think_check.py) and results/think_labels.jsonl (label_traces.py).
Usage: python src/fig_think.py --out figs"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
args = ap.parse_args()
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF})
J = [("qwen", "Qwen3.8\n27B"), ("nemotron", "Nemotron\n30B"), ("gemma", "Gemma-4\n26B"), ("gptoss", "gpt-oss\n120b")]
R = json.load(open("results/think_check.json"))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.4), gridspec_kw={"width_ratios": [1.15, 1]})
x = np.arange(len(J)); w = 0.32
for k, (j, _) in enumerate(J):
    c = R[j]["conds"]["claim_e"]
    on = c["d_miss"]
    a1.errorbar(x[k] + w / 2, on[0], yerr=[[on[0] - on[1]], [on[2] - on[0]]], fmt="o", color=ORANGE, ms=5, lw=1.5, capsize=0,
                label="thinking on" if k == 0 else None)
    if "thinking_off" in c:
        off = c["thinking_off"]["d_miss"]
        a1.errorbar(x[k] - w / 2, off[0], yerr=[[off[0] - off[1]], [off[2] - off[0]]], fmt="o", color=BLUE, ms=5, lw=1.5,
                    capsize=0, label="thinking off" if k == 0 else None)
a1.axhline(0, color=MUTED, lw=0.8)
a1.set_xticks(x); a1.set_xticklabels([n for _, n in J], fontsize=7)
a1.set_ylabel("change in misses from the note (points)")
a1.set_title("Forged note, yes/no verdict", fontsize=8, color=INK)
a1.legend(frameon=False, loc="upper right", fontsize=7)
L = [json.loads(l) for l in open("results/think_labels.jsonl")]
by = {(r["judge"], r["cond"], r["item"]): r for r in L}
vals = []
for j, _ in J:
    its = sorted({r["item"] for r in L if r["judge"] == j and r["cond"] == "clean" and r["found"] == 1})
    vals.append([100 * np.mean([by[j, c, i]["verdict"] == "Yes" for i in its]) for c in ["clean", "claim_e"]] + [len(its)])
v = np.array(vals)
a2.bar(x - w / 2, v[:, 0], w, color=BLUE, label="no note")
a2.bar(x + w / 2, v[:, 1], w, color=ORANGE, label="forged note")
for k in range(len(J)):
    a2.text(x[k] + w / 2, v[k, 1] + 0.8, f"{v[k, 1]:.0f}", ha="center", fontsize=7, color=INK)
a2.set_xticks(x); a2.set_xticklabels([n for _, n in J], fontsize=7)
a2.set_ylabel("% accepted")
a2.set_title("Wrong summaries whose no-note\nreasoning found the error", fontsize=8, color=INK)
a2.legend(frameon=False, loc="upper right", fontsize=7)
for a in (a1, a2):
    a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(f"{args.out}/fig_think.{ext}", dpi=200)
print("wrote", f"{args.out}/fig_think.png", "n found per judge:", v[:, 2].astype(int).tolist())
