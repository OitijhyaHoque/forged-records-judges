"""Report figure for the judge-internals results (Qwen3.5-9B only, the G1 judge).
Left: restoration of the claim_e - claim_o gap when sentence e's tokens are patched from claim_o at one layer only
(stage patch2), with Qwen's full-attention layers marked. Right: restoration from interventions at all layers
(stage patch, patch2, knock, patch3), median over the R4 items, with 95% bootstrap intervals over documents.
Input: the --json output of k1_readouts.py. Colors: the first two slots of the dataviz reference palette
(validated light mode: CVD delta E 24.7, normal-vision 33.6).
Usage: python internals/k1_report_fig.py internals/results/readouts_qwen9.json internals/results/figs/fig_internals_qwen9"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"yn": "#2a78d6", "score": "#eb6834"}
NAME = {"yn": "yes/no (L)", "score": "score (E[s])"}
FULL_ATTN = list(range(3, 32, 4))

R = json.load(open(sys.argv[1]))["qwen9"]
out = sys.argv[2]

rows = [  # (group, label, getter)
    ("Patch claim_o -> claim_e", "summary", lambda X: X["patch"]["into_e_summ"]),
    ("Patch claim_o -> claim_e", "sentence e", lambda X: X["patch"]["into_e_e"]),
    ("Patch claim_o -> claim_e", "summary without sentence e", lambda X: X["patch2"]["into_e_not_e"]),
    ("Patch claim_o -> claim_e", "question tokens", lambda X: X["patch"]["into_e_q"]),
    ("Block attention (claim_e run)", "sentence e -> note", lambda X: X["knock"]["e_note_all"]),
    ("Block attention (claim_e run)", "sentence e -> its quote in the note", lambda X: X["knock"]["e_quote_all"]),
    ("Block attention (claim_e run)", "question tokens -> note", lambda X: X["knock"]["q_note_all"]),
    ("Patch pointer_e -> claim_e", "sentence e", lambda X: X["patch3"]["into_e_e"]),
    ("Patch pointer_e -> claim_e", "document + summary", lambda X: X["patch3"]["into_e_docsumm"]),
]

plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "font.family": "DejaVu Sans"})
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1, 1.15]}, facecolor=SURF)
for ax in (a, b):
    ax.set_facecolor(SURF)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

# left panel
for l in FULL_ATTN:
    a.axvline(l, color=GRID, lw=6, zorder=0)
a.axhline(0, color=INK2, lw=0.8); a.axhline(1, color=INK2, lw=0.8, ls=(0, (3, 3)))
for fmt in ["yn", "score"]:
    E = np.array(R[fmt]["patch2"]["e_layer"])
    x = np.arange(len(E))
    a.fill_between(x, E[:, 1], E[:, 2], color=COL[fmt], alpha=0.15, lw=0)
    a.plot(x, E[:, 0], color=COL[fmt], lw=2, marker="o", ms=3.5, label=NAME[fmt])
a.set_xlabel("layer (shaded: full-attention layers 3, 7, ..., 31)")
a.set_ylabel("share of the claim_e - claim_o gap restored")
a.set_title("Patch sentence e at one layer only", loc="left", fontsize=9.5)
a.set_xlim(-0.5, 31.5); a.set_ylim(-0.25, 1.15)
a.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.62))

# right panel
y, ticks, labels, groups = 0, [], [], []
last = None
for g, lab, get in rows:
    if g != last:
        if last is not None:
            y += 0.6
        groups.append((y, g)); last = g
    for k, fmt in enumerate(["yn", "score"]):
        v = get(R[fmt])
        yy = y + (k - 0.5) * 0.32
        b.plot([v[1], v[2]], [yy, yy], color=COL[fmt], lw=2, solid_capstyle="round")
        b.plot(v[0], yy, "o", color=COL[fmt], ms=6, mec=SURF, mew=1.5, label=NAME[fmt] if y == 0 else None)
    ticks.append(y); labels.append(lab)
    y += 1
b.set_yticks(ticks, labels); b.invert_yaxis()
for gy, g in groups:
    b.text(-0.52, gy - 0.62, g, fontsize=8.5, fontweight="bold", color=INK, transform=b.get_yaxis_transform())
b.axvline(0, color=INK2, lw=0.8); b.axvline(1, color=INK2, lw=0.8, ls=(0, (3, 3)))
b.set_xlim(-0.2, 1.4)
b.set_xlabel("share of the gap restored")
b.set_title("Interventions at all layers", loc="left", fontsize=9.5)
b.tick_params(axis="y", length=0)
b.legend(frameon=False, loc="lower right")
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(f"{out}.{ext}", dpi=200, facecolor=SURF)
print("wrote", out)
