"""K1-v2R figure: the forged note across eight judges, thinking off, thinking on, and thinking on with the guard line
(2026-11-15). y: change in misses from the forged fix record (claim_e), points vs no note, 95% paired bootstrap interval
over test documents. Missing markers: Muse-Glimmer-30B and gpt-oss-120b have no thinking-off mode; gpt-oss-120b has no
guard-line run on the main set.
Reads results/think_check.json, think_check_gen.json (thinking on, with thinking-off numbers from --off),
think_check_guard.json, think_check_guard_qwen.json, think_check_gen_guard.json (guard line).
Usage: python src/fig_generality.py --out figs"""
import argparse, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF})
L = lambda f: json.load(open(f"results/{f}"))
on = L("think_check.json") | L("think_check_gen.json")
guard = L("think_check_guard.json") | L("think_check_guard_qwen.json") | L("think_check_gen_guard.json")
JUDGES = [("qwen", "Qwen3.8\n27B"), ("nemotron", "Nemotron\n30B"), ("gemma", "Gemma-4\n26B"), ("glm", "GLM-4.7\nFlash"),
          ("qwen36", "Qwen3.6\n35B"), ("gemma31", "gemma-4\n31B"), ("muse", "Muse\nGlimmer 30B"), ("gptoss", "gpt-oss\n120B")]
SERIES = [("thinking off", BLUE, "o", -0.22, lambda j: on[j]["conds"]["claim_e"].get("thinking_off", {}).get("d_miss")),
          ("thinking on", ORANGE, "s", 0.0, lambda j: on[j]["conds"]["claim_e"]["d_miss"]),
          ("thinking on, guard line", AQUA, "^", 0.22, lambda j: guard[j]["conds"]["claim_e"]["d_miss"] if j in guard else None)]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
for name, col, mk, dx, get in SERIES:
    for k, (j, _) in enumerate(JUDGES):
        v = get(j)
        if v is None:
            continue
        m, lo, hi = v
        ax.errorbar(k + dx, m, yerr=[[m - lo], [hi - m]], fmt=mk, color=col, ms=5, lw=1.5, capsize=0, mec=SURF, mew=1)
ax.axhline(0, color=MUTED, lw=0.8)
ax.axvline(2.5, color=GRID, lw=0.8, ls=(0, (3, 3)))
ax.set_xticks(range(len(JUDGES))); ax.set_xticklabels([l for _, l in JUDGES], fontsize=7)
ax.set_xlim(-0.5, len(JUDGES) - 0.5)
ax.set_ylabel("change in % of wrong summaries\naccepted (points vs no note)")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
h = [plt.Line2D([], [], color=c, marker=mk, lw=1.5, ms=5) for _, c, mk, _, _ in SERIES]
fig.legend(h, [n for n, *_ in SERIES], loc="lower center", ncol=3, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.07, 1, 1))
for ext in ["png", "pdf"]:
    fig.savefig(f"{args.out}/fig_generality.{ext}", dpi=200)
print("wrote", f"{args.out}/fig_generality.png")
