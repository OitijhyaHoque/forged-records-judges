"""Figure 2 of the report: the forged record on LLM-AggreFact (real system outputs, human labels), reasoning judges (2026-11-16).
y: change in misses from the forged fix record (claim_e), points vs no note in the same run, 95% paired bootstrap interval
over test documents (542 test claims, 277 unsupported). Series: no defense; guard line in the prompt; guard line plus a
trace monitor, gemma-4-26B or Qwen3.8-27B (rule +found of monitor_check.py, which counts a missing verdict as reject).
The "unverified" label is in the report's Table A2 only. Colors match fig_monitor (gemma monitor violet, Qwen monitor
magenta).
Reads results/think_check_aggrefact.json, think_check_aggrefact_guard.json (think_check.py) and
monitor_check_{gemma,qwen}_aggrefact.json (monitor_check.py), all written by scripts/08_readouts.sh.
Style and colors as fig_defenses.py (no defense orange, full guard line aqua; palette checked there with the dataviz
validator, adjacent CVD separation needs the markers). Usage: python src/fig_aggrefact.py --out figs"""
import argparse, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
ORANGE, VIOLET, MAGENTA, AQUA = "#eb6834", "#6250d6", "#e87ba4", "#1baf7a"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF})
L = lambda f: json.load(open(f"results/{f}"))
NAMES = {"qwen": "Qwen3.8\n27B", "gemma": "Gemma-4\n26B", "qwen36": "Qwen3.6\n35B", "nemotron": "Nemotron\n30B",
         "glm": "GLM-4.7\nFlash", "gptoss": "gpt-oss\n120B"}
J6 = ["qwen", "gemma", "qwen36", "nemotron", "glm", "gptoss"]

on = L("think_check_aggrefact.json")
guard = L("think_check_aggrefact_guard.json")
MON = {m: L(f"monitor_check_{m}_aggrefact.json") for m in ["gemma", "qwen"]}
dm = lambda d, c: (lambda j: d[j]["conds"][c]["d_miss"] if j in d else None)
gm = lambda m: (lambda j: MON[m].get(f"{j}_aggrefact_guard", {}).get("+found", {}).get("claim_e", {}).get("d_miss"))
S = [("no defense", ORANGE, "s", dm(on, "claim_e")),
     ("guard line in the prompt", AQUA, "^", dm(guard, "claim_e")),
     ("guard line + monitor gemma-4-26B", VIOLET, "D", gm("gemma")),
     ("guard line + monitor Qwen3.8-27B", MAGENTA, "v", gm("qwen"))]

fig, ax = plt.subplots(figsize=(7.2, 3.1))
n = len(S)
for s, (name, col, mk, get) in enumerate(S):
    dx = (s - (n - 1) / 2) * 0.18
    for k, j in enumerate(J6):
        v = get(j)
        if v is None:
            continue
        m, lo, hi = v
        ax.errorbar(k + dx, m, yerr=[[m - lo], [hi - m]], fmt=mk, color=col, ms=5, lw=1.5, capsize=0, mec=SURF, mew=1)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(range(len(J6))); ax.set_xticklabels([NAMES[j] for j in J6], fontsize=7)
ax.set_xlim(-0.5, len(J6) - 0.5)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.set_ylabel("change in % of unsupported claims\naccepted (points vs no note)")
fig.tight_layout(rect=(0, 0.08, 1, 1))
h = [plt.Line2D([], [], color=c, marker=mk, lw=1.5, ms=5) for _, c, mk, _ in S]
fig.legend(h, [nm for nm, *_ in S], loc="lower center", ncol=4, frameon=False, fontsize=7)
for ext in ["png", "pdf"]:
    fig.savefig(f"{args.out}/fig_aggrefact.{ext}", dpi=200)
print("wrote", f"{args.out}/fig_aggrefact.png")
for name, _, _, get in S:
    print(f"{name:45s}", "  ".join(f"{j} {get(j)[0]:+5.1f}" if get(j) else f"{j} -" for j in J6))
