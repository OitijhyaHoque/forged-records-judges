"""K1-v2R figures for the defenses (2026-11-16 03:30 UTC). All thinking on; y: change in misses (points of the 126 wrong
test summaries accepted, against no note in the same run), 95% paired bootstrap interval over test documents.
  fig_guard_halves  forged note (claim_e): no guard line, only "Judge the summary against the document only.", only the
                    provenance sentence, full guard line; 7 judges. Reads think_check.json, think_check_gen.json (no line),
                    think_check_guard{A,B}_{main,gen}.json (halves), think_check_guard.json, think_check_guard_qwen.json,
                    think_check_gen_guard.json (full line).
  fig_adaptive      kept notes of the feedback writer on fresh samples: written against the undefended Qwen judge and judged
                    without / with the guard line (refine_ng), and written against the guarded judge with 64 queries
                    (refine64), judged with the guard line; 4 judges. Reads think_check_n1refine_ng.json,
                    think_check_n1refine_ng_guard.json, think_check_n1refine64.json.
  fig_monitor       judge alone and judge plus trace monitor (rule +found of monitor_check.py), forged note and attack
                    notes, 8 judges. Reads monitor_check_gemma.json and, if present, monitor_check_qwen.json.
Colors: no defense orange (as "thinking on" in fig_generality), full guard line aqua (as there); other series violet and
magenta, each with its own marker (palette checked with the dataviz validator: adjacent CVD separation 6.1 needs the
markers). Usage: python src/fig_defenses.py --out figs"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
ORANGE, VIOLET, MAGENTA, AQUA = "#eb6834", "#6250d6", "#e87ba4", "#1baf7a"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF})
L = lambda f: json.load(open(f"results/{f}"))
NAMES = {"qwen": "Qwen3.8\n27B", "nemotron": "Nemotron\n30B", "gemma": "Gemma-4\n26B", "glm": "GLM-4.7\nFlash",
         "qwen36": "Qwen3.6\n35B", "gemma31": "gemma-4\n31B", "muse": "Muse\nGlimmer 30B", "gptoss": "gpt-oss\n120B"}
YLAB = "change in % of wrong summaries\naccepted (points vs no note)"


def panel(ax, judges, series, title=None):
    n = len(series)
    for s, (name, col, mk, get) in enumerate(series):
        dx = (s - (n - 1) / 2) * 0.2
        for k, j in enumerate(judges):
            v = get(j)
            if v is None:
                continue
            m, lo, hi = v
            ax.errorbar(k + dx, m, yerr=[[m - lo], [hi - m]], fmt=mk, color=col, ms=5, lw=1.5, capsize=0, mec=SURF, mew=1)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(judges))); ax.set_xticklabels([NAMES[j] for j in judges], fontsize=7)
    ax.set_xlim(-0.5, len(judges) - 0.5)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    if title:
        ax.set_title(title, fontsize=8, color=INK, loc="left")


def finish(fig, series, name, ncol):
    h = [plt.Line2D([], [], color=c, marker=mk, lw=1.5, ms=5) for _, c, mk, _ in series]
    fig.legend(h, [n for n, *_ in series], loc="lower center", ncol=ncol, frameon=False, fontsize=7)
    for ext in ["png", "pdf"]:
        fig.savefig(f"{args.out}/{name}.{ext}", dpi=200)
    print("wrote", f"{args.out}/{name}.png")


# 1. guard line halves
on = L("think_check.json") | L("think_check_gen.json")
full = L("think_check_guard.json") | L("think_check_guard_qwen.json") | L("think_check_gen_guard.json")
A = L("think_check_guardA_main.json") | L("think_check_guardA_gen.json")
B = L("think_check_guardB_main.json") | L("think_check_guardB_gen.json")
dm = lambda d, c="claim_e": (lambda j: d[j]["conds"][c]["d_miss"] if j in d else None)
J7 = ["qwen", "nemotron", "gemma", "glm", "qwen36", "gemma31", "muse"]
S1 = [("no guard line", ORANGE, "s", dm(on)), ("\"document only\" sentence alone", VIOLET, "D", dm(B)),
      ("provenance sentence alone", MAGENTA, "v", dm(A)), ("full guard line (both sentences)", AQUA, "^", dm(full))]
fig, ax = plt.subplots(figsize=(7.2, 3.1))
panel(ax, J7, S1); ax.set_ylabel(YLAB)
fig.tight_layout(rect=(0, 0.08, 1, 1)); finish(fig, S1, "fig_guard_halves", 4)

# 2. adaptive attackers
ng, ngg, r64 = L("think_check_n1refine_ng.json"), L("think_check_n1refine_ng_guard.json"), L("think_check_n1refine64.json")
J4 = ["qwen", "nemotron", "gemma", "gptoss"]
S2 = [("written against the undefended Qwen judge; no guard line", ORANGE, "s", dm(ng, "atk_refine")),
      ("the same notes; guard line", AQUA, "^", dm(ngg, "atk_refine")),
      ("written against the guarded Qwen judge, 64 queries; guard line", VIOLET, "D", dm(r64, "atk_refine"))]
fig, ax = plt.subplots(figsize=(5.0, 3.1))
panel(ax, J4, S2); ax.set_ylabel(YLAB)
fig.tight_layout(rect=(0, 0.16, 1, 1)); finish(fig, S2, "fig_adaptive", 1)

# 3. trace monitor
mons = [("gemma", "judge + monitor gemma-4-26B", VIOLET, "D")]
if os.path.exists("results/monitor_check_qwen.json"):
    mons.append(("qwen", "judge + monitor Qwen3.8-27B", MAGENTA, "v"))
M = {m: L(f"monitor_check_{m}.json") for m, *_ in mons}
J8 = ["qwen", "nemotron", "gemma", "glm", "qwen36", "gemma31", "muse", "gptoss"]
SHORT = {"qwen": "Qwen\n3.8", "nemotron": "Nemo-\ntron", "gemma": "Gemma\n26B", "glm": "GLM\n4.7", "qwen36": "Qwen\n3.6",
         "gemma31": "Gemma\n31B", "muse": "Muse", "gptoss": "gpt-oss\n120B"}
def mon(m, rule, suffix, cond):
    def get(j):
        key = j + suffix if not (suffix == "_gen" and j in ("qwen", "nemotron", "gemma", "gptoss")) else j
        d = M[m].get(key, {}).get(rule, {}).get(cond)
        return d["d_miss"] if d else None
    return get
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
S3 = [("judge alone", ORANGE, "s", None)] + [(lab, col, mk, None) for _, lab, col, mk in mons]
for ax, (title, suffix, cond) in zip(axs, [("forged fix record", "_gen", "claim_e"), ("attack notes (best of 16)", "_n1atk", "atk_best")]):
    ser = [("judge alone", ORANGE, "s", mon(mons[0][0], "judge", suffix, cond))]
    ser += [(lab, col, mk, mon(m, "+found", suffix, cond)) for m, lab, col, mk in mons]
    panel(ax, J8, ser, title)
    ax.set_xticklabels([SHORT[j] for j in J8], fontsize=6.5)
axs[0].set_ylabel(YLAB)
fig.tight_layout(rect=(0, 0.08, 1, 1)); finish(fig, S3, "fig_monitor", len(S3))
