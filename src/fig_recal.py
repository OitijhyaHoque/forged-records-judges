"""K1-v2R, exploratory figures (2026-11-15, after all results were seen). Test part only.
fig_roc_shift: ROC of each judge's yes/no verdict, clean against claim_e (note on every item). Marked points: the clean
  threshold (10% quantile of clean calibration consistent scores) applied to each condition.
fig_pitfall: change in miss rate (claim_e - clean, same threshold for both) as a function of the operating point (clean
  test false-alarm rate), for the verdict (yn) and the question about sentence e (sent). Points mark the thresholds E1
  used (each set by the 10% rule on calibration items).
Usage: python src/fig_recal.py --out figs"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--res", default="results")
args = ap.parse_args()
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF})
JUDGES = [("qwen", f"{args.res}/qwen_full.jsonl", "Qwen3.8-27B"), ("gemma", f"{args.res}/gemma_full.jsonl", "Gemma-4-26B"),
          ("nemotron", f"{args.res}/nemotron_full.jsonl", "Nemotron-30B"),
          ("qwen9", "results/qwen9_e0_full.jsonl", "Qwen3.5-9B"),
          ("gemma12", "results/gemma12_e0_full.jsonl", "Gemma-4-12B")]


def load(paths, fmt):
    S, meta = {}, {}
    for p in paths:
        for r in map(json.loads, open(p)):
            if r["fmt"] != fmt or r["cond"] not in ("clean", "claim_e"):
                continue
            z = r["p"][5] + r["p"][6]
            S[(r["cond"], r["item"])] = r["p"][5] / z if z > 0 else np.nan
            meta[r["item"]] = (r["label"], r["part"])
    its = sorted(meta)
    y = np.array([meta[i][0] for i in its]); part = np.array([meta[i][1] for i in its])
    return {c: np.array([S[(c, i)] for i in its]) for c in ("clean", "claim_e")}, y, part


def load_min(path_full, path_e1b):  # every-sentence check: lowest sentence Yes share (E1b)
    _, y, part = load([path_full], "yn")
    M = {}
    for r in map(json.loads, open(path_e1b)):
        if r["fmt"] == "sent" and r["cond"] in ("clean", "claim_e"):
            z = r["p"][5] + r["p"][6]
            M.setdefault((r["cond"], r["item"]), []).append(r["p"][5] / z if z > 0 else np.nan)
    its = sorted({i for _, i in M})
    return {c: np.array([np.nanmin(M[(c, i)]) for i in its]) for c in ("clean", "claim_e")}, y, part


def roc(s, y):  # positive = wrong summary caught (score below threshold)
    ts = np.unique(np.concatenate([s[~np.isnan(s)], [np.inf]]))
    fa = np.array([np.nanmean(s[y == 1] < t) for t in ts]); hit = np.array([np.nanmean(s[y == 0] < t) for t in ts])
    return np.r_[0, fa], np.r_[0, hit]


fig, axes = plt.subplots(1, 5, figsize=(10, 2.35), sharey=True)
for ax, (j, path, name) in zip(axes, JUDGES):
    S, y, part = load([path], "yn")
    thr = np.nanquantile(S["clean"][(part == "calib") & (y == 1)], 0.10)
    t = part == "test"
    ax.plot([0, 1], [0, 1], color=GRID, lw=0.8, zorder=0)
    for c, col, mk, lab in [("clean", BLUE, "o", "clean"), ("claim_e", ORANGE, "s", "forged note on every item")]:
        fa, hit = roc(S[c][t], y[t])
        ax.plot(fa, hit, color=col, lw=1.6, label=lab, zorder=2)
        pfa, phit = np.nanmean(S[c][t][y[t] == 1] < thr), np.nanmean(S[c][t][y[t] == 0] < thr)
        ax.plot(pfa, phit, marker=mk, ms=6, color=col, mec=SURF, mew=1.2, zorder=4)
    p0 = (np.nanmean(S["clean"][t][y[t] == 1] < thr), np.nanmean(S["clean"][t][y[t] == 0] < thr))
    p1 = (np.nanmean(S["claim_e"][t][y[t] == 1] < thr), np.nanmean(S["claim_e"][t][y[t] == 0] < thr))
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="->", color=INK, lw=0.9), zorder=5)
    ax.set_title(name, fontsize=8.5, color=INK); ax.set_xlim(0, 0.6); ax.set_ylim(0, 1.0)
    ax.grid(color=GRID, lw=0.5, zorder=0); ax.set_xticks([0, 0.2, 0.4, 0.6])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("wrong summaries caught")
fig.supxlabel("correct summaries rejected (false alarms)", fontsize=8, color=INK, y=0.02)
axes[0].legend(loc="lower right", fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0.04, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(f"{args.out}/fig_roc_shift.{ext}", dpi=200)
plt.close(fig)

# pitfall: delta miss vs operating point, verdict against sentence-level readouts, same context
AQUA = "#1baf7a"
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
for ax, (j, name) in zip(axes, [("qwen", "Qwen3.8-27B"), ("gemma", "Gemma-4-26B")]):
    series = [("verdict (whole summary)", BLUE, "o", load([f"{args.res}/{j}_full.jsonl"], "yn"), (6, 4)),
              ("question about sentence e only", ORANGE, "s", load([f"{args.res}/{j}_e1.jsonl"], "sent"), (6, -10)),
              ("check of every sentence", AQUA, "^", load_min(f"{args.res}/{j}_full.jsonl", f"{args.res}/{j}_e1b.jsonl"), (-60, 8))]
    for lab, col, mk, (S, y, part), off in series:
        t = part == "test"; ok = S["clean"][t][y[t] == 1]
        xs, ds = [], []
        for a in np.linspace(0.02, 0.40, 39):
            th = np.nanquantile(ok, a)
            xs.append(100 * np.nanmean(ok < th))
            ds.append(100 * (np.nanmean(S["claim_e"][t][y[t] == 0] >= th) - np.nanmean(S["clean"][t][y[t] == 0] >= th)))
        ax.plot(xs, ds, color=col, lw=1.5, label=lab, zorder=2)
        th = np.nanquantile(S["clean"][(part == "calib") & (y == 1)], 0.10)
        px = 100 * np.nanmean(ok < th)
        py = 100 * (np.nanmean(S["claim_e"][t][y[t] == 0] >= th) - np.nanmean(S["clean"][t][y[t] == 0] >= th))
        ax.plot(px, py, marker=mk, ms=6.5, color=col, mec=SURF, mew=1.2, zorder=4)
        print(f"  {name} {lab}: calibrated point {py:+.1f} points at {px:.1f}% clean test FA")
    ax.set_title(name, fontsize=8.5, color=INK); ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.grid(color=GRID, lw=0.5, zorder=0); ax.set_xlim(0, 41)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
axes[0].set_ylabel("miss rate, note minus clean (points)")
fig.supxlabel("operating point: false alarms on clean correct summaries (%)", fontsize=8, color=INK, y=0.02)
axes[1].legend(loc="upper right", fontsize=6.6, frameon=False)
fig.tight_layout(rect=(0, 0.04, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(f"{args.out}/fig_pitfall.{ext}", dpi=200)
print("wrote", args.out)
