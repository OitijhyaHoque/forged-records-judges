"""K1-v2R E1b analysis (pre-registered E1/E1b plan, section 8c).
Inputs per judge: results/<judge>_e1b.jsonl (the sent question about every sentence; clean, claim_e, claim_o,
claim_e_unverified), results/<judge>_full.jsonl (T4; the holistic yn rows), and results/<judge>_e1.jsonl (E1; only for
the check of the k = ke rows, skipped if missing).
Scores: sentence accept score a = Yes share. Every-sentence check: A = min over the summary's sentences of a, so the
check rejects the summary if any sentence is rejected. Holistic: H = Yes share of the yn row.
Threshold per score: the 10% quantile of clean scores on calibration consistent items, as in sprint_check.py.
Miss, FA, AUROC and the paired bootstrap over test documents (1,000 resamples, seed 0) as in e1_check.py.
Readouts (fixed before the run): B1 to B6 as in the plan.
Usage: python src/e1b_check.py qwen gemma [--res results] [--json results/e1b_check.json]"""
import argparse, json, os
import numpy as np

FA_Q, NBOOT = 0.10, 1000
CONDS = ["clean", "claim_e", "claim_o", "claim_e_unverified"]
ap = argparse.ArgumentParser()
ap.add_argument("judges", nargs="+"); ap.add_argument("--res", default="results"); ap.add_argument("--json", default=None)
args = ap.parse_args()

yes = lambda p: p[5] / (p[5] + p[6]) if p[5] + p[6] > 0 else np.nan

def auroc(s, y):
    ok = ~np.isnan(s)
    pos, neg = s[ok & (y == 1)][:, None], s[ok & (y == 0)][None]
    return (pos > neg).mean() + 0.5 * (pos == neg).mean()

out = {}
for judge in args.judges:
    hol = [r for r in map(json.loads, open(f"{args.res}/{judge}_full.jsonl")) if r["fmt"] == "yn"]
    sen = [json.loads(l) for l in open(f"{args.res}/{judge}_e1b.jsonl")]
    its = sorted({r["item"] for r in hol}); pos = {i: k for k, i in enumerate(its)}
    first = {}
    for r in hol:
        first.setdefault(r["item"], r)
    y = np.array([first[i]["label"] for i in its]); doc = np.array([first[i]["doc"] for i in its])
    part = np.array([first[i]["part"] for i in its]); ke = np.array([first[i]["ke"] for i in its])
    H = {c: np.full(len(its), np.nan) for c in CONDS}
    for r in hol:
        if r["cond"] in H:
            H[r["cond"]][pos[r["item"]]] = yes(r["p"])
    per = {}  # (cond, item index) -> {k: a}
    for r in sen:
        per.setdefault((r["cond"], pos[r["item"]]), {})[r["k"]] = yes(r["p"])
    A = {c: np.full(len(its), np.nan) for c in CONDS}; amin = {c: np.zeros(len(its), int) for c in CONDS}
    for (c, i), d in per.items():
        k = min(d, key=d.get); A[c][i], amin[c][i] = d[k], k
    low = sum(r["mass"] < 0.5 for r in sen)
    missing = {c: int(np.isnan(A[c]).sum()) for c in CONDS if np.isnan(A[c]).any()}

    cal = (part == "calib") & (y == 1)
    thr = {"all": float(np.nanquantile(A["clean"][cal], FA_Q)), "hol": float(np.nanquantile(H["clean"][cal], FA_Q))}
    S = {"all": A, "hol": H}
    ti = np.where(part == "test")[0]
    tdocs = np.unique(doc[ti]); rng = np.random.default_rng(0)
    boots = [np.concatenate([ti[doc[ti] == d] for d in rng.choice(tdocs, len(tdocs))]) for _ in range(NBOOT)]
    acc = lambda s, c, ix: S[s][c][ix] >= thr[s]
    rej = lambda s, c, ix: S[s][c][ix] < thr[s]
    miss = lambda s, c, ix: 100 * np.mean(acc(s, c, ix)[y[ix] == 0])
    fa = lambda s, c, ix: 100 * np.mean(rej(s, c, ix)[y[ix] == 1])
    def ci(fn):
        bs = np.array([fn(ix) for ix in boots], dtype=float)
        return [float(fn(ti)), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
    fmtci = lambda v: f"{v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}]"

    print(f"\n===== {judge}: {len(its)} items, test {len(ti)} ({(y[ti] == 0).sum()} inconsistent), {len(tdocs)} docs; "
          f"E1b rows {len(sen)}, mass < 0.5: {low}; items without E1b scores: {missing or 'none'}")
    print(f"thresholds: every-sentence {thr['all']:.3f}, holistic yn {thr['hol']:.3f}")
    print(f"   {'condition':20s} {'miss_all':>8s} {'FA_all':>6s} {'AUROC_all':>9s} {'miss_hol':>8s} {'FA_hol':>6s} {'AUROC_hol':>9s}")
    res = {"thr": thr, "conds": {}, "readouts": {}}
    for c in CONDS:
        v = {"miss_all": miss("all", c, ti), "fa_all": fa("all", c, ti), "auroc_all": float(auroc(A[c][ti], y[ti])),
             "miss_hol": miss("hol", c, ti), "fa_hol": fa("hol", c, ti), "auroc_hol": float(auroc(H[c][ti], y[ti]))}
        res["conds"][c] = v
        print(f"   {c:20s} {v['miss_all']:8.1f} {v['fa_all']:6.1f} {v['auroc_all']:9.3f} {v['miss_hol']:8.1f} "
              f"{v['fa_hol']:6.1f} {v['auroc_hol']:9.3f}")

    R = res["readouts"]
    for c in ["clean", "claim_e"]:
        R[f"B1 miss_all - miss_hol ({c})"] = ci(lambda ix: miss("all", c, ix) - miss("hol", c, ix))
        R[f"B1 FA_all - FA_hol ({c})"] = ci(lambda ix: fa("all", c, ix) - fa("hol", c, ix))
    d_all = ci(lambda ix: miss("all", "claim_e", ix) - miss("all", "clean", ix))
    d_hol = ci(lambda ix: miss("hol", "claim_e", ix) - miss("hol", "clean", ix))
    R["B2 d_all (claim_e - clean)"], R["B2 d_hol (claim_e - clean)"] = d_all, d_hol
    R["B2 d_all - d_hol"] = ci(lambda ix: (miss("all", "claim_e", ix) - miss("all", "clean", ix))
                               - (miss("hol", "claim_e", ix) - miss("hol", "clean", ix)))
    half, beats = d_all[0] <= d_hol[0] / 2, R["B1 miss_all - miss_hol (claim_e)"][2] < 0
    label = {2: "resists", 1: "partly", 0: "no"}[half + beats]
    if R["B1 FA_all - FA_hol (claim_e)"][1] > 0:
        label += " at a false-alarm cost"
    res["label"] = label

    def flipped(ix):
        sel = (y[ix] == 0) & acc("hol", "claim_e", ix) & rej("hol", "clean", ix)
        return 100 * rej("all", "claim_e", ix)[sel].mean() if sel.any() else np.nan
    nf = int(((y[ti] == 0) & acc("hol", "claim_e", ti) & rej("hol", "clean", ti)).sum())
    R[f"B3 flipped by the note, every-sentence check rejects (n={nf})"] = ci(flipped)
    R["B4 miss_all claim_o - clean"] = ci(lambda ix: miss("all", "claim_o", ix) - miss("all", "clean", ix))
    R["B4 FA_all claim_o - clean"] = ci(lambda ix: fa("all", "claim_o", ix) - fa("all", "clean", ix))
    R["B5 miss_all unverified - claim_e"] = ci(lambda ix: miss("all", "claim_e_unverified", ix) - miss("all", "claim_e", ix))
    R["B5 FA_all unverified - claim_e"] = ci(lambda ix: fa("all", "claim_e_unverified", ix) - fa("all", "claim_e", ix))
    for c in ["clean", "claim_e"]:
        def loc(ix, c=c):
            sel = (y[ix] == 0) & rej("all", c, ix)
            return 100 * (amin[c][ix] == ke[ix])[sel].mean() if sel.any() else np.nan
        n = int(((y[ti] == 0) & rej("all", c, ti)).sum())
        R[f"B6 rejected wrong summaries, lowest sentence is e ({c}, n={n})"] = ci(loc)
    print(f"   label (B1 + B2): {label}")
    for k, v in R.items():
        print(f"     {k:62s} {fmtci(v):>24s}")

    p_e1 = f"{args.res}/{judge}_e1.jsonl"
    if os.path.exists(p_e1):
        e1 = {(r["item"], r["cond"]): yes(r["p"]) for r in map(json.loads, open(p_e1)) if r["fmt"] == "sent"}
        diff = np.array([abs(per[(r["cond"], pos[r["item"]])][r["k"]] - e1[(r["item"], r["cond"])])
                         for r in sen if r["is_e"]])
        res["check_e1"] = {"n": len(diff), "mean_abs": float(diff.mean()), "max_abs": float(diff.max())}
        print(f"   check, k = ke rows against E1 sent: n {len(diff)}, Yes share |diff| mean {diff.mean():.4f}, "
              f"max {diff.max():.4f}, share > 0.05: {(diff > 0.05).mean():.3f}")
    out[judge] = res
if args.json:
    json.dump(out, open(args.json, "w"), indent=1)
    print(f"\nwrote {args.json}")
