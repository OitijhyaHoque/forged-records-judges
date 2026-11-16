"""T3 of the judge-internals plan: linear probes for "consistent vs inconsistent" on the lens-stage cache.
Pre-registered internals plan, step T3 and readouts R2, R3.

Input: results/<judge>_lens_<fmt>.jsonl and results/<judge>_cache_<fmt>.npy from k1_internals.py --stage lens.
Sites: 0 answer position, 1 mean over sentence e, 2 last summary token.
Training conditions: clean, neutral, pointer_e, claim_o (metadata lines that make no claim about sentence e).
Model: standardize, then L2 logistic regression.
Layer and C: per site, fit on all test documents (training conditions), score the calibration documents (training
conditions pooled), take the (layer, C) with the best calibration AUROC.
Per layer and site (C fixed to the site's choice): 5-fold cross-validation grouped by document on the test documents;
each fold's probe scores the held-out documents under all 5 conditions (out-of-fold scores).
Readouts at the chosen layer: AUROC per condition, and AUROC(claim_o) - AUROC(claim_e) with a 1,000-resample
bootstrap over test documents (seed 0). R2 is site 1 (sentence e), R3 is site 0 (answer position).
Labels for R2: "evidence intact" if the point estimate is at most 0.03 and the interval includes 0; "evidence
degraded" if the lower bound is above 0; else "unclear". Same labels for R3 at the decision site.
Output: results/<judge>_probe_<fmt>.json, plus out-of-fold scores at the chosen layers in
results/<judge>_probe_<fmt>_oof.jsonl.
Usage: python internals/k1_probe.py --judge qwen9 [--fmts yn,score] [--jobs 8]"""
import argparse, json, os, time
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
CONDS = ["clean", "neutral", "pointer_e", "claim_o", "claim_e"]
TRAIN = ["clean", "neutral", "pointer_e", "claim_o"]
SITES = ["answer", "e", "summary_last"]
CS = [0.001, 0.01, 0.1]
NBOOT = 1000


def auroc(s, y):
    pos, neg = s[y == 1][:, None], s[y == 0][None]
    if len(pos) == 0 or neg.size == 0:
        return float("nan")
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def fit_score(Xtr, ytr, Xte, C):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xtr), ytr)
    return m.decision_function(sc.transform(Xte))


def layer_site_job(X, meta, C_site, l, s):
    """Without C_site: calibration AUROC for each C. With C_site: out-of-fold test scores for that C."""
    part, cond, y, doc = meta
    tr_all = (part == "test") & np.isin(cond, TRAIN)
    cal = (part == "calib") & np.isin(cond, TRAIN)
    res = {"l": l, "s": s, "cal": {}}
    if C_site is None:
        for C in CS:
            res["cal"][C] = auroc(fit_score(X[tr_all], y[tr_all], X[cal], C), y[cal])
        return res
    test = part == "test"
    oof = np.full(len(y), np.nan)
    idx = np.where(test)[0]
    for tr_i, te_i in GroupKFold(5).split(idx, groups=doc[idx]):
        tr = idx[tr_i][np.isin(cond[idx[tr_i]], TRAIN)]
        oof[idx[te_i]] = fit_score(X[tr], y[tr], X[idx[te_i]], C_site)
    res["oof"] = oof
    return res


def boot_diff(scores, y, cond, doc, item, a="claim_o", b="claim_e", seed=0):
    """AUROC(a) - AUROC(b) on the given rows, paired bootstrap over documents."""
    docs = np.unique(doc)
    rng = np.random.default_rng(seed)
    by_doc = {d: np.where(doc == d)[0] for d in docs}
    def stat(ix):
        ma, mb = ix[cond[ix] == a], ix[cond[ix] == b]
        return auroc(scores[ma], y[ma]) - auroc(scores[mb], y[mb])
    point = stat(np.arange(len(y)))
    bs = []
    for _ in range(NBOOT):
        pick = rng.choice(docs, len(docs))
        bs.append(stat(np.concatenate([by_doc[d] for d in pick])))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return point, float(lo), float(hi)


def label(point, lo, hi, kind):
    if lo > 0:
        return f"{kind} degraded"
    if point <= 0.03 and lo <= 0 <= hi:
        return f"{kind} intact"
    return "unclear"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True); ap.add_argument("--fmts", default="yn,score")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    for fmt in args.fmts.split(","):
        t0 = time.time()
        rows = [json.loads(l) for l in open(f"{HERE}/results/{args.judge}_lens_{fmt}.jsonl")]
        cache = np.load(f"{HERE}/results/{args.judge}_cache_{fmt}.npy", mmap_mode="r")
        assert len(rows) == cache.shape[0]
        nL = cache.shape[1]
        part = np.array([r["part"] for r in rows]); cond = np.array([r["cond"] for r in rows])
        y = np.array([r["label"] for r in rows]); doc = np.array([r["doc"] for r in rows])
        item = np.array([r["item"] for r in rows])
        meta = (part, cond, y, doc)
        out = {"judge": args.judge, "fmt": fmt, "n_layers": nL, "sites": {}}
        oof_rows = {}
        for s, sname in enumerate(SITES):
            X = np.ascontiguousarray(cache[:, :, s, :]).astype(np.float32)  # [N, nL, d]
            # pass 1: calibration AUROC for every layer and C
            cal = Parallel(n_jobs=args.jobs)(delayed(layer_site_job)(X[:, l], meta, None, l, s) for l in range(nL))
            grid = {(r["l"], C): r["cal"][C] for r in cal for C in CS}
            best_l, best_C = max(grid, key=lambda k: (np.nan_to_num(grid[k], nan=-1), -k[0]))
            # pass 2: out-of-fold test scores for every layer with the chosen C
            res = Parallel(n_jobs=args.jobs)(delayed(layer_site_job)(X[:, l], meta, best_C, l, s) for l in range(nL))
            per_layer = []
            for r in res:
                sc = r["oof"]
                te = part == "test"
                per_layer.append({c: auroc(sc[te & (cond == c)], y[te & (cond == c)]) for c in CONDS}
                                 | {"cal_auroc": grid[(r["l"], best_C)]})
            sc = res[best_l]["oof"]
            te = np.where(part == "test")[0]
            pt, lo, hi = boot_diff(sc[te], y[te], cond[te], doc[te], item[te])
            # mean probe score of inconsistent test items, claim_e minus claim_o (a shift toward "consistent" is > 0)
            inc = te[y[te] == 0]
            d_inc = float(sc[inc][cond[inc] == "claim_e"].mean() - sc[inc][cond[inc] == "claim_o"].mean())
            kind = "evidence" if sname == "e" else "decision" if sname == "answer" else "summary"
            out["sites"][sname] = {"layer": int(best_l), "C": best_C, "cal_auroc": grid[(best_l, best_C)],
                                   "auroc": per_layer[best_l], "diff_o_minus_e": [pt, lo, hi],
                                   "label": label(pt, lo, hi, kind), "inc_score_shift_e_minus_o": d_inc,
                                   "per_layer": per_layer}
            for i in te:
                oof_rows.setdefault(i, {"row": int(i), "item": int(item[i]), "cond": str(cond[i]), "label": int(y[i]),
                                        "doc": str(doc[i])})[sname] = round(float(sc[i]), 4)
            print(f"{args.judge} {fmt} site {sname}: layer {best_l} C {best_C} cal {grid[(best_l, best_C)]:.3f}; "
                  f"AUROC " + " ".join(f"{c}={per_layer[best_l][c]:.3f}" for c in CONDS) +
                  f"; o-e {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {out['sites'][sname]['label']}; "
                  f"{time.time() - t0:.0f}s", flush=True)
            del X
        json.dump(out, open(f"{HERE}/results/{args.judge}_probe_{fmt}.json", "w"), indent=1)
        with open(f"{HERE}/results/{args.judge}_probe_{fmt}_oof.jsonl", "w") as f:
            for i in sorted(oof_rows):
                f.write(json.dumps(oof_rows[i]) + "\n")


if __name__ == "__main__":
    main()
