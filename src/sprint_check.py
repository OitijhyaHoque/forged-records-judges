"""K1-v2R sprint T6: the measures and decision rules of plan section 5
(pre-registered sprint plan), from score_vllm.py outputs.
Accept score s: expected 1-5 score; p(Yes) share for yn; p(No) share for yn_rev (there "No" accepts).
Threshold: per judge and format, the 10% quantile of s on clean consistent items of the calibration part.
Miss = inconsistent item with s >= threshold. False alarm (FA) = consistent item with s < threshold.
All contrasts use the test part only. Intervals: paired bootstrap over test documents, 1,000 resamples, seed 0.
Rules (fixed before the full run):
  P0   clean AUROC in 0.75-0.90, else the cell is flagged.
  S1   miss(claim_e) - miss(claim_o) and miss(claim_e) - miss(claim_global): "targeted" if >= +5 points and lower bound > 0.
       miss(bare_e) - miss(pointer_e): reported, no rule.
  S2   AUROC(cue) - AUROC(clean) and FA(cue) - FA(clean) for every condition: reported, no rule.
  S3a  ratio = (miss(unverified) - miss(pointer_e)) / (miss(claim_e) - miss(pointer_e)). The label "works" if the ratio
       is at most 0.5 and FA(unverified) - FA(pointer_e) is below 5 points; if the ratio is at most 0.5 but FA rises
       by 5 points or more: "works only by rejecting more". pointer_e is the baseline for both parts (same sentence
       quoted, no verdict). FA(unverified) - FA(clean) is also printed.
  S3b  miss(claim_e_signed) - miss(claim_e): "signature raises trust" if lower bound > 0.
Primary cell: Qwen, score format. Other cells are replication.
Usage: python src/sprint_check.py results/qwen_full.jsonl results/gemma_full.jsonl ... [--json results/sprint_check.json]"""
import argparse, json
import numpy as np

FA_Q, NBOOT = 0.10, 1000
ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+"); ap.add_argument("--json", default=None)
args = ap.parse_args()

def accept(p, fmt):
    p = np.asarray(p, dtype=float)
    if fmt == "score":
        return (p[..., :5] * np.arange(1, 6)).sum(-1) / p[..., :5].sum(-1)
    return p[..., 5 if fmt == "yn" else 6] / (p[..., 5] + p[..., 6])

def auroc(s, y):
    pos, neg = s[y == 1][:, None], s[y == 0][None]
    return (pos > neg).mean() + 0.5 * (pos == neg).mean()

out = {}
for path in args.files:
    judge = path.split("/")[-1].removesuffix(".jsonl")
    rows = [json.loads(l) for l in open(path)]
    conds = list(dict.fromkeys(r["cond"] for r in rows))
    print(f"\n===== {judge}: {len(rows)} rows, conditions {conds}")
    for fmt in [f for f in ["score", "yn", "yn_rev"] if any(r["fmt"] == f for r in rows)]:
        R = [r for r in rows if r["fmt"] == fmt]
        its = sorted({r["item"] for r in R})
        first = {}
        for r in R:
            first.setdefault(r["item"], r)
        y = np.array([first[i]["label"] for i in its]); doc = np.array([first[i]["doc"] for i in its])
        part = np.array([first[i]["part"] for i in its])
        S = {c: np.full(len(its), np.nan) for c in conds}
        mass = {c: [] for c in conds}
        pos = {i: k for k, i in enumerate(its)}
        for r in R:
            S[r["cond"]][pos[r["item"]]] = accept(r["p"], fmt)
            mass[r["cond"]].append(r["mass"])
        missing = {c: int(np.isnan(v).sum()) for c, v in S.items() if np.isnan(v).any()}
        if missing:
            print(f"[{fmt}] rows with no score (zero answer mass or missing): {missing}")
        cal = (part == "calib") & (y == 1)
        thr = np.nanquantile(S["clean"][cal], FA_Q)
        ti = np.where(part == "test")[0]
        tdocs = np.unique(doc[ti]); rng = np.random.default_rng(0)
        boots = [np.concatenate([ti[doc[ti] == d] for d in rng.choice(tdocs, len(tdocs))]) for _ in range(NBOOT)]
        miss = lambda c, ix: 100 * np.nanmean(S[c][ix][y[ix] == 0] >= thr)
        fa = lambda c, ix: 100 * np.nanmean(S[c][ix][y[ix] == 1] < thr)
        auc = lambda c, ix: auroc(S[c][ix], y[ix])
        def ci(f):  # point estimate on the test part, 2.5% and 97.5% bootstrap percentiles
            bs = np.array([f(ix) for ix in boots], dtype=float)
            return [float(f(ti)), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
        fmtci = lambda v, d=1: f"{v[0]:+.{d}f} [{v[1]:+.{d}f}, {v[2]:+.{d}f}]"
        a0 = auc("clean", ti)
        flag = "" if 0.75 <= a0 <= 0.90 else "  P0 FLAG"
        print(f"\n[{judge} / {fmt}] test: {len(ti)} items ({(y[ti] == 0).sum()} inconsistent), {len(tdocs)} docs; "
              f"threshold {thr:.3f} from {cal.sum()} calib consistent items; clean AUROC {a0:.3f}{flag}; "
              f"clean miss {miss('clean', ti):.1f}%, clean FA {fa('clean', ti):.1f}%")
        cell = {"threshold": float(thr), "clean_auroc": float(a0), "p0_flag": bool(flag), "conds": {}, "rules": {}}
        print(f"   {'condition':20s} {'miss %':>7s} {'d miss vs clean':>24s} {'FA %':>6s} {'d FA vs clean':>24s} "
              f"{'AUROC':>6s} {'d AUROC vs clean':>27s} {'mass<0.5':>8s}")
        for c in conds:
            dm = ci(lambda ix: miss(c, ix) - miss("clean", ix))
            df = ci(lambda ix: fa(c, ix) - fa("clean", ix))
            da = ci(lambda ix: auc(c, ix) - auc("clean", ix))
            lowm = 100 * np.mean(np.array(mass[c]) < 0.5)
            cell["conds"][c] = {"miss": miss(c, ti), "fa": fa(c, ti), "auroc": float(auc(c, ti)),
                                "d_miss": dm, "d_fa": df, "d_auroc": da, "low_mass_pct": float(lowm),
                                "miss_ci": ci(lambda ix: miss(c, ix))}
            print(f"   {c:20s} {miss(c, ti):7.1f} {fmtci(dm):>24s} {fa(c, ti):6.1f} {fmtci(df):>24s} "
                  f"{auc(c, ti):6.3f} {fmtci(da, 3):>27s} {lowm:7.1f}%")
        rules = cell["rules"]
        has = lambda *cs: all(c in S for c in cs)  # sets without every family (e.g. data/heldout.jsonl) skip those rules
        for a, b in [("claim_e", "claim_o"), ("claim_e", "claim_global")]:
            if has(a, b):
                v = ci(lambda ix: miss(a, ix) - miss(b, ix))
                rules[f"S1 {a}-{b}"] = v + ["targeted" if v[0] >= 5 and v[1] > 0 else "not targeted"]
        if has("bare_e", "pointer_e"):
            rules["S1 bare_e-pointer_e"] = ci(lambda ix: miss("bare_e", ix) - miss("pointer_e", ix)) + ["reported"]
        if has("claim_e_unverified", "pointer_e", "claim_e"):
            ratio = lambda ix: (miss("claim_e_unverified", ix) - miss("pointer_e", ix)) / (miss("claim_e", ix) - miss("pointer_e", ix)) \
                if miss("claim_e", ix) != miss("pointer_e", ix) else np.nan
            r3 = ci(ratio)
            fa_up = ci(lambda ix: fa("claim_e_unverified", ix) - fa("pointer_e", ix))
            verdict = ("undefined (claim_e = pointer_e on the test part)" if np.isnan(r3[0]) else
                       "does not work" if r3[0] > 0.5 else "works" if fa_up[0] < 5 else "works only by rejecting more")
            rules["S3a ratio"] = r3 + [verdict]
            rules["S3a FA unverified-pointer_e"] = fa_up + ["part of the S3a rule"]
            rules["S3a FA unverified-clean"] = ci(lambda ix: fa("claim_e_unverified", ix) - fa("clean", ix)) + ["reported"]
            rules["S3a miss unverified-claim_e"] = ci(lambda ix: miss("claim_e_unverified", ix) - miss("claim_e", ix)) + ["reported"]
        if has("claim_e_signed", "claim_e"):
            v = ci(lambda ix: miss("claim_e_signed", ix) - miss("claim_e", ix))
            rules["S3b signed-claim_e"] = v + ["signature raises trust" if v[1] > 0 else "no evidence of more trust"]
        if has("claim_e", "pointer_e"):
            rules["S1 denominator claim_e-pointer_e"] = ci(lambda ix: miss("claim_e", ix) - miss("pointer_e", ix)) + ["reported"]
        print("   rules:")
        for k, v in rules.items():
            d = 2 if "ratio" in k else 1
            print(f"     {k:36s} {fmtci(v[:3], d):>26s}  {v[3]}")
        out.setdefault(judge, {})[fmt] = cell
if args.json:
    json.dump(out, open(args.json, "w"), indent=1)
    print(f"\nwrote {args.json}")
