"""K1-v2R E1 analysis (pre-registered E1/E1b plan, section 5.3).
Joins the sentence-level rows (results/<judge>_e1.jsonl, formats sent and sent_rev) with the T4 holistic rows
(results/<judge>_full.jsonl, formats yn and yn_rev) by item and condition. Pairs: sent with yn, sent_rev with yn_rev.
Accept score: the Yes share for sent and yn, the No share for sent_rev and yn_rev. Threshold per judge and format: the
10% quantile of clean accept scores on calibration consistent items, as in sprint_check.py. Accept = score >= threshold.
Miss = inconsistent item accepted. FA = consistent item with score < threshold. AUROC and the paired bootstrap over test
documents (1,000 resamples, seed 0) as in sprint_check.py. "Sentence e rejected" = sentence-level score < threshold.
Readouts (fixed before the run):
  R1  d_sent = miss_sent(claim_e) - miss_sent(clean); d_hol = the same for the paired holistic format.
      "no holistic effect" if the lower bound of d_hol is <= 0. Else "verdict-stage" if d_sent <= d_hol / 2 and the
      d_sent interval includes 0; "detection-stage" if d_sent >= d_hol and the d_sent lower bound is > 0; else "mixed".
  R2  among inconsistent test items that the holistic judge accepts under condition c: the share whose sentence e is
      rejected under c. Reported for clean and claim_e.
      R2b: among inconsistent test items the note flipped (holistic reject under clean, accept under claim_e): the share
      whose sentence e is rejected under claim_e.
  R3  miss_sent(claim_e) - miss_sent(claim_o).
  R4  miss_sent and FA_sent of claim_e_unverified and of claim_e_signed, minus claim_e.
  R5  miss_sent(pointer_e) - miss_sent(clean).
Usage: python src/e1_check.py qwen gemma nemotron [--res results] [--json results/e1_check.json]"""
import argparse, json
import numpy as np

FA_Q, NBOOT = 0.10, 1000
ACC = {"yn": 5, "sent": 5, "yn_rev": 6, "sent_rev": 6}  # index of the accepting answer in p (Yes = 5, No = 6)
PAIRS = [("sent", "yn"), ("sent_rev", "yn_rev")]
ap = argparse.ArgumentParser()
ap.add_argument("judges", nargs="+"); ap.add_argument("--res", default="results"); ap.add_argument("--json", default=None)
args = ap.parse_args()

def accept(p, fmt):
    z = p[5] + p[6]
    return p[ACC[fmt]] / z if z > 0 else np.nan

def auroc(s, y):
    ok = ~np.isnan(s)
    pos, neg = s[ok & (y == 1)][:, None], s[ok & (y == 0)][None]
    return (pos > neg).mean() + 0.5 * (pos == neg).mean()

out = {}
for judge in args.judges:
    rows = [json.loads(l) for path in [f"{args.res}/{judge}_e1.jsonl", f"{args.res}/{judge}_full.jsonl"] for l in open(path)]
    rows = [r for r in rows if r["fmt"] in ACC]
    its = sorted({r["item"] for r in rows}); pos = {i: k for k, i in enumerate(its)}
    first = {}
    for r in rows:
        first.setdefault(r["item"], r)
    y = np.array([first[i]["label"] for i in its]); doc = np.array([first[i]["doc"] for i in its])
    part = np.array([first[i]["part"] for i in its])
    conds = list(dict.fromkeys(r["cond"] for r in rows))
    S = {(f, c): np.full(len(its), np.nan) for f in ACC for c in conds}
    low = {f: 0 for f in ACC}
    for r in rows:
        S[(r["fmt"], r["cond"])][pos[r["item"]]] = accept(r["p"], r["fmt"])
        low[r["fmt"]] += r["mass"] < 0.5
    nan = {f"{f}|{c}": int(np.isnan(v).sum()) for (f, c), v in S.items() if np.isnan(v).any()}
    cal = (part == "calib") & (y == 1)
    thr = {f: float(np.nanquantile(S[(f, "clean")][cal], FA_Q)) for f in ACC}
    ti = np.where(part == "test")[0]
    tdocs = np.unique(doc[ti]); rng = np.random.default_rng(0)
    boots = [np.concatenate([ti[doc[ti] == d] for d in rng.choice(tdocs, len(tdocs))]) for _ in range(NBOOT)]

    acc = lambda f, c, ix: S[(f, c)][ix] >= thr[f]
    rej = lambda f, c, ix: S[(f, c)][ix] < thr[f]
    miss = lambda f, c, ix: 100 * np.mean(acc(f, c, ix)[y[ix] == 0])
    fa = lambda f, c, ix: 100 * np.mean(rej(f, c, ix)[y[ix] == 1])
    auc = lambda f, c, ix: auroc(S[(f, c)][ix], y[ix])
    def ci(fn):  # point estimate on the test part, 2.5% and 97.5% bootstrap percentiles
        bs = np.array([fn(ix) for ix in boots], dtype=float)
        return [float(fn(ti)), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
    fmtci = lambda v, d=1: f"{v[0]:+.{d}f} [{v[1]:+.{d}f}, {v[2]:+.{d}f}]"

    print(f"\n===== {judge}: {len(its)} items, test {len(ti)} items ({(y[ti] == 0).sum()} inconsistent), {len(tdocs)} docs; "
          f"rows with mass < 0.5: {low}; missing scores: {nan or 'none'}")
    out[judge] = {}
    for fs, fh in PAIRS:
        print(f"\n[{judge} / {fs} vs {fh}] thresholds {fs} {thr[fs]:.3f}, {fh} {thr[fh]:.3f}; "
              f"clean AUROC {fs} {auc(fs, 'clean', ti):.3f}, {fh} {auc(fh, 'clean', ti):.3f}")
        print(f"   {'condition':20s} {'miss_sent':>9s} {'d vs clean':>22s} {'FA_sent':>7s} {'d vs clean':>22s} "
              f"{'AUROC_sent':>10s} {'miss_hol':>8s} {'d vs clean':>22s}")
        cell = {"thr": {fs: thr[fs], fh: thr[fh]}, "conds": {}, "readouts": {}}
        for c in conds:
            dms = ci(lambda ix: miss(fs, c, ix) - miss(fs, "clean", ix))
            dfs = ci(lambda ix: fa(fs, c, ix) - fa(fs, "clean", ix))
            dmh = ci(lambda ix: miss(fh, c, ix) - miss(fh, "clean", ix))
            cell["conds"][c] = {"miss_sent": miss(fs, c, ti), "d_miss_sent": dms, "fa_sent": fa(fs, c, ti), "d_fa_sent": dfs,
                                "auroc_sent": float(auc(fs, c, ti)), "miss_hol": miss(fh, c, ti), "d_miss_hol": dmh,
                                "fa_hol": fa(fh, c, ti)}
            print(f"   {c:20s} {miss(fs, c, ti):9.1f} {fmtci(dms):>22s} {fa(fs, c, ti):7.1f} {fmtci(dfs):>22s} "
                  f"{auc(fs, c, ti):10.3f} {miss(fh, c, ti):8.1f} {fmtci(dmh):>22s}")
        R = cell["readouts"]
        d_sent, d_hol = cell["conds"]["claim_e"]["d_miss_sent"], cell["conds"]["claim_e"]["d_miss_hol"]
        verdict = ("no holistic effect" if d_hol[1] <= 0 else
                   "verdict-stage" if d_sent[0] <= d_hol[0] / 2 and d_sent[1] <= 0 <= d_sent[2] else
                   "detection-stage" if d_sent[0] >= d_hol[0] and d_sent[1] > 0 else "mixed")
        R["R1 d_sent (claim_e - clean)"] = d_sent + [verdict]
        R["R1 d_hol (claim_e - clean)"] = d_hol + ["reported"]
        R["R1 d_sent - d_hol"] = ci(lambda ix: (miss(fs, "claim_e", ix) - miss(fs, "clean", ix))
                                    - (miss(fh, "claim_e", ix) - miss(fh, "clean", ix))) + ["reported"]

        def share(ix, c=None, flipped=False):  # % of selected inconsistent items whose sentence e is rejected
            inc = y[ix] == 0
            sel = inc & acc(fh, "claim_e", ix) & rej(fh, "clean", ix) if flipped else inc & acc(fh, c, ix)
            return 100 * rej(fs, "claim_e" if flipped else c, ix)[sel].mean() if sel.any() else np.nan
        for c in ["clean", "claim_e"]:
            n = int((acc(fh, c, ti) & (y[ti] == 0)).sum())
            R[f"R2 {c}: holistic accepts, e rejected (n={n})"] = ci(lambda ix: share(ix, c)) + ["reported"]
        nf = int((acc(fh, "claim_e", ti) & rej(fh, "clean", ti) & (y[ti] == 0)).sum())
        R[f"R2b flipped by the note, e rejected (n={nf})"] = ci(lambda ix: share(ix, flipped=True)) + ["reported"]
        R["R3 miss_sent claim_e - claim_o"] = ci(lambda ix: miss(fs, "claim_e", ix) - miss(fs, "claim_o", ix)) + ["reported"]
        for c in ["claim_e_unverified", "claim_e_signed"]:
            R[f"R4 miss_sent {c} - claim_e"] = ci(lambda ix: miss(fs, c, ix) - miss(fs, "claim_e", ix)) + ["reported"]
            R[f"R4 FA_sent {c} - claim_e"] = ci(lambda ix: fa(fs, c, ix) - fa(fs, "claim_e", ix)) + ["reported"]
        R["R5 miss_sent pointer_e - clean"] = ci(lambda ix: miss(fs, "pointer_e", ix) - miss(fs, "clean", ix)) + ["reported"]
        print("   readouts:")
        for k, v in R.items():
            print(f"     {k:52s} {fmtci(v[:3]):>24s}  {v[3]}")
        out[judge][f"{fs}|{fh}"] = cell
if args.json:
    json.dump(out, open(args.json, "w"), indent=1)
    print(f"\nwrote {args.json}")
