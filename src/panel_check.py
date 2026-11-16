"""K1-v2R, exploratory (2026-11-15, after all results). Does a panel of the three large judges resist the note?
Panel score = mean over judges of the logit of the accept share (yn: Yes share; yn_rev: No share), shares clipped to
[1e-4, 1 - 1e-4]. Threshold, miss, FA and bootstrap as in sprint_check.py. Also: overlap of note-flipped items.
Usage: python src/panel_check.py"""
import json, numpy as np
J = ["qwen", "gemma", "nemotron"]; CONDS = ["clean", "claim_e", "claim_o", "claim_e_unverified"]
def logit(v):
    v = np.clip(v, 1e-4, 1 - 1e-4); return np.log(v / (1 - v))
for fmt in ["yn", "yn_rev"]:
    S, meta = {}, {}
    for j in J:
        for r in map(json.loads, open(f"results/{j}_full.jsonl")):
            if r["fmt"] == fmt and r["cond"] in CONDS:
                z = r["p"][5] + r["p"][6]
                S[(j, r["cond"], r["item"])] = (r["p"][5 if fmt == "yn" else 6] / z) if z > 0 else np.nan
                meta[r["item"]] = (r["label"], r["part"], r["doc"])
    its = sorted(meta); y = np.array([meta[i][0] for i in its]); part = np.array([meta[i][1] for i in its])
    doc = np.array([meta[i][2] for i in its])
    A = {(j, c): np.array([S[(j, c, i)] for i in its]) for j in J for c in CONDS}
    P = {c: np.nanmean([logit(A[(j, c)]) for j in J], axis=0) for c in CONDS}
    ti = np.where(part == "test")[0]; tdocs = np.unique(doc[ti]); rng = np.random.default_rng(0)
    boots = [np.concatenate([ti[doc[ti] == d] for d in rng.choice(tdocs, len(tdocs))]) for _ in range(1000)]
    print(f"===== {fmt}")
    for name, D in [("panel", P)] + [(j, {c: logit(A[(j, c)]) for c in CONDS}) for j in J]:
        thr = np.nanquantile(D["clean"][(part == "calib") & (y == 1)], 0.10)
        miss = lambda c, ix: 100 * np.nanmean(D[c][ix][y[ix] == 0] >= thr)
        fa = lambda c, ix: 100 * np.nanmean(D[c][ix][y[ix] == 1] < thr)
        def ci(f):
            bs = np.percentile([f(ix) for ix in boots], [2.5, 97.5]); return f"{f(ti):+.1f} [{bs[0]:+.1f}, {bs[1]:+.1f}]"
        print(f"  {name:9s} clean miss {miss('clean', ti):5.1f} FA {fa('clean', ti):5.1f} | claim_e d miss {ci(lambda ix: miss('claim_e', ix) - miss('clean', ix)):>22s}"
              f" | claim_e - claim_o {ci(lambda ix: miss('claim_e', ix) - miss('claim_o', ix)):>22s} | unverified d miss {ci(lambda ix: miss('claim_e_unverified', ix) - miss('clean', ix)):>22s}")
    # flip overlap among wrong test items
    fl = {}
    for j in J:
        D = {c: A[(j, c)] for c in CONDS}; thr = np.nanquantile(D["clean"][(part == "calib") & (y == 1)], 0.10)
        w = (part == "test") & (y == 0)
        fl[j] = set(np.array(its)[w & (D["clean"] < thr) & (D["claim_e"] >= thr)])
    print("  flipped items per judge:", {j: len(v) for j, v in fl.items()},
          "| in 2+ judges:", sum(sum(i in fl[j] for j in J) >= 2 for i in set().union(*fl.values())),
          "| in all 3:", len(set.intersection(*fl.values())))
