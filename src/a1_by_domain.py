# Per-subset readout of the A1 (LLM-AggreFact) thinking-on runs: miss and FA, clean vs claim_e, 95% bootstrap over documents.
import json, sys, numpy as np
suffix = sys.argv[1]; judges = sys.argv[2].split(",")
for j in judges:
    T = {}
    for l in open(f"results/think_{j}{suffix}.jsonl"):
        r = json.loads(l)
        if r["part"] == "test": T[r["cond"], r["item"]] = r
    its = sorted(i for c, i in T if c == "clean")
    print(f"== {j} ({suffix})")
    for dom in sorted({T["clean", i]["domain"] for i in its}):
        ix = [i for i in its if T["clean", i]["domain"] == dom]
        y = np.array([T["clean", i]["label"] for i in ix]); doc = np.array([T["clean", i]["doc"] for i in ix])
        V = {c: np.array([np.nan if T[c, i]["verdict"] is None else float(T[c, i]["verdict"] == "Yes") for i in ix]) for c in ["clean", "claim_e"]}
        docs = np.unique(doc); byd = {d: np.where(doc == d)[0] for d in docs}; rng = np.random.default_rng(0)
        miss = lambda c, k: 100 * np.nanmean(V[c][k][y[k] == 0]); fa = lambda c, k: 100 * np.nanmean(1 - V[c][k][y[k] == 1])
        def ci(f):
            full = np.arange(len(ix)); bs = []
            for _ in range(1000):
                k = np.concatenate([byd[d] for d in rng.choice(docs, len(docs))]); bs.append(f(k))
            return f"{f(full):+5.1f} [{np.nanpercentile(bs, 2.5):+5.1f}, {np.nanpercentile(bs, 97.5):+5.1f}]"
        print(f"  {dom:16s} n={len(ix)} ({(y == 0).sum()} wrong, {len(docs)} docs): miss {miss('clean', np.arange(len(ix))):.1f} -> "
              f"{miss('claim_e', np.arange(len(ix))):.1f}, d miss {ci(lambda k: miss('claim_e', k) - miss('clean', k))}; "
              f"FA {fa('clean', np.arange(len(ix))):.1f} -> {fa('claim_e', np.arange(len(ix))):.1f}")
