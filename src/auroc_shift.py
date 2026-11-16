"""K1-v2R, exploratory (written after E1b): threshold-free view of the note's effect.
For each judge and question: AUROC on the test part under clean and claim_e, and the change with a paired bootstrap over
test documents (1,000 resamples, seed 0). Also the mean accept-score shift (claim_e - clean) by label.
Questions: yn and yn_rev (T4 holistic), sent and sent_rev (E1, sentence e), all (E1b, min over sentences; if present).
If the note moves both labels' scores up by similar amounts, AUROC barely changes while misses at a fixed threshold rise:
the note shifts the decision criterion more than it removes the ability to tell the items apart.
Usage: python src/auroc_shift.py qwen gemma nemotron [--res results]"""
import argparse, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("judges", nargs="+"); ap.add_argument("--res", default="results")
args = ap.parse_args()
ACC = {"yn": 5, "sent": 5, "yn_rev": 6, "sent_rev": 6}
acc = lambda p, f: p[ACC[f]] / (p[5] + p[6])
def auroc(s, y):
    pos, neg = s[y == 1][:, None], s[y == 0][None]
    return (pos > neg).mean() + 0.5 * (pos == neg).mean()
for j in args.judges:
    S = {}
    for path in [f"{args.res}/{j}_full.jsonl", f"{args.res}/{j}_e1.jsonl"]:
        for r in map(json.loads, open(path)):
            if r["fmt"] in ACC and r["cond"] in ("clean", "claim_e") and r["part"] == "test":
                S.setdefault((r["fmt"], r["cond"]), {})[r["item"]] = (acc(r["p"], r["fmt"]), r["label"], r["doc"])
    if os.path.exists(f"{args.res}/{j}_e1b.jsonl"):
        m = {}
        for r in map(json.loads, open(f"{args.res}/{j}_e1b.jsonl")):
            if r["cond"] in ("clean", "claim_e") and r["part"] == "test":
                k = (r["cond"], r["item"]); m[k] = min(m.get(k, (9,))[0], acc(r["p"], "sent")), r["label"], r["doc"]
        for (c, i), v in m.items():
            S.setdefault(("all", c), {})[i] = v
    its = sorted(S[("yn", "clean")]); y = np.array([S[("yn", "clean")][i][1] for i in its])
    doc = np.array([S[("yn", "clean")][i][2] for i in its]); docs = np.unique(doc); rng = np.random.default_rng(0)
    boots = [np.concatenate([np.where(doc == d)[0] for d in rng.choice(docs, len(docs))]) for _ in range(1000)]
    print(f"\n== {j} (test items {len(its)})")
    for f in ["yn", "sent", "all", "yn_rev", "sent_rev"]:
        if (f, "clean") not in S:
            continue
        a = {c: np.array([S[(f, c)][i][0] for i in its]) for c in ["clean", "claim_e"]}
        d = lambda ix: auroc(a["claim_e"][ix], y[ix]) - auroc(a["clean"][ix], y[ix])
        bs = np.array([d(ix) for ix in boots])
        sh = {lab: (a["claim_e"] - a["clean"])[y == lab].mean() for lab in [0, 1]}
        print(f"   {f:8s} AUROC clean {auroc(a['clean'], y):.3f} claim_e {auroc(a['claim_e'], y):.3f} "
              f"change {d(np.arange(len(its))):+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]"
              f" | accept-score shift: inconsistent {sh[0]:+.3f}, consistent {sh[1]:+.3f}")
