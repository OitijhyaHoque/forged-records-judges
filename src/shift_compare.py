"""K1-v2R, exploratory (written after E1b): does the note move the sentence-level score as much as the verdict score?
Threshold-free. For inconsistent test items: shift = accept score under claim_e minus under clean, for the holistic
question (yn, yn_rev; T4) and the sentence-e question in the same context (sent, sent_rev; E1). Reported: mean shift of
each, and the paired difference (sentence minus verdict) with a bootstrap over test documents (1,000 resamples, seed 0).
Same for consistent items.
Usage: python src/shift_compare.py qwen gemma nemotron [--res results]"""
import argparse, json
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("judges", nargs="+"); ap.add_argument("--res", default="results")
args = ap.parse_args()
ACC = {"yn": 5, "sent": 5, "yn_rev": 6, "sent_rev": 6}
for j in args.judges:
    S = {}
    for path in [f"{args.res}/{j}_full.jsonl", f"{args.res}/{j}_e1.jsonl"]:
        for r in map(json.loads, open(path)):
            if r["fmt"] in ACC and r["cond"] in ("clean", "claim_e") and r["part"] == "test":
                S[(r["fmt"], r["cond"], r["item"])] = (r["p"][ACC[r["fmt"]]] / (r["p"][5] + r["p"][6]), r["label"], r["doc"])
    its = sorted({k[2] for k in S}); y = np.array([S[("yn", "clean", i)][1] for i in its])
    doc = np.array([S[("yn", "clean", i)][2] for i in its]); docs = np.unique(doc); rng = np.random.default_rng(0)
    boots = [np.concatenate([np.where(doc == d)[0] for d in rng.choice(docs, len(docs))]) for _ in range(1000)]
    sh = {f: np.array([S[(f, "claim_e", i)][0] - S[(f, "clean", i)][0] for i in its]) for f in ACC}
    print(f"\n== {j}")
    for fs, fh in [("sent", "yn"), ("sent_rev", "yn_rev")]:
        for lab, name in [(0, "inconsistent"), (1, "consistent")]:
            m = lambda f, ix: sh[f][ix][y[ix] == lab].mean()
            d = lambda ix: m(fs, ix) - m(fh, ix)
            bs = np.percentile([d(ix) for ix in boots], [2.5, 97.5])
            all_ix = np.arange(len(its))
            print(f"   {fs:8s} vs {fh:6s} {name:12s}: shift sentence {m(fs, all_ix):+.3f}, verdict {m(fh, all_ix):+.3f}, "
                  f"difference {d(all_ix):+.3f} [{bs[0]:+.3f}, {bs[1]:+.3f}]")
