"""K1-v2R, exploratory (2026-11-15, after all results). Selection by the judge (best-of-N), from existing scores.
Within each test document, every (correct, wrong) summary pair is compared by accept score.
Win rate of the wrong summary = share of pairs where the wrong one scores higher (ties count 0.5).
  clean:  both without a note.
  attack: the wrong summary carries condition C (claim_e, claim_o, claim_global, neutral, claim_e_unverified); the correct one is clean.
Top pick: per document with at least one correct and one wrong summary, the chance that the top-scored summary is wrong
(ties broken at random, expected value). Bootstrap over test documents (1,000 resamples, seed 0).
Usage: python src/select_check.py"""
import json, numpy as np
FILES = [("qwen", "results/qwen_full.jsonl"), ("gemma", "results/gemma_full.jsonl"), ("nemotron", "results/nemotron_full.jsonl"),
         ("qwen9", "results/qwen9_e0_full.jsonl"), ("gemma12", "results/gemma12_e0_full.jsonl")]
ATT = ["claim_e", "claim_o", "claim_global", "neutral", "claim_e_unverified"]
def acc(p, fmt):
    if fmt == "score":
        p = np.asarray(p[:5]); return float((p * np.arange(1, 6)).sum() / p.sum())
    z = p[5] + p[6]; return p[5 if fmt == "yn" else 6] / z if z > 0 else np.nan
for j, path in FILES:
    rows = [json.loads(l) for l in open(path)]
    for fmt in ["score", "yn", "yn_rev"]:
        S, meta = {}, {}
        for r in rows:
            if r["fmt"] == fmt and r["part"] == "test":
                S[(r["cond"], r["item"])] = acc(r["p"], fmt); meta[r["item"]] = (r["label"], r["doc"])
        if not meta:
            continue
        docs = {}
        for i, (lab, d) in meta.items():
            docs.setdefault(d, {0: [], 1: []})[lab].append(i)
        D = [d for d, v in docs.items() if v[0] and v[1]]
        def win(cond, ds):
            w = []
            for d in ds:
                for a in docs[d][0]:
                    for b in docs[d][1]:
                        sa, sb = S[(cond, a)], S[("clean", b)]
                        w.append(0.5 if sa == sb else float(sa > sb))
            return 100 * np.mean(w)
        def top(cond, ds):
            t = []
            for d in ds:
                sc = [(S[(cond, a)], 0) for a in docs[d][0]] + [(S[("clean", b)], 1) for b in docs[d][1]]
                m = max(s for s, _ in sc); best = [lab for s, lab in sc if s == m]
                t.append(best.count(0) / len(best))
            return 100 * np.mean(t)
        rng = np.random.default_rng(0); B = [list(rng.choice(D, len(D))) for _ in range(1000)]
        def ci(f):
            bs = np.percentile([f(b) for b in B], [2.5, 97.5]); return f"{f(D):+5.1f} [{bs[0]:+5.1f}, {bs[1]:+5.1f}]"
        out = f"{j:8s} {fmt:6s} docs {len(D)}  clean: wrong wins {win('clean', D):4.1f}%, top pick wrong {top('clean', D):4.1f}%"
        out += f" | claim_e on wrong: d win {ci(lambda ds: win('claim_e', ds) - win('clean', ds))}, d top {ci(lambda ds: top('claim_e', ds) - top('clean', ds))}"
        out += " | d win other: " + ", ".join(f"{c} {win(c, D) - win('clean', D):+.1f}" for c in ATT[1:])
        print(out)
