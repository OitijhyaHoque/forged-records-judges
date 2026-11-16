"""K1-v2R sprint T3: check a smoke-run output of score_vllm.py.
Per format: answer-token mass (mean, min, share below 0.5), the most common top-1 tokens, the share of rows where
some answer token fell outside the top K (p = 0), and the mean accept score by label.
Accept score: expected 1-5 score; p(Yes) share for yn; p(No) share for yn_rev (there "No" accepts).
Usage: python src/smoke_check.py results/qwen_smoke.jsonl [...]"""
import json, sys
from collections import Counter

ANS_IDX = {"score": range(5), "yn": [5, 6], "yn_rev": [5, 6], "sent": [5, 6], "sent_rev": [5, 6]}  # sent*: E1

def accept(p, fmt):
    if fmt == "score":
        z = sum(p[:5])
        return sum((i + 1) * p[i] for i in range(5)) / z if z else float("nan")
    z = p[5] + p[6]
    return (p[5] if fmt in ("yn", "sent") else p[6]) / z if z else float("nan")

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    print(f"===== {path}: {len(rows)} rows, prompt tokens max {max(r['n_tok'] for r in rows)}")
    for fmt in ANS_IDX:
        R = [r for r in rows if r["fmt"] == fmt]
        if not R:
            continue
        m = [r["mass"] for r in R]
        zero = sum(any(r["p"][i] == 0 for i in ANS_IDX[fmt]) for r in R)
        top1 = Counter(r["top"][0][0] for r in R).most_common(5)
        acc = {y: [accept(r["p"], fmt) for r in R if r["label"] == y] for y in [1, 0]}
        acc = {("consistent" if y else "inconsistent"): round(sum(v) / len(v), 3) if v else None for y, v in acc.items()}
        print(f"{fmt:7s} n={len(R):3d} mass mean {sum(m) / len(m):.3f} min {min(m):.3f} <0.5: {sum(x < 0.5 for x in m)}"
              f" | answer token outside top K: {zero} | top-1 {top1} | mean accept {acc}")
    low = sorted(rows, key=lambda r: r["mass"])[:3]
    for r in low:
        print(f"  lowest mass {r['mass']} ({r['fmt']}, {r['cond']}): top {r['top']}")
