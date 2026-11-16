"""K1-v2R: rows for reasoning judges (think_judge.py) from N1 notes (2026-11-15).
For each test summary: atk_best = the note with the highest Qwen yes/no accept score among its 16 (score_vllm.py output
results/qwen_<tag>.jsonl); atk_rand = a note picked by a fixed hash of the item id. Prompts come from data/<tag>.jsonl
(yn rows). Clean yn rows of data/sprint.jsonl (all items, both parts) are added, so the output works with think_judge.py.
Guarded run (N1-D): --tag n1atk_guard --prompts data/n1atk.jsonl picks with the guarded Qwen scores and writes the
unguarded prompts; think_judge.py --guard then adds the guard line to every row.
Selection by a reasoning judge (N1-A, added 2026-11-15 19:10 UTC): --select-think <think_judge.py output with rows
free00..free15> picks, per summary, the note with verdict Yes and the highest p_yes (notes without Yes rank below).
Usage: python src/build_n1_think.py [--tag n1atk] [--out data/n1atk_think.jsonl] [--prompts data/<tag>.jsonl]"""
import argparse, hashlib, json

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="n1atk"); ap.add_argument("--out", default=None)
ap.add_argument("--prompts", default=None, help="row file for the prompts (default data/<tag>.jsonl); e.g. data/n1atk.jsonl "
                "when the tag is a defended run and think_judge.py --guard adds the guard line itself")
ap.add_argument("--select-think", default=None, help="think_judge.py output; pick by (verdict == Yes, p_yes)")
args = ap.parse_args()
out = args.out or f"data/{args.tag}_think.jsonl"
acc = lambda p: p[5] / (p[5] + p[6]) if p[5] + p[6] > 0 else -1
best = {}
src = args.select_think or f"results/qwen_{args.tag}.jsonl"
for r in map(json.loads, open(src)):
    if r["fmt"] == "yn" and r["cond"].startswith("free") and r["part"] == "test":
        s = (r["verdict"] == "Yes", r["p_yes"] or 0.0) if args.select_think else acc(r["p"])
        if r["item"] not in best or s > best[r["item"]][0]:
            best[r["item"]] = (s, r["cond"])
rand = {i: f"free{int(hashlib.md5(f'{i}|rand'.encode()).hexdigest(), 16) % 16:02d}" for i in best}
n = 0
with open(out, "w") as f:
    for r in map(json.loads, open("data/sprint.jsonl")):
        if r["fmt"] == "yn" and r["cond"] == "clean":
            f.write(json.dumps(r) + "\n"); n += 1
    for r in map(json.loads, open(args.prompts or f"data/{args.tag}.jsonl")):
        if r["fmt"] != "yn" or r["item"] not in best:
            continue
        for name, pick in [("atk_best", best[r["item"]][1]), ("atk_rand", rand[r["item"]])]:
            if r["cond"] == pick:
                f.write(json.dumps(r | {"cond": name, "src_cond": pick}) + "\n"); n += 1
print(n, "rows ->", out, "; summaries:", len(best))
