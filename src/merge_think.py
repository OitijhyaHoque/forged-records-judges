"""K1-v2R (2026-11-15 20:10 UTC): merge think_judge.py outputs of one run whose conds were split across GPUs.
The merged file has the rows of the unsplit run, in the order of the data file (think_judge.py keeps that order), so ties
in build_n1_think.py --select-think resolve as in the unsplit run. A row id found in more than one part (the calib rows)
is taken from the first part. Stops if a row of the unsplit run is missing or a part has a row the unsplit run lacks.
Usage: python src/merge_think.py --data data/n1aware.jsonl --conds clean,free00,...,free15 --out <merged.jsonl> <part.jsonl> ..."""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--conds", required=True); ap.add_argument("--out", required=True)
ap.add_argument("parts", nargs="+")
args = ap.parse_args()
conds = args.conds.split(",")
want = [r["id"] for r in map(json.loads, open(args.data)) if r["fmt"] == "yn" and (  # the row filter of think_judge.py
    (r["part"] == "test" and r["cond"] in conds) or (r["part"] == "calib" and r["cond"] == "clean" and r["label"] == 1))]
assert len(set(want)) == len(want), "row ids in the data file are not unique"
got, n_part = {}, {}
for p in args.parts:
    rows = [json.loads(l) for l in open(p)]
    n_part[p] = len(rows)
    for r in rows:
        got.setdefault(r["id"], r)
missing, extra = [i for i in want if i not in got], set(got) - set(want)
assert not missing and not extra, f"missing {len(missing)} rows (e.g. {missing[:3]}), extra {len(extra)}"
with open(args.out, "w") as f:
    for i in want:
        f.write(json.dumps(got[i]) + "\n")
print(len(want), "rows ->", args.out, "; rows per part:", n_part)
