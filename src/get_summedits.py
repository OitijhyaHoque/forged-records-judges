"""Download SummEdits (Laban et al. 2023; Hugging Face dataset Salesforce/summedits, CC BY 4.0, 6,348 rows) and write
data/summedits.json in the row order and layout of the copy used for the runs.
build.py draws items with a seeded random generator, so the row order must be the same: meta/summedits_ids.txt lists the
ids in that order. The copy used for the runs has md5 20457d772cb05fefa5cedf19c7937ab9. If your md5 differs (for example
because a field type changed on the Hub), check the built prompt sets with md5sum -c meta/data_md5.txt; those are what
matter for the results.
Usage: python src/get_summedits.py [--out data/summedits.json]"""
import argparse, hashlib, json, os
from datasets import load_dataset

KEYS = ["domain", "id", "doc", "summary", "label", "seed_summary", "edit_types"]
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/summedits.json"); ap.add_argument("--ids", default="meta/summedits_ids.txt")
args = ap.parse_args()
dd = load_dataset("Salesforce/summedits")
rows = {}
for split in dd:
    for r in dd[split]:
        rows[r["id"]] = r
order = [l.strip() for l in open(args.ids) if l.strip()]
missing = [i for i in order if i not in rows]
if missing:
    raise SystemExit(f"{len(missing)} ids of {args.ids} are not in the download, e.g. {missing[:3]}")
data = [{k: rows[i][k] for k in KEYS} for i in order]
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as f:
    json.dump(data, f, indent=4)
md5 = hashlib.md5(open(args.out, "rb").read()).hexdigest()
print(f"wrote {len(data)} rows to {args.out}, md5 {md5}",
      "(same as the runs)" if md5 == "20457d772cb05fefa5cedf19c7937ab9" else "(differs from the runs' copy)")
