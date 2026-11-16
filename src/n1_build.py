"""K1-v2R N1: judge rows for the free notes, in the data/sprint.jsonl layout, for score_vllm.py.
Rows:
  cond free00..free15  each note of n1_write.py in the "Metadata:" line (the slot of claim_e), summaries of the notes file
  cond clean           all items and both parts (threshold and a check against the earlier run)
  cond claim_e         the summaries of the notes file (the hand-written note, same run)
Formats yn, yn_rev, score; rows are ordered by format in that order, so partial output covers yn first.
Usage: python src/n1_build.py [--notes data/n1_notes.jsonl] [--sprint data/sprint.jsonl] [--out data/n1.jsonl]"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--notes", default="data/n1_notes.jsonl"); ap.add_argument("--sprint", default="data/sprint.jsonl")
ap.add_argument("--out", default="data/n1.jsonl")
args = ap.parse_args()

notes = {r["item"]: r for r in map(json.loads, open(args.notes))}
sprint = [json.loads(l) for l in open(args.sprint)]
FMTS = ["yn", "yn_rev", "score"]
out = []
for r in sprint:
    if r["cond"] == "clean" or (r["cond"] == "claim_e" and r["item"] in notes):
        out.append(r)
    if r["cond"] == "clean" and r["item"] in notes:
        head, body = r["prompt"].split("\nDocument: ", 1)
        for j, note in enumerate(notes[r["item"]]["notes"]):
            out.append(r | {"id": f"{r['item']}|free{j:02d}|{r['fmt']}", "cond": f"free{j:02d}", "role": "n1", "ctrl": None,
                            "wording": j, "prompt": f"{head}\nMetadata: {note}\nDocument: {body}"})
out.sort(key=lambda r: FMTS.index(r["fmt"]))  # stable: keeps the sprint order inside a format
with open(args.out, "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(len(out), "rows ->", args.out, {fm: sum(r["fmt"] == fm for r in out) for fm in FMTS})
