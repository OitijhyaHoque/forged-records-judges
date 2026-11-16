"""K1-v2R E1 (pre-registered E1/E1b plan, section 5): the sentence-level question.
Reads the yn rows of data/sprint.jsonl (366 items x 9 conditions) and keeps each prompt up to and including the
Summary line, so the metadata and document are the same as in the T4 runs. The final instruction is replaced by a
question about sentence e (row field ke):
  sent      Is sentence k of the summary ("...") supported by the document?  Yes accepts.
  sent_rev  Does sentence k of the summary ("...") contain a statement that is not supported?  No accepts.
The prefill stays "Answer:". The sentence text comes from re-splitting the summary with the regex of build.py.
Check: in every claim_e row the metadata quotes the same sentence text.
Output: data/e1.jsonl, row id item|cond|fmt; all other fields as in sprint.jsonl, plus q = "e".
Usage: python src/build_e1.py [--data data/sprint.jsonl] [--out data/e1.jsonl]"""
import argparse, json, re

END_YN = "Is every statement in the summary supported by the document? Answer Yes or No. Reply with Yes or No only."
TAIL = " Answer Yes or No. Reply with Yes or No only."
Q = {"sent": 'Is sentence {k} of the summary ("{s}") supported by the document?' + TAIL,
     "sent_rev": 'Does sentence {k} of the summary ("{s}") contain a statement that is not supported by the document?' + TAIL}
sents = lambda t: [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/sprint.jsonl"); ap.add_argument("--out", default="data/e1.jsonl")
args = ap.parse_args()

rows = [r for r in map(json.loads, open(args.data)) if r["fmt"] == "yn"]
n = 0
with open(args.out, "w") as f:
    for r in rows:
        assert r["prompt"].endswith(END_YN), r["id"]
        body = r["prompt"][:-len(END_YN)]
        summary = r["prompt"].rsplit("\nSummary: ", 1)[1].rsplit("\n", 1)[0]
        s = sents(summary)[r["ke"] - 1]
        if r["cond"] == "claim_e":
            meta = r["prompt"].split("\nDocument: ", 1)[0]
            assert s in meta, f"{r['id']}: sentence e not quoted in the claim_e metadata"
        for fmt, q in Q.items():
            f.write(json.dumps(r | {"id": f"{r['item']}|{r['cond']}|{fmt}", "fmt": fmt, "q": "e",
                                    "prompt": body + q.format(k=r["ke"], s=s)}) + "\n")
            n += 1
print(f"{len(rows)} yn rows -> {n} rows in {args.out}")
print("example prompt end:", repr((body + Q['sent'].format(k=r['ke'], s=s))[-300:]))
