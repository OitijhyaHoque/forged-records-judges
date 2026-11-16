"""K1-v2R E1b (pre-registered E1/E1b plan, section 8c): the E1 `sent` question about
every sentence of the summary, with no oracle location.
Reads the yn rows of data/sprint.jsonl for the conditions clean, claim_e, claim_o and claim_e_unverified. Keeps each
prompt up to and including the Summary line, as build_e1.py does, and asks about each sentence k = 1..n:
  sent  Is sentence k of the summary ("...") supported by the document?  Yes accepts.
Check: the row for k = ke has the same prompt as the E1 sent row in data/e1.jsonl.
Output: data/e1b.jsonl, row id item|cond|sent|k; all other fields as in sprint.jsonl, plus q = "all", k and is_e.
Usage: python src/build_e1b.py [--data data/sprint.jsonl] [--e1 data/e1.jsonl] [--out data/e1b.jsonl]"""
import argparse, json, re

END_YN = "Is every statement in the summary supported by the document? Answer Yes or No. Reply with Yes or No only."
Q = 'Is sentence {k} of the summary ("{s}") supported by the document? Answer Yes or No. Reply with Yes or No only.'
CONDS = ["clean", "claim_e", "claim_o", "claim_e_unverified"]
sents = lambda t: [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/sprint.jsonl"); ap.add_argument("--e1", default="data/e1.jsonl")
ap.add_argument("--out", default="data/e1b.jsonl")
args = ap.parse_args()

e1 = {r["id"]: r["prompt"] for r in map(json.loads, open(args.e1)) if r["fmt"] == "sent"}
rows = [r for r in map(json.loads, open(args.data)) if r["fmt"] == "yn" and r["cond"] in CONDS]
n, same = 0, 0
with open(args.out, "w") as f:
    for r in rows:
        assert r["prompt"].endswith(END_YN), r["id"]
        body = r["prompt"][:-len(END_YN)]
        summary = r["prompt"].rsplit("\nSummary: ", 1)[1].rsplit("\n", 1)[0]
        for k, s in enumerate(sents(summary), 1):
            prompt = body + Q.format(k=k, s=s)
            if k == r["ke"]:
                assert prompt == e1[f"{r['item']}|{r['cond']}|sent"], f"{r['id']}: k = ke prompt differs from E1"
                same += 1
            f.write(json.dumps(r | {"id": f"{r['item']}|{r['cond']}|sent|{k}", "fmt": "sent", "q": "all", "k": k,
                                    "is_e": k == r["ke"], "prompt": prompt}) + "\n")
            n += 1
print(f"{len(rows)} yn rows ({len(CONDS)} conditions) -> {n} rows in {args.out}; {same} rows with k = ke match E1")
