"""Readout of the trace labels (Section 4.2, Figure 3 right): does the judge find the error and then defer to the note?
Reads results/think_labels.jsonl (label_traces.py; one row per wrong test summary, judge and condition clean or claim_e,
with found = the reasoning says some statement is not supported, defers = the reasoning uses the note as a reason to accept).
Per judge it prints:
  found clean    wrong summaries whose no-note trace found the error
  also found     share of those whose forged-note trace also found it
  acc clean/note share of those accepted without and with the forged note
  flips          wrong summaries rejected without the note and accepted with it; how many of the flipped forged-note
                 traces state the error (found), and how many use the note as a reason to accept (defers)
Usage: python src/label_check.py [--labels results/think_labels.jsonl]"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--labels", default="results/think_labels.jsonl")
args = ap.parse_args()
L = [json.loads(l) for l in open(args.labels)]
by = {(r["judge"], r["cond"], r["item"]): r for r in L}
print(f"{len(L)} labelled traces from {args.labels}")
for j in ["qwen", "nemotron", "gemma", "gptoss"]:
    items = sorted({r["item"] for r in L if r["judge"] == j})
    if not items:
        continue
    found = [i for i in items if by[j, "clean", i]["found"] == 1]
    also = sum(by[j, "claim_e", i]["found"] == 1 for i in found)
    acc0 = sum(by[j, "clean", i]["verdict"] == "Yes" for i in found)
    acc1 = sum(by[j, "claim_e", i]["verdict"] == "Yes" for i in found)
    flips = [i for i in items if by[j, "clean", i]["verdict"] != "Yes" and by[j, "claim_e", i]["verdict"] == "Yes"]
    f_found = sum(by[j, "claim_e", i]["found"] == 1 for i in flips)
    f_defer = sum(by[j, "claim_e", i]["defers"] == 1 for i in flips)
    f_lost = sum(by[j, "clean", i]["found"] == 1 and by[j, "claim_e", i]["found"] == 0 for i in flips)
    print(f"{j:9s} wrong {len(items)}; found clean {len(found)}; also found with note {100 * also / max(len(found), 1):.1f}%; "
          f"accepted clean {100 * acc0 / max(len(found), 1):.1f}%, with note {100 * acc1 / max(len(found), 1):.1f}%")
    print(f"{'':9s} flips {len(flips)}: note trace finds the error {f_found}, uses the note to accept {f_defer}, "
          f"found without the note but not with it {f_lost}")
