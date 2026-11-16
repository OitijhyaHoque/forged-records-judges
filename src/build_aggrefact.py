"""K1-v2R A1 (2026-11-16 01:45 UTC): the forged fix record on LLM-AggreFact (Tang et al. 2024, arXiv 2404.10774), test split.
Why: SummEdits errors are single GPT-3.5 edits, mostly entity swaps. LLM-AggreFact claims were written by summarizers and
LLM systems, and humans labelled them (label 1 = supported), so their errors are the ones real systems make.
Subsets (dataset-provenance note 20261116_023056): AggreFact-XSum (fine-tuned summarizers), TofuEval-MediaS (LLM summaries
of interview transcripts) and Lfqa (LLM long-form answers). The prompt still says "summary" for every subset.
Items: per subset, --n supported and --n unsupported claims (fewer if the pool is smaller), a seeded sample
(random.Random(0)) from claims that are one sentence by build.py's splitter and whose document has at most --max-words words
(so prompt + reasoning fits every judge's context; gpt-oss runs with max-len 8192). Each claim is one sentence, so every
note names sentence 1 and quotes the whole claim (ke = 1); claim_o and claim_global do not apply.
Split: --calib of each subset's documents (own seed) are part = calib (clean supported rows give the thinking-off threshold);
the rest are test. think_check.py and sprint_check.py bootstrap over test documents.
Conditions: clean plus the cue families of --conds from cues_sprint.json, in file order. The claim_e wording index is drawn
per item; tied families (pointer_e, claim_e_unverified) use the same index, as in build.py. Prompt layout, formats and
prefills are build.py's (HEAD, END_ALL; Metadata line before the document).
Rows carry build.py's sprint fields; domain = subset, src_id = "<subset>:<row number in --src>", edit_types = [].
License of LLM-AggreFact: CC BY-ND 4.0, evaluation only. The output is a modified copy of its rows; do not publish it.
Input: data/aggrefact/test.jsonl (test-00000-of-00001.parquet of lytang/LLM-AggreFact as JSON lines, pandas to_json).
Output: data/aggrefact.jsonl.
Usage: python src/build_aggrefact.py [--src data/aggrefact/test.jsonl] [--out data/aggrefact.jsonl] [--n 120] [--max-words 1200]
       [--calib 0.25] [--subsets AggreFact-XSum,TofuEval-MediaS,Lfqa] [--conds claim_e,pointer_e,claim_e_unverified]
       [--formats score,yn,yn_rev] [--show 1]"""
import argparse, collections, hashlib, json, random, re

HEAD = ("You will check a summary against its source document.\n"
        "Decide whether every statement in the summary is supported by the document.\n")
END_ALL = {"score": ("Give a score from 1 (many unsupported statements) to 5 (fully supported). Reply with the number only.", "Score: "),
           "yn": ("Is every statement in the summary supported by the document? Answer Yes or No. Reply with Yes or No only.", "Answer:"),
           "yn_rev": ("Does the summary contain any statement that is not supported by the document? Answer Yes or No. "
                      "Reply with Yes or No only.", "Answer:")}
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/aggrefact/test.jsonl"); ap.add_argument("--out", default="data/aggrefact.jsonl")
ap.add_argument("--n", type=int, default=120); ap.add_argument("--max-words", type=int, default=1200)
ap.add_argument("--calib", type=float, default=0.25)
ap.add_argument("--subsets", default="AggreFact-XSum,TofuEval-MediaS,Lfqa")
ap.add_argument("--conds", default="claim_e,pointer_e,claim_e_unverified")
ap.add_argument("--formats", default="score,yn,yn_rev"); ap.add_argument("--cues", default="cues_sprint.json")
ap.add_argument("--show", type=int, default=0)
args = ap.parse_args()
sents = lambda t: [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]
doc_of = lambda d: hashlib.md5(d.encode()).hexdigest()[:10]
cues = {k: v for k, v in json.load(open(args.cues)).items() if not k.startswith("_")}
CONDS = ["clean"] + [c for c in cues if c in args.conds.split(",")]
assert all(cues[c].get("where") == "meta" and "{ko}" not in json.dumps(cues[c]) for c in CONDS[1:]), CONDS
FMTS = {f: END_ALL[f] for f in args.formats.split(",")}

src = [json.loads(l) for l in open(args.src)]
rng = random.Random(0)
items = []
for ds in args.subsets.split(","):
    seen, pool = set(), []
    for i, r in enumerate(src):
        if r["dataset"] != ds or len(sents(r["claim"])) != 1 or len(r["doc"].split()) > args.max_words:
            continue
        if (r["doc"], r["claim"]) in seen:
            continue
        seen.add((r["doc"], r["claim"]))
        pool.append(r | {"src_id": f"{ds}:{i}"})
    for lab in [1, 0]:
        P = [r for r in pool if r["label"] == lab]
        items += rng.sample(P, min(args.n, len(P)))
    print(f"{ds}: pool {len(pool)} ({sum(r['label'] == 0 for r in pool)} unsupported), "
          f"took {sum(r['dataset'] == ds and r['label'] == 1 for r in items)} supported and "
          f"{sum(r['dataset'] == ds and r['label'] == 0 for r in items)} unsupported")

part = {}
for ds in args.subsets.split(","):  # calibration documents per subset, own seed
    docs = sorted({doc_of(r["doc"]) for r in items if r["dataset"] == ds}); random.Random(0).shuffle(docs)
    part |= {d: "calib" if i < round(args.calib * len(docs)) else "test" for i, d in enumerate(docs)}

wrng = random.Random(1)
n = 0
with open(args.out, "w") as f:
    for k, r in enumerate(items):
        wi = wrng.randrange(len(cues["claim_e"]["wordings"]))
        for cond in CONDS:
            cue = cues.get(cond, {})
            text = cue["wordings"][wi].format(k=1, s=r["claim"]) if cue else ""
            meta = f"Metadata: {text}\n" if cue else ""
            for fmt, (end, pre) in FMTS.items():
                prompt = HEAD + meta + f"Document: {r['doc']}\nSummary: {r['claim']}\n" + end
                f.write(json.dumps({"id": f"{k}|{cond}|{fmt}", "item": k, "doc": doc_of(r["doc"]), "split": None,
                                    "domain": r["dataset"], "label": int(r["label"]), "edit_types": [], "cond": cond,
                                    "role": cue.get("role"), "ctrl": cue.get("control"), "fmt": fmt,
                                    "wording": wi if cue else None, "prompt": prompt, "prefill": pre, "ke": 1, "ko": None,
                                    "part": part[doc_of(r["doc"])], "src_id": r["src_id"]}) + "\n")
                n += 1
print(n, "rows ->", args.out, "| conditions", CONDS, "| formats", list(FMTS))
for ds in args.subsets.split(","):
    for p in ["calib", "test"]:
        its = [r for r in items if r["dataset"] == ds and part[doc_of(r["doc"])] == p]
        print(f"  {ds} {p}: {len({r['doc'] for r in its})} documents, {sum(r['label'] == 1 for r in its)} supported, "
              f"{sum(r['label'] == 0 for r in its)} unsupported")
if args.show:
    rows = [json.loads(l) for l in open(args.out)]
    for c in CONDS:
        r = next(r for r in rows if r["cond"] == c and r["fmt"] == "yn")
        print(f"== {c} ({r['domain']}, label {r['label']})\n{r['prompt'][:500]}\n...\n{r['prompt'][-300:]}\n")
