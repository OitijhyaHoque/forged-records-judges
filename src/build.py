"""Build the K1-v2 pilot-2 prompts from SummEdits. Same prompt, domains and split rule as pilot 1 (an October 2026 pilot).
Changes from pilot 1:
- N items (default 1000). The N/2 inconsistent items are stratified by edit type (N/8 each), so the harder
  types (hallucinated insertion, negation) are no longer rare (pilot 1 had 10-11 of each in 300).
  Most items have 2 edit types; an item's stratum is the rarest type it contains (TYPES is ordered rarest first).
- Conditions are clean plus every family in cues.json, in file order. Rows carry the family's role and control.
Output: data/pilot.jsonl, one row per item x condition x format.
Rescue run (--single --cues cues_rescue.json --exclude data/pilot.jsonl): N/2 consistent items and N/2 inconsistent
items with one edit in one sentence, none from pilot 2. Cue wordings may name that sentence ({k}, {s}).
'turn2' cues add a `turn2` field (a second user turn, see score.py); their prompt is the clean prompt.
Sprint run K1-v2R (pre-registered sprint plan):
  --single --multi --cues cues_sprint.json --exclude-ids meta/summedits_exclude_ids.txt (the items of the October pilots)
  --n-inc 200 --n-con 200 --formats score,yn,yn_rev --calib 0.25
  --multi keeps summaries with 2+ sentences. Cues may also name another sentence ({ko}, {so}; fixed pick by item id).
  A cue with "tie" uses the same wording index as the named family. --n-inc/--n-con take at most what the pool has.
  yn_rev asks whether any statement is unsupported, so "No" accepts. --calib marks a fraction of documents
  (own seed) as `part` = calib; the rest are test. Sprint rows also carry ke/ko (1-based sentence numbers).
Held-out domains (2026-11-15): --domains podcast,sales_call --conds claim_e,pointer_e,claim_e_unverified with the sprint flags.
With --cues cues.json or cues_rescue.json (pilot wordings, not included here) and without the sprint flags, the output is
byte-identical to pilot 2 and the rescue run."""
import argparse, json, random, hashlib, re

DOMAINS = ["news", "samsum", "sales_email", "scitldr", "ectsum", "qmsumm"]  # podcast, sales_call kept for a held-out domain
TYPES = ["negation_insertion_removal", "hallucinated_fact_insertion", "antonym_swap", "entity_modification"]
HEAD = ("You will check a summary against its source document.\n"
        "Decide whether every statement in the summary is supported by the document.\n")
END = {"score": ("Give a score from 1 (many unsupported statements) to 5 (fully supported). Reply with the number only.", "Score: "),
       "yn": ("Is every statement in the summary supported by the document? Answer Yes or No. Reply with Yes or No only.", "Answer:")}
END_ALL = END | {"yn_rev": ("Does the summary contain any statement that is not supported by the document? Answer Yes or No. "
                            "Reply with Yes or No only.", "Answer:")}

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1000); ap.add_argument("--out", default="data/pilot.jsonl")
ap.add_argument("--summedits", default="data/summedits.json")
ap.add_argument("--cues", default="cues_sprint.json")
ap.add_argument("--single", action="store_true", help="inconsistent items: one edit type, one changed sentence")
ap.add_argument("--exclude", nargs="+", help="earlier builds; their items are not sampled")
ap.add_argument("--exclude-ids", help="file with one SummEdits id per line; these items are not sampled "
                "(meta/summedits_exclude_ids.txt holds the items of the October pilots, the same set as --exclude with those builds)")
ap.add_argument("--multi", action="store_true", help="only summaries with 2 or more sentences")
ap.add_argument("--n-inc", type=int, help="inconsistent items (default n/2); at most the pool size")
ap.add_argument("--n-con", type=int, help="consistent items (default n/2); at most the pool size")
ap.add_argument("--formats", default="score,yn", help="comma list from: " + ",".join(END_ALL))
ap.add_argument("--calib", type=float, help="fraction of documents marked part=calib (threshold data)")
ap.add_argument("--domains", help="comma list of SummEdits domains (default: the six of DOMAINS); e.g. podcast,sales_call")
ap.add_argument("--conds", help="comma list of cue families to write (default: all); wording draws still use every family")
args = ap.parse_args()
if args.domains:
    DOMAINS = args.domains.split(",")
FMTS = {f: END_ALL[f] for f in args.formats.split(",")}
sprint = args.multi or args.calib is not None or args.formats != "score,yn"

cues = {k: v for k, v in json.load(open(args.cues)).items() if not k.startswith("_")}
CONDS = ["clean"] + [c for c in cues if not args.conds or c in args.conds.split(",")]
rng = random.Random(0)
data = [r for r in json.load(open(args.summedits)) if r["domain"] in DOMAINS]
doc_of = lambda r: hashlib.md5(r["doc"].encode()).hexdigest()[:10]
sents = lambda t: [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]
changed = lambda r: [i for i, s in enumerate(sents(r["summary"])) if s not in sents(r["seed_summary"])]
pool = data
for path in args.exclude or []:
    old = [json.loads(l) for l in open(path)]
    old = {(o["doc"], o["prompt"].rsplit("\nSummary: ", 1)[1].rsplit("\n", 1)[0]) for o in old if o["cond"] == "clean"}
    n0, pool = len(pool), [r for r in pool if (doc_of(r), r["summary"]) not in old]
    print(f"--exclude: {n0 - len(pool)} of {n0} rows were in {path}")
if args.exclude_ids:
    ex = {l.strip() for l in open(args.exclude_ids) if l.strip()}
    n0, pool = len(pool), [r for r in pool if r["id"] not in ex]
    print(f"--exclude-ids: {n0 - len(pool)} of {n0} rows are in {args.exclude_ids}")
if args.multi:
    pool = [r for r in pool if len(sents(r["summary"])) >= 2]
    print(f"--multi: {len(pool)} rows have 2 or more sentences")
take = lambda rows, n: rng.sample(rows, min(n, len(rows)))
con = [r for r in pool if r["label"] == 1]
items = rng.sample(con, args.n // 2) if args.n_con is None else take(con, args.n_con)
nc = len(items)
if args.single:
    inc = [r for r in pool if r["label"] == 0 and len(r["edit_types"] or []) == 1 and len(changed(r)) == 1]
    items += rng.sample(inc, args.n // 2) if args.n_inc is None else take(inc, args.n_inc)
    print(f"items: {nc} consistent of {len(con)}, {len(items) - nc} inconsistent of {len(inc)}")
else:
    stratum = lambda r: next((t for t in TYPES if t in (r["edit_types"] or [])), None)
    for t in TYPES:
        items += rng.sample([r for r in pool if r["label"] == 0 and stratum(r) == t], args.n // 8)

def target(r):  # (number, text) of the sentence that differs from the seed summary; else a fixed pick by item id
    c, s = changed(r), sents(r["summary"])
    k = c[0] if len(c) == 1 else int(hashlib.md5(r["id"].encode()).hexdigest(), 16) % len(s)
    return k + 1, s[k]

def other(r, ke):  # (number, text) of another sentence of the summary, a fixed pick by item id; None if only one
    s = sents(r["summary"])
    rest = [i for i in range(len(s)) if i != ke - 1]
    if not rest:
        return None, None
    k = rest[int(hashlib.md5((r["id"] + "|o").encode()).hexdigest(), 16) % len(rest)]
    return k + 1, s[k]

docs = sorted({doc_of(r) for r in data}); rng.shuffle(docs)
split = {d: "probe" if i < .5 * len(docs) else "choice" if i < .7 * len(docs) else "test" for i, d in enumerate(docs)}
if args.calib is not None:  # own rng, so the item and wording draws do not change
    cdocs = sorted({doc_of(r) for r in items}); random.Random(0).shuffle(cdocs)
    part = {d: "calib" if i < round(args.calib * len(cdocs)) else "test" for i, d in enumerate(cdocs)}

lens = {}
with open(args.out, "w") as f:
    for k, r in enumerate(items):
        w = {c: rng.randrange(len(cues[c]["wordings"])) for c in cues if "tie" not in cues[c]}
        w |= {c: w[cues[c]["tie"]] for c in cues if "tie" in cues[c]}
        tk, ts = target(r)
        ko, so = other(r, tk)
        for cond in CONDS:
            cue = cues.get(cond, {})
            text = cue["wordings"][w[cond]].format(k=tk, s=ts, ko=ko, so=so) if cue else ""
            meta = f"Metadata: {text}\n" if cue.get("where") == "meta" else ""
            pad = text + " " if cue.get("where") == "pad" else ""
            lens.setdefault(cond, []).append(len(text))
            for fmt, (end, pre) in FMTS.items():
                prompt = HEAD + meta + f"Document: {r['doc']}\nSummary: {pad}{r['summary']}\n" + end
                row = {"id": f"{k}|{cond}|{fmt}", "item": k, "doc": doc_of(r), "split": split[doc_of(r)],
                       "domain": r["domain"], "label": r["label"], "edit_types": r["edit_types"],
                       "cond": cond, "role": cue.get("role"), "ctrl": cue.get("control"), "fmt": fmt,
                       "wording": w.get(cond), "prompt": prompt, "prefill": pre}
                if sprint:
                    row |= {"ke": tk, "ko": ko, "part": part[doc_of(r)] if args.calib is not None else None,
                            "src_id": r["id"]}
                if cue.get("where") == "turn2":
                    row["turn2"] = f"{text}\n{end}"
                f.write(json.dumps(row) + "\n")
print(len(items) * len(CONDS) * len(FMTS), "rows ->", args.out)
if sprint:
    print("mean cue length in characters:", {c: round(sum(v) / len(v)) for c, v in lens.items()})
    if args.calib is not None:
        for p in ["calib", "test"]:
            its = [r for r in items if part[doc_of(r)] == p]
            print(f"part {p}: {sum(d == p for d in part.values())} documents, {sum(r['label'] == 1 for r in its)} consistent,"
                  f" {sum(r['label'] == 0 for r in its)} inconsistent")
