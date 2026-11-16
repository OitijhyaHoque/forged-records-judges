"""K1-v2R E-R readout: reasoning judges (think_judge.py), yes/no verdict with thinking on, test part.
Accept = verdict Yes. Miss = wrong summary accepted; FA = correct summary rejected. Rows without a verdict are left out
(counted). Every condition is compared with clean of the same file. Paired bootstrap over test documents (1,000, seed 0).
For comparison (files without a suffix only): the same judge with thinking off (results/<judge>_full.jsonl, yn format,
threshold = 10% quantile of clean calibration consistent accept scores), same test items and conditions.
Also: share of reasoning traces with a note that mention it, flips (wrong summaries rejected clean and accepted with the
note), and reasoning excerpts from flips.
--suffix _guard reads results/think_<judge>_guard.jsonl (prompt with the GUARD line, see think_judge.py).
--base-from: also compare each condition with clean of another file (for example the unguarded file), so the guard's
cost on clean items is visible.
--merge-from '' adds the base file's conditions that a run lacks (the _forms run has no clean rows).
--pairs claim_e-claim_o adds contrasts between two conditions.
--off results/{j}_heldout.jsonl compares with that thinking-off file for any suffix (held-out domains: --suffix _heldout).
Usage: python src/think_check.py [--judges nemotron,gemma,qwen,gptoss] [--suffix ""] [--examples 3] [--json results/think_check.json]"""
import argparse, json, os, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--judges", default="nemotron,gemma,qwen,gptoss"); ap.add_argument("--examples", type=int, default=3)
ap.add_argument("--suffix", default=""); ap.add_argument("--json", default=None)
ap.add_argument("--base-from", default=None, help="suffix of a second file whose clean rows are a second baseline ('' = unguarded)")
ap.add_argument("--merge-from", default=None, help="suffix of a second file; its test rows of conditions missing here are added "
                "(e.g. clean and claim_e from the base file for the _forms run)")
ap.add_argument("--pairs", default="", help="extra contrasts a-b of miss and FA, comma list, e.g. claim_e-claim_o,bare_e-pointer_e")
ap.add_argument("--off", default=None, help="thinking-off file, {j} = judge (e.g. results/{j}_heldout.jsonl); used with any suffix")
args = ap.parse_args()
NOTE = re.compile(r"metadata|version [12]\b|revision [12]\b|draft [12]\b|re-check|fact-check|writer's fix|reviewer note|"
                  r"version history|was fixed|corrected sentence", re.I)
out = {}

def boot(ix_by_doc, f, nb=1000):
    docs = sorted(ix_by_doc); rng = np.random.default_rng(0)
    full = np.concatenate([ix_by_doc[d] for d in docs])
    bs = [f(np.concatenate([ix_by_doc[d] for d in rng.choice(docs, len(docs))])) for _ in range(nb)]
    return [float(f(full)), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
fci = lambda v: f"{v[0]:+5.1f} [{v[1]:+5.1f}, {v[2]:+5.1f}]"

def load(path):
    rows = [json.loads(l) for l in open(path)]
    return rows, {(r["cond"], r["item"]): r for r in rows if r["part"] == "test"}

for j in args.judges.split(","):
    path = f"results/think_{j}{args.suffix}.jsonl"
    if not os.path.exists(path):
        print(f"\n===== {j}: {path} missing"); continue
    rows, T = load(path)
    if args.merge_from is not None:
        have = {c for c, _ in T}
        rows0, T0m = load(f"results/think_{j}{args.merge_from}.jsonl")
        T |= {k: v for k, v in T0m.items() if k[0] not in have}
        rows = rows + [r for r in rows0 if r["part"] == "test" and r["cond"] not in have]
    conds = [c for c in dict.fromkeys(r["cond"] for r in rows if r["part"] == "test") if c != "clean"]
    its = sorted({i for c, i in T if c == "clean"})
    y = np.array([T["clean", i]["label"] for i in its]); doc = np.array([T["clean", i]["doc"] for i in its])
    verdict = lambda r: np.nan if r["verdict"] is None else float(r["verdict"] == "Yes")
    A = {c: np.array([verdict(T[c, i]) for i in its]) for c in ["clean"] + conds}
    if args.base_from is not None:
        _, T0 = load(f"results/think_{j}{args.base_from}.jsonl")
        A["clean_base"] = np.array([verdict(T0["clean", i]) for i in its])
    nov = {c: int(np.isnan(v).sum()) for c, v in A.items()}
    trunc = sum(r["finish"] == "length" for r in rows)
    bydoc = {d: np.where(doc == d)[0] for d in np.unique(doc)}
    allix = np.arange(len(its))
    miss = lambda c, ix: 100 * np.nanmean(A[c][ix][y[ix] == 0])
    fa = lambda c, ix: 100 * np.nanmean(1 - A[c][ix][y[ix] == 1])
    res = {"file": path, "n_items": len(its), "no_verdict": nov, "truncated": trunc, "conds": {}}
    print(f"\n===== {j} (thinking on, file {path}): {len(its)} test items ({(y == 0).sum()} wrong); no verdict {nov}; "
          f"cut at max tokens {trunc}")
    print(f"  clean: miss {miss('clean', allix):.1f}%, FA {fa('clean', allix):.1f}%, mean generated tokens "
          f"{np.mean([T['clean', i]['n_gen'] for i in its]):.0f}")
    res["clean"] = {"miss": miss("clean", allix), "fa": fa("clean", allix)}
    if "clean_base" in A:
        dmb = boot(bydoc, lambda ix: miss("clean", ix) - miss("clean_base", ix))
        dfb = boot(bydoc, lambda ix: fa("clean", ix) - fa("clean_base", ix))
        res["clean_vs_base"] = {"d_miss": dmb, "d_fa": dfb}
        print(f"  clean of this file vs clean of the base file ({args.base_from!r}): d miss {fci(dmb)}, d FA {fci(dfb)}")
    off = (args.off or "results/{j}_full.jsonl").format(j=j)
    O = None
    if (args.suffix == "" or args.off) and os.path.exists(off):
        R = [json.loads(l) for l in open(off)]
        acc = lambda p: p[5] / (p[5] + p[6]) if p[5] + p[6] > 0 else np.nan
        cal = [acc(r["p"]) for r in R if r["fmt"] == "yn" and r["cond"] == "clean" and r["part"] == "calib" and r["label"] == 1]
        t = np.quantile(cal, 0.10)
        O = {(r["cond"], r["item"]): float(acc(r["p"]) >= t) for r in R if r["fmt"] == "yn" and r["part"] == "test"}
        res["thinking_off_threshold"] = float(t)
    for c in conds:
        dm = boot(bydoc, lambda ix: miss(c, ix) - miss("clean", ix))
        df = boot(bydoc, lambda ix: fa(c, ix) - fa("clean", ix))
        men = np.array([bool(NOTE.search(T[c, i]["reasoning"])) for i in its])
        flips = [k for k in range(len(its)) if y[k] == 0 and A["clean"][k] == 0 and A[c][k] == 1]
        back = [k for k in range(len(its)) if y[k] == 0 and A["clean"][k] == 1 and A[c][k] == 0]
        r = {"miss": miss(c, allix), "fa": fa(c, allix), "d_miss": dm, "d_fa": df, "note_mentioned_pct": float(100 * men.mean()),
             "flips_wrong_to_accept": len(flips), "flips_wrong_to_reject": len(back), "flip_items": [int(its[k]) for k in flips],
             "mean_n_gen": float(np.mean([T[c, i]["n_gen"] for i in its]))}
        print(f"  {c}: miss {r['miss']:.1f}% (d {fci(dm)}), FA {r['fa']:.1f}% (d {fci(df)}); reasoning mentions the note "
              f"{r['note_mentioned_pct']:.1f}%; wrong summaries flipped to accept {len(flips)}, to reject {len(back)}; "
              f"mean generated tokens {r['mean_n_gen']:.0f}")
        if "clean_base" in A:
            r["d_miss_vs_base"] = boot(bydoc, lambda ix: miss(c, ix) - miss("clean_base", ix))
            print(f"      vs clean of the base file: d miss {fci(r['d_miss_vs_base'])}")
        if O is not None and all((c, i) in O for i in its):
            B = {cc: np.array([O[cc, i] for i in its]) for cc in ["clean", c]}
            om = lambda cc, ix: 100 * np.mean(B[cc][ix][y[ix] == 0]); of = lambda cc, ix: 100 * np.mean(1 - B[cc][ix][y[ix] == 1])
            r["thinking_off"] = {"miss_clean": om("clean", allix), "miss": om(c, allix), "fa_clean": of("clean", allix),
                                 "fa": of(c, allix), "d_miss": boot(bydoc, lambda ix: om(c, ix) - om("clean", ix)),
                                 "d_fa": boot(bydoc, lambda ix: of(c, ix) - of("clean", ix))}
            o = r["thinking_off"]
            print(f"      thinking off: miss clean {o['miss_clean']:.1f}, {c} {o['miss']:.1f}, d {fci(o['d_miss'])}; "
                  f"FA clean {o['fa_clean']:.1f}, {c} {o['fa']:.1f}, d {fci(o['d_fa'])}")
        for k in flips[:args.examples]:
            rs = T[c, its[k]]["reasoning"]
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", rs) if NOTE.search(s)]
            print(f"    item {its[k]} ({T[c, its[k]]['edit_types']}): " + " | ".join(s[:200] for s in sents[:3]))
            print(f"      end of reasoning: {rs[-220:]!r}")
        res["conds"][c] = r
    res["pairs"] = {}
    for pr in [x for x in args.pairs.split(",") if x]:
        a_, b_ = pr.split("-")
        if a_ in A and b_ in A:
            dm = boot(bydoc, lambda ix: miss(a_, ix) - miss(b_, ix)); df = boot(bydoc, lambda ix: fa(a_, ix) - fa(b_, ix))
            res["pairs"][pr] = {"d_miss": dm, "d_fa": df}
            print(f"  pair {a_} - {b_}: d miss {fci(dm)}, d FA {fci(df)}")
    out[j] = res
js = args.json or f"results/think_check{args.suffix}.json"
json.dump(out, open(js, "w"), indent=1)
print("\nwrote", js)
