"""K1-v2R N1 readout: best-of-k free reviewer notes (n1_write.py, n1_build.py, score_vllm.py).
Accept score, threshold and miss/FA as in sprint_check.py: threshold = 10% quantile of clean accept scores on calibration
consistent items, from this run's clean rows. Test part only. Bootstrap over test documents (1,000 resamples, seed 0).
Best-of-k: the 16 notes of a summary are cut into 16/k disjoint groups; in each group the selector (judge, format) keeps
the note with the highest accept score; the evaluator (judge, format) scores the kept note; rates are averaged over groups.
k = 1 is the average single note. Selector = evaluator is in-sample; another judge as evaluator is held out.
Parts:
  1  check: clean scores of this run against the earlier run (same judge, same prompts)
  2  miss and FA by k for selector Qwen yn, every evaluator; clean and claim_e (hand-written note) for reference
  3  in-sample ceiling: every judge-format cell selects for itself
  4  note content: labels A (claims checked/accurate) and B (flags an error) among all notes and among kept notes
  5  which notes raise acceptance: accept minus the summary's mean over its 16 notes, by label class
  6  selection within a document (as select_check.py): top pick wrong and pairwise win rate
Usage: python src/n1_check.py [--notes data/n1_notes.jsonl] [--json results/n1_check.json]"""
import argparse, json, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--notes", default="data/n1_notes.jsonl"); ap.add_argument("--json", default="results/n1_check.json")
ap.add_argument("--judges", default="qwen,nemotron")
ap.add_argument("--tag", default="n1", help="judge files results/<judge>_<tag>.jsonl (n1: cooperative writer; n1atk: attack prompt)")
ap.add_argument("--fmts", default="yn,yn_rev,score", help="formats in the judge files (defense runs: yn)")
args = ap.parse_args()
JUDGES, FMTS, KS, NB = args.judges.split(","), args.fmts.split(","), [1, 2, 4, 8, 16], 1000
SEL = ("qwen", "yn")

def accept(p, fmt):
    p = np.asarray(p, dtype=float)
    if fmt == "score":
        return float((p[:5] * np.arange(1, 6)).sum() / p[:5].sum())
    z = p[5] + p[6]
    return float(p[5 if fmt == "yn" else 6] / z) if z > 0 else np.nan

notes = {r["item"]: r for r in map(json.loads, open(args.notes))}
its = sorted(notes)
n, NN = len(its), len(notes[its[0]]["notes"])
y = np.array([notes[i]["label"] for i in its]); doc = np.array([notes[i]["doc"] for i in its])
pos = {i: k for k, i in enumerate(its)}
S, THR, out = {}, {}, {"n_items": n, "n_notes": NN}
print(f"N1: {n} test summaries ({(y == 0).sum()} wrong, {(y == 1).sum()} correct), {NN} notes each, {len(set(doc))} documents")

print("\n== 1. clean scores of this run vs the earlier run")
for j in JUDGES:
    rows = [json.loads(l) for l in open(f"results/{j}_{args.tag}.jsonl")]
    old = {(r["item"], r["fmt"]): r for r in map(json.loads, open(f"results/{j}_full.jsonl")) if r["cond"] == "clean"}
    for fmt in FMTS:
        R = [r for r in rows if r["fmt"] == fmt]
        A = {"clean": np.full(n, np.nan), "claim_e": np.full(n, np.nan), "free": np.full((NN, n), np.nan)}
        cal = []
        for r in R:
            s = accept(r["p"], fmt)
            if r["cond"] == "clean" and r["part"] == "calib" and r["label"] == 1:
                cal.append(s)
            if r["item"] not in pos:
                continue
            if r["cond"].startswith("free"):
                A["free"][int(r["cond"][4:]), pos[r["item"]]] = s
            elif r["cond"] in A:
                A[r["cond"]][pos[r["item"]]] = s
        THR[j, fmt] = float(np.nanquantile(cal, 0.10))
        S[j, fmt] = A
        d = [abs(accept(r["p"], fmt) - accept(old[r["item"], fmt]["p"], fmt)) for r in R if r["cond"] == "clean"]
        nan = {c: int(np.isnan(v).sum()) for c, v in A.items()}
        print(f"  {j:8s} {fmt:6s} threshold {THR[j, fmt]:.3f} from {len(cal)} items; |clean - earlier| max {max(d):.4f}, "
              f"mean {np.mean(d):.5f}; missing scores {nan}")

ti = np.arange(n); tdocs = np.unique(doc); rng = np.random.default_rng(0)
BOOT = [np.concatenate([ti[doc == d] for d in rng.choice(tdocs, len(tdocs))]) for _ in range(NB)]
def ci(f):
    bs = np.array([f(ix) for ix in BOOT]); return [float(f(ti)), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
fci = lambda v: f"{v[0]:+5.1f} [{v[1]:+5.1f}, {v[2]:+5.1f}]"

def kept(sel, k):  # (groups, items) index of the kept note per group
    F = S[sel]["free"]
    return np.stack([g * k + np.argmax(np.nan_to_num(F[g * k:(g + 1) * k], nan=-1), axis=0) for g in range(NN // k)])

def rates(sel, ev, k, ix):  # miss %, FA % of the kept notes under the evaluator, items ix
    K, F, t = kept(sel, k), S[ev]["free"], THR[ev]
    s = F[K[:, ix], ix]
    return 100 * np.nanmean(s[:, y[ix] == 0] >= t), 100 * np.nanmean(s[:, y[ix] == 1] < t)

base = lambda ev, c, ix: (100 * np.nanmean(S[ev][c][ix][y[ix] == 0] >= THR[ev]), 100 * np.nanmean(S[ev][c][ix][y[ix] == 1] < THR[ev]))

def table(sel, evs, title):
    print(f"\n== {title}")
    res = {}
    for ev in evs:
        tag = "in-sample" if ev == sel else ("same judge" if ev[0] == sel[0] else "held-out judge")
        r = {"clean": base(ev, "clean", ti), "claim_e": base(ev, "claim_e", ti)}
        r |= {f"best_of_{k}": rates(sel, ev, k, ti) for k in KS}
        d16 = ci(lambda ix: rates(sel, ev, 16, ix)[0] - base(ev, "clean", ix)[0])
        d1 = ci(lambda ix: rates(sel, ev, 1, ix)[0] - base(ev, "clean", ix)[0])
        dk = ci(lambda ix: rates(sel, ev, 16, ix)[0] - rates(sel, ev, 1, ix)[0])
        dce = ci(lambda ix: rates(sel, ev, 16, ix)[0] - base(ev, "claim_e", ix)[0])
        res[f"{ev[0]}/{ev[1]}"] = {"rates": r, "d_miss_16_clean": d16, "d_miss_1_clean": d1, "d_miss_16_1": dk, "d_miss_16_claim_e": dce}
        print(f"  evaluator {ev[0]} {ev[1]} ({tag}), threshold {THR[ev]:.3f}")
        print("    miss %: " + "  ".join(f"{c} {v[0]:.1f}" for c, v in r.items()))
        print("    FA %:   " + "  ".join(f"{c} {v[1]:.1f}" for c, v in r.items()))
        print(f"    d miss: best-of-1 vs clean {fci(d1)}; best-of-16 vs clean {fci(d16)}; best-of-16 vs best-of-1 {fci(dk)}; "
              f"best-of-16 vs claim_e {fci(dce)}")
    return res

out["selector_qwen_yn"] = table(SEL, [(j, f) for j in JUDGES for f in FMTS], "2. selector Qwen yn; miss and FA by k")
if "nemotron" in JUDGES:  # reverse direction (added after the first readout): Nemotron yn picks, every cell scores
    out["selector_nemotron_yn"] = table(("nemotron", "yn"), [(j, f) for j in JUDGES for f in FMTS],
                                        "2r. selector Nemotron yn (reverse direction); miss and FA by k")
out["in_sample"] = {}
print("\n== 3. in-sample: each cell selects for itself (miss %: clean, claim_e, best-of-1, best-of-16; FA % best-of-16)")
for c in [(j, f) for j in JUDGES for f in FMTS]:
    r = {"clean": base(c, "clean", ti), "claim_e": base(c, "claim_e", ti), "b1": rates(c, c, 1, ti), "b16": rates(c, c, 16, ti)}
    out["in_sample"][f"{c[0]}/{c[1]}"] = r
    print(f"  {c[0]:8s} {c[1]:6s} miss clean {r['clean'][0]:5.1f}  claim_e {r['claim_e'][0]:5.1f}  best-of-1 {r['b1'][0]:5.1f}  "
          f"best-of-16 {r['b16'][0]:5.1f} | FA clean {r['clean'][1]:5.1f}  best-of-1 {r['b1'][1]:5.1f}  best-of-16 {r['b16'][1]:5.1f}")

print("\n== 4. note content (labels by the writer model; A = claims checked or accurate, B = flags an error)")
C = np.array([[c if c is not None else [np.nan, np.nan] for c in notes[i]["cls"]] for i in its], dtype=float)  # (n, NN, 2)
print(f"  unparsed labels: {int(np.isnan(C[..., 0]).sum())} of {C[..., 0].size}; "
      f"notes cut at max tokens: {sum(f == 'length' for i in its for f in notes[i]['finish'])}")
VER = re.compile(r"\b(verif\w*|double[- ]check\w*|cross[- ]check\w*|cross[- ]referenc\w*|checked|confirm\w*|fact[- ]check\w*|ensured)\b",
                 re.I)
R = np.array([[bool(VER.search(t)) for t in notes[i]["notes"]] for i in its])
out["content"] = {}
for sel in [SEL, ("nemotron", "yn")] if "nemotron" in JUDGES else [SEL]:
    K16 = kept(sel, 16)[0]; Kmin = np.argmin(np.nan_to_num(S[sel]["free"], nan=2), axis=0)
    for lab, name in [(0, "wrong"), (1, "correct")]:
        m = y == lab
        row = {}
        for q, qi in [("A", 0), ("B", 1)]:
            allv = 100 * np.nanmean(C[m][..., qi]); top = 100 * np.nanmean(C[m, K16[m], qi]); low = 100 * np.nanmean(C[m, Kmin[m], qi])
            row[q] = [allv, top, low]
        row["regex_checked"] = [100 * R[m].mean(), 100 * R[m, K16[m]].mean(), 100 * R[m, Kmin[m]].mean()]
        out["content"][f"{sel[0]}/{sel[1]}/{name}"] = row
        print(f"  selector {sel[0]} {sel[1]}, {name:7s} summaries: share % among all notes / kept best-of-16 / lowest-scored: "
              + "; ".join(f"{q} {v[0]:.1f} / {v[1]:.1f} / {v[2]:.1f}" for q, v in row.items()))
    dA = ci(lambda ix: 100 * (np.nanmean(C[ix, K16[ix], 0][y[ix] == 0]) - np.nanmean(C[ix][y[ix] == 0][..., 0])))
    print(f"    wrong summaries, A share kept minus all notes: {fci(dA)}")
    out["content"][f"{sel[0]}/{sel[1]}/dA_wrong"] = dA
    nflag = np.nansum(C[..., 1], axis=1)
    # wording patterns, found by reading the flipped items (exploratory): the note names an error but vouches for the rest,
    # or it excuses the error
    REST = re.compile(r"\b(all|every|each) other\b|\bthe rest of (the|my) summary\b|\b(aside|apart) from th(is|at)\b|\botherwise\b"
                      r"|\bremaining (details|statements|points)\b", re.I)
    EXC = re.compile(r"\binference\b|\bintended\b|\btypo\b|\bminor\b|\binterpret|\bgeneral sentiment\b|\bslight\b|\bsimplif", re.I)
    FIX = re.compile(r"\b(have|has|had)( now| since| already)? (been )?(corrected|fixed|revised|updated|amended|rectified|addressed)\b"
                     r"|\bI (corrected|fixed|revised|updated|amended|rectified)\b|\b(corrected|fixed|revised|updated) (version|summary)\b"
                     r"|\bhas been (corrected|fixed|revised|updated)\b", re.I)  # claims a fix that is not in the shown summary
    for pname, rx in [("rest_supported", REST), ("excuse", EXC), ("fix_claim", FIX)]:
        M = np.array([[bool(rx.search(t)) for t in notes[i]["notes"]] for i in its])
        m = (y == 0) & (nflag > 0)
        d = ci(lambda ix: 100 * (M[ix, K16[ix]][m[ix]].mean() - M[ix][m[ix]].mean()))
        v = [100 * M[m].mean(), 100 * M[m, K16[m]].mean(), 100 * M[m, Kmin[m]].mean()]
        out["content"][f"{sel[0]}/{sel[1]}/{pname}_wrong_flagged"] = {"all_kept_lowest": v, "d_kept_all": d}
        print(f"    wrong summaries with 1+ flagging notes ({m.sum()}), pattern {pname}: all {v[0]:.1f}%, kept {v[1]:.1f}%, "
              f"lowest {v[2]:.1f}%; kept minus all {fci(d)}")
    for lab, name in [(0, "wrong"), (1, "correct")]:  # summaries where some but not all notes flag a problem: is the kept note one of them?
        m = (y == lab) & (nflag > 0) & (nflag < NN)
        if m.any():
            base_b = np.nanmean(C[m][..., 1], axis=1)  # share of flagging notes per summary
            kept_b = C[m, K16[m], 1]
            print(f"    {name} summaries with 1-{NN - 1} flagging notes: {m.sum()} of {(y == lab).sum()}; kept note flags in "
                  f"{100 * np.nanmean(kept_b):.1f}%; expected under random choice {100 * base_b.mean():.1f}%")
            out["content"][f"{sel[0]}/{sel[1]}/flag_filter_{name}"] = [int(m.sum()), float(100 * np.nanmean(kept_b)),
                                                                      float(100 * base_b.mean())]

print("\n== 4b. wrong summaries grouped by how many of their notes flag a problem (selector Qwen yn): miss %")
nflag = np.nansum(C[..., 1], axis=1)
groups = {"no flag": (y == 0) & (nflag == 0), f"1-{NN - 1} flag": (y == 0) & (nflag > 0) & (nflag < NN), f"all {NN} flag": (y == 0) & (nflag == NN)}
out["by_flag_group"] = {}
for ev in [(j, f) for j in JUDGES for f in ["yn", "yn_rev"] if f in FMTS]:
    t = THR[ev]; F = S[ev]["free"]; K16 = kept(SEL, 16)[0]
    parts = []
    for g, m in groups.items():
        v = {"n": int(m.sum()), "clean": 100 * np.nanmean(S[ev]["clean"][m] >= t), "claim_e": 100 * np.nanmean(S[ev]["claim_e"][m] >= t),
             "best_of_1": 100 * np.nanmean(F[:, m] >= t), "best_of_16": 100 * np.nanmean(F[K16[m], np.where(m)[0]] >= t)}
        out["by_flag_group"][f"{ev[0]}/{ev[1]}/{g}"] = v
        parts.append(f"{g} (n={v['n']}): clean {v['clean']:.1f}, claim_e {v['claim_e']:.1f}, best-of-1 {v['best_of_1']:.1f}, "
                     f"best-of-16 {v['best_of_16']:.1f}")
    print(f"  {ev[0]:8s} {ev[1]:6s} " + " | ".join(parts))

print("\n== 5. accept score minus the summary's mean over its notes, by label class (wrong summaries; A only, B only, both, neither)")
cls = np.where(C[..., 0] == 1, np.where(C[..., 1] == 1, 2, 0), np.where(C[..., 1] == 1, 1, 3))  # 0 A only, 1 B only, 2 both, 3 neither
out["by_class"] = {}
for c in [(j, f) for j in JUDGES for f in FMTS]:
    F = S[c]["free"].T  # (n, NN)
    dev = F - np.nanmean(F, axis=1, keepdims=True)
    w = y == 0
    v = {nm: (float(np.nanmean(dev[w][cls[w] == k])) if (cls[w] == k).any() else np.nan, int((cls[w] == k).sum()))
         for k, nm in enumerate(["A_only", "B_only", "both", "neither"])}
    out["by_class"][f"{c[0]}/{c[1]}"] = v
    print(f"  {c[0]:8s} {c[1]:6s} " + "  ".join(f"{nm} {m:+.3f} (n={k})" for nm, (m, k) in v.items()))

print("\n== 6. selection within a document (selector Qwen yn picks each summary's note; evaluator as named)")
docs = {}
for k, i in enumerate(its):
    docs.setdefault(doc[k], {0: [], 1: []})[y[k]].append(k)
D = [d for d, v in docs.items() if v[0] and v[1]]
K16 = kept(SEL, 16)[0]
out["selection"] = {}
for ev in [(j, f) for j in JUDGES for f in FMTS]:
    F, cl, ce = S[ev]["free"], S[ev]["clean"], S[ev]["claim_e"]
    scen = {"clean": (lambda a: cl[a], lambda b: cl[b]),
            "claim_e on wrong": (lambda a: ce[a], lambda b: cl[b]),
            "best16 on wrong": (lambda a: F[K16[a], a], lambda b: cl[b]),
            "best16 on both": (lambda a: F[K16[a], a], lambda b: F[K16[b], b]),
            "random note on both": None}
    def top(sw, sc, ds):
        t = []
        for d in ds:
            sc_ = [(sw(a), 0) for a in docs[d][0]] + [(sc(b), 1) for b in docs[d][1]]
            m = max(s for s, _ in sc_); best = [lab for s, lab in sc_ if s == m]
            t.append(best.count(0) / len(best))
        return 100 * np.mean(t)
    def topf(name, ds):
        if name == "random note on both":
            return np.mean([top(lambda a: F[j, a], lambda b: F[j, b], ds) for j in range(NN)])
        return top(*scen[name], ds)
    rngb = np.random.default_rng(0); B = [list(rngb.choice(D, len(D))) for _ in range(NB)]
    def cid(name):
        f = lambda ds: topf(name, ds) - topf("clean", ds)
        bs = np.percentile([f(b) for b in B], [2.5, 97.5]); return [float(f(D)), float(bs[0]), float(bs[1])]
    r = {nm: float(topf(nm, D)) for nm in scen}
    dd = {nm: cid(nm) for nm in ["claim_e on wrong", "best16 on wrong", "best16 on both"]}
    out["selection"][f"{ev[0]}/{ev[1]}"] = {"top_pick_wrong": r, "d_vs_clean": dd}
    print(f"  {ev[0]:8s} {ev[1]:6s} docs {len(D)}; top pick wrong %: " + "; ".join(f"{k} {v:.1f}" for k, v in r.items()))
    print("      d vs clean: " + "; ".join(f"{k} {fci(v)}" for k, v in dd.items()))

json.dump(out, open(args.json, "w"), indent=1, default=float)
print("\nwrote", args.json)
