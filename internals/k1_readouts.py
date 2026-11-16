"""Readouts G1 and R1-R6 of the pre-registered judge-internals plan (section 4). Runs locally on copies of the result files. numpy and matplotlib only.

Inputs in --dir (default results/ next to this file):
  <judge>_e0_full.jsonl            vllm scores of all sprint rows (step T1)
  <judge>_lens_<fmt>.jsonl         k1_internals.py lens stage (T2)
  <judge>_probe_<fmt>.json         k1_probe.py (T3)
  <judge>_patch_<fmt>.jsonl        k1_internals.py patch stage (T4)
  <judge>_ablate_<fmt>.jsonl       k1_internals.py ablate stage (T5)
Missing files are skipped.

Definitions (fixed in the plan unless marked "choice of 2026-11-14"):
  accept score: E0 as in sprint_check.py (E[s] for score, p(Yes) share for yn); threshold = 10% quantile of clean
    accept scores on calibration consistent items; miss = inconsistent item accepted.
  G1: claim_e - claim_o miss difference on test items (bootstrap over test documents) and the number of flipped items
    (inconsistent, accepted under claim_e, rejected under claim_o; all 166 inconsistent items, and test only).
  Target t: yn L = log p(Yes) - log p(No); score E[s].
  R1: first layer where mean lens t(claim_e) - t(claim_o) on inconsistent test items reaches half of the final-layer
    difference (same sign). "late" if in the last third of the layers.
  R4: restoration (t_patched - t_e) / (t_o - t_e), median over inconsistent test items with |t_o - t_e| > 0.5 (yn).
    For score the threshold is 0.25 on the 1-5 scale (choice of 2026-11-14, made before any patch result).
    "mediated" if P2 (summary tokens) >= 0.5; "direct" if P2 < 0.2 and P1 reaches 0.8 at some layer; else "mixed".
    Extra groups (choice of 2026-11-14): docsumm (document + summary), e (sentence e only = P3), q (tokens after the
    summary except the answer position). P4 = the same groups copied from claim_e into claim_o; its "transfer" is
    (t_patched - t_o) / (t_e - t_o).
  R5: share of the claim_e effect lost when one note unit is removed: (t_e - t_drop) / (t_e - t_clean), median over
    items with |t_e - t_clean| above the R4 threshold, per unit role.
  R6: weight of sentence e under condition c: w_c = t(sentence e removed, c) - t(c). R6 = mean(w_claim_e - w_claim_o)
    with a bootstrap over test documents. Interval below 0: the note lowers the weight of the evidence.
Post-hoc readouts (added 2026-11-14 23:30 UTC, after the main results; not in the plan):
  knock: attention knockout (stage_knock), restoration as in R4.
  patch3: pointer_e versus claim_e (stage_patch3). into_e: (t_patched - t_e) / (t_p - t_e) with source pointer_e;
    into_p: (t_patched - t_p) / (t_e - t_p) with source claim_e. Items with |t_e - t_p| above the R4 threshold.
  probe_threat: miss rate of each signal (output accept score from E0, and the out-of-fold probe scores at the 3 sites)
    at a threshold fixed on clean scores: the 10% quantile of clean consistent test items, per signal. Same rule as
    the internal review of the probes (point 1).
Usage: python internals/k1_readouts.py [--dir DIR] [--judges qwen9,gemma12] [--figs DIR] [--json out.json]"""
import argparse, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NBOOT, SEED = 1000, 0
THR = {"yn": 0.5, "score": 0.25}


def jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else None


def boot(items_doc, f, seed=SEED):
    """items_doc: array of documents per unit; f(index array) -> value. Returns point, lo, hi."""
    docs = np.unique(items_doc)
    by = {d: np.where(items_doc == d)[0] for d in docs}
    rng = np.random.default_rng(seed)
    bs = [f(np.concatenate([by[d] for d in rng.choice(docs, len(docs))])) for _ in range(NBOOT)]
    return [float(f(np.arange(len(items_doc)))), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]


def accept(p, fmt):
    p = np.asarray(p, float)
    if fmt == "score":
        return (p[:5] * np.arange(1, 6)).sum() / p[:5].sum()
    return p[5] / (p[5] + p[6])


def g1(rows, fmt):
    R = [r for r in rows if r["fmt"] == fmt]
    S = {}
    meta = {}
    for r in R:
        S[(r["item"], r["cond"])] = accept(r["p"], fmt)
        meta[r["item"]] = r
    its = sorted(meta)
    y = np.array([meta[i]["label"] for i in its]); part = np.array([meta[i]["part"] for i in its])
    doc = np.array([meta[i]["doc"] for i in its])
    s = lambda c: np.array([S[(i, c)] for i in its])
    thr = np.quantile(s("clean")[(part == "calib") & (y == 1)], 0.10)
    acc = {c: s(c) >= thr for c in ["clean", "pointer_e", "claim_o", "claim_e"]}
    inc = y == 0
    flip = inc & acc["claim_e"] & ~acc["claim_o"]
    te = np.where((part == "test") & inc)[0]
    d = boot(doc[te], lambda ix: 100 * (acc["claim_e"][te][ix].mean() - acc["claim_o"][te][ix].mean()))
    dc = boot(doc[te], lambda ix: 100 * (acc["claim_e"][te][ix].mean() - acc["clean"][te][ix].mean()))
    return {"threshold": float(thr), "miss": {c: float(100 * acc[c][te].mean()) for c in acc},
            "d_miss_e_minus_o": d, "d_miss_e_minus_clean": dc, "flipped_all": int(flip.sum()),
            "flipped_test": int(flip[part == "test"].sum()), "flipped_items": [its[i] for i in np.where(flip)[0]],
            "passes": bool(d[1] > 0 and flip.sum() >= 25)}


def lens_readout(rows, fmt, nL):
    inc = [r for r in rows if r["label"] == 0 and r["part"] == "test"]
    items = sorted({r["item"] for r in inc})
    by = {(r["item"], r["cond"]): r for r in inc}
    doc = np.array([by[(i, "clean")]["doc"] for i in items])
    T = {c: np.array([by[(i, c)]["lens_t"] for i in items]) for c in ["clean", "neutral", "pointer_e", "claim_o", "claim_e"]}
    final = {c: np.array([by[(i, c)]["t"] for i in items]) for c in T}

    def crit(ix):
        d = T["claim_e"][ix].mean(0) - T["claim_o"][ix].mean(0)
        half = 0.5 * d[-1]
        hit = np.where(np.sign(half) * d >= abs(half))[0] if half != 0 else []
        return float(hit[0]) if len(hit) else np.nan
    c = boot(doc, crit)
    return {"critical_layer": c, "late": bool(c[0] >= 2 * nL / 3), "n_layers": nL, "n_items": len(items),
            "mean_lens": {k: v.mean(0).round(4).tolist() for k, v in T.items()},
            "mean_final": {k: float(v.mean()) for k, v in final.items()},
            "final_d_e_minus_o": boot(doc, lambda ix: final["claim_e"][ix].mean() - final["claim_o"][ix].mean())}


def patch_readout(rows, fmt):
    rows = [r for r in rows if r["part"] == "test"]
    thr = THR[fmt]
    keep = [r for r in rows if abs(r["t_o"] - r["t_e"]) > thr]
    doc = np.array([r["doc"] for r in keep])
    te, to = np.array([r["t_e"] for r in keep]), np.array([r["t_o"] for r in keep])
    gap = to - te
    out = {"n_items": len(rows), "n_kept": len(keep), "threshold": thr}
    P1 = np.array([r["P1"] for r in keep])
    rest = lambda v: (v - te) / gap
    out["P1"] = [boot(doc, lambda ix, l=l: np.median(rest(P1[:, l])[ix])) for l in range(P1.shape[1])]
    for g in ["summ", "docsumm", "e", "q"]:
        v = np.array([r["into_e"][g] for r in keep])
        out[f"into_e_{g}"] = boot(doc, lambda ix: np.median(rest(v)[ix]))
        w = np.array([r["into_o"][g] for r in keep])
        out[f"into_o_{g}"] = boot(doc, lambda ix: np.median(((w - to) / (te - to))[ix]))
    p2 = out["into_e_summ"][0]
    p1max = max(x[0] for x in out["P1"])
    out["R4"] = "mediated" if p2 >= 0.5 else "direct" if (p2 < 0.2 and p1max >= 0.8) else "mixed"
    out["P1_first_layer_below_0.5"] = next((l for l, x in enumerate(out["P1"]) if x[0] < 0.5), None)
    return out


def patch2_readout(rows, fmt):
    """Supplementary patching: sentence e at one layer; summary minus e and sentence ko at all layers."""
    rows = [r for r in rows if r["part"] == "test"]
    keep = [r for r in rows if abs(r["t_o"] - r["t_e"]) > THR[fmt]]
    doc = np.array([r["doc"] for r in keep])
    te, to = np.array([r["t_e"] for r in keep]), np.array([r["t_o"] for r in keep])
    EL = np.array([r["e_layer"] for r in keep])
    out = {"n_kept": len(keep),
           "e_layer": [boot(doc, lambda ix, l=l: np.median(((EL[:, l] - te) / (to - te))[ix])) for l in range(EL.shape[1])]}
    for g in ["not_e", "ko"]:
        v = np.array([r["into_e"][g] for r in keep]); w = np.array([r["into_o"][g] for r in keep])
        out[f"into_e_{g}"] = boot(doc, lambda ix: np.median(((v - te) / (to - te))[ix]))
        out[f"into_o_{g}"] = boot(doc, lambda ix: np.median(((w - to) / (te - to))[ix]))
    return out


def knock_readout(rows, fmt):
    """Attention knockout: restoration (t - t_e) / (t_o - t_e) for claim_e variants; for claim_o variants the change
    (t - t_o) / (t_e - t_o). Median over test items with a gap above the R4 threshold."""
    rows = [r for r in rows if r["part"] == "test"]
    keep = [r for r in rows if abs(r["t_o"] - r["t_e"]) > THR[fmt]]
    doc = np.array([r["doc"] for r in keep])
    te, to = np.array([r["t_e"] for r in keep]), np.array([r["t_o"] for r in keep])
    out = {"n_kept": len(keep)}
    for nm in keep[0]["knock"]:
        v = np.array([r["knock"][nm] for r in keep])
        f = (lambda ix: np.median(((v - to) / (te - to))[ix])) if nm.startswith("o_") else \
            (lambda ix: np.median(((v - te) / (to - te))[ix]))
        out[nm] = boot(doc, f)
    out["none_max_abs_dt"] = float(np.max(np.abs(np.array([r["knock"]["none"] for r in keep]) - te)))
    return out


def patch3_readout(rows, fmt):
    rows = [r for r in rows if r["part"] == "test"]
    keep = [r for r in rows if abs(r["t_p"] - r["t_e"]) > THR[fmt]]
    doc = np.array([r["doc"] for r in keep])
    te, tp = np.array([r["t_e"] for r in keep]), np.array([r["t_p"] for r in keep])
    out = {"n_items": len(rows), "n_kept": len(keep), "median_t_e_minus_t_p": float(np.median(te - tp))}
    for g in ["e", "summ", "docsumm"]:
        v = np.array([r["into_e"][g] for r in keep])
        out[f"into_e_{g}"] = boot(doc, lambda ix: np.median(((v - te) / (tp - te))[ix]))
    for g in ["e", "summ"]:
        v = np.array([r["into_p"][g] for r in keep])
        out[f"into_p_{g}"] = boot(doc, lambda ix: np.median(((v - tp) / (te - tp))[ix]))
    return out


def probe_threat_readout(oof, e0, fmt):
    conds = ["clean", "neutral", "pointer_e", "claim_o", "claim_e"]
    its = sorted({r["item"] for r in oof}); pos = {i: k for k, i in enumerate(its)}
    S = {}
    for r in oof:
        for s in ["e", "summary_last", "answer"]:
            S.setdefault((s, r["cond"]), np.full(len(its), np.nan))[pos[r["item"]]] = r[s]
    for r in e0:
        if r["fmt"] == fmt and r["cond"] in conds and r["item"] in pos:
            S.setdefault(("output", r["cond"]), np.full(len(its), np.nan))[pos[r["item"]]] = accept(r["p"], fmt)
    lab = {r["item"]: r["label"] for r in oof}; dc = {r["item"]: r["doc"] for r in oof}
    y = np.array([lab[i] for i in its]); doc = np.array([dc[i] for i in its])
    sig = ["output", "e", "summary_last", "answer"]
    thr = {s: np.quantile(S[(s, "clean")][y == 1], 0.10) for s in sig}
    mi = lambda s, c, ix: 100 * np.mean(S[(s, c)][ix][y[ix] == 0] >= thr[s])
    out = {"n_items": len(its), "n_inconsistent": int((y == 0).sum())}
    for s in sig:
        out[s] = {"miss": {c: mi(s, c, np.arange(len(its))) for c in conds},
                  "e_minus_clean": boot(doc, lambda ix: mi(s, "claim_e", ix) - mi(s, "clean", ix)),
                  "e_minus_o": boot(doc, lambda ix: mi(s, "claim_e", ix) - mi(s, "claim_o", ix))}
        if s != "output":
            out[s]["e_minus_o_minus_output"] = boot(doc, lambda ix: (mi(s, "claim_e", ix) - mi(s, "claim_o", ix))
                                                    - (mi("output", "claim_e", ix) - mi("output", "claim_o", ix)))
    return out


def ablate_readout(rows, fmt):
    rows = [r for r in rows if r["part"] == "test"]
    thr = THR[fmt]
    out = {"n_items": len(rows)}
    roles = {}
    for r in rows:
        eff = r["base"]["claim_e"] - r["base"]["clean"]
        if abs(eff) <= thr:
            continue
        for role, t in zip(r["roles"], r["drop_unit"]):
            roles.setdefault(f"w{r['wording']}:{role}", []).append((r["doc"], (r["base"]["claim_e"] - t) / eff))
    out["R5_share_lost"] = {}
    for k, v in sorted(roles.items()):
        doc = np.array([d for d, _ in v]); x = np.array([s for _, s in v])
        out["R5_share_lost"][k] = boot(doc, lambda ix: np.median(x[ix])) + [len(v)]
    doc = np.array([r["doc"] for r in rows])
    w = {c: np.array([r["drop_e"][c] - r["base"][c] for r in rows]) for c in ["clean", "claim_o", "claim_e"]}
    out["weight_mean"] = {c: boot(doc, lambda ix: w[c][ix].mean()) for c in w}
    out["R6_e_minus_o"] = boot(doc, lambda ix: (w["claim_e"] - w["claim_o"])[ix].mean())
    out["R6_e_minus_clean"] = boot(doc, lambda ix: (w["claim_e"] - w["clean"])[ix].mean())
    lo, hi = out["R6_e_minus_o"][1:]
    out["R6"] = "note lowers the weight of sentence e" if hi < 0 else \
        "note raises the weight of sentence e" if lo > 0 else "no difference"
    return out


def figs(res, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(figdir, exist_ok=True)
    for judge, R in res.items():
        for fmt, X in R.items():
            if "lens" not in X:
                continue
            fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
            for c, col in [("clean", "#555"), ("claim_o", "#1f77b4"), ("claim_e", "#d62728"), ("pointer_e", "#999")]:
                ax[0].plot(X["lens"]["mean_lens"][c], label=c, color=col)
            ax[0].set_title(f"{judge} {fmt}: logit lens, inconsistent test items"); ax[0].set_xlabel("layer")
            ax[0].set_ylabel("L" if fmt == "yn" else "E[s]"); ax[0].legend(fontsize=7)
            if "probe" in X:
                for s, col in [("answer", "#2ca02c"), ("e", "#9467bd"), ("summary_last", "#8c564b")]:
                    pl = X["probe"]["sites"][s]["per_layer"]
                    ax[1].plot([p["claim_o"] for p in pl], color=col, label=f"{s} claim_o")
                    ax[1].plot([p["claim_e"] for p in pl], color=col, ls="--", label=f"{s} claim_e")
                ax[1].set_title("probe AUROC (out of fold, test)"); ax[1].set_xlabel("layer"); ax[1].legend(fontsize=6)
            if "patch" in X:
                P1 = np.array(X["patch"]["P1"])
                ax[2].plot(P1[:, 0], color="k", label="P1 answer position")
                ax[2].fill_between(range(len(P1)), P1[:, 1], P1[:, 2], color="k", alpha=.15)
                for g, col in [("summ", "#1f77b4"), ("docsumm", "#ff7f0e"), ("e", "#9467bd"), ("q", "#2ca02c")]:
                    ax[2].axhline(X["patch"][f"into_e_{g}"][0], color=col, ls=":", label=f"{g}, all layers")
                if "patch2" in X:
                    E = np.array(X["patch2"]["e_layer"])
                    ax[2].plot(E[:, 0], color="#9467bd", label="sentence e, one layer")
                    ax[2].fill_between(range(len(E)), E[:, 1], E[:, 2], color="#9467bd", alpha=.15)
                ax[2].set_ylim(-0.3, 1.3); ax[2].set_title("restoration, claim_o -> claim_e"); ax[2].set_xlabel("layer")
                ax[2].legend(fontsize=6)
            fig.tight_layout()
            for ext in ["png", "pdf"]:
                fig.savefig(f"{figdir}/internals_{judge}_{fmt}.{ext}", dpi=150)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=f"{HERE}/results"); ap.add_argument("--judges", default="qwen9,gemma12")
    ap.add_argument("--figs", default=""); ap.add_argument("--json", default="")
    args = ap.parse_args()
    res = {}
    f3 = lambda v, d=3: f"{v[0]:+.{d}f} [{v[1]:+.{d}f}, {v[2]:+.{d}f}]"
    for judge in args.judges.split(","):
        e0 = jl(f"{args.dir}/{judge}_e0_full.jsonl")
        for fmt in ["yn", "score"]:
            X = res.setdefault(judge, {}).setdefault(fmt, {})
            if e0:
                X["g1"] = g = g1(e0, fmt)
                print(f"\n== {judge} {fmt}  G1: miss clean {g['miss']['clean']:.1f}, claim_o {g['miss']['claim_o']:.1f}, "
                      f"claim_e {g['miss']['claim_e']:.1f}; claim_e - claim_o {f3(g['d_miss_e_minus_o'], 1)} points; "
                      f"claim_e - clean {f3(g['d_miss_e_minus_clean'], 1)}; flipped {g['flipped_all']} "
                      f"(test {g['flipped_test']}); passes={g['passes']}")
            lens = jl(f"{args.dir}/{judge}_lens_{fmt}.jsonl")
            if lens:
                nL = len(lens[0]["lens_t"])
                X["lens"] = L = lens_readout(lens, fmt, nL)
                print(f"   R1 critical layer {f3(L['critical_layer'], 1)} of {nL} -> late={L['late']}; "
                      f"final t: " + ", ".join(f"{k} {v:+.3f}" for k, v in L["mean_final"].items()) +
                      f"; claim_e - claim_o {f3(L['final_d_e_minus_o'])}")
            pr = f"{args.dir}/{judge}_probe_{fmt}.json"
            if os.path.exists(pr):
                X["probe"] = P = json.load(open(pr))
                for s, tag in [("e", "R2 evidence site"), ("answer", "R3 decision site"), ("summary_last", "summary end")]:
                    q = P["sites"][s]
                    print(f"   {tag}: layer {q['layer']}, AUROC claim_o {q['auroc']['claim_o']:.3f}, claim_e "
                          f"{q['auroc']['claim_e']:.3f}, clean {q['auroc']['clean']:.3f}; o - e {f3(q['diff_o_minus_e'])}"
                          f" -> {q['label']}")
            pa = jl(f"{args.dir}/{judge}_patch_{fmt}.jsonl")
            if pa:
                X["patch"] = Q = patch_readout(pa, fmt)
                print(f"   R4 ({Q['n_kept']} of {Q['n_items']} test items with gap > {Q['threshold']}): "
                      + ", ".join(f"{g} {f3(Q['into_e_' + g], 2)}" for g in ["summ", "docsumm", "e", "q"])
                      + f"; P1 max {max(x[0] for x in Q['P1']):.2f}, first layer below 0.5: "
                        f"{Q['P1_first_layer_below_0.5']} -> {Q['R4']}")
                print("      P4 transfer claim_e -> claim_o: " +
                      ", ".join(f"{g} {f3(Q['into_o_' + g], 2)}" for g in ["summ", "docsumm", "e", "q"]))
                print("      P1 by layer: " + " ".join(f"{x[0]:.2f}" for x in Q["P1"]))
            p2 = jl(f"{args.dir}/{judge}_patch2_{fmt}.jsonl")
            if p2:
                X["patch2"] = Q2 = patch2_readout(p2, fmt)
                print(f"   patch2 ({Q2['n_kept']} items): into claim_e: summary minus e {f3(Q2['into_e_not_e'], 2)}, "
                      f"sentence ko {f3(Q2['into_e_ko'], 2)}; transfer into claim_o: summary minus e "
                      f"{f3(Q2['into_o_not_e'], 2)}, sentence ko {f3(Q2['into_o_ko'], 2)}")
                print("      sentence e, one layer: " + " ".join(f"{x[0]:.2f}" for x in Q2["e_layer"]))
            kn = jl(f"{args.dir}/{judge}_knock_{fmt}.jsonl")
            if kn:
                X["knock"] = KN = knock_readout(kn, fmt)
                print(f"   knockout ({KN['n_kept']} items; 'none' max |t - t_e| {KN['none_max_abs_dt']:.3f}): " +
                      ", ".join(f"{k} {f3(v, 2)}" for k, v in KN.items() if isinstance(v, list)))
            p3 = jl(f"{args.dir}/{judge}_patch3_{fmt}.jsonl")
            if p3:
                X["patch3"] = Q3 = patch3_readout(p3, fmt)
                print(f"   patch3 ({Q3['n_kept']} of {Q3['n_items']} items with |t_e - t_p| > {THR[fmt]}; median t_e - t_p "
                      f"{Q3['median_t_e_minus_t_p']:+.2f}): pointer_e -> claim_e " +
                      ", ".join(f"{g} {f3(Q3['into_e_' + g], 2)}" for g in ["e", "summ", "docsumm"]) +
                      "; claim_e -> pointer_e " + ", ".join(f"{g} {f3(Q3['into_p_' + g], 2)}" for g in ["e", "summ"]))
            oof = jl(f"{args.dir}/{judge}_probe_{fmt}_oof.jsonl")
            if oof and e0:
                X["probe_threat"] = PT = probe_threat_readout(oof, e0, fmt)
                for sg in ["output", "e", "summary_last", "answer"]:
                    q = PT[sg]
                    print(f"   fixed-threshold miss, {sg}: " + " ".join(f"{c} {v:.1f}" for c, v in q["miss"].items()) +
                          f"; claim_e - clean {f3(q['e_minus_clean'], 1)}; claim_e - claim_o {f3(q['e_minus_o'], 1)}" +
                          (f"; minus output {f3(q['e_minus_o_minus_output'], 1)}" if sg != "output" else ""))
            ab = jl(f"{args.dir}/{judge}_ablate_{fmt}.jsonl")
            if ab:
                X["ablate"] = A = ablate_readout(ab, fmt)
                print(f"   R6 weight of sentence e: " + ", ".join(f"{c} {f3(v)}" for c, v in A["weight_mean"].items())
                      + f"; claim_e - claim_o {f3(A['R6_e_minus_o'])} -> {A['R6']}")
                print("   R5 share of the claim_e effect lost: " + "; ".join(
                    f"{k} {v[0]:.2f} (n={v[3]})" for k, v in A["R5_share_lost"].items()))
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)
    if args.figs:
        figs(res, args.figs)


if __name__ == "__main__":
    main()
