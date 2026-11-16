"""K1-v2R E1b, exploratory (written after the B1-B6 readouts were seen; not part of the fixed readouts).
Why does the every-sentence check lose E1's advantage? Two candidate reasons:
  (a) threshold: the min over 2-4 sentences, calibrated to 10% false alarms, gets a lower threshold than sentence e alone;
  (b) mixing: the note raises the score of sentence e only, so the min mixes a shifted sentence with unshifted ones.
Table 1: miss of the sentence-e score alone (from the E1b rows with k = ke) at the E1 sent threshold and at the E1b
every-sentence threshold, under clean and claim_e (test part, point estimates).
Table 2: mean change in the Yes share (claim_e - clean), for sentence e and for the other sentences, by label.
Usage: python src/e1b_explore.py qwen gemma [--res results]"""
import argparse, json
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("judges", nargs="+"); ap.add_argument("--res", default="results")
args = ap.parse_args()
yes = lambda p: p[5] / (p[5] + p[6])
chk = json.load(open(f"{args.res}/e1b_check.json")); e1c = json.load(open(f"{args.res}/e1_check.json"))
for j in args.judges:
    rows = [json.loads(l) for l in open(f"{args.res}/{j}_e1b.jsonl")]
    t_all, t_e1 = chk[j]["thr"]["all"], e1c[j]["sent|yn"]["thr"]["sent"]
    test = [r for r in rows if r["part"] == "test"]
    E = {(r["item"], r["cond"]): yes(r["p"]) for r in test if r["is_e"]}
    items = {r["item"]: r["label"] for r in test}
    print(f"\n== {j}: thresholds E1 sent {t_e1:.3f}, E1b every-sentence {t_all:.3f}")
    for name, t in [("E1 sent thr", t_e1), ("E1b thr", t_all)]:
        m = {c: 100 * np.mean([E[(i, c)] >= t for i, y in items.items() if y == 0]) for c in ["clean", "claim_e"]}
        fa = {c: 100 * np.mean([E[(i, c)] < t for i, y in items.items() if y == 1]) for c in ["clean", "claim_e"]}
        print(f"   sentence e alone at {name} ({t:.3f}): miss clean {m['clean']:.1f}, claim_e {m['claim_e']:.1f}, "
              f"d {m['claim_e'] - m['clean']:+.1f}; FA clean {fa['clean']:.1f}, claim_e {fa['claim_e']:.1f}")
    S = {(r["item"], r["cond"], r["k"]): (yes(r["p"]), r["is_e"], r["label"]) for r in test}
    for is_e in [True, False]:
        for y in [0, 1]:
            d = [S[(i, "claim_e", k)][0] - a for (i, c, k), (a, e, lab) in S.items()
                 if c == "clean" and e == is_e and lab == y]
            print(f"   Yes share change claim_e - clean, {'sentence e' if is_e else 'other sentences'}, "
                  f"{'consistent' if y else 'inconsistent'}: mean {np.mean(d):+.3f} (n={len(d)})")
