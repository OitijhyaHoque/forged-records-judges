"""K1-v2R sprint scorer on vllm (one A100 80 GB). One prefill and 1 generated token per prompt. No hidden states.
Prompt as in score.py: one user turn with thinking off (enable_thinking=False); the assistant turn starts with
`prefill` ("Score: " or "Answer:"), so the next token is the answer.
Answer token ids are found per format by tokenizing context + answer and checking that the context tokens stay the
same and exactly one token is added. If that fails and the prefill ends with a space, the space moves into the
answer ("Score:" + " 1"). Otherwise the run stops before loading the model.
Probabilities come from the top-K logprobs at the answer position. An answer token outside the top K gets p = 0;
`floor` (the smallest top-K probability) is an upper bound for it.
Output <out>.jsonl: each input row without `prompt`, plus
  p     probabilities of ANS = 1..5, Yes, No (entries of the other format are 0), same layout as score.py
  mass  sum of the format's answer probabilities (a low value means the judge wants another token)
  n_tok prompt length in tokens; top: the 5 most likely tokens; floor
Usage: python src/score_vllm.py --model <path> --data data/sprint.jsonl --out results/<judge> [--sample 120]"""
import argparse, json, math, os, random, sys, time

ANS = ["1", "2", "3", "4", "5", "Yes", "No"]

# Everything runs inside main(): vllm may start its engine process with 'spawn', which re-imports this file.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=0, help="score a seeded random sample of this many rows (smoke run)")
    ap.add_argument("--k", type=int, default=20, help="top-K logprobs per prompt")
    ap.add_argument("--max-len", type=int, default=16384); ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--chunk", type=int, default=1000, help="rows per generate call (progress and partial output)")
    ap.add_argument("--llm-kw", default="{}", help="extra vllm.LLM keyword arguments as JSON")
    ap.add_argument("--dry", action="store_true", help="tokenize and check answer tokens, then stop (no GPU)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(l) for l in open(args.data)]
    if args.sample:
        rows = random.Random(0).sample(rows, min(args.sample, len(rows)))
    if any(r.get("turn2") for r in rows):
        sys.exit("turn2 rows are not supported by this scorer")

    def context(r):
        return tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) + r["prefill"]

    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]
    # sent and sent_rev: the E1 sentence-level questions (build_e1.py), answered Yes or No like yn.
    FMT_ANS = {"score": ANS[:5], "yn": ANS[5:], "yn_rev": ANS[5:], "sent": ANS[5:], "sent_rev": ANS[5:]}
    how = {}  # fmt -> (strip the prefill's trailing space?, {answer: token id})
    for fmt in sorted({r["fmt"] for r in rows}):
        text = context(next(r for r in rows if r["fmt"] == fmt))
        for strip in [False, True]:
            if strip and not text.endswith(" "):
                break
            base = text.rstrip(" ") if strip else text
            b, ids = enc(base), {}
            for a in FMT_ANS[fmt]:
                add = (" " + a) if strip or not base.endswith(" ") else a
                full = enc(base + add)
                if full[:len(b)] == b and len(full) == len(b) + 1:
                    ids[a] = full[-1]
            if len(ids) == len(FMT_ANS[fmt]):
                how[fmt] = (strip, ids)
                break
        if fmt not in how:
            sys.exit(f"no single-token answers for format {fmt}; context ends with {text[-60:]!r}")
        print(f"format {fmt}: strip space={how[fmt][0]}, answer tokens",
              {a: (i, tok.decode([i])) for a, i in how[fmt][1].items()}, flush=True)

    prompts = []
    for r in rows:
        t = context(r)
        prompts.append(enc(t.rstrip(" ") if how[r["fmt"]][0] else t))
    n_tok = [len(p) for p in prompts]
    print(f"{len(rows)} rows; prompt tokens: max {max(n_tok)}, mean {sum(n_tok) / len(n_tok):.0f}", flush=True)
    if max(n_tok) >= args.max_len:
        sys.exit(f"{sum(n >= args.max_len for n in n_tok)} prompts reach --max-len {args.max_len}; raise it")
    if args.dry:
        print("--- context end of the first row:", repr(tok.decode(prompts[0][-40:])))
        sys.exit("dry run: stopping before the model loads")

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_len, max_logprobs=args.k,
              enable_prefix_caching=True, **json.loads(args.llm_kw))
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=args.k)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0, low = time.time(), 0
    with open(args.out + ".jsonl", "w") as f:
        for c0 in range(0, len(rows), args.chunk):
            outs = llm.generate([{"prompt_token_ids": p} for p in prompts[c0:c0 + args.chunk]], sp, use_tqdm=False)
            for r, p, o in zip(rows[c0:c0 + args.chunk], prompts[c0:c0 + args.chunk], outs):
                lp = o.outputs[0].logprobs[0]  # {token id: Logprob}
                ids = how[r["fmt"]][1]
                pv = [round(math.exp(lp[ids[a]].logprob), 6) if a in ids and ids[a] in lp else 0.0 for a in ANS]
                mass = sum(pv)
                low += mass < 0.5
                top = sorted(lp.items(), key=lambda kv: -kv[1].logprob)
                f.write(json.dumps({k: v for k, v in r.items() if k != "prompt"} | {
                    "p": pv, "mass": round(mass, 4), "n_tok": len(p),
                    "top": [[tok.decode([i]), round(math.exp(v.logprob), 4)] for i, v in top[:5]],
                    "floor": round(math.exp(top[-1][1].logprob), 6)}) + "\n")
            f.flush()
            n = min(c0 + args.chunk, len(rows))
            print(f"{n}/{len(rows)} {time.time() - t0:.0f}s ({n / (time.time() - t0):.1f} rows/s), mass<0.5: {low}", flush=True)
    print(f"done {len(rows)} rows in {time.time() - t0:.0f}s; rows with mass < 0.5: {low}")


if __name__ == "__main__":
    main()
