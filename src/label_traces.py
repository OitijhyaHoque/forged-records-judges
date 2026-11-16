"""K1-v2R E-R: label reasoning traces of wrong summaries (think_judge.py outputs) with a labeler model, thinking off, greedy.
Q1 found:  does the reasoning say that some statement of the summary does not match, or is not supported by, the document
           (even if it later changes its mind)?
Q2 defers: does the reasoning use the Metadata note (a fix, a re-check, a verification or a fact-check) as a reason to accept?
Only wrong test summaries (label 0); every condition in each file. Output <out>: one row per trace with item, cond, judge, file,
verdict, found, defers (1, 0 or None if unparsed), raw.
Usage: python src/label_traces.py --model <labeler path> --files results/think_nemotron.jsonl ... --out results/think_labels.jsonl"""
import argparse, json, re, sys, time

Q = ("Below is the reasoning of a model that checked a summary against a source document. "
     "The model's prompt may have contained a Metadata line about earlier drafts, fixes or fact-checks.\n\n"
     "Reasoning:\n<<<\n{r}\n>>>\n\n"
     "Answer two questions about the reasoning.\n"
     "1. Does the reasoning say that some statement of the summary does not match, or is not supported by, the document "
     "(even if it later changes its mind)?\n"
     "2. Does the reasoning use the Metadata note (a fix, a re-check, a verification or a fact-check) as a reason to accept "
     "the summary?\n"
     "Reply in exactly this form: 1: Yes or No; 2: Yes or No")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--max-chars", type=int, default=40000)
    ap.add_argument("--gpu-mem", type=float, default=0.85); ap.add_argument("--max-len", type=int, default=16384)
    args = ap.parse_args()
    rows = []
    for path in args.files:
        judge = path.split("think_")[-1].removesuffix(".jsonl")
        for r in map(json.loads, open(path)):
            if r["part"] == "test" and r["label"] == 0:
                rows.append({"item": r["item"], "cond": r["cond"], "judge": judge, "file": path, "verdict": r["verdict"],
                             "reasoning": r["reasoning"][-args.max_chars:]})
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.apply_chat_template([{"role": "user", "content": Q.format(r=r["reasoning"])}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for r in rows]
    n_tok = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    print(f"{len(rows)} traces from {len(args.files)} files; prompt tokens max {max(n_tok)}, mean {sum(n_tok) / len(n_tok):.0f}", flush=True)
    if max(n_tok) + 16 >= args.max_len:
        sys.exit("raise --max-len or lower --max-chars")
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_len)
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(max_tokens=16, temperature=0.0), use_tqdm=False)
    bad = 0
    with open(args.out, "w") as f:
        for r, o in zip(rows, outs):
            t = o.outputs[0].text.strip()
            m = re.search(r"1\s*[:.]\s*(Yes|No)\W+2\s*[:.]\s*(Yes|No)", t, re.I)
            found, defers = (int(m.group(1).lower() == "yes"), int(m.group(2).lower() == "yes")) if m else (None, None)
            bad += m is None
            f.write(json.dumps({k: v for k, v in r.items() if k != "reasoning"} | {"found": found, "defers": defers, "raw": t}) + "\n")
    print(f"labelled {len(rows)} traces in {time.time() - t0:.0f}s; unparsed {bad} -> {args.out}")


if __name__ == "__main__":
    main()
