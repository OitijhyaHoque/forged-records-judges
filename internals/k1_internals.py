"""Judge internals for the forged-note sprint (pre-registered internals plan).
Runs on one RTX 5090 (32 GB) with HF transformers and plain forward hooks. Reads ../data/sprint.jsonl.

Stages (--stage):
  smoke   T0. 20 yn rows (the rows of the vllm smoke output if --ref is given). Prints p(Yes), p(No) next to the vllm
          values, checks that the last lens layer equals the model output, and times forward passes.
  lens    T2. Conditions clean, neutral, pointer_e, claim_o, claim_e; formats yn and score; all 366 items.
          Per prompt: answer logits at the output, logit lens at the answer position after every layer, and the
          residual stream after every layer at 3 sites (answer position, mean over sentence e, last summary token).
  patch   T4. Inconsistent items. Activation patching between the claim_o and claim_e runs of the same item.
  ablate  T5. Inconsistent items. Remove one sentence (unit) of the claim_e note; remove sentence e from the summary.
  patch2  Supplementary patching added after the first patch results (see stage_patch2).
  knock   Attention knockout on Qwen's full-attention layers, added after the patch2 results (see stage_knock).
  patch3  Patching between pointer_e and claim_e (quote versus claim), added after the knockout (see stage_patch3).

Targets: yn: L = z(" Yes") - z(" No"), which equals log p(Yes) - log p(No). score: E[s], the mean of 1..5 under a
softmax over the 5 answer logits. z are final logits after Gemma's soft cap. Every output row keeps the raw answer
logits, so other targets can be computed later.

Prompt and answer tokens follow ../src/score_vllm.py: chat template with enable_thinking=False, then the
prefill; "Score: " loses its trailing space and the answers are " 1".." 5"; "Answer:" is followed by " Yes" / " No".

Position alignment for patching: claim_e and claim_o differ only in the "Metadata:" line. The tokens from the
"Document:" line to the end must be identical; the item is skipped (and logged) if they are not. Token j of that tail
in one run maps to token j of the tail in the other run.

Usage (from the repository root): python internals/k1_internals.py --judge qwen9 --stage smoke [--ref results/qwen9_e0_smoke.jsonl]
       python internals/k1_internals.py --judge qwen9 --stage lens|patch|ablate [--fmts yn,score] [--limit N]
Outputs go to results/ next to this file: <judge>_<stage>_<fmt>.jsonl, and for lens also
<judge>_cache_<fmt>.npy (float16, shape [prompts, layers, 3 sites, hidden]) with row order = the jsonl order.
A <judge>_<stage>.done file marks a clean finish."""
import argparse, json, os, re, sys, time
import numpy as np
import torch

# Hugging Face model ids and the revisions used in the runs.
MODELS = {"qwen9": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
          "gemma12": ("google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7")}
CONDS = ["clean", "neutral", "pointer_e", "claim_o", "claim_e"]
FMT_ANS = {"yn": ["Yes", "No"], "score": ["1", "2", "3", "4", "5"]}
sents = lambda t: [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]
HERE = os.path.dirname(os.path.abspath(__file__))


def load(judge, attn="sdpa"):
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(MODELS[judge][0], revision=MODELS[judge][1])
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(MODELS[judge][0], revision=MODELS[judge][1], dtype=torch.bfloat16, device_map="cuda",
                                                        attn_implementation=attn)
    model.eval()
    if hasattr(model.model, "visual"):  # Qwen3.5 vision tower, not used
        del model.model.visual
    torch.cuda.empty_cache()
    print(f"loaded {judge} ({attn} attention) in {time.time() - t0:.0f}s; {type(model).__name__}; "
          f"GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    return tok, model


class Judge:
    """Language model trunk + head, answer token ids, hooks on every decoder layer output."""

    def __init__(self, judge, tok, model):
        self.tok, self.model = tok, model
        self.lm = model.model.language_model
        self.layers = self.lm.layers
        self.nL = len(self.layers)
        self.W = model.lm_head.weight  # [vocab, d]
        self.cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
        self.patch = None  # {layer: [(batch index, target positions LongTensor, source tensor [n, d])]}
        self.capture = None  # callable(layer, hidden [B, T, d])
        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(self._hook(i))
        # Attention knockout: {layer: bool mask [B, 1, T, T], True = may attend}; only softmax-attention layers.
        self.knock = None
        self.attn_layers = [i for i, layer in enumerate(self.layers) if hasattr(layer, "self_attn")]
        for i in self.attn_layers:
            self.layers[i].self_attn.register_forward_pre_hook(self._knock_hook(i), with_kwargs=True)
        self.ans = {}  # fmt -> (strip trailing space of the prefill?, LongTensor of answer token ids)
        self.dev = "cuda"

    def _hook(self, i):
        def f(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            if self.patch is not None and i in self.patch:
                for b, tgt, src in self.patch[i]:
                    h[b, tgt] = src
            if self.capture is not None:
                self.capture(i, h)
            return out
        return f

    def _knock_hook(self, i):
        def f(mod, args, kwargs):
            if self.knock is not None and i in self.knock:
                kwargs["attention_mask"] = self.knock[i]
            return args, kwargs
        return f

    def context(self, prompt, prefill):
        return self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False) + prefill

    def setup_answers(self, rows):
        enc = lambda t: self.tok(t, add_special_tokens=False)["input_ids"]
        for fmt in FMT_ANS:
            r = next(r for r in rows if r["fmt"] == fmt)
            text = self.context(r["prompt"], r["prefill"])
            for strip in [False, True]:
                if strip and not text.endswith(" "):
                    break
                base = text.rstrip(" ") if strip else text
                b, ids = enc(base), []
                for a in FMT_ANS[fmt]:
                    full = enc(base + ((" " + a) if strip or not base.endswith(" ") else a))
                    if full[:len(b)] == b and len(full) == len(b) + 1:
                        ids.append(full[-1])
                if len(ids) == len(FMT_ANS[fmt]):
                    self.ans[fmt] = (strip, torch.tensor(ids, device=self.dev))
                    break
            assert fmt in self.ans, f"no single-token answers for {fmt}"
            print(f"format {fmt}: strip={self.ans[fmt][0]}, tokens",
                  [(i, self.tok.decode([i])) for i in self.ans[fmt][1].tolist()], flush=True)

    def encode(self, prompt, prefill, fmt):
        """Token ids and offsets of the full context; char positions of the prompt's segments in that text."""
        text = self.context(prompt, prefill)
        if self.ans[fmt][0]:
            text = text.rstrip(" ")
        e = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return text, e["input_ids"], e["offset_mapping"]

    def ans_logits(self, h_normed, fmt):
        z = h_normed.float() @ self.W[self.ans[fmt][1]].float().T
        if self.cap:
            z = self.cap * torch.tanh(z / self.cap)
        return z

    def forward(self, ids):
        """ids: LongTensor [B, T]. Returns the final normed hidden state at the last position [B, d]."""
        out = self.lm(input_ids=ids, use_cache=False)
        return out.last_hidden_state[:, -1]


def target(z, fmt):
    """z [..., n answers] float -> L (yn) or E[s] (score)."""
    if fmt == "yn":
        return z[..., 0] - z[..., 1]
    p = torch.softmax(z, -1)
    return (p * torch.arange(1, 6, device=z.device, dtype=p.dtype)).sum(-1)


def summary_of(prompt):
    return prompt.rsplit("\nSummary: ", 1)[1].rsplit("\n", 1)[0]


def sent_span(summary, k):
    """Char span [a, b) of sentence k (1-based) inside the summary string, using the regex split of build.py."""
    s = sents(summary)
    pos, a = 0, None
    for i, x in enumerate(s):
        j = summary.index(x, pos)
        if i == k - 1:
            return j, j + len(x)
        pos = j + len(x)
    raise ValueError("sentence index out of range")


def spans(text, offsets, prompt, ke, ko=None):
    """Token index sets in the full context. Raises if a segment is not found. ko: also return sentence ko's tokens."""
    p0 = text.index(prompt[:200])
    doc_c = text.index("Document: ", p0)
    s_c = text.rindex("\nSummary: ") + len("\nSummary: ")
    summ = summary_of(prompt)
    assert text[s_c:s_c + len(summ)] == summ
    s_end = s_c + len(summ)
    ea, eb = sent_span(summ, ke)
    ea, eb = s_c + ea, s_c + eb
    T = len(offsets)
    over = lambda a, b: [i for i, (x, y) in enumerate(offsets) if x < b and y > a]
    tail0 = over(doc_c, doc_c + 1)[0]
    summ_idx = over(s_c, s_end)
    doc_idx = list(range(tail0, summ_idx[0]))  # "Document: ..." up to the summary text (includes "Summary: ")
    e_idx = over(ea, eb)
    q_idx = list(range(summ_idx[-1] + 1, T - 1))  # after the summary, excluding the answer position
    meta_idx = over(text.index("Metadata: ", p0), doc_c) if "Metadata: " in text[p0:doc_c] else []
    out = {"tail0": tail0, "summ": summ_idx, "doc": doc_idx, "e": e_idx, "q": q_idx, "s_last": summ_idx[-1],
           "last": T - 1, "T": T, "meta": meta_idx}
    if ko:
        oa, ob = sent_span(summ, ko)
        out["ko"] = over(s_c + oa, s_c + ob)
    return out


def rows_by(rows):
    d = {}
    for r in rows:
        d[(r["item"], r["cond"], r["fmt"])] = r
    return d


def meta_keep(r):
    return {k: r[k] for k in ["id", "item", "doc", "label", "part", "cond", "fmt", "wording", "ke", "ko", "domain",
                              "edit_types"]}


# ---------------------------------------------------------------- stages

@torch.inference_mode()
def stage_smoke(J, rows, args):
    ref = {}
    if args.ref:
        ref = {json.loads(l)["id"]: json.loads(l) for l in open(args.ref)}
        pick = [r for r in rows if r["id"] in ref and r["fmt"] == "yn"][:20]
        pick += [r for r in rows if r["id"] in ref and r["fmt"] == "score"][:10]
    else:
        pick = [r for r in rows if r["fmt"] == "yn"][:20]
    times, worst, dmax = [], 0.0, 0.0
    for r in pick:
        text, ids, off = J.encode(r["prompt"], r["prefill"], r["fmt"])
        x = torch.tensor([ids], device="cuda")
        store = torch.empty(J.nL, 1, J.W.shape[1], device="cuda", dtype=torch.bfloat16)
        J.capture = lambda i, h: store[i].copy_(h[0, -1:])
        torch.cuda.synchronize(); t0 = time.time()
        hl = J.forward(x)[0]
        torch.cuda.synchronize(); times.append((len(ids), time.time() - t0))
        J.capture = None
        full = (J.W @ hl).float()  # bf16 matmul; a float32 copy of W would take 4 GB
        if J.cap:
            full = J.cap * torch.tanh(full / J.cap)
        lp = torch.log_softmax(full, -1)
        ai = J.ans[r["fmt"]][1]
        p = lp[ai].exp().tolist()
        lens_last = J.ans_logits(J.lm.norm(store[-1, 0]), r["fmt"])
        dz = (lens_last - J.ans_logits(hl, r["fmt"])).abs().max().item()
        worst = max(worst, dz)
        top = [J.tok.decode([i]) for i in lp.topk(3).indices.tolist()]
        vp = ref.get(r["id"], {}).get("p")
        vv = ([vp[5], vp[6]] if r["fmt"] == "yn" else vp[:5]) if vp else None
        if vv:
            dmax = max(dmax, max(abs(a - b) for a, b in zip(p, vv)))
        print(f"{r['id']:<22} n={len(ids):<5} hf p={[round(v, 4) for v in p]} vllm p={vv} mass={sum(p):.3f} "
              f"top={top}", flush=True)
    print(f"max |last lens layer - output| over answer logits: {worst:.4f}")
    print(f"max |p hf - p vllm| over answer tokens: {dmax:.4f}")
    ok = worst < 0.05 and dmax < 0.08
    for n, t in times[1:6]:
        print(f"forward: {n} tokens in {t:.3f}s")
    print(f"mean forward time {np.mean([t for _, t in times[1:]]):.3f}s at mean {np.mean([n for n, _ in times[1:]]):.0f} tokens")
    # batch timing for patching
    r = pick[0]
    text, ids, off = J.encode(r["prompt"], r["prefill"], r["fmt"])
    for B in [4, 8]:
        x = torch.tensor([ids] * B, device="cuda")
        torch.cuda.synchronize(); t0 = time.time()
        J.forward(x)
        torch.cuda.synchronize()
        print(f"batch {B} x {len(ids)} tokens: {time.time() - t0:.3f}s, peak GPU "
              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    if not ok:
        sys.exit("smoke check failed: HF does not match vllm or the lens check failed")
    print("smoke check passed")


@torch.inference_mode()
def stage_lens(J, rows, args):
    d = J.W.shape[1]
    for fmt in args.fmts:
        sel = [r for r in rows if r["fmt"] == fmt and r["cond"] in CONDS]
        sel.sort(key=lambda r: (r["item"], CONDS.index(r["cond"])))
        if args.limit:
            sel = sel[:args.limit]
        out = f"{HERE}/results/{args.judge}_lens_{fmt}"
        cache = np.lib.format.open_memmap(f"{HERE}/results/{args.judge}_cache_{fmt}.npy", mode="w+",
                                          dtype=np.float16, shape=(len(sel), J.nL, 3, d))
        store = torch.empty(J.nL, 3, d, device="cuda", dtype=torch.float32)
        t0 = time.time()
        with open(out + ".jsonl", "w") as f:
            for n, r in enumerate(sel):
                text, ids, off = J.encode(r["prompt"], r["prefill"], fmt)
                sp = spans(text, off, r["prompt"], r["ke"])
                e_idx = torch.tensor(sp["e"], device="cuda")

                def cap(i, h):
                    store[i, 0] = h[0, -1]
                    store[i, 1] = h[0, e_idx].float().mean(0)
                    store[i, 2] = h[0, sp["s_last"]]
                J.capture = cap
                hl = J.forward(torch.tensor([ids], device="cuda"))[0]
                J.capture = None
                z = J.ans_logits(hl, fmt)
                zl = J.ans_logits(J.lm.norm(store[:, 0].to(torch.bfloat16)), fmt)  # [nL, n answers]
                full = (J.W @ hl).float()  # bf16 matmul; a float32 copy of W would take 4 GB
                if J.cap:
                    full = J.cap * torch.tanh(full / J.cap)
                mass = torch.softmax(full, -1)[J.ans[fmt][1]].sum().item()
                cache[n] = store.cpu().numpy().astype(np.float16)
                f.write(json.dumps(meta_keep(r) | {
                    "row": n, "n_tok": len(ids), "n_e": len(sp["e"]), "z": [round(v, 4) for v in z.tolist()],
                    "t": round(target(z, fmt).item(), 4), "mass": round(mass, 4),
                    "lens_t": [round(v, 4) for v in target(zl, fmt).tolist()],
                    "lens_z": [[round(v, 3) for v in zz] for zz in zl.tolist()]}) + "\n")
                if n % 200 == 0:
                    f.flush()
                    print(f"lens {fmt} {n}/{len(sel)} {time.time() - t0:.0f}s", flush=True)
        cache.flush(); del cache
        print(f"lens {fmt} done: {len(sel)} prompts in {time.time() - t0:.0f}s", flush=True)


def run_capture_tail(J, ids, tail0):
    """One forward; returns (final normed last hidden [d], tail states [nL, T - tail0, d] bf16)."""
    T = len(ids)
    buf = torch.empty(J.nL, T - tail0, J.W.shape[1], device="cuda", dtype=torch.bfloat16)
    J.capture = lambda i, h: buf[i].copy_(h[0, tail0:])
    hl = J.forward(torch.tensor([ids], device="cuda"))[0]
    J.capture = None
    return hl, buf


def run_patched(J, ids, plan, B):
    """plan: list (one entry per batch element) of {layer: (target position slice, source [n, d])}."""
    res, c = [], 0
    while c < len(plan):
        chunk = plan[c:c + B]
        J.patch = {}
        for b, p in enumerate(chunk):
            for l, v in p.items():
                for tgt, src in (v if isinstance(v, list) else [v]):
                    J.patch.setdefault(l, []).append((b, tgt, src))
        try:
            hl = J.forward(torch.tensor([ids] * len(chunk), device="cuda"))
        except torch.cuda.OutOfMemoryError:
            J.patch = None
            torch.cuda.empty_cache()
            if B == 1:
                raise
            B = max(1, B // 2)
            print(f"OOM at {len(ids)} tokens; batch now {B}", flush=True)
            continue
        J.patch = None
        res.append(hl)
        c += len(chunk)
    return torch.cat(res)


@torch.inference_mode()
def stage_patch(J, rows, args):
    by = rows_by(rows)
    items = sorted({r["item"] for r in rows if r["label"] == 0})
    if args.limit:
        items = items[:args.limit]
    for fmt in args.fmts:
        out = f"{HERE}/results/{args.judge}_patch_{fmt}.jsonl"
        t0, skipped = time.time(), 0
        with open(out, "w") as f:
            for n, it in enumerate(items):
                re_, ro = by[(it, "claim_e", fmt)], by[(it, "claim_o", fmt)]
                te, ie, oe = J.encode(re_["prompt"], re_["prefill"], fmt)
                to, io, oo = J.encode(ro["prompt"], ro["prefill"], fmt)
                se, so = spans(te, oe, re_["prompt"], re_["ke"]), spans(to, oo, ro["prompt"], ro["ke"])
                if ie[se["tail0"]:] != io[so["tail0"]:]:
                    skipped += 1
                    print(f"skip item {it}: tails differ", flush=True)
                    continue
                he, be = run_capture_tail(J, ie, se["tail0"])
                ho, bo = run_capture_tail(J, io, so["tail0"])
                # Position groups as slices of claim_e positions; all groups are contiguous token ranges.
                sl = lambda idx: (idx[0], idx[-1] + 1)
                groups = {"summ": sl(se["summ"]), "docsumm": (se["tail0"], se["summ"][-1] + 1), "e": sl(se["e"]),
                          "q": sl(se["q"])}
                d_eo = so["tail0"] - se["tail0"]  # claim_o position = claim_e position + d_eo
                re0 = se["tail0"]
                last = se["last"]
                # into claim_e from claim_o: P1 (answer position, one layer at a time), then each group at all layers
                plan_e = [{l: (slice(last, last + 1), bo[l, last - re0:last - re0 + 1])} for l in range(J.nL)]
                for g, (a, b) in groups.items():
                    plan_e.append({l: (slice(a, b), bo[l, a - re0:b - re0]) for l in range(J.nL)})
                # into claim_o from claim_e, same groups
                plan_o = [{l: (slice(a + d_eo, b + d_eo), be[l, a - re0:b - re0]) for l in range(J.nL)}
                          for g, (a, b) in groups.items()]
                hpe = run_patched(J, ie, plan_e, args.batch)
                hpo = run_patched(J, io, plan_o, args.batch)
                ze, zo = J.ans_logits(he, fmt), J.ans_logits(ho, fmt)
                zpe, zpo = J.ans_logits(hpe, fmt), J.ans_logits(hpo, fmt)
                tpe, tpo = target(zpe, fmt).tolist(), target(zpo, fmt).tolist()
                rd = lambda v: [round(x, 4) for x in v]
                f.write(json.dumps({"item": it, "doc": re_["doc"], "part": re_["part"], "fmt": fmt,
                                    "wording": re_["wording"], "ke": re_["ke"], "ko": re_["ko"],
                                    "n_tok_e": se["T"], "n_tok_o": so["T"], "n_tail": se["T"] - se["tail0"],
                                    "n_e": len(se["e"]), "n_summ": len(se["summ"]), "n_q": len(se["q"]),
                                    "t_e": round(target(ze, fmt).item(), 4), "t_o": round(target(zo, fmt).item(), 4),
                                    "z_e": rd(ze.tolist()), "z_o": rd(zo.tolist()),
                                    "P1": rd(tpe[:J.nL]),
                                    "into_e": dict(zip(groups, rd(tpe[J.nL:]))),
                                    "into_o": dict(zip(groups, rd(tpo)))}) + "\n")
                f.flush()
                del be, bo
                if n % 10 == 0:
                    print(f"patch {fmt} {n}/{len(items)} {time.time() - t0:.0f}s", flush=True)
        print(f"patch {fmt} done in {time.time() - t0:.0f}s; skipped {skipped}", flush=True)


@torch.inference_mode()
def stage_patch2(J, rows, args):
    """Supplementary patching (added 2026-11-14 21:40 UTC, after the first patch results):
    e_layer[l]: sentence e's tokens at one layer l only, claim_o -> claim_e (when does sentence e hold the effect);
    not_e: all summary tokens except sentence e, all layers, both directions (specificity control);
    ko: sentence ko's tokens (the sentence claim_o names), all layers, both directions."""
    by = rows_by(rows)
    items = sorted({r["item"] for r in rows if r["label"] == 0})
    if args.limit:
        items = items[:args.limit]
    for fmt in args.fmts:
        out = f"{HERE}/results/{args.judge}_patch2_{fmt}.jsonl"
        t0, skipped = time.time(), 0
        with open(out, "w") as f:
            for n, it in enumerate(items):
                re_, ro = by[(it, "claim_e", fmt)], by[(it, "claim_o", fmt)]
                te, ie, oe = J.encode(re_["prompt"], re_["prefill"], fmt)
                to, io, oo = J.encode(ro["prompt"], ro["prefill"], fmt)
                se = spans(te, oe, re_["prompt"], re_["ke"], ko=re_["ko"])
                so = spans(to, oo, ro["prompt"], ro["ke"])
                if ie[se["tail0"]:] != io[so["tail0"]:]:
                    skipped += 1
                    continue
                he, be = run_capture_tail(J, ie, se["tail0"])
                ho, bo = run_capture_tail(J, io, so["tail0"])
                re0, d_eo = se["tail0"], so["tail0"] - se["tail0"]
                ea, eb = se["e"][0], se["e"][-1] + 1
                sa, sb = se["summ"][0], se["summ"][-1] + 1
                ka, kb = se["ko"][0], se["ko"][-1] + 1
                not_e = [(a, b) for a, b in [(sa, ea), (eb, sb)] if b > a]
                plan_e = [{l: (slice(ea, eb), bo[l, ea - re0:eb - re0])} for l in range(J.nL)]
                plan_e.append({l: [(slice(a, b), bo[l, a - re0:b - re0]) for a, b in not_e] for l in range(J.nL)})
                plan_e.append({l: (slice(ka, kb), bo[l, ka - re0:kb - re0]) for l in range(J.nL)})
                plan_o = [{l: [(slice(a + d_eo, b + d_eo), be[l, a - re0:b - re0]) for a, b in not_e] for l in range(J.nL)},
                          {l: (slice(ka + d_eo, kb + d_eo), be[l, ka - re0:kb - re0]) for l in range(J.nL)}]
                tpe = target(J.ans_logits(run_patched(J, ie, plan_e, args.batch), fmt), fmt).tolist()
                tpo = target(J.ans_logits(run_patched(J, io, plan_o, args.batch), fmt), fmt).tolist()
                rd = lambda v: [round(x, 4) for x in v]
                f.write(json.dumps({"item": it, "doc": re_["doc"], "part": re_["part"], "fmt": fmt,
                                    "ke": re_["ke"], "ko": re_["ko"], "n_e": eb - ea, "n_ko": kb - ka,
                                    "t_e": round(target(J.ans_logits(he, fmt), fmt).item(), 4),
                                    "t_o": round(target(J.ans_logits(ho, fmt), fmt).item(), 4),
                                    "e_layer": rd(tpe[:J.nL]),
                                    "into_e": {"not_e": round(tpe[J.nL], 4), "ko": round(tpe[J.nL + 1], 4)},
                                    "into_o": {"not_e": round(tpo[0], 4), "ko": round(tpo[1], 4)}}) + "\n")
                f.flush()
                del be, bo
                if n % 20 == 0:
                    print(f"patch2 {fmt} {n}/{len(items)} {time.time() - t0:.0f}s", flush=True)
        print(f"patch2 {fmt} done in {time.time() - t0:.0f}s; skipped {skipped}", flush=True)


@torch.inference_mode()
def stage_patch3(J, rows, args):
    """Quote versus claim (added 2026-11-14 23:20 UTC, after the knockout results and the internal
    review, point 3). pointer_e quotes sentence e like claim_e
    but makes no claim. Patching between pointer_e and claim_e runs of the same item, all layers:
    into_e (source pointer_e, target claim_e): e, summ, docsumm; into_p (source claim_e, target pointer_e): e, summ.
    If sentence e's state carries the claim, into_e["e"] moves t_e to t_p. If it only carries the copy of the quote,
    it stays near t_e."""
    by = rows_by(rows)
    items = sorted({r["item"] for r in rows if r["label"] == 0})
    if args.limit:
        items = items[:args.limit]
    for fmt in args.fmts:
        out = f"{HERE}/results/{args.judge}_patch3_{fmt}.jsonl"
        t0, skipped = time.time(), 0
        with open(out, "w") as f:
            for n, it in enumerate(items):
                re_, rp = by[(it, "claim_e", fmt)], by[(it, "pointer_e", fmt)]
                te, ie, oe = J.encode(re_["prompt"], re_["prefill"], fmt)
                tp, ip, op = J.encode(rp["prompt"], rp["prefill"], fmt)
                se, sp = spans(te, oe, re_["prompt"], re_["ke"]), spans(tp, op, rp["prompt"], rp["ke"])
                if ie[se["tail0"]:] != ip[sp["tail0"]:]:
                    skipped += 1
                    print(f"skip item {it}: tails differ", flush=True)
                    continue
                he, be = run_capture_tail(J, ie, se["tail0"])
                hp, bp = run_capture_tail(J, ip, sp["tail0"])
                re0, d = se["tail0"], sp["tail0"] - se["tail0"]  # pointer_e position = claim_e position + d
                sl = lambda idx: (idx[0], idx[-1] + 1)
                groups = {"e": sl(se["e"]), "summ": sl(se["summ"]), "docsumm": (re0, se["summ"][-1] + 1)}
                plan_e = [{l: (slice(a, b), bp[l, a - re0:b - re0]) for l in range(J.nL)} for a, b in groups.values()]
                plan_p = [{l: (slice(a + d, b + d), be[l, a - re0:b - re0]) for l in range(J.nL)}
                          for g, (a, b) in groups.items() if g != "docsumm"]
                tpe = target(J.ans_logits(run_patched(J, ie, plan_e, args.batch), fmt), fmt).tolist()
                tpp = target(J.ans_logits(run_patched(J, ip, plan_p, args.batch), fmt), fmt).tolist()
                f.write(json.dumps({"item": it, "doc": re_["doc"], "part": re_["part"], "fmt": fmt,
                                    "wording": re_["wording"], "ke": re_["ke"],
                                    "t_e": round(target(J.ans_logits(he, fmt), fmt).item(), 4),
                                    "t_p": round(target(J.ans_logits(hp, fmt), fmt).item(), 4),
                                    "into_e": dict(zip(groups, [round(x, 4) for x in tpe])),
                                    "into_p": dict(zip(["e", "summ"], [round(x, 4) for x in tpp]))}) + "\n")
                f.flush()
                del be, bp
                if n % 20 == 0:
                    print(f"patch3 {fmt} {n}/{len(items)} {time.time() - t0:.0f}s", flush=True)
        print(f"patch3 {fmt} done in {time.time() - t0:.0f}s; skipped {skipped}", flush=True)


def knock_masks(J, T, variants):
    """variants: list of (layers, [(row slice, col slice), ...]) per batch element -> {layer: bool mask [B,1,T,T]}.
    Layers where no element blocks keep the model's own mask. Only for models without sliding-window layers."""
    causal = torch.ones(T, T, dtype=torch.bool, device="cuda").tril()
    layers = sorted({l for ls, _ in variants for l in ls})
    out = {}
    for l in layers:
        m = causal.expand(len(variants), 1, T, T).clone()
        for b, (ls, blocks) in enumerate(variants):
            if l in ls:
                for rs, cs in blocks:
                    m[b, 0, rs, cs] = False
        out[l] = m
    return out


@torch.inference_mode()
def stage_knock(J, rows, args):
    """Attention knockout (added 2026-11-14 22:00 UTC, after the patch2 results; Qwen only, its 8 full-attention layers).
    In the claim_e run, block attention from a set of query tokens to a set of key tokens in a set of layers:
      none        plain causal mask at every full-attention layer (numerical check against t_e)
      e_note_all  sentence e -> note, all full-attention layers; e_note_L3/L7/L11: one layer; e_note_late: layers >= 15
      e_quote_all sentence e -> the note's quote of sentence e only, all full-attention layers
      q_note_all  tokens after the summary (question and answer position) -> note, all full-attention layers
      post_note_all  every token after the note -> note, all full-attention layers
    claim_o run: o_e_note_all and o_post_note_all, the same blocks (controls).
    Linear-attention layers are not touched, so information can still pass from the note through them."""
    assert not any(t == "sliding_attention" for t in getattr(J.model.config.get_text_config(), "layer_types", [])), \
        "knockout masks do not handle sliding-window layers"
    by = rows_by(rows)
    A = J.attn_layers
    items = sorted({r["item"] for r in rows if r["label"] == 0})
    if args.limit:
        items = items[:args.limit]
    names_e = ["none", "e_note_all", "e_note_L3", "e_note_L7", "e_note_L11", "e_note_late", "e_quote_all",
               "q_note_all", "post_note_all"]
    names_o = ["o_e_note_all", "o_post_note_all"]
    for fmt in args.fmts:
        out = f"{HERE}/results/{args.judge}_knock_{fmt}.jsonl"
        t0 = time.time()
        with open(out, "w") as f:
            for n, it in enumerate(items):
                re_, ro = by[(it, "claim_e", fmt)], by[(it, "claim_o", fmt)]
                te, ie, oe = J.encode(re_["prompt"], re_["prefill"], fmt)
                to, io, oo = J.encode(ro["prompt"], ro["prefill"], fmt)
                se, so = spans(te, oe, re_["prompt"], re_["ke"]), spans(to, oo, ro["prompt"], ro["ke"])
                # the note's quote of sentence e, inside the metadata line of the claim_e prompt
                s_e = sents(summary_of(re_["prompt"]))[re_["ke"] - 1]
                m0 = te.index("Metadata: ")
                qa = te.index(s_e, m0)
                assert qa < te.index("Document: ", m0)
                quote = [i for i, (x, y) in enumerate(oe) if x < qa + len(s_e) and y > qa]
                sl = lambda idx: slice(idx[0], idx[-1] + 1)
                note_e, note_o = sl(se["meta"]), sl(so["meta"])
                post_e, post_o = slice(se["tail0"], se["T"]), slice(so["tail0"], so["T"])
                q_e = slice(se["q"][0], se["T"])
                late = [l for l in A if l >= 15]
                var_e = [(A, []), (A, [(sl(se["e"]), note_e)]), ([3], [(sl(se["e"]), note_e)]),
                         ([7], [(sl(se["e"]), note_e)]), ([11], [(sl(se["e"]), note_e)]), (late, [(sl(se["e"]), note_e)]),
                         (A, [(sl(se["e"]), sl(quote))]), (A, [(q_e, note_e)]), (A, [(post_e, note_e)])]
                var_o = [(A, [(sl(so["e"]), note_o)]), (A, [(post_o, note_o)])]
                he = J.forward(torch.tensor([ie], device="cuda"))[0]
                ho = J.forward(torch.tensor([io], device="cuda"))[0]
                res = {}
                for ids, var, names in [(ie, var_e, names_e), (io, var_o, names_o)]:
                    for c in range(0, len(var), args.batch):
                        chunk = var[c:c + args.batch]
                        J.knock = knock_masks(J, len(ids), chunk)
                        h = J.forward(torch.tensor([ids] * len(chunk), device="cuda"))
                        J.knock = None
                        for nm, t in zip(names[c:c + args.batch], target(J.ans_logits(h, fmt), fmt).tolist()):
                            res[nm] = round(t, 4)
                f.write(json.dumps({"item": it, "doc": re_["doc"], "part": re_["part"], "fmt": fmt,
                                    "t_e": round(target(J.ans_logits(he, fmt), fmt).item(), 4),
                                    "t_o": round(target(J.ans_logits(ho, fmt), fmt).item(), 4),
                                    "n_quote": len(quote), "n_meta": len(se["meta"]), "knock": res}) + "\n")
                f.flush()
                if n % 20 == 0:
                    print(f"knock {fmt} {n}/{len(items)} {time.time() - t0:.0f}s", flush=True)
        print(f"knock {fmt} done in {time.time() - t0:.0f}s", flush=True)


def note_units(cues, wording):
    """Units of the claim_e template wording: split after . ! ? or a closing quote."""
    tpl = cues["claim_e"]["wordings"][wording]
    return [u for u in re.split(r'(?<=[.!?"])\s+', tpl) if u]


ROLES = {0: ["prior_score", "error_report", "fix_claim", "quote", "verify", "version"],
         1: ["score_error", "fix_claim", "quote", "verify"],
         2: ["prior_score", "error_report", "fix_quote", "verify"]}


def drop_sentence(prompt, ke):
    """The prompt with sentence ke removed from the summary (and the whitespace after it, or before it if last)."""
    summ = summary_of(prompt)
    a, b = sent_span(summ, ke)
    if b < len(summ):
        m = re.match(r"\s*", summ[b:])
        new = summ[:a] + summ[b + m.end():]
    else:
        m = re.search(r"\s*$", summ[:a])
        new = summ[:m.start()]
    i = prompt.rindex("\nSummary: ") + len("\nSummary: ")
    assert prompt[i:i + len(summ)] == summ
    return prompt[:i] + new + prompt[i + len(summ):]


@torch.inference_mode()
def stage_ablate(J, rows, args):
    cues = json.load(open(f"{HERE}/../cues_sprint.json"))
    by = rows_by(rows)
    items = sorted({r["item"] for r in rows if r["label"] == 0})
    if args.limit:
        items = items[:args.limit]
    for fmt in args.fmts:
        out = f"{HERE}/results/{args.judge}_ablate_{fmt}.jsonl"
        t0, bad = time.time(), 0

        def tval(prompt, prefill):
            text, ids, off = J.encode(prompt, prefill, fmt)
            z = J.ans_logits(J.forward(torch.tensor([ids], device="cuda"))[0], fmt)
            return round(target(z, fmt).item(), 4), [round(v, 4) for v in z.tolist()]

        with open(out, "w") as f:
            for n, it in enumerate(items):
                re_ = by[(it, "claim_e", fmt)]
                pre = re_["prefill"]
                s_e = sents(summary_of(re_["prompt"]))[re_["ke"] - 1]
                units = note_units(cues, re_["wording"])
                filled = [u.format(k=re_["ke"], s=s_e) for u in units]
                note = " ".join(filled)
                if f"Metadata: {note}\n" not in re_["prompt"]:
                    bad += 1
                    print(f"item {it}: rebuilt note not found in the prompt", flush=True)
                    continue
                row = {"item": it, "doc": re_["doc"], "part": re_["part"], "fmt": fmt, "wording": re_["wording"],
                       "ke": re_["ke"], "roles": ROLES[re_["wording"]]}
                row["base"] = {c: tval(by[(it, c, fmt)]["prompt"], pre)[0] for c in ["clean", "claim_o", "claim_e"]}
                row["drop_unit"] = []
                for j in range(len(filled)):
                    nj = " ".join(filled[:j] + filled[j + 1:])
                    row["drop_unit"].append(tval(re_["prompt"].replace(f"Metadata: {note}\n", f"Metadata: {nj}\n"), pre)[0])
                assert re_["prompt"].replace(f"Metadata: {note}\n", "") == by[(it, "clean", fmt)]["prompt"]
                row["drop_e"] = {c: tval(drop_sentence(by[(it, c, fmt)]["prompt"], re_["ke"]), pre)[0]
                                 for c in ["clean", "claim_o", "claim_e"]}
                f.write(json.dumps(row) + "\n")
                f.flush()
                if n % 20 == 0:
                    print(f"ablate {fmt} {n}/{len(items)} {time.time() - t0:.0f}s", flush=True)
        print(f"ablate {fmt} done in {time.time() - t0:.0f}s; bad notes {bad}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, choices=list(MODELS))
    ap.add_argument("--stage", required=True, nargs="+", choices=["smoke", "lens", "patch", "ablate", "patch2", "knock", "patch3"])
    ap.add_argument("--data", default=f"{HERE}/../data/sprint.jsonl")
    ap.add_argument("--fmts", default="yn,score")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ref", default="")
    ap.add_argument("--attn", default="sdpa", help="attention implementation: sdpa or eager")
    args = ap.parse_args()
    args.fmts = args.fmts.split(",")
    os.makedirs(f"{HERE}/results", exist_ok=True)
    rows = [r for r in map(json.loads, open(args.data)) if r["fmt"] in FMT_ANS and r["cond"] in CONDS]
    print(f"{len(rows)} rows", flush=True)
    tok, model = load(args.judge, args.attn)
    J = Judge(args.judge, tok, model)
    J.setup_answers(rows)
    for st in args.stage:
        t0 = time.time()
        {"smoke": stage_smoke, "lens": stage_lens, "patch": stage_patch, "ablate": stage_ablate,
         "patch2": stage_patch2, "knock": stage_knock,
         "patch3": stage_patch3}[st](J, rows, args)
        print(f"stage {st} finished in {time.time() - t0:.0f}s", flush=True)
        if not args.limit and st != "smoke":
            open(f"{HERE}/results/{args.judge}_{st}.done", "w").close()


if __name__ == "__main__":
    main()
