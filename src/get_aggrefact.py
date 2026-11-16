"""Download the LLM-AggreFact test split (Tang et al. 2024, arXiv 2404.10774; Hugging Face dataset lytang/LLM-AggreFact)
and convert it to JSON lines for build_aggrefact.py.
The dataset is gated: accept its terms on Hugging Face, then log in (hf auth login) or set HF_TOKEN.
License: CC BY-ND 4.0, evaluation only. Do not publish the converted file or any rows built from it (data/aggrefact.jsonl).
The runs used data/test-00000-of-00001.parquet (17,298,205 bytes, md5 7955fe24d83ea3622938595a77d1c810), converted with
pandas 3.0.6 and pyarrow 25.0.1 (output md5 in meta/data_md5.txt).
Usage: python src/get_aggrefact.py [--parquet <local parquet file, skips the download>] [--out data/aggrefact/test.jsonl]"""
import argparse, hashlib, os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", default=None); ap.add_argument("--out", default="data/aggrefact/test.jsonl")
args = ap.parse_args()
p = args.parquet
if p is None:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("lytang/LLM-AggreFact", "data/test-00000-of-00001.parquet", repo_type="dataset")
print("parquet md5", hashlib.md5(open(p, "rb").read()).hexdigest())
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
pd.read_parquet(p).to_json(args.out, orient="records", lines=True, force_ascii=False)
print("wrote", args.out, "md5", hashlib.md5(open(args.out, "rb").read()).hexdigest())
