#!/bin/bash
# Step 7: inside Qwen3.5-9B and gemma-4-12B-it (report Appendix D). HF transformers in bf16 with forward hooks, one GPU
# with 32 GB (the runs used one RTX 5090). Needs data/sprint.jsonl and the thinking-off scores of the two small judges
# (scripts/02_thinking_off.sh: results/qwen9_e0_full.jsonl, results/gemma12_e0_full.jsonl).
# Usage: GPU=0 bash internals/run_internals.sh   Outputs and logs go to internals/results/.
# Order and flags are those of the original runs. Each "--limit 2" call is a two-item test before the full stage.
set -e
cd "$(dirname "$0")/.."
source ./env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
P=$PYTHON
I=internals
mkdir -p $I/results
# vllm reference scores for the smoke check, and the E0 scores the readouts read from internals/results/.
bash scripts/run_sprint.sh qwen9 ${GPU:-0} e0_smoke
cp results/qwen9_e0_full.jsonl results/gemma12_e0_full.jsonl $I/results/
# Qwen3.5-9B: smoke and two-item test, then lens (with the hidden-state cache for the probes), patching, note-part removal,
# single-layer patching (patch2), attention knockout (knock) and pointer_e versus claim_e patching (patch3).
$P $I/k1_internals.py --judge qwen9 --stage smoke patch ablate --limit 2 --ref results/qwen9_e0_smoke.jsonl > $I/results/qwen9_test.log 2>&1
$P $I/k1_internals.py --judge qwen9 --stage lens patch ablate > $I/results/qwen9_main.log 2>&1
$P $I/k1_internals.py --judge qwen9 --stage patch2 > $I/results/qwen9_patch2.log 2>&1
$P $I/k1_internals.py --judge qwen9 --stage knock > $I/results/qwen9_knock.log 2>&1
$P $I/k1_internals.py --judge qwen9 --stage patch3 > $I/results/qwen9_patch3.log 2>&1
# gemma-4-12B-it: HF and vllm scores matched only with eager attention, so its main run used eager attention.
$P $I/k1_internals.py --judge gemma12 --stage smoke --attn eager --ref results/gemma12_e0_full.jsonl > $I/results/gemma12_smoke_eager.log 2>&1
$P $I/k1_internals.py --judge gemma12 --stage lens patch ablate patch2 --attn eager > $I/results/gemma12_main.log 2>&1
$P $I/k1_internals.py --judge gemma12 --stage patch3 > $I/results/gemma12_patch3.log 2>&1
# Linear probes on the lens cache (CPU; the runs capped memory at 16 GB).
for J in qwen9 gemma12; do $P $I/k1_probe.py --judge $J --jobs 6 > $I/results/${J}_probe.log 2>&1; done
# Readouts and the report figure.
mkdir -p $I/results/figs
$P $I/k1_readouts.py --figs $I/results/figs --json $I/results/readouts_all.json > $I/results/readouts_all.txt
$P $I/k1_readouts.py --judges qwen9 --json $I/results/readouts_qwen9.json > /dev/null
$P $I/k1_report_fig.py $I/results/readouts_qwen9.json $I/results/figs/fig_internals_qwen9
