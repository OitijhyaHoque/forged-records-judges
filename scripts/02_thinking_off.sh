#!/bin/bash
# Step 2: judges without reasoning (one answer token, calibrated threshold). Sections 4.1 and 4.5, Figure 1 and Table A1 first column.
# Usage: GPU=0 bash scripts/02_thinking_off.sh   (one A100 80 GB; about 10 to 30 minutes per judge and set)
# The original runs used two GPUs at once; the order below is the order of the results, not a requirement.
set -e
cd "$(dirname "$0")/.."
G=${GPU:-0}
S=scripts/run_sprint.sh
# Main set, three large judges. The original runs checked a 120-row smoke run first:
#   bash $S <judge> $G smoke && python src/smoke_check.py results/<judge>_smoke.jsonl
for J in qwen gemma nemotron; do bash $S $J $G full; done
# Three more judge families.
for J in glm qwen36 gemma31; do bash $S $J $G full; done
# Two small judges (originally on one RTX 5090 with vLLM 0.29, --gpu-mem 0.88).
for J in qwen9 gemma12; do bash $S $J $G e0_full; done
# Held-out domains.
for J in qwen nemotron gemma; do bash $S $J $G heldout; done
# Sentence-level questions: sentence e (E1), every sentence (E1b). E1b ran on a shared GPU with these memory settings.
for J in gemma nemotron qwen; do bash $S $J $G e1; done
GPU_MEM=0.48 LLM_KW='{"max_num_seqs": 96}' bash $S qwen $G e1b
GPU_MEM=0.8 bash $S gemma $G e1b
# Placement of the record, with and without the guard line.
for J in qwen nemotron gemma; do bash $S $J $G place_guard; bash $S $J $G place; done
