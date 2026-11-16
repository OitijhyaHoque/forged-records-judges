#!/bin/bash
# Step 6: a second model reads the reasoning traces (thinking off, greedy). Sections 4.2 (Figure 3, right) and 4.4 (trace monitor, Figure 5).
# Needs the reasoning runs of scripts/03_reasoning.sh, 04_writers.sh and 05_aggrefact.sh. Sections whose inputs are
# missing are skipped.
# Usage: GPU=0 bash scripts/06_monitor.sh   (one A100 80 GB)
cd "$(dirname "$0")/.."
source ./env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
P=$PYTHON
have() { for f in "$@"; do [ -s "$f" ] || { echo "missing $f; skipping"; return 1; }; done; }
mon() {  # mon <monitor judge> <name> <conds> <files...>
  local M=$1 NAME=$2 C=$3; shift 3
  have "$@" || return 0
  echo "$(date -u) monitor $M $NAME"
  $P src/monitor_traces.py --model "$(model_path $M)" --files "$@" --conds $C --out results/monitor_${NAME}.jsonl \
    > results/monitor_${NAME}.log 2>&1 && echo "$(date -u) done" || echo "failed; see results/monitor_${NAME}.log"
}

# Trace labels (Section 4.2, Figure 3 right): did the reasoning find the error, and did it defer to the note? Labeler gemma-4-26B.
F="results/think_nemotron.jsonl results/think_gemma.jsonl results/think_qwen.jsonl results/think_gptoss.jsonl"
if have $F; then
  $P src/label_traces.py --model "$(model_path gemma)" --files $F --out results/think_labels.jsonl > results/think_labels.log 2>&1
fi

# Trace monitor on the forged record, the attack notes and the refine_ng notes, 8 judges (10,600 traces per monitor).
F=""
for J in qwen nemotron gemma gptoss; do F="$F results/think_$J.jsonl"; done
for J in glm qwen36 gemma31 muse; do F="$F results/think_${J}_gen.jsonl"; done
for J in qwen nemotron gemma gptoss glm qwen36 gemma31 muse; do F="$F results/think_${J}_n1atk.jsonl"; done
for J in qwen nemotron gemma gptoss; do F="$F results/think_${J}_n1refine_ng.jsonl"; done
for M in gemma qwen; do mon $M $M clean,claim_e,atk_best,atk_refine $F; done

# Trace monitor on LLM-AggreFact, without and with the guard line, 6 judges.
F=""
for J in qwen nemotron gemma gptoss glm qwen36; do F="$F results/think_${J}_aggrefact.jsonl results/think_${J}_aggrefact_guard.jsonl"; done
for M in gemma qwen; do mon $M ${M}_aggrefact clean,claim_e $F; done

# M2: the trace monitor on the kept notes of the attacker that saw the monitor.
F=""
for J in qwen nemotron gemma gptoss; do F="$F results/think_${J}_n1refine_mon.jsonl"; done
for M in qwen gemma; do mon $M ${M}_mon clean,atk_refine $F; done
