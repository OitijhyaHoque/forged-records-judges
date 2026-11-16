#!/bin/bash
# Step 5: LLM-AggreFact (real system outputs, human labels). Sections 4.1 and 4.4, Figure 2, Table A2, and the thinking-off numbers.
# Needs data/aggrefact.jsonl (scripts/01_data.sh after src/get_aggrefact.py).
# Usage: GPU=0 bash scripts/05_aggrefact.sh   (one A100 80 GB; about 5 to 25 minutes per judge and run)
set -e
cd "$(dirname "$0")/.."
G=${GPU:-0}
T=scripts/run_think.sh
A=data/aggrefact.jsonl
SIX="qwen nemotron gemma gptoss glm qwen36"
think() {  # think <tag> <judge> <think_judge.py arguments...>
  local TG=$1 J=$2; shift 2
  EXTRA="$*" TAG=$TG LOG=results/think_full${TG}_$J bash $T $G full $J
}
for J in $SIX; do think _aggrefact $J --data $A --conds clean,claim_e; done
for J in $SIX; do think _aggrefact_guard $J --guard --data $A --conds clean,claim_e; done
# The "unverified" label: only claim_e_unverified here; the readout takes clean from the _aggrefact run (--merge-from).
for J in $SIX; do think _aggrefact_unv $J --data $A --conds claim_e_unverified; done
# Thinking off, all conditions and formats.
for J in qwen nemotron gemma glm qwen36; do bash scripts/run_sprint.sh $J $G aggrefact; done
