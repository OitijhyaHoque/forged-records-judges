#!/bin/bash
# Step 3: reasoning judges (thinking on) on SummEdits. Sections 4.1, 4.2 and 4.4.
# Usage: GPU=0 bash scripts/03_reasoning.sh   (one A100 80 GB; about 10 to 60 minutes per judge and run)
# Commands, data, conditions, tags and seeds are those of the original runs. The original runs spread these jobs over two
# GPUs; the order does not change any result, because every prompt has its own seed (1000 + item).
set -e
cd "$(dirname "$0")/.."
G=${GPU:-0}
T=scripts/run_think.sh
ALL8="qwen nemotron gemma gptoss glm qwen36 gemma31 muse"
GEN="glm qwen36 gemma31 muse"
PLACE=clean,claim_e,claim_e_after,claim_e_insum,claim_e_inject
think() {  # think <tag> <judge> <think_judge.py arguments...>: one judge, its own log
  local TG=$1 J=$2; shift 2
  EXTRA="$*" TAG=$TG LOG=results/think_full${TG}_$J bash $T $G full $J
}

# Main set: clean and the forged record (claim_e).
for J in nemotron gemma qwen gptoss; do think "" $J; done
# Defenses for the three core judges: the "unverified" label in the record, and the guard line in the prompt.
for J in nemotron gemma qwen; do think _unv $J --conds clean,claim_e_unverified; done
for J in nemotron gemma qwen; do think _guard $J --guard; done
# gpt-oss-120b: every record form (second run of clean and claim_e), and low reasoning effort.
think _all gptoss --conds clean,neutral,pointer_e,claim_e,claim_o,claim_global,bare_e,claim_e_unverified,claim_e_signed
think _low gptoss --effort low
# Which part of the record moves the judge (claim about another sentence, whole summary, bare verdict, pointer).
for J in nemotron gemma qwen; do think _forms $J --conds claim_o,claim_global,bare_e,pointer_e; done
# Four more judge families: clean, claim_e and the label; then the guard line.
for J in $GEN; do think _gen $J --conds clean,claim_e,claim_e_unverified; done
for J in $GEN; do think _gen_guard $J --guard --conds clean,claim_e; done
# Held-out domains, without and with the guard line (gpt-oss with --max-tokens 5000 so prompt and answer fit 8192 tokens).
for J in nemotron gemma qwen $GEN; do think _heldout $J --data data/heldout.jsonl --conds clean,claim_e,claim_e_unverified; done
think _heldout gptoss --data data/heldout.jsonl --conds clean,claim_e --max-tokens 5000
for J in nemotron gemma qwen glm qwen36 gemma31 muse; do think _heldout_guard $J --guard --data data/heldout.jsonl --conds clean,claim_e; done
think _heldout_guard gptoss --guard --data data/heldout.jsonl --conds clean,claim_e --max-tokens 5000
# Placement of the record, without and with the guard line.
for J in $ALL8; do think _place_guard $J --guard --data data/place.jsonl --conds $PLACE; think _place $J --data data/place.jsonl --conds $PLACE; done
# The two halves of the guard line (the half is in the data, so no --guard).
for J in qwen nemotron gemma $GEN; do
  for H in B A; do think _guard$H $J --data data/sprint_guard$H.jsonl --conds clean,claim_e; done
done
