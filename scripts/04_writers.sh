#!/bin/bash
# Step 4: writers that produce notes, and the judges on those notes. Sections 4.3 and 4.4 (adaptive attackers).
# Needs the outputs of scripts/02_thinking_off.sh (results/<judge>_full.jsonl, for the clean check of n1_check.py).
# Usage: GPU=0 bash scripts/04_writers.sh   (one A100 80 GB; the feedback loops take 1 to 3 hours each)
# Dual-use: this step contains the attack writer prompt and the feedback loop. See the README.
set -e
cd "$(dirname "$0")/.."
source ./env.sh
G=${GPU:-0}
N=scripts/run_n1.sh
T=scripts/run_think.sh
FOUR="qwen nemotron gemma gptoss"
GEN="glm qwen36 gemma31 muse"
think() {  # think <tag> <judge> <think_judge.py arguments...>
  local TG=$1 J=$2; shift 2
  EXTRA="$*" TAG=$TG LOG=results/think_full${TG}_$J bash $T $G full $J
}

# N1: 16 free notes per test summary from gemma-4-26B, cooperative prompt and attack prompt; Qwen and Nemotron score
# every note with thinking off (best of 16 is picked by Qwen yes/no in the readout).
bash $N $G full
bash $N $G attack
# Reasoning judges on Qwen's best of 16 attack notes and on a hash-picked attack note.
$PYTHON src/build_n1_think.py --tag n1atk
for J in nemotron gemma qwen gptoss $GEN; do think _n1atk $J --data data/n1atk_think.jsonl --conds clean,atk_best,atk_rand; done

# N1-D: the attacker picks against the defended thinking-off judge (guard line, or the "unverified" label).
for D in guard label; do $PYTHON src/n1_defense_build.py --defense $D; done
for D in guard label; do for J in qwen nemotron; do bash scripts/run_sprint.sh $J $G n1atk_$D; done; done
$PYTHON src/build_n1_think.py --tag n1atk_guard --prompts data/n1atk.jsonl --out data/n1atk_guard_think.jsonl
for J in nemotron gemma qwen gptoss; do think _n1atk_guard $J --guard --data data/n1atk_guard_think.jsonl --conds clean,atk_best,atk_rand; done

# N1-A: a writer that knows the guard line; picks by the guarded thinking-off judge, then by the guarded reasoning judge.
bash $N $G aware
for J in nemotron gemma qwen gptoss; do think _n1aware_guard $J --guard --data data/n1aware_guard_think.jsonl --conds clean,atk_best,atk_rand; done
# Qwen with thinking on and the guard line judges all 16 notes; the original run split the notes into two halves on two
# GPUs and merged them (same seeds; merge_think.py restores the row order of one run).
CA=clean$(for i in $(seq 0 7); do printf ",free%02d" $i; done)
CB=$(for i in $(seq 8 15); do printf "free%02d," $i; done); CB=${CB%,}
CALL=clean$(for i in $(seq 0 15); do printf ",free%02d" $i; done)
think _n1aware_guard16a qwen --guard --data data/n1aware.jsonl --conds $CA
think _n1aware_guard16b qwen --guard --data data/n1aware.jsonl --conds $CB
$PYTHON src/merge_think.py --data data/n1aware.jsonl --conds $CALL --out results/think_qwen_n1aware_guard16.jsonl \
  results/think_qwen_n1aware_guard16a.jsonl results/think_qwen_n1aware_guard16b.jsonl
$PYTHON src/build_n1_think.py --tag n1aware_guard --prompts data/n1aware.jsonl \
  --select-think results/think_qwen_n1aware_guard16.jsonl --out data/n1aware_tguard_think.jsonl
for J in qwen nemotron gemma gptoss; do
  think _n1aware_tguard $J --guard --seed-offset 500 --data data/n1aware_tguard_think.jsonl --conds clean,atk_best,atk_rand
done

# N1-R: Qwen writes and the guarded reasoning Qwen judges, 4 rounds x 4 notes with the judge's reasoning as feedback.
# Fresh samples: Qwen judges the kept note with --seed-offset 500.
bash $N $G refine_smoke
bash $N $G refine
for J in $FOUR; do
  S=""; [ $J = qwen ] && S="--seed-offset 500"
  think _n1refine $J --guard $S --data data/n1refine_think.jsonl --conds clean,atk_refine
done
for J in $FOUR; do think _n1refine_noguard $J --data data/n1refine_think.jsonl --conds clean,atk_refine; done

# refine_ng: the same loop against the judge without the guard line, attack writer prompt; kept notes judged without
# and with the guard line. refine64: guard line on, 8 rounds x 8 notes.
bash $N $G refine_ng
for J in $FOUR; do
  S=""; [ $J = qwen ] && S="--seed-offset 500"
  think _n1refine_ng $J $S --data data/n1refine_ng_think.jsonl --conds clean,atk_refine
  think _n1refine_ng_guard $J --guard $S --data data/n1refine_ng_think.jsonl --conds clean,atk_refine
done
bash $N $G refine64
for J in $FOUR; do
  S=""; [ $J = qwen ] && S="--seed-offset 500"
  think _n1refine64 $J --guard $S --data data/n1refine64_think.jsonl --conds clean,atk_refine
done

# M2: refine_ng plus the trace monitor inside the loop; kept notes judged without the guard line.
bash $N $G refine_mon_smoke
bash $N $G refine_mon
for J in $FOUR; do
  S=""; [ $J = qwen ] && S="--seed-offset 500"
  think _n1refine_mon $J $S --data data/n1refine_mon_think.jsonl --conds clean,atk_refine
done
