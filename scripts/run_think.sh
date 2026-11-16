#!/bin/bash
# Reasoning judges (src/think_judge.py): the yes/no prompt with thinking on, one sample per prompt.
# Usage: [EXTRA="..."] [TAG=_x] [LOG=results/name] scripts/run_think.sh <gpu index> <smoke|full> <judge> [<judge> ...]
#   judges: qwen gemma nemotron gptoss glm qwen36 gemma31 muse (vLLM 0.30, one A100 80 GB each)
#   EXTRA: more think_judge.py arguments, for example "--guard" or "--data data/heldout.jsonl --conds clean,claim_e".
#   TAG: file tag; output results/think_<judge><TAG>.jsonl (smoke: <TAG>_smoke, first 3 test items).
#   LOG: log and .done path without the extension (default results/think_<mode><TAG>).
# Seeds: 1000 + item (+ --seed-offset), so clean and note prompts of one summary share a seed.
# Runs the judges in sequence in the foreground; LOG.done marks a clean finish of all of them.
set -e
cd "$(dirname "$0")/.."
source ./env.sh
G=$1; MODE=$2; shift 2
X=(); SUF=""
[ "$MODE" = smoke ] && X=(--limit 3) && SUF=_smoke
O=${LOG:-results/think_$MODE$TAG}
mkdir -p results
rm -f "$O.done"
echo "$(date -u) start think $MODE ($*) on GPU $G, log $O.log"
: > "$O.log"
for J in "$@"; do
  case $J in
    qwen|gemma|nemotron|qwen36|gemma31) E=() ;;
    gptoss) E=(--effort medium --gpu-mem 0.95 --max-len 8192 --max-tokens 6144) ;;
    glm) E=(--gpu-mem 0.92) ;;
    muse) E=(--strength medium --gpu-mem 0.92) ;;
    *) echo "unknown judge $J"; exit 1 ;;
  esac
  M=$(model_path "$J")
  # EXTRA is split on spaces on purpose, as in the original runs (no argument in it contains a space).
  CUDA_VISIBLE_DEVICES=$G "$PYTHON" src/think_judge.py --model "$M" --out "results/think_$J$TAG$SUF" "${E[@]}" "${X[@]}" $EXTRA >> "$O.log" 2>&1
done
touch "$O.done"
echo "$(date -u) done think $MODE ($*)"
