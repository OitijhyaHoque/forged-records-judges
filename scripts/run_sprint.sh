#!/bin/bash
# Thinking-off scoring (src/score_vllm.py): one prefill and one generated token per prompt, top-20 log-probabilities.
# Usage: [GPU_MEM=0.5] [LLM_KW='{"max_num_seqs": 96}'] scripts/run_sprint.sh <judge> <gpu index> <mode>
#   judge: qwen gemma nemotron glm qwen36 gemma31 (vLLM 0.30, one A100 80 GB) or qwen9 gemma12 (modes e0_full, e0_smoke;
#          one RTX 5090 32 GB, vLLM 0.29)
#   mode (prompt set):
#     smoke        120 sampled rows of data/sprint.jsonl     full        data/sprint.jsonl (main set)
#     e1smoke      120 sampled rows of data/e1.jsonl         e1          data/e1.jsonl
#     e1b          data/e1b.jsonl                            heldout     data/heldout.jsonl
#     place        data/place.jsonl                          place_guard data/place_guard.jsonl
#     n1atk_guard  data/n1atk_guard.jsonl                    n1atk_label data/n1atk_label.jsonl
#     aggrefact    data/aggrefact.jsonl                      e0_full     data/sprint.jsonl, small judges (gpu-mem 0.88)
#     e0_smoke     120 sampled rows of data/sprint.jsonl, small judges (reference for internals/run_internals.sh)
# GPU_MEM is passed as --gpu-mem (vllm memory share), LLM_KW as --llm-kw (extra vllm.LLM arguments as JSON).
# Runs in the foreground. Output results/<judge>_<mode>.jsonl, log results/<judge>_<mode>.log, and
# results/<judge>_<mode>.done after a clean finish.
set -e
cd "$(dirname "$0")/.."
source ./env.sh
J=$1; G=$2; MODE=$3
case $MODE in
  smoke) X=(--sample 120); D=data/sprint.jsonl ;;
  full) X=(); D=data/sprint.jsonl ;;
  e1smoke) X=(--sample 120); D=data/e1.jsonl ;;
  e1) X=(); D=data/e1.jsonl ;;
  e1b) X=(); D=data/e1b.jsonl ;;
  heldout) X=(); D=data/heldout.jsonl ;;
  n1atk_guard|n1atk_label|place|place_guard) X=(); D=data/$MODE.jsonl ;;
  aggrefact) X=(); D=data/aggrefact.jsonl ;;
  e0_full) X=(); D=data/sprint.jsonl; GPU_MEM=${GPU_MEM:-0.88} ;;
  e0_smoke) X=(--sample 120); D=data/sprint.jsonl; GPU_MEM=${GPU_MEM:-0.88} ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
[ -n "$GPU_MEM" ] && X+=(--gpu-mem "$GPU_MEM")
[ -n "$LLM_KW" ] && X+=(--llm-kw "$LLM_KW")
M=$(model_path "$J")
mkdir -p results
O=results/${J}_$MODE
rm -f "$O.done"
echo "$(date -u) start $J $MODE on GPU $G, log $O.log"
CUDA_VISIBLE_DEVICES=$G "$PYTHON" src/score_vllm.py --model "$M" --data "$D" --out "$O" --max-len 4096 "${X[@]}" > "$O.log" 2>&1
touch "$O.done"
echo "$(date -u) done $J $MODE"
