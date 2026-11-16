#!/bin/bash
# Step 1: build every prompt set used in the report. CPU only, a few minutes.
# Needs data/summedits.json (src/get_summedits.py) and, for LLM-AggreFact, data/aggrefact/test.jsonl
# (src/get_aggrefact.py, gated dataset). Run from the repository root: bash scripts/01_data.sh
# Writer-note sets (n1*.jsonl) are built later by scripts/04_writers.sh, because they need model outputs.
# The md5 sums of the files from the original runs are in meta/data_md5.txt; check with: md5sum -c meta/data_md5.txt
set -e
cd "$(dirname "$0")/.."
source ./env.sh
mkdir -p data
SPRINT="--single --multi --cues cues_sprint.json --exclude-ids meta/summedits_exclude_ids.txt --n-inc 200 --n-con 200 --formats score,yn,yn_rev --calib 0.25"
# Main set: 366 SummEdits items x 9 conditions x 3 formats.
$PYTHON src/build.py $SPRINT --summedits data/summedits.json --out data/sprint.jsonl
# Held-out domains podcast and sales_call.
$PYTHON src/build.py $SPRINT --summedits data/summedits.json --domains podcast,sales_call \
  --conds claim_e,pointer_e,claim_e_unverified --out data/heldout.jsonl
# Sentence-level questions (thinking off): sentence e only (E1), every sentence (E1b).
$PYTHON src/build_e1.py
$PYTHON src/build_e1b.py
# Placement of the record (after the summary, inside it, as a Review log line), with and without the guard line.
$PYTHON src/build_place.py
$PYTHON src/n1_defense_build.py --defense guard --src data/place.jsonl --out data/place_guard.jsonl --fmts yn
# The two halves of the guard line.
$PYTHON src/build_guardsplit.py --half A
$PYTHON src/build_guardsplit.py --half B
# LLM-AggreFact (only if its test split was downloaded).
if [ -s data/aggrefact/test.jsonl ]; then
  $PYTHON src/build_aggrefact.py
else
  echo "data/aggrefact/test.jsonl is missing; skipping LLM-AggreFact (see src/get_aggrefact.py)"
fi
