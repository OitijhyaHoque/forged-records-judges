#!/bin/bash
# Step 8: every readout behind the report, from the run outputs in results/. CPU only, a few minutes.
# Run from the repository root: bash scripts/08_readouts.sh
# Each readout writes results/<name>.txt (and a .json where the script has one). A readout whose inputs are missing
# prints FAIL and the script goes on. expected/ holds the .txt files that this script wrote from the original run outputs
# (compare with: diff -r expected results, or diff one file). The README lists the known differences.
cd "$(dirname "$0")/.."
source ./env.sh
P=$PYTHON
mkdir -p results
out() {  # out <name> <command...>: run the command, write its output to results/<name>.txt
  local N=$1; shift
  if "$@" > "results/$N.txt"; then echo "ok    results/$N.txt"; else echo "FAIL  results/$N.txt (missing inputs?)"; fi
}
tc() { local N=$1; shift; out "$N" $P src/think_check.py --examples 0 --json "results/$N.json" "$@"; }
FOUR=qwen,nemotron,gemma,gptoss
GEN=glm,qwen36,gemma31,muse
ALL8=qwen,nemotron,gemma,gptoss,glm,qwen36,gemma31,muse
PLACE="--pairs claim_e_after-claim_e,claim_e_insum-claim_e,claim_e_inject-claim_e"

# Section 4.1, Figure 1 and Table A1, thinking off (yes/no column; other formats in the same files), and the small judges.
out sprint_check $P src/sprint_check.py results/{qwen,gemma,nemotron}_full.jsonl --json results/sprint_check.json
out sprint_check_gen $P src/sprint_check.py results/{glm,qwen36,gemma31}_full.jsonl --json results/sprint_check_gen.json
out sprint_check_qwen9 $P src/sprint_check.py results/qwen9_e0_full.jsonl --json results/sprint_check_qwen9.json
out sprint_check_gemma12 $P src/sprint_check.py results/gemma12_e0_full.jsonl --json results/sprint_check_gemma12.json
out sprint_check_heldout $P src/sprint_check.py results/{qwen,nemotron,gemma}_heldout.jsonl --json results/sprint_check_heldout.json
out sprint_check_place $P src/sprint_check.py results/{qwen,nemotron,gemma}_place.jsonl results/{qwen,nemotron,gemma}_place_guard.jsonl \
  --json results/sprint_check_place.json

# Section 4.1, Figure 1 and Table A1, reasoning judges: main set, held-out domains, guard line.
tc think_check --judges nemotron,gemma,qwen,gptoss
tc think_check_all --judges gptoss --suffix _all
tc think_check_low --judges gptoss --suffix _low
tc think_check_gen --judges $GEN --suffix _gen --off 'results/{j}_full.jsonl'
tc think_check_heldout --judges nemotron,gemma,qwen,gptoss --suffix _heldout --off 'results/{j}_heldout.jsonl'
tc think_check_heldout_gen --judges $GEN --suffix _heldout
tc think_check_guard --judges nemotron,gemma --suffix _guard
tc think_check_guard_qwen --judges qwen --suffix _guard
tc think_check_gen_guard --judges $GEN --suffix _gen_guard --base-from _gen
tc think_check_heldout_guard --judges $ALL8 --suffix _heldout_guard --base-from _heldout
# Placement of the record (Section 4.1; Section 4.4 for the guard line).
tc think_check_place --judges $ALL8 --suffix _place $PLACE
tc think_check_place_guard --judges $ALL8 --suffix _place_guard $PLACE

# Sections 4.1 and 4.4, Figure 2 and Table A2: LLM-AggreFact.
tc think_check_aggrefact --judges qwen,nemotron,gemma,gptoss,glm,qwen36 --suffix _aggrefact --off 'results/{j}_aggrefact.jsonl'
tc think_check_aggrefact_guard --judges qwen,nemotron,gemma,gptoss,glm,qwen36 --suffix _aggrefact_guard --base-from _aggrefact
tc think_check_aggrefact_unv --judges qwen,nemotron,gemma,gptoss,glm,qwen36 --suffix _aggrefact_unv --merge-from _aggrefact \
  --pairs claim_e_unverified-claim_e
out think_check_aggrefact_by_domain $P src/a1_by_domain.py _aggrefact qwen,nemotron,gemma,gptoss,glm,qwen36
out think_check_aggrefact_guard_by_domain $P src/a1_by_domain.py _aggrefact_guard qwen,nemotron,gemma,gptoss,glm,qwen36
out sprint_check_aggrefact $P src/sprint_check.py results/{qwen,nemotron,gemma,glm,qwen36}_aggrefact.jsonl --json results/sprint_check_aggrefact.json

# Section 4.2, Figure 3: trace labels (results/think_labels.jsonl from scripts/06_monitor.sh). The found-then-defer counts:
out label_check $P src/label_check.py
# Note forms with thinking on (claim_e minus claim_o, bare_e):
tc think_check_forms --judges nemotron,gemma,qwen --suffix _forms --merge-from '' --off 'results/{j}_full.jsonl' \
  --pairs claim_e-claim_o,claim_e-claim_global,bare_e-pointer_e

# Section 4.3, Figure 4: writers.
out n1_check $P src/n1_check.py
out n1atk_check $P src/n1_check.py --tag n1atk --notes data/n1atk_notes.jsonl --json results/n1atk_check.json
tc think_check_n1atk --judges nemotron,gemma,qwen --suffix _n1atk --merge-from '' --pairs atk_best-atk_rand,atk_best-claim_e
tc think_check_n1atk_gptoss --judges gptoss --suffix _n1atk --merge-from '' --pairs atk_best-atk_rand,atk_best-claim_e
tc think_check_n1atk_gen --judges $GEN --suffix _n1atk --pairs atk_best-atk_rand
out n1_refine_check $P src/n1_refine_check.py --examples 0
out n1_refine_check_ng $P src/n1_refine_check.py --examples 0 --notes data/n1refine_ng_notes.jsonl --json results/n1_refine_check_ng.json
tc think_check_n1refine_ng --judges $FOUR --suffix _n1refine_ng
tc think_check_n1refine_noguard --judges $FOUR --suffix _n1refine_noguard

# Section 4.4, Table 1, Figures 5, A1 and A2: defenses.
out defense_check $P src/defense_check.py
tc think_check_guardA_main --judges qwen,nemotron,gemma --suffix _guardA --base-from ''
tc think_check_guardB_main --judges qwen,nemotron,gemma --suffix _guardB --base-from ''
tc think_check_guardA_gen --judges $GEN --suffix _guardA --base-from _gen
tc think_check_guardB_gen --judges $GEN --suffix _guardB --base-from _gen
# Adaptive attackers against the guarded judge.
out n1_check_n1atk_guard $P src/n1_check.py --tag n1atk_guard --fmts yn --notes data/n1atk_notes.jsonl --json results/n1_check_n1atk_guard.json
out n1_check_n1atk_label $P src/n1_check.py --tag n1atk_label --fmts yn --notes data/n1atk_notes.jsonl --json results/n1_check_n1atk_label.json
out n1_check_n1aware_guard $P src/n1_check.py --tag n1aware_guard --fmts yn --notes data/n1aware_notes.jsonl --json results/n1_check_n1aware_guard.json
tc think_check_n1atk_guard --judges nemotron,gemma,qwen --suffix _n1atk_guard --base-from _n1atk --pairs atk_best-atk_rand
tc think_check_n1atk_guard_gptoss --judges gptoss --suffix _n1atk_guard --base-from _n1atk --pairs atk_best-atk_rand
tc think_check_n1aware_guard --judges nemotron,gemma,qwen,gptoss --suffix _n1aware_guard --base-from _n1atk_guard --pairs atk_best-atk_rand
tc think_check_n1aware_tguard --judges $FOUR --suffix _n1aware_tguard --pairs atk_best-atk_rand
tc think_check_n1refine --judges $FOUR --suffix _n1refine
tc think_check_n1refine64 --judges $FOUR --suffix _n1refine64
tc think_check_n1refine_ng_guard --judges $FOUR --suffix _n1refine_ng_guard
out n1_refine_check_64 $P src/n1_refine_check.py --examples 0 --notes data/n1refine64_notes.jsonl --json results/n1_refine_check_64.json
# Second judges and panels (thinking on), and the trace monitor.
out think_panel_check $P src/think_panel_check.py
out monitor_check_gemma $P src/monitor_check.py results/monitor_gemma.jsonl --json results/monitor_check_gemma.json
out monitor_check_qwen $P src/monitor_check.py results/monitor_qwen.jsonl --json results/monitor_check_qwen.json
out monitor_check_gemma_aggrefact $P src/monitor_check.py results/monitor_gemma_aggrefact.jsonl --json results/monitor_check_gemma_aggrefact.json
out monitor_check_qwen_aggrefact $P src/monitor_check.py results/monitor_qwen_aggrefact.jsonl --json results/monitor_check_qwen_aggrefact.json
# M2: the feedback attacker that sees the trace monitor.
out n1_refine_check_mon $P src/n1_refine_check.py --examples 0 --notes data/n1refine_mon_notes.jsonl --json results/n1_refine_check_mon.json
tc think_check_n1refine_mon --judges $FOUR --suffix _n1refine_mon
out monitor_check_gemma_mon $P src/monitor_check.py results/monitor_gemma_mon.jsonl --json results/monitor_check_gemma_mon.json
out monitor_check_qwen_mon $P src/monitor_check.py results/monitor_qwen_mon.jsonl --json results/monitor_check_qwen_mon.json

# Section 4.5 and Figure A3: thinking off, criterion shift, operating point, selection, sentence-level checks, panel.
out recal_sdt $P src/recal_sdt.py results/qwen_full.jsonl results/gemma_full.jsonl results/nemotron_full.jsonl \
  results/qwen9_e0_full.jsonl results/gemma12_e0_full.jsonl --json results/recal_sdt.json
out select_check $P src/select_check.py
out panel_check $P src/panel_check.py
out e1_check $P src/e1_check.py qwen gemma nemotron --json results/e1_check.json
out e1b_check $P src/e1b_check.py qwen gemma --json results/e1b_check.json
out e1b_explore $P src/e1b_explore.py qwen gemma
out auroc_shift $P src/auroc_shift.py qwen gemma nemotron
out shift_compare $P src/shift_compare.py qwen gemma nemotron
