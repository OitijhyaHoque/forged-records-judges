#!/bin/bash
# Step 9: the figures, from the readout JSON files of scripts/08_readouts.sh. CPU only (matplotlib).
# Run from the repository root: bash scripts/09_figures.sh   Output: figs/*.png and figs/*.pdf
cd "$(dirname "$0")/.."
source ./env.sh
P=$PYTHON
mkdir -p figs
run() { "$@" && echo "ok    $*" || echo "FAIL  $* (missing inputs?)"; }
run $P src/fig_generality.py --out figs   # fig_generality: the record across 8 judges, thinking off, on, guard line
run $P src/fig_think.py --out figs        # fig_think: thinking off vs on; traces that found the error
run $P src/fig_forms_think.py --out figs  # fig_forms_think: which part of the record moves the judge
run $P src/fig_n1_attack.py --out figs    # fig_n1_attack: best of k writer notes, cooperative vs attack prompt
run $P src/fig_defenses.py --out figs     # fig_guard_halves, fig_adaptive, fig_monitor
run $P src/fig_recal.py --out figs        # fig_roc_shift, fig_pitfall (operating point)
run $P src/make_figs.py results/sprint_check.json --out figs   # fig_miss, fig_auroc (thinking off, all conditions)
[ -s src/fig_aggrefact.py ] && run $P src/fig_aggrefact.py --out figs   # fig_aggrefact: LLM-AggreFact (Figure 2, Table A2)
