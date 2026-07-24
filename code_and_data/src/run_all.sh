#!/usr/bin/env bash
# =====================================================================
# DR-PINNs -- pelny pakiet replikacyjny (samograj).
#
# Wykonuje w kolejnosci:
#   1. run_experiment_61.py  (Example 6.1: linear control + Quickhull)
#   2. run_experiment_62.py  (Example 6.2: rotating ellipse + companion)
#   3. run_experiment_63.py  (Example 6.3: relay parabolic inclusion)
#   4. sync_figures.sh       (kopiuje wszystkie figury do figures/)
#   5. pdflatex x2           (jesli dostepny; sprawdza brak [TBD:])
#
# Kazdy skrypt zapisuje ../generated/results_6x.tex + manifest_6x.json;
# paper wczytuje makra automatycznie. Flaga --smoke przekazywana do
# wszystkich eksperymentow (szybki test end-to-end, wyniki NIE nadaja
# sie do publikacji).
#
# Uzycie:  ./run_all.sh [--smoke]
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

SMOKE="${1:-}"
if [[ -n "$SMOKE" && "$SMOKE" != "--smoke" ]]; then
    echo "Uzycie: $0 [--smoke]" >&2
    exit 1
fi
[[ "$SMOKE" == "--smoke" ]] && \
    echo ">>> TRYB SMOKE: budzety iteracji obciete, wyniki niepublikowalne."

t0=$(date +%s)
for exp in 61 62 63; do
    echo
    echo "=============================================================="
    echo ">>> Experiment 6.${exp:1} -- run_experiment_${exp}.py $SMOKE"
    echo "=============================================================="
    python3 "run_experiment_${exp}.py" $SMOKE
done

echo
echo ">>> Synchronizacja figur do figures/ ..."
./sync_figures.sh

cd ..
if command -v pdflatex > /dev/null 2>&1 && [[ -f dr-pinns.tex ]]; then
    echo ">>> Kompilacja dr-pinns.tex (2 przebiegi) ..."
    pdflatex -interaction=nonstopmode dr-pinns.tex > /dev/null
    pdflatex -interaction=nonstopmode dr-pinns.tex > /dev/null
    if command -v pdftotext > /dev/null 2>&1; then
        if pdftotext dr-pinns.pdf - 2>/dev/null | grep -q "\[TBD:"; then
            echo "!!! W PDF pozostaly znaczniki [TBD:] -- niepelna synchronizacja." >&2
            exit 2
        fi
        echo ">>> PDF bez znacznikow [TBD:] -- pelna synchronizacja."
    fi
else
    echo ">>> pdflatex niedostepny lub brak dr-pinns.tex -- pomijam kompilacje (uruchom recznie)."
fi

echo
echo ">>> Gotowe w $(( $(date +%s) - t0 )) s. Manifesty: generated/manifest_6{1,2,3}.json"
