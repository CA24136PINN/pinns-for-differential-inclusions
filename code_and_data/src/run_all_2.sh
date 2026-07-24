#!/usr/bin/env bash

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
