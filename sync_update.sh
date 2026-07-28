#!/usr/bin/env bash
# =====================================================================
# sync_update.sh -- selectively apply the referee-revision update from
# code_and_data_updated/ into code_and_data/ WITHOUT touching any
# GPU-expensive outputs (training runs, checkpoints, analysis arrays).
#
# Usage:
#   bash sync_update.sh                # DRY RUN: show what would change
#   bash sync_update.sh --apply        # copy, backing up overwritten files
#   bash sync_update.sh SRC DST [--apply]
#
# Only the files on the explicit whitelist below are ever copied.
# Everything under results/raw/exp63/runs/, .../checkpoints/ and the
# analysis arrays is hard-blocked as an extra safety net.
# Overwritten files are saved to DST/_backup_<timestamp>/ first.
# =====================================================================
set -euo pipefail

SRC="code_and_data_updated"
DST="code_and_data"
APPLY=0
for a in "$@"; do
    case "$a" in
        --apply) APPLY=1 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) if [[ "$SRC" == "code_and_data_updated" && -d "$a" ]]; then SRC="$a"
           elif [[ "$DST" == "code_and_data" ]]; then DST="$a"
           else echo "Unknown argument: $a" >&2; exit 1; fi ;;
    esac
done
# If the unpacked zip contains a nested code_and_data/, descend into it.
[[ -d "$SRC/code_and_data" && ! -d "$SRC/experiments" ]] && SRC="$SRC/code_and_data"

[[ -d "$SRC" ]] || { echo "!! source not found: $SRC" >&2; exit 1; }
[[ -d "$DST" ]] || { echo "!! destination not found: $DST" >&2; exit 1; }

# ---- whitelist: the ONLY files this script will ever write -----------
FILES=(
    experiments/run_experiment_63.py
    experiments/convergence_study_63.py
    paper/dr-pinns.tex
    paper/dr-pinns.pdf
    paper/generated/table_63_solver_convergence.tex
    paper/generated/results_63.tex
    results/raw/exp63/reference.npz
    results/raw/exp63/reference_manifest.json
    results/raw/exp63/manifest_63.json
    results/raw/exp63/convergence_ref.json
    results/aggregated/manifest_63.json
    results/aggregated/table_63_solver_convergence.tex
    results/aggregated/results_63.tex
    results/figures/solver_convergence.png
)
# stale file superseded by convergence_ref.json (removed, not copied)
STALE=(
    results/raw/exp63/convergence_results_63.json
)
# ---- hard guard: never write anything matching these -----------------
FORBIDDEN='(^|/)(runs|checkpoints)(/|$)|analysis\.npz|raw_data\.npz'

for f in "${FILES[@]}"; do
    if [[ "$f" =~ $FORBIDDEN ]]; then
        echo "!! INTERNAL ERROR: whitelist entry '$f' hits the GPU-results" \
             "guard -- refusing to run." >&2
        exit 2
    fi
done

TS=$(date +%Y%m%d_%H%M%S)
BACKUP="$DST/_backup_$TS"
n_new=0 n_mod=0 n_same=0 n_miss=0

status_of() {  # status_of <src> <dst>
    if [[ ! -e "$2" ]]; then echo NEW
    elif cmp -s "$1" "$2"; then echo IDENTICAL
    else echo DIFFERS; fi
}

echo "Source:      $SRC"
echo "Destination: $DST"
[[ $APPLY -eq 1 ]] && echo "Mode:        APPLY (backup: $BACKUP)" \
                   || echo "Mode:        DRY RUN (nothing will be written;" \
                           "re-run with --apply)"
echo
printf "%-10s %-58s %s\n" "STATUS" "FILE" "DETAILS"
printf "%-10s %-58s %s\n" "------" "----" "-------"

for f in "${FILES[@]}"; do
    s="$SRC/$f"; d="$DST/$f"
    if [[ ! -f "$s" ]]; then
        printf "%-10s %-58s %s\n" "MISSING" "$f" "(not in source -- skipped)"
        n_miss=$((n_miss+1)); continue
    fi
    st=$(status_of "$s" "$d")
    case "$st" in
        NEW)
            printf "%-10s %-58s %s\n" "NEW" "$f" \
                   "$(stat -c%s "$s") B"
            n_new=$((n_new+1)) ;;
        IDENTICAL)
            printf "%-10s %-58s %s\n" "identical" "$f" "(skipped)"
            n_same=$((n_same+1)); continue ;;
        DIFFERS)
            det="dst: $(stat -c%s "$d") B, $(date -d @"$(stat -c%Y "$d")" '+%F %H:%M')"
            det+="  ->  src: $(stat -c%s "$s") B"
            printf "%-10s %-58s %s\n" "UPDATE" "$f" "$det"
            # short content summary for text files
            if file -b "$d" | grep -qi text; then
                nch=$( (diff -u "$d" "$s" 2>/dev/null || true) \
                       | grep -c '^[+-][^+-]' || true)
                printf "%-10s %-58s %s\n" "" "" "($nch changed lines)"
            fi
            n_mod=$((n_mod+1)) ;;
    esac
    if [[ $APPLY -eq 1 ]]; then
        if [[ -e "$d" ]]; then
            mkdir -p "$BACKUP/$(dirname "$f")"
            cp -p "$d" "$BACKUP/$f"
        fi
        mkdir -p "$DST/$(dirname "$f")"
        cp -p "$s" "$d"
    fi
done

echo
for f in "${STALE[@]}"; do
    d="$DST/$f"
    if [[ -f "$d" ]]; then
        printf "%-10s %-58s %s\n" "STALE" "$f" \
               "(superseded by convergence_ref.json)"
        if [[ $APPLY -eq 1 ]]; then
            mkdir -p "$BACKUP/$(dirname "$f")"
            mv "$d" "$BACKUP/$f"
        fi
    fi
done

echo
echo "Summary: $n_new new, $n_mod updated, $n_same identical," \
     "$n_miss missing in source."
if [[ $APPLY -eq 1 ]]; then
    [[ -d "$BACKUP" ]] && echo "Backups of every overwritten/removed file:" \
                               "$BACKUP" \
                       || echo "Nothing was overwritten -- no backup created."
    echo "NOT touched (guaranteed): results/raw/exp63/runs/," \
         "results/raw/exp63/checkpoints/, analysis.npz, raw_data.npz," \
         "exp61/, exp62/ and everything else off the whitelist."
else
    echo "Dry run only. Re-run with --apply to perform the copy."
fi
