"""Audit paper/figures against the manuscript and the unified figure style.

Checks, for every figure referenced by paper/dr-pinns.tex:

  1. the PNG exists in paper/figures/;
  2. it was exported at the unified paper resolution (dpi = 180, see
     code/src/paper_style.py) -- figures still at dpi = 150 predate the
     style unification and must be regenerated;
  3. conversely, no orphan PNGs sit in paper/figures/ without being
     referenced by the manuscript.

Run from the repository root:

    python scripts/check_figures.py

Exit code 0 iff everything is consistent.
"""

import os
import re
import sys

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required (pip install pillow).", file=sys.stderr)
    sys.exit(2)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(REPO, "paper", "dr-pinns.tex")
FIGDIR = os.path.join(REPO, "paper", "figures")
UNIFIED_DPI = 180
DPI_TOL = 1.0  # matplotlib writes e.g. 180.0098


def referenced_figures():
    with open(TEX, encoding="utf-8") as f:
        tex = f.read()
    # \includegraphics[...]{figures/xxx.png}
    return sorted(set(re.findall(r"\\includegraphics\[[^\]]*\]\{figures/([^}]+)\}", tex)))


def main():
    refs = referenced_figures()
    on_disk = sorted(f for f in os.listdir(FIGDIR) if f.lower().endswith(".png"))
    ok = True

    print(f"Manuscript references {len(refs)} figure(s); "
          f"{len(on_disk)} PNG(s) in paper/figures.\n")
    print(f"{'figure':<42} {'exists':>6} {'dpi':>8} {'unified style':>14}")
    for name in refs:
        path = os.path.join(FIGDIR, name)
        if not os.path.exists(path):
            print(f"{name:<42} {'NO':>6} {'-':>8} {'-':>14}")
            ok = False
            continue
        dpi = Image.open(path).info.get("dpi", (None, None))[0]
        unified = dpi is not None and abs(dpi - UNIFIED_DPI) <= DPI_TOL
        print(f"{name:<42} {'yes':>6} "
              f"{(f'{dpi:.0f}' if dpi else '?'):>8} "
              f"{('OK' if unified else 'STALE (regen!)'):>14}")
        if not unified:
            ok = False

    orphans = [f for f in on_disk if f not in refs]
    if orphans:
        ok = False
        print("\nOrphan PNGs in paper/figures (not referenced by the manuscript):")
        for f in orphans:
            print(f"  - {f}")

    print("\n" + ("ALL CHECKS PASSED." if ok else
                  "PROBLEMS FOUND: see 'STALE'/'NO'/orphans above. "
                  "Regenerate via scripts/reproduce.sh, then re-run this check."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
