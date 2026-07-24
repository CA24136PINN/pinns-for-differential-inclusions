# DR-PINNs — poprawki Example 6.1 i 6.3 (odpowiedź na recenzję)

## Co jest w pakiecie

```
dr-pinns.tex                 poprawiony LaTeX (sekcje 6.1, 6.2, 6.3)
src/run_all.sh               SAMOGRAJ: wszystkie eksperymenty + figury + PDF
src/run_experiment_61.py     jednoprzebiegowy pipeline Example 6.1
src/run_experiment_62.py     jednoprzebiegowy pipeline Example 6.2
src/run_experiment_63.py     jednoprzebiegowy pipeline Example 6.3
src/sync_figures.sh          kopiuje 12 figur papera z results/ do figures/
src/paper_style.py           wspólny styl figur (bez zmian)
generated_clean/             pusty katalog `generated/` do finalnego runu
```

Katalogi `generated/`, `results/`, `checkpoints/` w tym pakiecie pochodzą ze
**smoke testu** (kilkadziesiąt epok) — służą wyłącznie weryfikacji, że
pipeline działa end-to-end. Do submission trzeba je nadpisać pełnym runem.

## Mechanizm synchronizacji (kluczowa zmiana strukturalna)

Recenzent dwukrotnie wskazał desynchronizację tekst/wykres/notebook (6.1:
loss 0.21→1.2e-12 vs 0.527→2.7e-15, d_H 0.0445 vs 0.0416, czasy scalability;
6.3: Figure 11 z innego uruchomienia niż tekst). Zamiast ręcznie przepisywać
liczby, każdy skrypt w **jednym udokumentowanym uruchomieniu** generuje:

1. wszystkie figury sekcji,
2. `generated/results_6x.tex` — każda liczba cytowana w tekście jako makro
   LaTeX (`\LCMeanDH`, `\RelayValLoss`, …),
3. `generated/manifest_6x.json` — seedy, wersje bibliotek, sprzęt, czasy,
   wszystkie raportowane wartości,
4. (6.3) checkpointy wag — Figure 11 jest odtwarzalna bit-po-bicie.

`dr-pinns.tex` wczytuje te pliki przez `\InputIfFileExists`; gdy ich brak,
fallbacki `\providecommand` renderują czerwone `[TBD: run 6.x]` — żadna
przeterminowana wartość nie może zostać w tekście. Czasy referencyjne
solvera 6.3 (0.1845/0.1645/0.1505) są deterministyczne i zostały
zweryfikowane niezależnie, więc mają zwykłe fallbacki.

## Mapowanie poprawek na punkty recenzji

### Example 6.1
- **Integrator (blokujący):** każdy odcinek stałego sterowania jest całkowany
  na dokładnym przedziale `[t_j, t_{j+1}]` z `dense_output`; stan przekazywany
  dalej to stan dokładnie w chwili przełączenia. Dodatkowo każda trajektoria
  jest porównywana z zamkniętą propagacją macierzowo-wykładniczą
  `x(t_{j+1}) = e^{AΔ}x(t_j) + A^{-1}(e^{AΔ}-I)Bu`; run przerywa się przy
  odchyleniu > 1e-7 (obserwowane ~5e-10). Weryfikacja niezależna: stary
  schemat vs. exact daje max ~0.046, mean ~0.009 — ten sam rząd co 0.0445,
  zgodnie z diagnozą recenzenta.
- **0.0445:** usunięte; `\LCMeanDH` z pełnego runu.
- **150 vs 200 punktów:** ujednolicone na 200 (paper i kod).
- **Walidacja dense-grid:** nowy akapit; mean/max inclusion distance na 5000
  losowych czasów poza siatką kolokacyjną, dla każdego członka ensemble
  (najgorsze wartości raportowane makrami).
- **Scalability:** warm-up (odrzucany) + 5 powtórzeń na konfigurację,
  mean±std, error bary na wykresie, metodologia opisana w tekście, sprzęt w
  manifeście. Wartości K̃ w tabelach (6,…,6 i 12,23,32,32) odtwarzają się
  deterministycznie ze seedów — smoke test to potwierdził.

### Example 6.2
- Poprawki narracyjne z recenzji: „at every evaluation point” → „at every
  time-grid node”; „strictly below ℓ=1 at all times” → „below ℓ=1 on the
  dense validation grid”; analogicznie podpis figury selektora.
- **Brak źródeł w pakiecie:** oryginalny pakiet zawierał tylko artefakty
  6.2 (checkpoint .pt, metrics.json, history.csv) bez kodu.
  `run_experiment_62.py` odtwarza pełny eksperyment z opisu w paperze:
  * **Część A** — optymalizacja selektora hard-admissible: 24 węzły
    kontrolne, interpolacja liniowa na siatkę n_t=120, RK4, kontynuacja
    λ_ξ∈{1e-3,1e-2,1e-1} z warm-startem, L-BFGS-B z gradientem
    analitycznym (TF autograd przez cały rollout RK4). Do tego a-posteriori
    check dopuszczalności na 20 001 punktach pośrednich i sonda
    ekstremalna −e₂ (wartość x₂(T)≈−0.05 z tekstu jest teraz liczona,
    nie deklarowana).
  * **Część B** — companion DR-PINN: ansatz endpoint-hard, czysty rezidual
    odległościowy, projekcja na elipsę Newtonem (30 iteracji, równanie
    sekularne), **cały rzut trzymany jako stała w backward pass**
    (dokładny gradient envelope; check FD vs autograd raportowany makrem),
    float64, 7000 epok Adam hold-then-decay, checkpoint .weights.h5.
  * Wszystkie liczby sekcji (tabela kontynuacji 3×3, dense check, sonda
    ekstremalna, 6 metryk companion) przeniesione na makra
    `\Ellip...` — ten sam mechanizm co 6.1/6.3.
  * Stary artefakt torch (drpinn_companion_seed0.pt) jest zastąpiony
    implementacją TF — jeden spójny zestaw zależności dla całego pakietu
    (numpy, scipy, tensorflow, matplotlib).

### Example 6.3
- **Φ_ε (Opcja A):** trening używa dokładnego Φ (ε=0; gałąź przedziałowa
  tylko przy dokładnym zerze zmiennoprzecinkowym — zdarzenie miary zero dla
  gładkiego ansatzu). Akapit o Φ_ε zastąpiony jawnym wzorem na dist²(z,Φ(s))
  i **testem wrażliwości po treningu**: rezidual przetrenowanej sieci
  przeliczany na tym samym batchu walidacyjnym dla ε∈{0,1e-8,1e-6,1e-4};
  maksymalne względne odchylenie raportowane makrem (`\RelayEpsSens`).
  Uwaga: dla ε=1e-4 odchylenie może być niezerowe, bo plateau po extinction
  ma rząd 1e-4 — dlatego zdanie w paperze raportuje wartość, nie przesądza
  „insensitive”.
- **Figure 11:** generowana z tej samej sieci (i zapisanego checkpointu), z
  której pochodzą wszystkie liczby sekcji; TODO w podpisie usunięte, w
  podpisie dodane zdanie o pochodzeniu z tego samego runu.
- **Tabela i narracja:** wszystkie wartości sieciowe na makrach z jednego
  runu; wartości referencyjne zweryfikowane. Zdanie o „growing modestly with
  λ” przeformułowane tak, by nie zależało od monotoniczności błędów w
  konkretnym rerunie (argument λ² zostaje jako oczekiwanie).

## Jak wykonać finalny run

Samograj (wszystko jedną komendą):

```bash
cd src
./run_all.sh              # 6.1 + 6.2 + 6.3 + figury + pdflatex + check TBD
./run_all.sh --smoke      # szybki test end-to-end całego pipeline'u (~5 min)
```

`run_all.sh` kończy się kodem !=0, jeśli w PDF zostały znaczniki `[TBD:]`
(niepełna synchronizacja) — nadaje się do CI. Orientacyjne czasy pełnego
runu: 6.1 ~30–60 min (CPU), 6.2 ~10–15 min (CPU), 6.3 — 4 treningi po 40k
epok (GPU zalecane; na CPU wiele godzin). Eksperymenty można też odpalać
pojedynczo (`python run_experiment_6x.py [--smoke]`).

## Status po tych zmianach względem recenzji

- Matematyka: bez zmian (recenzja: zasadniczo poprawna).
- 6.1: wszystkie punkty blokujące zaadresowane w kodzie i tekście; liczby
  czekają na pełny run.
- 6.3: kwestia ε rozstrzygnięta (Opcja A), synchronizacja Figure 11
  strukturalnie zagwarantowana; liczby czekają na pełny run.
- 6.2: pełny kod źródłowy odtworzony (wcześniej w pakiecie były tylko
  artefakty), wszystkie liczby na makrach.
- Stare notebooki (`dr_pinn_linear_control_qp_quickhull.ipynb`,
  `dr_pinn_relay_parabolic_experiment.ipynb`) oraz artefakty torch z
  `results_rotating_ellipse/` są zastąpione skryptami — w pakiecie
  replikacyjnym warto je usunąć albo zostawić z adnotacją „superseded”,
  żeby nie wróciła desynchronizacja.
