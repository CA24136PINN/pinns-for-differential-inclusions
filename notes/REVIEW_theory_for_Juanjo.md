# Mathematical review of dr-pinns.tex — issues OUTSIDE Section 6

Reviewer: Kris (per Paco's division of labour, only Juanjo edits the theory
part of the .tex; the items below are therefore reported, not fixed).
Line numbers refer to the 2026-07-22 draft.

## Errors (should be fixed before submission)

1. **Definition of lower semicontinuity is wrong (Sec. 2.1, ~l. 324).**
   Current text: "for every y in F(x) and every neighborhood U of y, there
   exists a neighborhood V of x such that F(x) ∩ V ≠ ∅ for all x ∈ V."
   Two problems: (i) the intersection must be with U (a subset of Y), not
   with V (a subset of X) — as written it is type-incorrect; (ii) the bound
   variable x is reused. Correct statement: "... there exists a neighborhood
   V of x such that F(x') ∩ U ≠ ∅ for every x' ∈ V."

2. **Attainment of the distance is false in general normed spaces
   (Sec. 2.2, ~l. 385).** "If C ⊆ X is a closed set, then the infimum ...
   is always attained" — false in infinite-dimensional normed spaces
   (nearest points to closed, even closed convex, sets need not exist
   without reflexivity/proximinality). Since the paper immediately
   specializes to R^n, the simplest fix is to state attainment for
   nonempty closed sets in finite-dimensional spaces (or add
   "boundedly compact C").

3. **Hard-coded citation numbers "[13]" and "[14, Chapter VIII]"
   (Sec. 4.3, proof of Theorem consistency, ~l. 1180).** The text reads
   "(\cite{berkovitz_1974}; see also Ioffe [13] and [14, Chapter VIII])".
   With the current elsarticle-num bibliography, [13] is Lions–Magenes and
   [14] is Evans — i.e. the numbers point at the wrong works and will drift
   with any bibliography change. Replace with proper \cite commands (the
   intended references appear to be Ioffe's lower-semicontinuity paper(s),
   e.g. SIAM J. Control Optim. 15 (1977) 521–538, and a monograph such as
   Ekeland–Temam or Cesari for the Chapter-VIII pointer) and add the
   entries to references.bib.

4. **Variable mismatch in Theorem 3.? (General consistency, ~l. 752).**
   Hypothesis says "u_n → x in X", conclusion says "D(u) ∈ F(u)"; the limit
   should be called u throughout. In the proof: "by appling" → "by
   applying"; "with with" → "with".

## Imprecisions / presentation (recommended)

5. **Gradient formula in the abstract remark (Sec. 3, ~l. 700).** The
   displayed "∇R(u) = 2(D(u) − Π(D(u)))" is the derivative of
   v ↦ dist²(v, F(u)) evaluated at v = D(u), not the derivative of
   u ↦ R(u) (which also sees the u-dependence of F(u) and of D). Suggest
   writing ∇_v dist²(v, F(u))|_{v=D(u)} as done correctly in Sec. 4.1, and
   noting that in a general Banach space Y this requires Hilbert (or
   smooth-norm) structure; in the paper's concrete settings Y is L² or R^d,
   so no issue there.

6. **Pointwise vs. global residual notation (Sec. 3 definition).** R(u) is
   defined as a single scalar dist²(D(u), F(u)), but the loss is written
   (1/N) Σ R(u_θ)(z_i), which silently reinterprets R(u) as a pointwise
   residual field z ↦ dist²(D(u)(z), F(u)(z)). Suggest defining the
   pointwise residual explicitly.

7. **"Since F(t,x) is compact, the projections are unique" (Lemma normal
   integrand, DI case).** Compactness gives attainment; uniqueness comes
   from convexity + strict convexity of the norm (stated one line later
   anyway). Also "for all k ∈ N" should be "for all n ∈ N"; the proof
   refers to parts (a)–(c) while the statement numbers them 1.–3.

8. **Triangle inequality typo (Theorem consistency proof, DI case,
   ~l. 1120).** "|x_n(t)| = |x_0| + |x_n(t) − x_0|" — the "=" must be "≤".
   The subsequent Grönwall argument is unaffected.

9. **Muddled justification of weak derivative convergence (same proof).**
   "because the weak limit is determined by the bilinear form, and strong
   convergence of x_n, by Rellich–Kondrachov, forces the derivative
   component to converge weakly" — the standard one-line argument is:
   weak convergence in W^{1,2} implies ẋ_n ⇀ (x*)' in L² by testing
   against C_c^∞ functions (∫ ẋ_n φ = −∫ x_n φ̇ → −∫ x* φ̇).
   Rellich–Kondrachov is not needed for this step.

10. **AC domain typo (Sec. 2.3, ~l. 480).** "absolutely continuous
    functions on [0,T] × R^d" → "on [0,T] with values in R^d".

11. **Initial-condition well-posedness (Sec. 2.3, ~l. 545).** "The compact
    embedding into C([0,T];L²) ensures ... u(0,·) = u_0 is well defined" —
    it is the *continuous* embedding into C([0,T];L²) (indeed into
    C([0,T];H_0^1)) that gives this; compactness is used only later for
    the convergence argument.

12. **∀t vs a.e. t (Sec. 5, Definition of the admissible class A).** The
    admissible class imposes the boundary trace condition "for all
    t ∈ [0,T]" while Definition strong-solution uses "a.e. t"; for
    u ∈ W(0,T) the trace is defined a.e., so "a.e." is the consistent
    choice in both places.

13. **Existence proof citation (Sec. 2.3, Theorem existence-pde).** The
    proof invokes "standard existence theory for semilinear parabolic
    equations (see, e.g., [Evans, Ch. 7])" for a merely *continuous* f with
    linear growth. Evans Ch. 7 treats linear equations (semilinear with
    Lipschitz f appears in §9.2). For continuous f a
    compactness/Schauder-type fixed-point argument (or mollification of f)
    is the standard route; suggest adjusting the citation or adding one
    line of proof sketch.

## Checked and found correct

- Lemma closed-graph and its proof; general consistency theorem modulo
  item 4.
- Lemma normal-integrand (both DI and parabolic versions): convexity, lsc,
  and measurability arguments are sound.
- Theorem consistency (DI): growth/Grönwall/L² bounds, Arzelà–Ascoli,
  weak-compactness, and the Berkovitz/Ioffe lower-semicontinuity step are
  correct.
- Theorem consistency-parabolic: energy estimate, Grönwall, maximal L²
  parabolic regularity, compact embedding, closedness of the Dirichlet
  subspace under weak limits, and the lsc step — all correct.
- Section 6.3 relay example: reference splitting scheme and its reported
  extinction times t* = 0.1845 / 0.1645 / 0.1505 were independently
  recomputed and are dt- and grid-robust; energy identity, (P1)–(P3)
  verification (after the small (P2) fix), and all table arithmetic check
  out. Sign errors in the maximal-monotone remarks were fixed by Kris in
  Section 6.
