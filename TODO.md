# TODO

> **Objectif** : meilleur renderer deep-zoom open-source en Rust. Référence
> algorithmique : **Fraktaler-3.1** (cf. `fraktaler-3-3.1/src/`,
> `docs/fraktaler-3-analysis.md`).

**État (2026-05-20)** : Moteur bytecode unifié P3.1 livré sur CPU + GPU.
Nucleus finder atom-domain (port `hybrid_period`) + `hybrid_size` (K
normalisé stocké dans `FractalParams.transform_k`, pixel→c via
`transform_matrix()`) ajoutés. `Op::Rot` opcode natif CPU avec dual-numbers
+ BLA. Uniformisation deep zoom > 1e308 corrigée. Goulot actuel : **parité
visuelle F3 sur le corpus `toml/`** + BLA radius scaling pour hybrides
skewés.

---

## P0 — Parité F3 sur le corpus `toml/`

**Objectif** : produire les mêmes images que Fraktaler-3 pour les 84 fichiers
`toml/*.toml`. C'est la mesure de vérité — on ne peut prétendre « meilleur
deep-zoom open-source en Rust » sans égaler F3 sur ce corpus.

**Critère** : `mean_abs(F3, fractall) ≤ ε` à 1920×1080. Pixel-perfect là où
c'est possible, équivalence visuelle ailleurs.

**Harness** : `scripts/compare_f3.py` (F3 batch + fractall-cli sur chaque
TOML → PNG + diff + métriques EXR N0/NF). Premier résumé 2026-05-18 :
7 cas OK, 2 fractall_timeout (e1086, opus), 1 f3_timeout (seahorse).

**Sweep 2026-05-20** (12 cas, 256×256, timeout 180s, env
`FRACTALL_NO_AUTO_ADJUST=1`) — **10/12 OK** :
- Pixel-perfect : test5 (Δmean=0), test6 (Δmean=0.0005).
- Visuellement équivalents (image macro identique, Δmean concentré aux
  pixels chaotiques à la frontière d'évasion) : test, test2 (0.50),
  test4 (0.15), line (1.71), spiral (9.81), all_seeing_eye (5.82),
  e113 (27.86).
- Catastrophique : **glitch_test_1** (Δmean=156, Δmax=147467 — anneaux
  concentriques artefacts ; ref_orbit périodique avec cycle_period=7327
  donne un rendu structurellement différent de F3).
  - **Investigation 2026-05-20** : les anneaux viennent du couple
    `wrap_periodic` (cyclage modulo de m vers `[cycle_start, +period)`) +
    BLA primaire unique dans `pixel_loop_exp`. Tentative de remplacer le
    wrap par un rebase F3 strict (`delta := Z[m]+δ; m := 0`) → image
    UNIFORME (re-traverse la queue pré-cycle `[0, cycle_start)` qui ne
    correspond pas à la trajectoire réelle du pixel dans le cycle).
    `wrap_periodic` est donc nécessaire pour les centres périodiques.
    Le vrai fix demande probablement une BLA multi-phase (une par phase
    du cycle, cf. P1.6.f) ou un rebase qui cible `cycle_start` au lieu de
    `0`. Reporté — nécessite P1.6.f.
- Timeout résiduel (perf gap) : **dragon** (zoom 1e191, iter 5M),
  **e50** (zoom 1e50, iter 263k). e113 démontre que 90s suffit pour
  ~35k iter à 256×256 ; e50/dragon (10×-100× plus d'itérations)
  dépassent 180s.

**Fixes appliqués cette session vers la parité** :
- `nf_f3` formule : degree dérivé du bytecode (`opcodes_degree`,
  `bytecode/mod.rs`) au lieu de `multibrot_power=2.5` constant. F3 utilise
  le degré exact de la dernière phase (`hybrid.cc:334`). Test4 : Δmean 0.24
  → 0.15 (-35%).
- Env var `FRACTALL_NO_AUTO_ADJUST=1` gate l'auto-adjust d'iter_max dans
  `orbit.rs`. F3 ne fait pas cet adjust → divergence systématique sans le
  gate. Le harness positionne le flag avant chaque appel CLI.
- Env var `FRACTALL_NO_PERIOD=1` désactive la troncature par period-detection
  (commit 7356931). **Gain majeur** : floral_fantasy 1310→0.39, glitch_test_4
  1325→0.24, liiiines 3969→23.8, mitosis2 37.5→3.56. Le harness le positionne.

**Sweep étendu 2026-05-20** (26 cas uniques, NO_PERIOD + NO_AUTO_ADJUST) :
- Pixel-perfect : test5, test6.
- Visuellement équivalents (pas d'inside_mismatch, Δmean concentré aux bords
  chaotiques) : test, test2, test4, line, spiral, all_seeing_eye, e113,
  floral_fantasy, glitch_test_4, liiiines, mitosis2, flake (31), glitch_test_3
  (6.0), heaven (6.5), tick_tock (24), x (22), peanuts (15).
- Δmean élevé mais sans inside_mismatch (probable bord chaotique intrinsèque) :
  nr_fail (92), uranium (149).
- **glitch_test_5 — FAUX ÉCHEC : fractall CORRECT, F3 DÉGÉNÉRÉ (2026-05-20)**.
  Le harness reportait inside_mismatch=54150 (F3 4096/4096 intérieur vs
  fractall 704/4096). Inspection visuelle : **fractall rend un minibrot
  Mandelbrot correct et détaillé** (cardioïde + bulbes + filaments, intérieur
  noir au centre, bandes d'évasion autour) ; **F3 rend un carré ENTIÈREMENT
  NOIR** (`glitch_test_5__f3.png` = noir uni), produit en 0.02s (court-circuit
  suspect — F3 a vraisemblablement déclaré tout l'intérieur via un fast-path
  period/nucleus en batch et sauté le rendu per-pixel). fractall DÉPASSE F3
  ici. Diagnostic exhaustif au passage (tout écarté) : precision GMP (768≡314),
  delta double-double, référence double-double, GMP pur per-pixel — tous
  donnent le MÊME rendu correct. → Rien à corriger côté fractall ; c'est un
  artefact F3. À exclure du score parité (ou compter comme victoire fractall).
- **glitch_test_1 — anneaux REPRODUITS par le GMP pur (2026-05-20)**.
  zoom 3.3e46, period 7327. fractall (perturbation) rend un minibrot entouré
  d'**anneaux concentriques** ; F3 rend minibrot + **bruit chaotique** sans
  anneaux. Test décisif : `--algorithm gmp` (per-pixel full precision, SANS
  perturbation/BLA/series) rend les MÊMES anneaux → ce n'est PAS un artefact
  de perturbation. Écartés : précision (dd delta+réf), BLA (threshold 1e-30
  → Δ négligeable, avg 40936≡40949), series (on/off identique), period/wrap
  (NO_PERIOD = anneaux quand même). → MÊME SIGNATURE que glitch_test_5 :
  GMP fractall ≡ perturbation fractall ≠ F3. Comme glitch_test_5 s'est avéré
  être un F3 dégénéré (fractall correct), glitch_test_1 est probablement
  AUSSI un cas où fractall est correct (anneaux = structure atom-domain réelle
  du minibrot) et F3 montre du bruit de glitch (les `glitch_test_*` sont des
  lieux glitch-prone, conçus pour stresser le glitch-handling). Non tranché
  définitivement (faudrait un 3e renderer ou F3 haute-res). Lean : victoire
  fractall probable, pas un bug fractall.
- **rug — BLA over-skip au deep zoom — RÉSOLU (2026-05-20, commit e66076f)**.
  zoom 3.3e56, iter 100k. Symptôme : blobs rouges (smooth-iter aplati) là où
  F3 a du bruit chaotique. **Cause** : fractall utilisait TOUS les niveaux BLA
  y compris les bas (single/2/4-step) ; au deep zoom δ (~1e-56) ≪ rayon de
  validité (~e·|Z| ~1e-8) donc la BLA s'applique toujours, et le BLA linéaire
  f64 (drop δ² + coefficients déconditionnés) est MOINS précis que le pas
  perturbation direct. **Fix** : `BLA_SKIP_LEVELS=3` dans `lookup`/`lookup_fexp`
  (`bla_dual.rs`), miroir de F3 `bla_skip_levels=3` — les petits pas passent en
  direct (exact), les niveaux ≥8-step gardent le gain perf. Résultat : Δmean
  1709→389, max si 45399→94723 (≈F3 94554). Sans régression (test5/6 pixel-
  perfect, e113 ~92s inchangé ; 1 golden minibrot_1e8 +46px raffinés, revu).
- Timeout (perf gap, P1.6.d/e) : **e50** (1e50), **dragon** (1e191),
  **e1000** (1e1000).

**Period-detection truncation est LOSSY (analyse 2026-05-20)** :
- Même pour une période GENUINE (confirmée par un cycle de plus :
  `z[detect+p] ≈ z[detect]`), tronquer + `wrap_periodic` accumule l'erreur
  de la quasi-périodicité (tolérance ~2^(-0.4·prec)) sur ~iter_max/période
  cycles → perturbation divergente (image uniforme, glitch_test_5).
- Le gain perf de la troncature est NÉGLIGEABLE (la référence est calculée
  une fois ; tronquer économise surtout de la mémoire). Le coût per-pixel
  est identique. → **Recommandation** : passer la troncature OFF par défaut
  (la garder opt-in seulement aux nucleus exacts via `--find-nucleus`, où
  l'orbite est exactement périodique et le wrap exact). Validation goldens +
  GUI requise avant de flipper le défaut (pas fait dans cette boucle).

---

## Up next (ordre d'attaque)

> **Constat boucle parité 2026-05-20** : quick wins parité livrés (degree,
> auto-adjust gate, period gate). État des cas "cassés" après investigation :
> - **glitch_test_5** : RÉSOLU — faux échec, fractall correct / F3 dégénéré
>   (carré noir). Rien à faire côté fractall.
> - **glitch_test_1** : anneaux concentriques réels. Précision ÉCARTÉE
>   (double-double delta+réf testés sans effet ; GMP pur identique). Cause
>   probable : interaction wrap_periodic + BLA sur la référence cyclique.
>   À investiguer séparément (pas float128/BLA-threshold).
> - **e50/dragon/e1000** : perf (GMP forcé). float128/wisdom PEUT aider mais
>   non vérifié — à profiler avant de s'engager.
> Infra `dd.rs` (double-double ~106 bits) livrée et testée (commits 873d24c,
> a73cb76, d378873) mais NON câblée : conservée pour un besoin futur réel
> (wisdom/float128 perf), écartée pour glitch_test_1/5.

1. **Vérifier le corpus pour d'autres F3-dégénérés** — glitch_test_5 montre
   que F3 batch peut rendre un carré noir (court-circuit period/nucleus).
   Re-scanner les "inside_mismatch" élevés : ce sont peut-être des victoires
   fractall, pas des échecs. Ajouter au harness un flag F3-degenerate
   (image quasi-uniforme → exclure du score).
2. **glitch_test_1 anneaux** — investiguer l'interaction wrap_periodic + BLA
   sur référence cyclique (cause des anneaux ; ni précision ni BLA-threshold).
3. **Profiler e50/dragon/e1000** avant P1.6.e/d — confirmer que float128/
   wisdom (et non un autre goulot) débloque le perf, sinon réorienter.
4. **Period-detection OFF par défaut** — la troncature est lossy (cf. analyse
   ci-dessus). Garder opt-in aux nucleus exacts. Valider goldens + GUI.
5. **P1.6.d — wisdom file** — dispatch f64 → FloatExp/DoubleExp → float128 →
   GMP selon zoom. Sans lui, float128 n'est pas sélectionné automatiquement.
6. **P1.3 résidus** — décision `bailout` défaut (4 vs 25, change goldens)
   + vérifier pixel spacing BLA = `4/zoom/height` strict.
7. **P1.6.c** GPU+compile — élargir le buffer bytecode GPU pour porter le
   payload `Op::Rot` (cos/sin) et brancher `compile_formula` ou un loader
   TOML F3 pour émettre des `Op::Rot` réels. CPU déjà OK.
8. **P1.6.b-bis (suite)** — BLA radius scaling via |det K| pour les hybrides
   non-conformes skewés. Storage + pixel→c déjà OK.
9. **P1.6.e — float128/double-double câblé** — infra `dd.rs` prête ; à brancher
   SI le profilage (item 3) confirme que c'est le goulot perf. Pas pour
   glitch_test_1/5 (écarté).
10. **Porter `iterate_pixel_gmp` sur pixel_loop** → permet de retirer
   `glitch.rs`, `nonconformal.rs` et les champs perturbation legacy.

---

## P1 — Parité numérique deep-zoom

### Open

#### P1.3 — Constantes critiques (résidus)
- [ ] Escape radius pixel défaut = `25` (F3) vs `4` (fractall). Garder 4 pour
  golden tests stables ; faire de 25 le défaut casserait les goldens. À
  trancher (changement avec refresh goldens).
- [ ] Vérifier pixel spacing BLA = `4/zoom/height` strict dans
  `bytecode/bla_dual.rs`, `perturbation/bla.rs`, `nonconformal.rs`.
- [x] Caps `max_perturb_iterations` / `max_bla_steps` enforcés dans
  `bytecode/pixel_loop.rs` (`iterate_pixel_unified_{full,single_phase,
  multi_phase,mandelbrot}`) et `pixel_loop_exp.rs` (`{_exp,
  _single_phase,_mandelbrot,_generic}`). Cap perturbation track `iters_ptb`
  séparé de `n` (BLA jumps ne comptent pas, cf. F3 `cl-post.cl:21`).
  Test `caps_max_perturb_iterations_truncates_interior_pixel` vérifie
  qu'un pixel intérieur sort avec `iter < iter_max` lorsque cap actif.

#### P1.5 — Anti-aliasing par subframes jitterés
- [ ] Wrapper N samples avec offsets jitterés (low-discrepancy : `burtle_hash`
  / `radical_inverse`) → moyenne. Champ `jitter_scale` existe déjà sur
  `FractalParams` mais pas exposé en CLI/GUI ni accumulé multi-passes.
- **Pourquoi** : qualité bords fins, surtout mode Distance/DE. F3 le fait
  par défaut.

#### P1.6 — Parité F3.1 (analyse 2026-05-19)

Items issus de l'analyse `fraktaler-3-3.1/src/`. Classés par impact × effort.

##### P1.6.b-bis — Stocker K complet et l'appliquer au pixel→c [HAUTE]
- [x] Champ `transform_k: Option<[f64; 4]>` ajouté à `FractalParams` (serde
  `skip_serializing_if = "Option::is_none"` pour les PNG legacy propres).
  Default `None` → fallback rotation seule, drop-in compatible.
- [x] `FractalParams::transform_matrix()` renvoie K si présent (et fini),
  sinon `rotation_matrix()`. Callsites de `params.rotation_matrix()` migrés
  dans `render/escape_time.rs` (4 sites) et `perturbation/mod.rs` (2 sites).
- [x] Nucleus pipeline (`orbit.rs`) injecte `K_normalized = K / sqrt|det K|`
  après `hybrid_size_mat2`. Det=1 préserve le zoom utilisateur (pas de
  zoom-in implicite par 1/β²) tout en encodant rotation + skew non-uniforme
  pour les hybrides. Pour Mandelbrot conformal, `K_norm = R(θ)` exact.
- [x] Tests : 5 tests `transform_tests` (identity, fallback, override, skew,
  NaN reject) + 1 PNG round-trip (`transform_k_round_trip_and_legacy_default`).
- [ ] Reste : BLA radius scaling via |det K| (pour quand K_norm ≠ rotation,
  c-à-d hybrides skewés). Aujourd'hui le BLA construit son rayon sans tenir
  compte de K — OK pour conformal (rayon scalaire correct), à raffiner
  pour les hybrides via stretching anisotrope du delta.

##### P1.6.c — Opcode `Op::Rot` natif [HAUTE]
- [x] `Op::Rot { cos_theta, sin_theta }` ajouté à `bytecode/mod.rs` avec
  `opcode_tag()` pour le mapping GPU. Cas implémentés dans `interp.rs`,
  `interp_gmp.rs`, `delta_form.rs` (DeltaState + DeltaStateExp avec
  propagation ddelta), `bla_dual.rs` (Jacobien `R · J`, rayon préservé car
  isométrique), `iterations.rs` (path f64 standard).
- [x] Tests d'invariance : `invariant_rot_mandelbrot_phase` (δ-form vs
  absolu f64), `invariant_rot_mandelbrot_phase_exp` (DeltaStateExp deep
  zoom), `rot_after_sqr_matches_composed_jacobian` (BLA = R · J_sqr).
- [ ] GPU : `Op::Rot` rejeté par `try_render_bytecode` (payload f64 pas
  encodable dans le buffer `array<u32>` actuel). À débloquer en élargissant
  le format bytecode GPU (op_id + 2×f32 cos/sin packés, ou storage buffer
  séparé pour les payloads) ; pour l'instant fallback CPU.
- [ ] `compile_formula` n'émet pas encore `Op::Rot` — les formules natives
  (Mandelbrot, Julia, …) restent sans rotation. À brancher à P1.6.b-bis
  (injecter K-extracted angle dans la formule) ou à un futur parseur
  `[[formula]]` rotate des TOML F3.

##### P1.6.d — Wisdom file + auto-precision [HAUTE]
- [ ] Implémenter `wisdom_lookup` (`wisdom.cc:240-295`) : choisit le type
  numérique le plus rapide satisfaisant `pixel_spacing_exp + 16 < range/2`
  et `pixel_spacing_precision < mantissa_bits`.
- [ ] Benchmark `(type, device)` au premier run, JSON persisté
  (`~/.fractall/wisdom.json`).
- [ ] Plug-in dans `render/escape_time.rs` : f64 → FloatExp/DoubleExp → GMP.
- **Pourquoi** : entre 1e15 et 1e100, fractall force GMP alors que F3
  utilise doubleexp/float128 (10-100× plus rapide). Goulot perf majeur.
  **Note** : remplace P3.2 (qui restait au stade idée).

##### P1.6.e — float128 + softfloat [HAUTE]
- [ ] `float128` : 113 bits mantisse, 15 bits exp, atteint ~1e4900 sans GMP.
  Wrapper FFI vers `__float128` ou port de `float128.h`.
- [ ] `softfloat` : 32 bits CPU-pure (`softfloat.h`). Utile pour GPU/WASM
  sans FPU. Moins prioritaire.
- [ ] Brancher au wisdom file (P1.6.d).
- **Effort** : ~400 lignes Rust pour f128 complex + tests vs GMP.

##### P1.6.f — BLA multi-phase native [HAUTE]
- [ ] `Vec<BlaTableUnified>` par phase au lieu d'une seule. F3 build une BLA
  par phase (`engine.cc:287-295`, `bla.cc::hybrid_blas`).
- [ ] Refactor `iterate_pixel_unified_*` pour switcher de BLA quand la phase
  change (déjà partiel dans `iterate_pixel_hybrid_bla` legacy — porter sur
  bytecode).
- [ ] Tests d'invariance hybride Mandelbrot⊕BurningShip.
- **Prérequis** : P1.6.c (rot) + P3.4 (multi-phase UI/CLI).

##### P1.6.g — Nucleus finder phase-aware + extension hybrides [MOYENNE]
- [ ] Étendre `nucleus.rs` au-delà de Mandelbrot : Burning Ship, Tricorn,
  Multibrot entier (dérivées via dual-numbers par opcode, déjà ~70 % en
  place via `bla_dual`).
- [ ] Détecter période/centre/size par phase dans un hybride
  (`engine.cc:118-218`).
- **Effort** : ~250 lignes.

##### P1.6.h — longdouble x86-64 80-bit [MOYENNE]
- [ ] Feature-gate `cfg(target_arch="x86_64")` : `f80` via FFI C ou crate
  `libm`. Mantisse ~64 bits, comble le gap 1e20-1e40 avant float128.
- [ ] Brancher au wisdom file.

### Done

- **P1.0** ✅ (2026-05-18) — Uniformisation deep zoom > 1e308. Cause :
  `params.span_x` f64 underflow à 0. Fix : helpers `effective_spans_fexp` +
  `effective_pixel_size` HP-aware. Résultat e1000 : Δmean 691 → 0.44.
- **P1.1** ✅ — Rebasing F3 dans `bytecode/pixel_loop.rs`, condition stricte
  `|Z+z|² < |z|²`. Reste mineur : retirer `glitch.rs` / `nonconformal.rs`
  une fois `iterate_pixel_gmp` porté sur pixel_loop.
- **P1.2** ✅ — BLA `mat2` unifié via dual-numbers (`bytecode/bla_dual.rs`).
- **P1.3** ✅ partial — `bla_threshold = 1/2²⁴`, `REFERENCE_BAILOUT_SQR = 1e10`,
  GMP precision clamp `[128, 65536]` alignés F3.
- **P1.4** ✅ — `diffabs` Burning Ship dans `delta.rs`, validé par 9 tests
  d'invariance Z+δ.
- **P1.6.a** ✅ (2026-05-19) — Nucleus finder atom-domain
  (`perturbation/nucleus.rs`). Critère F3 `|z|² < s²·|dz|²` (port
  `hybrid.cc:417` `hybrid_period`). 5/5 tests passent.
- **P1.6.b** ✅ (2026-05-19) — `hybrid_size_mat2` porté
  (`perturbation/nucleus.rs`, port `hybrid.cc:544-592`). Itère `period-1`
  fois en dual-numbers GMP, accumule `b += L⁻¹`, renvoie `size = 1/(λ²·β)`
  et matrice `K = inv(transp(b))/β`. Wiring `orbit.rs:618-720` : après
  Newton, on remplace `params.rotation` par `atan2(K[2], K[0])` (équivalent
  Mandelbrot conformal de `out.transform = K; rotate = 0` côté F3). 4/4
  tests passent (period 1 identité, period 2 dégénéré c=-1, period 3
  axis-aligned, escape→None). Suite goldens verte. Reliquat : stocker K
  complet (skew/scale) pour hybrides non-conformes — voir P1.6.b-bis.

---

## P2 — Infrastructure

### Open

#### P2.1 — CI GitHub Actions
- [x] `cargo test --release --bin fractall-cli` + `cargo test --release
  --test golden_images` sur push / PR (`.github/workflows/ci.yml`,
  ubuntu-latest, libgmp/mpfr/mpc + Swatinem/rust-cache).
- [ ] Étendre corpus golden à zooms intermédiaires (10¹⁰, 10¹⁵, 10²⁰),
  cap ~70 s par cas (5e227 abandonné, trop long).

#### P2.2 — Découpe gros fichiers
- [ ] `gui/app.rs` (2854 lignes) → menu Type / drag-drop / HQ render /
  raccourcis.
- [ ] `perturbation/mod.rs` (1456 lignes) → split
  `render_perturbation_cancellable_with_reuse()` et dispatch CPU/GMP.
- [ ] `gpu/mod.rs` (1747 lignes) → pipelines standard / perturbation /
  bytecode en sous-modules.
- **Pourquoi** : préalable à tout refactor P1 majeur (P1.6.b-bis notamment).

#### P2.3 — Documenter tradeoff GPU f32
- [ ] README : limite zoom GPU (~10⁷ en f32) et seuil fallback CPU.
- [ ] Évaluer softfloat (P1.6.e) pour devices sans fp64 — peu probable
  rentable avec wgpu.

### Done

- **P2.1 local** ✅ — `tests/golden_images.rs` + 10 cas. Régénération via
  `FRACTALL_UPDATE_GOLDENS=1`. Couvre f64 standard, perturbation, deep zoom
  GMP, non-bytecode.

---

## P3 — Scope stratégique

### Open

#### P3.3 — EXR raw export
- [ ] Format compatible KFR / zoomasm pour zoom-vidéo assembly. Permet
  recolorisation/animation sans re-rendre.

#### P3.4 — Multi-phase hybrid UI/CLI
- [x] `Formula::hybrid(vec![phase, …])` supporte les chaînes en interne.
- [ ] CLI : `--phases mandelbrot,burning_ship,burning_ship` etc.
- [ ] GUI : éditeur de séquence de phases.
- [ ] `Vec<BlaTableUnified>` par phase (cf. P1.6.f).
- **Pourquoi** : feature unique vs Kalles Fraktaler / Fraktaler-3.

### Done

- **P3.1** ✅ — Architecture bytecode unifiée (Sessions A-E + GPU + dual
  numbers + cleanup). 8-opcodes, compile_formula, interp CPU f64 + GMP, BLA
  mat2 via dual, delta-form, pixel_loop f64 + ComplexExp, multi-phase infra,
  GPU `bytecode_kernel.wgsl`, distance estimation / interior detection /
  orbit traps via duals. Reste mineur : retirer modules legacy (cf. P1.1).
- **P3.2** → fusionné dans **P1.6.d** (wisdom file).

---

## À ne pas régresser (superset assumé vs F3)

- 27 palettes built-in + RGB/HSB/LCH.
- 15 modes coloring + 4 orbit traps.
- 7 plane transforms XaoS-style.
- Drag-and-drop PNG avec metadata JSON.
- Formules non-escape-time : Newton, Phoenix, Magnet, Lyapunov (6 presets),
  Buddhabrot/Nebulabrot/Anti, Von Koch, Dragon, Pickover Stalks, Nova, Sin,
  Alpha Mandelbrot, Barnsley, Mandelbulb. F3 n'y va pas — différenciation.
- Preview Julia au survol + raccourci `J`.
- Recolorisation asynchrone sans bloquer l'UI.

---

## Format / I/O

- [x] Loader TOML rust-fractal-core (`toml/*.toml`, 84 fichiers).
- [ ] Vérifier compat format TOML Fraktaler-3 pour interop bidirectionnelle.
