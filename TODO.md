# TODO

> **Objectif** : meilleur renderer deep-zoom open-source en Rust. Référence
> algorithmique : **Fraktaler-3.1** (cf. `fraktaler-3-3.1/src/`,
> `docs/fraktaler-3-analysis.md`).

**État (2026-05-19)** : Moteur bytecode unifié P3.1 livré sur CPU + GPU.
Nucleus finder atom-domain (port `hybrid_period`) + `hybrid_size` (rotation
extraite de K) ajoutés. Uniformisation deep zoom > 1e308 corrigée. Goulot
actuel : **parité visuelle F3 sur le corpus `toml/`** + propager la matrice
K complète (scale/skew non-conformal) au pixel→c mapping.

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

---

## Up next (ordre d'attaque)

1. **Investigation fails Fractall** (e1086, opus)
   Probable GMP-per-pixel quand BLA ne couvre pas → timeout 60s. À profiler.
2. **P1.3 résidus** — décision `bailout` défaut (4 vs 25, change goldens)
   + vérifier pixel spacing BLA = `4/zoom/height` strict.
3. **P1.6.c** GPU+compile — élargir le buffer bytecode GPU pour porter le
   payload `Op::Rot` (cos/sin) et brancher `compile_formula` ou un loader
   TOML F3 pour émettre des `Op::Rot` réels. CPU déjà OK.
4. **P1.6.b-bis** — Stocker K complet (mat2) dans `FractalParams` et l'appli-
   quer dans le pixel→c mapping. Aujourd'hui on n'extrait que l'angle ;
   pour les minibrots hybrides non-conformes (BS, Tricorn), K a une partie
   skew/scale non-uniforme qu'il faut propager.
5. **P1.6.d + P1.6.e** — Wisdom file + float128 (bundle perf deep zoom).
   10-100× speedup pour zooms 1e30-1e100. Couple obligé : float128 sans
   dispatcher wisdom-driven n'a pas de sens.
6. **Porter `iterate_pixel_gmp` sur pixel_loop** → permet de retirer
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
- [ ] Stocker `K` (mat2 row-major) dans `FractalParams` à côté de `rotation`.
- [ ] Injecter K dans le pipeline (`engine.cc:235-236`) : mapping pixel→c
  via `K * (4/zoom)`, rayon BLA scalé via |det K|, period detection prend
  K en entrée.
- [ ] Pour les minibrots Mandelbrot conformes, K = (1/β²)·R(θ) ; on extrait
  déjà θ et on l'écrase dans `rotation`. Pour les hybrides non-conformes
  (BS, Tricorn), K a une partie skew/scale non-uniforme à propager fidèle-
  ment (sinon corruption sur flake / olbaid*).
- **Effort** : ~120 lignes (storage + pixel→c + BLA scaling).

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
