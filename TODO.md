# TODO — Roadmap fractall vers l'excellence

> **Mission** : le meilleur renderer de fractales **deep-zoom open-source en Rust**.
> **Référence algorithmique** : **Fraktaler-3.1** (`fraktaler-3-3.1/src/`,
> `docs/fraktaler-3-analysis.md`). Quand fractall diverge de F3, c'est fractall
> qu'on corrige — sauf preuve que F3 est dégénéré sur le cas.
> **Différenciation assumée** : superset de F3 côté *features* (27 palettes,
> 15 modes de coloring, 7 plane transforms, fractales non-escape-time, GUI
> interactive, drag-drop PNG). Voir [§Ne pas régresser](#-ne-pas-régresser).

**État (2026-05-20)** — Socle solide :
- Moteur **bytecode unifié** (8 opcodes) sur CPU + GPU : un seul pixel-loop
  couvre Mandelbrot/Julia/BurningShip/Tricorn/Celtic/Buffalo/PerpBS/Multibrot
  + variantes Julia.
- **Perturbation** + BLA `mat2` (dual-numbers) + **rebasing F3** + FloatExp
  (`ComplexExp`) pour zoom > 1e13, GMP pour l'orbite référence.
- **Nucleus finder** atom-domain (`hybrid_period` + `hybrid_center` +
  `hybrid_size` → K dans `transform_k`).
- Coordonnées **HP > 1e308**, escape radius **aligné F3 (25)**, **BLA
  pixel-spacing corrigé**, **anti-aliasing multi-sample**, **rotation GPU
  corrigée**.

**Goulots restants** (les 2 vrais chantiers) :
1. **Parité visuelle F3** sur le corpus `toml/` (mesurer, élucider les
   divergences restantes) → **G1**, **G3**.
2. **Performance deep-zoom 1e15–1e1000** : on force GMP là où float128 /
   doubleexp seraient 10–100× plus rapides → **G2**.

---

## ⭐ Définition de l'excellence (critères mesurables)

Fractall sera « excellent » quand les 5 piliers sont atteints :

1. **Correction** — pixel-équivalent à Fraktaler-3 sur les **84** configs
   `toml/*.toml` à 1920×1080 (`mean_abs ≤ ε`, divergence résiduelle concentrée
   *uniquement* aux bords chaotiques intrinsèques). **Zéro sortie fausse
   silencieuse** sur toute combinaison path × feature.
2. **Performance** — deep zoom 1e15–1e1000 à **≤ ~2× le wall-clock de F3**.
   Aucun path ne force GMP là où float128/doubleexp/longdouble suffit.
3. **Robustesse** — aucun artefact (anneaux, blobs, glitches) ; rendu correct
   au-delà de 1e308 ; golden tests déterministes **verts en CI**.
4. **UX & différenciation** — features hors-F3 sans régression ; GUI fluide
   (progressif + recolorisation async + AA).
5. **Maintenabilité** — aucun fichier > **~800 lignes** ; paths perturbation
   *legacy* retirés ; un seul moteur (bytecode).

---

## 🎯 Goals

### G1 — Parité visuelle F3 sur le corpus `toml/` · `[P0 · métrique de vérité]`

**Objectif** : produire les mêmes images que F3 pour les 84 `toml/*.toml`.
**Harness** : `scripts/compare_f3.py` (F3 batch + `fractall-cli` → PNG + diff +
métriques EXR N0/NF). Flags de parité : `FRACTALL_NO_AUTO_ADJUST=1`,
`FRACTALL_NO_PERIOD=1` (F3 ne fait ni l'auto-adjust d'iter_max ni la troncature
par période).
**État** : dernier sweep (26 cas uniques, 256²) → majorité visuellement
équivalents (Δmean concentré aux bords). Pixel-perfect : test5, test6.

**Done when** :
- [ ] **Re-sweep complet** des 84 cas à 1920×1080 : rapport `mean_abs` +
  classification par cas (pixel-equiv / bord-chaotique / F3-dégénéré / perf).
- [ ] **Quantifier le gain ER=25** : l'escape radius a changé (4→25) sans
  re-sweep ; mesurer la baisse de Δmean à la frontière d'évasion.
- [ ] **Détecteur F3-dégénéré** dans le harness : image quasi-uniforme rendue
  en < 0.1 s ⇒ exclure du score (cf. glitch_test_5, §G3).
- [ ] Chaque cas restant : pixel-équivalent **ou** divergence expliquée et
  classée (pas de FAIL non élucidé).

### G2 — Performance deep-zoom : dispatch wisdom + types intermédiaires · `[P0 · perf]`

**Problème** : entre ~1e15 et ~1e1000, fractall reste sur le path `ComplexExp`
(FloatExp normalisé via `frexp` à chaque op, ~35 ns/iter, ~6× le f64) ou force
GMP, là où F3 utilise doubleexp/float128/longdouble (10–100× plus rapide).
Cas symptomatiques (timeout > 180 s à 256²) : **e50** (1e50, 263k iter),
**dragon** (1e191, 5M iter), **e1000**, e1121/e1200/golden_spider/leaded_glass/
test3/virus/windmill. On NE PEUT PAS retomber sur f64 pur (`z_ref + δ` collapse
quand `|δ| < 2.2e-16·|z_ref|`, cf. seuil `PIXEL_SIZE_EXP_THRESHOLD=1e-13`,
`delta.rs:89`).

**Done when** (3 leviers, du plus rentable au moins) :
- [ ] **Wisdom file + auto-precision** (port `wisdom.cc:240-295`). Choisit le
  type le plus rapide satisfaisant `pixel_spacing_exp + 16 < range/2` et
  `pixel_spacing_precision < mantissa_bits`. Benchmark `(type, device)` au 1er
  run, JSON persisté (`~/.fractall/wisdom.json`). Plug-in dans
  `render/escape_time.rs` : f64 → FloatExp/DoubleExp → float128 → GMP. *Sans
  lui, aucun type intermédiaire n'est sélectionné automatiquement — c'est le
  débloqueur des 2 autres leviers.*
- [ ] **float128** (113 bits mantisse, exp 15 bits → ~1e4900 sans GMP). Wrapper
  FFI `__float128` ou port de `float128.h`. ~400 lignes + tests vs GMP.
  Optionnel : `longdouble` x86-64 80-bit (gap 1e20–1e40) ; `softfloat` 32-bit
  CPU-pure (GPU/WASM sans FPU).
- [ ] **Boucle interne f64-scaled** (alternative/complément) : delta f64 +
  exposant partagé rescalé périodiquement (façon rust-fractal-core) ; ops
  internes f64 pures, bailout/rebase en FloatExp seulement. Réécrit le hot-loop
  exp.
- **Acceptation** : e50/dragon/e1000 rendus en < 180 s à 256² ; profil confirme
  que le path choisi domine (pas un autre goulot).

> ⚠️ **Profiler avant de coder** : confirmer sur e50/dragon que c'est bien
> l'arithmétique exp (et non orbit/BLA) qui domine, sinon réorienter.
> Infra `dd.rs` (double-double ~106 bits) déjà livrée et testée (`ComplexDDExp`,
> commits 873d24c/a73cb76/d378873), **non câblée** — candidate pour le wisdom.

### G3 — Élucider les divergences ouvertes · `[P0 · correction]`

Trois divergences restent non closes. Toutes pointent probablement vers
« fractall correct / F3 ou harness en cause », mais doivent être tranchées.

- [ ] **glitch_test_1 — anneaux concentriques** (zoom 3.3e46, period 7327).
  fractall (perturbation **ET** `--algorithm gmp` pur) rend des anneaux ;
  F3 rend du bruit chaotique. Précision écartée (double-double δ+réf), BLA
  écarté (threshold 1e-30 ≡), series écarté, period/wrap écarté (NO_PERIOD =
  anneaux quand même). **Signature identique à glitch_test_5** (GMP fractall ≡
  perturbation fractall ≠ F3). Lean : victoire fractall probable (anneaux =
  structure atom-domain réelle), mais non tranché — faudrait un 3e renderer ou
  F3 haute-résolution.
- [ ] **seahorse-valley FAIL pré-existant** (Mandelbrot 1e8). `fractall-quality`
  pert-vs-GMP : **div_ratio 0.627** (63 % des pixels !), max_diff/p99 = 3072, à
  128² ET 256². **Indépendant** du bailout (4 ≡ 25) et du fix BLA pixel-spacing
  (testé avant/après = identique) → bug pré-existant ailleurs (orbite réf ?
  harness GMP ? seuil d'activation perturbation à 1e8 ?). Anormal sur le preset
  « shallow deep-zoom » phare — à investiguer.
- [ ] **Period-detection truncation = LOSSY** → passer **OFF par défaut**.
  Même pour une période *genuine*, `truncate + wrap_periodic` accumule l'erreur
  de quasi-périodicité (~2^(-0.4·prec)) sur ~iter_max/période cycles →
  perturbation divergente (image uniforme, glitch_test_5). Gain perf de la
  troncature **négligeable** (orbite calculée une fois). **Done when** :
  troncature off par défaut, opt-in seulement aux nucleus exacts
  (`--find-nucleus`) où l'orbite est exactement périodique. Valider goldens +
  GUI avant de flipper.

### G4 — Hybrides multi-phase : la feature unique · `[P1 · différenciation]`

Chaîner des formules par phase (Mandelbrot ⊕ Burning Ship ⊕ …) — feature
absente de Kalles Fraktaler, partielle dans F3. L'infra `Formula::hybrid(vec)`
existe déjà ; il manque la BLA par phase, le nucleus phase-aware, et l'UI/CLI.

**Done when** :
- [ ] **BLA multi-phase native** : `Vec<BlaTableUnified>` (une par phase) au lieu
  d'une seule ; `iterate_pixel_unified_*` switche de BLA au changement de phase
  (F3 `engine.cc:287-295`, `bla.cc::hybrid_blas`). Tests d'invariance hybride
  Mandelbrot⊕BurningShip.
- [ ] **Nucleus finder phase-aware** : étendre `nucleus.rs` au-delà de
  Mandelbrot (Burning Ship, Tricorn, Multibrot entier — dérivées dual-numbers
  par opcode, ~70 % en place via `bla_dual`) ; période/centre/size par phase
  (`engine.cc:118-218`).
  - [ ] **BLA radius scaling σ₁(K)** (rapatrié de l'ex-P1.6.b-bis) : dès qu'un K
    *skewé* (non-conforme) est produit ici, scaler le merge `c` par σ₁(K) (plus
    grande valeur singulière ; det=1 ⇒ σ₁=1/σ₂≥1) dans `delta.rs`/`bla.rs`, ou
    validité anisotrope `|K⁻¹δ| < r`. **No-op tant que K reste conforme** (le
    seul K produit aujourd'hui est `R(θ)`, σ₁=1 ; F3 lui-même ne scale pas son
    BLA par K) → testable seulement avec un K skewé réel, donc lié à ce goal.
- [ ] **CLI/GUI** : `--phases mandelbrot,burning_ship,…` + éditeur de séquence
  GUI.
- [ ] (Optionnel) **`Op::Rot` per-phase** : l'opcode existe (CPU, dual + BLA)
  mais n'est jamais émis ; le câbler seulement si un parseur `[[formula]] rotate`
  des TOML F3 en a besoin (≠ rotation de vue, déjà gérée au pixel→c).

### G5 — Architecture & nettoyage · `[P1 · maintenabilité]`

**Done when** :
- [ ] **Retirer les modules perturbation legacy** : porter `iterate_pixel_gmp`
  sur `pixel_loop`, puis supprimer `glitch.rs`, `nonconformal.rs` et les champs
  perturbation legacy (`max_secondary_refs`, `min_glitch_cluster_size`,
  `glitch_tolerance`, …). Un seul moteur.
- [ ] **Découper les gros fichiers** (< ~800 lignes chacun) :
  `gui/app.rs` (~2.9k) → menu Type / drag-drop / HQ render / raccourcis ;
  `perturbation/mod.rs` (~1.5k) → dispatch CPU/GMP ; `gpu/mod.rs` (~1.8k) →
  pipelines standard / perturbation / bytecode.
- [ ] **GPU : K natif dans `perturbation.wgsl` + shaders f32 dédiés** (basse
  prio) — aujourd'hui fallback CPU quand une transform est active (garde-fou
  contre la sortie non-tournée silencieuse) ; l'appliquer nativement évite le
  fallback sur ces cas niches (plane ≠ Mu, perturbation GPU mid-zoom).

### G6 — Robustesse & infra qualité · `[P1]`

**Done when** :
- [ ] **CI : étendre le corpus golden** à zooms intermédiaires (1e10, 1e15,
  1e20), cap ~70 s/cas. (CI de base déjà en place : unit + golden sur push/PR.)
- [ ] **Vérifier visuellement la GUI AA** (env de dev headless ici → non testé
  à l'écran ; la logique compile et le CLI est vérifié).
- [ ] **AA polish** : per-pixel decorrelation (`burtle_hash`, utile à bas N) ;
  exposer `--jitter-scale` dans la GUI ; AA sur le path GPU.

### G7 — I/O & interop · `[P2]`

**Done when** :
- [ ] **EXR raw export** compatible KFR / zoomasm (assemblage zoom-vidéo) →
  recolorisation/animation sans re-rendre. (Base EXR N0/NF déjà présente via
  `--export-iterations`.)
- [ ] **Interop TOML Fraktaler-3** bidirectionnelle (lire/écrire le format F3
  natif, pas seulement le format léger rust-fractal-core).

---

## ✅ Shipped (condensé, le plus récent en haut)

**2026-05-20** :
- **Rotation GPU** corrigée au pixel→c (`bytecode_kernel.wgsl` applique K depuis
  `transform_matrix()`) — bug : le GPU rendait la vue **non tournée** en silence.
  Garde-fous CPU-fallback sur les autres paths GPU (perturbation, f32 dédiés).
- **Anti-aliasing multi-sample** « per-frame » : module `fractal/jitter.rs`
  (Halton `radical_inverse` + tente `triangle`, port F3 `hybrid.h`), offset
  sous-pixel appliqué aux 4 paths, moyenne RGB. CLI `--aa-samples`/
  `--jitter-scale` + dropdown GUI.
- **BLA pixel-spacing** corrigé vs F3 (2 bugs) : merge `c` = rayon image
  (`pixel_spacing·pixel_precision`, pas `|cref|`) + formule `(R_y−|B_x|·c)/|A_x|`
  (`bla.h:37`). Validation pert-vs-GMP : e17 div 0.96 %→0.037 % (×26).
- **Escape radius** aligné F3 : 4 → **25** (`ESCAPE_TIME_BAILOUT=625`) pour la
  famille escape-time → parité N0/NF à la frontière + smooth coloring plus
  propre. 10 goldens régénérés + revus.

**Antérieur** :
- **P3.1** — Architecture bytecode unifiée (8 opcodes, `compile_formula`, interp
  CPU f64 + GMP, BLA mat2 via duals, delta-form, pixel_loop f64 + ComplexExp,
  GPU `bytecode_kernel.wgsl`, distance/interior/orbit-traps via duals).
- **P1.0** — Uniformisation deep zoom > 1e308 (helpers HP-aware
  `effective_spans_fexp`/`effective_pixel_size`). e1000 : Δmean 691 → 0.44.
- **P1.1** — Rebasing F3 (`|Z+δ|² < |δ|²`) remplace la glitch detection.
- **P1.2** — BLA `mat2` unifié via dual-numbers.
- **P1.3** — `bla_threshold = 1/2²⁴`, `REFERENCE_BAILOUT_SQR = 1e10`, clamp GMP
  `[128, 65536]`, caps `max_perturb_iterations`/`max_bla_steps`.
- **P1.4** — `diffabs` Burning Ship (9 tests d'invariance Z+δ).
- **P1.6.a/b** — Nucleus finder atom-domain (`hybrid_period` `|z|² < s²·|dz|²`)
  + `hybrid_size_mat2` (matrice K).
- **P1.6.b-bis** — `transform_k` stocké + `transform_matrix()` appliqué au
  pixel→c (4 sites render + 2 perturbation) ; nucleus injecte `K/√|det K|`.
- **P1.6.c** — Opcode `Op::Rot` natif CPU (interp/delta-form/BLA dual) + tests
  d'invariance.
- **Parité corpus** — `nf_f3` degree dérivé du bytecode ; gates
  `FRACTALL_NO_AUTO_ADJUST` / `FRACTALL_NO_PERIOD` ; `BLA_SKIP_LEVELS=3` (fix
  over-skip « rug » 1e56, miroir F3 `bla_skip_levels`).
- **CI** — `.github/workflows/ci.yml` (unit + golden, ubuntu, gmp/mpfr/mpc).
- **glitch_test_5** — classé **F3-dégénéré** (F3 rend un carré noir via
  fast-path period/nucleus ; fractall rend un minibrot correct). Victoire
  fractall, à exclure du score parité.

---

## 🛡️ Ne pas régresser (superset assumé vs F3)

- 27 palettes built-in + espaces RGB/HSB/LCH.
- 15 modes de coloring + 4 orbit traps.
- 7 plane transforms XaoS-style.
- Drag-and-drop PNG avec metadata JSON (restauration exacte de l'état).
- Fractales non-escape-time (F3 n'y va pas) : Newton, Phoenix, Magnet, Lyapunov
  (6 presets), Buddhabrot/Nebulabrot/Anti, Von Koch, Dragon, Pickover Stalks,
  Nova, Sin, Alpha Mandelbrot, Barnsley, Mandelbulb.
- Preview Julia au survol + raccourci `J`.
- Recolorisation asynchrone sans bloquer l'UI ; rendu progressif multi-passes.
- Anti-aliasing multi-sample (CLI + GUI).
- `bailout` reste configurable (retrouver ER=4 si besoin).

---

## 📌 Notes de référence (où regarder dans F3)

| Sujet | F3 |
|-------|-----|
| Escape radius | `param.h:41` (`escape_radius = 625`) |
| BLA merge / lookup | `bla.h:27-41`, `bla.cc` ; `c = pixel_spacing·pixel_precision` (`engine.cc:282`) |
| pixel→c + transform K | `hybrid.cc:233-265` (`c = K·c + offset`) |
| Single-step BLA | `hybrid.h:142` (ignore `c`) |
| Jitter AA | `hybrid.h:16-68` (`burtle_hash`/`radical_inverse`/`triangle`/`jitter`) |
| Nucleus | `hybrid.cc` (`hybrid_period`/`hybrid_center`/`hybrid_size`) |
| Wisdom / auto-precision | `wisdom.cc:240-295`, `render.cc:219` |
| Hybrides multi-phase | `engine.cc:118-295` |
