# Changelog

Tous les changements notables de **fractall-rust** (renderer de fractales
deep-zoom, référence algorithmique Fraktaler-3.1).

Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/) et adapté :
regroupement par type (Ajouté / Corrigé / Performance / Modifié), et sections par
jalon (`Gx`) pour l'historique. Ce fichier est un résumé actionnable — le détail
technique vit dans `TODO.md`, `CLAUDE.md`, `SCORECARD.md` et l'historique git.

## [Non publié]

### Ajouté
- **G9.5 auto-device** : `wisdom::select_device(params, gpu_available) -> Device`
  (`src/fractal/wisdom.rs`) arbitre CPU/GPU par débit benché machine, sous
  garde-fou correction — GPU routé UNIQUEMENT dans la plage deep both-perturbation
  (~1e12–4e37, les deux devices font de la perturbation), JAMAIS sur les shaders
  std f32 (24 b = faux). CLI : `--gpu`/`--no-gpu` deviennent des OVERRIDES, sinon
  AUTO par défaut. GUI : menu « Tech: 🔄 Auto » = device auto. Sur GPU grand public
  (f64 1:64) l'auto reste CPU (gpu_perturb ~7.6× plus lent).
- **G9.5 bench GPU** : `--wisdom-bench` mesure aussi la clé `gpu_perturb_f64`
  (`src/fractal/wisdom_bench.rs`) → `~/.config/fractall/wisdom.toml`, consommée par
  l'arbitrage device.
- **G4 hybrides multi-phase (jalons 1-4)** : les fractales HYBRIDES rendent —
  CLI **`--phases mandelbrot,burning_ship`** (types escape-time itérés
  cycliquement). `params.hybrid_phases` + `formula_for_params` +
  `compile_hybrid_formula` (`src/fractal/bytecode/`). Rendu par le path f64
  standard, **et en perturbation deep** sur TOUT le range : f64
  (`iterate_pixel_unified_multi_phase`, ~zoom 1e13–1e280, jalon 3) puis
  **ComplexExp** (`iterate_pixel_unified_exp_multi_phase`, deep > 1e280,
  jalon 4) — sans BLA, cyclant `phases[n % len]` + rebasing F3. `[M,M]`
  pixel-exact == Mandelbrot (invariants testés : f64-std + deep-perturbation
  3e10 + deep-exp **1e1000**), `[M,BS]` = hybride genuine. `render_dispatch`
  renvoie `None` (GPU ne cycle pas). Verrous : 3 unit/render-level tests +
  golden `mandelbrot_hybrid_burningship`. Reste (jalon 5) : BLA par phase +
  nucleus phase-aware + éditeur GUI.

### Corrigé
- **Perturbation réf-intérieure >512²** : un 2e bloc de résolution glitch récursive
  (`perturbation/mod.rs`) n'était pas gaté `!bytecode_path` → il supprimait le
  fallback GMP → ~3.4 % de structure spurious. Gate ajouté.
- **GUI warp G10.1** : signe Y inversé de l'aperçu zoom molette hors-centre
  (`compute_warp_norm` : `0.5 + dy`, `src/gui/app.rs`).

### Performance
- **Fallback dd réf-intérieure** : quand `glitch_ratio > 0.30` (Mandelbrot
  bytecode), escalade au tier dd (~106 b) au lieu du full-GMP per-pixel — ~4-8×
  plus rapide, pixel-exact GMP (`GLITCH_FALLBACK_THRESHOLD`, `perturbation/mod.rs`).

### Modifié
- **Couverture >512² durcie** : nouveau golden `mandelbrot_interior_ref_640`
  (seul cas >512², exerce l'escalade dd) ; audit des gates `small_image` (tous
  sains).

**État moteur** : 0 gap mesuré (harness quick + standard-speed + wisdom-optimality),
bat F3 partout (geomean ~0.18, 25/25 wins speed). 271 tests unit CLI + 24 golden
pixel-exact + quality 15/15 PASS.

## [G10] — Plan XaoS : réutilisation inter-frame & tuiles

### Ajouté
- **G10.5 file de tuiles priorité-centre** (`src/render/tiles.rs`) : les 4 boucles
  pixel CPU tournent sur une work-queue de tuiles ordonnée par distance au curseur ;
  streaming intra-passe GUI (la zone sous le curseur devient nette en premier).
  L'ordre ne change AUCUN pixel (verrou `tiled_render_identical_across_priorities`).
- **G10.4 réutilisation pixels inter-frame XaoS** (`src/fractal/xaos.rs`) : en
  pan/zoom sans rotation, colonnes/lignes matchées ≤ 0.5 px copiées de la frame
  précédente (~×40 en pan). Zoom (2026-07-16) : matching injectif par axe + refine
  union silencieux à l'idle.
- **G10.2** : réutilisation orbite référence inter-frame off-center (perturbation).
- **G10.1** : warp GPU de la dernière frame — aperçu fluide pendant le rendu.

### Corrigé
- **G10.4b** : bruit deep-zoom en réutilisation cache orbite (régime atom-domain :
  la troncature atom-domain dépend de l'échelle de vue → réf invalidée si span
  différent). Zoom injectif + molette ancrée. Écho XaoS et reuse basse-résolution
  rendus mutuellement exclusifs (verrou `echo_pass_ignores_coarse_pass_reuse`).

### Performance
- **G10.3** : recolorisation sans clone 74 Mo — `Arc` partagé.

## [G9] — Moteur multi-techniques orchestré par le wisdom

### Ajouté
- **G9.1** : `wisdom` = planificateur unique. `select_algorithm(params, device)`
  + `variants` consommés par les dispatchers CPU/GPU/GUI (`src/fractal/wisdom.rs`).
- **G9.2** : benchmarks machine persistés — `--wisdom-bench` mesure le débit
  effectif par technique (`src/fractal/wisdom_bench.rs`).
- **G9.3** : routage Harmonic LA par le wisdom — AUTO par défaut sur la classe
  « période courte » (probe `period0 ≤ 100`, gagne 1.2-5.9×).
- **G9.4a/b** : kernel GPU perturbation F3-strict en f64 natif
  (`src/gpu/perturbation.wgsl`) — parité CPU-grade 1e4→1e8, réfs tronquées
  (rebase-at-end F3) + range 1e8→~4e37. Juge `gpu-suite` = GMP pur.
- **Verrous** : `fractall-quality gpu-suite`/`gpu-compare` (parité GPU↔CPU),
  tests quality en CI, axe harness `wisdom-optimality` (le plan auto jamais battu
  >10 % par un forcé).

### Corrigé
- **G9 fix epsilon BLA** : epsilon de validité BLA f64 = 2⁻⁵³ (mantisse f64) au
  lieu de 2⁻²⁴ (f32) — corrige les scènes f64 deep silencieusement fausses.

## [G8.2] — Techniques post-F3 (compression, Harmonic LA)

### Ajouté
- **Compression d'orbite** (`PTWithCompression`) : orbite O(waypoints) env-gated,
  path f64 Mandelbrot, opt-in `FRACTALL_COMPRESS_REF`.
- **Harmonic MLA/LLA** : prototype segmenté aux dips + ré-ascension gardée
  (3× sur période courte).

### Performance
- **dd-BLA** : epsilon 2⁻¹⁰⁶ → 2⁻⁸⁰ calibré par bissection QA — e50 quality ~2.3×.
- **Pixel-loop** : cache tête-de-boucle (réutilise `Z[m']+δ'` déjà calculé, −4.5 %).
- **PNG save** : 5.6× plus rapide (fdeflate).

## [G2/G3] — Correction & vitesse deep-zoom

### Corrigé / Performance
- **Rebase-at-end F3 strict** (`perturbation/delta.rs` → `bytecode/pixel_loop*.rs`) :
  garde les centres escape-time profonds sur le path perturbation au lieu de tomber
  en GMP per-pixel — débloque la perf deep-zoom (e50 544→1.6 s, e1000 742→0.5 s,
  dragon ~6 h→6.5 s à 256²).
- **G3** : verrous pert-vs-GMP Celtic/Buffalo/PerpBS (frontières lisses).
- Escalade full-GMP des pixels encore glitchés (sous-cas glitch réf-unique).

## [G1] — Base moteur unifié

- **Moteur bytecode P3.1** (`src/fractal/bytecode/`) : formule compilée unifiée,
  path CPU f64 / CPU perturbation (f64/exp/dd) / GPU. Défaut du dispatcher.
- **Perturbation + BLA** (`src/fractal/perturbation/`) : orbite référence GMP,
  BLA mat2 dual-number, atom-domain nucleus finder (`nucleus.rs`).
- **Dispatch unique CLI↔GUI** (`src/render/escape_time.rs`) : un seul dispatcher
  `render_escape_time_cancellable_with_reuse`.
