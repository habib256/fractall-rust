# TODO

> **Objectif** : être le meilleur renderer deep-zoom open-source en Rust. Fraktaler-3 = source de vérité algorithmique (cf. `docs/fraktaler-3-analysis.md`).

État au **2026-05-18** : P3.1 livré end-to-end (Sessions A-E + GPU + dual numbers + cleanup). Le moteur bytecode unifié (BLA mat2 + rebasing F3) est actif par défaut sur CPU et GPU. Harness `scripts/compare_f3.py` opérationnel : compare EXR (N0/NF) entre F3 et Fractall pour les 84 fichiers de `toml/`. **Premier résumé (résolution 200×200, ER=4)** : 7 cas OK avec mean Δsi ∈ [0.65, 1342], 2 fractall_fail (e1086, opus, timeout 60s), 1 f3_fail (seahorse). La dette restante est documentée ci-dessous.

---

## P0 — Goal actif : parité F3 sur le corpus `toml/`

**Objectif** : Fractall doit produire les mêmes images que Fraktaler-3 pour les
**84 fichiers** de `toml/` (tous Mandelbrot deep zoom au format léger
rust-fractal-core — aucun n'utilise `[[formula]]` / `bailout.*` / `transform.*`
F3-natif). C'est la **mesure de vérité** de l'objectif global du projet : on n'est
pas crédible comme "meilleur deep-zoom open-source en Rust" tant qu'on n'égale
pas F3 visuellement sur ce corpus.

**Critère d'acceptation** : pour chaque `toml/X.toml`, le diff `mean_abs(F3, fractall) ≤ ε`
sur une résolution standard (1920×1080 par défaut), avec ε à définir empiriquement
(viser pixel-perfect là où c'est possible, équivalence visuelle ailleurs).

**Plan d'attaque (sous-tâches trackées)** :
1. ✅ CLI `--toml <path>` pour `fractall-cli` (charge real/imag/zoom/iterations + rotate).
2. ✅ Support du champ `rotate` dans le loader TOML (plusieurs fichiers du corpus
   l'utilisent : olbaid1/3/4/5, opus, opus2, x). Appliqué via `K = mat2(cos,-sin,sin,cos)`
   au mapping pixel→c (aligné F3 `hybrid.cc:265`). Champ `FractalParams::rotation`
   + helper `rotation_matrix()`, câblé dans le path standard f64, GMP escape-time,
   perturbation/bytecode `FloatExp`, et `DcGmpContext::compute_dc` pour le path
   GMP-per-pixel. CLI flag `--rotation` aussi disponible (override le TOML).
   **Path legacy non couvert** : la résolution de glitches Pauldelbrot dans
   `perturbation/mod.rs` (lignes ~1109, 1247) ne tient pas compte de la rotation
   pour les références secondaires — à corriger en même temps que la suppression
   du path legacy (cf. P1.1 reste).
3. ✅ Harness `scripts/compare_f3.py` qui lance F3 batch + fractall-cli sur chaque TOML,
   produit PNG + diff + métriques. F3 binaire dispo à `fraktaler-3-3.1/fraktaler-3.macos`.
   Sortie : `bench/compare/_summary.{csv,md}` + PNG side-by-side. Export EXR
   Fractall via `--export-iterations` (N0+NF format F3, NBIAS=1024). NF formula
   `nf_f3` dans `src/io/exr.rs` vérifiée pixel-pour-pixel identique à F3
   `hybrid.cc:350`.
4. **Audit initial fait (2026-05-18)**, premier résumé :
   - cas OK : `test` (Δmean=0.65), `spiral` (10.45), `e50` (108), `e1121` (119),
     `glitch_test_1` (191), `e1000` (691), `olbaid1` (1342)
   - fails : `e1086`, `opus` (fractall timeout 60s), `seahorse` (f3 timeout)
   - **Pattern critique sur deep zooms (e1000, e1121)** : Fractall produit une
     image **uniforme** (tous pixels au même iter ≈ 691), F3 montre la structure
     fractale détaillée. mean ≈ rms ≈ max signale un offset quasi-constant.
   - **Hypothèse cause** : détection de période sur l'orbite référence
     (`ReferenceOrbit::cycle_period`) + `wrap_periodic` ramène tous les pixels
     au même point du cycle court. Le fix récent dans `perturbation/delta.rs`
     (path legacy) traite le cas pour les centres intérieurs, mais pas
     symétriquement appliqué dans `bytecode/pixel_loop_exp.rs` (deep zoom
     ComplexExp) ni `perturbation/delta.rs::iterate_pixel_gmp`. À corriger.
5. **P1.3 quick wins** : audit des constantes (voir détail mis à jour ci-dessous) :
   - ✅ `bla_threshold = 1.0/2²⁴` (déjà aligné F3)
   - ✅ `REFERENCE_BAILOUT_SQR = 1e10` (déjà aligné F3, cf. `perturbation/orbit.rs:26`)
   - ⚠️ `bailout` (escape radius pixel) défaut Fractall = `4` ; F3 défaut = `25`
     (ER² = 625). Pour la comparaison on aligne explicitement à 4 côté F3 via le
     wrapper TOML. Pour matcher visuellement F3 hors comparaison, garder 25 en
     défaut (mieux conditionné pour smooth iteration). À discuter : faire de
     `25` le défaut Fractall et mettre à jour goldens.
   - ✅ Clamp GMP précision relâché à `[128, 65536]` (cf. `mod.rs:497`). F3 n'a
     pas de clamp haut ; 65536 couvre le corpus (max ≈ 26500 bits à zoom 1e8000).
   - ⏳ `iters_ptb < PerturbIterations` et `steps_bla < BLASteps` (F3) : caps
     anti-runaway par pixel, NON enforcés dans `bytecode/pixel_loop.rs`
     (seul `n < iteration_max` est testé). À ajouter pour matcher F3, qui sort
     en `n = current_iter` (pas `iter_max`) quand cap atteint → effet visuel
     différent sur pixels intérieurs.
6. **P1.5 AA subframes jitterés** — F3 le fait par défaut, indispensable pour parité
   visuelle bords fins.
7. Itérer jusqu'à convergence sur l'ensemble du corpus (élargir résolution / cas
   testés après les premières corrections deep-zoom).

---

## Priorité 1 — Qualité numérique deep-zoom

### P1.0 — Uniformisation aux deep zooms (e1000, olbaid1) ✅ FIXÉ (2026-05-18)
- **Symptôme initial** : à zoom > 1e308, Fractall produisait une image quasi-
  uniforme (`avg_iter/px == max_iter/px`), F3 rendait la structure fractale
  détaillée. Bench : mean Δsi ≈ rms ≈ max (e1000: 691/691/707, olbaid1:
  1342/1652/4312).
- **Cause racine** : `params.span_x` / `params.span_y` sont f64. À zoom > 1e308,
  `span = 4/zoom` underflow à 0 en f64 (même si `span_x_hp` HP est correct).
  Tout le calcul `dc_re_fexp[i] = FloatExp::from_f64(((i+0.5)/width - 0.5) * x_range)`
  produit donc 0 pour tous les pixels → dc nul → uniformisation.
- **Fix** (`perturbation/mod.rs`) :
  1. Helper `effective_spans_fexp(params)` qui parse `span_x_hp` / `span_y_hp`
     en `Float(1024 bits)` puis convertit en `FloatExp` (mantisse f64 + exp i32).
     Couvre des magnitudes jusqu'à ±2^31, soit zoom jusqu'à ~1e10⁸.
  2. Helper `effective_pixel_size(params)` HP-aware utilisé par les
     dispatchers de path.
  3. `bytecode_path_label` et `try_bytecode_unified_path` (`delta.rs`) :
     utilisent `effective_pixel_size`, retirent le seuil GMP qui était
     surconservateur — bytecode_exp gère arbitrairement profond via ComplexExp.
  4. `should_use_full_gmp_perturbation` court-circuite à `false` quand
     bytecode est applicable (évite GMP-per-pixel inutile + très lent).
  5. `render_perturbation_with_cache` : remplace `x_range = params.span_x`
     (f64 underflow) par `x_range_fexp` (FloatExp HP-aware) dans la
     construction de `dc_re_fexp` / `dc_im_fexp` / chemin jitter.
- **Résultat** (bench 200×200) :
  | Cas | Avant Δmean | Après Δmean | Statut |
  |-----|------------:|------------:|--------|
  | e1000 | 691.13 | **0.44** | ✅ parité quasi-F3 |
  | olbaid1 | 1341.98 | **52.89** | ✅ progrès énorme |
  | e1121 | 119.18 | 116.89 | ⏳ bug indépendant |
  | e50 | 107.73 | 107.73 | ⏳ idem |
- **Restant** :
  1. Glitch-correction paths (`perturbation/mod.rs:1195, 1292, 1332`) utilisent
     toujours `x_range` f64 pour le dc secondaire — à porter sur FloatExp
     quand le path legacy sera retiré.
  2. Investigation e1121 / e50 (offset ~100, pas une uniformisation pure).

### P1.1 — Rebasing proactif ✅ FAIT (P3.1 Sessions A-E)
- [x] Rebasing F3 dans `fractal/bytecode/pixel_loop.rs` avec condition F3 stricte `|Z+z|² < |z|²`.
- [x] Validé sur Tricorn (100 % pixel-perfect), BurningShip (5 % diff visuellement équivalent), Mandelbrot deep zoom 1e6.
- [x] Activé par défaut depuis Session E. `--no-bytecode` retombe sur le path legacy (glitch detection Pauldelbrot).
- [ ] **Reste** : supprimer `glitch.rs`, `nonconformal.rs` et les champs `glitch_tolerance` / `max_secondary_refs` / `min_glitch_cluster_size` une fois que le path GMP deep zoom (`perturbation/delta.rs::iterate_pixel_gmp`) sera lui aussi porté sur pixel_loop. Le path bytecode CPU/GPU n'utilise plus ces modules.

### P1.2 — BLA `mat2` unifié ✅ FAIT (P3.1 Sessions B + D)
- [x] `BlaTableUnified` construit via dual-numbers walking le bytecode (`fractal/bytecode/bla_dual.rs`).
- [x] `sup_norm` en formule fermée 2×2.
- [x] Branche conformal / non-conformal supprimée du path bytecode unifié.
- [ ] `bla.rs` (conformal) et `nonconformal.rs` (matriciel) restent pour le path GMP deep zoom — à retirer en même temps que P1.1.

### P1.3 — Aligner les constantes critiques sur F3 (quick wins)
- [x] Reference bailout = `1e10` hardcoded — `REFERENCE_BAILOUT_SQR` dans
      `perturbation/orbit.rs:26`, utilisé f64 (ligne 27) et GMP (ligne 268).
- [ ] Escape radius pixel par défaut = `25` (carré 625) — meilleur conditionnement
      smooth. Actuellement défaut Fractall = `4`. Toucher au défaut casse les
      tests goldens : à coupler avec un refresh des golden PNG.
- [x] `bla_threshold = 1.0/2²⁴ ≈ 5.96e-8` — défini dans `definitions.rs:33` et
      `types.rs::default_bla_threshold`.
- [x] Précision GMP : clamp relâché à `[128, 65536]` (`mod.rs:497, 522`). F3 n'a
      pas de clamp haut ; suffisant pour le corpus actuel (max ≈ 26500 bits).
- [ ] Vérifier pixel spacing BLA = `4/zoom/height` strict dans tous les paths
      (`bytecode/bla_dual.rs`, `perturbation/bla.rs`, `nonconformal.rs`).
- [ ] **NOUVEAU** — caps `max_perturb_iterations` / `max_bla_steps` non enforcés
      dans `bytecode/pixel_loop{,_exp}.rs`. F3 sort `n = current_iter` quand
      le cap est atteint, ce qui colore le pixel comme « échappé tard » au lieu
      d'intérieur. À ajouter pour parité.

### P1.4 — `diffabs` Burning Ship ✅ FAIT
- [x] `diffabs(c, d)` dans `delta.rs::diffabs`, utilisé par le bytecode delta-form (`bytecode/delta_form.rs`) pour AbsX/AbsY, validé par 9 tests d'invariance Z+δ.

### P1.5 — Anti-aliasing par subframes jitterés
- [ ] Wrapper N samples avec offsets jitterés (`burtle_hash` / `radical_inverse` ou équivalent low-discrepancy) → moyenne.
- [ ] Combinable avec progressive rendering (1 subframe = 1 pass).
- [ ] Le champ `jitter_scale` existe déjà sur `FractalParams` mais n'est pas exposé en CLI/GUI ni accumulé en multi-passes.
- **Pourquoi** : AA propre pour les bords fins, surtout en mode DE/Distance. F3 le fait par défaut.

---

## Priorité 2 — Infrastructure

### P2.1 — CI + tests d'images de référence
- [x] **Harness golden images local** : `tests/golden_images.rs` + 10 cas dans `tests/golden/`. Couvre Mandelbrot / Julia / BS / Tricorn / Celtic défauts, Multibrot pow 3, perturbation 1e6, BS perturbation 1e3, Newton, minibrot zoom 1e8, et un deep zoom Mandelbrot 5e113 comme garde-fou GMP+BLA. Comparaison pixel exact. Régénération via `FRACTALL_UPDATE_GOLDENS=1`.
- [ ] **GitHub Actions** : `cargo test --release --bin fractall-cli` + `cargo test --release --test golden_images` sur push / PR.
- [ ] Étendre le corpus à des zooms intermédiaires (10¹⁰, 10¹⁵, 10²⁰) sans dépasser ~70 s par cas (le 5e227 testé a été abandonné, trop long).
- **Pourquoi** : sans ça, chaque modif de perturbation / BLA / bytecode est une roulette russe.

### P2.2 — Découpe des gros fichiers
- [ ] **`gui/app.rs` (2854 lignes)** : extraire menu Type, drag-and-drop, rendu HQ asynchrone, raccourcis clavier dans des modules séparés.
- [ ] **`perturbation/mod.rs` (1456 lignes)** : éclater `render_perturbation_cancellable_with_reuse()` et le dispatch CPU/GMP.
- [ ] **`gpu/mod.rs` (1747 lignes)** : séparer les pipelines (standard, perturbation, bytecode) en sous-modules.
- **Pourquoi** : à faire avant tout gros refactor P1 pour pouvoir naviguer.

### P2.3 — Documenter le tradeoff GPU f32
- [x] Le GPU n'utilise plus que des shaders f32 (les *_f64.wgsl ont été retirés).
- [ ] Documenter dans README la limite de zoom GPU (~10⁷ en f32) et le seuil de fallback CPU.
- [ ] Évaluer softfloat (F3 §11) pour devices sans fp64 — peu probable que ça vaille le coup avec wgpu.

---

## Priorité 3 — Scope stratégique

### P3.1 — Architecture hybride bytecode ✅ FAIT (Sessions A-E + GPU + cleanup)
- [x] Bytecode 8-opcodes `Sqr/Mul/Store/AbsX/AbsY/NegX/NegY/Add` (`fractal/bytecode/mod.rs`).
- [x] `compile_formula` pour Mandelbrot/Julia/BS/Tricorn/Celtic/Buffalo/PerpBS/Multibrot puissance entière + variantes Julia.
- [x] Interpréteur CPU f64 + GMP (`bytecode/interp{,_gmp}.rs`).
- [x] BLA mat2 unifié via dual-numbers (`bytecode/bla_dual.rs`) — Vec multi-niveaux avec merge F3.
- [x] Delta-form interpreter (`bytecode/delta_form.rs`) — `DeltaState` (f64) + `DeltaStateExp` (ComplexExp).
- [x] Pixel loop unifié (`bytecode/pixel_loop.rs` + `pixel_loop_exp.rs`) — BLA mat2 + delta-form + rebasing F3.
- [x] Intégration dans `delta.rs::iterate_pixel` avec cache thread-local.
- [x] Activé par défaut depuis Session E. Pixel-perfect ou diff < 5 % vs legacy.
- [x] **ComplexExp dans pixel_loop** pour deep zoom > 1e13 — pixel-perfect à zoom 1e15 et 1e30.
- [x] **Multi-phase** infrastructure (`Formula::hybrid`) prête (manque UI/CLI).
- [x] **GPU bytecode runtime** (`pipeline_bytecode` + `bytecode_kernel.wgsl`) — Mandelbrot/Julia/BS pixel-perfect ou diff < 1.5 %. Étend la couverture GPU à Tricorn/Celtic/Buffalo/PerpBS/Multibrot sans shaders dédiés.
- [x] **Dual numbers dans bytecode** (`iterations.rs::iterate_via_bytecode` et `pixel_loop.rs`) — distance estimation, interior detection et orbit traps pixel-perfect vs legacy, sur le path standard ET en mode perturbation.
- [x] **Cleanup** : suppression du champ `use_legacy_glitch_detection`, de `iterate_pixel_with_duals` et des modules `dual` orphelins.
- [ ] **Reste mineur** : retirer définitivement `glitch.rs` et `nonconformal.rs` — bloqué tant que `perturbation/delta.rs::iterate_pixel_gmp` (path GMP deep zoom pur) n'a pas été ported sur pixel_loop. Voir « Ordre d'attaque » plus bas.

### P3.2 — Wisdom-driven backend selection
- [ ] Benchmark `(device, type)` au premier run, JSON persisté.
- [ ] Choisir le backend le plus rapide viable pour le zoom courant.
- **Pourquoi** : vraie valeur sur matériel exotique, pas critique sur desktop moderne.

### P3.3 — EXR raw export
- [ ] Format compatible KFR / zoomasm pour assemblage vidéo-zoom.
- [ ] Permet de recolorer / animer sans re-rendre.
- **Pourquoi** : niche mais différenciateur pour les créateurs de zoom-vidéos.

### P3.4 — Multi-phase hybrid formula UI/CLI
- [x] `Formula::hybrid(vec![phase, ...])` supporte déjà les chaînes de phases en interne.
- [ ] Exposer en CLI : `--phases mandelbrot,burning_ship,burning_ship` etc.
- [ ] Exposer en GUI : éditeur de séquence de phases.
- [ ] Construire `Vec<BlaTableUnified>` par phase au lieu d'une seule.
- **Pourquoi** : feature unique vs Kalles Fraktaler / Fraktaler-3.

---

## À NE PAS régresser (superset assumé vs F3)

- 27 palettes built-in + RGB/HSB/LCH (UX win).
- 15 modes coloring + 4 orbit traps built-in.
- 7 plane transforms XaoS-style.
- Drag-and-drop PNG avec metadata JSON.
- Catalogue de formules non-escape-time : Newton, Phoenix, Magnet, Lyapunov (6 presets), Buddhabrot/Nebulabrot/Anti, Von Koch, Dragon, Pickover Stalks, Nova, Sin, Alpha Mandelbrot, Barnsley, Mandelbulb — F3 n'y va pas, c'est notre terrain de différenciation.
- Preview Julia au survol + raccourci `J`.
- Recolorisation asynchrone sans bloquer l'UI.

---

## Format / I/O

- [x] Loader TOML de fichiers de paramètres (`toml/*.toml`) — large corpus inclus.
- [ ] Vérifier la compatibilité avec le format TOML de Fraktaler-3 si interop souhaitée.

---

## Ordre d'attaque recommandé

P3.1 est clos pour le hot path (CPU bytecode + GPU bytecode + dual numbers). Le
nouveau goulot est la **parité visuelle vs F3 sur le corpus `toml/`** (P0).

1. **P1.0 — fix uniformisation deep-zoom** (e1000, e1121, olbaid1). Bug
   bloquant pour P0. Cause = traitement de la périodicité dans `pixel_loop_exp`
   et `iterate_pixel_gmp`. Sans ça, P0 est inatteignable.
2. **P2.1 GitHub Actions** : indispensable avant tout autre gros refactor — la
   confiance vient des golden tests.
3. **Investiguer fails Fractall** (e1086, opus). Probablement GMP-per-pixel
   coûteux quand BLA ne couvre pas → 60s timeout. À profiler.
4. **P1.3 quick wins restants** : caps `max_perturb_iterations` / `max_bla_steps`
   dans `bytecode/pixel_loop{,_exp}.rs`, plus la décision sur `bailout` défaut
   (4 vs 25).
5. **Porter `iterate_pixel_gmp` sur pixel_loop** (path GMP deep zoom pur, sans
   rebase F3 bytecode). Une fois fait, on retire `glitch.rs`, `nonconformal.rs`,
   les champs perturbation legacy, et la branche `--no-bytecode` peut devenir
   une garde de debug uniquement.
6. **P2.2 découpe des gros fichiers** : préalable à tout refactor architectural.
7. **P1.5 AA subframes jitterés** : qualité visuelle, surtout en mode Distance.
8. **P3.4 multi-phase UI/CLI** : feature de différenciation, infrastructure
   déjà prête.
