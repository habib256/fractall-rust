# TODO

> **Objectif** : être le meilleur renderer deep-zoom open-source en Rust. Fraktaler-3 = source de vérité algorithmique (cf. `docs/fraktaler-3-analysis.md`).

État au **2026-05-17** : P3.1 livré end-to-end (Sessions A-E + GPU + dual numbers + cleanup). Le moteur bytecode unifié (BLA mat2 + rebasing F3) est actif par défaut sur CPU et GPU. La dette restante est documentée ci-dessous.

---

## Priorité 1 — Qualité numérique deep-zoom

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
- [ ] Reference bailout = `1e10` hardcoded dans `hybrid_reference` équivalent.
- [ ] Escape radius pixel par défaut = `25` (carré 625) — meilleur conditionnement smooth.
- [ ] `bla_threshold` = `1/2²⁴ ≈ 6e-8`, indépendant du type (actuellement `1e-8`).
- [ ] Précision GMP : retirer le clamp `[128, 8192]` ou passer à `[24, ∞]`.
- [ ] Pixel spacing BLA = `4/zoom/height` strictement (revérifier).

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

P3.1 est clos pour le hot path (CPU bytecode + GPU bytecode + dual numbers). Il reste deux blocs avant de pouvoir supprimer le code legacy :

1. **Porter `iterate_pixel_gmp` sur pixel_loop** (path GMP deep zoom pur, sans rebase F3 bytecode). Une fois fait, on retire `glitch.rs`, `nonconformal.rs`, les champs perturbation legacy, et la branche `--no-bytecode` peut devenir une garde de debug uniquement.
2. **P2.1 GitHub Actions** : indispensable avant tout autre gros refactor — la confiance vient des golden tests.

Une fois ces deux items livrés, les chantiers ouverts en priorité :

3. **P1.3 quick wins** : alignement des constantes F3 (peu de code, gros gain de cohérence).
4. **P2.2 découpe des gros fichiers** : préalable à tout refactor architectural.
5. **P1.5 AA subframes jitterés** : qualité visuelle, surtout en mode Distance.
6. **P3.4 multi-phase UI/CLI** : feature de différenciation, infrastructure déjà prête.
