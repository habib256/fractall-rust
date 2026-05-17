# TODO

> **Objectif declare** : etre le meilleur renderer deep-zoom open-source en Rust. Fraktaler-3 = source de verite algorithmique (cf. `docs/fraktaler-3-analysis.md`).

---

## Roadmap strategique

### Priorite 1 — Qualite numerique deep-zoom

Changements qui rendent les images plus correctes et les zooms plus profonds plus rapides.

#### P1.1 — Migrer vers le rebasing proactif ✅ FAIT (P3.1 Sessions A-E)
- [x] Rebasing F3 implémenté (`fractal/bytecode/pixel_loop.rs`) avec condition F3 stricte `|Z+z|² < |z|²`.
- [x] Validé sur Tricorn (100% pixel-perfect), BurningShip (5% diff visuellement équivalent), Mandelbrot deep zoom (93% diff fines, 8% lourd).
- [x] Activé par défaut depuis P3.1 Session E.
- [ ] Retirer `glitch.rs`, `glitch_tolerance`, `max_secondary_refs`, `min_glitch_cluster_size` — partiellement fait (marqué deprecated), retrait complet bloqué par le path GMP deep zoom + dual numbers (distance/interior) qui en dépendent encore. Reste : étendre pixel_loop à ComplexExp + dual numbers, puis supprimer.

#### P1.2 — Unifier BLA en `mat2` partout ✅ FAIT (P3.1 Session B + D)
- [x] BLA mat2 unifié construit via dual-numbers walking le bytecode (`fractal/bytecode/bla_dual.rs::BlaTableUnified`).
- [x] `sup_norm` implémenté formule fermée 2×2.
- [x] Branche conformal/non-conformal supprimée du **path bytecode unifié** (par défaut depuis Session E).
- [ ] `bla.rs` et `nonconformal.rs` restent pour le path GMP deep zoom (à retirer une fois ComplexExp dans pixel_loop).

#### P1.3 — Aligner les constantes critiques sur F3 (quick wins, <1h chacun)
- [ ] Reference bailout = `1e10` hardcoded dans `hybrid_reference` equivalent.
- [ ] Escape radius pixel par defaut = `25` (carre 625) si pas deja le cas — meilleur conditionnement smooth.
- [ ] BLA threshold = `1/2²⁴ ≈ 6e-8`, independant du type (actuellement `1e-8`).
- [ ] Precision GMP : retirer le clamp `[128, 8192]` ou passer a `[24, ∞]`.
- [ ] Pixel spacing BLA = `4/zoom/height` strictement (reverifier apres bugfix #11).

#### P1.4 — Verifier `diffabs` Burning Ship ✅ FAIT
- [x] `diffabs(c, d)` existe dans `delta.rs::diffabs` et est utilisé par le bytecode delta-form (`bytecode/delta_form.rs`) pour AbsX/AbsY, validé par 9 tests d'invariance Z+δ.

#### P1.5 — Anti-aliasing par subframes jitterés
- [ ] Implementer wrapper N samples avec offsets jitterés (`burtle_hash`/`radical_inverse` ou equivalent low-discrepancy) → moyenne.
- [ ] Combinable avec progressive rendering (1 subframe = 1 pass).
- **Pourquoi** : AA propre pour les bords fins, surtout en mode DE/Distance. F3 le fait par defaut.

### Priorite 2 — Infrastructure

#### P2.1 — CI + tests d'images de reference (non-negociable)
- [x] **Harness golden images local** : `tests/golden_images.rs` + 10 cas dans `tests/golden/` couvrant Mandelbrot/Julia/BS/Tricorn/Celtic/Multibrot pow 3/perturbation 1e6/perturbation 1e3/Tricorn perturb 1e2/Newton. Comparaison pixel exact. Regen via `FRACTALL_UPDATE_GOLDENS=1`. Documenté dans CLAUDE.md.
- [ ] GitHub Actions : `cargo test --release --bin fractall-cli` + `cargo test --release --test golden_images` sur push/PR.
- [ ] Inclure zooms encore plus profonds (10¹⁰, 10¹⁵, 10²⁰) une fois le rebasing en place.
- **Pourquoi** : sans ça, chaque modif de perturbation/BLA est une roulette russe. Indispensable vu l'objectif.

#### P2.2 — Decoupe gros fichiers
- [ ] **`gui/app.rs` (2854 lignes)** : extraire le menu Type, drag-and-drop, rendu HQ asynchrone, raccourcis clavier dans modules separes.
- [ ] **`perturbation/mod.rs` (1451 lignes)** : `render_perturbation_cancellable_with_reuse()` et dispatch CPU/GMP eclates.
- **Pourquoi** : a faire avant les gros refactors P1 pour pouvoir naviguer.

#### P2.3 — Documenter le tradeoff GPU f32
- [ ] Documenter dans README/CLAUDE.md la limite de zoom GPU (~10⁷ en f32).
- [ ] Indiquer le seuil de fallback CPU dans la doc utilisateur.
- [ ] Evaluer si softfloat (cf. F3 §11) pertinent pour devices sans fp64 — peu probable que ça vaille le coup avec wgpu.

### Priorite 3 — Scope strategic (a arbitrer)

#### P3.1 — Architecture hybride bytecode ✅ FAIT (Sessions A-E + cleanup, 18 docs)
- [x] Bytecode 8-opcodes `Sqr/Mul/Store/AbsX/AbsY/NegX/NegY/Add` (`fractal/bytecode/mod.rs`).
- [x] `compile_formula` pour Mandelbrot/Julia/BS/Tricorn/Celtic/Buffalo/PerpBS/Multibrot puiss. entière.
- [x] Interpréteur CPU f64 + GMP (`bytecode/interp{,_gmp}.rs`).
- [x] BLA mat2 unifié via dual-numbers (`bytecode/bla_dual.rs`) — Vec multi-niveaux avec merge F3.
- [x] Delta-form interpreter (`bytecode/delta_form.rs`) — f64 + ComplexExp (`DeltaState`/`DeltaStateExp`).
- [x] Pixel loop unifié (`bytecode/pixel_loop.rs` + `pixel_loop_exp.rs`) — BLA mat2 + delta-form + rebasing F3.
- [x] Intégration end-to-end dans `delta.rs::iterate_pixel` avec cache thread-local.
- [x] Activé par défaut depuis Session E. Pixel-perfect ou diff < 5 % vs legacy.
- [x] **ComplexExp dans pixel_loop** pour deep zoom > 1e13 — pixel-perfect à zoom 1e15 et 1e30.
- [x] **Multi-phase** infrastructure (`Formula::hybrid`) prête (UI/CLI manque).
- [x] **GPU bytecode runtime intégré** (`pipeline_bytecode` + `bytecode_kernel.wgsl`) — Mandelbrot/Julia/BS pixel-perfect ou diff < 1.5 %. Étend la couverture GPU à Tricorn/Celtic/Buffalo/PerpBS/Multibrot sans shaders dédiés.
- [x] **Dual numbers dans bytecode** (`iterations.rs::iterate_via_bytecode`) — distance estimation, interior detection, orbit traps pixel-perfect vs legacy en f64 standard.
- [x] **Cleanup partiel** : suppression `use_legacy_glitch_detection` field + CLI flag.
- [ ] **Reste mineur** : suppression complète de `glitch.rs` et `nonconformal.rs`. Bloquée par `iterate_pixel_with_duals` (perturbation + dual numbers) qui les utilise encore. Pour les retirer définitivement : étendre `pixel_loop.rs` avec dual-numbers tracking en mode perturbation (~300 lignes, session dédiée).

#### P3.2 — Wisdom-driven backend selection
- [ ] Benchmark `(device, type)` au premier run, JSON persiste.
- [ ] Choisir le type le plus rapide viable pour le zoom courant.
- **Pourquoi** : vraie valeur sur materiel exotique, pas critique sur desktop moderne.

#### P3.3 — EXR raw export
- [ ] Format compatible KFR/zoomasm pour assemblage vidéo-zoom.
- [ ] Permet de recolorer/animer sans re-rendre.
- **Pourquoi** : niche mais différenciateur si tu cibles les createurs de zoom-videos.

### A NE PAS regresser (superset assume vs F3)

- 27 palettes built-in + RGB/HSB/LCH (UX win).
- 15 modes coloring + 4 orbit traps built-in.
- 7 plane transforms XaoS-style.
- Drag-and-drop PNG avec metadata JSON.
- Catalogue de formules non-escape-time : Newton, Phoenix, Magnet, Lyapunov (6 presets), Buddhabrot/Nebulabrot/Anti, Von Koch, Dragon, Pickover Stalks, Nova, Sin, Alpha Mandelbrot, Barnsley, Mandelbulb — F3 n'y va pas, c'est notre terrain de differenciation.
- Preview Julia au survol + raccourci `J`.
- Recolorisation asynchrone.

---

## Format / I/O

- [ ] **TOML parameter files** : F3 utilise TOML (et KFR pour raw + params). On a deja un loader TOML (cf. commit `ef5ffea`). Verifier compatibilite avec format F3 si interop souhaitee.

---

## Ordre d'attaque recommande

État au 2026-05-17 : P3.1 fondations + P1.1 + P1.2 + P1.4 livrés via 14 commits
sur la branche `Stable` (P3.1 Sessions A-E). Le path bytecode unifié est
activé par défaut. Voir `docs/p3-1-task6-session-{a,c,e}-findings.md`.

Pour fermer P3.1 complètement :

1. **ComplexExp dans `pixel_loop.rs`** pour deep zoom > 1e13. Débloque le retrait
   de `glitch.rs` (utilisé seulement par GMP perturbation path).
2. **Dual numbers dans `pixel_loop.rs`** pour distance estimation / interior
   detection. Débloque le retrait de `iterate_pixel_with_duals`.
3. **GPU bytecode integration** : encoding `bytecode[]` storage buffer +
   nouveau pipeline dans `gpu/mod.rs`. Supprime les 3 shaders dupliqués.
4. **Multi-phase / hybrides** : UI/CLI pour définir une formule multi-phases
   (ex. `--phases mandelbrot,burning_ship,burning_ship`), `Vec<BlaTableUnified>`
   par phase. Feature unique vs Kalles Fraktaler.
5. **Cleanup final** : supprimer `glitch.rs`, `nonconformal.rs`, params CLI
   `--no-legacy-glitch-detection`, champ `use_legacy_glitch_detection`.

Reste P1.3 (constantes F3 quick wins) et P1.5 (AA subframes) — non-bloqués.
