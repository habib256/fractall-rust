# TODO

## Dette technique / vigilance

- [ ] **Découpe `gui/app.rs` (2854 lignes)** : trop gros, candidat a refacto avant que la dette ne s'installe. Pistes : extraire le menu Type, la gestion drag-and-drop, le rendu HQ asynchrone, les raccourcis clavier dans des modules separes.
- [ ] **Découpe `perturbation/mod.rs` (1451 lignes)** : meme remarque. La fonction `render_perturbation_cancellable_with_reuse()` et le dispatch CPU/GMP peuvent etre eclates.
- [ ] **Mettre en place une CI** : sur ce projet les regressions numeriques sont silencieuses (un pixel faux != un crash). Workflow minimal :
  - `cargo test --release --lib` sur push/PR
  - Tests d'images de reference : rendre quelques locations connues (depuis `locations/`) et comparer un hash (perceptual hash ou hash exact sur des zones stables)
  - Coût faible, gain enorme pour detecter les regressions sur perturbation / BLA / glitch.
- [ ] **GPU f32 uniquement : documenter explicitement le tradeoff**. Probablement choix conscient (compatibilite Metal qui ne supporte pas f64), mais il faudrait :
  - Documenter dans le README / CLAUDE.md la limite de zoom GPU (~10^7 en f32)
  - Indiquer clairement le seuil de fallback CPU dans la doc utilisateur
  - Eventuellement evaluer si `softfloat` (cf. Fraktaler-3 §11) serait pertinent pour les devices sans fp64

## Aligner sur Fraktaler-3 (source de verite)

Voir `docs/fraktaler-3-analysis.md` pour le detail. Ecarts notables a evaluer :

- [ ] **Rebasing proactif vs glitch detection a posteriori** : Fraktaler-3 a abandonne Pauldelbrot et le clustering au profit d'un rebasing par pixel/iteration (`if |Z+z| < |z| → reseat`). Notre implementation (`glitch_tolerance`, `max_secondary_refs`, `min_glitch_cluster_size`, `glitch.rs`) suit l'ancienne approche Kalles Fraktaler. Decision a prendre : garder, hybrider, ou migrer.
- [ ] **BLA matrices 2x2 partout** : F3 utilise `mat2<real>` unconditionnellement, meme pour Mandelbrot (conformal). Notre code branche conformal/non-conformal. A unifier pour simplifier ?
- [ ] **Formule de precision GMP** : la valeur reelle dans F3 est `prec = max(24, 24 + (zoom * height).exp)` sans clamp `[128, 8192]`. Notre clamp est une protection raisonnable mais a documenter comme un ecart conscient.
- [ ] **Reference bailout `1e10` hardcoded** : F3 utilise 1e10 pour le bailout de l'orbite reference (different du bailout pixel `ER=25`). Verifier notre valeur.
- [ ] **Series approximation** : F3 ne l'implemente pas (BLA suffit, et series est incompatible avec hybrides). Notre `series_order` est un superset legitime, a garder.
- [ ] **Pixel spacing pour BLA** : F3 utilise `4/zoom/height` (verbatim `engine.cc`). A reverifier apres bugfix #11.
- [ ] **Seuil BLA `1/2^24 ≈ 6e-8`** : hardcoded dans F3, non type-dependant. Notre `bla_threshold = 1e-8` est plus strict mais comparable.
- [ ] **Architecture hybride / opcode interpreter** : F3 n'a pas d'enum FractalType ; tout est compile en bytecode (8 opcodes : add, store, sqr, mul, absx, absy, negx, negy) et tous les Mandelbrot/Burning Ship/Tricorn/Celtic/Buffalo/Multibrot/hybrides decoulent de combinaisons de phases. C'est un changement architectural majeur, a evaluer pour une v2 si on veut ajouter des hybrides.
- [ ] **`diffabs(c, d)` pour Burning Ship perturbation** : 4-case piecewise qui evite la cancellation catastrophique sur `|c+d| - |c|`. Verifier qu'on a un equivalent.
- [ ] **Phase-aware reference orbits** : pour hybrides, F3 stocke `k` orbites (une par rotation de phase) et le rebasing choisit entre elles. Pas applicable tant qu'on n'a pas d'hybrides.
- [ ] **Wisdom-driven backend selection** : F3 benchmark chaque `(device, type)` au premier run et choisit le plus rapide viable. Tres elegant, a considerer si on veut multi-GPU.

## Format / I/O

- [ ] **TOML parameter files** : F3 utilise TOML (et KFR pour raw + params). On a deja un loader TOML (cf. commit `ef5ffea`). Verifier la compatibilite avec le format F3 si on veut interop.
- [ ] **EXR raw export** : F3 sauve les donnees brutes d'iteration en OpenEXR (compatible KF2+ et `zoomasm`). Permet de recolorer sans re-rendre. A considerer pour les rendus longs.
