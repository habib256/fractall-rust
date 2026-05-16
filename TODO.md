# TODO

> **Objectif declare** : etre le meilleur renderer deep-zoom open-source en Rust. Fraktaler-3 = source de verite algorithmique (cf. `docs/fraktaler-3-analysis.md`).

---

## Roadmap strategique

### Priorite 1 — Qualite numerique deep-zoom

Changements qui rendent les images plus correctes et les zooms plus profonds plus rapides.

#### P1.1 — Migrer vers le rebasing proactif (gros chantier, gros gain)
- [ ] Implementer le rebasing F3 (`if |Z+z| < |z| → reseat sur reference la plus proche, m=0`) a cote de l'existant, derriere un flag `use_rebasing: bool`.
- [ ] Tests A/B sur les locations connues (`locations/`) : comparer images, profondeurs atteintes, temps de calcul vs path Pauldelbrot+clustering.
- [ ] Si valide, retirer `glitch.rs`, `glitch_tolerance`, `max_secondary_refs`, `min_glitch_cluster_size`, `min_glitch_cluster_size` des params + UI.
- **Pourquoi** : F3 a demontre (blog 2022-02-21) que le rebasing elimine la classe entiere de bugs glitch. Code plus simple, moins de params a tuner, prerequis pour hybrides.
- **Risque** : peut etre moins efficient sur certaines locations specifiques. Rare.

#### P1.2 — Unifier BLA en `mat2` partout
- [ ] Remplacer le path conformal (complex) par `mat2<real>` partout, comme F3.
- [ ] Verifier que `sup`/`inf` (normes operateur 2 via trace/det de AᵀA, formule fermee — cf. analyse §4) sont implementees.
- [ ] Retirer la branche conformal/non-conformal dans `bla.rs`, `delta.rs`, `nonconformal.rs`.
- **Pourquoi** : simplification code, prerequis hybrides, coût constant ~1.5-2× sur Mandelbrot pur (negligeable vs gains BLA eux-memes).

#### P1.3 — Aligner les constantes critiques sur F3 (quick wins, <1h chacun)
- [ ] Reference bailout = `1e10` hardcoded dans `hybrid_reference` equivalent.
- [ ] Escape radius pixel par defaut = `25` (carre 625) si pas deja le cas — meilleur conditionnement smooth.
- [ ] BLA threshold = `1/2²⁴ ≈ 6e-8`, independant du type (actuellement `1e-8`).
- [ ] Precision GMP : retirer le clamp `[128, 8192]` ou passer a `[24, ∞]`.
- [ ] Pixel spacing BLA = `4/zoom/height` strictement (reverifier apres bugfix #11).

#### P1.4 — Verifier `diffabs` Burning Ship
- [ ] Verifier qu'on a un equivalent de `diffabs(c, d)` 4-cas piecewise dans `nonconformal.rs` ou `delta.rs`.
- **Pourquoi** : `|c+d| − |c|` en perturbation Burning Ship peut subir cancellation catastrophique. F3 a la fonction depuis 2018-01-04 (blog).

#### P1.5 — Anti-aliasing par subframes jitterés
- [ ] Implementer wrapper N samples avec offsets jitterés (`burtle_hash`/`radical_inverse` ou equivalent low-discrepancy) → moyenne.
- [ ] Combinable avec progressive rendering (1 subframe = 1 pass).
- **Pourquoi** : AA propre pour les bords fins, surtout en mode DE/Distance. F3 le fait par defaut.

### Priorite 2 — Infrastructure

#### P2.1 — CI + tests d'images de reference (non-negociable)
- [ ] GitHub Actions : `cargo test --release --lib` sur push/PR.
- [ ] Job rendu d'images : 5-10 locations dans `locations/`, headless, comparer hash exact (ou diff perceptuel avec seuil) vs golden image versionnee.
- [ ] Inclure zooms profonds (10¹⁰, 10¹⁵, 10²⁰) sur Mandelbrot/BS/Tricorn/Julia pour couvrir tous les paths.
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

#### P3.1 — Architecture hybride bytecode (le gros refactor v2.0)
- [ ] Decision strategique avant tout autre P1 majeur : on s'engage ou pas ?
- [ ] Remplacer enum `FractalType` (33 valeurs pour la famille escape-time) par systeme phase + bytecode 8-opcodes : `add, store, sqr, mul, absx, absy, negx, negy`.
- [ ] Compilateur formule → bytecode (cf. F3 `param.cc::compile_formula`).
- [ ] Interpreteur CPU + generateur kernel GPU a partir du bytecode.
- [ ] Phase-aware reference orbits : pour hybrides, stocker `k` orbites (une par rotation de phase), rebasing choisit entre elles.
- **Pourquoi** : permet hybrides first-class (Mandelbrot×3 + BS×2, etc.) — feature unique vs Kalles Fraktaler et competiteurs. Code unifie. Newton/Phoenix/Magnet/Lyapunov/Buddhabrot/Mandelbulb restent en codepaths speciaux.
- **Coût** : refonte de `fractal/types.rs`, `iterations.rs`, `perturbation/`, `gpu/`. 1-3 semaines.
- **Si on s'engage** : faire avant P1.1/P1.2 pour eviter de refactorer deux fois.

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

1. **Cette semaine** : P1.3 (quick wins constantes) + P2.1 (CI + golden images) en parallele.
2. **Decision strategique** : P3.1 (hybride bytecode) on s'engage ou pas ?
3. Si P3.1 = oui → faire P3.1, puis P1.1 + P1.2 dans la foulee.
4. Si P3.1 = non/plus tard → P2.2 (decoupe), puis P1.1 (rebasing), puis P1.2 (BLA mat2), puis P1.4 (diffabs), puis P1.5 (AA subframes).
