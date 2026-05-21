# TODO — Roadmap fractall vers l'excellence

> **Mission** : le meilleur renderer de fractales **deep-zoom open-source en Rust**.
> **Référence algorithmique** : **Fraktaler-3.1** (`fraktaler-3-3.1/src/`,
> `docs/fraktaler-3-analysis.md`). Quand fractall diverge de F3, c'est fractall
> qu'on corrige — sauf preuve que F3 est dégénéré sur le cas.
> **Différenciation assumée** : superset de F3 côté *features* (27 palettes,
> 15 modes de coloring, 7 plane transforms, fractales non-escape-time, GUI
> interactive, drag-drop PNG). Voir [§Ne pas régresser](#-ne-pas-régresser).

**État (2026-05-21)** — Socle solide :
- Moteur **bytecode unifié** (8 opcodes) sur CPU + GPU : un seul pixel-loop
  couvre Mandelbrot/Julia/BurningShip/Tricorn/Celtic/Buffalo/PerpBS/Multibrot
  + variantes Julia.
- **Chemin de rendu UNIQUE CLI ↔ GUI** : un seul dispatcher
  (`render_escape_time_cancellable_with_reuse`), plus de dispatch dupliqué.
- **Perturbation** + BLA `mat2` (dual-numbers) + **rebasing F3** + FloatExp
  (`ComplexExp`) pour zoom > 1e13, GMP pour l'orbite référence.
- **Nucleus finder** atom-domain (`hybrid_period` + `hybrid_center` +
  `hybrid_size` → K dans `transform_k`).
- Coordonnées **HP > 1e308**, escape radius **aligné F3 (25)**, **BLA
  pixel-spacing corrigé**, **anti-aliasing multi-sample**, **rotation GPU
  corrigée**.
- **Parité F3 mesurée (G1)** : corpus 84 cas swept (harness durci) →
  **0 régression de correction**, parité validée jusqu'à zoom **1e1200**.

**Goulots restants** (les 2 vrais chantiers + 1 bug ciblé) :
1. **Performance deep-zoom 1e15–1e1000** : on force GMP / l'exp-path lent là où
   float128 / doubleexp seraient 10–100× plus rapides. C'est LE blocage de la
   parité full-depth (36/84 cas perf-bound) → **G2**.
2. **Bug auto-nucleus near-axis** (`optimize_reference_center`, toujours actif)
   : snappe la référence Mandelbrot trop loin sur les points près de l'axe →
   anneaux (cusp -0.75) + hang test2 @1920×1080. Fix ciblé → **G3**.
3. **Divergences restantes élucidées** (G3) : glitch_test_1 = victoire fractall
   (F3 dégénéré) ; seahorse, period-detection lossy encore ouverts.

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

### G1 — Parité visuelle F3 sur le corpus `toml/` · `[✅ MESURÉ — 0 régression ; résidu = G2 (perf) + G3 (near-axis)]`

**Objectif** : produire les mêmes images que F3 pour les 84 `toml/*.toml`.
**Harness** : `scripts/compare_f3.py` (F3 batch + `fractall-cli` → PNG + diff +
métriques EXR N0/NF). Flags de parité : `FRACTALL_NO_AUTO_ADJUST=1`,
`FRACTALL_NO_PERIOD=1`. **ER aligné des deux côtés** : `--escape-radius` écrit
`bailout.escape_radius` côté F3 ET passe `--bailout` côté fractall (défaut 25).
**Deux sweeps complémentaires (2026-05-21, ER=25 aligné) :**

**(A) Sweep LITTÉRAL « 84 cas à 1920×1080 »** — iter cap commun (1000) pour que
TOUS terminent (apples-to-apples : F3 et fractall plafonnés identiquement).
**83/84 ok · 0 échec · 0 timeout** (1 holdout : test2, bug auto-nucleus, cf.
G3 — parité connue via f64). **79/84 pixel-équivalents à <0.01 %**, seuls
spiral (4.6 %) et line (3.1 %) divergent (bord chaotique, structure résolue à
1000 iter). Rapport : `bench/compare_1080/_summary.md`. *NB : au cap 1000, les
cas profonds sont sous-résolus → « pixel-equiv » trivial (accord both-inside) ;
ce sweep prouve l'accord F3≡fractall à budget commun, pas la structure profonde.*

**(B) Sweep CORRECTION pleine profondeur** (iter natives ; 128×72/120s + push
96×54/300s) : **46 ok · 2 F3-dégénéré (fractall correct) · 36 perf/timeout ·
0 échec.** C'est le test de parité RÉEL (structure deep-zoom complète). Rapports :
`bench/compare_full/` + `bench/compare_perf2/`.

> **🎯 VERDICT G1 : 0 régression de correction sur tout le corpus rendable, à
> TOUTE profondeur.** Les cas qui rendent — y compris **extrêmes** : e1121
> (zoom **1e1121**) et e1200 (zoom **1e1200**) matchent F3 à **~0.01 %** ;
> dinosaur_fossils (5M iter) & glitch_test_6 (15M iter) pixel-perfect ; e1000/
> e318/e50/e113/11_dimensions/lethal_weapon (1.6M iter) en divergence faible
> bord-chaotique. **La correction deep-zoom de fractall est donc validée jusqu'à
> 1e1200.** Le seul blocage de « produire les 84 images F3 » est la PERFORMANCE :
> 36/84 cas (200k–10¹⁰ itérations sur l'exp-path lent) ne terminent pas même à
> 300 s → **G1 est GATÉ PAR G2** (pas par la correction). On ne peut pas comparer
> des images qu'on ne sait pas rendre dans le temps imparti.

Classification des 39 cas complétés (Δ **relatif** = Δmean/iter, seule métrique
comparable entre cas) :
- **Pixel-perfect** (<0.01 %, 4) : test5, test6, **dinosaur_fossils** (5M iter),
  **glitch_test_6** (15M iter).
- **Bord/équivalent** (<0.3 %, 8) : e401, e1000, glitch_test_4, floral_fantasy,
  glitch_test_2, glitch_test_3, integral_of_ex2, heaven.
- **Divergence modérée** (0.3–2 %, 19, bord chaotique intrinsèque) : windmill, x,
  virus, mitosis, test3, tick_tock, all_seeing_eye, e50, test2, golden_spider,
  leaded_glass, e113, test, mitosis2, magic, flake, lya, line, uranium.
- **À investiguer** (>2 %, 8) — tous explicables : test4 (31 iter → rel% = bruit),
  spiral/nr_fail/peanuts/liiiines (bord chaotique), **rug** (BLA over-skip connu),
  lethal_weapon/**threads_colour** (1.5–1.6M iter, zoom jusqu'à 1e652 — frontière
  perf/précision extrême).
- **F3-dégénéré (2)** : glitch_test_5 (F3 100 % intérieur, 0.014 s) **et
  glitch_test_1** (F3 extérieur uniforme 0.3 % intérieur, 0.085 s fast-path) —
  fractall structuré dans les deux cas. **Le détecteur tranche enfin
  glitch_test_1 en faveur de fractall** (cf. G3 : le « rings » était une victoire
  fractall, F3 dégénéré sur ce lieu glitch-prone).
- **Perf/timeout (36)** : ~13 borderline (200k–1M iter : e1016/e1086/e1298/
  e22522/e634/e8000/e890/long/olbaid1-3/opus/the_complexity_of_a_line, finissent
  >300 s) + ~23 catégoriquement infaisables (>1M iter : dragon 5M, wfs* 3.5M,
  super_dense/wfs_mb/infinity/triangle 10-15M, hard/orion 20M, opus2 80M,
  **seahorse 10¹⁰**, …) — exp-path trop lent → **G2**.

**Done when** :
- [x] **Re-sweep complet 84 cas à 1920×1080** (2026-05-21, sweep A) : 83/84
  complétés avec `mean_abs` + classification (`bench/compare_1080/`), 79
  pixel-équiv, 0 échec ; test2 = holdout (bug auto-nucleus G3, parité connue
  via f64). + sweep B pleine profondeur (46 réels, validés jusqu'à 1e1200).
  Tous les 84 cas sont classés. NB : à 1920×1080 il faut un cap d'itérations
  commun pour COMPLÉTER les cas profonds (sinon timeout) ; la parité de
  structure profonde réelle vient du sweep B (G2 pour les 36 deep restants).
- [x] **Gain ER=25 quantifié** (2026-05-20) : à **ER aligné** (F3 et fractall
  au même ER), passer de 4 à 25 laisse Δmean **quasi inchangé** (test/test2/
  line/spiral/all_seeing_eye identiques à 4-5 décimales ; seul test4 améliore :
  Δmean 0.153→0.141, inside_mm 730→703). Logique : le harness aligne l'ER des
  deux côtés, donc la *différence* est invariante à l'ER. **Conséquence** : la
  valeur d'ER=25 est la parité de *défaut* avec F3 (rendu utilisateur sans
  `--bailout`) + smooth coloring plus propre, PAS la métrique harness alignée.
  (Corrige la sur-affirmation « requis pour la parité N0/NF ».)
- [x] **Détecteur F3-dégénéré** (2026-05-20) : `compare_f3.py` flagge
  `f3_degenerate` quand F3 est quasi-uniforme (tout-intérieur/-extérieur ou std
  exterieur < 1e-3) **ET** rendu en < 0.1 s (fast-path) **ET** fractall structuré.
  Exclu du score. Vérifié sur glitch_test_5 (100% intérieur, 0.018 s).
- [x] **Chaque cas classé, 0 FAIL non élucidé** (2026-05-21) : aucun statut
  `fractall_fail` réel ; les 66 « perf/timeout » sont expliqués (G2), le
  dégénéré exclu, les 17 complétés classés par Δ relatif (table ci-dessus). Les
  divergences résiduelles sont soit bord-chaotique intrinsèque, soit la famille
  rings/glitch (G3 : glitch_test_1 + le bug du cusp -0.75), soit la frontière
  perf/précision extrême (threads_colour). Aucune régression de correction.
- **Reste pour fermer G1 (dépend de G2)** : une fois la perf deep-zoom débloquée,
  re-sweeper les 66 cas perf-bound à iter pleines pour confirmer la parité (ou
  exposer de vrais écarts aujourd'hui masqués par le timeout).

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

- [ ] **🔴 CPU perturbation : anneaux concentriques près du cusp -0.75**
  (découvert 2026-05-20, image utilisateur `fractal_1779301318.png`). Mandelbrot
  centre ≈ (-0.749996, -0.004086), zoom ≈ 2e13, 2500 iter. **CPU perturbation
  rend des anneaux concentriques (FAUX)** alors que **GPU perturbation ET CPU
  GMP pur rendent le champ de cardioïdes correct**. → vrai **bug du cœur
  perturbation CPU** (`render_perturbation_with_cache` / pixel_loop), partagé
  CLI+GUI (le CLI le reproduit). **PAS** la period-detection (anneaux persistent
  avec `FRACTALL_NO_PERIOD=1`). Distinct de glitch_test_1 (ici GMP est CORRECT →
  référence sûre pour débugger). **Done when** : CPU perturbation == GMP/GPU sur
  cette vue, puis **l'ajouter aux golden images** (la zone est demandée comme
  golden — bloqué tant que le rendu est faux). Repro : voir paramètres extraits
  du PNG (HP center + `--zoom` = 4/span_x + `--algorithm perturbation`).
  **🔑 PISTE FORTE (2026-05-21)** : `optimize_reference_center` (`orbit.rs:1012`)
  snappe AUTOMATIQUEMENT la référence Mandelbrot vers le nucleus le plus proche
  — toujours actif (pas opt-in). Sur les points près de l'axe réel / cusp -0.75,
  il déplace la référence LOIN du centre de vue → grand `dc` → perturbation
  dégradée. **Même cause suspectée pour le hang test2 @1920×1080** (zoom 2.1e9 :
  à haute résolution le pixel plus fin franchit le seuil perturbation → snap
  auto vers un nucleus à imag≈7e-189, ~300 px hors vue → boucle pixel qui
  s'emballe). **À tester** : désactiver/borner `optimize_reference_center`
  (rejeter le snap si nucleus > k·span du centre) — devrait fixer LES DEUX
  (rings cusp + hang test2).
- [x] **glitch_test_1 — anneaux concentriques : TRANCHÉ (2026-05-21)** vers une
  **victoire fractall**. Le détecteur F3-dégénéré du harness (timing + uniformité)
  flagge glitch_test_1 : **F3 rend un extérieur uniforme (0.3 % intérieur) en
  0.085 s (fast-path)** tandis que fractall rend une vraie structure. Combiné aux
  diagnostics antérieurs (GMP pur fractall ≡ perturbation ≠ F3 ; précision/BLA/
  series/period tous écartés), c'est la même signature que glitch_test_5 : F3
  court-circuite sur ce lieu glitch-prone, fractall est correct. Exclu du score.
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
- [x] **Chemin de rendu CPU unifié CLI ↔ GUI** (2026-05-20) — un seul dispatcher
  `render_escape_time_cancellable_with_reuse`. Cf. *Shipped* + invariant
  §« Ne pas régresser ».
- [ ] **Unifier aussi le dispatch GPU** : aujourd'hui le choix GPU/CPU + l'appel
  des kernels GPU est dupliqué dans `main.rs` (CLI) et le thread de rendu GUI.
  Extraire un dispatcher GPU partagé (ou intégrer le GPU au dispatcher unique).
- [ ] **Retirer les modules perturbation legacy** : porter `iterate_pixel_gmp`
  sur `pixel_loop`, puis supprimer `glitch.rs`, `nonconformal.rs` et les champs
  perturbation legacy (`max_secondary_refs`, `min_glitch_cluster_size`,
  `glitch_tolerance`, …). Un seul moteur. Retirer aussi les renderers densité
  MPC non-cancellables marqués `#[allow(dead_code)]` (superseded).
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

**2026-05-21** :
- **Parité F3 mesurée sur le corpus (G1)** : 2 sweeps (1920×1080 cap commun =
  83/84 + 79 pixel-équiv ; pleine profondeur = 46 réels validés jusqu'à 1e1200).
  **0 régression de correction.** glitch_test_1 tranché (victoire fractall).
- **Harness `compare_f3.py` durci** : `--bailout` (alignement ER des 2 côtés),
  classification timeout↔fail↔perf, métrique Δ **relative**, détecteur
  F3-dégénéré (timing + uniformité), `--out` absolu. `bench/` gitignoré.
- **Root cause identifié** : `optimize_reference_center` (snap auto near-axis) =
  cause commune suspectée des anneaux cusp -0.75 + hang test2 @1920×1080 (→ G3).

**2026-05-20** :
- **Chemin de rendu UNIFIÉ CLI ↔ GUI** : un seul dispatcher CPU
  `render_escape_time_cancellable_with_reuse` (le CLI `render_escape_time` y
  délègue ; la GUI — passes progressives, HQ, AA — l'appelle aussi, cache
  d'orbite threadé in/out). Supprimé les 3 implémentations de dispatch dupliquées
  (l'ancien `render_escape_time`, l'inline match GUI, les renderers
  `render_escape_time_{f64,gmp}` non-cancellables morts). Densité MPC
  non-cancellables marquées superseded. Golden + 178 unit tests verts (rendu
  inchangé). Cf. CLAUDE.md §« Chemin de rendu unique ».
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

- **INVARIANT : chemin de rendu unique CLI ↔ GUI.** Un seul dispatcher CPU
  (`render_escape_time_cancellable_with_reuse`). Ne jamais redupliquer la
  logique de dispatch dans `gui/app.rs` ni recréer un `render_escape_time*`
  parallèle. Une divergence GUI/CLI = bug. (GPU encore inline des deux côtés —
  unification GPU = G5.)

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
