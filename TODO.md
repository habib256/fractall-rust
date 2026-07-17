# TODO — Roadmap fractall vers l'excellence

> **Mission** : le meilleur renderer de fractales **deep-zoom open-source en Rust**.
> **Référence algorithmique** : **Fraktaler-3.1** (`fraktaler-3-3.1/src/`,
> `docs/fraktaler-3-analysis.md`). Quand fractall diverge de F3, c'est fractall
> qu'on corrige — sauf preuve que F3 est dégénéré sur le cas.
> **Différenciation assumée** : superset de F3 côté *features* (27 palettes,
> 15 modes de coloring, 7 plane transforms, fractales non-escape-time, GUI
> interactive, drag-drop PNG). Voir [§Ne pas régresser](#-ne-pas-régresser).

---

## 🧭 État actuel (2026-07-17) — lire ceci en premier

**Moteur : 0 gap mesuré.** harness quick + standard-speed + wisdom-optimality
verts, bat F3 partout (geomean speed ~0.18, 25/25 wins). 271 unit CLI + 24 golden
pixel-exact + quality 15/15 PASS.

**FAIT :**
- **Correction (G1-G3)** : parité F3 sur le corpus (84 cas), 0 régression, validée
  jusqu'à zoom **1e1200**. Période Brent OFF par défaut (opt-in `FRACTALL_PERIOD=1`).
- **Perf deep-zoom (G2)** : **rebase-at-end F3** → plus de fallback GMP par-pixel
  (e50 544→1.6 s, e1000 742→0.5 s, dragon ~6 h→6.5 s à 256²). Escalade **tier dd**
  quand `glitch_ratio > 0.30` (Mandelbrot bytecode) au lieu du full-GMP per-pixel
  (~4-8× plus rapide, pixel-exact GMP ; `perturbation/mod.rs:1777`).
- **Wisdom (G9.1-G9.5)** : plan unique {device, algo, tier, variantes} +
  benchmarks machine (`--wisdom-bench` → `~/.config/fractall/wisdom.toml`).
  **Auto-device G9.5** : `wisdom::select_device(params, gpu_available)`
  (`wisdom.rs:294`) arbitre CPU/GPU par débit benché SOUS garde-fou correction
  (GPU routé uniquement dans la plage deep both-perturbation ~1e12–4e37, JAMAIS
  sur shaders std f32 24 b). CLI `--gpu`/`--no-gpu` = overrides, sinon Auto ;
  GUI menu « Tech: 🔄 Auto ». Sur GPU grand public (f64 1:64) l'auto reste CPU.
- **GUI temps-réel (G10)** : réutilisation orbite (G10.2), recolorisation
  sans clone (G10.3), pixels XaoS colonnes/lignes (G10.4/b), file de tuiles
  priorité-centre + streaming (G10.5), warp GPU molette (G10.1, signe Y corrigé).
- **Hybrides (G4 jalon 1-5a)** : les hybrides multi-phase **RENDENT** (CLI
  `--phases mandelbrot,burning_ship`) via le path f64 standard. `hybrid_phases` +
  `formula_for_params` + `compile_hybrid_formula`. [M,M]==Mandelbrot pixel-exact.
  Deep OK en perturbation sur TOUT le range : f64 (jalon 3, ~1e13–1e280) +
  ComplexExp (jalon 4, > 1e280, vérifié `[M,M]==M` @ **1e1000**). **Jalon 5a :
  les hybrides GENUINE ([M,BS]) deep sont CORRECTS** — réfs par phase (port F3)
  + tracking `(phase+m)≡n (mod N)` + gates `!is_hybrid` sur tous les chemins
  z²+c hardcodés ; verrou grille vs GMP-cyclant ([M,M] 160/160). Jalons
  5b-5f : BLA par phase (6.7×), éditeur GUI, atom-domain mat2, parité F3
  NATIF (Δ=0 @e13), nucleus phase-aware, + BLA radius scaling σ₁(K).
- **Durcissements** : gate `!bytecode_path` sur le 2e bloc glitch récursif
  (`mod.rs:1634`, supprimait ~3.4 % structure spurious à >512²) ; golden
  `mandelbrot_interior_ref_640` (seul cas >512², exerce l'escalade dd).

**RESTE :**
- **G4 : SOLDÉ INTÉGRALEMENT (2026-07-18)** — jalon 5g (fast-path inline
  multi-phase, [M,M] e50 boucle pixel 2.3× pixel-identique) + `Op::Rot`
  per-phase câblé (parseur opcodes F3 `--opcodes`/`[[formula]]`, verrou GMP
  160/160 + juge F3 natif).
- **G6** : durcir/étendre le corpus golden.
- **G9.6** : fiabilité → escalade tier auto (px→dd/frame→dd) — marginal.

---

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

**✅ Performance deep-zoom RÉSOLUE (2026-05-21, sur `main`)** — c'était LE
goulot. Cause : les cas perf-bound rendaient en **fallback GMP par-pixel** (réf
escape-time tronquée → `ref_truncated` → GMP). Fix (~20 lignes) : **rebase-at-end
F3** pour orbites escape-time (`pixel_loop_exp.rs`) → plus de fallback GMP.
**e50 544→1.57 s, e1000 742→0.53 s (pixel-identique GMP), dragon ~6 h→6.46 s** à
256². Les 36 cas « perf-bound » de G1 ne le sont plus. 178 unit + golden verts.
(Bonus : BLA aligned-lookup + table ~8× plus petite.)

**Goulots restants** :
1. **✅ Bug auto-nucleus test2 @1920×1080 — RÉSOLU (2026-05-22)**. Ce n'était pas
   qu'un hang : `optimize_reference_center` (auto-snap réf → nucleus voisin) rendait
   FAUX (test2 82 %, mandelbrot_perturb_1e6 93.8 % ≠ GMP — le golden p6 encodait
   l'image fausse). **Désactivé par défaut** (le rebase-at-end G2 subsume l'anti-glitch ;
   réf = centre de vue, standard F3). p6 golden corrigé. Re-sweep snap-off identique
   au baseline (0 régression).
2. **Divergences restantes élucidées** (G3) : glitch_test_1/5 = victoire fractall
   (F3 dégénéré) ; period-detection faux-positif RÉSOLU (tol 0.85). Seahorse =
   perf (10¹⁰ iter), pas correction.
   - ⚠️ **CORRIGÉ 2026-07-07 — la tol 0.85 ne résolvait PAS glitch_test_5.** Le
     harness le masquait (`NO_PERIOD=1` sur speed ET parité) alors que la PROD
     tournait Brent **ON** par défaut → glitch_test_5 (9.7e83) rendait **75 %
     des pixels FAUX vs GMP** (12259/16384 ; période OFF = **0 diff, pixel-exact
     GMP**). Grazes vs vraies périodes = indistinguables par seuil scalaire (4
     sessions, cf. G2). **Fix : Brent OFF par défaut** (opt-in `FRACTALL_PERIOD=1`
     pour le speedup risqué ; F3 n'utilise PAS ce Brent — il a l'atom-domain
     F3-exact `FRACTALL_ATOM_PERIOD`). Prod == config harness désormais (mesure
     honnête). Verrou : golden `mandelbrot_glitch5`. 196 unit + goldens + quality
     11 PASS 🟢. Coût : glitch_test_2 0.04→0.17 s en prod (mais glitch_test_5
     2.16→0.01 s ET correct). `orbit.rs::enable_period_detection`.
3. **✅ Re-sweep corpus complet** (G1) : 84 cas, 0 échec de correction. Seuls
   e22522 (hors plage précision, désormais **averti** au lieu de faux-silencieux)
   + 6 perf-timeout restent.

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

### G1 — Parité visuelle F3 sur le corpus `toml/` · `[✅ MESURÉ — 0 régression ; résidu perf DISSOUS par G2]`

> **MISE À JOUR (2026-05-21, après G2 résolu)** : les **36 cas « perf/timeout »**
> du sweep (B) ne le sont plus — ils tombaient en fallback GMP (réf évadante) ;
> avec le rebase-at-end F3 ils rendent en perturbation en **<7 s à 256²**
> (e1000 pixel-identique GMP, e50/e113 == GMP). **TODO : re-sweep complet pleine
> profondeur** (désormais tractable) pour confirmer la parité F3 sur ces 36 cas.

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
  spiral/nr_fail/peanuts/liiiines (bord chaotique), **rug** (glitch réf unique —
  cf. diag 2026-07-12 ci-dessous, l'étiquette « BLA over-skip » est INFIRMÉE),
  lethal_weapon/**threads_colour** (1.5–1.6M iter, zoom jusqu'à 1e652 — frontière
  perf/précision extrême).

> **🔬 DIAG rug 1e56 (2026-07-12) — « BLA over-skip » INFIRMÉ ; glitch réf unique.**
> vs **GMP pur** (ground truth, pas F3) : FAIL max_diff=298, p99=78, div_ratio=4.2 %,
> **escape_disagreement=0** (aucun pixel ne flippe intérieur/extérieur — pur écart
> de *compte* d'itération → banding). Signature : les pixels divergents sont des
> **paires point-symétriques** (±dc autour du centre) au **z_pert bit-identique**
> mais z_gmp différent → la perturbation impose une fausse symétrie δ↔−δ. 4 hypothèses
> réfutées par expérience contrôlée (chacune → résultat **bit-identique** au défaut) :
> **(1)** BLA off (`bytecode_f64` direct) ; **(2)** tier **dd** (Z 106 b non-arrondi
> dans `2·Z·δ`, `path=bytecode_dd` confirmé) ; **(3)** précision réf+GMP **1024 b** ;
> **(4)** **rebasing désactivé**. z_pert invariant aux 4 → l'écart est **structurel,
> pas numérique**. Cause réelle : **glitch de référence unique** — à n≈40327 la réf
> intérieure a une grande excursion (|Z|≈44 > ER) qui pousse `Z+δ` au bailout
> (fausse évasion) alors que le vrai pixel (GMP) s'évade à 40625 ; δ a décorrélé de
> `z_pixel−Z_ref` sans que ni le critère de rebase (`|Z+δ|²<|δ|²`) ni la validité BLA
> ne le captent. **fractall matche F3** ici (parité inside_mismatch=10/16384) → PAS
> une régression vs F3 ; divergence partagée, inhérente à la perturbation **réf
> unique**. Vrai correctif = **références secondaires / glitch-correction** (machinerie
> Pauldelbrot retirée au profit du rebasing F3-strict) — gros chantier, hors périmètre
> d'une itération. Distinct de la classe **dd-sensibilité** (seahorse/e50 : dd CORRIGE ;
> rug : dd n'aide PAS). Verrou impossible tant que non corrigé (le gate FAIL).
>
> **✅ SOUS-CAS RÉSOLU (2026-07-14) — escalade full-GMP des pixels encore glitchés.**
> Le correctif « références secondaires » complet reste un chantier, MAIS le sous-cas
> **fausse évasion** (blob intérieur faussement évadé, `escape_disagreement>0`) est
> résolu sans machinerie lourde : dans `render_perturbation_with_cache`, les pixels
> qui reviennent **encore glitchés** de la correction `iterate_pixel_gmp` (précision
> GMP sur la même réf → toujours glitché = glitch structurel réf-unique) sont escaladés
> vers `iterate_point_mpc` (full GMP par-pixel, **indépendant de la référence**). Avant,
> seul le sous-cas ref-exhausted (`cap_iter<iter_max`) déclenchait le full GMP ; le
> sous-cas δ-trop-grand sur réf pleine (`cap_iter==iter_max`) gardait la valeur glitchée.
> Débloqué par le fuzz `mandelbrot -0.615+0.401i zoom 6e7` (WARN div_ratio 0.78 %) et,
> **bonus**, corrige le golden `mandelbrot_cusp_m075` qui verrouillait 201 px FAUX
> (vs GMP : **FAIL** max_diff 1723 p99 1722 div_ratio 1.26 % → **near-exact** max_diff 159
> p99 0 div_ratio 0.006 %). Verrou : preset quality `single-ref-glitch-interior` +
> golden cusp régénéré. Coût : full GMP sur les seuls px encore glitchés (rare, borné,
> < seuil 30 % sinon fallback total). **Reste ouvert** : le sous-cas *banding* de rug
> (`escape_disagreement=0`, pas de flip intérieur/extérieur) — non couvert ici car les
> px n'y sont pas forcément flaggés glitchés ; à vérifier séparément (full corpus).
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
  - **e52465 quarantainé (2026-07-14) — infaisable AU PLANCHER GMP, pas un bug.**
    Preflight flip ok(48.5 s)→timeout : le « ok » du 2026-07-12 (89b15b1) datait
    d'AVANT la suppression du plafond de précision (e6f958c/db6407b) — rendu
    rapide-mais-FAUX à 65 536 b (l'orbite complète à 65 k b coûterait déjà ~574 s,
    le 48.5 s n'a jamais calculé l'orbite entière). Correct = 174 319 b × 2.87 M
    iters : mesuré **970 µs/iter** dans le moteur vs **860 µs/iter** GMP brut
    (`examples/bench_gmp_iter.rs`) → +13 % du plancher matériel, ~46 min
    d'orbite incompressible. Bisecté : PAS une régression (binaire pré-f9030c9
    identique) ; F3 timeout aussi (journal). Centre ≈ −2 (Misiurewicz, pas de
    période atom) → aucune troncature possible. Quarantaine = garde-fou harness
    conforme à la décision « aucune limite moteur ». Sous-produit : fix
    progression `Ref[%]` live (l'orbite restait affichée 0 % → ressemblait à un
    hang, c'est ce qui a coûté le diagnostic).
- **⚠️ Hazard mémoire / crash OS — RÉSOLU pour 3/5 cas (2026-07-10)** :
  RSS explosait — **wfs_mb 28 GB, orion 29.5 GB, opus2 28.6 GB** — faisant
  **tomber l'OS** pendant les sweeps `full` sur machine 40 GB. `harness.py
  preflight` les quarantaine (cap RLIMIT_AS) → les sweeps les skippent.
  - **CAUSE RACINE CORRIGÉE (2026-07-10)** : l'hypothèse « orbites GMP stockées
    pleine précision » était **FAUSSE** — le path bytecode ne stocke PAS le GMP
    dense (`store_dense_gmp=false`). Mesuré (`FRACTALL_MEMDBG`, wfs_mb) : **1453
    o/itér venaient de la `BlaTable` conformale historique** (orbit.rs:927,
    ~13 nœuds/itér × 112 o) — **jamais lue par le path bytecode** (qui a sa
    propre `BlaTableUnified`), pur poids mort. z_ref+z_ref_f64 = 48 o/itér ;
    le reste (2900→48 o/itér) = cette table. **Fix** : la SAUTER (`BlaTable::
    empty()`) quand `bytecode_path_label().is_some()` **et** `iteration_max >
    1 M` (sous 1 M on la garde : le shader perturbation GPU legacy la lit, et
    le GPU f32 n'est viable qu'à iter modérées — filet de sécurité : le GPU la
    rebuild localement si vide, `gpu/mod.rs`). Le fallback CPU legacy avec table
    vide reste CORRECT (num_levels()==0 → pas de saut BLA = pas direct = exact).
    **Résultats** (16², RSS mesuré via VmHWM) : wfs_mb 10 M **28 GB→2.0 GB**
    (14×), orion 20 M **29.5 GB→<6 GB** (exit 0 sous cap 6 GB), opus2 80 M
    **28.6 GB→13.9 GB**. Tous < seuil quarantaine 20 GB → **dé-quarantainés**.
    NB opus2 : le résidu venait du pic de BUILD de la `BlaTableUnified` *utilisée*
    (tous les niveaux ~2m nœuds coexistaient avant le clear des niveaux 0-2, F3
    `bla_skip_levels`).
    Verrous : goldens 10/10 pixel-exact (path <1 M inchangé, cf. e113 35 k iters),
    196 + 2 unit PASS (`conformal_bla_skip_tests`).
  - **✅ SUITE (2026-07-10) — pic de BUILD `BlaTableUnified` abaissé ~2m→0,75m** :
    le build matérialisait tous les niveaux (level0 = m nœuds = le plus gros) puis
    vidait 0-2 À LA FIN. Fix (`bla_dual.rs::build`) : (a) construire le **level 1
    en streaming** depuis l'orbite (2 single-steps fusionnés/nœud) SANS matérialiser
    le level 0 — bit-identique car `single(i)==level0[i]` ; (b) **vider les niveaux
    skip (1,2) DANS la boucle** dès qu'ils ont servi à merger le niveau du dessus.
    Pic build 2m→~0,75m nœuds. **opus2 80 M : 13.9 GB→8.0 GB (−42 %)**. Bénéficie
    à TOUT rendu perturbation deep (table BLA = terme dominant après le fix
    conformale). Verrous : goldens 10/10 pixel-exact, 32 tests `bla_dual*`
    (dont `table_build_levels_8_iterations`), 198 unit PASS.
  - **✅ SUITE (2026-07-10) — réservation d'orbite pathologique bornée
    (`orbit_reserve`)** : `Vec::with_capacity(iteration_max + 1)` réservait une
    capacité proportionnelle à `iteration_max` (venu du TOML utilisateur). Pour
    **seahorse** (`iterations=1e10` → clampé `u32::MAX≈4.3e9`), `z_ref` réservait
    **137 GB d'un coup** (`memory allocation of 137438953472 bytes failed`) **avant
    même de lancer l'orbite** → abort en 1 s qui, hors cap, faisait **tomber l'OS**
    pendant un sweep (et a collatéralement quarantainé e22522, in-flight au moment
    du crash). Fix (`orbit.rs`) : plafonner la pré-réserve à `MAX_ORBIT_RESERVE`
    (32 M entrées) sur les 8 sites `with_capacity` d'orbite — **no-op pour toute
    orbite légitime du corpus** (≤ ~15 M iters, glitch_test_6) qui réserve sa
    taille exacte ; seuls les `iteration_max` pathologiques sont bornés, le `Vec`
    croissant ensuite à la demande. seahorse ne réserve plus 137 GB → **échoue
    proprement** (timeout/`killed_oom` sous cap) au lieu de crasher l'OS. Verrou :
    `orbit_reserve_caps_pathological_iteration_max` (200 unit PASS), goldens 10/10.
  - **✅ HORS-PLAGE PRÉCISION RÉSOLU (2026-07-12) : plafond 65 536 → 262 144 b
    (`MAX_PERTURB_PRECISION_BITS`), e22522 + e52465 CORRIGÉS.** F3 n'a AUCUN
    plafond (`param.cc:132`) ; le nôtre datait d'une crainte mémoire infondée
    (l'orbite est stockée ComplexExp/f64, la précision ne porte que sur quelques
    temporaires GMP ≈ 32 Ko à 262 144 b). À pleine précision, l'atom-domain FIRE
    (la réf sous-précise cassait le critère) : e22522 réf 1 M→213 574 iters,
    orbite 87.6→52.7 s, **53.1 s vs F3 56.3 s (WIN, était 1.58× ET faux)**,
    parité Δmean=1.96/inside_mm=0 ✓ ; e52465 (174 322 b) réf 2.88 M→639 392,
    rend structuré en 576 s @96². GUI `hp_arith_precision` aligné sur le même
    plafond. Verrou : unit `precision_bits_covers_ultra_deep_corpus`.
  - **✅ e22522 DÉ-QUARANTAINÉ (2026-07-10)** : vérifié — rend en fractall en
    **92 s / pic RSS 1.23 GB @256²** (orbite GMP 88 s à 65 536 b, pixels 3.6 s),
    exit 0, **aucun OOM**. Ce n'était PAS un OOM fractall : `died_uncleanly`
    collatéral (in-flight quand seahorse a crashé l'OS via la réserve 137 GB,
    désormais corrigée). C'est un cas **hors-plage précision** (zoom 1e22522 →
    ~74 852 b requis > plafond 65 536 b ⇒ image dégradée/uniforme, *averti*),
    lent mais robuste et < timeout `full` (600 s). Tombstone `resolved.json`.
  - **✅ QUARANTAINE VIDE (2026-07-11) — corpus 84 cas 100 % memory-safe.**
    Preflight full (`f5b7c90`, cap 34 GB) : **les 84 cas rendent, exit 0, 0 OOM**,
    pic RSS max = opus2 **8.0 GB** (< seuil 20.4 GB). seahorse DÉ-QUARANTAINÉ :
    la réserve 137 GB corrigée, il rend en **216 s / pic RSS 5.1 GB @256²** (orbite
    escape-time ~50 M iters, prec 4667 b — SUFFISANTE pour 1e1392). Le mécanisme
    de quarantaine ne vise que les **crashers** (HARNESS.md) ; seahorse ne crashe
    plus → hors quarantaine (tombstone `resolved.json`, fix vérifié). But projet
    « corpus complet supporté » **atteint côté robustesse/mémoire**.
  - **✅ seahorse ADJUGÉ CORRECT (2026-07-12) — l'image uniforme est la VRAIE
    image, pas un bug.** Ground truth par GMP PUR per-pixel (probe 4×2, 8 px
    à ~49.7 M iters chacun, 4654 b, 38 min de calcul) : **max_diff=0** — la
    perturbation reproduit EXACTEMENT les itérations vraies. Le spread réel de
    la vue n'est que ~6 445 iters sur 49.7 M (1.3e-4 relatif) → une seule
    couleur, légitimement. L'ancienne hypothèse « nucleus-centering requis »
    est INFIRMÉE (il n'y a pas de structure à révéler à cette résolution).
    F3 crashe sur ce cas (sig 6) → fractall va au-delà de F3 ici. Piège
    d'instrumentation corrigé : le log quality « bits=256 » affichait le champ
    PLANCHER utilisateur, pas la précision effective (4654 b) — faux signal de
    probe invalide. Vitesse (216 s, orbite ~50 M iters GMP) : seul levier
    restant = float128/wisdom sur l'orbite, orthogonal.
  - Reste orthogonal pour la PERF (pas la mémoire) de ces cas intérieurs :
    period-aware reference (G2, cf. plus bas, 4 sessions brûlées — bloqué sur le
    critère atom-domain F3 exact).
  - **✅ REGISTRE `slow-safe` — déconflation timeout≠crash (2026-07-15, /improve)** :
    le sweep FULL vitesse (84 cas) ne laissait qu'UN gap : `e52465`
    quarantiné, étiqueté « crash/OOM connu » severité-2 robustesse. **Mesuré**
    (256², cap 20 GB) : e52465 (zoom 1e52465, 2.88 M iters) **rend correctement
    exit 0 en 661.7 s, pic RSS 237 MB** (path=bytecode_exp, orbite GMP tronquée
    atom-domain à 639 k pas × 174 325 b, single-thread = 660 s ; F3 aussi hors
    budget). **Ce n'est ni un crash ni un OOM** — juste hors budget TEMPS des
    sweeps. Cause du faux gap : `run_case_measured` renvoie `peak_rss_kb=None`
    sur timeout (le `time -v` est tué avec le process) → preflight ne peut pas
    distinguer lent-safe d'un runaway, donc conflatait `timeout` avec les
    crashes (ligne préflight `crashed = st in (…"timeout")`) ET le gap generator
    étiquetait tout quarantiné « crash/OOM ». Or `HARD_CRASH_OUTCOMES` EXCLUT
    déjà `timeout` — l'incohérence était interne. **Fix** : registre
    `harness/slow-safe.json` (miroir de `f3-degenerate.json`), attesté PAR
    L'OPÉRATEUR après mesure (exit 0 + pic RSS < cap) ; `score`/`preflight` le
    skippent SANS gap robustesse (`slow_safe` a `ratio=None`, exclu des agrégats
    comme f3-degenerate), affiché à part dans le scorecard ; sous-commande
    `harness.py slow-safe {list,add,remove}` (`add` retire de la quarantaine).
    e52465 reclassé quarantaine→slow-safe. **La mécanique OOM/crash est
    INCHANGÉE** (crash/OOM/RSS-haute → quarantaine severité-2 comme avant ;
    seule une branche skip attestée est ajoutée avant le run). Résultat : sweep
    full vitesse = **0 gap** (geomean quick 0.298, 10/10 wins ; e52465 correct
    hors enveloppe excellence perf ≤1e1000).
- **⚠️ Trou d'invariant garde-fou RÉPARÉ (2026-07-07)** : `quarantine.json` est
  versionné → il peut **dériver** (revert/reset/checkout git) et *dé-quarantainer
  silencieusement* un cas qui a fait tomber la machine. Cas réel : **e22522**
  (`died_uncleanly` sur le côté **F3** d'un sweep speed, 1e22522 → orbite GMP
  ~9 GB @74 852 b) journalisé mais **absent** de la quarantaine → il pouvait
  rejoindre un `full` sweep et re-crasher l'OS (preflight ne vette QUE le côté
  fractall, pas F3). Fix : le **journal append-only est la vérité de terrain** ;
  `reconcile_quarantine_from_journal()` (appelé au démarrage de `score`/`preflight`)
  ré-quarantaine tout cas `died_uncleanly`/`killed_oom`/`aborted`/`killed` non
  couvert, en respectant un **tombstone `resolved.json`** posé par
  `quarantine remove` (fix attesté → pas de ré-ajout, sauf incident postérieur).
  Verrou : `scripts/test_harness_guard.py` (5 scénarios). e22522 ré-exclu.

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

### G2 — Performance deep-zoom · `[✅ RÉSOLU 2026-05-21 — rebase-at-end F3, fallback GMP éliminé]`

> **🏁 Sweep vitesse ciblé i7-10700F (2026-07-12) — les résidus « Xeon 4c » sont
> DISSOUS sur cette machine.** 16 cas deep orbite-bound à 256² (`--axes speed`),
> **tous WIN vs F3** (ratio fractall/F3, <1 = win) : threads_colour **0.06**,
> e634 0.06, e533 0.07, olbaid4 0.08, lethal_weapon 0.08, e1086 0.11, e1016 0.13,
> e890 0.14, e1298 0.14, rug 0.16, wfs 0.25, wfs2 0.28, olbaid5 0.31, wfs4 0.33,
> e227 0.34, wfs_extended 0.48. Parité de ces cas : 14/14 **ok** vs F3
> (inside_mismatch ≤ 321 à zoom jusqu'à 1e652). Les ratios 1.92 (glitch_test_2) /
> 1.24 (dragon) ci-dessous étaient **spécifiques au Xeon 4 cœurs** ; sur l'i7 8c/16t
> ils WIN aussi (quick tier : geomean **0.247**, 10/10 wins). Aucun déficit vitesse
> mesurable hors la classe ultra-deep-intérieur bloquée (période-aware, cf. plus bas).
>
> **Résidu vitesse mesuré (2026-07-11, harness quick vs F3 sur Xeon 4c)** — après
> le fix `bla_exp` (cf. Shipped), les 2 seuls cas où fractall reste plus lent que
> F3 sont **orbite-bound sur référence INTÉRIEURE pleine longueur** :
> - `glitch_test_2` (zoom 1.53e115, réf 250 001 iters, prec 414 b) : ratio **1.92** ;
>   orbite GMP **0.163 s** > rendu TOTAL F3 (0.168 s).
> - `dragon` (zoom 9.47e193, réf 5 M iters, prec 676 b) : ratio **1.24** ;
>   orbite GMP **3.196 s** = 70 % du rendu (BLA 0.13 s, pixels 1.27 s).
>
> Les deux références n'ÉVADENT PAS (centre intérieur → boucle jusqu'à
> `iteration_max`) et fractall calcule l'orbite GMP pleine longueur, alors que
> **F3 tronque via son atom-domain period detection** (réf = 1 période ≪ 250 k /
> 5 M). Précision IDENTIQUE à F3 (formule `24+⌊log2(zoom·h)⌋` portée), donc pas
> d'over-provisioning ; l'arithmétique GMP est déjà optimale (`mpc_sqr`). **Le
> seul levier = tronquer la réf via atom-domain sur le path f64** — c'est
> exactement le blocage `atom_period_enabled` gaté `< 1e-280` (path exp only,
> cf. §G3 « period-aware reference », **4 sessions brûlées**) : les grazes de la
> réf tronquée ~1e-56…1e-8000 tuent la reconstruction d'évasion en f64. Candidat
> mémoire adjacent : `z_ref`/`z_ref_dd` (ComplexExp, 32 o/iter → dragon **160 Mo**)
> sont MORTS sur le path f64 (lus seulement par exp/dd/legacy/hybrid) mais leur
> skip demande un gate 4-conditions sur le hot-loop orbite → différé (risque >
> gain, n'aide pas les OOM qui sont exp-path). **Non re-litigé ici** (cas tranché).

> **🏆 RÉSOLU (2026-05-21)** — fix de **~20 lignes** : rebase à `m=0` au bout de
> la référence pour les orbites escape-time (F3 `hybrid.cc:301` : `m+1==size` ⇒
> `z=Z+δ, m=0`), au lieu d'abandonner (`ref_exhausted` → fallback GMP). La
> perturbation devient utilisable sur les centres escape-time → **plus de
> fallback GMP**. Résultats 256² (étaient en GMP) :
> | cas | avant (GMP) | après (perturbation) | correction |
> |-----|-------------|----------------------|------------|
> | **e50** | 544 s | **1.57 s** (~346×) | == GMP (mean 0.074) |
> | **e1000** | 742 s | **0.53 s** (~1400×) | **pixel-identique GMP** |
> | **dragon** | ~6 h | **6.46 s** (~3350×) | OK |
> | e113 | golden | 0.52 s | == GMP (mean 0.089), golden régénéré + revu |
>
> **dragon n'était PAS physiquement impossible** : le BLA skippe l'immense
> majorité de ses 5M itérations une fois la perturbation active. **178 unit +
> golden verts** (seul e113 régénéré, validé). `pixel_loop_exp.rs` `_mandelbrot`
> + `_generic`. ⇒ **Les 36 cas perf-bound de G1 ne le sont plus.**

> **🎯 BREAKTHROUGH (2026-05-21)** — Les cas perf-bound (e50/e1000/dragon/…)
> **NE RENDAIENT PAS via perturbation** : leur orbite référence s'évade
> (`ref_truncated`) → le gate routait TOUS les pixels vers **GMP par-pixel**
> (~1 µs/iter). **Tout mon profilage antérieur (« memory-bound », « ~32 ns/iter »,
> « BLA inefficace », SIMD, float128) était FAUX** : il supposait que ces cas
> utilisaient le path perturbation. Ils ne l'utilisaient pas.

**Preuves (e50 @48×48) :**
- baseline = 20.3 s pour 2304 px = **8.7 ms/px ≈ 1 µs/iter** → signature **GMP
  par-pixel** (pas perturbation).
- `--precision-bits 1024` = 56.5 s, **image identique** (0 diff) → l'évasion de
  la référence n'est PAS un problème de précision (genuine escape).
- `--find-nucleus` (référence non-évadante) = **3.07 s** (dont 2.6 s Newton fixe
  → rendu ~0.4 s = **~50× plus rapide**) → avec une référence valide, la
  perturbation + BLA **skippe** les itérations → rapide. Mais le nucleus
  **change la vue** (re-centre + rotation 16.8°, 1351/2304 px diff) ⇒ pas valide
  pour la parité F3 telle quelle.

> Les figures « 256² : e50 544 s / e1000 742 s / dragon 6 h » sont RÉELLES mais
> reflètent le **fallback GMP**, pas la perturbation. Le compteur d'itérations
> retourné par le path tronqué est 0 → les « ns/iter » dérivés étaient du bruit.

**VRAIE CAUSE** : `compute_reference_orbit` tronque (ou le gate `ref_truncated`
de `pixel_loop_exp` traite) la référence qui s'évade → fallback GMP. F3, lui,
**continue la référence au-delà de l'évasion** (trajectoire pleine longueur) et
laisse la perturbation **rebaser** les pixels qui s'évadent → reste sur le path
rapide. **Le fix = rendre la perturbation utilisable avec une référence évadante
À LA MÊME VUE** (pas le nucleus qui change la vue), en évitant le bug d'image
uniforme qui a motivé le gate `ref_truncated` (cf. e113).

**Done when** :
- [x] **🟢 Fallback GMP éliminé** (2026-05-21) : rebase-at-end F3 pour orbites
  escape-time dans `pixel_loop_exp.rs` (`_mandelbrot` + `_generic`) ; gate
  `ref_truncated` retiré. e50/e1000/dragon en **<7 s** à 256² (étaient 544 s /
  742 s / ~6 h). e1000 pixel-identique GMP ; e50/e113 == GMP (mean <0.1).
- [x] **F3 confirmé comme référence** : `hybrid.cc:296-307` rebase à `m=0` sur
  `|Z+δ|²<|δ|²` OU `m+1==size` (bout de réf). On a porté la 2e branche.
- [x] **dragon réévalué** : **6.46 s** à 256² (BLA skippe l'essentiel des 5M
  iter). N'était pas physiquement impossible — il rendait juste en GMP.
- [x] **BLA lookup aligned-start** + **libération niveaux BLA inutilisés**
  (`bla_dual.rs`, committé `fd9ce4a`). Aide TOUS les cas perturbation — qui
  incluent désormais les 36 ex-perf-bound. + gros gain mémoire (table ~8×).
- [x] r2_fexp retiré ; boucle f64-scaled retirée (0 % — sur le mauvais path).
- [x] **Clone d'orbite hybride supprimé sans cycle** (2026-07-03, `eb7287e`) :
  `build_hybrid_bla_references` renvoyait `single(primary.clone())` dès
  qu'aucun cycle n'était détecté (cas escape-time = majorité des deep zooms),
  clonant z_ref+z_ref_f64+**z_ref_gmp** (~170 o/iter → 850 Mo à 5 M iters/dragon)
  alors que le cache stocke déjà l'orbite. Renvoyer `None` (dispatch route déjà
  vers `cache.orbit` via `iterate_pixel`, bit-identique). **dragon 3.60→3.37×,
  glitch_test_2 4.37→3.75× (pire ratio), ~850 Mo libérés**. Parité/quality/
  goldens inchangés.
- [x] **Série non construite si path bytecode l'ignore** (2026-07-03, `d385674`) :
  gate `series_will_be_used = auto_adjust_enabled || !uses_bytecode_path`. En
  parité-F3 (`NO_AUTO_ADJUST`) le build série était mort (bytecode a sa propre
  BLA). **dragon série 0.77→0 s, total 11.57→10.61 s, ratio 3.33→3.12**. Auto-
  adjust (défaut) inchangé. Bit-identique (série jamais lue dans ce cas).
- [x] **Orbite `z_ref` ComplexExp dérivée du f64** (2026-07-04, `9e2aeb6`) :
  `from_gmp` clonait 2 Floats GMP/itér. → fast-path `from_complex64` du f64 déjà
  calculé quand z est dans le range f64 normal (bit-identique ; fallback from_gmp
  si underflow near-0). dragon orbite ~5.06→4.74 s, ratio 2.97→2.94, geomean
  0.759. Verrou goldens deep-zoom + bit-identité dragon.
- [x] **Pré-check f64 du bailout orbite référence** (2026-07-04, `3f59745`) :
  la norme GMP du test de bailout (2 carrés 676 b/itér.) tournait à chaque tour
  même sur les orbites bornées qui ne s'évadent jamais (dragon 5 M iters). Pré-
  check f64 sur `z_ref_f64.last()` (bit-identique, marge 1 % ≫ l'erreur ~4e-6) →
  norme GMP calculée seulement près de la frontière. **dragon orbite 4.4→3.26 s
  (−26 %), ratio 2.94→2.50 ; glitch_test_2 3.76→3.15**. Diagnostic préalable :
  interpréteur GMP + copie `z.assign` = 0 gain (A/B) — c'était la norme.
- [x] **Wisdom mid-tier : orbite référence double-double** (2026-07-04,
  `51f7950` Phase 1a + `b757e65` Phase 1b) : le module `dd.rs` (double-double
  ~106 b, dormant) est câblé comme moteur d'orbite référence. Quand Mandelbrot
  standard ∧ path bytecode ∧ requirement F3 non clampé ≤ 96 b (~1e13–1e19),
  l'orbite est itérée en dd au lieu de GMP (`dd_reference_orbit_mandelbrot`,
  gate sur les bits formule bruts car `compute_perturbation_precision_bits`
  clampe à ≥128). **Orbite 14× plus rapide** (2.6→0.2 ms), **e17 render 66→30 ms,
  e13 86→55 ms**. Insight : l'orbite est ITÉRÉE haute précision mais STOCKÉE en
  53 b → dd (106 b) est un drop-in de l'itération GMP pour cette tranche. Pas de
  period-detection (rebase-at-end F3). Goldens 🟢 (orbites courtes → dd et GMP
  arrondissent au même 53 b), quality e13/e17 PASS (= GMP), parité inchangée.
  Verrou `dd_orbit_matches_gmp_midrange` + presets e13/e17/e18. **Réalise le
  goal G2 « aucun path ne force GMP là où doubledouble suffit » pour la
  tranche moyenne.**
  - [x] **Phase 2 — Julia dd** (2026-07-04, `bf88e4d`) : `dd_reference_orbit`
    généralisé `(z0, c)` — Mandelbrot `z0=0,c=cref` / Julia `z0=cref,c=seed`.
    julia-siegel-disk orbite GMP→dd, PASS max_diff=0. Verrou
    `dd_orbit_matches_gmp_julia`. Tranche dd couvre Mandelbrot ET Julia.
  - [ ] Reste Phase 2+ : quad-double pour 1e28–1e60 ; câbler le pixel loop dd
    (P1.6.e original : delta quasi-périodique, glitch_test_1/5).
- [x] **Orbite : `square_mut` (mpc_sqr) au lieu de `z*=z` (mpc_mul)** (2026-07-04) :
  un sweep du corpus COMPLET (cf. [[harness-sweep-full-corpus]]) a révélé les cas
  ultra-profonds orbite-bound (wfs*, olbaid5) plus lents que F3. Le pas Mandelbrot
  `Z²+c` faisait `z *= z` (mpc_mul, ~3 mults réelles) alors que z² se calcule en
  **~2 mults** (mpc_sqr via `Complex::square_mut`) — comme F3 (`sqr`). Appliqué à
  `interp_gmp.rs::Op::Sqr` (path bytecode, défaut) + `orbit.rs` (Mandelbrot/Julia/
  BS/Tricorn direct). Correctement arrondi = même valeur que z·z → **bit-identique**
  (goldens 🟢 inchangés). **Orbite ~2.1× plus rapide** à haute précision (wfs4 6338 b :
  30.6→14.6 s). Résultats 256² : wfs4 2.56→**1.30**, wfs2 1.51→**0.84**, wfs_extended
  1.89→**1.05**, wfs→0.50, olbaid5 1.41→**0.88**, **dragon 1.9→0.79**, e227→0.53.
  Aide TOUT orbite deep Mandelbrot/Julia/BS/Tricorn. 187 unit + quality 11 PASS +
  parité inchangés.
  - [ ] **Gap CARACTÉRISÉ (2026-07-04) : classe « ultra-deep INTÉRIEUR »**
    (wfs_mb 6.8×, e8000, e22522, e52465 — orbites 10-50 M iters, 6-50 kbits) :
    **la vue entière est INTÉRIEURE** (wfs_mb 64² : avg_iter/px = max = 10 M = tous
    les pixels atteignent le cap sans s'évader → orbite référence PÉRIODIQUE, cycle
    attracteur). fractall force les 10 M itérations GMP (77 s @ 6750 b) pour
    conclure « intérieur ». F3 fait wfs_mb en **9.33 s @ ~107 % CPU** (orbite-bound
    single-thread) → il calcule ~1.3 M iters, PAS 10 M : **F3 détecte la période via
    atom-domain** (`hybrid_period`, `|z|² < s²|dz|²`) et confirme l'intérieur tôt.
    Diagnostic vérifié : la period-detection Brent de fractall (a) est désactivée par
    `NO_PERIOD=1` (garde anti-faux-positifs, cf. glitch_test_5/floral_fantasy grazes),
    (b) coûte 2×/iter quand activée (check GMP sub+norm par tour), (c) NE détecte PAS
    wfs_mb en 150 s → **net perdant**. Le squaring est déjà optimal (mpc_sqr).
    **Fix = period-aware reference via atom-domain** (porter `hybrid_period` de
    nucleus.rs DANS `compute_reference_orbit` : trouver P, calculer 1 période, cycler).
    GROS + DÉLICAT (faux positifs = intérieur erroné/image uniforme = la raison de
    NO_PERIOD) → verrous stricts requis (goldens interior + glitch_test_5/1). Reporté :
    cas extrême (1e2020+), ROI faible vs risque correction. Le reste du corpus est
    déjà ≤ F3 (geomean 0.29). Cf. CLAUDE.md « nucleus phase-aware » (TODO G2/G4).
  - [ ] **TENTATIVE 2026-07-05 (REVERTÉE) — atom-domain + cyclage MODULO** : porté
    dans `compute_reference_orbit` (dz suivi en ComplexExp ~35 ns/iter, critère
    `z_f64==0 ∧ |z|²<s²|dz|²`, troncation → `cycle_period` → `wrap_periodic` modulo).
    Résultats mesurés (64², diff vs orbite pleine) :
    - **wfs_mb : truncation à ~271 k iters, 86 s→2.5 s (35×), PIXEL-IDENTIQUE** avec
      `z_f64==0` SEUL. e8000 : 6→2.6 s, pixel-identique. **Marche pour les vrais
      nucleus super-attracteurs** (z_P underflow f64 = 0).
    - **MAIS e22522 CASSÉ** (escape-time à 433 k iters) : `z_f64==0` fire à tort car
      à prec ≫ 1023 l'orbite FRÔLE 0 (z~1e-500, underflow f64) sans être périodique.
      Diagnostic [ATOMDIAG] : e22522 z_f64==0 à 1668, 3336, 5005… (espacé régulier,
      « near-périodique ») MAIS escape à 433 k = 260 périodes plus loin.
    - **LIMITE FONDAMENTALE du cyclage MODULO** : au point de troncation, un vrai
      intérieur (wfs_mb) et un near-périodique-qui-escape (e22522) sont
      INDISTINGUABLES sans calculer ~toute l'orbite. Aucun seuil sur `|z|`/`|dz|`/prec
      ne les sépare (critère atom `|z|<s|dz|` rejette LES DEUX ; `z_f64==0` accepte
      les deux). La vérif « z revient au même point à 2P » est bernée (e22522 semble
      périodique 260× avant d'escape).
    - **LE VRAI FIX ENVISAGÉ = cyclage REBASE-AT-END (F3 `hybrid.cc:301`)** : le
      rebase absorbe Z dans δ → cens é tolérer une troncation fausse (le delta compense).
  - [ ] **TENTATIVE 2 (2026-07-05, REVERTÉE) — rebase-at-end + dz-stabilité** :
    - **rebase-at-end MARCHE pour wfs_mb** (garder z≈0 en dernier élément, cycle_
      period=0 → l'exp loop lit z_ref[last]≈0 ⇒ delta petit) : 86→2.4 s PIXEL-IDENTIQUE.
      (Le bug « delta ~O(1) » de la tentative 1 venait de tronquer à z~O(1), pas z≈0.)
    - **MAIS rebase-at-end NE tolère PAS e22522** (tronqué à 1668 → faussement
      intérieur, l'orbite pleine escape à 433 k). La « tolérance F3 » ne vaut que pour
      une SUR-troncation LÉGÈRE près de la vraie période, pas 1/260ᵉ. Et e8000 (vrai
      intérieur) CASSE aussi en rebase-at-end alors qu'il marchait en MODULO → rebase
      vs modulo ont des domaines de validité DIFFÉRENTS par cas, pas de gagnant clair.
    - **dz-stabilité RÉFUTÉE** : [ATOMDIAG] montre que LES 4 cas (wfs_mb, e8000,
      e22522, e52465) ont `|dz|²` STABLE (Δlog2≈0) entre passages z≈0 consécutifs
      (tous « paraissent » périodiques). Le multiplieur dérivée ne distingue PAS
      intérieur/escape. z≈0 à multiples réguliers pour les 4.
    - **Critère F3 exact mal reproduit** : `|Z_i| < |radius·dZdC|` avec radius=4/zoom
      (analysis) → mon calcul REJETTE wfs_mb (|Z_P|≈1e-342 > radius·|dZdC|≈1e-1343).
      Donc je ne comprends pas encore la sémantique exacte de `radius`/`dZdC` de F3
      (échelle ? dZdC = matrice 2×2, pas scalaire ? phase ?). **BLOQUANT** : sans le
      critère F3 exact, aucune heuristique testée (z_f64==0, atom-scalaire, dz-stab)
      ne sépare {wfs_mb,e8000} de {e22522,e52465}.
    - **PROCHAINE FOIS** : relire `hybrid.cc:79-97` LIGNE À LIGNE (surtout la
      construction de `radius`=`K` dans engine.cc:235 et le type `mat2` de dZdC) et
      reproduire le calcul matriciel EXACT avant de re-tenter. Ne PAS improviser un
      critère scalaire. 2 sessions déjà brûlées ici ; c'est le point dur. Revert
      propre à chaque fois, 0 régression. Le reste du moteur reste ≤ F3 (geomean 0.29).
  - [x] **TENTATIVE 3 (2026-07-05) — CRITÈRE F3 EXACT IMPLÉMENTÉ → HYPOTHÈSE RÉFUTÉE** :
    extrait le calcul EXACT de F3 (`hybrid.cc:92` `abs(inverse(radius·dZdC)·Zp[i])<1`,
    `matrix.h`, `engine.cc:235`). Cas conforme Mandelbrot = **`|Zp[i]| < (4/zoom)·|dZ/dC|`**
    avec `|radius|=4/zoom` EXACT (pas 2/zoom ni 4/zoom/height). Implémenté ce critère
    (dZdC = dérivée ComplexExp, testé CHAQUE itér.). Résultat DÉCISIF [ATOMDIAG-F3] :
    - **wfs_mb : le critère F3 NE FIRE JAMAIS** (min ratio 2^+2150, jamais proche).
      **F3 NE TRONQUE PAS wfs_mb par l'atom-domain de référence.** Son centre est
      DANS le bulbe période-P mais LOIN du nucleus (|Z_P|~2^-1135 ⇒ offset c-space
      ~2^-3386 ≫ vue 2^-6710). L'atome ne couvre pas la vue.
    - e8000 : fire à 44899 (vrai nucleus dans la vue). e22522 : ne fire jamais (correct).
      Donc le critère F3 est bien reproduit (fire e8000, pas e22522) — MAIS il ne
      cible PAS wfs_mb.
    - **CONCLUSION : la « period-aware reference via atom-domain » est un CUL-DE-SAC
      pour wfs_mb.** F3 fait wfs_mb en **4.84 s** (orbite ~single-thread) vs fractall
      75 s → **~15×/iter à 6750 b**. 0.48 µs/iter est PHYSIQUEMENT impossible en
      full-precision 6750 b (une mul 105-limb ~1-2 µs) ⇒ **F3 calcule BEAUCOUP moins
      d'itérations** — probablement via le **NEWTON `hybrid_period`** (hybrid.cc:417,
      mécanisme SÉPARÉ de l'atom-domain de référence) + **reference period-lock**
      (engine.cc:225 `lock_maximum_reference_iterations_to_period`). Le `--find-nucleus`
      de fractall (finder apparenté) timeout > 150 s → pas un quick win non plus.
    - **ACQUIS solides** : (a) le MÉCANISME troncation+cyclage MARCHE (wfs_mb tronqué
      = pixel-identique + 35× quand la détection fire) ; (b) le critère atom-domain de
      référence n'est PAS le bon outil pour wfs_mb.
  - [x] **TENTATIVE 4 (2026-07-05) — period-lock aussi ÉCARTÉ ; gap = ORBITE GMP** :
    vérifié `lock_maximum_reference_iterations_to_period` = **`false` PAR DÉFAUT**
    (param.cc:208, param.h:47) ET exige `reference.period > 0` connu. Le wrapper batch
    F3 (compare_f3.py) ne l'active PAS et ne fournit pas de période. **Donc F3 N'A
    AUCUNE troncation active pour wfs_mb** (period-lock off, atom-domain ne fire pas,
    pas d'escape — centre intérieur) et calcule quand même la référence PLEINE (10 M)
    en **4.84 s** (32² orbite-dominé) vs fractall **75 s** ⇒ **~15×/iter à 6737 b**.
    - **Les DEUX mécanismes période sont donc écartés** (atom-domain + period-lock).
      Le gap wfs_mb = **vitesse orbite GMP par-itération** (rug/MPC vs MPFR raw F3).
    - ⚠️ **MAIS 0.48 µs/iter pour un carré complexe 6737 b est SUSPECT** (théorique
      ~1.5 µs pour une seule mul 105-limbes ; carré complexe = 2-3 muls). Soit F3 a un
      bignum ~10× plus rapide (improbable, MPFR dessous des deux), soit un DÉTAIL F3
      non identifié (précision réduite ? représentation ? une troncation que je n'ai
      pas trouvée ?). **Non élucidé.**
    - **CONCLUSION FERME (4 sessions)** : wfs_mb NE se résout PAS par la détection de
      période (les 2 variantes F3 réfutées expérimentalement). Le levier restant =
      orbite GMP raw-MPFR (gros, incertain, toucherait TOUS les deep zooms) OU
      élucider le 15×/iter F3 (mesure/profil dédié). **Recommandation : clore** — cas
      extrême (1e2020), moteur ≤ F3 partout ailleurs (geomean 0.29), 4 sessions déjà
      investies. Ne rouvrir que sur profil précis du 15×/iter, PAS sur la période.
  - [x] **PERCÉE (2026-07-05, tentative 5, env-gated) — détection RÉSOLUE, cf. verdict ci-dessous** : le « 15×/iter » de la
    session 4 était FAUX. Mesure fine (F3 orbite vs cap d'itérations) : **F3 a le MÊME
    per-iter que fractall (~8 µs)** ; le 15× vient de la **TRONCATION** (F3 tronque
    wfs_mb à ~540 k = 2 périodes). Instrumenté F3 lui-même (print `hybrid.cc:92` +
    rebuild `make SYSTEM=linux-batch`) : **F3 fire à i=542080**. Reverse-eng : F3
    `|dZdC|`=2^3359 vs mon 2^2251 → **BUG TROUVÉ** : je calculais dZdC avec `z_curr_f64`
    (f64) qui UNDERFLOW pour les Z≈0 near-période → dZdC trop petit de ~2^1108. **FIX**:
    dZdC en ComplexExp (`dZdC=2·Z·dZdC+1`, Z ComplexExp).
    - **Critère F3 EXACTEMENT reproduit** — points de fire IDENTIQUES à F3 vérifiés :
      wfs_mb 542080, e50 86614, e113 11380, dragon 2046924, floral 1704, e8000 44899.
      (C'ÉTAIT LE BLOCAGE des 4 sessions — détection enfin RÉSOLUE.)
    - **wfs_mb : truncation + rebase-at-end = PIXEL-IDENTIQUE full-ref, 86→4.9 s (17.6×).**
    - **RESTE : le CYCLAGE**. rebase-at-end marche pour wfs_mb/e52465/floral mais diffère
      pour e8000/e1000/e401/dragon (comparés à FULL-REF — mais si F3 tronque aussi,
      full-ref est le MAUVAIS baseline → comparer à F3 EXR). `pixel_loop.rs` (f64) bail
      en GMP à end_of_ref (rebase-at-end pas porté) → mid-zoom non géré, gaté au path exp.
    - **Env-gated `FRACTALL_ATOM_PERIOD=1`, OFF par défaut** (goldens + 187 unit verts,
      0 régression).
  - [~] **VERDICT tentative 6 (2026-07-05) — RÉVISÉ tentative 7, voir ci-dessous.**
    Mesure ATOM on/off vs **F3 EXR direct** (`compare_f3.py`, pas le full-ref) :
    e1000/e8000/e401 atom-ON flip TOUT intérieur (inside_mm=25600) ; wfs_mb identique.
    J'avais conclu « F3 NE tronque pas e8000, full-ref = bon baseline » — **CONCLUSION
    FAUSSE**, réfutée par instrumentation directe de F3 (tentative 7). Gardée ici pour
    trace ; ne pas s'y fier.
  - [~] **TENTATIVE 7 (2026-07-05) — GROUND TRUTH F3 : F3 TRONQUE TOUT ; le bug est le
    CYCLAGE de fractall, pas le critère.** Instrumenté `hybrid_reference` (F3
    `hybrid.cc:82-97`, print gaté `F3_REFDIAG`, rebuild `make SYSTEM=linux-batch`) :
    **la boucle de RÉFÉRENCE elle-même tronque** via le check périodicité
    `abs(inverse(radius·dZdC)·Zp[i]) < 1` → `M=i+1; break` — INCONDITIONNEL, PAS gaté
    par newton/period-lock. Longueurs de référence F3 mesurées :
    | cas | F3 tronque à | cap toml | réf full-escape (atom-off fractall) |
    |-----|-------------:|---------:|--------------------------:|
    | e1000 | **4499** | 32000 | — |
    | e401  | **13729** | 212138 | — |
    | e8000 | **44900** | 807345 | escape à 105606 |
    | wfs_mb | **542081** | 10000000 | jamais (vrai intérieur) |
    - **Le critère atom de fractall est EXACT** (fire aux mêmes i que F3 : e8000 44900).
      **F3 tronque e8000 à 44900 ET rend correctement** (inside_mm=0 vs full-ref). Donc
      la troncature N'EST PAS le problème — **le CYCLAGE de la réf tronquée l'est.**
    - **Root cause du flip-intérieur (instrumenté `FRACTALL_PIXDIAG`, e8000 atom-on)** :
      tous les pixels IDENTIQUES, `n=807345=imax` (→ marqués intérieur), rebase=17
      (~18 périodes de 44900), **`|δ|` COINCÉ à exp 0 (O(1)) sur toutes les périodes —
      δ ne CROÎT jamais** vers l'évasion. La contribution pixel (dc) est noyée →
      collapse uniforme. `Z[end]=Z[44899]` lu **= 0.0 exact en f64** (underflow : la vraie
      valeur ~1e-8000 au graze est zéroée).
    - **DIAGNOSTIC** : à e8000 la réf tronquée est un GRAZE périodique (pas un vrai
      cycle : l'orbite pleine s'évade à 105606 = 2,3 périodes). L'évasion doit être
      reconstruite par la CROISSANCE de δ sur ~2,3 périodes rebasées. F3 y arrive
      (réf + BLA en **floatexp**, exposant étendu → grazes ~1e-8000 préservés). Fractall
      **stocke la réf en `Complex64` (f64) ET la BLA en `mat2<f64>`** → les valeurs de
      graze ~1e-8000 underflow à 0, les coeffs BLA `A=∏2Z` (~1e444/période) overflow →
      la croissance de δ est perdue → tout intérieur. (Le multiplieur orbital λ=∏2Z NE
      sépare PAS intérieur/graze : Z[0]=0 ⇒ λ linéaire par période = 0 pour LES DEUX ;
      l'évasion est NON-linéaire. Pas de gate interior cheap : e22522 graze 260 périodes
      avant d'évader.)
    - **CE QU'IL FAUT (= vrai fix, TODO G2/wisdom)** : porter la réf **ET** la BLA du
      path atom-tronqué en précision étendue (floatexp/ComplexExp) comme F3, pour que le
      cyclage reconstruise l'évasion. `ReferenceOrbit.z_ref` (ComplexExp) existe déjà ;
      la BLA `mat2<f64>` est le second morceau (overflow à 1e444). NON trivial —
      ~réécriture de `pixel_loop_exp.rs` + `bla_dual` en exposant étendu. Gain visé :
      wfs_mb 89→~5 s (16×) + tous les deep-interior, ET correction des grazes.
    - **DÉCISION** : `FRACTALL_ATOM_PERIOD` reste gaté OFF (le cyclage tronqué est FAUX
      tant que réf+BLA sont f64). Le critère de troncature (F3-exact) est BON à garder.
      Rouvrir = chantier précision-étendue réf+BLA, pas un gate interior. Moteur ≤ F3
      partout ailleurs (geomean 0.29).
  - [x] **TENTATIVE 8 (2026-07-05) — RÉFÉRENCE COMPLEXEXP = CORRECTNESS RÉSOLUE (vérifié
    F3 EXR).** Ajouté `iterate_pixel_unified_exp_mandelbrot_hp` (pixel_loop_exp.rs, gaté
    `FRACTALL_ATOM_PERIOD`) : **réf lue en `ComplexExp` (`ref_orbit.z_ref`) + pas directs
    ComplexExp, PAS de BLA.** Résultat e8000 atom-ON vs **F3 EXR** : **Δmean=0.2033,
    inside_mm=0 — BIT-IDENTIQUE à atom-off/F3** (avant : flip total intérieur). Confirme
    que **la précision de la réf ÉTAIT la cause** (les grazes ~1e-8000 zéroés en f64
    tuaient la reconstruction). avg_iter/px=105243 (= full-ref). Goldens 🟢 (path gaté).
    - **RESTE = perf.** HP sans BLA = 24 s (vs atom-off 6 s) sur e8000 ; INUTILISABLE
      sur wfs_mb (intérieur → iteration_max=10M pas/px sans skip). **Le gain 16× exige
      une BLA en exposant étendu** : la BLA f64 (`bla_dual.rs` `mat2<f64>`) est bâtie
      depuis `z_ref_f64` (grazes zéroés) ET ses coeffs `A=∏2Z` (skip traversant un graze
      ~1e-8000·… ou ~1e444/période) underflow/overflow f64. Chantier suivant :
      `bla_dual_exp.rs` (miroir FloatExp de Mat2/DualComplex2/single-step/merge/table),
      bâtie depuis `z_ref` (ComplexExp), câblée+cachée dans le path HP. Oracle de
      correction : `compare_f3.py --only e8000` doit rester Δmean≈0.20 inside_mm=0, ET
      passer sous ~6 s ; puis wfs_mb doit rendre correct + ~5 s (16×).
  - [x] **TENTATIVE 9 (2026-07-05) — BLA FloatExp IMPLÉMENTÉE + VÉRIFIÉE. Port complet.**
    Nouveau `bla_dual_exp.rs` (563 l : `Mat2Exp`/`DualComplex2Exp`/`BlaSingleStepExp`/
    `BlaMultiStepExp::merge`/`BlaTableUnifiedExp`, miroir FloatExp fidèle de `bla_dual.rs`
    — f64 INTOUCHÉ). `FloatExp::{div,sqrt,min,max}` ajoutés (types.rs). BLA exp bâtie 1×/réf
    depuis `z_ref` (ComplexExp), cachée dans `BlaUnifiedCacheEntry` (delta.rs), câblée au
    path HP UNIQUEMENT quand `FRACTALL_ATOM_PERIOD=1 && Mandelbrot`. Boucle HP réordonnée
    en rebase-avant-lookup (F3). **Vérifié vs F3 EXR (128²) :**
    | cas | atom-ON + BLA exp | note |
    |-----|-------------------|------|
    | e8000 | Δmean=0.25, **inside_mm=0**, Fr **2.75 s** | = F3, **9× vs no-BLA (24 s), < F3 (4.8 s)** |
    | e1000 | Δmean=0.54, inside_mm=0, 0.12 s | = F3 |
    | e401  | Δmean=10.8, inside_mm=0, 0.14 s | ≈ atom-off (e401 jamais parfait même off) |
    - **Régression = ZÉRO** : 196 unit (+9) 🟢, goldens 🟢, `bla_dual.rs` f64 intouché,
      path défaut (atom-off) inchangé (BLA exp bâtie/utilisée seulement si atom on).
    - **wfs_mb (1e2020) = ÉCART PRÉ-EXISTANT, PAS une régression** : atom-OFF (path défaut,
      hors atom) diverge DÉJÀ vs F3 (Δmean=0 sur pixels d'accord MAIS **inside_mm=16370/16384**
      — désaccord massif intérieur/extérieur au cap 10M). atom-ON fait MIEUX sur la frontière
      (inside_mm=**6**) mais Δmean=4605 (~0,08 % relatif sur 5,7 M iters) → limite de précision
      à 6749 b / cyclage ~10 périodes. **wfs_mb est un bug SÉPARÉ du path deep par défaut
      (classe « ultra-deep intérieur »), à traiter à part — pas bloquant pour ce port.**
    - **BILAN** : la troncature atom est désormais CORRECTE (réf+BLA exposant-étendu) pour
      les cas où le path deep défaut l'est (e8000/e1000/e401). Gain perf modeste sur ces cas
      (troncature réf : e8000 105606→44900, e1000 32000→4499). Le gros gain visé (wfs_mb 16×)
      reste bloqué par l'écart pré-existant wfs_mb, PAS par le cyclage (enfin correct).
    - **DÉCISION** : hook reste gaté OFF (validation corpus complet + fix wfs_mb requis avant
      défaut), mais il est maintenant FONCTIONNEL & sans régression. Rien commité.
  - [~] **VALIDATION CORPUS ATOM (2026-07-11) — snapshot post-TENTATIVE 9 + réfutation
    micro-opt.** Gaps vitesse actuels (3 runs médiane, corpus complet propre, quarantaine
    vide) : **e8000 1.336** (fr 6.58 s / F3 4.92 s, **98 % orbite** : 807 k iters × 26 616 b),
    **glitch_test_2 1.302** (0.22/0.17), **wfs4 1.175** (15.5/13.2). Tous Mandelbrot,
    orbite-bound. Le levier est la **troncature atom** (TENTATIVE 9) : e8000 atom-ON =
    3.03 s **< F3**, **PIXEL-IDENTIQUE** atom-off (revérifié).
    - **Snapshot atom-ON vs atom-OFF, 17 cas ultra-deep (>1e280), 256²** — étape « validation
      corpus complet » demandée par T9 : **7 sûrs** (identiques/bruit bord : e1000, e8000,
      e401, e634, e890, e1016 [10 px], e1298 [1.7 %]) — tous **escape-time** ; **10 divergents**
      (**intérieur/near-interior** : wfs2, wfs4, wfs_extended, olbaid5, triangle = 100 % ;
      wfs_mb, e1121, e1200 ≈ 100 % ; olbaid1 21 % ; e1086 14 %). ⇒ atom **PAS enable-able tel
      quel**. Prochaine étape G2 : adjuger CHAQUE divergent vs **F3 EXR** (`compare_f3.py`) —
      certains (wfs_mb) atom-ON est PLUS correct (T9 inside_mm 16370→6), d'autres atom-ON casse.
      Hypothèse : atom sûr ⇔ réf **escape-time** ; buggé ⇔ réf **bornée/intérieure** (cyclage).
    - **RÉFUTÉ — l'overhead orbite n'est PAS la copie `z.assign(&state.z)`** : bypasser
      l'interpréteur bytecode pour l'orbite Mandelbrot (hand-path `square_mut`+add, bit-identique,
      goldens 🟢) = **0 gain mesuré** (e8000 orbite 6.02→6.02 s ; glitch_test_2 0.128 s inchangé).
      Confirme le verdict T7 (« per-iter ≈ F3 ~8 µs, rug≈MPFR ») : le gap est la **longueur de
      réf** (troncature), pas le coût par itération. Reverté. ⇒ le SEUL levier vitesse deep-zoom
      reste **le cyclage atom correct sur les cas intérieurs** (chantier G2, non trivial).
  - [x] **⭐ ATOM-PERIOD ON PAR DÉFAUT (2026-07-11) — CORRECTION MAJEURE deep-interior +
    vitesse. La conclusion « atom-ON casse » ci-dessus était FAUSSE** (comparait atom-ON à
    atom-OFF en supposant atom-OFF correct). **Adjugé vs F3 EXR** (`compare_f3.py`, 17 cas
    ultra-deep) : c'est **atom-OFF (le défaut !) qui DIVERGE de F3**, atom-ON qui matche.
    | cas | atom-OFF rel Δ% vs F3 | atom-ON rel Δ% | temps off→on |
    |-----|------:|------:|-----|
    | wfs_mb | inside_mm **65479/65536** (tout NOIR) | inside_mm 12 | 79→**5 s** (15.8×) |
    | wfs4 | 1.296 | **0.003** | 15.2→**4.3 s** |
    | e1121 | 0.283 | **0.0002** | — |
    | wfs2 | 0.400 | **0.002** | 10.4→**4.0 s** |
    | wfs_extended | 0.376 | **0.002** | — |
    | e1200 | 0.044 | **0.0003** | — |
    | olbaid5 | 0.022 | **0.0002** | — |
    | triangle | 0.034 | **0.0006** | 28.8→**16 s** |
    | e8000/e1000/e401/e634/e890/e1086/olbaid1 | (déjà bas) | **= atom-OFF** | e8000 6.6→3.0 |
    - **atom-ON n'est JAMAIS pire** (égal sur les escape-time déjà corrects, 50–1400× meilleur
      sur l'intérieur) ET 2–16× plus rapide. Visuel wfs_mb : atom-OFF = **écran noir**, atom-ON =
      structure F3 (diff quasi-blanc). ⇒ le path deep-interior par défaut produisait des images
      FAUSSES depuis toujours (masqué par le gate parité laxiste — inside_mm=0 sur les cas où les
      masques concordent mais smooth-iter faux).
    - **Décision** : `atom_hp_enabled()` (flag canonique `delta.rs`, partagé par les 3 portes
      orbit/delta/pixel_loop_exp) **default ON** ; `FRACTALL_ATOM_PERIOD=0` force OFF (debug).
      N'active QUE Mandelbrot + path exp >1e280 (mid-range/shallow/goldens INTOUCHÉS). Ferme le
      chantier T9 (« validation corpus + fix wfs_mb requis » → fait : wfs_mb 15.8× + correct).
    - **Verrous** : golden `mandelbrot_e1200_interior` (deep-interior, 45 k iters), 200 unit +
      11 goldens 🟢, parité full corpus atom-ON. Reste : wfs_mb résidu inside_mm=12 (≈parfait,
      vs 65479 avant) ; mid-range 1e13–1e280 pas encore atom (rebase-at-end pas porté au path f64,
      cf. glitch_test_2 1.30× — prochain levier).
    - **✅ wfs_mb inside_mm=12 ADJUGÉ : FRACTALL CORRECT, F3 FAUX (2026-07-12).**
      Les 12 px de désaccord bordent le cœur intérieur (~58 px au centre de la vue,
      évasions 7.7-9.7 M proches du cap 10 M). Probes GMP PUR 1×1 aux coordonnées
      exactes de 2 px opposés (orientation du mapping pixel→c validée : pert 1×1
      = EXR à 9 iters près sur 9.5 M) : (a) px(122,123) F3=intérieur/FR=évasion
      9 487 965 → GMP s'évade, **max_diff=1** ; (b) px(125,122) F3=évasion 8.5 M/
      FR=intérieur → GMP intérieur jusqu'à 12 M, **max_diff=0**. Dans les DEUX sens
      fractall = ground truth ; le résidu est l'erreur de F3 (précision de son type
      numérique à 1e2020 sur bord chaotique). **AUCUN déficit de correction
      fractall connu sur le corpus 84 cas.**
  - [x] **Auto-adjust × atom-truncated : orbite payée 2× en mode normal (2026-07-12).**
    L'heuristique auto-adjust (`skip_ratio > 25 % → double iteration_max + RECALCULE
    l'orbite`) prenait la réf ATOM-TRONQUÉE (ref_len ≪ iter_max, intentionnel) pour
    un signal « iter_max trop bas » : dragon 96² total 4.0 s (orbite 1.43 s × 2 +
    série × 2) et iter_max silencieusement doublé (pixels 5 M → 10 M, écart au TOML
    et à F3 qui n'ajuste jamais). INVISIBLE au harness (sweeps = NO_AUTO_ADJUST=1) —
    seul le mode normal CLI/GUI payait. Fix : l'heuristique ignore les réfs
    `atom_truncated` (elle reste active pour les réfs pleines/escape-truncated, son
    but d'origine). Vérif : goldens 18/18 pixel-exact (aucune image ne change : les
    cas concernés s'évadent sous l'iter_max demandé), quality 11 PASS, parité 10 ok.
    **Timing confirmé (3 runs A/B binaires, machine ~libre)** : dragon 96² mode
    normal 4.07-4.11 s → **2.05-2.11 s (2.0×)**.
    **Suivi FAIT (même jour)** : série gatée off quand `atom_truncated` + path
    bytecode (elle ne nourrissait que l'heuristique, désormais inerte sur ces réfs) —
    dragon 96² phase réf 1.98 → **1.40 s** (−30 %), série=0 aussi sur e50/gt2/e22522.
    `force_series` (debug) continue de forcer. Legacy path : série conservée.
  - [x] **TENTATIVE 10 — mid-range atom FAIT (2026-07-12) : gate atom étendu à
    pixel_size < 1e-13 + rebase-at-end porté au path f64 + guard BLA.** Changements :
    (1) `orbit.rs` gate `atom_period_enabled` : seuil `PIXEL_SIZE_EXP_THRESHOLD`
    (1e-280) → `ATOM_PERIOD_PIXEL_SIZE_THRESHOLD` (1e-13, delta.rs). NB : le
    fast-path dd (≤96 b, ~zoom<1e19) court-circuite la boucle GMP → tranche
    1e13–1e19 non tronquée (réf déjà bon marché). (2) Nouveau champ
    `ReferenceOrbit::atom_truncated` ; `pixel_loop.rs` (3 sites end-of-ref) rebase
    par `δ:=Z[end]+δ, m:=0` au lieu de `ref_exhausted`→GMP per-pixel.
    (3) **PIÈGE trouvé — guard BLA `lands_on_ref_end`** : sans lui, un pas BLA qui
    atterrit SUR la dernière entrée (graze |Z[end]|~4e-58 pour glitch_test_2) fait
    `continue` par-dessus le check end-of-ref → le pas direct suivant part du graze
    (δ'≈δ²+dc ~1e-115) puis le rebase ajoute Z[end] ≫ δ' qui ABSORBE δ en f64 →
    tous les pixels identiques → **image intérieure uniforme (tout noir)**. F3
    checke rebase/end AVANT chaque pas (`hybrid.cc:295-308`) ; notre boucle le
    fait après les pas directs seulement → on interdit au BLA d'atterrir sur la
    dernière entrée (réfs atom-tronquées uniquement, zéro impact ailleurs).
    **Résultats** : glitch_test_2 réf 250 k→1143 (orbite 0.226→0.002 s), total
    0.27→**0.022 s** (F3 0.17 s → WIN ~0.15×), avg_iter/px=7779 exact ; e50 réf
    1.05 M→86 615, parité Fr 0.55 s vs F3 3.35 s ; dragon réf 5 M→2.05 M, parité
    4.10 s vs atom-off 7.48 s (F3 9.9 s) ; floral 1705, gt1/gt3/gt4/heaven ✓.
    **Validation** : parité EXR 10 cas mid-range tous ✓, inside_mismatch
    IDENTIQUES atom-ON/OFF (dragon 28, gt1 44, gt2 12 = pré-existants) ; quality
    suite (voir commit) ; goldens : e50 178 px/max Δ15, e113 15 px/max Δ101
    (bruit dispersé plancher f64, revus visuellement), 15 autres PIXEL-IDENTIQUES.
    **Verrou** : golden `mandelbrot_glitch_test_2_atom` (régression guard BLA =
    tout noir). ~~Latent : même flaw dans pixel_loop_exp ?~~ **RÉFUTÉ (2026-07-12)** :
    la boucle HP (`iterate_pixel_unified_exp_mandelbrot_hp`) checke déjà le
    rebase/end-of-ref EN TÊTE de boucle (ordre F3, cf. son commentaire) — wfs_mb
    inside_mm=12 n'est PAS causé par ça. Le flaw ne subsiste que dans les loops
    exp non-HP (`_exp_mandelbrot` = dead code sauf FRACTALL_ATOM_PERIOD=0 ;
    `_exp_generic` = types non-Mandelbrot deep, jamais atom-tronqués) → bénin.
  - [~] **DIAGNOSTIC mid-range atom (2026-07-11) — glitch_test_2 = SEUL perdant mid-range,
    le fix est scopé mais coûte cher.** Mesure 3-runs des cas 1e13–1e280 : glitch_test_2
    **1.60×** (0.27/0.17 s) est le seul >1 ; integral_of_ex2/x/mitosis/virus/glitch_test_3/4
    = tous WINS (0.2–0.7×). glitch_test_2 orbite-bound (0.226 s / 0.239 s, réf INTÉRIEURE
    court les 250 k iters pleins, avg_iter/px=7779 max=250000). **Test throwaway** (gate atom
    abaissé à pixel_size<1e-13) : **l'atome FIRE — réf 250000→1143 iters** (période ~1143),
    orbite **0.226→0.001 s**. Image quasi-correcte (24 px/0.04 % vs atom-off). **MAIS pixels
    = 51 s** : le path f64 `pixel_loop.rs` n'a PAS le rebase-at-end (ligne ~293 : bail GMP
    per-pixel à end_of_ref) → les pixels qui dépassent 1143 retombent en GMP → catastrophe.
    - **Fix scopé** : (a) porter le rebase-at-end de `pixel_loop_exp.rs:332-340` (`δ:=z_ref[m]+δ ;
      m:=0`) dans `pixel_loop.rs` — SÛR dans le domaine f64 (mid-range : grazes >1e-280 ne
      sous-débordent pas, contrairement au deep qui exigeait ComplexExp) ; (b) étendre le gate
      atom à mid-range (`pixel_size < ~1e-13`).
    - **Pourquoi PAS fait maintenant (ROI/risque)** : gain = 100 ms sur 1 cas ; mais le gate large
      ferait FIRER l'atome sur floral_fantasy (période 284), glitch_test_5, glitch_test_1
      (intérieurs mid-range) → **change plusieurs goldens** + exige une validation mid-range
      complète vs F3 EXR. C'est un chantier « TENTATIVE 10 » (probablement CORRECTION+vitesse
      comme le deep, pas juste 100 ms) qui mérite une itération dédiée, pas un rush. Prérequis
      validé : le critère atome fire correctement mid-range, le rebase-at-end f64 est sûr.
- [x] **Path f64 étendu à 1e280 (seuil 1e-200 → 1e-280)** (2026-07-04) : après
  l'extension initiale à 1e-200, un sweep vitesse du corpus STANDARD/full a révélé
  4 cas encore sur le path exp lent (zoom > 1e200) donc PLUS LENTS que F3 :
  integral_of_ex2 1e202 (1.76×), x 1e235 (1.62×), mitosis 1e270 (1.61×), virus
  1e224 (1.57×). Validé f64 ≡ exp **pixel-identique** (0 diff) jusqu'à mitosis
  1e270 ; **casse à safari 1e307** (3 % px : δ~pixel_size devient SUBNORMAL sous
  2.2e-308). Seuil abaissé à 1e-280 (exp gardé pour pixel_size < 1e-280). Résultats :
  integral_of_ex2 1.76→**0.35**, x 1.62→**0.39**, virus 1.57→**0.62**, mitosis
  1.61→**0.70** (+ dinosaur_fossils, test3, 11_dimensions, e227, evolution_trees,
  adventurous_forest tous < 1). safari 1e307 correctement sur exp (0.057× — F3 lent).
  AUCUN golden touché (aucun dans 1e200–1e280). 187 unit + quality 11 PASS + goldens
  🟢 + parité inchangés. **Leçon : le tier quick (10 cas) ne couvre pas les gaps du
  corpus profond — sweeper `--axes speed --cases <full>` périodiquement.**
- [x] **Deep-zoom perf ~RÉSOLU — path pixel f64 étendu à 1e13–1e200** (2026-07-04) :
  **la plus grosse victoire de la session.** Le seuil `PIXEL_SIZE_EXP_THRESHOLD`
  forçait le path ComplexExp (mantisse+exp, ~10-20× plus lent/op via frexp) dès
  1e13. Mais F3 (`wisdom`, analysis §12) utilise **`double` jusqu'à ~1e300** : le
  delta est SUIVI séparément de z_ref, donc son itération `δ'=2Zδ+δ²+dc` est exacte
  en f64 tant que δ ne sous-déborde pas (~1e-308) ; z_ref+δ ne perd δ que quand il
  est déjà négligeable pour le bailout. Le seuil 1e-13 était un fossile d'avant les
  fix rebase-at-end (G2) + tolérance period-detection. Abaissé à **1e-200**.
  **Validation f64 ≡ GMP** (max_diff 0, ou divergence IDENTIQUE au path exp sur les
  pixels de bord) : seahorse 1e8…e18, **floral_fantasy 1e85** (l'ancien épouvantail
  « image uniforme » → PASS max_diff=0), glitch_test_2 1e112, e113, dragon 1e191
  (1 px/16384). **Gains 256²** : glitch_test_2 pixels 0.20→0.014 s (total 0.345→
  **0.155**, ratio 2.24→**1.30**) ; dragon pixels 3.66→0.18 s (total 7.0→**3.67**,
  ratio 1.9→**0.95 — passe DEVANT F3**) ; e50 0.22×, e113 0.37×, floral 0.21×.
  **geomean vitesse 0.65→0.33 ; 9 wins/10 ; AUCUN gap.** Golden `mandelbrot_e50`
  régénéré (5 px/16000, Δcouleur ≤2 — imperceptible, revu visuellement ; e113 et 13
  autres pixel-identiques f64==exp). 187 unit + quality 11 PASS + parité inchangés.
  Reste : glitch_test_2 1.30× = son **orbite GMP** (0.13 s/0.21 s), pas les pixels
  → levier orbite dd/float128 (tranche > 96 b). Path exp gardé pour zoom > 1e200
  (δ sous-déborde f64). Seuil overridable `FRACTALL_EXP_THRESHOLD`.
  - [x] **Skip du terme δ² quand droppé par l'add (bit-identique)** (2026-07-04) :
    dans le pas direct `δ'=2Zδ+δ²+dc`, l'add FloatExp rend `self` inchangé dès
    `exp(2Zδ) − exp(δ²) ≥ 54`. À deep zoom δ est minuscule ⇒ vrai sur la plupart
    des pas → on saute le calcul de δ² (`delta.mul(delta)` = 6 frexp + 2 des adds).
    Condition conservatrice SANS calculer δ² : borne sup `exp(δ²) ≤ 2·max(exp δ)+1`
    vs les deux composantes NON NULLES de 2Zδ (garde `mantissa != 0` : à l'itér. 0
    Z=0 ⇒ 2Zδ=0, l'add ne dropperait pas). **Bit-identique** (l'add renvoie
    exactement `self`). glitch_test_2 pixels 0,216→0,202 s (−6,5 %), total
    0,360→0,345 s (5 runs interleaved vs F3 0,11 s stable). dragon **plat** (réf.
    BORNÉE → δ reste ~O(1) près des rebases, δ² rarement droppable ; la garde ne
    coûte rien quand non prise). Goldens 🟢 + 187 unit + quality 11 PASS + parité
    inchangés. ⚠️ Le ratio harness quick est sous le plancher de bruit thermique
    pour ce petit cas (mesurer interleaved standalone, cf. mémoire).
  - [x] **frexp sans division (bit-identique) + DIAGNOSTIC CORRIGÉ** (2026-07-04) :
    l'instrumentation `[DIAG BLA]` (compteurs bla_steps/direct_steps) a RÉFUTÉ
    l'IMPASSE #2 « memory-bound ». Faits mesurés : dragon = **286,6 M pas DIRECTS**
    vs 9,9 M pas BLA (les BLA couvrent 256,7 G itérations, avg_skip 25968, mais ne
    sont que ~3 % des tours de boucle) ; 286,6 M × ~13 ns ≈ 3,7 s ≈ toute la phase
    pixel. glitch_test_2 idem (15,6 M directs ≈ 0,2 s ≈ sa phase pixel). Les pas
    directs walkent `m` séquentiellement (préfetch → PAS cache-miss) → le goulot
    est **le pas direct `δ'=2Zδ+δ²+dc` en ComplexExp**, jamais optimisé (les IMPASSE
    #1/#2 visaient le pas BLA = mauvaise cible). Le pas direct fait ~16 `frexp` (1
    par `FloatExp::new`/`Mul`/`Add`). Fix : chemin normal de `frexp` sans conversion
    int→f64 ni division — on force le champ exposant biaisé à 1022 via bits
    (`from_bits((bits & 0x800f…) | 1022<<52)`), **bit-identique** (même significande,
    exposant −1 → mantisse ∈ [0.5,1)). dragon total 7,55→**7,06 s** (ratio 2,07→1,91),
    glitch_test_2 pixels 0,259→0,216 s (−17 %), total 0,404→**0,360 s** (ratio
    2,85→2,24). geomean vitesse 0,765→**0,699**. Goldens 🟢 pixel-exact + 187 unit +
    quality 11 PASS + parité inchangés. **Prochain levier (glitch_test_2)** :
    réduire le NOMBRE de frexp/pas direct — normalisation spécialisée Mul (produit
    de 2 mantisses ∈ [0.5,1) ⇒ ∈ [0.25,1), 1 seule branche `<0.5 ? ×2` au lieu de
    frexp), ou pas direct scaled-f64 (non-bit-identique).
  - dragon : orbite GMP ~3.26 s (mul complexe 676 b × 5 M iters, cœur
    incompressible sans float128) + pixels ComplexExp ~5.5 s. Leviers restants
    = wisdom/float128 (orbite, cf. AskUser : middle-range only) ET boucle pixel
    exp f64-scaled (non-bit-identique, cf. P1.6.d/e).
  - glitch_test_2 : fractall 0.605 s stable vs **F3 0.113 s stable** → ~3.5-5.4×
    (bruit F3). Zoom 1e112, orbite 0.23 s + pixels ComplexExp 0.33 s.
  - **IMPASSE (2026-07-04) : spécialiser `FloatExp::mul`/`sqr` (éviter `frexp`
    via une branche, bit-identique) = 0 gain mesuré** sur glitch_test_2 ET
    dragon (pixels inchangés). Cause : le pas hot du `pixel_loop_exp` est le
    **pas BLA** (`a.m00*δ.re + a.m01*δ.im + b.…`, `pixel_loop_exp.rs:201-204`)
    qui utilise `Mul<f64> for FloatExp` (multiplicateur mat2 non normalisé →
    frexp obligatoire) + `Add` (frexp). Le `δ.mul(δ)` direct (FloatExp×FloatExp,
    la seule op que la spéc accélère) ne tourne que sur les *rares* pas
    non-skippés par la BLA.
  - **IMPASSE #2 (2026-07-04) — ⚠️ RÉFUTÉE plus bas (frexp DIAG)** : la conclusion
    « memory-bound » était FAUSSE (le spike f64-scaled visait le pas BLA, ~3 % des
    tours ; le vrai goulot est le pas DIRECT, arithmetic-bound). Conservée pour la
    leçon : ne pas conclure « memory-bound » sans compter les pas directs vs BLA.
    **IMPASSE #2 (texte d'origine) : pas BLA f64-scaled = ~0 gain.** Spike : produit mat2-vecteur
    `A·δ + B·dc` extrait à exposant commun + f64 pur + 1 seul frexp final (vs
    ~14 frexp/pow2i op-par-op). Mesuré : glitch_test_2 pixels 0.33→0.317 s
    (**~4 %**), **dragon pixels inchangés** (~5.6 s). Cause : `z_ref_f64[m]` est
    lu à chaque itér. ; dragon a 5 M entrées × 16 o = **80 Mo ≫ L3** → chaque
    lookup = cache miss qui domine, réduire l'arithmétique ne change rien.
    glitch_test_2 (250 k × 16 = 4 Mo, tient en cache) montre le petit 4 %. **Le
    vrai levier pixel = MÉMOIRE** (layout cache-friendly de `z_ref_f64`, ou
    réduire sa taille / stride BLA), PAS l'arithmétique f64-scaled. Spike
    reverté (non-bit-identique + gain trop faible pour justifier une régén
    goldens). Confirme l'IMPASSE #1 (FloatExp::mul = 0 %) : 3e preuve que
    l'arithmétique ComplexExp n'est pas le goulot.
  - [x] **`z_ref_gmp` dense = build PARESSEUX** (2026-07-04, `deac768`) : le
    stockage GMP dense (clone par itér., ~0.7 s + 850 Mo sur dragon) n'est LU que
    par `iterate_pixel_gmp` (rendu full-GMP + correction glitch). Sauté sur le
    path bytecode non-full-GMP via `compute_reference_orbit(_,_,force_dense_gmp)` ;
    la pass glitch le recompute UNE fois (même chemin bytecode + `true` →
    bit-identique) quand des pixels glitchent. Garde-fou `is_valid_for`.
    **dragon orbite 5.48→4.83 s, ratio 3.05→2.97, 850 Mo libérés, pixel-
    identique**. Verrou : golden `mandelbrot_cusp_m075` (glitche → exerce le
    recompute) + bit-identité dragon. Les deep zooms sans glitch
    (dragon/e50/e100 : corrections=0) ne paient plus le stockage mort.
  - [x] **`BlaTableUnified` construite UNE fois, partagée via `Arc`** (2026-07-04) :
    le cache BLA était **thread-local par worker rayon** → chaque worker
    reconstruisait la table entière (16 builds redondants + 16 copies en RAM).
    Sur dragon (4 M nœuds) : build ~1.2 s **contendu ×16** (1.19→1.67 s wall) qui
    dominait le début de la phase pixel ET saturait la bande passante mémoire (le
    goulot pixel identifié memory-bound ci-dessus). Fix : `GLOBAL_BLA` (Mutex) — le
    premier worker construit sous le lock, les autres clonent l'`Arc` ; le
    thread-local ne cache plus que l'`Arc` (accès par-pixel lock-free). **Un seul
    build, une seule table.** dragon pixels 5.71→4.69 s (−18 %), total 8.92→7.91 s,
    ratio 2.63→2.29 ; glitch_test_2 pixels 0.342→0.270 s (−21 %), ratio 3.48→3.14 ;
    e113 sorti des gaps. **Bit-identique** (même table, goldens 🟢 pixel-exact +
    quality 11 PASS + parité inchangés).
  - [x] **Build BLA parallélisé + pré-chauffé hors du lock** (2026-07-04) : suite du
    point ci-dessus. `BlaTableUnified::build` map level-0 + merges en rayon
    (order-preserving → bit-identique) au-delà de `PAR_BLA_MIN=65536` nœuds ; sous
    le seuil (goldens ~2,5 k, e113 35 k) reste serial (pas d'overhead). Le build est
    **pré-chauffé** par `delta::prewarm_bla_entry(params, &cache.orbit)` appelé dans
    `render_perturbation_with_cache` AVANT la boucle pixel (pool rayon libre) : sous
    le lock global depuis un worker, les 15 autres sont parqués → rayon ne les vole
    pas, donc le parallélisme n'a d'effet QUE pré-chauffé. dragon build serial ~1,2 s
    → parallèle **~0,11 s**, total 7,91→7,55 s (ratio 2,19→2,07) ; glitch_test_2 plat
    (table 250 k déjà ~20 ms). Sauté sur hybrides (`hybrid_refs.is_some()` → autres
    orbites). Goldens 🟢 pixel-exact + 187 unit + quality 11 PASS + parité inchangés.
    Reste : le pixel loop deep-zoom lui-même (memory-bound, cf. IMPASSE #2) — dragon
    pixels ~4,3 s, levier = layout cache-friendly `z_ref_f64` / stride BLA.
- [x] **✅ Re-sweep corpus complet avec le fix `[2026-07-14]`** : sweep full
  post-escalade (cf. G8.2) — **80/80 wins, geomean 0.193, plus aucun perdant** ;
  les ex-perf rendent et matchent (seul e52465 quarantainé, adjugé plancher-GMP).
- **Acceptation : ✅ ATTEINTE** — e50 **1.57 s**, e1000 **0.53 s**, dragon
  **6.46 s** à 256² (cible <180 s, marge 30–340×). L'objectif RECALIBRÉ
  (e50/e1000 <180 s ; dragon relatif à F3) est **dépassé** : l'absolu d'origine
  est atteint pour les trois.

> ⚠️ **Leçon (énorme)** : j'ai optimisé pendant des heures le path perturbation
> (memory/BLA/SIMD) alors que les cas cibles **ne l'utilisaient même pas** — ils
> tombaient en GMP. **Toujours confirmer QUEL path s'exécute** (compteur par
> fonction) AVANT d'analyser sa perf. Le compteur `EXP_PIXELS=1` vs
> `entry_calls=2304` a été le révélateur. (Note annexe : le throttling thermique
> du M4 a aussi biaisé les A/B back-to-back — contrôler l'état thermique.)

### G3 — Élucider les divergences ouvertes · `[P0 · correction]`

- [x] **✅ Réf INTÉRIEURE frôlant zéro → structure spurious à >512² — RÉSOLU
  (2026-07-17)**. Trouvé via screenshot GUI (centre -0.5622-0.6428i, zoom
  ~2.66e10). La réf ne s'évade pas (intérieure) et **frôle zéro**
  (min|Z|²≈1.5e-11) → annulation f64 au rebase → 36 % de pixels flaggés glitch.
  **Cause** : le 2ᵉ bloc de résolution glitch récursive (`mod.rs` ~l.1642)
  n'était PAS gaté `!bytecode_path` (contrairement à son frère l.1534) → à
  **>512² uniquement** (`!small_image`) il re-rendait les pixels flaggés via
  l'`iterate_pixel` LEGACY (réf secondaire, MÊME imprécision f64), les
  dé-flaggait (`glitch_mask=false`), faisait tomber le glitch_ratio sous
  `GLITCH_FALLBACK_THRESHOLD` (0.30) et **supprimait le fallback full-GMP** →
  **15048 px (3.44 %) de structure spurious** à 800×547 (PASS à ≤512²). **Fix**
  (1 ligne) : gate `&& !bytecode_path` → glitch_ratio reste 0.365 > 0.30 →
  fallback GMP → **640² == GMP pixel-exact (0 spurious)**. Diagnostic 3-voies :
  f64-standard OK (65 flips), dd-tier PARFAIT, **GPU perturbation PARFAIT (20 px)**
  → le screenshot GPU de l'utilisateur était en fait CORRECT ; seul le CPU f64
  perturbation était faux. Verrous : preset `mandelbrot-interior-ref` (coords +
  comment ; ⚠️ ne reproduit qu'à >512², suite 256² PASS) + comment de garde
  `mod.rs`. 258 unit + 21 golden + QA 256² sans régression. **Reste** : le
  fallback full-GMP est LENT (~2 min à 800²) sur ce cas pathologique rare (GUI =
  GPU, non concerné) — un fallback **dd** (≈25× plus rapide que GMP) est
  l'optimisation perf naturelle (G9.6, nécessite le câblage orbite dd).
  ⚠️ **NB méthodo** : l'hypothèse initiale « plancher de précision f64 exigeant
  dd auto-dispatch » était FAUSSE — le fallback GMP existant suffisait, il était
  juste court-circuité par un bloc de correction legacy mal gaté.
  - [x] **✅ Fallback dd (perf) — FAIT (2026-07-17)**. Le fallback full-GMP
    (`iterate_point_mpc`, ~1 µs/iter) sur ce régime (glitch_ratio > 0.30,
    Mandelbrot bytecode) est remplacé par une **escalade tier dd** : re-rendu de
    la frame avec `use_dd_tier=true` (récursion 1 niveau, garde `!use_dd_tier`).
    Le pixel loop dd NE flagge PAS ces pixels (`glitched_initial=0`) → il rend
    proprement sans re-déclencher le fallback → **pixel-exact GMP** (vérifié 640²
    == GMP, 0 spurious) en **~14 s** (640²) vs full-GMP ~60-120 s+ (≈4-8×). Le
    full-GMP reste le **backstop** si dd échoue/flagge encore (`delta.rs`). ⚠️ Ne
    se déclenche qu'à **>512²** (à ≤256² le glitch_ratio reste < 0.30 → correction
    per-pixel, cas non touché → suite quality/goldens inchangés). Verrou = comment
    `mod.rs` (même limite de résolution que le gate ci-dessus).

- [x] **✅ CPU perturbation : anneaux concentriques près du cusp -0.75 — RÉSOLU
  (2026-05-21)**. Centre ≈ (-0.749996, -0.004086), zoom ≈ 2.8e10, 2500 iter.
  **Vraie cause** : `max_perturb_iterations` (cap des pas DIRECTS) défaut **1024**
  < iter requis (~1700). Comme `iters_ptb ≤ n < iteration_max`, ce cap ne mord
  QUE s'il est < iteration_max → les pixels tronquent tôt avec un compte
  d'itération ~radial ⇒ anneaux. Le loader TOML faisait `= iters` (donc OK en
  CLI --toml) mais GUI + CLI non-TOML restaient à 1024. **Fix** : clamp
  `max_perturb_iterations`/`max_bla_steps` à `≥ iteration_max` dans
  `render_perturbation_with_cache` (chemin commun CLI+GUI+quality), alignement F3
  (`maximum_perturb_iterations = iterations`). Résultat : cusp == GMP (mean 1.29
  vs 56.87 avant) ; e113 golden encore plus proche du GMP pur (mean 0.005 vs
  0.089) → régénéré + revu. **Fausses pistes écartées** : BLA (off = pire),
  précision (256/1024/4096 identique), period-detection (NO_PERIOD identique),
  optimize_reference_center (pas de snap ici), exp-vs-f64 (les deux ringaient).
  178 unit + golden verts. *(Zone figée en golden : `mandelbrot_cusp_m075`.)*
- [x] **✅ floral_fantasy : image UNIFORME en chemin par défaut — RÉSOLU
  (2026-05-21)**. zoom 1.55e85. Défaut (period ON) → 1 couleur ; NO_PERIOD → 194
  (correct). **Cause** : période détectée = **FAUX-POSITIF** (graze) — tolérance
  Brent `2^(-prec·0.4)` trop lâche → troncation + `wrap_periodic` sur réf fausse
  → uniforme. **Fix** : tolérance resserrée à `2^(-prec·0.85)` (orbit.rs) — les
  vraies périodes (~2^(-prec)) restent détectées, les grazes rejetés → escape-time
  + rebase-at-end (correct). Défaut == NO_PERIOD == F3. Verrouillé : golden
  `mandelbrot_floral`. Scan corpus : 0 nouveau uniforme, 0 régression perf.
- [x] **glitch_test_1 — anneaux concentriques : TRANCHÉ (2026-05-21)** vers une
  **victoire fractall**. Le détecteur F3-dégénéré du harness (timing + uniformité)
  flagge glitch_test_1 : **F3 rend un extérieur uniforme (0.3 % intérieur) en
  0.085 s (fast-path)** tandis que fractall rend une vraie structure. Combiné aux
  diagnostics antérieurs (GMP pur fractall ≡ perturbation ≠ F3 ; précision/BLA/
  series/period tous écartés), c'est la même signature que glitch_test_5 : F3
  court-circuite sur ce lieu glitch-prone, fractall est correct. Exclu du score.
- [x] **✅ julia-siegel-disk FAIL — RÉSOLU (2026-07-03, BLA over-skip)**.
  `fractall-quality` pert-vs-GMP : décalage UNIFORME de +2 iters (div_ratio
  1.0, max_diff=2, p99=2) sur TOUTE l'image — signature d'un bug systématique,
  pas du bruit de bord. **Cause** : à c=-0.8+0.156i l'orbite de référence
  (critique, centre 0,0) est BORNÉE (siegel) mais les pixels s'échappent vers
  iter ~254. Un pas **BLA multi-étapes** est linéarisé autour de la référence
  bornée → aveugle à la divergence propre du pixel → il **saute par-dessus
  l'évasion** et rapporte l'iter d'escape jusqu'à `l-1` trop tard (F3 fait le
  même over-skip en `hybrid.cc:316-320` ; le GMP pur, sans BLA, non → PASS
  max_diff=0 en désactivant la BLA le confirme). **Fix** (`pixel_loop.rs`,
  single-phase + hot-path Mandelbrot) : rejeter un saut BLA `l≥2` dont le point
  d'arrivée `Z[m']+δ'` est déjà échappé (escape irréversible car |z|>ER≫|c| ⇒
  test du seul endpoint suffit) → single-step pour l'iter exacte. Rend fractall
  **strictement plus correct que F3** ici. Verrou : test unit
  `bla_no_overskip_past_escape_julia_siegel` (BLA ≡ f64 direct, 9216/9216
  pixels, biais 0) + preset quality `julia-siegel-disk` (FAIL→PASS).
  - **✅ Guard porté au path exp (2026-07-10)** : `pixel_loop_exp.rs` partageait
    le même bloc BLA SANS le guard, sur les **3** fonctions (mandelbrot inline,
    mandelbrot_hp `bla_exp`, generic Julia/autres). Bug latent (exp seulement
    utilisé > 1e278 en prod), mais **reproduit** en forçant le path via
    `FRACTALL_EXP_THRESHOLD=1` sur `julia-siegel-disk` : **FAIL div_ratio 1.0,
    +2 uniforme** (signature over-skip identique au f64 pré-fix) → **PASS
    max_diff 0** avec le guard. Test chaque endpoint `Z[m']+δ'` (ComplexExp/
    FloatExp selon le bloc) ; escape irréversible → un seul point suffit.
    Verrou : unit `exp_bla_no_overskip_past_escape_julia_siegel` (BLA exp ≡ f64
    direct, biais ~0). Goldens 10/10 pixel-exact (e1000 exerce le path exp,
    inchangé — les Mandelbrot deep à réf longue n'over-skippent pas).
- [x] **✅ seahorse-valley / e13 / e17 « FAIL » = bruit de bord — RÉSOLU via
  recalibration G6 (2026-07-10)**. Ces cas ont ~2-24 pixels **dispersés** au bord
  (plancher f64 : delta grandi, |dz| ~ 200-430) → `max_diff` grand (137/210/7)
  mais `p95=p99=0`, `div_ratio` minuscule (2e-5…1.5e-3). L'ANCIEN gate FAILait sur
  le `max` outlier → FAIL permanent, noyant le vrai signal du loop /improve.
  **Fix** (`quality/metrics.rs::classify`) : gate robuste — FAIL si `p99 > 1`
  (divergence LARGE) OU `div_ratio > warn` (SYSTÉMATIQUE, ex. over-skip = +N
  uniforme div_ratio 1.0) ; le `max` outlier seul → **WARN**. Résultat mesuré
  256² : seahorse/e13/e17 **FAIL→WARN**, misiurewicz/julia-siegel/… restent PASS.
  Le vrai bug (over-skip forced-exp, div_ratio 1.0) resterait FAIL. Verrous : unit
  `verdict_widespread_and_systematic_fail` + `verdict_pass_warn_fail_tiers` (maj).
  `max` toujours rapporté dans report.md. NB : la divergence physique (plancher
  f64) est réelle — seul `--dd-tier` la supprime (cf. wisdom auto-dispatch, G3) ;
  la recalibration classe juste correctement « bruit épars » vs « régression ».
- [x] **✅ Tier double-double (~106 b, « float128 » pur-Rust) livré (2026-07-04)**.
  Diagnostic : e30/e50 FAIL = plancher mantisse **f64** (53 b). `z_ref`
  (Complex64) ET `delta` (ComplexExp) stockent Z et δ à 2⁻⁵² relatif ; en
  spirale ultra-sensible (centre `-0.04947…−0.67478…`) l'amplification de
  Lyapunov sur ~25000 iters transforme ce 2⁻⁵² en O(1) → ±50-200 iters vs GMP
  (`escape_disagreement=0`, pixels les plus profonds, dispersés — PAS un rebase
  manqué : hypothèse rebase-avant-BLA testée + revert, div_ratio bit-identique).
  **Fix** : tier dd opt-in (`params.use_dd_tier` / CLI `--dd-tier`) qui **stocke
  la référence en dd** (`ReferenceOrbit::z_ref_dd` via `gmp_float_to_ddexp`,
  indispensable car Z entre non-arrondi dans `2·Z·δ`) et **itère le delta en dd
  sans BLA** (`pixel_loop_dd.rs` ; la BLA f64 réintroduirait 2⁻⁵² tôt quand δ est
  minuscule). Noyau `ComplexDDExp` préexistant (`dd.rs`).
  **Résultats finaux** (quality 96²) : **TOUTE la suite pixel-identique GMP** —
  5P/1W/5F (début) → **11 PASS / 0 WARN / 0 FAIL**. Tous les cas précision-limités
  (spirales e30/e50/e100 + seahorse 1e8 + Misiurewicz 1e12 + minibrot 1e18)
  passent max_diff=0. Parité (voire mieux) avec le float128 de F3. Coût : ~4-10×
  plus lent (pas de BLA) → opt-in. Verrous : presets quality (dd activé sur les
  cas sensibles) + tests `pixel_loop_dd`/`dd` (incl. `dd_div`).
  - [x] **✅ `dc` en dd** (le levier des pixels de bord). Le `dc` ComplexExp (53 b :
    span f64 × fraction f64) laissait ~1-3 pixels de bord (grand |dc|) diverger
    (e30 max_diff 106, e50 8) — pixel calculé à un `c` décalé de 2⁻⁵²·|dc| vs GMP.
    Fix : `effective_spans_dd` (span depuis la string HP → 106 b), fraction pixel
    via **division dd** (`DoubleDouble::div`/`DoubleDoubleExp::div`, algo Bailey),
    threadé via `iterate_pixel_with_dd(dc_dd)`. → e30/e50 **max_diff 0**. (Mon
    hypothèse initiale « plancher d'itération 106 b » était fausse : c'était le dc.)
  - [x] **✅ dispatch dd hoisté à tout range** (avant le split exp/f64) : le dd loop
    marche à n'importe quel zoom (réf dd + dc dd construits sans gating de zoom), donc
    couvre aussi le path f64 (< 1e13). → **seahorse 1e8 / Misiurewicz 1e12 / minibrot
    1e18 FAIL/WARN→PASS max_diff=0**. La **sensibilité** (pas la profondeur) dicte le
    besoin dd. NB : le dc-en-dd étant nécessaire (n'a rien à voir avec la BLA), la
    cause est bien la **précision**, pas un over-skip BLA masqué par le no-BLA.
  - [x] **✅ dd-BLA** (`bytecode/bla_dd.rs`) : table BLA à coefficients `Mat2Dd`
    (~106 b) depuis `z_ref_dd`, single-step Mandelbrot `A=2Z` + merge F3, appliquée
    dans `pixel_loop_dd` (`δ:=A·δ+B·dc` via `DoubleDoubleExp::mul_dd`) avec
    rebase-avant-BLA (F3 `hybrid.cc:295`). Câblée/cachée dans `delta.rs`
    (`dd_table` par orbite). **Clé** : epsilon = **2⁻¹⁰⁶** (epsilon machine dd),
    pas `bla_threshold` 2⁻²⁴ — sinon la BLA borne δ² à ~24 b et masque la précision
    dd (div_ratio revenait au plancher f64). Correction préservée (suite reste
    **11/0/0 max_diff=0**). Gain : **e100 2426→845ms (~3×)** ; e50 29→21s. Modeste
    sur e30/e50 : leur δ (1e-30/1e-50) + forte amplification effondrent les rayons
    mergés → surtout des pas directs (inhérent, comme F3 float128 à zoom modéré).
    Verrous : tests `bla_dd` (single/merge vs f64 BLA, table+lookup).
  - [x] **✅ Tuning perf dd-BLA (2026-07-15)** : epsilon **2⁻¹⁰⁶ → 2⁻⁸⁰**
    (constante calibrée, pas d'adaptatif). Fondement : l'erreur relative par
    saut accepté est ~ε/2 quelle que soit sa longueur (δ² droppé borné par
    r = ε·|Z|) → 2⁻⁸⁰ reste ≫ sous le bruit f64 (2⁻⁵²) qui motive le tier dd,
    en élargissant les rayons ×2²⁶. Bissection QA (presets dd 96², max_diff
    vs GMP) : 2⁻⁸⁰/2⁻⁷² exacts partout, **2⁻⁶⁴ casse e30 (max_diff=14)** →
    2⁻⁸⁰ = 16 bits de marge sous la falaise observée (falaise DÉPENDANTE du
    cas → pas de constante plus lâche sans nouveau verrou). Gains :
    **e50 18.7→8.0 s (~2.3×)**, e100 744→505 ms (1.5×), e30 ~neutre (bruit),
    presets peu profonds inchangés (δ relatif au-dessus des rayons avant
    comme après). Suite 15/15 PASS max_diff=0 (le preset e30 EST le verrou
    anti-ε-lâche, prouvé par la bissection). L'epsilon ADAPTATIF (par
    δ/sensibilité) reste ouvert mais bloqué par le même mur que l'auto-
    dispatch dd : pas de détecteur de sensibilité cheap fiable (cbits réfuté).
  - [x] **✅ Échelle wisdom (module `fractal/wisdom.rs`, 2026-07-12)**. Source
    UNIQUE de la sélection de tier numérique (`number_tier` : dd demandé > exp
    > f64), consommée par `bytecode_path_label` ET `try_bytecode_unified_path`
    (dédup de la logique jadis copiée aux deux endroits). Calcule le
    `WisdomPlan` inspectable (algorithme + tier + exposant/mantisse requis
    F3-style `render.cc:219` + précision GMP orbite) ; log `[WISDOM]` via
    `FRACTALL_WISDOM=1`. **Seuils calibrés préservés** (aucune régression :
    208 unit + 21 golden pixel-exact + sweep-lock `tier_matches_legacy_thresholds`).
    Confirme empiriquement le modèle F3 : à 1e300 l'escalade f64→exp se fait sur
    l'**exposant** (`req_exp=1003`) tandis que `req_prec` reste 8 b ≪ 53 b — la
    mantisse ne force JAMAIS l'escalade sur une frame centrée. `log2_zoom`
    extrait (HP-aware, partagé avec `compute_perturbation_precision_bits`).
  - [ ] **auto-dispatch dd (wisdom, suite)** : sélectionner dd automatiquement
    selon une estimation de sensibilité/conditionnement par frame (F3-style), au
    lieu de l'opt-in `use_dd_tier`. **Hors périmètre de l'échelle ci-dessus**
    (décision 2026-07-12 : variante sûre d'abord) : la viabilité F3 ne réclame
    jamais dd (mantisse requise ~8 b), le besoin vient de la sensibilité pixel
    qu'aucun détecteur cheap fiable ne capte (proxy `cbits` réfuté, cf. ci-dessous).
    Point d'accroche prêt dans `wisdom.rs` (`required_precision` vs
    `tier.mantissa_bits()`). Prérequis perf : dd-BLA ✅ (epsilon 2⁻⁸⁰ calibré).
    - [ ] **📐 DÉTECTEUR PROPOSÉ (survey web 2026-07-15) : borne d'erreur
      propagée par-pixel + test de shadowing — « Reliable Mandelbrot »
      (Claude Heiland-Allen, https://mathr.co.uk/web/m-reliable.html).**
      Idée : une orbite f64 bruitée est CORRECTE au sens pixel tant que son
      erreur, ramenée en espace-c via la dérivée, reste sous ~1 pixel
      (shadowing) : `fiable ⟺ Δz_n < κ·pixel_size·|dδ_n/dδc|` (κ ≈ ½…1).
      Borne `Δz` propagée par récurrence (u = 2⁻⁵³), transposée à nos trois
      événements de `pixel_loop.rs` :
      · pas direct : `Δz ← Δz·(2·|Z+δ| + Δz) + u·2·|Z|·|δ|` (normes déjà
        cachées en tête de boucle) ;
      · saut BLA (l iters) : `Δz ← ‖A‖·Δz + (ε_bla + u)·|z_land|` ;
      · **rebase `δ:=Z+δ` : `Δz ← Δz + u·|Z_m|`** (ajout fractall — capture
        EXACTEMENT le mécanisme e13 : la cancellation révèle l'arrondi f64 du
        Z de référence face au δ minuscule).
      La dérivée `dδ_n/dδc` = le tracking dual-number EXISTANT du path
      distance-estimation (`UnifiedOptions`/ddelta) — rien à inventer.
      **Pourquoi ça sépare là où cbits échouait** : cbits comptait l'activité
      de cancellation sans propagation multiplicative NI normalisation par la
      dérivée. Vérifié contre les 3 points de données du repo : e13 (2 px,
      amplification Lyapunov ≠ entre px à cbits voisins → Δz/|dδdc| sépare) ;
      e50 (3.2 % px cbits≥8 STABLES : |dδdc| énorme à deep zoom → tolérance
      `pixel_size·|dδdc|` explose → pas de faux positifs) ; presets dd
      shallow (seahorse 1e8 : grande Δz / dérivée modeste → flag, le cas
      invisible au req_prec F3). mathr conclut comme nous : « per-pixel
      status » obligatoire, aucun proxy par-frame ne marche. AUCUNE
      implémentation existante (KF/F3 sélectionnent par profondeur seulement)
      — première pratique, adossée à l'analyse de l'auteur de F3.
      **Plan en 2 itérations /improve** :
      1. Détecteur seul en OBSERVATION (`FRACTALL_RELIABILITY=1`) : tracking
         Δz + dual dans le path f64, flag dans `UnifiedPixelResult`, ligne
         `[RELIABILITY] flagged=N/total`. Validation = précision/rappel vs
         vérité GMP per-pixel sur les presets QA (doit attraper les 2 px
         d'e13@256² + les divergents seahorse/misiurewicz/minibrot en f64
         forcé ; ~0 faux positif sur e50/e113). Coût attendu ~10-25 % quand
         activé (dual 2 muls + borne 3-4 flops/iter ; borne cheb majorante
         si le sqrt gêne).
      2. ESCALADE : px flaggés ré-itérés sur `pixel_loop_dd` (même motif que
         l'escalade still-glitched→GMP, verrou `single-ref-glitch-interior`) ;
         fraction flaggée > seuil (~5-10 %) → bascule frame entière dd = le
         wisdom auto-dispatch débloqué. e13 : 2 px ≈ gratuit ; seahorse :
         flag massif → dd frame, la bonne réponse.
      **Risques** : borne conservatrice (produit de majorations) → sur-flag,
      κ/seuil à calibrer sur corpus QA (méthode bissection, cf. epsilon dd) ;
      coût du tracking permanent → si > ~10 %, opt-in `Auto` gaté wisdom
      (zoom < 1e18 + iters élevées, la zone documentée du besoin dd) ; test
      en fin d'orbite seulement (suffisant a priori — l'erreur qui fausse un
      escape est dans l'état final) à confirmer en validation 1.
    - **Cas moteur pour le wisdom (2026-07-10, diagnostic)** : preset quality
      `mandelbrot-e13` (centre `-1.7499537683537087`, zoom 1e13, 16384 iters —
      juste au-dessus du seuil d'activation perturbation). **FAIL à ≥128²**
      (2 px symétriques, `iter_diff=201` : perturbation f64 escape à n=1261 avec
      `|z|=33.6` alors que le GMP ground truth n'escape qu'à n=1462), **PASS à 96²**
      (les pixels pathologiques ne tombent pas sur la grille). Donc INVISIBLE au
      tier `quick` (quality 96²) mais réel au `suite` défaut (256²) — les 2 px
      apparaissent aux coords (131,118)/(131,137) à 256². **Racine** : plancher
      f64 du δ. L'orbite de c≈-1.75 (pointe de l'antenne période-2) repasse
      périodiquement près de |Z|≈0.027 → le pixel rebase 11× et chaque rebase
      `δ:=Z+δ` (Z,δ comparables) subit une cancellation catastrophique ; sur
      ~1261 iters le 2⁻⁵² relatif devient O(1) → faux escape anticipé. Les boucles
      pixel f64 (nôtre) et F3 (`hybrid.cc`) sont **algorithmiquement équivalentes**
      pour Mandelbrot (rebase-avant-step au même index m, escape post-rebase) —
      ce n'est PAS un bug d'ordre. **`--dd-tier` corrige à 100 %** (0 px divergent
      vs GMP), confirmant que c'est bien la précision du δ, pas la BLA
      (dd tourne sans BLA). C'est le **même plancher que e30/e50** (cf. dd-tier
      livré ci-dessus), juste à zoom plus faible et sur 2 px seulement.
    - **🔬 TROIS VOIES fractall / F3 / GMP (2026-07-13, F3 Linux enfin dispo).**
      Question longtemps ouverte : le WARN e13 est-il un **retard sur F3** (F3
      rendrait ces px justes → porter sa précision) ou une **limite partagée** ?
      Réponse tranchée par mesure (256², ER 625 aligné, escape-count ENTIER vs
      GMP per-pixel ground truth) : **fractall-pert 16/65536 px faux** (99.98 %
      exact, dont 8 à ±1, 6 à ±2-5, **2** dd-sensibles à +210) ; **F3 9391/65536
      px faux** (14 %, mean 3.74, max 1582, 1612 px >5). Corr(pert,gmp)=0.9999,
      corr(F3,gmp)=0.928. Centre + coins : N entier **identique** F3=GMP=fractall.
      → **fractall est ~600× plus fidèle à GMP que F3 sur cette scène.** Le WARN
      e13 n'est PAS un retard sur F3 ; c'est fractall qui DOMINE F3 en exactitude,
      avec 2 px dd-résiduels comme seul défaut (F3 fait pire *partout*, sans doute
      parce que son wisdom par défaut choisit un tier bas — `float` 24 b — sur
      cette profondeur modérée, là où fractall tient f64 53 b minimum). ⚠️ Deux
      corrections au diagnostic 2026-07-10 ci-dessus : (a) direction — fractall
      **sur-compte** (escape TARDIF pert=867 > gmp=657), pas « anticipé » ;
      (b) « F3 algorithmiquement équivalent » vaut pour la *boucle* mais PAS pour
      la *précision effective* — reproductible via `compare_f3.py --only <probe>`
      (Δmean 3.75). **Conséquence méta** : F3 n'est PAS ground truth aux zooms
      modérés (tier bas) — l'axe parité mesure « match l'affichage F3 », l'axe
      qualité (vs GMP) reste le vrai juge de correction. Verrou : commentaire
      `mandelbrot-e13` (presets.rs) porte le chiffre 16 vs 9391.
    - **🔬 GÉNÉRALISÉ + ROOT-CAUSE (2026-07-13, `scripts/three_way_gmp.py`).**
      L'arbitre 3-voies (fractall-pert / F3 / GMP per-pixel, escape-count entier,
      seuil « grosse erreur » >5 px pour ôter le bruit ±1 cross-implémentation)
      confirme le pattern sur **4 scènes Mandelbrot zoom modéré** (192², erreurs
      >5 vs GMP) : seahorse-1e8 **fractall 41 / F3 15766** (385×), misiurewicz-1e12
      66 / 4341 (66×), e13 0 / 1087 (1087×), e17 2 / 1357 (678×). **Root cause
      côté F3** (`wisdom.cc::wisdom_enumerate`) : la wisdom PAR DÉFAUT (aucune
      `wisdom.toml` → `wisdom_default` = enumerate) donne `nt_float` **vitesse 0.6
      = la plus rapide**, et sa viabilité (`render.cc:243`) n'exige que
      `mantissa ≥ max(24, 24−pixel_precision)` ≈ **24 b** pour une frame centrée
      (même modèle `req_prec≈log2(diag)` que notre `wisdom.rs`). Donc F3 tourne en
      **float 24 b** tant que l'exposant float tient (jusqu'à ~1e38), et sa
      précision s'effondre là où la sensibilité de Lyapunov l'amplifie (filaments,
      antenne). fractall plancher **f64 53 b** → ~66-1087× plus fidèle. **Contrôle
      décisif** : à deep zoom (>1e38, float non-viable par exposant) F3 bascule
      floatexp ⇒ parité EXACTE (quick parity e50/e113/e1000 `mean_abs=0.0`,
      `max_abs=0.0`) — la divergence n'apparaît QUE là où float est viable, ce qui
      **exclut un artefact de mapping** et prouve le mécanisme tier. NB : l'écart
      est souvent **sous-visible** en rendu colorisé (quelques iters sur filaments
      chaotiques) — F3 assume ce compromis vitesse ; fractall assume la précision.
      Outil réutilisable = arbitre « qui a raison » quand la parité diverge.
    - **Détecteur cheap réfuté** : hypothèse « flag les pixels à forte
      cancellation cumulée aux rebases (`Σ ½·log2(|δ|²/|Z+δ|²)`) → re-render GMP ».
      **Infirmé sur données** : les 2 px fautifs ont `cbits≈11.2`, mais des px
      **corrects** montent à `cbits≈11.6` → pas de séparation. Pire, `cbits` ne
      corrèle qu'à l'activité de rebasing, PAS à la justesse : e50 a **3.2 %** de px
      `cbits≥8` (1.37 % `≥10`) et e113 **1.2 %** (0.45 % `≥10`) — tous stables au
      golden — contre **0.098 %** (0.015 % `≥10`) pour e13@256². Un flag `cbits≥T`
      + cap de comptage ne « marcherait » sur e13 que parce qu'il est peu profond
      (peu de px rebasent) → hack qui game la métrique, pas un détecteur de justesse.
      Réintroduire une correction GMP par-pixel contredirait aussi l'archi bytecode
      (le rebasing F3 a **remplacé** la glitch-detection, cf. CLAUDE.md + mod.rs:1341).
      **Conclusion** : pas de fix cheap/principled — c'est exactement le boulot du
      **wisdom auto-dispatch dd** (choisir dd par sensibilité de frame). Le seul
      « fix » correct aujourd'hui = `--dd-tier` opt-in.
- [x] **✅ Period-detection truncation = LOSSY → OFF par défaut — FAIT
  (2026-07-07, cf. G2 point 2)**. Brent OFF par défaut dans
  `orbit.rs::enable_period_detection` (opt-in `FRACTALL_PERIOD=1`), verrou
  golden `mandelbrot_glitch5`. Le rationale de cet item (erreur de
  quasi-périodicité accumulée par `truncate + wrap_periodic`) est exactement
  ce qui a été observé sur glitch_test_5 (75 % px faux). L'opt-in reste l'env
  var (pas `--find-nucleus`) : aux nucleus exacts l'orbite est exactement
  périodique et le rebase-at-end suffit déjà.

- [x] **✅ Couverture perturbation Celtic/Buffalo/PerpBS + diag précision-sensibilité
  (2026-07-13)**. Ces 3 types bytecode non-conformes n'avaient AUCUN verrou
  perturbation↔f64 (contrairement à Mandelbrot/Julia/BurningShip). Enquête partie
  d'une divergence apparente pert-vs-GMP à l'antenne `-1.75` (zoom 1e6 : Buffalo
  big(>5)=1465, PerpBS=603, inside_mm=505/202). **Tranché par cross-check à 3
  voies** : (a) f64-standard ET perturbation **coïncident exactement** (big=0) →
  la perturbation n'est PAS en cause ; (b) `buffalo_mpc`/`perpendicular_burning_ship_mpc`
  (gmp.rs) **matchent** la formule bytecode → pas de bug de formule ; (c) **stabilité
  GMP par précision** : GMP-128 ≠ GMP-256 (big=25) mais GMP-256 == GMP-512 == GMP-1024
  → la scène **exige 256+ b**, GMP-128 (défaut à 1e6) est lui-même non convergé. ⇒
  **PAS un bug** : l'antenne de ces familles hirsutes est à sensibilité de précision
  extrême (même classe que e13/dd-sensibilité, mais bien pire — abs-après-Sqr crée
  des annulations à chaque passage près de zéro), f64 ne peut la résoudre. **Verrous
  posés** : `perturbation_matches_f64_{celtic,buffalo,perpendicular_burning_ship}`
  (pert dispatcher == f64 direct à zoom modéré, GMP-free donc robuste au confound
  de précision). Message trompeur du garde-fou `quality::compare` corrigé (« la
  perturbation ne marche que pour M/J/BS/Tricorn » → FAUX ; c'est la comparaison
  vs GMP qui n'est fiable que là).
  - [x] **✅ Presets QA Celtic/Buffalo/PerpBS livrés (2026-07-13)** : suite quality
    étendue à 14 presets (`celtic-fold-edge`, `buffalo-fold-edge`,
    `perpbs-escape-band`, zoom 1e9) + garde-fou `quality::compare` élargi aux 3
    types. Sélection outillée : descente de zoom sur cellules frontière à faible
    variance locale + échappement invariant à une perturbation 2⁻⁴⁶ de c
    (les frontières hirsutes de ces familles échouent ce critère : GMP-256 lui-même
    non convergé, div_ratio 0.3-0.9 vs GMP quel que soit le tier — confirmé
    expérimentalement à 1e6 ET 1e9, cohérent avec le diag antenne ci-dessus).
    Les 3 scènes : **PASS max_diff=0 à 256²** (pert bit-exact GMP) + contrôle
    ground truth GMP-256 == GMP-512. Celtic/Buffalo : ~57-62 % intérieur +
    échappements profonds ~4090 (orbites longues) ; PerpBS : tout-échappement
    ~3000 (réf évadante → rebase-at-end). Un bug classe over-skip (+N uniforme)
    ou coefficient BLA non-conforme y FAILerait immédiatement.

### G4 — Hybrides multi-phase : la feature unique · `[P1 · différenciation]`

Chaîner des formules par phase (Mandelbrot ⊕ Burning Ship ⊕ …) — feature
absente de Kalles Fraktaler, partielle dans F3. L'infra `Formula::hybrid(vec)`
existe déjà ; il manque la BLA par phase, le nucleus phase-aware, et l'UI/CLI.

**État infra (audit 2026-07-17)** : l'interpréteur GMP (`GmpInterpState::step`)
cycle DÉJÀ les phases ; `compute_reference_orbit` itère via cet interpréteur
(donc multi-phase-ready) ; `iterate_pixel_unified_multi_phase` (pixel loop f64,
pas directs SANS BLA) existe. Il manquait un moyen de COMPILER un hybride, +
le câblage params/render, + la BLA par phase + le nucleus phase-aware + l'UI.

**Done when** :
- [x] **✅ Jalon 1 — compilation hybride `[2026-07-17]`** :
  `compile_hybrid_formula(&[FractalType], power)` (refactor behavior-preserving de
  `compile_formula` via `phase_ops_for_type` — goldens par-type inchangés).
  Compose une phase par type escape-time (réutilise le bytecode existant, aucune
  nouvelle sémantique) : `[Mandelbrot, BurningShip]` = Mandel-Ship alternant,
  `[M,M,BS]` = 2×M puis 1×BS. `None` si vide/type non-bytecode/power non-entière.
  6 tests (dont `hybrid_MM_iterates_identically_to_single_M`).
- [x] **✅ Jalon 2 — les hybrides RENDENT `[2026-07-17]`** :
  `params.hybrid_phases: Option<Vec<FractalType>>` + `formula_for_params(params)`
  (source unique : hybride si `hybrid_phases`, sinon mono-formule). CLI
  **`--phases mandelbrot,burning_ship`** (+ `FractalType::from_hybrid_name`). Le
  path **f64 standard** (`iterate_bytecode_f64` cycle DÉJÀ les phases ;
  `iterate_via_bytecode` tracking-path cyclé aussi) rend les hybrides ;
  `select_algorithm` **force StandardF64** pour les hybrides (perturbation
  rejette phases>1, GMP par-pixel = z²+c hardcodé) ; `render_dispatch` renvoie
  `None` (GPU ne cycle pas → fallback CPU). Vérifié : **[M,M] pixel-exact ==
  Mandelbrot** (invariant), [M,BS] genuine (≠ M ET ≠ BS, 4440/2974 px). Verrous :
  unit test [M,M]==M + golden `mandelbrot_hybrid_burningship`. Single-phase
  INCHANGÉ (`phases[iter % 1]` = phases[0], 24 goldens verts).
- [x] **✅ Jalon 3 — hybrides DEEP en perturbation (f64) `[2026-07-17]`** :
  `try_bytecode_unified_path` (`delta.rs`) route le multi-phase vers
  `iterate_pixel_unified_full` (→ `iterate_pixel_unified_multi_phase`, pas
  directs f64 + rebasing, SANS BLA) ; dd/exp gatés single-phase (multi-phase
  deep exp = jalon 4). `select_algorithm` route un hybride vers Perturbation
  dans la bande f64-perturbation (`pixel ∈ [exp_threshold, perturb_threshold]`,
  ~zoom 1e10–1e13 selon largeur), StandardF64 sinon. Orbite référence itérée
  avec la formule hybride (`orbit.rs` → `formula_for_params`, GmpInterpState
  cycle). **Vérifié : [M,M] @3e10 PIXEL-EXACT == Mandelbrot perturbation**
  (verrou render-level `hybrid_mm_equals_mandelbrot_deep_perturbation`, sans
  GMP externe — le GMP par-pixel ne cycle pas). Single-phase INCHANGÉ (tout
  gaté `multi_phase`).
- [x] **✅ Jalon 5a — références PAR PHASE : correction hybrides GENUINE deep
  `[2026-07-17]`** : port F3 `hybrid_references`/`hybrid.cc:266-341`. Le verrou
  [M,M] était AVEUGLE par construction (phases identiques ⇒ désync sans effet) ;
  un [M,BS] genuine @3e10 rendait UNIFORME (1 couleur, 100 % faux). Trois bugs :
  **(1) désync phase/référence** — pas pixel `phases[n % N]` vs réf `phases[m %
  N]` divergent après tout rebase `m := 0` avec `n % N ≠ 0` ; fix = N orbites
  (réf p itère `phases[(p+i) % N]`, `compute_reference_orbit_phase` →
  `ReferenceOrbit.hybrid_phase_refs`, récursion prof. 1) + tracking de phase
  dans les boucles f64/exp (invariant `(phase+m) ≡ n (mod N)`, rebase ⇒
  `phase := (phase+m) % N ; m := 0`, rebase-at-end inconditionnel — exit
  wrap_periodic/ref_exhausted→GMP, tous deux faux pour un hybride).
  **(2) chemins z²+c hardcodés pris par les hybrides** (`fractal_type ==
  Mandelbrot`) : fast-path dd de l'orbite (~1e13–1e19, réf FAUSSE), atom-domain
  (`dZdC=2·Z·dz+1`), Brent, compresseur fantôme, `harmonic_candidate` → tous
  gatés `!is_hybrid`. **(3) cache d'orbite sans discriminant `hybrid_phases`**
  (réutilisation [M,BS]→[M] même centre, GUI). **Méthodo ground truth** : le
  f64-std n'est PAS un juge valable sur bord hirsute (chaos : 4→85 % de
  désaccord entre 300 et 2000 iters) ; le SEUL juge = GMP par-pixel CYCLANT
  (`GmpInterpState`). Vérifié : [M,M] grille 160/160 EXACT vs GMP ; [M,BS]
  95 exact + 6 off1 / 160, résidu adjugé PLANCHER BRUIT f64 des plis abs
  (diagnostics : vérité 256 b == 512 b sur 160/160 → vérité convergée ; 69/88
  points faux CHANGENT de réponse avec une réf décalée → bruit, pas
  systématique ; classe hirsute BS-famille, cf. G3). Verrous :
  `multi_phase_perturbation_matches_gmp_per_pixel` (grille GMP-cyclant) + 2
  diagnostics `--ignored` (truth-stability, ref-sensitivity) + verrous [M,M]
  jalons 3-4 inchangés.
- [x] **✅ Jalon 5b — BLA PAR PHASE (perf hybrides deep) `[2026-07-17]`** : port
  F3 `hybrid_blas` (`bla.cc` via `blasR2calc(Z[phase], opss, …, phase)`).
  `BlaTableUnified::build_cycled(refs[p], formula, p, …)` — le single-step à
  l'index i utilise `phases[(p+i) % N]` (même séquence que l'itération de
  `refs[p]`), merges phase-agnostiques (`build_with` factorisé, mono-phase
  bit-identique). Les boucles multi-phase f64/exp prennent `tables:
  &[BlaTableUnified]` (BLA active ssi `tables.len() == n_phases` ; anciens
  callers `&[]` = pas directs) : saut `tables[phase].lookup{,_fexp}(m, δ²)`,
  n et m avancent du même l → invariant `(phase+m) ≡ n` préservé SANS màj de
  phase (F3 same) ; PAS de rebase-check post-saut (mirror single-phase — un
  rebase mid-chaîne casserait la rampe géométrique des skips). DEUX chemins
  z²+c supplémentaires gatés `!is_hybrid` : **table série Taylor** (coefficients
  faux pour un hybride genuine ; son heuristique AUTO-ADJUST misfirait sur
  [M,M] — skip 32 % → iteration_max ×4 + re-rendus parasites) et rien d'autre.
  **Perf : le prewarm couvre le multi-phase** (`prewarm_bla_entry` passait par
  `compile_formula` + gate `phases==1` → l'entrée hybride, N tables × orbite
  pleine, se construisait sous le lock global workers parqués = ~TOUT le temps
  pixel). Mesuré [M,M] e50 96²/263k iters : **total 3.34 → 0.50 s (6.7×),
  pixels 3.15 → 0.30 s** ; e1000 (exp) pixels 0.049 s ≈ [M] single 0.042 s.
  Verrous : [M,M]==[M] e1000 pixel-exact AVEC BLA ; grille GMP-cyclant en
  config production (tables cyclées) inchangée ([M,M] 160/160) ; déterminisme
  run-to-run 0 px. Résidu [M,M] vs [M] à e50 : 5-6 px épars = troncatures de
  réf différentes ([M] atom-tronquée, refs hybrides pleines — l'atom-domain
  z²+c est gaté) + epsilon BLA, plancher bruit (résolu au jalon 5d).
- [x] **✅ Jalon 5c — éditeur GUI de séquence hybride `[2026-07-17]`** : menu
  Type → « Hybride (séquence) » (➕/⌫/🗑/Appliquer ≥2 phases/Désactiver),
  label « Hybride: M⊕BS », `apply_hybrid_sequence()` (reset convention
  Mandelbrot + `hybrid_phases` + `start_render`), drag-and-drop PNG restaure.
- [x] **✅ Jalon 5d — atom-domain GÉNÉRIQUE hybrides (mat2) `[2026-07-17]`** :
  port F3 `hybrid_reference` (`hybrid.cc:81-98` — dZdC **mat2** bas-précision,
  critère `|inv(radius·dZdC)·Z| < 1`). Récurrence par phase entière via les
  Jacobiens dual-numbers de la BLA : `J' = A_i·J + I` (`build_bla_single_step` ;
  B = I car `Op::Add` — seul op impliquant c — est TERMINAL dans toutes nos
  phases, cf. `from_single` b=IDENTITY), J en 4×FloatExp (croît ~λⁱ), critère
  sans inversion `|adj(J)·Z|² < r²·det(J)²`. Pour J conforme ([M,M]) ≡ critère
  complexe — vérifié NUMÉRIQUEMENT : les réfs [M,M] e50 tronquent au MÊME index
  que [M] single (86615/192865) ⟹ **[M,M] e50 == [M] à 0 px** (le résidu 6 px
  du jalon 5b venait du mismatch de troncature). Perf e50 96² : 0.50 → 0.38 s
  ([M] single 0.135 s — reste 2× orbite + loop générique sans fast-path inline).
  Verrou : `hybrid_mm_equals_mandelbrot_deep_f64_e50` (render-level, couvre
  d'un coup 5a réfs/tracking + 5b BLA cyclées/prewarm + 5d atom mat2) ; grille
  GMP-cyclant et e1000 inchangés. **Genuine deep validé end-to-end** :
  [M,BS] @ 1e30 (coordonnée zoom-hunt HP, 567 couleurs, diagnostic `--ignored`
  `multi_phase_deep_e30_genuine_diagnostic`) — réfs de phase tronquées à des
  index DIFFÉRENTS (533/496, attendu : séquences de phases différentes),
  grille vs GMP-cyclant 27/160 exact à 30k iters = **plancher chaos hirsute**
  (PROUVÉ hors de cause pour l'atom : résultat IDENTIQUE 27/160 avec
  `FRACTALL_ATOM_PERIOD=0`, cohérent avec la croissance chaos 4→85 % entre
  300 et 2000 iters mesurée à 3e10).
- [x] **✅ Jalon 5e — parité hybride vs F3 NATIF (juge externe) `[2026-07-17]`** :
  le format light-toml porte `phases = "mandelbrot,burning_ship"` (loader
  `main.rs::load_toml_params`, CLI `--phases` prioritaire ; `compare_f3.py`
  émet les blocs `[[formula]]\nopcodes = "…"` F3 par phase, mapping aligné
  `bytecode/compile.rs` ↔ `param.cc op_string`). Cas corpus (frontières LISSES
  — les zones speckle du set hybride BS-famille sont à sensibilité extrême,
  ~55 % de désaccord inside cross-engine par chaos, STRUCTURE macro identique,
  même classe que G3) : `hybrid_mbs_smooth_e8` (f64-std, Δmean 0.0026 = fraction
  NF, inside_mismatch 0) et **`hybrid_mbs_smooth_e13` (PERTURBATION multi-phase,
  pixel 3.4e-15 : Δmean = Δmax = 0.0000, inside 0 — PIXEL-IDENTIQUE à F3)**.
  Le stack G4 entier (réfs par phase + BLA cyclée + rebasing + atom mat2) a son
  juge externe. Coordonnées par zoom-hunt smoothness-guided (voisinage 5×5 le
  moins varié). Les cas rejoignent le corpus full automatiquement (tier quick =
  liste fixe, inchangée).
- [x] **✅ Jalon 5f — nucleus PHASE-AWARE `[2026-07-17]`** : port F3
  `hybrid_period`/`hybrid_center`/`hybrid_size` pour une `Formula` arbitraire.
  Cœur : **interpréteur GMP dual-mat2** (`nucleus.rs::GmpDualMat2` — valeur +
  Jacobien ∂(zx,zy)/∂seed en mat2 de GMP Floats, chain rule par opcode :
  Sqr/Mul/Rot = M(w)·J, AbsX/AbsY = négation de ligne conditionnelle, Add =
  +I en mode d/dC). Trois étages : `find_period_atom_domain_formula` (critère
  mat2 `|adj(J)·z|² < s²·det(J)²`, full-GMP comme la variante z²+c — F3 fait
  la détection en perturbation floatexp, plus rapide mais dépendante des réfs ;
  à porter si la perf bloque), `newton_refine_center_formula` (Newton 2D,
  solve J·Δ=−z), `hybrid_size_mat2_formula` (b += inv(J), degré = moyenne géo
  des degrés de phase ; ⚠️ récurrence **d/dC avec +I à l'Add** — la lettre F3
  met dC=0 mais notre variante z²+c validée corpus P1.6.b inclut le +I, et
  [M,M] doit lui être identique — vérifié : ratio size 0.258 avec d/dZ pur,
  1.0 exact avec d/dC). Routage `orbit.rs` : hybride → variantes formule,
  single-phase → z²+c historique (bit-identique). Vérifié : **[M,M] ==
  z²+c** (période/centre/size/K, unit) ; **genuine [M,BS]** : satellite période
  123 trouvé, Newton convergé 12 pas vers un point PÉRIODIQUE (|z_123| ≈ 0
  vérifié indépendamment), K **non-conforme** (K01=0.680 ≠ −K10=−0.362 — la
  raison d'être du mat2) ; render-level : `[M,M] --find-nucleus` @minibrot
  1e18 = période 445, K/size identiques, image PIXEL-IDENTIQUE à `[M]`.
  **✅ Jalon 5g `[2026-07-18]` — fast-path Mandelbrot inline multi-phase** :
  dans les boucles multi-phase f64 + exp, les phases [Sqr, Add] steppent
  inline `δ' = 2·Z·δ + δ² + dc` (bitmask par formule, mêmes opérandes/ordre
  que DeltaState{,Exp} → bit-identique) au lieu de l'interpréteur ; + cache
  de la réf de phase courante (slice/len/atom rechargés au rebase seulement),
  cache tête-de-boucle chaîné (mirror single-phase) et `node.z_land` dans la
  garde anti-over-skip (bit-copie de l'orbite de phase, plus d'accès tableau).
  Mesuré e50 [M,M] 256² : boucle pixel **2.26 → 0.98 s (2.3×)**, total
  2.43 → 1.12 s (vs [M] 0.78 s — 2× orbite inhérent aux N réfs) ; e1000 exp
  ~4 % (FloatExp domine). **Pixel-identique** old-vs-new : [M,M] e50, [M,M]
  e1000, [M,BS] e13, [M] e50 single (0 px) ; verrous e50/e1000/[M,M]@3e10 +
  grille GMP-cyclant + 25 goldens verts, harness quick 0 gap.
- [x] **✅ Jalon 4 — hybrides DEEP-EXP (ComplexExp, > 1e280) `[2026-07-17]`** :
  `iterate_pixel_unified_exp_multi_phase` (`pixel_loop_exp.rs`) — mirror du
  multi-phase f64 en `DeltaStateExp` (FloatExp survit à l'underflow f64 du delta),
  SANS BLA, cyclant `phases[n % len]` + rebasing F3 en FloatExp. `iterate_pixel_
  unified_exp` dispatche multi→ce loop (au lieu d'`assert!(phases==1)`) ;
  `delta.rs` route le multi-phase exp (`use_exp_path`) au lieu de `return None`.
  `select_algorithm` : gate `!wants_exp` SUPPRIMÉ — le tier f64/exp est choisi
  par pixel dans `try_bytecode_unified_path`. **Vérifié : [M,M] @ zoom 1e1000
  PIXEL-EXACT == Mandelbrot exp** (56 couleurs, escapes réels ; verrou render-level
  `hybrid_mm_equals_mandelbrot_deep_exp_e1000`, garde-fou anti-tuile-uniforme).
  Single-phase INCHANGÉ (fast path Mandelbrot exp non touché ; 269 unit + golden
  verts). **Reste jalon 5 : BLA par phase (perf) + nucleus phase-aware + éditeur GUI.**
- [x] **✅ BLA multi-phase native** — fait au **jalon 5b** (`build_cycled`, une
  table par phase, lookup `tables[phase]`).
- [x] **✅ Nucleus finder phase-aware** — fait au **jalon 5f** (`GmpDualMat2` +
  variantes `*_formula`, couvre toute formule bytecode donc BS/Tricorn/…
  en phase d'un hybride ; les types single-phase non-Mandelbrot restent
  sans `--find-nucleus` : le flag est gaté Mandelbrot, à ouvrir si demandé).
  - [x] **✅ BLA radius scaling σ₁(K) `[2026-07-17]`** (ex-P1.6.b-bis, débloqué
    par le K non-conforme du jalon 5f) : `FractalParams::transform_sigma1()`
    (plus grande valeur singulière de `transform_matrix()`, snap exact à 1.0
    pour K conforme) × le rayon `c_norm` BLA — centralisé `delta.rs::
    bla_c_norm` (prewarm + build f64/dd + `c_norm_fexp` exp). Sans lui, un K
    skewé (σ₁≈1.43 sur le blob [M,BS] p=304, K_norm=[0.230,1.238;0.676,-0.707])
    étire la grille δc → `max |δc|` sous-estimé → rayons de merge
    sur-permissifs → over-skip. **No-op bit-identique** hors K skewé (σ₁=1
    exact ; 275 unit + 24 goldens verts, quality 15/15). Verrous :
    `transform_sigma1_*` (types.rs) + `bla_c_norm_scales_with_sigma1_of_skewed_k`.
    Smoke end-to-end : [M,BS] `--find-nucleus` @1e15 → nucleus p=304, K
    non-conforme, rendu perturbation multi-phase propre (0 glitch/fallback).
    Non fait (rien ne le motive mesurablement) : validité anisotrope
    `|K⁻¹δ| < r` — σ₁ est la borne conservative, F3 ne fait ni l'un ni l'autre.
- [x] **✅ CLI/GUI** — fait aux **jalons 2 (CLI `--phases`) et 5c (éditeur GUI)**.
- [x] **✅ `Op::Rot` per-phase CÂBLÉ `[2026-07-18]`** : parseur
  `parse_opcodes_formula` (compile.rs, format F3 `param.cc::parse_opcodess` —
  mots `add sqr mul store absx absy negx negy rot{DEG}`, chaque `add` termine
  une phase, `need_store` mirroré, cos/sin **f32** parité F3 vérifiée
  bit-identique vs g++). Nouveau champ `params.hybrid_opcodes`
  (PNG-roundtrip), PRIORITAIRE sur `hybrid_phases` dans `formula_for_params` ;
  prédicat UNIQUE `FractalParams::is_hybrid_formula()` posé sur TOUS les gates
  z²+c (wisdom, GPU, série, dd, atom, Brent, harmonic, cache orbite +
  discriminant `hybrid_opcodes`). Entrées : CLI `--opcodes "sqr rot{30} add"`,
  TOML blocs `[[formula]]` F3 natifs (structurés abs/neg/power + `opcodes`),
  clé légère top-level `opcodes` (consommée par compare_f3.py, qui émet un
  bloc [[formula]] par phase). Rien à faire dans le moteur : Op::Rot était
  déjà couvert par interp f64/GMP, delta-form f64/exp, BLA dual, nucleus
  GmpDualMat2. Verrous : parseur (5 unit) + grille GMP-cyclant
  `opcodes_rot_perturbation_matches_gmp_per_pixel` (rot30 single **160/160
  exact** + rot30⊕BS genuine **160/160 exact**, centres zoom-huntés à vraie
  structure, garde anti-vacuité par spread — qui a révélé et durci le contrôle
  [M,M] historique, passé à 2000 iters) + juge F3 NATIF
  `hybrid_rot_smooth_e10/e13` (f64-std + perturbation : inside_mismatch=0,
  Δmean 0.033/0.053 = bruit F3 tier float, adjugé : grille GMP exacte sur le
  MÊME centre, coefficients rot bit-identiques). `[[formula]] rotate` (champ
  transform F3) reste distinct : c'est la rotation de vue, déjà `--rotation`.

### G5 — Architecture & nettoyage · `[P1 · maintenabilité]`

**Done when** :
- [x] **Chemin de rendu CPU unifié CLI ↔ GUI** (2026-05-20) — un seul dispatcher
  `render_escape_time_cancellable_with_reuse`. Cf. *Shipped* + invariant
  §« Ne pas régresser ».
- [x] **Dispatch GPU unifié** (2026-05-21) : `GpuRenderer::render_dispatch()`
  (`gpu/mod.rs`, renvoie `GpuDispatchResult`) est le point d'entrée unique
  partagé par le CLI (`main.rs`) et le thread GUI — calcul de `use_perturbation`
  + match par type + threading du cache d'orbite + fallback CPU via `None`. Fin
  de la duplication (~60 lignes). Vérifié : GPU standard/perturbation/fallback
  Tricorn OK, 178 unit + golden verts.
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

- [x] **✅ Durcissement couverture >512² `[2026-07-17]`** — le path de correction
  perturbation ne tourne qu'à `max_dim > 512` (`!small_image`) mais n'était
  JAMAIS testé au-dessus de 256² (le bug réf-intérieure y a vécu invisible).
  **Audit complet des gates `small_image`** :
  - `mod.rs:1417/1421` (flag `suspect` en grande image) : **no-op pour le
    bytecode** (`suspect` toujours `false` sur ce path) → sain.
  - `mod.rs:1479` (neighbor pass CPU) : gaté `!bytecode_path` → sauté → sain.
  - `mod.rs:1514/1634` (secondary-refs CPU) : gatés `!bytecode_path` (le 1634
    corrigé cette session) → sains.
  - `orbit.rs:1231` (build série) : path LEGACY seul (le bytecode lit la BLA,
    pas la série) → hors périmètre bytecode.
  - **`gpu/mod.rs:1031` (neighbor pass GPU) : PAS gaté sur le path F3 — mais
    VÉRIFIÉ BÉNIN** : GPU seahorse-1e8 @640² == GMP **pixel-exact (0 diff)** ;
    le neighbor pass + correction GMP AMÉLIORE la précision (corrige les px
    plancher-f64), il ne les casse pas. Laissé tel quel.
  → **Aucun nouveau bug latent >512² ; les paths sont robustes.** Verrou :
  golden `mandelbrot_interior_ref_640` (640×438, SEUL cas >512², exerce
  l'escalade dd, pixel-exact GMP à la génération) — rougit si le gate
  `!bytecode_path` ou l'escalade dd régresse. ⚠️ Coût ~14 s (escalade dd) —
  assumé pour combler le trou. NB : une passe systématique quality @>512² vs
  GMP est impraticable (GMP par-pixel ~13 min/preset à 640²) → le golden cible
  la classe de bug connue.

**Done when** :
- [x] **Golden : verrouiller les fixes deep-zoom** (2026-05-21) — ajout de
  `mandelbrot_e50` (1e50, rebase-at-end G2), `mandelbrot_e1000` (1e1000),
  `mandelbrot_cusp_m075` (cusp -0.75, fix max_perturb G3), `mandelbrot_floral`
  (1.55e85, fix period-detection G3). Rendus == chemin par défaut == GMP, revus
  visuellement, verts en CI (déjà câblée). 4 nouveaux goldens.
- [x] **CI : étendre le corpus golden** à zooms intermédiaires (2026-07-12) :
  `mandelbrot_e10` (centre minibrot, **PASS pixel-exact vs GMP** max_diff=0,
  dendrites), `mandelbrot_e15` + `mandelbrot_e20` (centre e113 180-digits,
  spirale/étoile structurées, bruit de bord dispersé p99=0 vs GMP — comme
  seahorse/e17). Comblent le trou golden entre 1e8 (minibrot) et 1e50+
  (escape-time). Path perturbation par défaut `bytecode_f64`. 21 goldens (était
  18), revus visuellement, déterministes (re-run vert). Rendu <50 ms/cas.
- [x] **Golden hybride genuine deep + nucleus phase-aware** (2026-07-17) :
  `hybrid_mbs_nucleus_5e28` ([M,BS] `--find-nucleus` @5e28, satellite p=304)
  — verrouille find_nucleus_formula + K non-conforme + BLA multi-phase
  σ₁(K) + réfs par phase + atom mat2 en un cas (~0.12 s, revu, déterministe).
  Le log `[NUCLEUS]` imprime désormais le centre raffiné (50 chiffres,
  reproductibilité F3-style). 25 goldens. NB : `FRACTALL_UPDATE_GOLDENS`
  réécrit les 22 anciens PNG (metadata `use_dd_tier: None→false`, pixels
  vérifiés 22/22 identiques) — churn non commité, à purger au prochain
  update légitime.
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

### G8 — Harness d'auto-amélioration · `[P0 · méta — égaler puis dépasser F3]`

> Protocole : `HARNESS.md`. Outil : `scripts/harness.py`. Itération : skill
> `/improve`. Mémoire : `SCORECARD.md` + `harness/history/*.json` (versionnés).
> Trois axes : vitesse (geomean ratio vs F3), parité (compare_f3), qualité
> (pert vs GMP + goldens). Priorités : correction > robustesse > vitesse >
> qualité. « Dépasser » = geomean ≤ 1.0 + correctness_wins ≥ 2 + 0 régression
> superset.

**Done when** :
- [x] **Protocole écrit** (`HARNESS.md`) + skill `/improve` (2026-07-03).
- [x] **`scripts/harness.py`** opérationnel (2026-07-12) : score/baseline/gaps,
  tiers quick/standard/full, JSON (`harness/history/*.json`) + SCORECARD.md +
  gaps triés — vérifié bout-en-bout sur le run standard (25 cas, 4 axes).
- [x] **✅ `fractall-quality` émet du JSON** (`suite-summary.json` toujours écrit
  par la commande suite, `report.json` par preset — `quality/report.rs`) ;
  `harness.py::parse_quality_json` le consomme avec fallback markdown.
- [x] **Binaire F3 Linux** buildé (`fraktaler-3-3.1/fraktaler-3-3.1.linux`) —
  la machine courante (i7-10700F 16 threads Linux) remplace le M4 ; utilisé par
  les axes speed/parity.
  - [x] **Build reproductible (`scripts/build_f3_linux.sh`, 2026-07-13)** : le
    `.linux` est gitignoré/par-machine → sur un nouveau container (ex. Xeon 4c CI)
    les axes speed/parity retombent dark faute de binaire. Le script rebuild un F3
    **batch GUI-free** (raw-EXR N0/NF → `batch()`, pas de SDL/GL/imgui — submodules
    absents) : deps apt + shim 3 symboles gui.cc/colour.cc + 15 TUs cœur. Rallume
    speed/parity sur toute machine Linux. Premier score Xeon 4c : geomean **0.282**
    (10/10 wins), parité 10/10 ok, quality 11 PASS, goldens 🟢
    (`20260713T093029Z`). Hint `compare_f3.py` + doc HARNESS.md alignés.
- [x] **Baseline v1 committée** sur cette machine (2026-07-12) : **tier standard**
  (256², runs=3 médiane, 25 cas) figée depuis `20260712T120028Z-cd05f6a.json`.
  **geomean vitesse 0.223** (25/25 wins, pire test5 0.580), parité 25/25 ok,
  quality 8 PASS / 3 WARN (seahorse/e13/e17 bruit de bord) / 0 FAIL, goldens
  verts. Remplace l'ancienne baseline quick/runs=1. Machine-spécifique (i7).
- [x] **Boucle exécutée au moins 1×** de bout en bout (`/improve`) avec
  amélioration mesurée committée + verrou posé (2026-07-03, `53a55cc` :
  latence progress reporter → geomean vitesse 1.712→0.966 ; rebaseline
  `a8bf871`).

**🧭 G8.2 — Après F3 : évolution des références de la boucle (survey 2026-07-14).**
Constat en 3 points : (a) **vitesse** — fractall bat F3 partout (quick 10/10,
standard 25/25, full **80/80 — sweep 2026-07-14 post-escalade : geomean 0.193,
pire cas e22522 0.943, plus AUCUN perdant** ; seul e52465 quarantainé, adjugé
plancher-GMP) : l'axe head-to-head F3 ne tire plus vers le haut ;
(b) **correction** — F3 n'est PAS ground truth aux zooms modérés (wisdom défaut
= tier float 24 b, cf. diag 3-voies G3 2026-07-13 : 9391 px faux vs 16 chez
fractall) — le vrai juge reste notre GMP per-pixel ; (c) **algorithmique** —
F3 3.1 (2025-06) reste la dernière stable, KF2+ est arrêté (jamais de BLA), mais
la frontière algo est chez **Imagina** (Zhuoran) et **FractalShark**. Paysage
vérifié (2026-07-14) :
- **Imagina** (AGPL-3, Windows, réécriture `ImaginaFractal/ImaginaCore` WIP) —
  réputé le plus rapide CPU. Surtout : **`ImaginaFractal/Algorithms`** (GPL-3)
  = implémentations de référence de **HarmonicLLA / HarmonicMLA / MipLA /
  PTWithCompression** (compression d'orbite de référence) — la génération
  d'après-BLA. À étudier pour les prochains goals perf (candidats : LA
  harmonique vs notre BLA mat2 ; compression de réf pour la mémoire deep).
- **FractalShark** (GPL-3, actif — release 0.532 du 2026-07-05, Mandelbrot
  SEULEMENT) : `FractalSharkCli` batch + port **Linux expérimental** (CMake/
  Clang) ; orbite de référence **GPU par NTT** (~10× vs CPU MPIR à précision
  extrême), 2 implémentations LA CUDA, compression de réf à la Zhuoran, custom
  float 2×32+exp GPU. **Cette machine a une RTX 4060 Ti 16 GB (driver 595)**
  → exploitable, MAIS `nvcc` absent : le build exige `sudo apt install
  cuda-toolkit` (action utilisateur, pas de sudo autonome).
Actions candidates pour la boucle (ordre suggéré) :
- [x] **✅ Arbitre 3-voies intégré au harness (2026-07-14)** :
  `harness.py adjudicate <stem>` (via `three_way_gmp.py --case`) rend
  fractall/F3/GMP et persiste le verdict (`fractall_wrong`/`f3_wrong`/
  `shared`/`both_match_gmp`) dans `harness/adjudications.json` (versionné) ;
  `compute_gaps` annote les gaps parité avec le verdict (un `f3_wrong` est
  déclassé sév. 2→4). Premiers adjugés : test5 `both_match_gmp` ; **rug
  `shared` MAIS fractall big=156 vs F3 big=3852 (max 28181) à 64² — fractall
  ~25× plus fidèle à GMP que F3 même sur notre seul déficit connu.**
- [x] **✅ Axe fuzz livré (2026-07-14)** : `scripts/fuzz_scenes.py` (scènes
  frontière stables déterministes — bump additif 2⁻⁴⁶, filtre axes de pliage,
  6 familles, zoom 1e5-1e11) + `axis_fuzz` dans harness.py (quick=3/standard=6/
  full=8 sondes 96², seed committée `FUZZ_SEED_DEFAULT=20260714`, cache
  bench/harness/fuzz/, section scorecard + gaps sév. 1 sur FAIL). Premier run :
  2 PASS + 1 WARN (mandelbrot 6e7, 75 px épars p99=0 — la classe plancher-f64
  visée, détectée du premier coup). Rotation de seed au rebaseline (HARNESS.md).
- [x] **✅ FractalSharkCli buildé + calibré (2026-07-14)** — CUDA 13.3 installé
  (user), build **gcc** OK (upstream = clang ; 1 patch local au clone :
  `#include <x86intrin.h>` pour `__rdtsc`, gcc-only ; script fallback gcc).
  Binaire : `~/src/FractalShark/build-release/FractalSharkCli/FractalSharkCli`.
  **Calibration** : `--zoom` = même sémantique que nous (span = 4/zoom,
  `PointZoomBBConverter.h Factor=2`) ; **axe y inversé** (math vs écran) —
  sans impact vitesse, flip vertical pour comparer des images. GPU (RTX 4060
  Ti) opérationnel : seahorse 1e8 256² CPU 2.5 s / GPU 1.7 s (≈1 s d'overhead
  init CUDA constant — comparer à grande résolution ou soustraire).
  - ⚠️ **Limites du port Linux (dev preview) — resserrées après duel réel
    (2026-07-14)** : à 1e50 le CPU sort UNIFORME (263k iters) et le GPU rend
    un **AUTRE lieu** (spirale ≠ notre étoile golden — précision de réf
    insuffisante, « structuré » ≠ « correct ») ; abort vers 1e100. Lieu
    vérifié correct à **1e30** (== fractall). Et le path CPU Linux est
    ~50× plus lent que fractall (62 s vs 1.25 s, e50-coords 1e30 1024²
    131k iters) → seul le **GPU** sert de barre.
  - **📊 Duel valide (1e30, 1024², 131k iters, 3 runs)** : fractall CPU
    **1.25 s** · FS CPU 62 s · **FS GPU (RTX 4060 Ti) ~0.6 s** après warm-up
    (~1 s d'init CUDA au 1er run). ⇒ La barre FS = GPU ~**2× notre CPU 16t**
    sur le mid-deep. Notre GPU wgpu (bytecode f32) s'arrête à ~1e7 → ouvre le
    goal « **GPU perturbation deep** » (kernel delta HDR wgpu), cible
    mesurable : 0.6 s. Câblage harness (colonne speed FS-GPU ≤1e30) : ROI
    faible tant que le range fiable est si étroit — suivre les releases
    upstream (le Windows gère e1000+), re-tester à chaque 0.5x.
- [x] **✅ Étude `ImaginaFractal/Algorithms` (2026-07-14)** — analyse complète
  dans **`docs/imagina-algorithms-analysis.md`** (pendant de
  fraktaler-3-analysis.md ; clone ~/src/Imagina-Algorithms, 5 évaluateurs,
  1169 lignes). Acquis : notre rebasing/boucle pixel = le canon de Zhuoran
  (aucune glitch-detection → architecture validée) ; MipLA ≈ notre BLA (on
  fait même mieux sur la mémoire via skip-levels) ; leur arrêt de réf =
  atom-domain inconditionnel ×2 (nous : gaté par plage). Deux techniques à
  porter, en ordre :
  - [~] **Port PTWithCompression — PHASE 1 LIVRÉE (2026-07-14) : structs +
    instrumentation, GO pour la phase 2.** `perturbation/compress.rs` :
    `ReferenceCompressor`/`ReferenceDecompressor` port fidèle d'Imagina
    (fantôme f64, cheb·2⁻³², waypoint au décrochage, reset() = rebase), 4 unit
    locks (roundtrip intérieur/chaotique, trap quasi-nucleus, replay reset).
    Hook `FRACTALL_COMPRESS_REF_STATS=1` dans le build d'orbite (Mandelbrot
    seed=0) → ligne `[COMPRESS]`. **Densités mesurées (corpus)** : dragon
    2.05M iters → 1 918 wp (**1067×**, 98 Mo→46 Ko) ; e22522 213 k → 769
    (**278×** — le piège |Z|≈0 sous-f64 est GÉRÉ : |Z[end]|=0.0 exact snappé,
    cf. unit lock) ; e50 116× ; glitch_test_6 47× ; glitch_test_2 39× ;
    wfs_mb 542 k → 38 721 (**14×**, le plus dense — orbite frôlant des zéros).
    ⚠️ Piège découvert en test : un orbite de VALIDATION calculée en f64 est
    bit-identique au fantôme → 0 waypoint ; toute validation doit sourcer
    l'orbite en GMP (comme la prod).
    **✅ PHASE 2 LIVRÉE (2026-07-14) — swap de stockage env-gated
    `FRACTALL_COMPRESS_REF=1`, path f64 Mandelbrot.** Trait `RefF64Source`
    (reset/advance/teleport/end_value/wrap) monomorphisé : `SliceSource`
    (défaut, **bit-identique à HEAD vérifié octet-à-octet** sur
    dragon/e50/glitch_test_2 + goldens) vs `CompressedSource` (décompresseur).
    Pont BLA : `BlaMultiStep.z_land` (+16 o/nœud) — la garde anti-over-skip lit
    z_land SANS toucher l'état, `teleport` (=`seek` depuis valeur exacte)
    seulement si saut accepté. Libération `z_ref_f64`+`z_ref` post-prewarm BLA
    via `Arc::try_unwrap` (refcount 1 : CLI/QA ; partagé GUI : sautée, path
    compressé tourne quand même) ; cache strippé jamais réutilisé
    (`is_valid_for`→false). Routage `delta.rs::compressed_ref_route_active`
    (Mandelbrot, tier F64, cycle_period=0, pas de distance/interior/traps) ;
    clé du cache BLA = identité des waypoints (stable au strip).
    **Mesuré (dragon 5M iters 256²)** : `path=bytecode_f64_compressed`,
    **98.3 Mo libérés** (orbite → 1 917 wp = 46 Ko), total +~1-3 % (pixels
    +20 % — replay ; le gain visé est la mémoire steady-state). opus2 25.9M
    iters : 1.24 GB → 0.58 Mo (1067×). Gate ON vs OFF : 2-244 px/65536
    divergents épars (replay ≤2⁻³², e13 vs GMP : métriques IDENTIQUES au
    défaut). 218 unit (3 nouveaux : seek mi-segment ≤ tol vs GMP, seek(0)==
    reset, seal idempotent) + goldens 🟢.
    **✅ VALIDATION CORPUS (2026-07-14, 83 cas 256², gate ON vs OFF + arbitre
    GMP)** : 0 crash, 0 timeout nouveau ; 38 cas routés compressés (tier F64
    jusqu'à ~1e280 — e50/e113/dragon/e227/lethal_weapon inclus), 45 correctement
    NON routés (exp/dd/features). Diffs ON-vs-OFF : majorité ≤ 300 px épars
    (classe replay ≤2⁻³²) ; outliers concentrés sur la classe « frontière
    précision extrême » DÉJÀ décorrélée de GMP au path par défaut :
    **lethal_weapon 37 607 px (57 %) MAIS arbitre GMP 8×8 : div_ratio
    IDENTIQUE 0.78125 aux deux gates** (OFF max_diff 21k / ON 33k — deux
    échantillons chaotiques équivalents, limite f64-δ documentée, PAS un défaut
    de compression) ; **e227 idem : div_ratio 0.10938 identique** (ON même
    nominalement plus proche : max_diff 1292 vs 2938). golden_spider/
    verstoppertje/evolution_trees/uranium = même classe (deep 1M+ iters) ;
    rug = classe glitch réf-unique connue.
    **Reste (phase 3, si besoin)** : compression `z_ref` ComplexExp (tier exp —
    fantôme FloatExp au voisinage des zéros sous-f64, wfs_mb) ; build streaming
    (pic RSS, aujourd'hui build-then-drop) ; **défaut ON** — critère atteint sur
    l'axe correction (jamais pire que le défaut vs GMP), bloqué par : regen des
    21 goldens + revue visuelle + décision utilisateur (les rendus par défaut
    bougent de quelques px) ; wrap_periodic Brent (waypoint forcé à
    cycle_start) si jamais réactivé par défaut.
  - [x] **Prototype Harmonic LA — CONCLU (2026-07-14) : jalons MLA + LLA
    livrés, verdict final = BLA mat2 reste le défaut, harmonic opt-in
    gagnant seulement sur orbites courtes.**
    `bytecode/harmonic_mla.rs` (~640 l.) : port fidèle d'Imagina HarmonicMLA —
    `LaStep` (Step/Composite chebyshev, ValidRadiusScale 2⁻²⁴), build
    étages harmoniques (période par chute des minima 2⁻⁴, coupe à la moyenne
    géométrique, plafond à la période, sentinelles), évaluateur par descente
    (1 check/segment, premier pas quadratique EXACT `dz·(Z+z)`), queue directe
    aux sémantiques fractall (escape 625, rebase F3, rebase-at-end atom).
    Écarts sûreté : cap `i+length ≤ iter_max`, garde anti-over-skip à
    l'atterrissage, plafond 64 étages. Routage delta.rs (`harmonic_route_active`,
    exclusif de COMPRESS_REF), label `bytecode_f64_harmonic_mla`.
    **A/B 256², phase pixels (3 runs)** : glitch_test_2 (period0=3, orbite
    1143) **0.021→0.007 s = 3×** 🎯 ; e113 −15 % ; **e50 +44 % ✗, dragon
    +47 % ✗** (orbites 86 k–2 M : rayons de validité MLA étroits → descente
    retombe en queue directe). Arbitre GMP e13 gate ON : WARN max_diff=221
    p99=0 div_ratio=0.00031 (≈ défaut 210/0.00012, même classe). Diff ON-vs-OFF
    0.03–4.3 % px (approximation LA ≠ BLA). 221 unit (3 verrous : period0=3
    détecté, éval == BLA 100/100 dc à mismatch 0) + goldens 🟢 gate OFF
    bit-identique.
    - [x] **✅ Jalon LLA LIVRÉ (2026-07-14)** — segmentation aux dips de rayon
      (`HarmonicLLA.cpp`, chute > 2⁻¹⁰/pas) portée dans le même module
      (LaStep + évaluateur communs, `new2`/`detect_dip`/flags dip) ; gate
      `FRACTALL_HARMONIC_LA=1|lla` → LLA, `mla` → MLA (A/B), label
      `bytecode_f64_harmonic_{lla,mla}`.
      **Hypothèse « les dips corrigent e50/dragon » INFIRMÉE** : A/B
      entrelacé 256² (phase pixels, médiane) LLA ≈ MLA — vs BLA :
      glitch_test_2 0.021→0.009 s (**2.3×** ✓), e113 0.023→0.027 (+17 %),
      e50 0.084→0.103 (+23 %), dragon 0.180→0.286 (+59 %). La segmentation
      n'est PAS le levier : la phase LA gère déjà les wraps en interne
      (rebase d'étage `j=begin`) et la couverture LA est déjà élevée
      (compteurs e50 : 92 segments appliqués/px pour 172k iters/px ≈
      ~1.9k iters/segment composite) — le déficit vs BLA vient de la
      granularité (frontières de segments fixes vs mip skip-anywhere).
      + **Écart fractall #4 — ré-ascension GARDÉE après rebase de la queue
      directe** (absent d'Imagina : leur queue ne ré-accélère jamais) :
      l'état post-rebase (`m=0`, `z=δ`) = état d'entrée de la phase LA →
      `continue 'render` vers l'étage sommet, SEULEMENT si le dernier
      passage LA a été productif (≥ 1 segment ; anti-ping-pong). A/B
      politiques (jamais/toujours/gardée, entrelacé ×3) : e50 **−24 %**
      (0.135→0.103), dragon/glitch_test_2 neutres, e113 +8 % ; gardée ≥
      toujours partout. Correction gate ON RENFORCÉE : e13 vs GMP pur
      **PASS max_diff=0** (jalon MLA : WARN 221). 224 unit (3 verrous LLA :
      period0=3 par premier dip, éval ≥ 99 % vs BLA, next_stage_la_index
      dans les bornes de l'étage précédent) + quality 15/15 PASS +
      goldens 🟢 (gate OFF bit-identique — path défaut non touché).
    - [ ] Prochain levier harmonic SI repris un jour : granularité de fin de
      segment (un pixel qui échoue au rayon d'un segment long retombe en
      direct pour TOUT le reste du segment — un mini-étage en pas 2^k
      pourrait combler) ; généralisation Mat2 (non-conforme) + tier exp
      seulement si un cas gagnant net apparaît.
  - [x] **Micro-A/B — ADJUGÉ (2026-07-15)** : (a) chaînage des sauts BLA =
    **déjà livré** (`2365da9`, cache tête-de-boucle −4.5 % phase pixels : les
    sauts s'enchaînent via `continue` sans rebase check intermédiaire, lookup
    démarrant au niveau aligné max via `trailing_zeros`) ; (b) chebyshev vs
    norm_sqr sur la validité BLA = **no-go par analyse** : à check égal la
    norme est déjà cachée (aucun gain de vitesse propre) ; mixer cheb|δ| avec
    nos rayons euclidiens (dérivation F3 blaR2) ÉLARGIT l'acceptation hors de
    la borne d'erreur (cheb ≤ euclid) = improvisation contre F3-source-de-
    vérité ; le faire proprement = re-dériver tout le build mat2 en métrique
    chebyshev (normes d'opérateur incluses) — pas un micro-A/B, ROI attendu
    faible (pixel loop memory-bound, cf. G2).
- [x] **✅ Cross-check ground truth indépendant (2026-07-14)** :
  `scripts/independent_probe.py` — mpmath pur-Python (zéro GMP/MPFR/rug),
  réplique exactement mapping pixel→c + boucle escape des 7 familles QA, lit
  les params dans le PNG (chunk fractall-params) et arbitre les top-divergents
  d'un report.json. Validé : e13 4/4 top-divergents → mpmath 512 b == juge GMP
  (pert diverge comme documenté) — notre juge est confirmé sans common-mode.

---

### G9 — Moteur multi-techniques orchestré par le wisdom · `[P0 · BUT FINAL — directive utilisateur 2026-07-15]`

**Vision** : le moteur dispose de PLUSIEURS techniques de rendu (tiers
numériques f64/exp/dd/GMP ; accélérations BLA mat2 / Harmonic LA / compression
d'orbite ; devices **CPU et GPU**) et le **wisdom choisit par rendu la
technique la plus rapide parmi les VIABLES** — viabilité = exposant + mantisse
requis + fiabilité pixel (détecteur shadowing), vitesse = mesurée sur LA
machine (benchmarks persistés, modèle F3 `wisdom.cc` complet). Le GPU est un
first-class citizen : sélectionné automatiquement quand il est plus rapide,
pas un flag `--gpu` opt-in.

État des briques (tout existe en pièces détachées — le goal = l'orchestration) :
- ✅ `wisdom::number_tier` (F64/Exp/Dd) + `WisdomPlan` inspectable — mais
  statique (seuils), ne couvre ni device ni variantes d'accélération.
- ✅ Dispatcher CPU unique + `GpuRenderer::render_dispatch` — mais la décision
  GPU est côté caller (`--gpu`), et le GPU s'arrête à ~1e7 (f32).
- ✅ Techniques env-gated prêtes à router : compression (`FRACTALL_COMPRESS_REF`,
  corpus-validée), Harmonic LA (`FRACTALL_HARMONIC_LA`, gagnant 2.3× sur
  orbites courtes — critère de routage MESURABLE : `period0 ≪ orbit_len`,
  connu au build de table), dd-BLA (epsilon calibré 2⁻⁸⁰).
- 📐 Détecteur de fiabilité proposé (mathr m-reliable, cf. G-dd) = la brique
  « viabilité mantisse » qui manque au wisdom.

Jalons (chacun ≈ 1-2 itérations /improve, ordre suggéré) :
- [x] **9.1 — Wisdom = planificateur UNIQUE** `[✅ 2026-07-15]` : `WisdomPlan`
  étendu à `{device (Cpu|Gpu), algorithme, tier, variantes (compression,
  harmonic)}`. `wisdom::select_algorithm(params, device)` est LA source de la
  sélection d'algorithme, consommée par le dispatcher CPU (`render/escape_
  time.rs`), `render_dispatch` GPU (seuil pert f32 ~1e5 via `Device::Gpu`),
  et les 3 sites GUI exacts-équivalents (`effective_cpu_mode` + 2×
  `will_use_perturbation`). `wisdom::variants(params)` porte la partie
  STATIQUE des prédicats de routage compression/harmonic ; `delta.rs::
  {compressed_ref,harmonic}_route_active` la composent avec les conditions
  orbite (plus de duplication env-gate). Comportement strictement préservé
  (227 unit + goldens + QA 15/15 PASS + quick 0 gap) ; `[WISDOM]` logue
  désormais `device=… variants=…`. Reste hors périmètre 9.1 : `effective_
  algorithm_mode` GUI (forme délibérée sans GMP, stats display).
- [x] **9.2 — Benchmarks machine persistés (F3 wisdom.cc-style)**
  `[✅ 2026-07-15]` : `fractall-cli --wisdom-bench` (explicite, jamais
  implicite au premier rendu — ~20 s) mesure le **débit effectif** (Σ iters /
  wall, skips BLA/harmonic + orbite + rayon inclus) de chaque technique sur
  des rendus RÉELS via le dispatcher unique : `cpu_std_f64` (vue défaut),
  `cpu_perturb_{f64,exp,dd}` (frames corpus e50/e318 embarquées, tests
  verrouillent frame→technique), `gpu_std_f32` (MÊME vue que cpu_std_f64 —
  comparabilité device directe). Persisté `~/.config/fractall/wisdom.toml`
  (override `FRACTALL_WISDOM_FILE`), module `fractal/wisdom_bench.rs`.
  Consommé par `WisdomPlan.bench_iters_per_sec` (ligne `[WISDOM] bench=`,
  fallback `-` sans fichier). Mesures i7-10700F : cpu_std 1.8e9, perturb_f64
  1.04e11, exp 1.8e10, dd 2.2e8, **gpu_f32 5.0e9 = 2.7× cpu_std** → la
  donnée d'arbitrage 9.5 existe. Raffinement possible à 9.5 : croître les
  itérations (pas seulement la taille) pour atteindre la durée cible sur les
  frames std (0.15-0.22 s au cap de taille).
- [x] **9.3 — Routage harmonic par le wisdom** `[✅ 2026-07-15]` : mode
  `FRACTALL_HARMONIC_LA` tri-état (unset/`auto` → **Auto, nouveau défaut** ;
  `1|lla|mla` → forcé ; `0|off|bla` → kill switch). Décision Auto au build de
  l'entrée cache BLA : probe `detect_period0` (scan premier-dip O(period0),
  réplique exacte de l'ouverture du build LLA, verrouillée par test) +
  politique `wisdom::route_harmonic_auto(period0)` = route si `1 ≤ period0 ≤
  100`. **Calibration corpus 256² (A/B pixels ×3)** : le discriminant est
  period0 SEUL, PAS le ratio orbit/period0 (e50 ratio 773 PERD +34 %) ni la
  longueur d'orbite (super_dense p9 orbite 695 k GAGNE 1.74×, −63 s). Gagnants
  p7-78 : flake 5.9×, gt5 5.8×, test3 5.7×, gt3 5.3×, mitosis 3.7×, gt1 3.7×,
  mitosis2 3×, gt2 2.2×, peanuts/all_seeing_eye 1.8×, leaded_glass 1.6×,
  dinosaur_fossils (p78) 1.2× ; perdants p112+ : e50 +34 %, e113 +13 %,
  dragon +59 % ; seuil 100 = milieu de la zone morte mesurée [79, 111].
  Per-pixel : routage sur PRÉSENCE de la table dans l'entrée (le probe ne
  tourne jamais par pixel) ; per-render : candidat `wisdom::harmonic_candidate`
  (le prédicat compressed reste per-pixel-safe via `compression_active`
  gate-first). Adjudication GMP pur (glitch_test_2 96²) : harmonic max_diff=11
  div=0.00033 vs BLA max_diff=38 div=0.00043 — **le path routé est plus proche
  de la vérité que BLA**. Verrous : golden `mandelbrot_glitch_test_2_harmonic`
  (routage auto + évaluateur LA) ; golden `_atom` épinglé
  `FRACTALL_HARMONIC_LA=0` (préserve le verrou du guard BLA lands_on_ref_end ;
  runner golden étendu aux env par cas) ; tests seuils politique + probe==build.
- [~] **9.4 — GPU perturbation deep (kernel delta HDR wgsl)** — **étape 1
  livrée 2026-07-15 : kernel F3-strict f64 natif, parité CPU-grade** :
  réécriture de `perturbation.wgsl` en port fidèle de `pixel_loop.rs::
  iterate_pixel_unified_mandelbrot` (n/m séparés — le legacy remettait n:=0 à
  chaque rebase, faussant les comptes ; rebasing strict `|Z+δ|²<|δ|²` sans
  hystérésis ; garde anti-over-skip BLA ; suppression glitch Pauldelbrot +
  correction GMP-CPU qui masquait l'imprécision f32 ; suppression du clamp
  `iter_max=min(iter_max, ref_len-1)` côté host — gate `ref_len-1 ≥ iter_max`
  sinon fallback CPU). Précision **f64 natif (`SHADER_F64`)**, buffers zref/
  BLA en f64. Résultat gpu-suite (juge GMP) : **WARN p99=0 div 0.0007-0.0015
  escape_disagree=0 sur tout le range perturbation 1e4→1e8** = le niveau exact
  du CPU-perturbation vs GMP (0.001). Sans SHADER_F64 (Metal) → fallback CPU.
  - **☠️ IMPASSE df64 (2×f32 double-float) sur la stack WGSL→naga→SPIR-V** :
    sans décorations NoContraction/precise, `fma(a,b,-(a·b))` peut être évalué
    non-fusionné (→0) et les EFT two_sum sont réassociées (`(a+b)-a → b`) —
    mesuré sur NVIDIA RTX 4060 Ti ET llvmpipe : le df64 s'effondre en f32 en
    contexte de boucle (div 0.067 = simulation f32 pure) alors que chaque
    primitive isolée passe. Garde runtime `×u(=1)` : défait en boucle aussi.
    Diagnostic reproductible : `cargo run --release --bin df64_gpu_probe`
    (sonde exactitude two_sum/fma/split + boucle perturbation par adaptateur).
  - **Étape 2 livrée 2026-07-15 (G9.4b) : réfs tronquées + range 1e8 → ~4e37** :
    le kernel gère la fin de référence comme le CPU (`pixel_loop.rs` étape 3) —
    wrap périodique (`cycle_start/cycle_period` Brent's en uniform), rebase-at-end
    atom-domain (`hybrid.cc:301`) + guard BLA `lands_on_ref_end` en WGSL ; le
    gate host ne retombe en CPU que pour les réfs tronquées par ESCAPE
    (per-pixel GMP requis, hors GPU). Range borné par le transport span/offset
    en paires f32 hi/lo : gate `GPU_SPAN_F32_MIN = 1e-37` (zoom ≲ 4e37, HP spans
    couverts par l'underflow f64 → 0) ; au-delà = tier HDR (exposant par pixel).
    Verrous gpu-suite : `gpu-mandelbrot-e13` (WARN max_diff=2 div 3e-5) +
    `gpu-mandelbrot-e30-truncated-ref` (WARN p99=0 div 7.6e-4) ; range 1e4→1e8
    inchangé (div 0.0007-0.0015).
  - **(a) perf MESURÉE 2026-07-15** : e30 1024² 131k iters (avg 25k/px) —
    kernel GPU 3.85 s vs CPU f64 16t 1.86 s → **GPU ~2× plus lent** (RTX 4060
    Ti, f64 1:64, comme anticipé). L'auto-GPU (9.5) ne doit PAS router le deep
    sur GeForce par vitesse ; la voie perf reste 2×f32 via passthrough SPIR-V
    décoré NoContraction, ou kernel exp-par-pixel f32.
  - **📌 Découverte correction — ✅ ÉLUCIDÉE + fix f64 livré (2026-07-15)** :
    sur mandelbrot-e30 (scène ultra-sensible) le CPU bytecode_f64 FAIL div 0.034
    vs GMP là où le kernel GPU f64 + BLA conforme tient div 3e-4. **Cause
    isolée** (pas la « conformité » ni le merge) : l'**epsilon de validité BLA**.
    F3 (`engine.cc:283`) hardcode `e = 1/2²⁴` avec un FIXME (« not enough in all
    circumstances ») parce que ses coefficients BLA sont en mantisse `float`
    (24 b). Nos tables mat2 sont en **f64 (53 b)** mais copiaient tel quel le
    2⁻²⁴ (`bla_threshold`), d'où une erreur ~2⁻²⁵ par saut (terme δ² droppé,
    borné par r=ε·|Z|) qui sature sur les scènes sensibles. **Fix** :
    `F64_BLA_EPSILON = 2⁻⁵³` pour le tier f64 (`delta.rs::build_bla_entry`),
    miroir fidèle de la convention F3 e=2^-mantissa_bits. Preuve (`fractall-
    quality compare`, GMP pur) : e30 f64 div 0.036→**3e-4**, e15/e20 WARN
    (max_diff 349/415) → **PASS pixel-exact GMP (max_diff=0)**, e50 f64 FAIL
    div 0.051 p99=73 → WARN div 0.0013 p99=0. Ces vues rendaient **silencieusement
    faux en prod Auto** (path f64, dd = opt-in). Coût : BLA moins agressive →
    e50 256² 0.21→0.83 s (~4×), e113 0.05→0.11 s (~2×), geomean quick 0.192→0.282
    (fractall gagne toujours 10/10 vs F3) — compromis correction>vitesse assumé.
    Verrous : goldens e10/e15/e20/e50/deep_e113/glitch_test_2_atom/cusp_m075
    régénérés (corrections, revus visuellement). 233 unit + 21 golden + QA 15/15
    PASS. **Reste** : (1) ~~porter le même fix au tier **exp**~~ **❌ RÉFUTÉ
    (2026-07-15, /improve)** : mesuré, le tier exp (FloatExp, `bla_exp`) N'a PAS
    l'over-skip epsilon et le passer à 2⁻⁵³ est une **pure régression vitesse**.
    Preuves : liiiines (zoom 3.16e321, seule scène exp GMP-tractable à 12.5 k
    iters, `path=bytecode_exp` atom-tronqué) donne div 0.00415 **identique** de
    ε=2⁻²⁴ à 2⁻⁵³ **à 1e-300** (BLA quasi-off) — l'over-skip n'apparaît qu'à
    ε=1.0 (div 0.33) ; golden exp e1000 **pixel-identique** aux deux ε. Mais le
    coût vitesse est réel : e401 256² 0.75→0.95 s (**+30 %**), e1000 0.66→0.73 s
    (**+10 %**) pour ZÉRO gain de correction. **Pourquoi exp diffère de f64** :
    aux zooms exp la réf est **tronquée atom-domain + cyclée** (période courte),
    la BLA n'atteint donc pas les longs sauts qui stressent le rayon de validité
    (contrairement aux spirales f64 e30/e50 à réf longue). Le résidu WARN de
    liiiines (div 0.004, p99=1) = **plancher de la mantisse FloatExp (53 b)**,
    PAS l'epsilon → ne se corrige qu'avec un tier exp à mantisse plus large
    (softfloat/float128-exp façon F3, feature majeure hors périmètre). ⇒ laisser
    `bla_exp` à `bla_threshold`=2⁻²⁴ (correct ET plus rapide). (2) **Récupérer la
    vitesse f64** via BLA **second ordre** (terme C·δ²,
    exactement ce que fait le kernel GPU conforme, `bla.rs::c_new`) : garde le
    rayon 2⁻²⁴ large ET la précision — mais exige des dérivées secondes
    (hyper-dual) dans la mat2 unifiée. (3) ✅ **FAIT (2026-07-15)** : réévalué les presets `use_dd_tier`. **misiurewicz-m32
    (1e12) et mandelbrot-e18-minibrot (1e18) sont pixel-exacts en f64 pur**
    depuis le fix epsilon (max_diff=0 vs GMP à 96²/128²/160²) → RETIRÉS de la
    liste dd (`quality/mod.rs`) : la QA vérifie désormais le path f64 que la
    prod utilise (dd = opt-in) = verrou anti-régression. **seahorse-valley
    (1e8) et les spirales e30/e50/e100 restent dd** : leur edge/spirale
    ultra-sensible garde un résidu f64 réel (seahorse f64 WARN div 0.0018 ;
    e30 f64 WARN div 3e-4) que seul le dd rend pixel-exact — vrai plancher de
    sensibilité, pas un artefact BLA.
- [x] **9.5 — Auto-GPU** `[✅ FONCTIONNELLEMENT COMPLET 2026-07-17 — device
  auto-arbitré CLI+GUI par benchmark + garde-fou correction ; overrides
  --gpu/--no-gpu (CLI) + menu CPU/GPU (GUI)]` :
  le plan choisit le device par benchmark 9.2 + viabilité ; `--gpu`/`--no-gpu`
  deviennent des overrides. Verrou : `fractall-quality gpu-compare/gpu-suite`
  (presets `GPU_PRESETS` seahorse 1e2→1e8, rapports `quality-reports/gpu/`) +
  tests quality en CI. **⚠️ Juge = GMP PUR depuis 2026-07-15** : le juge
  initial (CPU Auto = f64-std à ces zooms) divergeait LUI-MÊME de ~6 % de la
  vérité à 5000 iters sur bord chaotique (CPU-std vs GMP div 0.06 quand
  CPU-perturbation vs GMP div 0.001 — la perturbation ancrée sur référence
  GMP est plus juste que l'itération f64 directe) et aurait pénalisé un
  kernel GPU plus correct que lui. État verdicts (kernel f64 9.4) : WARN
  1e4→1e8 (= niveau CPU), FAIL 1e2-1e3 (shaders std f32, mantisse 24 b, p99
  215-578, leçon F3 : float 24 b = 9391 px faux) ⇒ le wisdom ne doit PAS
  router le GPU en dessous du seuil perturbation (pixel ≥ 1e-5) tant que les
  shaders std restent f32 — options : les passer f64 (SHADER_F64, coût DP),
  ou router la perturbation f64 aussi en shallow (δ grand + rebasing, à
  mesurer), ou tier std f64→GPU interdit. Arbitrage device par vitesse benchée
  (9.2) une fois la perf 9.4 mesurée.
  - [x] **✅ Étape 1 — logique d'arbitrage device (`wisdom::select_device`)
    `[2026-07-17]`** : fonction PURE (non encore consommée par le dispatch =
    zéro régression rendu) qui choisit CPU/GPU par arbitrage de débit benché
    SOUS contrainte de correction. **Garde-fou f32** : le GPU n'est routé QUE
    quand il rendrait en perturbation (kernel f64) ; en dessous du seuil
    (`select_algorithm(_, Gpu) != Perturbation` → std f32, faux) → **jamais
    GPU**. Plage transport f32 hi/lo bornée (`pixel_size ≥ 1e-37`). `arbitrate_
    device(gpu_bench, cpu_bench)` : GPU seulement s'il gagne d'au moins
    `GPU_SPEED_MARGIN=1.25`, CPU si l'un des deux n'est pas benché (conservateur).
    Clé bench `GpuPerturbF64` ajoutée (`wisdom_bench`, `for_plan(Gpu,
    Perturbation)`) ; **pas encore mesurée** → sur cette machine l'arbitrage
    retombe sur CPU (correct : GPU grand public f64 1:64 = plus lent ; GPU f32 =
    faux). 6 unit tests (no-gpu / shallow-f32-gate / deep-extreme / no-bench /
    marge). 263 unit + 21 golden verts.
  - [x] **✅ Étape 2 (a) — mesure GPU-perturb dans `--wisdom-bench` `[2026-07-17]`** :
    `run_bench` benche désormais **les deux** clés GPU avec la MÊME closure
    `render_dispatch` (auto std/perturb par zoom) — `gpu_std_f32` (vue shallow)
    ET `gpu_perturb_f64` (vue e30, DANS la plage kernel ≲ 4e37 ; e50 serait
    hors-plage → None). `measure_gpu` renvoie None si le GPU ne rend pas la vue
    (Metal sans SHADER_F64 → pas d'entrée → arbitrage CPU). **Mesuré RTX 4060 Ti**
    : `gpu_perturb_f64 = 2.13e9` iters/s vs `cpu_perturb_f64 = 1.62e10` (**GPU
    ~7.6× plus LENT**, f64 1:64) → `select_device` route **CPU** sur données
    RÉELLES (robuste au bruit de bench : GPU perd même au cpu-bas). `gpu_std_f32`
    (4.48e9, rapide) correctement IGNORÉ (garde-fou correction). Tests
    `for_plan(Gpu,Perturbation)` + frame GPU-perturb (perturbation ∧ pixel≥1e-37).
    266 unit + 21 golden verts.
  - [x] **✅ Étape 2 (b) — câblage CLI `[2026-07-17]`** : `main.rs` calcule
    `use_gpu` = override `--gpu`/**nouveau `--no-gpu`**, sinon **auto**
    `select_device(params, true) == Gpu`. **Raffinement crucial** : l'arbitrage
    ne compare QUE si le CPU aussi rendrait en perturbation (apples-to-apples) —
    là où le CPU utilise std-f64 (zoom ~1e5–1e12) on reste CPU (comparer std-f64
    vs GPU-perturb serait invalide ; le CPU-std y est déjà rapide+correct). Donc
    l'auto-GPU se limite à la plage deep both-perturbation (1e12…4e37). **Sur
    cette machine : auto ≡ CPU partout** (GPU-perturb 7.6× plus lent) → vérifié
    `auto == --no-gpu` BIT-À-BIT ; `--gpu` force toujours le GPU. **Déterminisme
    cross-machine** : les goldens forcent `--no-gpu` (sur un GPU f64 rapide une
    frame deep-perturb pourrait router GPU ≠ CPU bit-à-bit). Le tier quality
    passe par le dispatcher CPU direct (non concerné) ; le harness mesure l'auto
    (= CPU ici). 263 unit + 21 golden + harness quick (0 gap) verts.
  - [x] **✅ Étape 2 (c) — câblage GUI `[2026-07-17]`** : le menu « Tech: » a déjà
    la structure idéale — **🔄 Auto** (algorithm_mode=Auto) vs sous-menus
    explicites CPU/GPU. Wiring : quand `algorithm_mode == Auto`, le device est
    AUSSI auto (`select_device(params, gpu_disponible)`, même décision que le
    CLI) ; une sélection explicite CPU/GPU reste un override manuel
    (`self.use_gpu`). Sur cette machine : Auto → CPU (défaut GUI inchangé) ; sur
    un GPU f64 rapide, une frame deep-perturbation en mode Auto basculerait GPU.
    272 gui + 263 cli + 21 golden verts. **G9.5 FONCTIONNELLEMENT COMPLET** :
    le device est auto-arbitré (CLI + GUI) par benchmark machine + garde-fou
    correction, `--gpu`/`--no-gpu` (CLI) et le menu CPU/GPU (GUI) = overrides.
- [ ] **9.6 — Fiabilité → escalade de tier** (= G-dd auto-dispatch, plan déjà
  écrit) : détecteur shadowing en observation puis escalade px→dd /
  frame→dd ; ferme la boucle « le wisdom ne sous-provisionne jamais »
  (contrairement au wisdom F3, cf. diag 3-voies : F3 float 24 b = 9391 px faux).
- [ ] **9.7 — Le wisdom choisit aussi le TYPE de l'ORBITE référence**
  `[⏸ faible ROI — documenté, pas prioritaire]` : aujourd'hui l'orbite est
  **toujours en GMP**, clampée à **≥128 b** (`compute_perturbation_precision_
  bits`, `mod.rs:718`) ; le tier `dd` (~106 b) ne sert que la boucle PIXEL, pas
  l'orbite. La hiérarchie `wisdom` de F3 (double→long double→float128→floatexp→
  softfloat) sélectionne aussi le type de calcul de l'orbite : dans la fenêtre
  **mid-deep où la précision orbite requise est 53–113 b** (~zoom 1e13→1e30), un
  `long double`/`float128`/`dd` **natif** éviterait l'overhead GMP-128. Étend le
  BUT FINAL multi-technique (le wisdom orchestre TOUTES les techniques, orbite
  comprise). **⚠️ ROI mesuré faible, raisons (analyse /improve 2026-07-15)** :
  (1) **pas un déficit** — fractall bat DÉJÀ F3 sur tout le corpus (geomean quick
  0.298, full 10/10 wins) *avec* GMP-always, alors que F3 utilise ce tier ;
  (2) l'orbite domine rarement dans cette fenêtre (à ~50 k iters, GMP-128 ≈
  25 ms, la boucle pixel domine) — le gain n'est sensible que si l'orbite est
  LONGUE (millions d'iters) ET la précision tombe dans [53,113] b, tranche
  étroite ; (3) **ZÉRO effet sur les ultra-deep** type e52465 (174 325 b requis
  ≫ 113 b float128 → orbite intrinsèquement GMP/arbitraire, cf. slow-safe G8).
  Bencher d'abord (tier orbite dd vs GMP-128 sur un cas mid-deep à orbite longue)
  pour chiffrer le gain réel AVANT d'investir. À faire APRÈS 9.5/9.6 (plus fort
  impact).

**Critère d'excellence G9** : sur le corpus + presets QA, le plan choisi par
le wisdom n'est JAMAIS battu > 10 % par un plan alternatif (vitesse), et
JAMAIS moins correct que le tier au-dessus (correction) — mesurable par un
axe harness « wisdom-optimality » (sweep des plans sur un échantillon).

- [~] **Axe `wisdom-optimality` LIVRÉ (2026-07-15, /improve)** :
  `harness.py wisdom-optimality [--cases --size --runs]` chronométre le plan
  CHOISI par le wisdom (`auto`) contre les plans forcés alternatifs sur la
  dimension de routage active (harmonic vs BLA, `FRACTALL_HARMONIC_LA`), sur un
  échantillon 2-régimes (période courte→LLA gagne / longue→BLA gagne). Verdicts :
  **PASS** (dans 10 %), **FAIL** (plus lent À SORTIE IDENTIQUE = vrai gap),
  **ADJUDICATE** (plus lent mais l'alternative rend une AUTRE image → tradeoff
  correction possiblement voulu, vérifier vs GMP avant de recalibrer), avec
  plancher de bruit 60 ms (rendus <200 ms non fiables sur machine chargée).
  CPU-only, sans effet render. Rapport `bench/harness/wisdom-opt/`. Couvre la
  moitié VITESSE du critère sur la dimension harmonic ; restent (a) les
  dimensions tier (dd/exp, = 9.6) et device (= 9.5), (b) la moitié CORRECTION
  (jamais moins correct que le tier au-dessus).
  - **✅ FINDING ADJUGÉ vs GMP pur (2026-07-16, /improve) — wisdom (BLA) a
    RAISON, seuil `period0≤100` PRÉSERVÉ**. Sur les cas long-période deep, la LLA
    forcée est bien plus rapide post fix-epsilon (e50 3.48×, e113 1.81×,
    dragon 1.14× via l'axe) mais **faster-but-WRONG**. Adjudication 96² vs GMP
    pur (`fractall-quality compare`, `FRACTALL_HARMONIC_LA=lla|off`) :
    - **e50** : BLA **pixel-exact** (max_diff=0, p99=0, div_ratio=0) ;
      LLA **FAIL** (max_diff=418, p99=53, div_ratio=**0.036** = 3.6 % de pixels
      systématiquement faux, PAS du bruit de bord).
    - **e113** : BLA WARN div_ratio 0.00022 (max 10) ; LLA WARN div_ratio
      0.00119 (max 17), ~5× plus divergent — même sens.
    - **dragon** (3.7e191, 5M iters) : GMP intractable, mais period0=3164 ≫ 100
      et la tendance e50/e113 généralise → non routé.
    **Conclusion** : le fix-epsilon a inversé le classement de VITESSE mais pas
    la frontière de CORRECTION. Router LLA au-delà de 100 échangerait de la
    vitesse contre de la correction → refusé (critère G9). L'axe émet ADJUDICATE
    (pas FAIL) : c'est le comportement voulu (tradeoff correction assumé).
    **Verrou** : doc `HARMONIC_AUTO_PERIOD0_MAX` réancré sur la correction +
    test `route_harmonic_auto_calibrated_thresholds` (commentaire GMP) +
    preset quality `mandelbrot-e50` (BLA≡GMP, déjà PASS). Seuil robuste aux
    futurs changements de coût BLA (n'est plus ancré sur le débit relatif).

### G10 — Rendu GUI temps-réel : plan XaoS · `[P1 · navigation fluide — étude 2026-07-16]`

> **Vision** : la navigation (pan/zoom) doit être *fluide* (ressenti 30-60 fps),
> pas une reconstruction complète à chaque mouvement. Aujourd'hui chaque
> changement de géométrie jette l'orbite référence (`app.rs:1601/1635/1656`
> `orbit_cache=None`) ET recalcule tous les pixels depuis zéro ; **aucune
> réutilisation inter-frame**. Référence : la technique d'**approximation
> dynamique de XaoS** (réutilisation séparable colonnes/lignes) — que ni nous ni
> F3 n'implémentons. F3 réutilise seulement l'orbite/BLA (`render.cc:238
> reference_can_be_reused`), pas les pixels.

**Fait habilitant (vérifié)** : `colorize_to_rgb` (`io/png.rs:23`) ne consomme
que `iterations[idx]` (échappement) + `zs[idx]` (z à l'échappement), **absolus
et indépendants de l'orbite référence** → un pixel calculé à la coordonnée `c`
reste réutilisable pour le même `c` à la frame suivante, même en perturbation.
Le matching doit se faire en **espace pixel relatif** (transformée frame→frame
`x_old=a·x+b`, séparable si rotation=0), exact à toute profondeur sans HP dans
la boucle.

**Garde-fous** : fast-path gated sur `rotation=0 ∧ transform_k=None` (sinon
fallback rendu complet — la séparabilité casse en rotation) ; réutilisation
désactivée pour les modes à données par-pixel (`Distance*`, `OrbitTraps`,
`Wings` — même exclusion que `escape_time.rs:68 build_reuse`).

Jalons (ordre de ROI croissant en effort) :
- [x] **G10.1 — Warp GPU de la dernière frame** `[✅ 2026-07-16, c8edf51 +
  fix signe Y ae611c8]` :
  pendant qu'un rendu calcule, afficher la texture précédente **transformée par
  le pan/zoom courant** (quad texturé egui, offset/scale UV dérivés de
  `view_texture` vs `view_live` en HP). Donne le *ressenti* de fluidité XaoS
  immédiatement, sans toucher au compute. Aucune régression moteur (pur
  affichage). Snapshot de vue posé au chargement de texture.
- [x] **G10.2 — Réutilisation orbite référence inter-frame** `[✅ 2026-07-16]` :
  réutilisation OFF-CENTER prouvée correcte **par construction** — la référence
  cachée rend une nouvelle vue **contenue dans son empreinte** (`|Δcentre| +
  span_new/2 ≤ span_old/2`, `ReferenceOrbitCache::can_subset_reuse`) car chaque
  pixel a un `dc` que la référence a DÉJÀ rendu correctement (sous-ensemble).
  Couvre zoom-in + pan contenu (le cas interactif) ; zoom-out/grand pan → rebuild.
  La boucle calcule déjà `c = cref + dc` (`pixel_loop.rs:97`) ; il a suffi
  d'ajouter l'offset `(centre_vue − cref)` à la grille dc FloatExp séparable
  (`mod.rs`, nul quand centrée → **goldens pixel-exacts inchangés**). **Opt-in
  CPU-only** : `compute_reference_orbit_cached(..., allow_subset_reuse)` — seul le
  chemin CPU FloatExp passe `true` ; **GPU/dd/legacy/tests passent `false`**
  (exact-center, aucune régression). Exclusions : nucleus, rotation, dd.
  GUI : `zoom_at_point`/`zoom_to_rectangle` ne jettent plus `orbit_cache` (le
  moteur décide). Verrou : `subset_reuse_offcenter_matches_fresh_and_reuses`
  (rendu réutilisé == rendu frais à ≤1 % px + preuve que la réf n'a pas été
  rebuild). 235 unit + 21 golden + QA 15 PASS + quick 0 gap. Socle de G10.4.
- [x] **G10.3 — Recolorisation sans clone 74 Mo** `[✅ 2026-07-16]` : les 4
  buffers bruts (`iterations/zs/distances/orbits`) sont passés en `Arc<Vec<…>>`.
  La recolorisation (changement palette/color_repeat) clone désormais un **Arc
  (bump de refcount, ~gratuit)** au lieu de ~74 Mo (@1080p) memcpy **sur le thread
  UI** → plus de jank pendant un drag du slider palette. `Arc::make_mut` (COW) pour
  la normalisation de taille d'`orbits`. Deref coercion → tous les sites de lecture
  (`&[T]`) inchangés. GUI-only (moteur intact) : 244 gui + 235 cli tests verts.
- [x] **G10.4 — Réutilisation pixels XaoS (colonnes/lignes)** `[✅ 2026-07-16]` :
  `fractal/xaos.rs` — matching séparable en espace pixel relatif : transformée
  `x_old = a·(x+0.5)+B` par axe, dérivée de deux ratios O(1) calculés UNE fois
  en HP (`Δcentre/span`, `span_new/span_old`, même math que le warp G10.1) —
  exact à toute profondeur, aucun HP dans la boucle. **Anti-dérive** : chaque
  frame stocke `col_err`/`row_err` (position VRAIE des données par axe) ; le
  matching compare la grille nominale aux positions vraies (`k + err[k]`),
  tolérance 0.5 px → l'erreur ne s'accumule JAMAIS sur les pans enchaînés
  (vérité préservée à travers les copies, cf. test
  `chained_fractional_pans_do_not_accumulate_error`). Compatibilité vérifiée
  par **fingerprint JSON** des params non-géométriques (robuste aux champs
  futurs) + gates rotation=0/transform_k=None/find_nucleus=false/modes
  per-pixel/AA. Consommé par les 4 boucles pixel CPU (f64/GMP/perturbation/
  perturbation-GMP) via un param `xaos: Option<&XaosMap>` du dispatcher unique
  (CLI/quality/HQ/AA/Julia-preview passent `None` → goldens pixel-exacts
  inchangés). GUI : frame source stockée à chaque PassComplete **CPU** (les
  passes GPU f32 sont exclues), buffers en `Arc` bout-en-bout (PassComplete +
  TextureReadyMessage, supprime aussi le clone `previous_pass` par passe) ;
  **raffinement idle** : si la passe finale a copié des pixels, re-rendu exact
  silencieux (passe unique, XaoS off, label `≈XaoS` effacé) après 400 ms sans
  interaction. Mesure : pan 8 px @1024×768/20k iters seahorse → **98.8 % de
  pixels copiés, ×42.6 wall-clock** (17.5 s → 0.41 s), map build 108 µs
  (diagnostic `xaos_pan_speedup_diagnostic`, `--ignored`). Verrous : 11 unit
  xaos dont roundtrips pan-entier == rendu frais pixel-exact sur les paths f64
  ET perturbation ; 246+255+253 unit + 21 golden + QA 15 verts.
- [x] **G10.4b — XaoS zoom : injectivité + refine union + molette** `[✅ 2026-07-16]` :
  le zoom-in dupliquait des colonnes source jusqu'à couvrir 100 % de la cible
  (zoom ×2 aligné = « écho pur » : AUCUN pixel frais, image exacte seulement
  après idle 400 ms + refine total → clic-zoom plus lent et plus flou
  qu'avant G10.4). Fixes : (1) **matching injectif** par axe (une source → au
  plus une cible, la mieux alignée) — ≥ (1−a)·n colonnes fraîches garanties
  en zoom-in, no-op pan/zoom-out (pan ×39 préservé) ; (2) **refine union**
  (`build_refine_map`, `XaosMap::keep_union`) : conserve tout pixel dont un
  axe est ENTIÈREMENT exact (`col_exact`/`row_exact` dans la frame — piège :
  une ligne copiée « alignée » n'est PAS exacte si l'axe colonne est décalé,
  verrou `union_refine_rejects_aligned_but_shifted_frame`) et ne recalcule
  que les approximations → cycle zoom ×2 écho+refine = **107 %** d'un rendu
  frais, image visible à ~0.8× (mesure `xaos_zoom_cycle_diagnostic` : refine
  ×3.6, 75 % conservés) ; (3) refine/label `≈XaoS` gated sur l'erreur réelle
  (> ε) — plus de refine après copies exactes ; plus de label parasite sur
  les passes GPU (map non consommé) ; (4) une passe écho-pur ne remplace plus
  la frame source (dégradait la source en copies de copies pendant les zooms
  enchaînés qui annulent les rendus) ; (5) **zoom molette ancré au curseur**
  (`zoom_anchored_hp`, C′ = C + (r−0.5)·S·(1−1/f), HP ; ≈×1.2/cran) —
  documenté dans CLAUDE.md mais jamais implémenté ; c'est le chemin de zoom
  continu que l'écho XaoS sert le mieux. Boucles pixel unifiées sur
  `XaosMap::source_index`. **Suite (même jour)** : (6) invariant **écho XaoS ⊃
  pas de reuse basse-résolution** (dispatcher + perturbation) — le reuse copie
  des centres décalés de (ratio−1)/2 px qui contaminaient les colonnes/lignes
  déclarées FRAÎCHES du map (le refine union leur fait confiance → l'image
  « exacte » gardait des pixels jusqu'à 1.5 px à côté) ; verrou poison
  `echo_pass_ignores_coarse_pass_reuse` (sans map, le reuse reste actif) ;
  (7) les passes intermédiaires écho-pur sont SAUTÉES (contenu ⊂ warp G10.1
  affiché, en plus flou — supprime le pompage preview→full à chaque cran de
  molette, le travail frais démarre plus tôt) ; la passe finale tourne
  toujours. Verrous : 18 unit xaos dont roundtrip
  `zoom_then_exact_refine_matches_fresh_render` (écho ×2 → refine union ==
  frais pixel-exact) ; 252+261+259 unit + 21 golden verts.
- [x] **G10.5 — File de tuiles priorité-centre** `[✅ 2026-07-16]` :
  `render/tiles.rs` — les 4 boucles pixel CPU (f64, GMP, perturbation,
  perturbation-GMP) passent du `par_chunks_mut` monolithique à une work-queue
  de tuiles (16/32/64 px, ≥ 8 tuiles/thread) ordonnée par distance au point de
  priorité (curseur GUI via `hover_norm`, centre sinon). Deux propriétés que
  rayon seul ne donne pas : (1) **ordre d'exécution réel** — file atomique
  `fetch_add` sur l'ordre trié (le split binaire rayon disperse les fronts sur
  toute l'image) ; (2) **streaming intra-passe** — `TileOpts { priority, sink }`
  threadé dans le dispatcher unique, le sink GUI colorise chaque tuile, la
  blitte sur la passe précédente colorisée (base upscalée) et envoie un
  `RenderMessage::TileProgress` throttlé 100 ms → la zone sous le curseur
  devient nette EN PREMIER pendant la passe (zoom molette ancré = la zone
  d'intérêt d'abord, à chaque cran). Sûreté sans `unsafe` : buffers découpés
  en amont en segments de lignes disjoints par tuile (`TileGrid::split`,
  chaîne `split_at_mut`) distribués par `Mutex<Option<S>>` pris une fois.
  CLI/quality/HQ/AA/refine passent `None` (priorité centre, pas de sink) —
  l'ordre ne change pas les valeurs : verrous
  `tiled_render_identical_across_priorities` (bit-exact coin vs centre vs
  None), `tile_sink_covers_every_pixel_once_with_final_values`, + 4 unit
  tiles.rs (couverture/disjonction/ordre/cancel) ; 21 goldens pixel-exacts
  inchangés, QA e13/e30 PASS + seahorse WARN (baseline), quick 0 gap
  (geomean 0.218, bruit mono-run). Cancel : poll par tuile (exécuteur) + par
  ligne de tuile (paths GMP/perturbation, pixels lourds) + re-check final
  (un cancel intra-tuile en fin de file ne doit pas retourner un buffer
  troué). Gates streaming : refine silencieux, OrbitTraps/Wings (pas
  d'orbites par tuile), 1re passe (pas de base — le warp G10.1 affiche déjà
  mieux), GPU (dispatcher non appelé). Diagnostic
  `tile_priority_first_paint_diagnostic` (`--ignored`) : zone prioritaire
  (r = 25 % diag autour du coin) complète à ~1 % du wall-clock total
  @768×576/20k iters.

## ✅ Shipped (condensé, le plus récent en haut)

**2026-07-16b** (`/improve`, ré-vérification CORRECTION corpus complet) :
- **Parité F3 corpus complet RE-VÉRIFIÉE clean post fix PNG** (`a68630e`) :
  après le fix de sauvegarde (qui ne touche pas le rendu), sweep parité full
  (compare_f3, EXR N0/NF, seuil gap 2 % rel_dsi). **64 cas confirmés < 2 %**
  (pire `nr_fail` 1.96 %, puis lethal_weapon 0.53 %, dragon_detail 0.30 % —
  tous bord-chaotique connu, 0 inside-mismatch structurel) → **0 régression de
  correction**. Métriques calculées directement des EXR (`read_exr_iterations`
  + `smooth_iter`) : le sweep full-tier à itérations natives + timeouts 600 s
  (e52465 slow-safe ~661 s, super_dense/seahorse F3-timeout) est impraticable en
  une passe — salvage EXR = même signal sans l'attente. Corpus parité-clean
  confirmé au nouveau commit. Leçon harness : préférer un sous-ensemble ou un
  cap d'itérations pour la parité full (éviter les cas 600 s connus).
- **Cohérence encodage PNG rapide** (`quality/report.rs` `write_heatmap_png`) :
  `set_compression(Fast)` sur les heatmaps de rapport QA (~45 PNG/suite),
  aligné sur `io/png.rs`. Lossless inchangé, 234 unit + 21 golden + QA 15 PASS.

**2026-07-16** (`/improve`, axe vitesse — corpus complet) :
- **Fix perf — sauvegarde PNG 5.6× plus rapide** (`io/png.rs`
  `save_png_rgb_with_metadata`) : l'encodeur `png` 0.18 défautait à
  `Compression::Balanced` (zlib niveau 6) → la SAUVEGARDE dominait le wall-clock
  des rendus rapides. Diagnostic sur **glitch_test_5** (seul cas du corpus 84 où
  fractall PERDAIT vs F3, ratio 1.99 au sweep full 256²) : à 1024² le rendu ne
  prend que **130 ms** mais la sauvegarde PNG **280 ms** (2.2× le rendu !). Or le
  build **F3 batch Linux ne sauve RIEN** (`HAVE_EXR=0`, `save_exr` no-op) → le
  harness comparait fractall(rendu+colorize+PNG) à F3(rendu seul), gonflant le
  ratio des cas à rendu rapide (courte période → LLA). **Fix** :
  `set_compression(png::Compression::Fast)` (fdeflate ultra-fast) → sauvegarde
  **280→50 ms**, fichier **+6 %** seulement (les images fractales compressent
  bien même en fast). Résultats : gt5 1024² wall **410→180 ms** ; gt5 256²
  standard 3-run **ratio 1.99→0.72 (LOSS→WIN)** ; geomean quick **0.234→0.199**
  (bénéficie à TOUS les rendus). Le rendu lui-même (~130 ms à 1024²) n'était
  qu'à ~1.2× F3 — c'était bien la sauvegarde, pas le kernel. PNG reste lossless
  (pixels décodés bit-identiques). Verrou :
  `io::png::tests::save_rgb_fast_is_lossless_and_round_trips_metadata` (pixels
  bit-exacts + métadonnées drag-and-drop). 234 unit + 21 golden (pixels décodés,
  insensibles au niveau de compression) + QA 15 PASS + quick 0 gap.
  **Reste (documenté, pas un gap moteur)** : l'asymétrie de mesure « F3 ne sauve
  pas en batch Linux » subsiste pour les autres cas à rendu rapide (le harness
  mesure le wall CLI complet, sauvegarde comprise) ; la sauvegarde rapide la
  réduit fortement mais ne l'élimine pas. Rendre la mesure strictement équitable
  (exclure la sauvegarde, ou faire sauver F3) = raffinement harness futur.

**2026-07-11** (`/improve`, axe vitesse + infra harness) :
- **Fix harness — fausse « RÉGRESSION geomean » cross-machine** (`scripts/
  harness.py`) : `_regression_gaps` flaggait une régression de VITESSE dès que
  `geomean_ratio > baseline·1.10`, SANS vérifier que la baseline vient de la
  même machine. Or les ratios fractall/F3 sont **machine-sensibles** (cœurs +
  CPU → équilibre orbite/série différent) — HARNESS.md pose « baseline **par
  machine** ». Rejouer le harness sur une autre machine (baseline i7-10700F
  16 threads vs Xeon 4 threads → geomean 0.33→0.62) levait donc une fausse
  alarme qui masquait le vrai signal `/improve` (gap #1 fantôme). **Fix** :
  `_same_machine(card, base)` (cpu + nproc) gate la régression geomean ; les
  régressions de CORRECTION (quality FAIL, parité pixel-equiv, goldens) restent
  machine-INDÉPENDANTES (non gatées). Le SCORECARD note « baseline d'une autre
  machine — pas de delta vitesse » et masque le delta trompeur. Vérifié : sur
  le vrai card Xeon vs baseline i7 `compute_gaps` renvoie `[]` (0 gap fantôme) ;
  même-machine + geomean +25 % → toujours flaggé. Verrous :
  `test_harness_guard.py::run_regression_gates` (3 scénarios : même-machine
  flag, cross-machine no-flag vitesse, cross-machine quality-fail toujours flag).
- **Fix perf — table BLA FloatExp (`bla_exp`) morte sur le path f64** (`delta.rs`
  `build_bla_entry`) : `bla_exp` (`BlaTableUnifiedExp`, coefficients FloatExp)
  n'est **LUE** que par le pixel loop exp (`use_exp_path`, `pixel_size <
  pixel_size_exp_threshold()` = 1e-280, `try_bytecode_unified_path`). Or son
  build était gaté uniquement sur `atom_hp_enabled()` (ON par défaut) + type
  Mandelbrot → **construite pour TOUT Mandelbrot perturbation** y compris aux
  zooms f64 (~1e13 … 1e280) qui lisent `tables` et jamais `bla_exp`. Build
  FloatExp O(M) sur toute l'orbite ≈ **10× le build f64** → poids mort pur.
  Mesuré sur **glitch_test_2** (zoom 5.97e112, réf intérieure 250 001 iters,
  path `bytecode_f64`) : `prewarm_bla` **0.139 s → 0.014 s**, rendu total
  0.389 s → 0.261 s, wall-clock 0.44 s → 0.29 s. **Ratio vs F3 2.74 → 1.89**
  (c'était le pire cas de l'axe vitesse). Bit-identique (image pixel-exacte —
  `bla_exp` n'était pas lue). **Fix** : gate le build sur la MÊME condition que
  son consommateur (`effective_pixel_size(params) < pixel_size_exp_threshold()`).
  Bénéficie à TOUT rendu Mandelbrot perturbation zoom f64 (1e13-1e280). Verrous :
  unit `bla_exp_skipped_on_f64_path` + `bla_exp_built_on_exp_path` (202 tests
  PASS), goldens 10/10 pixel-exact (e1000/e401 exercent le path exp → `bla_exp`
  toujours construite là), quality 11 PASS. NB : vérifié sur une machine
  éphémère 4 cœurs Xeon (≠ baseline i7-10700F) → baseline/SCORECARD du
  mainteneur préservés (pas de rebaseline cross-machine).

**2026-07-03** (`53a55cc`, `a8bf871`, `d26274b`, `cdb4f1d`) :
- **Fix perf — latence progress reporter** (`53a55cc` puis `d26274b`) : la boucle
  du reporter dormait 500ms fixes ; `reporter.join()` post-rendu bloquait donc
  jusqu'à ~500ms sur les rendus rapides (latence wall-clock réelle, cas test5 &
  co.). D'abord réduit à un poll 20ms (`53a55cc`, geomean 1.712→0.966) — mais un
  rendu de 3ms payait encore 20ms de plancher (test5 flappait win↔loss,
  geomean 0.966↔1.041 en bruit). Fix robuste `d26274b` : ProgressState porte une
  Condvar + guard bool ; le reporter fait `wait_timeout(250ms)` et `finish()` le
  réveille aussitôt → **0 latence join, sans busy-spin**. test5 process 20ms→3ms
  (win robuste), **geomean 0.966→0.890**. Goldens verts, 179 tests, quality
  inchangée. Rebaselines explicites `a8bf871` puis `cdb4f1d`. Deux itérations
  `/improve` complètes sur i7-10700F.

**2026-05-21** (sur `main` `fd9ce4a`..`1d88d16` ; + branche `g3-cusp-rings-fix`
`239952e`/`aca861c` à fast-forward) :
- **Fix G3 — anneaux concentriques cusp -0.75** (`239952e`) : `max_perturb_iterations`
  défaut 1024 < iter requis (~1700) tronquait les pas directs → compte d'itération
  radial ⇒ anneaux. Clamp `max_perturb_iterations`/`max_bla_steps` à
  `≥ iteration_max` dans `render_perturbation_with_cache`. cusp == GMP (mean 1.29
  vs 56.87) ; e113 golden encore + proche GMP (régénéré + revu). cf. [[perturbation-cap-rings]].
- **Fix GUI — deep zoom uniforme après zoom** (`aca861c`) : `HP_PRECISION` FIXE
  (256 bits) tronquait le centre au zoom (à 1e235 il faut ~783 bits) → vue fausse
  → image uniforme. `hp_arith_precision()` scale avec le span (`-log2(span)+96`).
  cf. [[uniform-image-causes]].
- **Dispatch GPU unifié (G5)** : `GpuRenderer::render_dispatch()` → un seul point
  d'entrée GPU partagé CLI (`main.rs`) + GUI (`gui/app.rs`) ; fin de la
  duplication du choix perturbation/type/kernel (~60 lignes). Fallback CPU via
  `None`. Vérifié GPU standard/perturbation/fallback + 178 unit + golden.
- **🏆 G2 RÉSOLU — fallback GMP éliminé (rebase-at-end F3)** : les cas deep
  escape-time (e50/e1000/dragon/… 36 ex-perf-bound) rendaient en GMP par-pixel
  car leur réf s'évade (`ref_truncated` → exhausted → GMP). Fix : rebase à `m=0`
  au bout de la réf pour orbites escape-time (F3 `hybrid.cc:301`), gate retiré.
  **e50 544→1.57 s, e1000 742→0.53 s (== GMP), dragon ~6 h→6.46 s** à 256²
  (acceptation d'origine ATTEINTE pour les 3). dragon n'était PAS impossible — il
  rendait en GMP. 178 unit + golden verts (e113 régénéré + revu, == pure GMP).
- **Fix crash OOB** (`6fcee8a`) : le rebase-at-end lisait `z_ref[m]` avec
  `m == ref_len` sur orbites périodiques profondes (crash GUI au chargement de
  floral_fantasy 1.55e85 / glitch_test_1 3.35e46). Clamp de l'index → 13 cas
  profonds rendent sans crash.
- **GUI : clic droit dézoome vers le curseur** (`1d88d16`) : `zoom_out_at_point`
  ignorait son `point` (dézoom au centre) ; rendu symétrique du clic gauche.
- **Perf deep-zoom (G2, bonus)** : BLA lookup aligned-start + libération des
  niveaux BLA < skip après build → table ~8× plus petite (e50 ~40→5 Mo). Aide
  TOUS les cas perturbation (désormais incl. les 36 ex-perf-bound). f64-scaled
  testée puis retirée (0 %). *(Note : le diag intermédiaire « memory-bound /
  dragon impossible » était faux — la vraie cause était le fallback GMP.)*
- **Parité F3 mesurée sur le corpus (G1)** : 2 sweeps (1920×1080 cap commun =
  83/84 + 79 pixel-équiv ; pleine profondeur = 46 réels validés jusqu'à 1e1200).
  **0 régression de correction.** glitch_test_1 tranché (victoire fractall).
- **Harness `compare_f3.py` durci** : `--bailout` (alignement ER des 2 côtés),
  classification timeout↔fail↔perf, métrique Δ **relative**, détecteur
  F3-dégénéré (timing + uniformité), `--out` absolu. `bench/` gitignoré.
- **Piste `optimize_reference_center`** (snap auto near-axis) pour le hang test2
  @1920×1080 (→ G3). *(Hypothèse initiale « cause commune avec les anneaux
  cusp -0.75 » INFIRMÉE : les anneaux venaient de `max_perturb_iterations`, cf.
  fix `239952e`.)*

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

- **INVARIANT : chemin de rendu unique CLI ↔ GUI**, CPU **et GPU**. CPU : un seul
  dispatcher `render_escape_time_cancellable_with_reuse`. GPU : un seul
  `GpuRenderer::render_dispatch`. Ne jamais redupliquer la logique de dispatch
  (choix perturbation/type, sélection kernel) dans `gui/app.rs` ni dans
  `main.rs`. Une divergence GUI/CLI = bug.

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
