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
  - **⚠️ Gap résiduel seahorse (QUALITÉ + VITESSE, pas mémoire)** : image
    **uniforme** (magenta plat, avg_iter 49.7 M ≈ max 50.1 M, spread 0.8 % — pas
    de structure) car l'orbite référence escape-time à ~50 M ne révèle pas le
    minibrot sans **nucleus-centering** (cf. §Nucleus finder : réf périodique
    exacte requise à deep zoom). **F3 lui-même échoue** sur seahorse (`f3_fail`,
    parité `f5b7c90`) → pas de ground-truth, fractall est ici **plus robuste que
    F3**. Leviers : `--find-nucleus` sur ce type de vue ultra-deep ; et vitesse
    216 s (orbite 207 s = 4667 b × ~50 M iters GMP, incompressible sans float128).
    Comparable au cas hors-plage e22522 : *rend mais dégradé, à améliorer plus tard*.
  - Reste orthogonal pour la PERF (pas la mémoire) de ces cas intérieurs :
    period-aware reference (G2, cf. plus bas, 4 sessions brûlées — bloqué sur le
    critère atom-domain F3 exact).
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
- [ ] **Re-sweep corpus complet avec le fix** : confirmer que les 36 ex-perf
  cas rendent ET matchent F3 (probable vu e1000 pixel-identique + e113 == GMP).
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
  - [ ] **Tuning perf dd-BLA** : l'epsilon 2⁻¹⁰⁶ (précision max) restreint la BLA.
    Un epsilon adaptatif = précision *nécessaire* (fonction de δ/sensibilité, pas
    du type) élargirait le skip là où < 106 b suffit → plus rapide sans casser le
    max_diff. Lié au wisdom ci-dessous.
  - [ ] **auto-dispatch (wisdom)** : sélectionner dd automatiquement selon une
    estimation de sensibilité/conditionnement par frame (F3-style), au lieu de
    l'opt-in `use_dd_tier`. Prérequis perf : dd-BLA ✅ + epsilon adaptatif (sinon coût hors
    des cas sensibles).
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

**Done when** :
- [x] **Golden : verrouiller les fixes deep-zoom** (2026-05-21) — ajout de
  `mandelbrot_e50` (1e50, rebase-at-end G2), `mandelbrot_e1000` (1e1000),
  `mandelbrot_cusp_m075` (cusp -0.75, fix max_perturb G3), `mandelbrot_floral`
  (1.55e85, fix period-detection G3). Rendus == chemin par défaut == GMP, revus
  visuellement, verts en CI (déjà câblée). 4 nouveaux goldens.
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

### G8 — Harness d'auto-amélioration · `[P0 · méta — égaler puis dépasser F3]`

> Protocole : `HARNESS.md`. Outil : `scripts/harness.py`. Itération : skill
> `/improve`. Mémoire : `SCORECARD.md` + `harness/history/*.json` (versionnés).
> Trois axes : vitesse (geomean ratio vs F3), parité (compare_f3), qualité
> (pert vs GMP + goldens). Priorités : correction > robustesse > vitesse >
> qualité. « Dépasser » = geomean ≤ 1.0 + correctness_wins ≥ 2 + 0 régression
> superset.

**Done when** :
- [x] **Protocole écrit** (`HARNESS.md`) + skill `/improve` (2026-07-03).
- [ ] **`scripts/harness.py`** opérationnel (score/baseline/gaps, tiers
  quick/standard/full, JSON + SCORECARD.md, gaps triés) — en cours.
- [ ] **`fractall-quality` émet du JSON** (`suite-summary.json`,
  `report.json`) — en cours.
- [ ] **Binaire F3 Linux** buildé (`fraktaler-3-3.1/`) — la machine courante
  (i7-10700F Linux) remplace le M4 ; les baselines macOS ne sont plus
  comparables.
- [ ] **Baseline v1 committée** sur cette machine (tier standard).
- [x] **Boucle exécutée au moins 1×** de bout en bout (`/improve`) avec
  amélioration mesurée committée + verrou posé (2026-07-03, `53a55cc` :
  latence progress reporter → geomean vitesse 1.712→0.966 ; rebaseline
  `a8bf871`).

---

## ✅ Shipped (condensé, le plus récent en haut)

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
