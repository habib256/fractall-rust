# Changelog

Tous les changements notables de **fractall-rust** (renderer de fractales
deep-zoom, référence algorithmique Fraktaler-3.1).

Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/) et adapté :
regroupement par type (Ajouté / Corrigé / Performance / Modifié), et sections par
jalon (`Gx`) pour l'historique. Ce fichier est un résumé actionnable — le détail
technique vit dans `TODO.md`, `CLAUDE.md`, `SCORECARD.md` et l'historique git.

## [Unreleased] — nuit d'optimisation et de correction (2026-11-06, Apple M4)

### Performance
- **Path f64 standard par lots** (`bytecode/interp_batch.rs`) : 8 points en
  lockstep, bit-identique à l'interpréteur scalaire — phase pixel ×4,5 à ×7
  (Mandelbrot 1440p 0,31 → 0,057 s ; seahorse 1e6 16,6 → 2,4 s). C'est le
  chemin de toute navigation sous ~1e13. `FRACTALL_STD_BATCH=0` = scalaire.
- **Tier exp (> 1e280)** : parse GMP du span HP sorti de la boucle pixel
  (e401 ×3, e1000 ×6) ; boucle HP allégée (−22 à −28 %) ; boucle « double mis
  à l'échelle » (`bytecode/pixel_loop_scaled.rs`, ×2 à ×7, PNG identiques sur
  les 37 cas du tier exp du corpus). `FRACTALL_EXP_SCALED=0` = boucle FloatExp.
- **Coût fixe par pixel de la perturbation** : plus d'`Arc` cloné ni de
  `Formula` compilée par pixel (test5 ×7, flake/floral −27 %).
- **Correction dd** : chemin rapide double-double sans exposant (boucle dd
  ×2-5, bit-identique), tables BLA inutiles supprimées, capture paresseuse de
  la partie basse — pic RSS dragon 1 063 → 329 Mo, orbite dragon −40 %.
- **Pas perturbation générique** (Julia, Burning Ship, hybrides) sans dérivée
  quand elle n'est pas demandée : Julia −20 %.
- **GPU Metal** : sonde BLA ascendante dans le kernel f32 (×4 sur le temps
  kernel) ; routage Auto du GPU dans la plage perturbation f32 après
  `--wisdom-bench` (4-13× sur les scènes faciles, 1,3-2,3× sur les spirales).

### Corrigé
- **Détecteur de fiabilité** : la troncature d'un saut BLA de `l` pas vaut
  l·ε (elle était comptée ε : e227 avait 384 pixels faux non flaggés à 256²,
  golden_spider 169, dragon 36 à 128²) ; borne exp emballée qui retombait à
  « fiable » par débordement d'exposant ; test de détermination à l'évasion.
  Le traqueur du tier exp est actif par défaut (`FRACTALL_RELIABILITY_EXP=0`).
- **Harmonic LA** : rayon de validité 2⁻²⁴ → 2⁻⁵³ (leaded_glass 227 pixels
  faux → 0) ; golden `mandelbrot_glitch_test_2_harmonic` régénéré (l'ancien
  avait 11 pixels ≠ GMP pur, le nouveau 1).
- **Cache de table BLA** : clé par contenu d'orbite + rayon de vue (une
  table périmée pouvait être réutilisée après réallocation ou sur vue
  élargie).
- **GPU f32** : sous-dépassement de |δ|² au-delà de ~1e19.
- Diagnostics : zoom affiché au-delà de 1e308 ; harness sous macOS (nom de
  machine, RAM, binaire F3 non lançable détecté).

### Ajouté
- **Harness : `hotspots`** — profil échantillonné d'un cas avec un binaire
  symbolisé (`sample`/`perf`) : quelles fonctions consomment le temps.
  Phase `bla` et `ns/iter·thread` dans le profil.
- `fractall-quality gpu-bench` (temps kernel / correction CPU / % renvoyé).

## [Unreleased, précédent] — harnais d'optimisation

### Ajouté
- **Harness : axe `profile`** (où va le temps) — chaque cas du tier est rendu
  une fois avec les diagnostics du moteur capturés : path exécuté, ventilation
  du wall-clock orbite/pixels/post/sauvegarde/autre, ns/iter effectif, % de
  pixels flaggés par le détecteur de fiabilité, escalade dd. Versionné dans
  `harness/history`, affiché dans `SCORECARD.md`. Un gap est levé quand la
  sauvegarde domine le rendu ou quand `ns/iter` régresse de 15 % à path égal.
- **Harness : axe `bench`** — `fractall-cli --wisdom-bench` lancé isolé
  (`FRACTALL_WISDOM_FILE` dans `bench/`) → iters/s et ns/iter par technique
  (`cpu_std_f64`, `cpu_perturb_{f64,exp,dd}`, `gpu_*`), régression > 15 % =
  gap.
- **Harness : vitesse intrinsèque sans F3** — temps fractall par cas comparés
  à la baseline de la même machine (`speed_self`, geomean + pire cas) ; gap
  par cas au-delà de 15 % ET du bruit mesuré (`spread`, runs ≥ 2) ET de 50 ms.
- **Harness : `doctor`** (prérequis, axes mesurables sur la machine, baseline
  d'une autre machine, crash en suspens), **`trend`** (évolution des métriques
  dans l'historique, `--case`, `--bench`), **`render`** (re-génère le
  SCORECARD depuis un history sans re-mesurer) ; section « Tendance » dans le
  SCORECARD.
- **Harness : `ab`** — A/B isolé par variable d'environnement ou flag CLI
  (médiane + bruit sur N runs, ventilation par phase, path routé, identité
  des octets PNG entre bras) : l'outil de la méthode G14. Profil : **pic RSS**
  par cas (wrapper `getrusage`, sans `/usr/bin/time`) ; `score
  --parity-iterations N` plafonne les itérations des deux moteurs (parité
  `full` praticable). Le moteur émet la ligne `[FRACTALL] path=std_f64|std_gmp`
  sur les paths standard (un cas shallow sortait `path=?` dans le profil).
- **Harness : baselines par machine** (`harness/baselines/<cpu-threads>.json`,
  repli `baseline.json`) — une machine secondaire fige sa référence sans
  écraser celle de la machine de dev.
- `TODO.md` G14 : plan de campagne des optimisations, chaque item avec sa
  métrique harness ; CI exécute `scripts/test_harness_guard.py`.

### Corrigé
- **Le harness démarrait seulement avec numpy + OpenEXR** (`import compare_f3`
  en tête de fichier faisait `sys.exit`) : import paresseux, statut
  `f3_deps_missing` distinct de `f3_unavailable` ; la détection de F3 filtre
  par format binaire (le `.macos` versionné était « trouvé » sous Linux →
  10 `fail` silencieux de l'axe vitesse).
- `scripts/test_harness_guard.py` échouait depuis l'ajout de la règle « un
  crash côté F3 ne quarantaine pas » (fixture sur la phase `speed_f3`) ;
  fixture corrigée, règle verrouillée, parseur du profil et seuils de vitesse
  intrinsèque couverts.

### Modifié
- `scripts/harness.py` scindé en package `scripts/harnesslib/` (10 modules
  < 800 lignes, pilier 5) ; `harness.py` reste le point d'entrée. Schéma du
  history 2 (sections `profile`, `bench`, `speed_self` ; `trend` lit les deux).
- Docs : `HARNESS.md` (axes intrinsèques vs F3, checklist machine neuve),
  skill `/improve` (étape doctor, lecture du profil, `cargo test --lib`),
  `harness/README.md` (rôle de chaque registre), `CLAUDE.md`, README.

## [0.8.3] — 2026-08-30 — densité navigable, hybrides deep, distribution portable

### Ajouté
- **AppImage Linux x86_64** (`packaging/linux/`) : un fichier unique qui embarque
  les trois binaires de la distribution. `AppRun` choisit lequel lancer selon le
  nom d'invocation (`ln -s fractall-*.AppImage fractall-cli`) puis selon le
  premier argument, avec la GUI par défaut pour que le double-clic ouvre
  l'explorateur. Bâtie dans le MÊME conteneur bionic que le `tar.gz` — la glibc
  de la machine de build est le plancher de compatibilité — et son icône est un
  rendu frais du `fractall-cli` qu'elle emballe.
- **Format TOML natif Fraktaler-3** : import (`--toml` accepte les fichiers F3
  complets, blocs `[[formula]]` inclus) et export (`src/io/f3_toml.rs`), en plus
  du format léger déjà supporté.
- **Maps EXR lisibles par Kalles Fraktaler 2** (`src/io/exr.rs`) : les canaux
  bruts exportés suivent désormais la convention de nommage KF2.
- **Anti-aliasing sur GPU** : l'AA multi-passes n'est plus réservé au CPU
  (`src/gpu/`), avec la décorrélation Cranley-Patterson du jitter PAR PIXEL
  portée dans les kernels, et l'échelle du jitter exposée dans la GUI.
- **Buddhabrot navigable en profondeur** : nouvel échantillonnage par
  importance (**Metropolis-Hastings**, `fractal/density.rs`) pour Buddhabrot,
  Nebulabrot et Anti-Buddhabrot. Des chaînes de Markov ciblent la contribution
  à la fenêtre affichée au lieu de tirer les `c` au hasard sur tout le plan, ce
  qui garde l'image vivante à n'importe quel zoom — le tirage uniforme, lui,
  laissait la fenêtre vide dès ~×1000. Amorçage recuit (la fenêtre part du
  domaine entier et se resserre par octaves), projection pondérée pour rester
  non biaisée, accumulation déterministe. La vue par défaut, où le tirage
  uniforme ne souffre d'aucune famine, continue de l'utiliser.
- `FRACTALL_DENSITY_SAMPLING=uniform|metropolis` force le régime,
  `FRACTALL_MH_DEBUG=1` trace la survie des chaînes.
- **Tests e2e de navigation profonde** (`tests/deep_navigation.rs`, cible CI) :
  navigation réelle via `ViewHp` (molette ancrée, pan, sélection, resize) puis
  rendu par le dispatcher, pour les quatre types spéciaux. Verrouille la
  consommation effective des coordonnées HP au-delà du gel des miroirs `f64`,
  l'exactitude au pixel du pan sur le path MPC, le déterminisme en zoom profond
  et le découplage échantillonnage ↔ vue.
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
- **G4 jalon 5f — nucleus phase-aware** : port F3 `hybrid_period`/`center`/
  `size` pour formules hybrides via un interpréteur GMP dual-mat2 (Jacobien
  ∂z/∂c matriciel par opcode — les abs cassent la conformité). `--find-nucleus`
  marche sur les hybrides : [M,M]@minibrot 1e18 pixel-identique à [M] (période
  445, K/size identiques) ; [M,BS] genuine → satellite période 123, Newton
  convergé, K non-conforme.
- **G4 jalon 5e — parité hybride vs F3 natif** : light-toml `phases = "…"`
  (loader CLI `--toml` + `compare_f3.py` → blocs `[[formula]]` F3). Cas corpus
  `hybrid_mbs_smooth_e8` (f64-std, Δmean 0.0026) et `hybrid_mbs_smooth_e13`
  (**perturbation multi-phase : Δ = 0.0000, pixel-identique à F3**) — le juge
  EXTERNE du stack G4 hybride complet.
- **G4 jalon 5c — éditeur GUI de séquence hybride** : menu Type → « Hybride
  (séquence) » (composer ➕/⌫/🗑, Appliquer ≥ 2 phases, Désactiver) ; label
  « Hybride: M⊕BS » quand actif ; drag-and-drop PNG restaure la séquence.
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
- **Binaires de release inexécutables** (SIGILL au démarrage) : `.cargo/config.toml`
  est versionné avec `target-cpu=native` pour le confort du dev local, et aucun
  job de release ne le surchargeait sauf celui du Pi 400. Les artefacts étaient
  donc compilés pour le CPU du *runner* GitHub : la v0.8.2 linux-x86_64
  embarquait de l'AVX-512 (`vptestmb`) et mourait instantanément sur tout CPU
  qui n'en a pas — vérifié sur Core i7-8700, `fractall-gui` comme
  `fractall-cli`. Chaque job pose maintenant `RUSTFLAGS` explicitement
  (`x86-64` sur Linux/Windows x86_64 — la même baseline que `ci.yml`, qui
  posait déjà ce garde-fou pour les tests —, `apple-m1` sur macOS).
- **Export haute résolution en mode Distance / OrbitTraps** : le rendu 4K/8K
  partait sans le canal que le mode consomme et **échouait** à la colorisation
  vérifiée (« Erreur de colorisation »). La règle « ce mode exige ce canal »
  n'était posée que sur le clone du rendu fenêtre ; elle est désormais unique
  (`render::ensure_required_channels`) et appliquée à chaque frontière de
  rendu — CLI, fenêtre, export HQ, preview Julia.
- **Changement de type de fractale** : tout réglage absent d'une liste blanche
  de six champs était silencieusement réinitialisé — espace colorimétrique,
  décalage de palette, jitter d'anti-aliasing, tier dd, moteur bytecode, et
  huit des dix réglages de perturbation. Le transfert se fait maintenant
  **groupe par groupe** (`params_for_type_keeping_preferences`) : le type
  définit la formule et la géométrie, l'utilisateur garde couleur,
  échantillonnage, moteur, perturbation et canaux.
- **Densité : bord gauche et bord haut** — le path f64 tronquait les
  coordonnées au lieu de les arrondir vers le bas, repliant sur la première
  colonne et la première ligne des points situés JUSTE en dehors de la fenêtre.
  Le path MPC arrondissait déjà correctement.
- **Types de densité navigables** : Buddhabrot, Nebulabrot et Anti-Buddhabrot
  tiraient leurs paramètres `c` DANS la fenêtre affichée. Zoomer donnait donc
  une image **entièrement noire dès ×100** (les orbites quittent la fenêtre et
  n'y reviennent pas), et n'importe quelle fenêtre éloignée se remplissait
  d'une **nappe uniforme** (le premier itéré `z₁ = c` y retombait toujours).
  Les `c` sont désormais tirés sur le domaine canonique du plan des paramètres
  (centre `(-0.5, 0)`, span `4×3`), indépendamment de la vue : la fenêtre ne
  sert plus qu'à projeter les trajectoires. Les rendus de la vue par défaut
  restent bit-identiques.
- **G4 jalon 5a — hybrides GENUINE deep : références par phase** (port F3
  `hybrid_references`). Trois bugs latents corrigés, invisibles au verrou
  `[M,M]` (phases identiques ⇒ désync sans effet) : (1) **désync
  phase/référence au rebase** — le pas pixel `phases[n % N]` vs la réf itérée
  `phases[m % N]` divergeaient après tout rebase `m := 0` avec `n % N ≠ 0` ;
  fix = N orbites référence (la réf p itère `phases[(p+i) % N]`,
  `hybrid_phase_refs`) + tracking F3 `(phase + m) ≡ n (mod N)` dans les boucles
  f64/exp ; (2) **fast-path dd de l'orbite z²+c** pris par les hybrides
  (`fractal_type == Mandelbrot`) → référence fausse ; gates `!is_hybrid` sur
  dd/atom-domain/Brent/compresseur/harmonic ; (3) **cache d'orbite** sans
  discriminant `hybrid_phases` (réutilisation [M,BS]→[M] au même centre).
  Avant fix : [M,BS] @3e10 = image UNIFORME (100 % faux) ; après : structure
  exacte, [M,M] 160/160 pixel-exact vs GMP cyclant, résidu [M,BS] = plancher
  bruit f64 des plis abs (78 % dépendant de la réf, vérité GMP 256=512 stable).
  Verrou : grille GMP-cyclant `multi_phase_perturbation_matches_gmp_per_pixel`.
- **Perturbation réf-intérieure >512²** : un 2e bloc de résolution glitch récursive
  (`perturbation/mod.rs`) n'était pas gaté `!bytecode_path` → il supprimait le
  fallback GMP → ~3.4 % de structure spurious. Gate ajouté.
- **GUI warp G10.1** : signe Y inversé de l'aperçu zoom molette hors-centre
  (`compute_warp_norm` : `0.5 + dy`, `src/gui/app.rs`).

### Performance
- **G4 jalon 5d — atom-domain générique hybrides (mat2)** : troncature des
  réfs hybrides (port F3 `hybrid_reference` : dZdC mat2, critère
  `|inv(r·J)·Z| < 1` calculé sans inversion) via `J' = A_i·J + I` (Jacobiens
  dual-numbers par phase, FloatExp). [M,M] e50 tronque au même index que [M]
  ⟹ **pixel-exact == [M] (0 px, verrou e50)** et 0.50 → 0.38 s.
- **G4 jalon 5b — BLA par phase pour les hybrides deep** (port F3
  `hybrid_blas`) : `BlaTableUnified::build_cycled` (le pas i de la table de
  phase p utilise `phases[(p+i) % N]`, bâtie sur `refs[p]`) + saut
  `tables[phase].lookup{,_fexp}` dans les boucles multi-phase f64/exp.
  `prewarm_bla_entry` couvre désormais le multi-phase (sinon le build N-tables
  se faisait sous le lock global = tout le temps pixel) ; série Taylor gatée
  `!is_hybrid` (z²+c — son heuristique auto-adjust misfirait : iteration_max
  ×4 parasites). **[M,M] e50 96²/263k iters : 3.34 → 0.50 s (6.7×), pixels
  3.15 → 0.30 s** ; e1000 (exp) ≈ parité single-phase. [M,M]==[M] e1000
  pixel-exact avec BLA ; déterminisme 0 px.
- **Fallback dd réf-intérieure** : quand `glitch_ratio > 0.30` (Mandelbrot
  bytecode), escalade au tier dd (~106 b) au lieu du full-GMP per-pixel — ~4-8×
  plus rapide, pixel-exact GMP (`GLITCH_FALLBACK_THRESHOLD`, `perturbation/mod.rs`).

### Supprimé
- **Résolution de glitch Pauldelbrot** : passe voisinage, clustering spatial,
  références secondaires et résolution récursive par itération sont retirées du
  CPU comme du GPU, avec le module `glitch.rs` et trois réglages
  (`max_secondary_refs`, `min_glitch_cluster_size`, `glitch_neighbor_pass`).
  Fraktaler-3 n'en a aucune trace : le rebasing rend la détection inutile. Ces
  blocs étaient déjà inertes sur le chemin par défaut, et là où ils tournaient
  encore ils coûtaient un tiers du temps de rendu pour une image identique au
  pixel près. Rendus par défaut inchangés.

### Modifié
- **Empreinte de reprise vidéo aveugle au GROUPE couleur** : `map_fingerprint`
  neutralise `params.color` en entier au lieu d'énumérer cinq champs. Un
  réglage de colorisation ajouté plus tard reste automatiquement hors
  empreinte — l'oublier aurait invalidé toutes les keyframes d'un projet
  (des heures de calcul) pour un changement de palette.
- **Surface de sérialisation verrouillée** : la liste EXACTE des 49 clés à
  plat, un round-trip à valeurs **toutes** non-défaut (détecte un champ perdu
  et une collision de nom entre deux groupes `flatten`) et l'absence des deux
  champs d'état AA transitoire. Ce contrat porte les PNG, les `.fmap`, les
  TOML *et* les empreintes de cache : il ne doit plus bouger par effet de
  bord. Les trois verrous sont mutation-testés.
- **`FractalParams` regroupé** : les réglages de perturbation, de couleur,
  d'échantillonnage sous-pixel, de formule (Multibrot, hybrides, opcodes,
  séquence Lyapunov), de canaux annexes (distance, intérieur, orbit traps) et
  de moteur (mode d'algorithme, bytecode, tier dd, nucleus, précision GMP)
  vivent désormais dans six sous-structures (`params.perturbation.*`,
  `params.color.*`, `params.sampling.*`, `params.formula.*`,
  `params.channels.*`, `params.engine.*`) au lieu de flotter dans une struct
  plate d'une cinquantaine de champs. La **sérialisation ne change pas** : les
  clés restent à plat, donc les PNG, `.fmap` et TOML déjà écrits se relisent à
  l'identique.
- **Affichage des densités invariant d'échelle** : la compression
  logarithmique se rapporte désormais à un quantile haut de la densité, et non
  à sa valeur brute. Le contraste ne dépend plus du budget d'échantillons —
  donc de la résolution demandée — ni du régime d'échantillonnage, et quelques
  pixels chauds n'écrasent plus l'image.
- **Couverture >512² durcie** : nouveau golden `mandelbrot_interior_ref_640`
  (seul cas >512², exerce l'escalade dd) ; audit des gates `small_image` (tous
  sains).

**État moteur** : 0 gap mesuré (harness quick + standard-speed + wisdom-optimality),
bat F3 partout (geomean ~0.18, 25/25 wins speed). 271 tests unit CLI + 24 golden
pixel-exact + quality 15/15 PASS.

## [0.8.2] — 2026-08-23 — chasse aux bugs

### Corrigé
- **Zoom XaoS** : erreur de position par pixel (le modèle par axe dérivait en
  molette continue, 0,5 → 1,5 px), transformée HP à précision dynamique
  (faux au-delà de ~1e74), fin de la réutilisation grossière inter-passes
  (treillis 1/16 de pixels à 1,5 px, progressif et HQ), frame source non
  dégradée en zoom-out, `glitch_ratio` sur pixels calculés.
- **Perturbation** : K du nucleus finder enfin appliquée au pixel→c (golden
  `hybrid_mbs_nucleus_5e28` régénéré) ; `iterate_pixel_gmp` bouclait à
  l'infini sur un pixel intérieur qui rebase ; offset off-center et jitter
  AA propagés aux paths GMP legacy ; discriminants de formule du cache subset.
- **Couleur** : `--enable-interior-detection` noircissait ~50 % de
  l'extérieur ; couture sur `color_offset` négatif.
- **GUI** : resize en HP (centre préservé en deep zoom), clic-zoom par ratio
  pixel (zoomait au centre > 1e15), touche `C`, AA vs recolorisation,
  `TileProgress` upscalé, raccourcis inhibés en saisie.
- **Studio vidéo** : « Ré-assembler » re-planifiait sur le zoom de la vue
  (vidéo tronquée, manifest écrasé) ; miniatures provisoires périmées ;
  mini-rendus non annulables ; drag des points de vitesse.
- **CLI/vidéo/GPU** : AA multi-sample et vidéo perdaient distances/orbites
  (Smooth silencieux — vidéo : refus explicite), `color_repeat` forcé à 40,
  HP périmées après `--toml`, NaN dans les métadonnées PNG, dimensions 0 ;
  jamais de GPU quand distances/orbites/dd sont demandés ni avec
  `--find-nucleus`.

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
