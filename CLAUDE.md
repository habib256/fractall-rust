# CLAUDE.md

> Carte rapide du repo pour reprendre une session sans relire tout le code.
> Référence algorithmique : **Fraktaler-3.1** (`fraktaler-3-3.1/src/`,
> `fraktaler-3-3.1/fraktaler-3-analysis.md`).
> **Boucle d'auto-amélioration** : protocole dans `HARNESS.md`, outil
> `scripts/harness.py`, itération via le skill `/improve`, dernier état dans
> `SCORECARD.md` + `harness/history/`.

## Build & Run

```bash
cargo build --release
cargo run --release --bin fractall-cli -- --type 3 --output out.png
cargo run --release --bin fractall-gui
cargo run --release --bin fractall-quality -- suite
cargo run --release --bin fractall-video -- plan cfg.toml -p proj/   # G12 vidéo
cargo run --release --bin fractall-video-gui   # studio vidéo graphique (G12 j6)
```

Prérequis natifs : GMP / MPFR / MPC (pour `rug`).

## Tests

- **Unit tests** : `cargo test --release --lib` (~530 tests). ⚠️ Cible **lib**
  (`src/lib.rs`) : les binaires ne sont que des enveloppes, les tests des
  modules ne tournent plus en triple sous `--bin`.
  Couvre `perturbation/{bla,delta,series,nonconformal,distance,interior,
  orbit,types,nucleus,mod}`, `jitter` (AA), `lyapunov`, `progressive`,
  `wisdom` (select_algorithm/device, plan), et tout le `bytecode/` (`compile`
  + `compile_hybrid_formula`, `interp{,_gmp}`, `bla_dual`, `delta_form`,
  `pixel_loop{,_exp}`).
- **Golden image tests** (`tests/golden_images.rs`) : 29 cas pixel-exact
  (goldens forcent `--no-gpu` pour déterminisme cross-machine) — paths f64
  standard, perturbation, deep zoom GMP, types non-bytecode, zooms
  intermédiaires (1e10/1e15/1e20, path perturbation par défaut), atom-domain,
  et `mandelbrot_interior_ref_640` (SEUL cas >512², exerce l'escalade dd).
  - Lancer : `cargo test --release --test golden_images`
  - Régénérer : `FRACTALL_UPDATE_GOLDENS=1 cargo test --release --test
    golden_images`. **Toujours** vérifier visuellement les nouveaux PNG.
  - Note : les goldens ER=25 (2026-05-20) ; `mandelbrot_e10/e15/e20` (G6,
    2026-07-12) comblent le trou 1e8→1e50 (revus visuellement, e10 PASS
    pixel-exact vs GMP, e15/e20 bruit de bord dispersé p99=0).
- **Deep navigation tests** (`tests/deep_navigation.rs`) : e2e des types
  SPÉCIAUX (Buddhabrot, Nebulabrot, Anti-Buddhabrot, Lyapunov) — navigation
  réelle via `ViewHp` (molette ancrée / pan / sélection / resize) puis rendu par
  `render_request`. Verrouille : canaux complets et finis à toute profondeur,
  déterminisme, **consommation HP** (deux vues aux miroirs f64 identiques mais
  aux chaînes HP différentes doivent rendre différemment — sonde en zone
  Lyapunov CHAOTIQUE, les zones périodiques étant numériquement plates), pan de
  k pixels exact sur le path MPC, et découplage domaine d'échantillonnage ↔ vue
  pour la densité. `cargo test --release --test deep_navigation` (~9 s).
- **QA perturbation vs GMP** (`fractall-quality`) : compare le chemin
  perturbation au rendu GMP pur pixel-par-pixel. Voir §Quality plus bas.

CI : `.github/workflows/ci.yml` (unit + golden sur push/PR, ubuntu,
gmp/mpfr/mpc). Extension du corpus golden à venir (cf. TODO G6).

**Release** : `.github/workflows/release.yml` (tag `v*` ou lancement manuel) →
`dist/` via `.github/scripts/package.sh`. 4 cibles : linux-x86_64-glibc2.27
(conteneur Ubuntu 18.04 — ⚠️ PAS en `container:`, Node 20 des actions exige
glibc ≥ 2.28 ; on checkout sur l'hôte et on `docker run` le build seul),
linux-arm64-pi400 (runner ARM64 natif + conteneur bullseye = glibc 2.31 pour
couvrir Pi OS bullseye ET bookworm, `-C target-cpu=cortex-a72`),
windows-x86_64 (toolchain **GNU + MSYS2** — `gmp-mpfr-sys` compile GMP depuis
les sources et **ne supporte pas MSVC** ; les DLL MinGW sont copiées à côté
des .exe), macos-arm64.

**⚠️ RUSTFLAGS obligatoire par job** : `.cargo/config.toml` est versionné avec
`target-cpu=native` (confort local). Sans surcharge, un job de release compile
pour le CPU du RUNNER — la release v0.8.2 linux-x86_64 embarquait de l'AVX-512
(`vptestmb`) et mourait en **SIGILL** au démarrage sur tout CPU sans AVX-512.
Chaque job pose donc `RUSTFLAGS` (`x86-64` sur x86_64 — la baseline, comme
`ci.yml` —, `apple-m1` sur macOS,
`cortex-a72` sur Pi 400) — la variable d'env écrase `build.rustflags`.

**AppImage x86_64** (`packaging/linux/`, calqué sur POM1) : produite par le
MÊME job bionic que le tar.gz (`build_appimage.sh`, appelé avec
`FRACTALL_APPIMAGE_SKIP_BUILD=1`), car la glibc de la machine de build est le
plancher de compatibilité et linuxdeploy doit résoudre contre le rootfs 18.04.
`AppRun` dispatche GUI/CLI/quality (nom d'invocation `ARGV0`, puis 1er
argument, défaut GUI) ; l'icône est un **rendu frais de `fractall-cli`** (pas
d'asset versionné ; override `FRACTALL_APPIMAGE_ICON` quand l'hôte ne peut pas
exécuter le binaire produit). `usr/lib` reste VIDE et c'est normal : les
binaires ne listent que la glibc en `NEEDED`, winit/wgpu chargent
X11/xkbcommon/Wayland/GL par `dlopen` depuis le système.

## Harness d'auto-amélioration (HARNESS.md)

Axes **contre Fraktaler-3** (exigent F3 + numpy/OpenEXR) : **vitesse-ratio**
(wall-clock head-to-head, geomean des ratios), **parité** (`compare_f3.py`,
EXR N0/NF). Axes **intrinsèques** (tournent partout, dont CI/conteneur) :
**vitesse auto-relative** (temps fractall par cas vs baseline même machine),
**qualité** (`fractall-quality` vs ground truth GMP + goldens), **fuzz**
(sondes pert-vs-GMP), **truth** (rendu de PRODUCTION vs tier dd sur les cas
du corpus — le seul axe qui juge la sortie par défaut là où elle est
réellement utilisée ; a manqué jusqu'au 2026-11-06), **precision** (juge orbite à précision ×4 — seul axe
qui voit une précision d'orbite insuffisante : le juge GMP de quality partage
la formule du moteur), **profil** (où va le temps : path exécuté + ventilation
orbite/pixels/post/sauvegarde + ns/iter + % fiabilité), **bench** (iters/s par
technique via `--wisdom-bench` isolé).
**wisdom** (2026-11-19 — juge les DÉCISIONS automatiques : regret du plan
`auto` contre chaque plan forcé à sortie équivalente, adjudication GMP
automatique quand l'alternative rapide rend autre chose, et justesse du
plafond choisi par la sonde d'itérations contre le rendu à 4N), **types**
(les 33 types : vue d'accueil + vue de bord ×4096 figées dans
`harness/type-scenes.json`, vitesse en ns/pixel·itération et juge GMP à
convergence vérifiée — un écart à juge non convergé est `CHAOS`, pas un
gap). Cycle d'A/B léger : `score --tier micro` (128², axes
speed/truth/wisdom/types/profile). Sans gap, la
liste de travail est la section **Leviers** du scorecard (`harness.py
levers` : temps mesuré trié par ce qui le consomme).

```bash
python3 scripts/harness.py doctor                # prérequis + axes mesurables ICI
python3 scripts/harness.py score --tier micro    # cycle d'A/B léger (128², axes essentiels, tous types)
python3 scripts/harness.py score --tier quick    # référence avant commit (~10 cas, 256²)
python3 scripts/harness.py score --tier standard # 3 runs médiane, avant commit perf
python3 scripts/harness.py trend --case e50      # évolution d'un cas (temps, ns/iter, path)
python3 scripts/harness.py ab --cases e50 --env FRACTALL_RELIABILITY=off,escalate  # A/B isolé (G14)
python3 scripts/harness.py hotspots --case e1000 --size 1200  # profil ÉCHANTILLONNÉ : quelles fonctions (binaire symbolisé)
python3 scripts/harness.py levers                # où gagner : temps mesuré, trié (même sans gap)
python3 scripts/harness.py wisdom-optimality --only algorithm  # axe wisdom seul (harmonic|algorithm|iter)
python3 scripts/harness.py baseline              # fige la référence de CETTE machine (harness/baselines/)
python3 scripts/test_harness_guard.py            # verrous du harness (CI)
```

Sorties : `harness/history/<date>-<sha>.json` (mémoire longue, versionnée),
`SCORECARD.md` (dernier score + profil + gaps triés), artefacts lourds dans
`bench/` (gitignoré). Baseline **par machine** (`harness/baselines/<slug>.json`,
repli `baseline.json`). Binaire F3 : auto-détection
(`fraktaler-3-3.1/fraktaler-3-3.1.linux` Linux, `.macos` sinon, filtrée par
format ELF/Mach-O, override `F3_BIN`) — sans F3 le reste tourne. Code :
`scripts/harnesslib/` (fichiers < 800 lignes ; `harness.py` = enveloppe).
Priorités de gap : correction > robustesse > vitesse > qualité. Détail,
invariants et checklist machine neuve : `HARNESS.md` ; plan des optimisations
à venir : `TODO.md` G14.

## Architecture

```
src/
├── main.rs              # CLI fractall-cli (clap)
├── main_gui.rs          # GUI fractall-gui (egui/eframe)
├── main_quality.rs      # QA fractall-quality (clap subcommands)
├── main_video.rs        # fractall-video (G12) : plan / render / assemble
├── main_video_gui.rs    # fractall-video-gui : studio vidéo graphique (G12 j6)
├── video_gui/           # Studio vidéo eframe SÉPARÉ du générateur (gui/ intact)
│   ├── mod.rs              # VideoStudioApp : preview navigable + panneau + jobs
│   ├── nav.rs              # zoom ancré / pan HP purs (règle -log2(span)+96)
│   └── job.rs              # worker plan→render→assemble, manifest en mémoire
├── video/               # Pipeline vidéo zoom (G12, archi DeepDrill)
│   ├── mod.rs              # Manifest TOML, keyframes ×2 (spans GMP exacts), render résumable
│   ├── assemble.rs         # interpolation trilinear 2-keyframes + ffmpeg stdin / frames PNG
│   ├── spline.rs           # valeurs dynamiques (cubique monotone Fritsch-Carlson)
│   └── lighting.rs         # éclairage Lambert normal-map écran (« spatial images »)
├── fractal/
│   ├── mod.rs           # exports + default_params_for_type()
│   ├── types.rs         # FractalType, FractalParams, AlgorithmMode, …
│   ├── definitions.rs   # constantes par type + LyapunovPreset
│   ├── iterations.rs    # escape-time f64 + dispatch bytecode unifié
│   ├── cycle.rs         # raccourci des cycles EXACTS (points intérieurs, boucles scalaires)
│   ├── jitter.rs        # AA : radical_inverse + triangle + burtle_hash décorr. par pixel (port F3)
│   ├── xaos.rs          # G10.4 réutilisation pixels inter-frame (XaoS)
│   ├── gmp.rs           # précision arbitraire (rug / mpc)
│   ├── lyapunov.rs      # Lyapunov exponent
│   ├── buddhabrot.rs    # Buddhabrot / Nebulabrot / Anti-Buddhabrot (specs
│   │                    #   d'orbite + colorisation ; échantillonnage → density.rs)
│   ├── density.rs       # Échantillonneur densité : uniforme + Metropolis-Hastings
│   │                    #   (⚠️ `c` tirés sur le domaine CANONIQUE, jamais la vue)
│   ├── vectorial.rs     # Von Koch, Dragon
│   ├── orbit_traps.rs   # Point, Line, Cross, Circle
│   ├── bytecode/        # Moteur unifié Fraktaler-3 (P3.1, défaut)
│   │   ├── mod.rs          # Op, Phase, Formula
│   │   ├── compile.rs      # compile_formula + compile_hybrid_formula + formula_for_params (G4)
│   │   ├── interp.rs       # iterate_bytecode_f64 (scalaire, référence)
│   │   ├── interp_batch.rs # idem PAR LOTS, 8 voies entrelacées (path std, défaut)
│   │   ├── interp_gmp.rs   # interpréteur GMP pour orbite référence
│   │   ├── delta_form.rs   # DeltaState f64 + DeltaStateExp ComplexExp
│   │   ├── direct_dd.rs    # itération directe double-double tracée (étage f64 → GMP, Julia/Tricorn)
│   │   ├── direct_simd.rs  # étages directs dd/qd VECTORISÉS (8 pixels de front, sans branchement)
│   │   ├── bla_dual.rs     # BLA mat2 unifié via dual-numbers
│   │   ├── pixel_loop.rs   # BLA + delta-form + rebasing F3 (f64)
│   │   ├── pixel_loop_exp.rs # idem ComplexExp (tier exp, boucle FloatExp de référence)
│   │   ├── pixel_loop_scaled.rs # tier exp en « double mis à l'échelle » (défaut)
│   │   ├── pixel_loop_dd.rs # tier double-double ~106b (float128-like), opt-in
│   │   ├── pixel_loop_qd.rs # tier quad-double ~212b : pixels au-delà du dd, avant le GMP
│   │   └── bla_dd.rs       # table BLA à coefficients dd (skip sans perte 106b)
│   └── perturbation/    # Path GMP + ponts vers bytecode
│       ├── mod.rs          # render_perturbation_cancellable_with_reuse
│       ├── types.rs        # ComplexExp, FloatExp (mantisse + exposant)
│       ├── orbit.rs        # ReferenceOrbit{,Cache}, HybridBlaReferences
│       ├── orbit_par.rs    # itération GMP de l'orbite sur deux cœurs (bit-identique)
│       ├── qd.rs           # arithmétique quad-double (QD « sloppy », ~210 b mesurés)
│       ├── orbit_qd.rs     # orbite qd rejouée depuis les points de contrôle GMP
│       ├── correct_qd.rs   # étage qd de l'escalade (dc en GMP exact, plafond)
│       ├── nucleus.rs      # Atom-domain period + Newton (port F3)
│       ├── bla.rs          # BlaTable conformal (path GMP deep zoom)
│       ├── nonconformal.rs # BLA matriciel (path GMP)
│       ├── delta.rs        # iterate_pixel{,_gmp}
│       ├── series.rs       # Taylor series approximation
├── render/
│   ├── escape_time.rs    # dispatcher unique CLI↔GUI (sélection algo)
│   ├── std_reliability.rs # suivi + correction des pixels non fiables du path f64 standard
│   ├── scalar_reliability.rs # idem pour les boucles SCALAIRES hors bytecode (Barnsley, Alpha, Mandelbulb, Magnet, Sin : f64 tracé → dd → GMP)
│   ├── dd_trig.rs        # sin/cos/exp en double-double (étage dd du sinus)
│   └── tiles.rs          # G10.5 file de tuiles priorité-centre + sink streaming
├── gpu/                  # wgpu Vulkan/Metal/DX12
│   ├── mod.rs              # GpuRenderer + pipeline bytecode
│   ├── *_f32.wgsl          # paths legacy par type
│   ├── perturbation.wgsl   # kernel perturbation F3-strict f64 (SHADER_F64)
│   └── bytecode_kernel.wgsl # runtime bytecode unifié (P3.1)
├── gui/                  # FractallApp (egui)
├── color/                # 27 palettes + RGB/HSB/LCH + 15 modes
├── io/png.rs             # save/load avec métadonnées JSON
├── io/fmap.rs            # format map .fmap (G12) : canaux bruts compressés zlib
└── quality/              # fractall-quality QA suite
```

## Densité : Buddhabrot / Nebulabrot / Anti-Buddhabrot (`fractal/density.rs`)

Ces trois types n'évaluent pas un pixel : ils tirent des paramètres `c` sur le
**domaine canonique** du plan des paramètres (centre `(-0.5, 0)`, span `4×3` —
la vue par défaut), itèrent l'orbite, et projettent les points visités dans la
fenêtre. ⚠️ **La fenêtre ne définit JAMAIS où l'on échantillonne** : tirer les
`c` dans la vue (bug fermé le 2026-08-25) rendait tout zoom noir et remplissait
n'importe quelle fenêtre lointaine d'une nappe uniforme.

Deux régimes, choisis par `density::uses_metropolis` (seuil : vue < ¼ du
domaine) :

- **uniforme** — historique, non biaisé, trivialement parallèle. La fraction des
  orbites qui atteignent la fenêtre décroît comme sa SURFACE : vide dès ~×1000.
- **Metropolis-Hastings** (Boswell) — chaînes de Markov ciblant la
  CONTRIBUTION à la fenêtre. Nourrit la fenêtre à toute profondeur (vérifié
  jusqu'à ×1e8 et sur le path MPC au-delà de 1e16).

Invariants du régime MH, tous payés par un bug mesuré :
- **poids `1/f`** à la projection (`f` = points de l'orbite dans la fenêtre) :
  la chaîne échantillonne `∝ f`, sans ce poids l'image vaudrait `f·D` ;
- **noyau de mutation figé pendant la mesure** : une adaptation au taux
  d'acceptation égalise le pas DANS la fenêtre, donc annule le jacobien
  `dz/dc` — c'est-à-dire la densité elle-même (corrélation tombée à 0,06) ;
- **fenêtres d'amorçage concentriques** sur la vue et emboîtées : une
  interpolation qui déplace aussi le centre translate la fenêtre de dizaines de
  largeurs à la fin, la population s'éteint ;
- **ré-échantillonnage** des chaînes mortes à chaque barrière d'étape (~25 %
  d'attrition par octave, sinon zéro survivant en dix étapes) ;
- **cible = nombre de touches** + terme de proximité minuscule (un noyau de
  proximité de plein poids attire les orbites qui FRÔLENT la fenêtre) ;
- accumulation en **virgule fixe** (`AtomicU64`) : déterminisme indépendant des
  threads.

Verrous : `metropolis_estimates_the_same_field_as_uniform_sampling`
(corrélation > 0,85 contre un tirage uniforme à gros budget),
`metropolis_feeds_a_window_that_starves_uniform_sampling`, et
`zooming_density_types_keeps_structure` (e2e, ×10 → ×1e8).
Diagnostics : `FRACTALL_MH_DEBUG=1` (survie des chaînes par étape),
`FRACTALL_DENSITY_SAMPLING=uniform|metropolis` (forçage du régime),
`regimes_diagnostic` et `density_fill_by_zoom_diagnostic` (tests `--ignored`).

⚠️ La compression logarithmique d'affichage est **invariante d'échelle**
(rapport à un quantile haut, pas à la valeur brute) : sans quoi le contraste
dépendrait du budget d'échantillons — donc de la résolution — et du régime.

## Vidéo & maps (G12, 2026-08-11)

**Format `.fmap`** (`io/fmap.rs`) : sidecar calcul/rendu — canaux bruts du
dispatcher (iterations u32, zs 2×f64, distances f64 opt.) compressés zlib +
JSON `FractalParams` (HP incluses). Round-trip **bit-identique** (verrou ;
a nécessité serde_json `float_roundtrip` — sans la feature le parsing f64
perd le dernier ulp). CLI : `--output-map out.fmap` (avec le PNG) et
`--from-map out.fmap` (recolorisation SANS recalcul ; `--palette`/
`--color-repeat`/`--outcoloring` sont devenus des overrides optionnels pour
distinguer « explicite » du défaut).

**`fractall-video`** (binaire-enveloppe, logique `src/video/`) :
- `plan cfg.toml -p proj/` → `manifest.toml` (keyframes = ceil(log2(zoom)),
  centre fixe, spans 4/2^k **exacts** — sérialisés en expansion décimale
  COMPLÈTE, `Display` rug tronque à ~prec·log10(2) chiffres) ;
- `render proj/` → keyframes `.fmap` séquentielles via le dispatcher unique
  (`cache/xaos/tiles=None` ; pas de réutilisation d'orbite inter-échelle,
  cf. régime atom-domain). Reprise = skip par empreinte **couleur-blind**
  (changer la palette du manifest n'invalide pas les maps). ⚠️ Le canal
  `orbits` n'est PAS persisté : `assemble::video_outcoloring` REFUSE
  OrbitTraps/Wings au plan et à l'assemblage (sinon Smooth silencieux) ;
- `assemble proj/ -o out.mp4 | --frames-dir d/` → blend trilinear DeepDrill
  (curr à 1/z, next fenêtre `2·coord−0.5`, poids `z−1`), bilinéaire CPU+rayon
  (clamp-to-edge AVANT floor), ffmpeg stdin rawvideo. AA vidéo = keyframes
  supersamplées (`image.supersample`). Verrous : frame à z=1 == keyframe
  colorisée pixel-exacte, continuité au raccord, déterminisme.

**Studio graphique `fractall-video-gui`** (G12 jalon 6, 2026-08-11) :
application eframe DÉDIÉE (le générateur `gui/` n'est pas touché). Preview
navigable (molette = zoom ancré, glisser = pan, helpers HP purs
`video_gui/nav.rs`) via le dispatcher unique (2 passes ¼→pleine, cancel par
Arc neuf, cache d'orbite inter-passes) ; drag-and-drop d'un PNG fractall ou
d'un `.fmap` = adoption de la cible ; le manifest est construit EN MÉMOIRE
(`job::build_manifest` → `video::plan_from_manifest`), zéro fichier édité.
Boutons « Générer » (reprise auto) et « Ré-assembler » (recolorer/re-timer
sans recalcul ; cible = `location` du manifest SUR DISQUE, pas la vue —
inspecter une keyframe par clic ne re-planifie pas ; `job::check_assemble_only`
refuse AVANT toute écriture un nombre de keyframes ≠ ou une empreinte
keyframe 0 ≠ → renvoi vers Générer).
Hooks lib : `render_project_with_progress` / `assemble_project_with_progress`
(`&mut dyn FnMut(événement)`, cancel `Arc<AtomicBool>`, annulation = outcome
`Cancelled` PAS une erreur ; ffmpeg tué + .mp4 partiel supprimé) — les
anciennes signatures restent des enveloppes à sortie console identique.
⚠️ dimensions vidéo forcées PAIRES (`even_dims`, x264 yuv420p échoue sinon —
plancher 16 : demander 16×12 donne 16×16, piège des tests).

**Timeline studio (G13, 2026-08-13)** — panneau bas de `fractall-video-gui`,
logique pure dans `video_gui/timeline.rs`, workers dans `job.rs` :
- **Miniatures des keyframes** : cellule par keyframe 0..=n, remplies au fil
  du rendu (msg `VideoJobMsg::Thumb` = canaux BRUTS sous-échantillonnés
  nearest-centre, `thumb_channels` — l'UI colorise, donc changement de
  palette = recolorisation in-memory SANS relire les `.fmap`). Provisoires au
  lancement : keyframe 0 = mini-rendu vue pleine (itérations plafonnées
  5000, caché par type), keyframe n = copie réduite de la preview (gatée
  `view_is_project_target`). Reprise/adoption : `spawn_thumb_scan` repeuple
  depuis les maps existantes. Clic sur une miniature = la preview saute à
  cette profondeur (`span_at_keyframe`, centre du projet).
- **Ordre de rendu DICHOTOMIQUE** (`video::RenderOrder::Bisection`,
  `bisection_order` : 0, n, milieux récursifs) : la timeline se peuple à
  toutes les profondeurs tôt — sûr car keyframes indépendantes + reprise par
  empreinte (verrou `bisection_render_produces_same_maps_as_sequential`).
  CLI inchangé (`render_project_with_progress` = enveloppe Sequential).
- **Courbe de vitesse par zone** (`SpeedCurve`) : points (position keyframe,
  multiplicateur ×1/8..×8 log) édités sur la bande au-dessus des miniatures
  (double-clic = ajouter, glisser = déplacer, clic droit = supprimer),
  interpolation linéaire, compilés par `compile()` en spline temporelle
  `video.velocity` (intégration trapèze `t(p)=∫dp/v(p)` + nœud de garde
  +0.5 s pour que l'intégrateur point-milieu atteigne p=n). ⚠️ Courbe
  plate → constante Display EXACTE (verrou bit-identique). La vitesse
  n'étant consommée qu'à l'ASSEMBLAGE, éditer la courbe = « Ré-assembler »
  seulement (verrou : fingerprints de maps insensibles à la vitesse).
- **Scrubbing** : glisser sur la règle = frame vidéo interpolée dans la zone
  centrale par `spawn_scrub_worker` (MÊME code que l'assembleur :
  `colorize_keyframe` + `interpolate_frame`, cache LRU 4 keyframes
  colorisées, debounce par drain du canal ; verrou : p entier == keyframe
  colorisée pixel-exacte). Respawn du worker par empreinte
  (dossier/couleurs/génération). Toute interaction de navigation sort du
  mode scrub.
- **ETA pondérée** : régression linéaire `secondes = a + b·k` sur les durées
  mesurées (`eta_seconds`) — les keyframes profondes coûtent plus cher
  qu'une moyenne naïve ne le prédit.
- **Bandeau de divergence** : centre de la vue ≠ projet sur disque → warning
  (le zoom n'est PAS comparé : inspecter une keyframe via clic est légitime).

**Dynamiques** (`video/spline.rs`) : `video.velocity` et `[dynamics]
palette_offset` acceptent `"t/v,t/v,…"` (temps `M:S` ou s), cubique monotone ;
spline plate == constante **bit-identique** (`as_constant`). Le champ
`FractalParams::color_offset` (défaut 0.0 = `+0.0` bit-exact, goldens verts)
décale la palette en unités de cycle. **Lighting** (`video/lighting.rs`,
`[lighting] enable/alpha/beta`) : Lambert sur normale ÉCRAN (différences
finies du champ smooth-iteration) — PAS de canal dz (aurait touché toutes les
boucles pixel ; si un jour dz est exporté, brancher ici).

## Dépendances principales

| Crate | Version | Rôle |
|-------|---------|------|
| clap | 4.6 | CLI parsing (derive) |
| image | 0.25 | PNG support |
| num-complex | 0.4 | Arithmétique complexe |
| rayon | 1.12 | Parallélisme multi-thread |
| eframe / egui | 0.37 | GUI (wrapper egui, backend wgpu) |
| wgpu | 30.0 | GPU (Vulkan/Metal/DX12) |
| pollster | 1.0 | Runtime async GPU |
| bytemuck | 1.15 | Sérialisation buffers GPU |
| rug | 1.30 | Précision arbitraire (GMP/MPFR/MPC) |
| serde / serde_json | 1.0 | Métadonnées PNG |
| png | 0.18 | Accès chunks PNG bas niveau |
| toml | 1.2 | Loader paramètres (`toml/`) |
| naga (dev) | 30.0 | Validation WGSL en tests |

## Système de coordonnées

`center + span` (pas `xmin/xmax/ymin/ymax`) :

⚠️ Les réglages de `FractalParams` sont regroupés en sous-structures
(`params.perturbation.*` BLA/série/bornes, `params.color.*` palette et mode
extérieur, `params.sampling.*` jitter et état AA transitoire, `params.formula.*`
Multibrot/hybrides/opcodes/Lyapunov, `params.channels.*`
distance/intérieur/orbit traps, `params.engine.*` mode d'algorithme, bytecode,
tier dd, nucleus, GMP + précision — ce que lit `wisdom::plan`). Elles sont
**sérialisées à plat** (`#[serde(flatten)]`, noms de champs conservés) : les
PNG, `.fmap` et TOML antérieurs se relisent à l'identique — ne jamais renommer
un champ ni retirer le `flatten` sans casser cette compatibilité (verrous
`grouped_params_serialize_flat`, `legacy_png_populates_grouped_params`,
`params_serialization_surface_is_exhaustive` = liste EXACTE des 49 clés,
`params_round_trip_preserves_every_field` = toutes valeurs non-défaut, détecte
une collision de clés entre deux groupes `flatten`).

⚠️ **Raisonner par GROUPE, jamais par liste de champs** : tout site qui
énumère à la main les champs d'un groupe finit par en oublier un.
`params_for_type_keeping_preferences` (changement de type GUI : le type définit
formule/géométrie, l'utilisateur garde couleur/échantillonnage/moteur/
perturbation/canaux) et `video::map_fingerprint` (neutralise `params.color`
ENTIER, sinon un futur réglage de couleur invaliderait des heures de
keyframes) sont écrits ainsi et verrouillés par comparaison sérialisée du
groupe.

```rust
pub struct FractalParams {
    pub center_x: f64,           // f64 GPU/CPU standard
    pub center_y: f64,
    pub span_x: f64,
    pub span_y: f64,
    pub center_x_hp: Option<String>,  // HP GMP pour zooms > 10^15
    pub center_y_hp: Option<String>,
    pub span_x_hp: Option<String>,
    pub span_y_hp: Option<String>,
    pub rotation: f64,            // degrés, R = mat2(cos,-sin,sin,cos)
    pub find_nucleus: bool,       // opt-in Mandelbrot only
    pub transform_k: Option<[f64; 4]>, // mat2 row-major (F3 param.transform)
    // …
}
```

`transform_matrix()` est le point d'entrée pour le mapping pixel→c : K si
`transform_k` est `Some` (et fini), sinon retombe sur `rotation_matrix()`.
Drop-in compatible — quasi-tous les callsites consomment un tuple
`(m00, m01, m10, m11)` ou `None`.

Helpers HP-aware (`perturbation/mod.rs`) :
- `effective_spans_fexp(params)` → `(FloatExp, FloatExp)` survie f64
  underflow (zoom > 1e308).
- `effective_pixel_size(params)` → utilisé par les dispatchers de path.

## Métadonnées PNG

Chunk `tEXt` clé `fractall-params` = JSON sérialisé de `FractalParams`
(coordonnées HP incluses). API : `save_png_with_metadata()` /
`load_png_metadata()`. **Drag-and-drop** GUI restaure l'état exact.

## Dispatch rendu (`render/escape_time.rs`)

### ⚠️ Chemin de rendu UNIQUE CLI ↔ GUI (invariant)

**Le CLI et la GUI DOIVENT suivre exactement le même chemin de rendu CPU.**
Il existe **un seul dispatcher** : `render_escape_time_cancellable_with_reuse(
params, cancel, reuse, orbit_cache, xaos, tiles)`. Toute la sélection
type→algorithme (perturbation / GMP / f64 / spéciaux) et l'appel des renderers
vivent là. `xaos` (G10.4) et `tiles` (G10.5) sont les options GUI de
réutilisation inter-frame et de scheduling par tuiles — `None` partout ailleurs.

- **CLI** (`main.rs`) → `render_escape_time(params)` = thin wrapper sur le
  dispatcher (no cancel, no reuse, `&mut None` cache).
- **GUI** (`gui/app.rs`, chaque passe progressive + HQ + AA) → appelle le MÊME
  dispatcher. `orbit_cache: &mut Option<Arc<ReferenceOrbitCache>>` est threadé
  in/out pour réutiliser l'orbite référence entre passes (perturbation only).
- **`fractall-quality`** → idem (`&mut None`).

Le path perturbation passe par `render_perturbation_with_cache` (cœur commun) ;
le cache est géré DANS le dispatcher, pas par un appel parallèle côté GUI.

**Sortie TYPÉE `RenderOutput`** (G5, 2026-08-24, `render/output.rs`) : le
dispatcher renvoie une struct à champs nommés `{iterations, zs, orbits,
distances}` — plus de tuple dont on jette un canal sans s'en rendre compte.
Sémantique : canal non produit = Vec **VIDE** ; canal produit = width×height
entrées (verrou `channels_absent_unless_produced`). `orbits` : path f64 si
`enable_orbit_traps` ; `distances` : paths f64/perturbation si
`enable_distance_estimation` ; GMP/GPU/spéciaux : aucun.

**Ne JAMAIS** réintroduire une logique de dispatch dans `gui/app.rs` ou
dupliquer `render_escape_time*` : une divergence GUI/CLI = bug.

**Dispatch GPU UNIFIÉ** (2026-05-21) : `GpuRenderer::render_dispatch(params,
cancel, reuse, orbit_cache) -> Option<GpuDispatchResult>` (`gpu/mod.rs`) est le
point d'entrée unique partagé par le CLI (`main.rs`) et le thread GUI. Il calcule
`use_perturbation` (algorithm_mode + plane_transform), dispatche par type
(perturbation Mandelbrot/Julia/BurningShip, sinon shader f32), thread l'orbite
référence, et renvoie `None` quand le GPU ne peut pas → le caller fait le
fallback CPU via le dispatcher unique. **Ne plus dupliquer ce choix dans
`main.rs`/`gui/app.rs`.**

**Auto-device (G9.5)** : `wisdom::select_device(params, gpu_available) ->
Device` (`fractal/wisdom.rs`) arbitre CPU/GPU par débit benché machine
(`fractall-cli --wisdom-bench` → `~/.config/fractall/wisdom.toml`, clés dont
`gpu_perturb_f64`) SOUS garde-fou correction : GPU routé UNIQUEMENT dans la
plage deep both-perturbation (les deux devices font de la perturbation, ~1e12
à 4e37) ; JAMAIS sur les shaders std f32 (24 b = faux). Arbitrage final
`arbitrate_device` : GPU si `gpu_thr > cpu_thr · GPU_SPEED_MARGIN` (les deux
benchés). **`wisdom::gpu_lacks_features(params)`** (2026-08-23) gate
`select_device` ET `render_dispatch` : `GpuDispatchResult` n'a ni
`distances` ni `orbits` et le kernel est f64 pur → modes Distance*/
OrbitTraps/Wings, `enable_distance_estimation/orbit_traps`, `use_dd_tier`
⇒ jamais GPU (sinon Smooth silencieux / dd ignoré). Sur GPU grand public (f64 1:64) l'auto reste CPU. Consommé par
`main.rs` (`use_gpu`), `gui/app.rs` (menu « Tech: 🔄 Auto »), CLI overrides
`--gpu`/`--no-gpu`. Goldens forcent `--no-gpu`.

```
AlgorithmMode::Auto :
  - Spéciaux : VonKoch/Dragon → vectoriel ; Buddha* → densité ;
               Lyapunov → exposant ; Mandelbulb → 3D
  - Escape-time (Mandelbrot, Julia, BS, Tricorn, Celtic, Buffalo, PerpBS,
    Multibrot puiss. entière, + variantes Julia) :
      should_use_perturbation() (zoom > ~1e13) → perturbation + BLA
      should_use_gmp_reference() (zoom > 1e16) → GMP reference
      sinon                                    → CPU f64 standard
  - Perturbation incompatible avec plane_transform != Mu → fallback f64/GMP

Modes forcés : StandardF64 | Perturbation | ReferenceGmp
```

### Moteur bytecode (défaut depuis P3.1 Session E)

Si `params.use_bytecode_engine` ET `compile_formula(type, power)` réussit :

- **CPU f64 standard** : par défaut **par lots** —
  `bytecode/interp_batch.rs::iterate_bytecode_f64_batch` (2026-11-06) : les
  points à calculer d'une TUILE avancent à 8 en lockstep (chaque opcode
  appliqué à toutes les voies, voie rechargée dès qu'elle sort). Chaque voie
  fait la suite d'opérations exacte du scalaire → **bit-identique** (verrou
  `batch_matches_scalar_interpreter_bit_for_bit`), ×4,5 à ×7 sur la phase
  pixel (la boucle scalaire était bornée par la LATENCE de la chaîne z→z²+c,
  pas par le débit). Conditions : formule mono-phase, sans canal annexe.
  Sinon scalaire `iterations.rs::iterate_via_bytecode` (hybrides ; distance,
  intérieur, orbit traps via suivi de `dz`), formule compilée UNE fois par
  rendu (`standard_formula` + `iterate_point_with_formula` — `iterate_point`
  la recompilait deux fois par pixel). `FRACTALL_STD_BATCH=0` = scalaire.
  ⚠️ Toute nouvelle sémantique d'opcode doit être portée dans les DEUX
  interpréteurs (le verrou compare les deux sur toutes les familles).
  **Points intérieurs : raccourci des cycles EXACTS** (2026-11-06) — un
  point qui ne s'évade pas coûtait `iteration_max` passes. Une orbite
  attirée par un cycle devient exactement périodique en flottant : dès que
  l'état `(z, registre)` d'une voie reprend BIT POUR BIT un point de contrôle
  (Brent, test tous les 16 passes hors du corps chaud), la voie saute un
  nombre entier de cycles et finit normalement les passes restantes — même
  compte et même z final, par construction (verrou
  `cycle_shortcut_matches_full_iteration_bit_for_bit`). Vue Mandelbrot par
  défaut 1440p : phase pixels 0,140 → 0,058 s ; à 20 000 itérations ×11, à
  200 000 ×78 ; 0 % d'instructions en plus sur une vue sans intérieur.
  Même mécanisme pour les boucles scalaires des types sans bytecode
  (`fractal/cycle.rs::CycleCut` : Newton, Magnet, Multibrot non entier,
  Mandelbulb, Sin, Phoenix, Barnsley, Alpha, Pickover — Magnet Julia 720p
  6,15 → 0,06 s) et pour le path scalaire bytecode des hybrides
  (`interp::iterate_bytecode_f64_cut`, état relevé en début de cycle de
  phases ; [M,BS] 720p 0,64 → 0,10 s). `iterate_bytecode_f64` reste
  l'interpréteur de RÉFÉRENCE sans raccourci, contre lequel les verrous
  comparent. ⚠️ Ce n'est PAS un test de convergence : aucune tolérance
  (une quasi-période n'est pas un cycle). Conditions pour équiper une
  boucle : l'état comparé contient TOUT ce que lit le pas suivant (Phoenix :
  z ET y) et la boucle ne cumule rien qui dépende du nombre de passages (un
  min/max d'orbite est permis, une somme non — d'où l'absence de raccourci
  sur les canaux distance / pièges d'orbite). `FRACTALL_STD_CYCLE=0`.
  Sans objet en perturbation : les pixels intérieurs y sont déjà sautés par
  la BLA (dragon : 2,5 M itérations × 16 k pixels en 0,02 s).
- **CPU perturbation** : `perturbation/delta.rs::try_bytecode_unified_path`
  → `bytecode/pixel_loop.rs` (f64), ou au tier exp (pixel < 1e-280)
  `pixel_loop_scaled.rs` — δ = d·2^e en f64, pas FloatExp exact en repli,
  PNG identiques à `pixel_loop_exp.rs` sur les 37 cas exp du corpus, ×2-7 ;
  `FRACTALL_EXP_SCALED=0` = boucle FloatExp. La boucle rapide f64 prend son
  pas direct d'un noyau monomorphisé (`DirectKernel`) : Mandelbrot (tracé),
  Julia, Burning Ship, Tricorn — bit-identiques à la boucle générique, qui
  ne sert plus qu'aux canaux annexes et aux autres phases. ⚠️ Un noyau
  statique gagne ×1,4-1,9 ; la même boucle avec le pas bytecode non
  spécialisé ne gagne RIEN (mesuré) : le coût est le dispatch d'opcode. **Rebasing F3 strict** (`|Z[m] + δ|² < |δ|²` **OU bout de
  référence**) remplace la glitch detection Pauldelbrot legacy. Le **rebase-at-end**
  (F3 `hybrid.cc:301` : `z := Z+δ, m := 0` quand la réf est épuisée) garde les
  centres escape-time profonds sur le path perturbation au lieu de tomber en GMP
  par-pixel — c'est ce qui a débloqué la perf deep-zoom (e50 544→1.6 s, e1000
  742→0.5 s, dragon ~6 h→6.5 s à 256², cf. TODO G2). ⚠️ Lire `z_ref[m]` avec `m`
  clampé à `ref_len-1` (après un pas, `m` peut valoir `ref_len`). ⚠️ Un pixel
  qui atteint `iteration_max` PILE sur le dernier point de la réf a fini : ne
  JAMAIS le marquer `ref_exhausted` (garde `keep_iterating`, boucles f64 comme
  GMP) — sinon une vue entièrement intérieure flaggait 100 % des pixels →
  frame dd entière pour rien (vue GUI 264×156 : 5,0 → 0,41 s, 2026-11-18 ;
  verrou `pixel_finishing_on_last_reference_point_is_not_exhausted`).
  **Fallback correction** (Mandelbrot bytecode, `perturbation/mod.rs`) : quand
  `glitch_ratio > GLITCH_FALLBACK_THRESHOLD` (0.30), on **escalade au tier dd**
  (`[DD-ESCALATION]`, re-render pixel-exact GMP ~4-8× plus rapide) au lieu du
  backstop full-GMP per-pixel. ⚠️ **Plus AUCUNE résolution de glitch
  Pauldelbrot** (2026-08-25) : passe voisinage, clustering spatial, références
  secondaires et résolution récursive par itération sont SUPPRIMÉES — Fraktaler-3
  n'en a aucune trace (`grep -ri glitch` sur ses sources : zéro occurrence), le
  rebasing les rend structurellement inutiles, et elles coûtaient un tiers du
  temps de rendu du chemin legacy pour une image identique au pixel près.
  Restent : le critère par pixel `|δ|² > tol²·|Z|²` du chemin legacy (qui ne fait
  que router vers GMP) et l'escalade dd.
- **GPU** : `gpu/mod.rs::try_render_bytecode` → `bytecode_kernel.wgsl`.

Fallback legacy si : type non compilable, `--no-bytecode`, ou GMP deep zoom
+ features non encore portées sur pixel_loop.

### Nucleus finder (P1.6.a + P1.6.b, 2026-05-19)

`perturbation/nucleus.rs` : port de Fraktaler-3.1 `hybrid_period` +
`hybrid_center` + `hybrid_size` (`hybrid.cc:417,493,544`). Mandelbrot
uniquement, opt-in via CLI `--find-nucleus` ou `params.find_nucleus = true`.

Pipeline :
1. `find_period_atom_domain(cx, cy, max_iter, s, prec)` — critère atom-domain
   `|z|² < s²·|dz|²` avec `s` = échelle de vue. Plus fiable que min-|z|.
2. `newton_refine_center(…)` — Newton complexe via dual-numbers GMP,
   tracking best-seen pour retourner amélioration locale même si convergence
   stricte échoue.
3. `hybrid_size_mat2(…)` — itère `period-1` fois via dual-numbers 2D et
   accumule `b += L⁻¹`. Renvoie `size = 1/(λ²·β)` (taille canonique) et la
   matrice `K = inv(transp(b))/β` (orientation + scale). La rotation extraite
   via `atan2(K[2], K[0])` aligne le frame de rendu sur celui du minibrot —
   indispensable pour les minibrots non-axis-aligned (seahorse valley,
   flake, olbaid*). F3 `out.transform = K; out.p.transform.rotate = 0`
   (`engine.cc:208`) : on écrase de même `params.rotation` par la valeur
   dérivée de K. ⚠️ K est calculée AVEC l'orbite et transportée par
   `ReferenceOrbitCache::nucleus_transform` ; `render_perturbation_with_cache`
   l'applique au `params` effectif (transform_k + rotation ⇒ `transform_matrix`
   ET `transform_sigma1`). Avant 2026-08-23 elle mourait dans un clone local
   (golden `hybrid_mbs_nucleus_5e28` régénéré). `render_dispatch` GPU renvoie
   `None` sur `find_nucleus` (offset `center−cref` f64 + K non appliquée).

**Why** : à deep zoom, l'orbite référence escape avant `iteration_max` →
perturbation tronque. Le nucleus refine vers un centre périodique exact
dont l'orbite ne s'évade pas. Sans K, un minibrot rotated apparaît noir
ou en bandes diagonales car les pixels échantillonnent à travers les
branches voisines.

Reste pour la parité/perf F3 (cf. TODO G2/G4) : **auto-firing dd par
sensibilité** (l'échelle wisdom est livrée, cf. §Wisdom ci-dessous) et
**nucleus phase-aware** pour les hybrides. La rotation de vue est gérée au
pixel→c (CPU + GPU bytecode) ; `Op::Rot` reste un opcode CPU dormant, utile
seulement pour une rotation *par phase* (TODO G4).

### Hybrides multi-phase (G4 jalon 1-2, `bytecode/compile.rs`)

Chaîner des formules par phase, itérées cycliquement (`phases[n % len]`). CLI
**`--phases mandelbrot,burning_ship`** → `params.hybrid_phases:
Option<Vec<FractalType>>`. `formula_for_params(params)` = source UNIQUE de la
formule (hybride via `compile_hybrid_formula` si `hybrid_phases`, sinon
`compile_formula`). **Rendu par le path f64 standard** (`iterate_bytecode_f64`
cycle déjà les phases) : `select_algorithm` **force `StandardF64`** pour un
hybride, le GMP par-pixel `iterate_point_mpc` étant z²+c hardcodé (ne cycle
pas). `render_dispatch` renvoie `None` (GPU ne cycle pas → fallback CPU).
**Deep (jalons 3-4)** : dès que la perturbation est viable (`should_use_
perturbation`) `select_algorithm` route l'hybride vers **Perturbation** ;
`delta.rs::try_bytecode_unified_path` route le multi-phase (pas directs SANS BLA
+ rebasing F3, cyclant `phases[n % len]`) vers `iterate_pixel_unified_multi_
phase` en **f64** (jalon 3, zoom ~1e13–1e280) ou `iterate_pixel_unified_exp_
multi_phase` en **ComplexExp** (jalon 4, deep > 1e280, `wants_exp`) ; le tier
f64/exp est choisi par pixel (`wants_exp`), le dd reste gaté single-phase.
**Jalon 5a — références PAR PHASE (correction, port F3 `hybrid_references`)** :
un hybride genuine ([M,BS]) exige N orbites référence (la réf `p` itère
`phases[(p+i) % N]`, `orbit.rs::compute_reference_orbit_phase` →
`ReferenceOrbit.hybrid_phase_refs`) + tracking de phase dans les boucles
(invariant F3 `(phase + m) ≡ n (mod N)`, rebase ⇒ `phase := (phase+m) % N ;
m := 0`, `hybrid.cc:266-341`). Sans ça, tout rebase avec `n % N ≠ 0`
désynchronise pas pixel/référence → image garbage (prouvé : [M,BS] @3e10
uniforme). ⚠️ Un hybride porte `fractal_type == Mandelbrot` : TOUT chemin z²+c
hardcodé est gaté `!is_hybrid` (fast-path dd de l'orbite, atom-domain
`dZdC=2·Z·dz+1`, compresseur fantôme, Brent, `harmonic_candidate`) ; le cache
d'orbite discrimine `hybrid_phases` (`is_valid_for`). Le GROUND TRUTH hybride =
GMP par-pixel CYCLANT (`GmpInterpState`), PAS le f64-std (chaos : 4→85 % de
désaccord entre 300 et 2000 iters) ni `iterate_point_mpc` (z²+c). Verrous :
`[M,M] == Mandelbrot` pixel-exact — unit (f64-std) + render-level ~3e10 + exp
1e1000 — + grille GMP-cyclant (`multi_phase_perturbation_matches_gmp_per_pixel` :
[M,M] 160/160 exact, [M,BS] majorité exacte, résidu = plancher bruit f64 sur
plis abs, 69/88 dépendant de la réf — diagnostics `--ignored` truth-stability
et ref-sensitivity) + golden `mandelbrot_hybrid_burningship`. **Jalon 5b — BLA PAR PHASE (perf,
port F3 `hybrid_blas`)** : `BlaTableUnified::build_cycled(refs[p], formula, p,
…)` (pas i = `phases[(p+i) % N]`, merges phase-agnostiques) ; boucles
multi-phase : saut `tables[phase].lookup{,_fexp}` (BLA active ssi une table
par phase, sinon pas directs), n/m avancent ensemble → invariant préservé,
pas de rebase-check post-saut. Série Taylor gatée `!is_hybrid` (z²+c + son
auto-adjust misfirait). ⚠️ `prewarm_bla_entry` DOIT couvrir le multi-phase
(sinon build N-tables sous le lock global = tout le temps pixel). Mesuré
[M,M] e50 96² : 3.34 → 0.50 s (6.7×) ; e1000 exp ≈ parité single. **Jalon 5d —
atom-domain GÉNÉRIQUE hybrides (mat2, port F3 `hybrid_reference`)** :
troncature des réfs hybrides via `J' = A_i·J + I` (Jacobiens dual-numbers,
B = I car Add terminal), J en FloatExp, critère `|adj(J)·Z|² < r²·det(J)²`
(≡ critère complexe pour J conforme — [M,M] tronque au MÊME index que [M] ⟹
**[M,M] e50 == [M] à 0 px**, verrou `hybrid_mm_equals_mandelbrot_deep_f64_
e50`). Perf e50 : 0.50 → 0.38 s. **Jalon 5e — parité vs F3 NATIF** : light-toml
`phases = "…"` (loader CLI + `compare_f3.py` → blocs `[[formula]]` F3) ; cas
corpus `hybrid_mbs_smooth_e8` (f64-std, Δmean 0.0026) et `hybrid_mbs_smooth_
e13` (**perturbation multi-phase : Δ = 0.0000 PIXEL-IDENTIQUE à F3**) — juge
externe du stack G4. ⚠️ Les zones speckle hybrides BS-famille sont
non-comparables cross-engine (chaos, structure macro identique — classe G3).
**Jalon 5f — nucleus PHASE-AWARE** : `nucleus.rs::GmpDualMat2` (Jacobien mat2
GMP par opcode) + `find_{period_atom_domain,nucleus}_formula`,
`newton_refine_center_formula`, `hybrid_size_mat2_formula` — routés par
`orbit.rs` pour les hybrides, z²+c historique sinon (bit-identique). ⚠️ size :
récurrence d/dC avec +I à l'Add (sémantique z²+c validée, PAS la lettre F3
dC=0). Vérifié : [M,M]==z²+c exact ; [M,BS] genuine → satellite p=123, K
non-conforme ; `--find-nucleus` [M,M]@1e18 pixel-identique à [M]. **BLA
radius scaling σ₁(K)** (2026-07-17, débloqué par le K non-conforme du 5f) :
`transform_sigma1()` (types.rs, snap exact 1.0 si K conforme) × le rayon
`c_norm` BLA (`delta.rs::bla_c_norm`, + `c_norm_fexp` exp) — un K skewé
étire la grille δc, sans σ₁ `max|δc|` est sous-estimé → rayon de merge
sur-permissif (over-skip). No-op bit-identique hors K skewé. **Jalon 5g —
fast-path inline multi-phase** (2026-07-18) : boucles multi-phase f64/exp —
phases [Sqr, Add] steppées inline `δ' = 2·Z·δ + δ² + dc` (bitmask, bit-
identique à DeltaState{,Exp}), cache réf de phase (rechargé au rebase
seulement), tête-de-boucle chaînée + `z_land` dans la garde (mirror
single-phase). e50 [M,M] 256² : boucle pixel 2.3× (2.26→0.98 s),
pixel-identique partout ([M,M] e50/e1000, [M,BS] e13). **Op::Rot per-phase**
(2026-07-18) : `parse_opcodes_formula` (compile.rs) parse le format opcodes
F3 (`add sqr mul store absx absy negx negy rot{DEG}`, `add` termine une
phase, need_store mirroré, cos/sin **f32** parité F3) → champ
`params.hybrid_opcodes` (prioritaire sur `hybrid_phases`, PNG-roundtrip).
Entrées : CLI `--opcodes "sqr rot{30} add"`, blocs `[[formula]]` F3 natifs
dans les TOML (structurés abs/neg/power + `opcodes`), clé légère `opcodes`
(compare_f3.py). ⚠️ Prédicat UNIQUE `params.is_hybrid_formula()` (types.rs)
= hybride OU opcodes — TOUT gate z²+c hardcodé doit l'utiliser (une formule
opcodes MÊME mono-phase n'est pas z²+c). Verrous : grille GMP-cyclant rot30
160/160 exact (+ garde anti-vacuité spread) + juge F3 natif
`hybrid_rot_smooth_e10/e13` (résidu Δmean 0.03-0.05 = bruit F3 tier float,
adjugé GMP).
`fractal_type` sert la convention d'appel (Mandelbrot-like : δ₀=0, dc=pixel).

### Wisdom auto-dispatch (`fractal/wisdom.rs`, 2026-07-12 · G9.1 2026-07-15)

Source UNIQUE de la sélection **algorithme + tier + variantes** :
- `wisdom::select_algorithm(params, Device::Cpu|Gpu)` → `StandardF64 |
  Perturbation | ReferenceGmp`. Consommée par le dispatcher CPU
  (`render/escape_time.rs`, famille `perturbation_family` = M/J/BS/Tricorn),
  par `GpuRenderer::render_dispatch` (device Gpu : seuil perturbation f32 ~1e5
  vs ~1e12 CPU) et par les labels/passes GUI (`effective_cpu_mode`,
  `will_use_perturbation`). Couvre modes forcés, fallback plane≠Mu, Auto.
- `wisdom::number_tier(params)` → `F64 | Exp | Dd` (ordre dd demandé > exp
  `pixel_size<1e-280` > f64), consommée par `bytecode_path_label` ET le dispatch
  d'exécution (`try_bytecode_unified_path`).
- `wisdom::variants(params)` → `{compression, harmonic}` : partie STATIQUE des
  prédicats de routage. Compression : opt-in `FRACTALL_COMPRESS_REF`
  (`compression_active`, per-pixel-safe : gate d'abord). **Harmonic : routé
  AUTO par défaut (G9.3)** — `FRACTALL_HARMONIC_LA` tri-état (unset/`auto` →
  Auto ; `1|lla|mla` → forcé ; `0|off|bla` → kill switch) ; candidat
  `harmonic_candidate` (per-render), décision finale au build de l'entrée
  cache BLA : probe `detect_period0` + `route_harmonic_auto` (route si
  `1 ≤ period0 ≤ 100`, calibré corpus — gagne 1.2-5.9× jusqu'à p78, perd dès
  p112 ; la longueur d'orbite est hors de cause : super_dense p9/695 k gagne
  1.74×). Per-pixel : routage sur la PRÉSENCE de la table dans l'entrée.

`wisdom::plan(params)` / `plan_for(params, device)` renvoie un `WisdomPlan`
inspectable (device + algorithme + tier + variantes + **débit benché machine**
+ exposant/mantisse requis F3-style + précision GMP orbite) ; `FRACTALL_WISDOM=1`
logue la ligne `[WISDOM]` (`bench=` = iters/s mesurés, `-` si pas de fichier
wisdom). **Benchmarks machine (G9.2)** : `fractal/wisdom_bench.rs` +
`fractall-cli --wisdom-bench` → `~/.config/fractall/wisdom.toml` (override
`FRACTALL_WISDOM_FILE`) ; débit effectif par technique (cpu_std_f64,
cpu_perturb_{f64,exp,dd}, gpu_std_f32) sur rendus réels, modèle F3
`wisdom.cc:393`. Consommateur d'arbitrage device = jalon G9.5. Modèle F3 (`wisdom.cc:240` /
`render.cc:219`) : un type de mantisse `M` / exposant `E` est viable si
`req_exp+16 < 2^E/2` ET `req_prec < M` ; pour une frame **centrée**
`req_prec ≈ log2(hypot(w,h))` (~8-13 b), donc **f64 (53 b) suffit toujours sur la
mantisse** — l'escalade f64→exp se fait sur l'**exposant** (profondeur), jamais
la mantisse (vérifié : à 1e300, `req_exp=1003` mais `req_prec=8`). Le tier **dd
(~106 b)** reste opt-in (`--dd-tier`) : son besoin vient d'une sensibilité pixel
non captée par un détecteur cheap fiable (proxy `cbits` réfuté, cf. TODO G3).
Seuils calibrés **préservés** (271 unit + 24 golden pixel-exact + sweep-lock).

#### Décision « expérience utilisateur » du sage (`wisdom_iter`)

Module ré-exporté par `fractal::wisdom` (point d'entrée unique des choix
automatiques). Aucun champ `FractalParams` : « auto » est un état CLI/GUI,
seule la valeur résolue est stockée. ⚠️ Un sélecteur de mode de navigation
piloté par le sage (`wisdom_nav`, « Nav: Auto/Continuous/Step », coût mesuré
des rendus) a existé du 2026-11-17 au 2026-11-18 : RETIRÉ sur décision de
l'utilisateur (choix jugé inadapté, menu aussi) — retour à la case « XaoS »
(off par défaut). Ne pas le réintroduire sans qu'il le demande.

- **`wisdom_iter`** — nombre d'itérations automatique. Critère : un plafond
  `C` suffit si la fraction de pixels dont la classe évadé/plafonné change
  entre `C` et `4C` reste sous **`TARGET_CHANGE` = 0,2 %** — extrapolée de la
  queue de la distribution des évasions (loi de puissance sur le surplus
  `it − m`, `m` = quantile 1 %), sans rendu supplémentaire. Couche par image
  (`IterationStats::from_iterations` + `assess_iterations`, GUI) : croissance
  seulement au-delà de **`GROW_THRESHOLD` = 0,5 %** (vise alors 0,2 %),
  réduction seulement si `C/4` suffit déjà ; jamais sous 1,25 × le quantile
  99,9 % observé. Sonde ponctuelle (`auto_iterations_with(&params, &cancel,
  AutoIterOptions::default(), &mut orbit_cache)`, CLI + démarrage GUI) :
  **4096 pixels** au rapport d'aspect, CPU par le dispatcher unique, ≤ 7
  tours, plafond max 10 M. Départ (`starting_iterations`) : défaut du type,
  relevé à `250·log10(zoom)` au-delà de 1e10 (zoom = 4/span),
  borné à 250 000. Rien ne s'évade : ×4 au moins 2 fois puis tant que le tour
  suivant tient dans **8 s** de budget, sinon on rend le départ avec
  `no_escape`. Précision de l'orbite de la sonde alignée sur la vue PLEINE
  taille ⇒ le cache d'orbite sert au rendu final (verrou
  `probe_orbit_cache_is_reusable_by_final_render`). Types applicables
  (`auto_iterations_applicable`) : familles Mandelbrot/Julia bytecode (M, J,
  BS, Tricorn, Celtic, Buffalo, PerpBS, Multibrot + variantes Julia),
  hybrides/opcodes, Alpha Mandelbrot (+Julia), Mandelbrot Sin / Julia Sin,
  Phoenix, Barnsley (M + J) ; NON : vectoriels, densités, Lyapunov, Newton,
  Nova, Magnet, Mandelbulb, Pickover. Limites connues : vue dont rien ne
  s'évade dans le budget ⇒ départ + `no_escape` (intérieur indiscernable d'un
  plafond trop bas) ; sur vue très profonde la sonde coûte ≈ une orbite de
  référence ; `cancel` n'est pas relevé pendant la correction de fiabilité
  dd. **Queue lourde (2026-11-19)** — types « fuyants » dont l'intérieur
  apparent finit par s'évader (Barnsley, Multibrot non entier) : la masse
  par octave décroît de moins de 30 % (`HEAVY_TAIL_RATIO`) alors que le
  plafond est déjà loin de la médiane des évasions (`IN_TAIL_FACTOR` — sans
  cette condition seahorse 1e7 serait « lourd » à son plafond de départ).
  L'extrapolation géométrique n'y vaut rien : la sonde avance par ×4
  (`HEAVY_TAIL_GROWTH`) en remesurant et s'arrête dès que la prédiction
  passe sous `GROW_THRESHOLD` ; la marge de queue observée ne relève plus le
  plafond suffisant que de ×2 (`TAIL_MARGIN_MAX_LIFT`). Barnsley Mandelbrot
  ×5 : 2,9 M itérations → 52 000 (37 s → 0,66 s à 128×80). Juge : volet
  `iter` de l'axe `wisdom` du harness. Calibration rejouable : `cargo run
  --release --example calibrate_auto_iter`. `AutoIterResult::summary_line()` = ligne anglaise.
  **Sonde INCRÉMENTALE (2026-11-19)** — un pixel évadé sous le plafond C a
  le même compte et le même z sous 4C : à partir du 2e tour seuls les pixels
  PLAFONNÉS sont recalculés (éléphants 1e7 : 4131/4131/4131/4131 pixels par
  tour → 4131/2152/845/13, sonde 7,65 → 3,94 s ; N identique sur 83/83 vues
  du corpus). Mécanisme : la carte de raffinement XaoS (`build_refine_map`)
  sur une frame source dont l'erreur vaut 0 pour un pixel évadé et 1e6 pour
  un plafonné, passée au dispatcher unique — ⚠️ ne JAMAIS écrire un second
  mécanisme de rendu partiel dans `wisdom_iter`. Vue non éligible à la copie
  (rotation, K, nucleus, AA) ⇒ tour complet. ⚠️ La sonde a besoin de comptes
  CORRIGÉS par la fiabilité : sur des comptes f64 bruts le N choisi est
  gonflé jusqu'à ×1,56 (seahorse 1e7 : 90 000 → 140 000). ⚠️ Un tour
  incrémental ne calcule que les pixels difficiles : le ratio de pixels non
  fiables (calculé sur les pixels calculés) peut y déclencher l'escalade dd
  plein cadre plus souvent — correct, seulement plus cher. Verrous
  `incremental_probe_round_matches_full_round_bit_for_bit`,
  `incremental_probe_matches_full_probe`. `FRACTALL_AUTO_ITER_INCREMENTAL=0`
  (A/B), `FRACTALL_AUTO_ITER_DEBUG=1` (une ligne `[AUTO-ITER]` par tour) ;
  outils `examples/probe_bench.rs`, `calibrate_auto_iter -- --probe-ab`.
  **Câblage CLI** (`main.rs`) : `iterations_choice` (ordre de résolution, cf.
  §CLI) puis `apply_auto_iterations` AVANT le choix du device et l'export F3 ;
  le cache d'orbite de la sonde est passé à `render_planned` ; un TOML qui
  alignait `max_perturb_iterations`/`max_bla_steps` voit ses caps suivre la
  valeur auto. `[WISDOM]` gagne en FIN de ligne `iter=N iter_src=user|source|
  default|auto` (`wisdom::set_iteration_source`, global de processus déclaré
  par le CLI ; `-` ailleurs) — `_RE_WISDOM` ne lit que le préfixe.
  ⚠️ Toute invocation CLI à résultat déterministe (golden, baseline, icône)
  doit porter `--iterations` ou une source qui le fournit.
- **Câblage GUI** (`gui/app.rs`, décisions pures testées dans `gui/sage.rs`) :
  case « XaoS » (`xaos_nav_mode`, zoom continu au bouton maintenu, off par
  défaut) et case « Iter: Auto »
  (coupée par une valeur tapée, un PNG ou un TOML déposé). ⚠️ La GUI ne doit
  PAS lancer l'évaluation par image (`assess_iterations`) sur les frames
  d'écho XaoS (pixels copiés : la distribution ne décrit pas la vue) —
  `sage::iterations_assessable` : passe finale CPU pleine résolution, sans
  copie approximative, rendue à la limite courante. Garde-fous : au plus 3
  `Grow` de suite par vue (`AutoGrowGuard`), jamais pendant un geste ; un
  `Shrink` attend le rendu SUIVANT (`pending_iter_shrink` — l'appliquer tout
  de suite affichait une limite que l'image à l'écran n'avait pas).

### Détecteur de fiabilité par pixel (G9.6, `bytecode/reliability.rs`)

Le fast-path f64 Mandelbrot rendait des comptes FAUX sans rien signaler
(plancher de précision du δ : seahorse 1e8 max_diff 437, e50 23 px).
**Traqueur de borne d'erreur** injecté par monomorphisation (`StepObserver` ;
production = `NoObserve` ZST, verrou bit-identique
`tracked_mandelbrot_loop_is_bit_identical_and_quiet`) : borne absolue `E` sur
z propagée (pas direct `E·(2|z|+E)+γ·local`, saut BLA de `l` pas
`σ₁(A)·E+(l·ε+γ)·…`, rebase `+u·(|Z|+|z|)`) ; pixel non fiable si la borne **s'EMBALLE**
(`error_bound ≥ 1e30` ou NaN, `DeltaResult::error_bound`).
⚠️ Calibration (étude vs GMP, `quality::reliability_study`) : TOUS les pixels
faux ont une borne emballée ; un seuil z-space MODÉRÉ flagge 1-8 % (rejeté) ;
la normalisation par dz/dc (shadowing d'origine) ne départageait aucun pixel
faux → supprimée (gain de vitesse) ; **ne PAS linéariser** la propagation
(0/33 détecté : c'est le terme E² qui révèle la perte de détermination).
**Escalade par défaut** (`FRACTALL_RELIABILITY=escalate|observe|off`) :
pixels flaggés recalculés au **tier dd** sur ces seuls pixels
(`correct_pixels_dd`, orbite dd reconstruite depuis la partie basse
`z_ref_lo` capturée avec l'orbite primaire — jamais re-calculée ; repli
GMP), frame entière dd au-delà de 50 % seulement (le pixel par pixel est
toujours moins cher en dessous). Tier **exp** : traqueur
`ShadowTrackerExp` **actif par défaut** depuis 2026-11-06
(`FRACTALL_RELIABILITY_EXP=0` le coupe) — à 256² sans lui e401 6, olbaid2 48,
e1298 18 pixels faux, 0 avec ; porté dans la boucle scaled
(`direct_scaled` / `bla_scaled`, même borne au bit près). ⚠️ La boucle
scaled est bornée par le DÉBIT d'instructions (IPC ~7, ~250 instr./tour),
pas par la latence : chaque instruction du traqueur se paie. Son coût
(+42 % → +24 % d'instructions, 2026-11-06) était le chemin général du pas
direct (3-4 puissances de 2 et les gardes à chaque pas), PAS la racine (la
retirer : 0 %) ni σ₁ : puissances de 2 en cache par (cadre de δ, cadre de la
borne) — la borne vit le plus souvent AU-DESSUS du cadre de δ —, |d|² du
tour réutilisé, σ₁(A) précalculé par nœud (`BlaMultiStepExp::sigma_m/
sigma_e`). Verrou `light_tracker_matches_reference_bit_for_bit` (ancien
traqueur conservé verbatim dans `pixel_loop_scaled_track_ref.rs`). (Les erreurs exp
d'entrée — ε BLA 2⁻²⁴ → 2⁻⁵³, span f64 **sous-normal** 1e308-1e324,
`effective_spans_fexp` exige `is_normal` — avaient été corrigées avant.)
⚠️ **Trois règles payées par des pixels faux (2026-11-06)**, valables pour
tout traqueur : (1) la troncature d'un saut BLA de `l` pas vaut **l·ε**, pas
ε (chaque pas fusionné abandonne jusqu'à ε : avec ε seul la borne
sous-estimait l'erreur réelle ~40× — e227 384 pixels faux non flaggés à
256², golden_spider 169, dragon 36 à 128²) ; (2) une borne emballée doit
être **collante** (au tier exp l'exposant i32 débordait et la borne
retombait à « fiable ») ; (3) un pixel qui s'évade quand `E ≥ ||z|−R|`
n'est pas déterminé (`escape_undetermined`, actif en exp, opt-in f64
`FRACTALL_F64_MARGIN=1`).
**Tier dd : même garde-fou un niveau au-dessus** (`ShadowTrackerDd`, unité
2⁻¹⁰⁴) — le dd n'est PAS la vérité : seahorse 1e6, pixel (117,17), GMP
convergé 1598, dd 1838. Pixel dd dont la borne s'emballe → second essai
SANS BLA (borne réduite à l'arrondi dd ; évite d'envoyer au GMP des pixels
que le dd rend juste) → GMP par pixel. Vaut en correction, en frame dd
(`--dd-tier`, `[DD-ESCALATION]`) et pour les pixels flaggés du kernel GPU
f32. `FRACTALL_RELIABILITY_DD=0`. ε de la BLA dd = **2⁻⁸⁰, APPARIÉ au
traqueur dd** : nu, il laissait des pixels dd faux (troncature l·ε) ; à
2⁻¹⁰⁶ la table ne saute plus rien (+24-35 %) ; sous le traqueur un saut
dont la troncature compte emballe la borne et le pixel est refait sans BLA
— mêmes itérations que 2⁻¹⁰⁶ sur 75 cas du corpus. Ne jamais changer l'un
sans l'autre. **Plafond de l'escalade GMP** : tout ou rien, ssi le
nombre de pixels dd non fiables ≤ max(16, 0,1 % × pixels)
(`gmp_escalation_allowed`, `FRACTALL_RELIABILITY_GMP_MAX_FRACTION`, ≥ 1 =
illimité) ; au-delà, valeur dd de meilleur effort + avertissement stderr
inconditionnel — lethal_weapon (1,6 M itérations, 17 % de pixels au-delà du
dd) prendrait des heures sinon. **Tier quad-double (~212 b) entre le dd et
le GMP** (2026-11-06, `pixel_loop_qd.rs` + `perturbation/{qd,orbit_qd,
correct_qd}.rs`) : chaîne complète **f64 signalé → dd tracé → dd sans BLA →
qd → GMP sous plafond**. Boucle en pas directs sans BLA, orbite qd rejouée
EN PARALLÈLE depuis les points de contrôle GMP de la capture paresseuse, dc
construit en GMP exact (K, jitter AA et `centre_vue − cref` compris), son
propre plafond tout ou rien (`FRACTALL_QD_MAX_FRACTION`, défaut 1 % des
pixels, plancher 16), domaine |dc| ≥ 2⁻⁸⁴⁰ (au-delà : GMP). Itérations
identiques au GMP pur sur e227 (7/7), golden_spider (16/16), rug,
seahorse 1e6 ; e227 256² −25 % de cycles, golden_spider −20 %.
`FRACTALL_QD=0`. ⚠️ **Piège payé par des pixels faux** : le traqueur qd doit
compter l'arrondi de la RÉFÉRENCE (`2^-(prec−2)·|Z[m+1]|` par pas) — le
pixel hérite de l'arrondi de la pseudo-orbite ; sans ce terme, sur
lethal_weapon (orbite 290 b), 376 pixels étaient déclarés fiables dont 245
faux vs GMP 580 b (avec : 91 fiables, 91 justes). ⚠️ À la précision de
formule le GMP pur donne des ITÉRATIONS justes mais un z final non convergé
sur les pixels au-delà du dd (golden_spider : 605 b ≠ 1210 b == 2420 b) :
ne jamais juger ces pixels sur `zs`. ⚠️ Ne PAS ajouter au traqueur qd un
plancher « arrondi de la conversion en qd » : la référence arrondie
Ẑ = Z + ρ n'injecte que le terme croisé `2·δ·ρ` (∝ δ, déjà couvert par
γ·|2Z|·|δ|) — c'est pourquoi une référence f64 suffit en perturbation f64 ;
essayé, la borne de tous les pixels qd justes s'emballait. **Entrée dans le tier dd = `wisdom::
dd_tier_applicable`** (Mandelbrot ET non hybride/opcodes) partout —
`--dd-tier` sur `--opcodes` rendait un z²+c nu.
⚠️ `z_ref_lo` est à l'échelle de la MANTISSE de `z_ref` (ComplexExp), pas du
f64 : |Z| ~ 1e-1200 près de la période atom-domain. Correction dd : K (rotation)
appliquée au dc dd. Coût
(seahorse 1e8 512²) : boucle 2,0 → 2,5 ns/iter (traqueur gardé en LOCAL —
derrière `&mut` il coûtait ×3), total 0,72 → 1,10 s. Mesuré vs GMP :
seahorse WARN max_diff 437 → PASS ; e30 5/5, e50 17/18 pixels faux
détectés, 0 faux positif sur e13/e17/e18/e100. Suivis : fast-path f64
Mandelbrot, Harmonic LA, tier exp Mandelbrot (boucle scaled), tier dd.
**Julia : suivie par défaut** (fast-path f64, 2026-11-06 — 0,7 % de pixels
faux non signalés sans elle, jusqu'à 812 itérations d'écart) ; pas de tier
dd pour cette famille : pixels signalés recalculés en GMP pur
(`iterate_point_mpc`) sous le plafond 0,1 % OU un budget de 5 % des
itérations de la frame (`FAMILY_GMP_ITER_BUDGET`), valeur f64 + avertissement
au-delà. Burning Ship / Tricorn : opt-in `FRACTALL_RELIABILITY_FAMILY=all`
(107 scènes sondées, aucun pixel faux adjudicable). **Sans suivi** :
hybrides, DE, Julia au tier exp.
⚠️ **Path f64 STANDARD : suivi depuis le 2026-11-18** (chantier de
correction n° 1). Il était faux sans rien signaler sur les bords chaotiques à
fort plafond, bien avant d'être à court de mantisse (seahorse 1e10, 2500 it.,
256×160 : 7,8 % de pixels faux vs GMP pur) — l'arrondi de l'itération
directe est amplifié par le produit des `2|z|`. Le noyau par lots porte un
traqueur (`interp_batch::iterate_bytecode_f64_batch_tracked`, `const TRACK`,
bit-identique) : `E' = E·(2√n + E) + 4u·(n + |c|₁)`, signalé si `E ≥ 1e30`
OU, à l'évasion, `E ≥ |z| − R` ; les SEULS pixels signalés sont recalculés
par `render/std_reliability.rs` — Mandelbrot / Julia par
`perturbation::recompute_pixels_cpu` (perturbation tracée autour du centre
HP → dd → qd → GMP), Mandelbrot/Julia tournés et, en opt-in
`FRACTALL_RELIABILITY_STD=all`, BS / Tricorn / Celtic / Buffalo / PerpBS par
GMP pur sous le plafond des familles (0,1 % des pixels OU 5 % des
itérations). Mesuré (axe `truth`, volet `std:*`, 18 scènes 256×160) :
5 FAIL + 4 WARN → 18 PASS, pixels faux seahorse 1e10 3179 → 3, éléphants
1e10 1390 → 2 ; coût du traqueur seul +15-23 % (vues sans grand bord) à
+31-35 % (spirales) sur la phase pixels, total avec correction ×1,4 (défaut
720p) à ×2,5 (éléphants 1e7 720p), contre ×6 à ×90 pour un re-routage en
perturbation. ⚠️ Un pixel corrigé est mal conditionné : son z final (pas son
compte) dépend de la référence — deux vues dont les centres diffèrent au
17e chiffre (chaîne HP vs f64) ne sont plus égales au bit près sur `zs`.
Non suivis : formules à `Mul`/`Rot` (Multibrot, opcodes), hybrides et canaux
annexes (chemin scalaire), plan ≠ Mu, graine non nulle, types sans bytecode.
**Tricorn suivi par défaut + étage double-double DIRECT (2026-11-19)** —
l'axe `types` du harness a montré 2,5 % de pixels faux sur un bord Tricorn
×4096 (juge convergé). Tricorn est désormais traité comme Mandelbrot/Julia :
traqueur du path standard, pixels signalés recalculés par la perturbation
tracée (`family_tracking_enabled`), puis — nouveau, et valable aussi pour
Julia — par `bytecode/direct_dd.rs` : l'itération directe en double-double
(~106 b, 10-15 ns/itération) avec sa propre borne (`E_MAX`, marge
d'évasion), avant le GMP pur (150 ns/itération). Conditions
(`direct_dd_phase`) : formule mono-phase à opcodes simples, plan Mu, pixel
≥ 1e-22, PAS le Mandelbrot non hybride (son tier dd de perturbation passe
avant). Sortie identique au chemin tout-GMP ; Tricorn bord 128×80 :
correction 0,17-0,27 s → 0,033 s. Burning Ship reste en opt-in : son bord
est `CHAOS` (98 % de pixels signalés, juge 256 b ≠ 512 b).
**Étage dd direct dans la correction `GmpCapped` du path standard
(2026-11-19)** — `perturbation::direct_dd_pixels` (partagé avec
`recompute_pixels_cpu`) passe AVANT le GMP pur pour Mandelbrot/Julia tournés
et les familles à pliage ; le plafond GMP ne s'applique plus qu'au reste
(seahorse 1e7 tourné 30° : 310 px faux sans suivi → 0, correction 0,77 →
0,065 s). Les opcodes abs n'exigent aucun test au pli : abs est
1-lipschitzien, la borne absolue lui survit (verrou
`direct_dd_stage_agrees_with_gmp_on_fold_families`). **Suivies par défaut** :
variantes Julia de Burning Ship / Tricorn / Buffalo / Perp. BS (Perp. BS
Julia bord 9 px faux → 0). **Mode `StdCorrection::Budgeted`** (Perp. BS
Mandelbrot-like, accueil 2 px faux → 0) : rien n'est corrigé au-delà de 2 %
de pixels signalés (`FOLD_FLAG_BUDGET`), seuls les pixels ÉVADÉS sont tentés
(86 % des signalés sont des orbites chaotiques bornées que le dd ne garantit
jamais : les tenter coûtait ×3 pour rien), GMP sous le seul plafond en
pixels (le budget de 5 % des itérations y coûtait ×4). Restent en opt-in
`all` : BS / Celtic / Buffalo Mandelbrot-like (accueil 5-10 % de signalés,
bord CHAOS) et Celtic Julia (intérieur à attracteur chaotique borné : borne
emballée sur 52 % du bord sans compte faux, ×7). ⚠️ Le z final d'un pixel
corrigé par le dd n'est garanti qu'à `max(E_MAX, 0,01·|z|)` : un verrou ne
compare à l'octet que les ITÉRATIONS.
**Boucles scalaires suivies (`render/scalar_reliability.rs`, 2026-11-19)** —
les types hors bytecode n'avaient aucun suivi. Barnsley (Julia et
Mandelbrot) est le premier : la carte `z' = (z ∓ 1)·c` dilate de |c| par pas
(f64 épuisé vers 130 itérations) et sa branche `Re z ≥ 0` bascule sur un
bit — 5 % de pixels faux sur un bord Barnsley Julia, 1 % sur la vue
d'accueil. Boucle f64 tracée BIT-IDENTIQUE à `iterations::barnsley_*`
(verrou `tracked_barnsley_is_bit_identical_to_production_loop`), pixels
signalés refaits en double-double, reste en GMP sous le budget `GmpCapped`.
⚠️ Dans une boucle tracée : pas de racine carrée par itération (majorants
L1, tests sur `norm_sqr`) et suivi INERTE une fois le pixel signalé — la
première version coûtait ×2,8 sur les vues chaotiques, celle-ci +30 %.
Interrupteur : `FRACTALL_RELIABILITY_STD=0`. Étendre à un type = sa boucle
tracée f64 + dd et une entrée dans `scalar_reliability::plan`.
**Alpha Mandelbrot, Mandelbulb, Magnet Julia suivis (2026-11-19)** — bords
×4096 : 116 / 101 / 8 px faux → 0 (juge convergé). Coût : traqueur seul
+22 à +42 % (Alpha au-dessus de la cible), rendu de bord ×1,9-2,3 (17-22 %
de pixels signalés, dont un sur cinq réellement faux), accueil dans le
bruit. Étage dd à voies entrelacées monomorphisées (41-44 ns/it.) avec
raccourci des cycles exacts (Magnet : 40 → 1,3 ns/it.). ⚠️ Règles payées
cette nuit : (1) le multiplicateur d'un pas est le module de la dérivée
EXACTE (Alpha `|1+2m|`, pas `1+2|m|` qui signale 100 % des pixels ; Magnet
`|z−r|/|q|`) — l'inégalité triangulaire détruit les annulations ; (2) ne
jamais élever à une puissance un majorant sans racine de |z| (×1,083 devient
×1,74 sur |z|⁷) : suivre `g = E²`, multiplicateurs au carré exacts par
`norm_sqr`, sommes par `(x+y)² ≤ (1+δ)x² + (1+1/δ)y²` ; (3) une borne NON
linéarisée reste rigoureuse à grande erreur : seuil 0,25 au lieu de 1e-3,
c'est le test d'évasion qui décide ; (4) la chaîne de dépendance de `g` ne
doit pas être plus longue que celle de `z` (Alpha +114 % → +40 %) ; (5) le
raccourci de cycle de l'étage dd compare AUSSI la borne ; (6) l'addition dd
« rapide » a une erreur relative aux OPÉRANDES, pas au résultat. **Phoenix
suivi depuis le 2026-11-20** (`render/scalar_phoenix.rs`) : toute borne en
modules du couple (z, y) a un rayon spectral > 1 dès |z| > ¼ et s'emballe
sur 100 % du bord ; la borne est donc ELLIPSOÏDALE — matrice hermitienne
2×2 P avec e·eᴴ ⪯ P, transportée par le jacobien exact
`P' = (1+δ)·J·P·Jᴴ + (1+1/δ)·|w|²·e₁e₁ᴴ`, phases comprises —, même borne
en double-double (4 voies, raccourci de cycle sur z, y ET l'ellipsoïde),
puis GMP. ~100 vues de bord : 283 px faux → 0 ; signalés 7,6 % au bord
×4096. Coût : traqueur +55 % (au-dessus de la cible, boucle bornée par le
débit), 720p accueil +24 %, bord +47 %. ⚠️ Ne pas linéariser le terme `a²`
(100 % du bord signalé) ; majorer |y| par une constante : 7,6 → 9,3 %.
⚠️ **Dans un test, `ReferenceGmp` sans `use_gmp` n'est PAS un juge GMP**
pour un type hors famille perturbation (le rendu repasse par le path f64
suivi) : `tracked_scalar_edges_match_gmp` comparait la production à
elle-même jusqu'au 2026-11-20 (corrigé, ~50 s, tous types à 0 px faux).
Verrous
`tracked_scalar_loops_are_bit_identical_to_production_loops`,
`dd_lanes_match_single_pixel_loop`, `tracked_scalar_edges_match_gmp`.
**Magnet Mandelbrot, Mandelbrot Sin, Julia Sin suivis (2026-11-20)** — les
juges GMP accélérés ont montré 11 % de pixels faux sur les bords Magnet
Mandelbrot (273 px) et Mandelbrot Sin (294 px) : 0 après suivi. Magnet M =
la carte de Magnet Julia avec `s` = le pixel (`magnet_f64` commune). Sinus :
multiplicateur = dérivée EXACTE `|cos ẑ|² = cos²x + sinh²y`, reste
`½E²·cosh(|y|+E)` ; composantes par `iterations::csin_parts` (verrou
`csin_parts_rebuild_csin_bit_for_bit`) ; étage dd par `render/dd_trig.rs`
(sin/cos/exp double-double, constantes GMP en `OnceLock`, ~340 ns/it. contre
~15 µs en GMP ; verrou `dd_trig_matches_mpfr_within_bounds`). Traqueur seul
+1 à +26 %. ⚠️ **Le prix est au bord** : Mandelbrot Sin 256×160 1,8 → 13,9 s
(29 % de pixels signalés, 1,8 % au-delà du dd → GMP), Magnet M 0,65 → 3,2 s
— les pixels envoyés au GMP sont nécessaires (15-30 % divergent en dd mené à
son terme). ⚠️ Règles : (1) une borne qui décroît vers un point fixe nul
passe en SOUS-NORMAUX (assistance micro-code, +165 % sur l'intérieur de
Mandelbrot Sin) → plancher sur le facteur additif (`SIN_FLOOR`) ; (2) un
majorant `cosh y` du multiplicateur ne s'annule pas près de Re z = ±π/2 et
emballe l'intérieur ; (3) une sortie sur SEUIL (pôle de Magnet,
`|q|² < 1e-24`) se juge sur |q̂| + erreur, pas sur l'erreur seule (verrou
`magnet_pole_exit_is_judged_on_q_plus_error`) ; (4) « dd signalé ≠ GMP » ne
prouve rien : un pixel signalé s'arrête au pas du signalement — pour juger
le dd, le mener à son terme sans borne (`dd_untracked`) ; (5) l'étage GMP qui
arbitre le dd/qd ne descend JAMAIS sous 256 b (`GMP_FLOOR_BITS` ; avec
`precision_bits` abaissé il tombait à 128 b : Mandelbulb 1667 « corrigé » au
lieu de 1492 — même plancher dans `std_reliability`).
**Étage quad-double des boucles scalaires + voies vectorielles du sinus
(2026-11-20)** — `render/scalar_qd.rs` : étage qd tracé (unité 2⁻²⁰⁰) entre
le dd et le GMP pour Alpha, Mandelbulb, Magnet, Sin, Barnsley ; point du
pixel en GMP exact puis converti. `render/scalar_lanes.rs` : étages Sin
dd/qd et Magnet qd sur `DdV`/`QdV` de `direct_simd` ; `dd_trig` réécrit
(e^a et e^−a d'un seul polynôme, ln 2 façon Cody-Waite, Estrin). Bord
256×160 : Mandelbrot Sin 11-16 → 5,6 s (GMP 6,5-9,6 s → 0), Magnet M
≈ ×1,5, comptes d'itérations inchangés. ⚠️ Règles : (1) un étage plus
précis avant le GMP ne tourne QUE si le GMP aurait tourné (`gmp_would_run`
— sinon scène CHAOS : étage cher payé pour des pixels qui restent en f64) ;
(2) en arithmétique « sloppy » l'arrondi d'une somme se compte sur ses
OPÉRANDES ; (3) réduction `m·ln 2` sans Cody-Waite : erreur ∝ m·2⁻¹⁰⁶,
borne fausse dès |y| ≳ 50 ; (4) les voies vectorielles ne paient que si le
pas est long (Sin oui, Magnet dd non : 26 → 46-52 ns/it.) ; (5) un pas
vectoriel n'a pas les bits du scalaire : ses constantes d'erreur se
vérifient contre MPFR pour cette arithmétique-là. Reste : le traqueur
Magnet est pessimiste (94 % des signalés déjà justes en f64).
`FRACTALL_SCALAR_QD=0`, `FRACTALL_SIN_LANES=0`. Verrous
`qd_stage_agrees_with_gmp_beyond_dd`, `qd_trig_matches_mpfr_within_bounds`,
`sin_lanes_trig_matches_mpfr_within_bounds` ; banc `--ignored
bench_correction_stages`.
⚠️ **L'emballement ne voit pas un bord PARABOLIQUE** (2026-11-19) : près
d'un point parabolique l'erreur ne s'amplifie pas exponentiellement, elle
déplace le temps de transit du goulot, et l'orbite s'évade quelques pas plus
tard — la borne dd vaut alors une fraction de |z|, très loin de 1e30. Vallée
des éléphants 1e7 au plafond de la sonde auto (60 000) : 1,17 % de pixels
faux, tous des valeurs dd « fiables ». D'où le test d'évasion relatif
(`E ≥ 0,01·|z|` ⇒ second essai sans BLA, puis qd / GMP) dans
`correct_pixels_dd`, **en vue peu profonde seulement**
(`shallow_dd_escape_rel`, pixel ≥ 1e-13) : en profond il ne change aucun
pixel du corpus et coûte jusqu'à ×2. Le plafond FIXE (2500) de l'axe truth
ne pouvait pas le voir : c'est la scène à plafond auto qui l'a montré.
**Vue peu profonde : corriger par itération DIRECTE dd puis qd, jamais par
perturbation dd (2026-11-19)** — pixel ≥ 1e-13, Mandelbrot, dans
`recompute_pixels_cpu` et la correction plein cadre. La référence au centre
s'y évade tôt (1 767 points aux éléphants 1e7) : chaque pixel rebase sans
cesse, la BLA ne saute rien, et la perturbation dd coûte ~7× l'itération
directe dd. Chaîne : perturbation f64 tracée → **dd direct** → **qd direct**
(`direct_dd.rs` générique sur `DirectNum`, `iterate_direct_qd`) → GMP.
Éléphants 1e7, 60 000 it., 256×160 : GMP 1 848 px → 63, correction ÷2,5-3,
pixels faux 12 → 6 ; perturbation FORCÉE sur la même vue 4,2 % de pixels
faux → 6 px. Le qd direct règle ~98 % de ce que le dd refuse pour ~2,5×
moins cher que le GMP 256 b ; pas de plafond en fraction de pixels sur le
path standard (il ne remplace que du GMP), budget `SHALLOW_QD_ITER_BUDGET`
(Σ itérations ≤ 0,5 × frame) en perturbation plein cadre. `GmpCapped` gagne
le même étage qd. ⚠️ Au bord parabolique à fort plafond le traqueur NE
surestime PAS : 73 % des pixels signalés (12 944 / 17 747) sont réellement
faux en f64 brut — le levier est le prix par pixel, pas le détecteur.
**Référence de correction du path standard dans le cache d'orbite**
(`correct_std_pixels_cached`) : relue au rendu final après sonde et entre
passes GUI (pas entre tours de sonde : `is_valid_for` exige un plafond ≥).
Verrous `direct_qd_stage_agrees_with_gmp_beyond_dd`,
`std_correction_reference_goes_through_the_orbit_cache`,
`parabolic_edge_at_auto_cap_matches_gmp` (étendu à la perturbation forcée).
**Étages directs VECTORISÉS (`bytecode/direct_simd.rs`, 2026-11-19)** —
8 pixels de front, chaque composante dd/qd est un vecteur de 8 flottants,
voie rechargée dès qu'elle sort. dd : 27,7 → 8,0 ns/it. (natif), 33,5 →
15,4 (baseline x86-64 des releases) ; qd : 121,6 → 32,6 / 150,8 → 74,9.
Éléphants 1e7 60 000 it. 256×160 : correction 7,65 → 2,73 s. dd identique
au bit à `iterate_direct_dd` ; le qd a une renormalisation SANS branchement
(bits de z ≠ ancien `Qd` quand une composante est nulle, borne U_QD = 2⁻²⁰⁶
vérifiée contre rug) ; l'ancienne boucle reste en témoin
(`iterate_direct_qd_branchy`). ⚠️ Des voies entrelacées sur un dd/qd
scalaire à branches ne donnent que ×1,5 : le gain vient de la
VECTORISATION, donc d'une arithmétique sans branchement. ⚠️ Sans `fma` à la
compilation (releases x86-64 baseline), `f64::mul_add` est un APPEL libm, ni
inliné ni vectorisé : `two_prod` passe par `cfg!(target_feature = "fma")`
sinon découpage de Dekker (mêmes bits, verrou `dekker_matches_fma`).
**Étage f64 sauté quand la référence est courte** (`short_reference_skips_
f64` : vue peu profonde ET référence évadée avant le quart du plafond) :
une perturbation autour d'une référence évadée tôt n'est qu'une itération
directe avec frais généraux, et son traqueur ratait un décalage d'une
itération au bord parabolique (éléphants : 6 → 2 px faux). Référence
intérieure (seahorse) : l'étage f64 avec BLA reste 3-20× moins cher, gardé.
`FRACTALL_SHALLOW_SKIP_F64=0`. Banc : test `--ignored bench_direct_simd`.
**FMA / AVX2 choisis À L'EXÉCUTION dans les binaires baseline x86-64
(`fractal/eft.rs`, 2026-11-20)** — `eft::two_prod` (partagée par `dd.rs`,
`qd.rs`, `direct_simd`) : `mul_add` en ligne si FMA à la compilation,
fonction `#[target_feature(enable="fma")]` si le CPU l'a (détection
`OnceLock`), Dekker sinon. `eft::avx2_fma_dispatch()` aiguille vers des
copies `#[target_feature(enable="avx2,fma")]` des noyaux chauds : étages
directs dd/qd, noyau par lots f64 (`interp_batch`), boucles rapides de
perturbation f64 (`pixel_loop`) — identiques au bit près (Rust ne contracte
jamais `a*b+c`), seul `unsafe` = l'appel. Baseline ramené au natif :
éléphants 4,40 → 2,20 s, seahorse lots 2,27 → 1,62 s, phase pixels e50
×1,30. ⚠️ (1) En baseline `mul_add` appelle la glibc, qui passe par
l'instruction matérielle (~4 ns) : Dekker scalaire est PLUS LENT sur un CPU
à FMA, et Dekker vectorisé SSE2 pire encore (9,7 vs 5,6 s) — il ne sert
qu'aux CPU sans FMA ; (2) `inline(always)` sur le corps du noyau par lots
rend le natif ×2 plus lent : le poser par `cfg_attr(not(avx2+fma), …)` ;
(3) sans SSE4.1 `floor/round/trunc/ceil` sont logiciels — hors des boucles
chaudes. Reste : Lyapunov baseline ×1,2-1,4 ; MinGW et CPU sans AVX2 non
mesurés. `FRACTALL_SIMD_DISPATCH=0` (A/B).
⚠️ **Ne pas chercher à resserrer le terme `l·ε` du saut BLA f64** : à
ε = 2⁻⁵³ la troncature réelle est négligeable (borne du second ordre
implémentée et testée : la substituer revient à retirer le terme) et c'est
CE terme qui attrape les pixels faux — sans lui e227 1 → 390 faux non
signalés. Il sert de proxy de la masse d'erreur des `l` pas sautés. Environ
un pixel signalé sur deux est réellement faux ; le levier coût est le prix
de la correction, pas le détecteur (`scripts/reliability_confusion.py`).
⚠️ **Référence dont `Z[0] ≠ 0` (Julia : elle part du centre de la vue) :
le rebase est RELATIF à `Z[0]`** — critère `|z − Z[0]| < |δ|`,
`δ := z − Z[0]`, sur TOUS les chemins (fast-path, générique, HP exp, GMP,
kernels GPU f32 et f64). Deux défauts fermés le 2026-11-06 : (1) rebaser
comme si `Z[0] = 0` rendait 92-99 % de pixels faux sur toute vue Julia non
centrée en 0 (le preset `julia-siegel-disk`, centré en 0, ne pouvait pas le
voir) ; (2) `z − Z[0]` doit être formé `(Z[m] − Z[0]) + δ`, JAMAIS
`(Z[m] + δ) − Z[0]` : δ y est absorbé, et sur un centre PÉRIODIQUE (cible
d'un zoom auto-similaire) le pixel rebasait sur δ = 0 — 100 % de pixels
faux. Verrous `julia_rebase_is_relative_to_reference_origin`,
`julia_view_centred_on_a_fixed_point_matches_gmp`, preset `julia-edge-e20`.
Même classe : une famille Mandelbrot-like à **graine non nulle** (z₀ = seed,
métadonnées éditées) a aussi `Z[0] ≠ 0` — `wisdom::select_algorithm` ne la
route JAMAIS en perturbation (19 % de pixels faux mesurés ; verrou
`seeded_mandelbrot_is_never_rendered_by_perturbation`). L'axe fuzz du
harness sonde désormais Julia HORS CENTRE (`fuzz_scenes.generate_julia`).
⚠️ **Aucune lecture d'environnement hors `OnceLock` dans les fichiers du
chemin par pixel** (verrou de source
`hot_path_sources_only_read_env_through_once_lock`) : `std::env::var` par
pixel = ~250 ns + verrou global — réintroduit puis retiré la même nuit.
⚠️ **Tout chemin qui recalcule un pixel doit ajouter `centre_vue − cref`
au dc** (`resolved_sampling.ref_offset`, non nul quand l'orbite du cache est
réutilisée hors centre — `can_subset_reuse`, GUI seulement) : le plein cadre
et la correction GMP le faisaient, `correct_pixels_dd` non (39 pixels
corrigés sur 39 faux, verrou
`panned_view_with_reused_orbit_matches_fresh_render`). Les rendus sans cache
d'orbite (CLI, quality, harness) ne peuvent PAS voir cette classe de défaut.
Au tier exp (|dc| < 2⁻⁹⁰⁰, garde d'entrée du chemin rapide dd) le traqueur dd
est `reliability_dd_exp::ShadowTrackerDdExp` (borne et magnitudes en
`FloatExp`) : le traqueur f64 y sous-débordait et restait à 0.
⚠️ **Scène au-delà du dd = frame dd payée pour rien** (super_dense : 3 s →
132 s à 48² pour une image aussi fausse, avec avertissement) — TODO
G14.1 (k). Remède en OPT-IN : `FRACTALL_DD_PROBE=1` (sonde de ≤ 256 pixels
avant l'escalade frame ; majorité au-delà du dd ⇒ f64 gardé + avertissement ;
super_dense 132 → 10,4 s, lethal_weapon inchangé au pixel près).
⚠️ Le juge de ces
défauts n'était AUCUN axe de routine : c'est le rendu de production comparé
au tier dd (lui-même vérifié contre GMP) sur le corpus `toml/` qui les a
trouvés — quality juge 16 presets, precision juge l'orbite.

**Correction dd (2026-11-06)** : boucle dd en double-double « nu » mis à
l'échelle 2^s (`pixel_loop_dd.rs`, bit-identique à la boucle
`DoubleDoubleExp`, ×2-5 ; `FRACTALL_DD_FAST=0`) ; la correction ne bâtit que
la table BLA dd, niveaux ≥ `BLA_SKIP_LEVELS` (pic RSS dragon 1 063 → 329
Mo) ; partie basse capturée PARESSEUSEMENT (un Z GMP toutes les 1024
itérations dans `high_precision_data`, relecture parallèle seulement si un
pixel est flaggé ; `FRACTALL_LAZY_LO=0`).

### Interrupteurs d'environnement (A/B et diagnostic)

Tous lus UNE fois (`OnceLock`). Défaut = comportement de production.

| Variable | Effet |
|---|---|
| `FRACTALL_SIMD_DISPATCH=0` | binaire baseline x86-64 : pas de copies AVX2+FMA aiguillées à l'exécution |
| `FRACTALL_SCALAR_QD=0` | boucles scalaires suivies : pas d'étage quad-double avant le GMP |
| `FRACTALL_SIN_LANES=0` | étages dd/qd du sinus en boucle scalaire au lieu des voies vectorielles |
| `FRACTALL_STD_BATCH=0` | path f64 standard : interpréteur scalaire au lieu du noyau par lots |
| `FRACTALL_RELIABILITY_STD=0\|all` | path f64 standard : `0` coupe le traqueur et la correction ; défaut Mandelbrot + Julia (correction par perturbation) ; `all` ajoute BS / Tricorn / Celtic / Buffalo / PerpBS (GMP pur sous plafond) |
| `FRACTALL_STD_MARGIN=0` | path f64 standard : pas de test de détermination à l'évasion (emballement seul — moitié moins de pixels signalés, ~15× plus de pixels faux résiduels sur le volet `std:*`) |
| `FRACTALL_STD_CYCLE=0` | path f64 standard (lots ET boucles scalaires) : pas de raccourci des cycles exacts — chaque point intérieur itère jusqu'au plafond |
| `FRACTALL_ORBIT_PAR=0` | orbite de référence : itération GMP sur un seul cœur |
| `FRACTALL_ORBIT_PAR_MIN_BITS=n` | précision à partir de laquelle l'orbite utilise le fil assistant (défaut 2048) |
| `FRACTALL_ORBIT_RESUME=0` | orbite de référence recalculée de zéro quand le plafond monte (pas de prolongation) |
| `FRACTALL_CONFORMAL_BLA_EAGER=1` | table BLA conformale legacy construite avec l'orbite (au lieu de la première lecture) |
| `FRACTALL_AUTO_ITER_INCREMENTAL=0` | sonde d'itérations : chaque tour recalcule tous les pixels |
| `FRACTALL_AUTO_ITER_DEBUG=1` | une ligne `[AUTO-ITER]` par tour de sonde (plafond, durée, pixels calculés) |
| `FRACTALL_EXP_SCALED=0` | tier exp : boucle FloatExp au lieu du « double mis à l'échelle » |
| `FRACTALL_EXP_SCALED_STATS=1` | ventilation des tours de la boucle scaled (rapide / lent / causes de repli) |
| `FRACTALL_JULIA_FAST=0` | Julia en perturbation : pas générique au lieu du fast-path inline (⚠️ sans effet tant que le suivi de fiabilité Julia est actif : couper aussi `FRACTALL_RELIABILITY_FAMILY=0`) |
| `FRACTALL_PHASE_FAST=0` | Burning Ship / Tricorn : boucle générique au lieu des noyaux statiques |
| `FRACTALL_RELIABILITY=off\|observe\|escalate` | détecteur de fiabilité (défaut `escalate`) |
| `FRACTALL_RELIABILITY_EXP=0` | coupe le traqueur du tier exp |
| `FRACTALL_RELIABILITY_DD=0` | coupe le traqueur du tier dd (⚠️ rend l'ε 2⁻⁸⁰ de la BLA dd non gardé) |
| `FRACTALL_RELIABILITY_FAMILY=0\|all` | suivi de fiabilité hors Mandelbrot en perturbation : défaut Julia + Tricorn ; `all` ajoute Burning Ship ; `0` coupe |
| `FRACTALL_QD=0` | pas de tier quad-double : les pixels au-delà du dd vont droit au GMP |
| `FRACTALL_QD_MAX_FRACTION=f` | plafond du tier qd (défaut 0,01 ; plancher 16 pixels ; ≥ 1 = illimité) |
| `FRACTALL_DD_RETRY_NOBLA=0` | pas de second essai dd sans BLA avant le GMP |
| `FRACTALL_RELIABILITY_GMP_MAX_FRACTION=f` | plafond de l'escalade dd → GMP (défaut 0,001 ; ≥ 1 = illimité) |
| `FRACTALL_DD_PROBE=1` | sonde dd avant l'escalade frame (opt-in) : scène dont la majorité des pixels non fiables échappe aussi au dd ⇒ f64 gardé avec avertissement, frame dd non rendue |
| `FRACTALL_F64_MARGIN=1` | test de détermination à l'évasion au tier f64 (opt-in, +75-195 %) |
| `FRACTALL_SHALLOW_DIRECT=0` | vues peu profondes : correction par perturbation dd/qd au lieu des étages directs dd puis qd (93 px faux sur éléphants 1e7 en perturbation forcée) |
| `FRACTALL_SHALLOW_SKIP_F64=0` | vue peu profonde à référence courte : garde l'étage de perturbation f64 tracée avant le dd direct |
| `FRACTALL_DIRECT_QD=0` | coupe l'étage qd direct (les pixels que le dd direct refuse partent au GMP) |
| `FRACTALL_DIRECT_DD=0` | correction des familles sans tier dd (Julia, Tricorn) et correction `GmpCapped` du path standard : pas d'étage double-double direct, les pixels signalés partent droit au GMP pur (sortie identique, ×5-8 plus lent sur un bord) |
| `FRACTALL_DD_ESCAPE_SHALLOW=0\|f` | correction dd des vues PEU PROFONDES (pixel ≥ 1e-13) : `0` coupe le test d'évasion relatif (défaut 0,01·\|z\|), une valeur le remplace. Sans lui, bord parabolique à fort plafond (éléphants 1e7, 60 000 it.) : 1,2 % de pixels faux |
| `FRACTALL_DD_ESCAPE_REL=0.1` | tier dd : pixel non fiable si sa borne ≥ 0,1·\|z\| à l'évasion (opt-in — corrige le dernier pixel faux de seahorse-valley 256², coûte ×2 sur e227 pour 0 pixel changé) |
| `FRACTALL_DD_FAST=0` | boucle `DoubleDoubleExp` au lieu du chemin rapide dd |
| `FRACTALL_LAZY_LO=0` | capture eager de la partie basse de l'orbite |
| `FRACTALL_HARMONIC_LA=auto\|1\|0` | routage Harmonic LA |
| `FRACTALL_GPU_PERTURB_F32=1`, `FRACTALL_GPU_F32_RAW=1`, `FRACTALL_GPU_F32_FALLBACK_RATIO` | kernel GPU f32 : forçage, sortie brute sans correction, seuil de repli |
| `FRACTALL_PERTURB_STATS=1`, `FRACTALL_WISDOM=1` | lignes `[PERTURB PERF]` / `[RELIABILITY*]` / `[FRACTALL]`, `[WISDOM]` |

### Précision GMP perturbation

Formule C++ Fraktaler-3 : `bits = max(24, 24 + floor(log2(zoom * height)))`,
**+ 64 bits de marge** (`ORBIT_PRECISION_MARGIN_BITS`, 2026-11-05) ; clamp
`[128, …]`. Cf. `compute_perturbation_precision_bits()` ; le champ
utilisateur `precision_bits` sert de plancher. ⚠️ La formule F3 NUE laisse
2-4 % de pixels FAUX sur des scènes du corpus (golden_spider, e227,
evolution_trees — mesuré contre une orbite à précision ×4, juge convergé
×2 == ×4) ; +64 b les ramène à la vérité au pixel près.

**Boucles GMP PURES par pixel** (`gmp.rs::iterate_point_mpc` — juge de
`fractall-quality`, correction par pixel, repli GMP plein cadre) : sans
allocation ni norme pleine précision par itération depuis le 2026-11-06
(`mpc_below_bailout` : norme f64 d'abord ; `sqr_in_place` : carré arrondi
EXACTEMENT comme `mpc_sqr`, par deux carrés exacts à précision double et une
seule soustraction arrondie). ×1,8 à ×2,3, `.fmap` identiques. Ces boucles
ont aussi le raccourci des cycles exacts (`MpcCycle` : ×7 sur une vue à
intérieur à 20 000 itérations, 256 b). ⚠️ MPC alloue
ses temporaires à chaque appel (60 % du temps dans `malloc` au profil) : dans
une boucle GMP chaude, passer par MPFR sur tampons réutilisés — mais ne
JAMAIS former `fl(x²) − fl(y²)` (double arrondi) si l'on veut les bits de
`mpc_sqr`.

**Types à racine / sinus / division (2026-11-19)** — f64 : Multibrot
DEMI-entier (défaut 2,5) par `z^k·√z` algébrique (même branche que `powf`,
signe de ±0 sur l'axe réel négatif compris) : ×2,1-2,6 ; sinus complexe
`csin` par un `sincos` + un `expm1`, raccourci |Im z| < 2⁻²⁸ (sans lui
l'accueil Mandelbrot Sin, dont l'intérieur est sur l'axe réel, perdait
25 %) : ×1,1-1,3 ; Nova a le raccourci des cycles exacts. Juges GMP de ces
types (`gmp.rs::MpfrOps`, tampons MPFR réutilisés + `MpcCycle` + puissances
entières par multiplications) : Magnet M accueil 32 → 1,2 s, Multibrot
Julia bord 135 → 2,2 s, Mandelbrot Sin accueil > 150 → 8 s — des scènes de
l'axe `types` jusque-là `gmp:timeout` deviennent jugeables. `MpfrOps::mul`
rend les bits de `mpc_mul` (verrou) ; division, racine et sinus sont à
quelques ulp, non correctement arrondis. ⚠️ MPFR ne sous-déborde pas :
l'intérieur de Mandelbrot Sin tend vers 0 sans l'atteindre, le raccourci de
cycle ne mord jamais et `mpfr_sinh_cosh` travaille à p + log2(1/|t|) bits
(~600 µs/itération) — d'où le raccourci du sinus d'argument minuscule.
⚠️ Le `ns/px·it` de Pickover n'a pas de sens (sa sortie « itération » est
une valeur de piège). Reste : ces boucles sont bornées par la LATENCE
(sqrt → div, sincos/expm1) — le levier est un traitement par voies comme
`interp_batch`.

**Itération GMP de l'orbite** (`orbit.rs::fast_sqr_scratch`, Mandelbrot
mono-phase) : `re' = (re+im)(re−im) + cr`, `im' = 2·re·im + ci` en 2
multiplications MPFR à tampons réutilisés, au lieu de `mpc_sqr` (arrondi
correct de z² exact, ~700 ns à 676 b contre ~410). Les deux sont des
pseudo-orbites de même précision, aucune n'est « plus vraie » (écarts au juge
×4 équivalents) ; 31/38 cas du corpus pixel-identiques, 25 goldens
inchangés. Dragon : orbite −35 %, total −12 %. `FRACTALL_ORBIT_FAST_SQR=0`
revient à `mpc_sqr` (A/B).

**Orbite sur DEUX cœurs** (`perturbation/orbit_par.rs`, 2026-11-06) :
l'orbite de référence est le seul travail strictement séquentiel du rendu et
c'est ELLE qui fait le temps des zooms très profonds (corpus à 64² :
triangle 14 s sur 17, wfs_extended 16 s sur 19). Les deux multiplications du
pas sont indépendantes : `im' = 2·re·im + ci` part sur un fil assistant
(attente active, `Float` transférés par une case — aucun `unsafe`) pendant
que le fil appelant calcule `(re+im)(re−im)`. Mêmes opérations MPFR ⇒ orbite
**identique au bit près** (verrou `par_sqr_matches_serial_step_bit_for_bit`).
Garde-fous : reprise de la tâche si l'assistant ne l'a pas prise ; fenêtre
série à doublement exponentiel, assistant endormi, quand la machine est
saturée ; seuil de précision (sous ~2000 bits la synchronisation coûte plus
que la multiplication épargnée). ⚠️ `rayon::join` ne convient PAS ici
(mesuré ×0,14-0,67) : une itération dure ~1 µs, un réveil de fil 10-50 µs.
⚠️ **Une tâche se RÉCLAME par `compare_exchange`** (`POSTED → TAKEN` côté
assistant, `POSTED → IDLE` pour la reprise) et seul le gagnant vide la
case : la première version agissait sur un état LU plus tôt et pouvait
laisser l'appelant attendre `DONE` pour toujours (blocage trouvé à la
relecture le jour même, verrou
`claimed_handoff_survives_preemption_at_every_step`). Mesuré au calme :
×1,1 à 2 800 bits, ×1,4 à 5 000, ×1,8-1,9 au-delà de 10 000 (e8000 total
1,28 → 0,69 s) ; neutre au seuil.
Banc : `cargo run --release --example bench_orbit_join -- <bits> <iters>`.

**Cache de table BLA** (`delta.rs`, `GLOBAL_BLA` + thread-local) : clé =
identité de l'orbite (**adresse + longueur + empreinte de contenu**, cf.
`OrbitIdentity`) + `c_norm` + empreinte de la formule hybride/opcodes
(`[M,M]` et `[M]` ont la même orbite primaire) + type/puissance/seuil/tier dd. ⚠️ Ni l'adresse
seule (une orbite libérée puis une autre réallouée au même endroit
retrouverait la table de l'ancienne) ni l'oubli de `c_norm` (orbite
réutilisée pour une vue plus large → rayons de merge sur-permissifs) ne sont
acceptables : verrou `bla_cache_key_follows_orbit_content_and_view_radius`.
Le chemin par pixel ne clone PAS l'`Arc` de l'entrée (poignée `Rc` par
thread) et ne compile pas de `Formula` : sur les vues peu chères c'était
jusqu'à 85 % du temps pixel. Règle générale tirée de cette nuit : **rien de
ce qui est constant sur le rendu ne doit être recalculé dans
`try_bytecode_unified_path`** — `effective_pixel_size` y relisait les spans
HP en GMP à chaque pixel au-delà de 1e308 (mémo depuis, verrou
`hp_pixel_size_memo_follows_its_key`) ; l'outil pour le voir est
`harness.py hotspots`.

**Table BLA conformale PARESSEUSE** (`bla.rs::LazyBlaTable`, 2026-11-19) :
la table legacy n'est lue qu'en repli hors bytecode ; elle n'est plus
construite qu'à la première lecture (sonde seahorse 1e7 à 160 000 it. :
0,37 s → 0 ; e8000 −0,1 s). ⚠️ Ne jamais construire une table DANS
`OnceLock::get_or_init` ni sous un Mutex quand la lecture peut venir d'un
worker rayon : le build est parallèle, le thread constructeur peut voler une
tâche pixel qui relit la cellule → ré-entrée, blocage. Construire hors
verrou puis `set` (le premier gagne) ; les chemins qui la liront à coup sûr
la pré-construisent hors boucle ; la construire AVANT
`strip_orbit_arrays_for_compress` (qui vide `z_ref_f64`).
`FRACTALL_CONFORMAL_BLA_EAGER=1` (A/B).

**Orbite PROLONGÉE** (`orbit.rs::OrbitTail`, `compute_reference_orbit_
resuming`, 2026-11-19) : quand seul le plafond monte (sonde d'itérations,
auto-ajustement, GUI), la boucle GMP reprend à la fin de l'orbite du cache
au lieu de repartir de 0 — vue intérieure 1e300, sonde 76k → 304k → 1,2 M :
10,2 → 7,6 s. Identique au bit près à une orbite fraîche (verrou
`resumed_orbit_matches_fresh_orbit_bit_for_bit`). Périmètre : Mandelbrot non
hybride, hors fast-path dd ≤ 96 b, sans orbite dense / compression / Brent.
⚠️ La boucle GMP ne porte d'un tour à l'autre que z et dZ/dC : tout nouvel
état porté par la boucle doit entrer dans `OrbitTail` (ou exclure la
reprise). `FRACTALL_ORBIT_RESUME=0` (A/B).

**Cache d'orbite** (`ReferenceOrbitCache`) : orbite + BLA réutilisées si même
centre / type / précision. ⚠️ Le cache retient le plafond DEMANDÉ à sa
construction (`requested_iteration_max`) : le relèvement du plafond du rendu
à `cache.iteration_max` ne vaut que pour l'auto-ajustement de la série, donc
seulement si la vue demande au moins autant — un cache bâti à 5000 puis
réutilisé à 2000 (GUI) rendait TOUS les pixels au plafond du cache (défaut
ancien fermé le 2026-11-20, verrou
`lowering_the_cap_with_a_reused_orbit_matches_fresh_render`). ⚠️ **Régime atom-domain** (`is_valid_for` +
`can_subset_reuse` → `atom_regime_scale_mismatch`) : la troncature atom-domain
de la référence dépend de l'ÉCHELLE de vue (`atom_radius_sqr = span_vue²`), donc
une référence est baked à son span de construction. En zoom profond
(`pixel_size < 1e-13`) une référence bâtie à une échelle différente (±1/16) est
INVALIDÉE (rebuild) — sinon sa troncature ne correspond pas à un build frais →
bruit sel-et-poivre (~1.7 % px) sur le rendu inter-frame (bug corrigé 2026-07-16 ;
le pan à profondeur fixe garde span constant → réutilisation préservée). N'affecte
que la réutilisation GUI multi-frame ; les rendus single-shot (CLI/quality/harness,
`cache=None`) ne consultent jamais ces prédicats.

## Paramètres perturbation

| Champ | Description | Défaut |
|-------|-------------|--------|
| `bla_threshold` | seuil delta BLA | 1/2²⁴ ≈ 5.96e-8 |
| `bla_validity_scale` | multiplicateur rayon BLA | 1.0 |
| `glitch_tolerance` | seuil `\|δ\|²>tol²·\|Z\|²` → GMP (chemin legacy) | 1e-4 |
| `series_order` | ordre série (0 = off) | 2 |
| `max_perturb_iterations` | cap pas DIRECTS par pixel (⚠️ voir note) | 1024 |
| `max_bla_steps` | cap pas BLA par pixel | 1024 |
| `use_reference_precision_formula` | formule C++ F3 | true |
| `use_bytecode_engine` | path unifié BLA mat2 + rebasing F3 | true |
| `use_dd_tier` | tier double-double ~106b Mandelbrot deep (float128-like, table BLA dd `bla_dd.rs`) | false |
| `find_nucleus` | nucleus Mandelbrot avant orbit | false |
| `jitter_scale` | amplitude AA sous-pixel (px) | 0.0 |
| `aa_subpixel_offset` | offset AA uniforme transitoire legacy (`#[serde(skip)]`) | `[0,0]` |
| `aa_jitter` | sample AA par pixel `Some((k,scale))` (`#[serde(skip)]`, décorr. Cranley-Patterson F3, prioritaire) | `None` |
| `rotation` | degrés CCW, mat2(cos,-sin,sin,cos) | 0.0 |

⚠️ **`max_perturb_iterations` / `max_bla_steps` : clampés à `≥ iteration_max`**
dans `render_perturbation_with_cache` (chemin commun). Comme `iters_ptb ≤ n <
iteration_max`, un cap < iteration_max ne fait que tronquer les pas directs tôt →
compte d'itération ~radial ⇒ **anneaux concentriques** (cf. cusp -0.75 : défaut
1024 < ~1700 requis). Le défaut 1024 reste le champ utilisateur, mais le clamp
garantit l'absence de troncature parasite. F3 met `maximum_perturb_iterations =
iterations`. Le loader TOML faisait déjà `= iters` ; le clamp couvre GUI + CLI
non-TOML.

**Escape radius** : champ `bailout`. Défaut **25** (`bailout_sqr=625`,
`const ESCAPE_TIME_BAILOUT`, `definitions.rs`) pour la famille escape-time
bytecode+perturbation — aligné F3 (`escape_radius=625`). Les types à sémantique
d'évasion particulière (Newton, Magnet, Sin, Nova, Pickover, densité, vectoriel,
AlphaMandelbrot) gardent leur propre bailout (souvent 4). Configurable par
pixel via `--bailout`/PNG.

**Anti-aliasing multi-sample** (`fractal/jitter.rs`) : chaque sample jitère le
mapping pixel→c d'un offset sous-pixel low-discrepancy (Halton `radical_inverse`
+ tente `triangle`, port F3 `hybrid.h`), moyenne des rendus colorisés en RGB.
**Décorrélation Cranley-Patterson PAR PIXEL** (`jitter::pixel_offset`, port fidèle
F3 `jitter` `hybrid.h:62`) : chaque pixel reçoit une rotation
`h = burtle_hash(j·w+i)/2³²` ajoutée à la valeur de Halton avant la tente → les
pixels voisins tirent des sous-positions **décorrélées** (dithering), supprimant
l'aliasing corrélé du schéma per-frame uniforme à **bas N**. Porté par le champ
transitoire `aa_jitter: Option<(k, scale)>` (prioritaire sur `aa_subpixel_offset`
uniforme legacy), appliqué au mapping des paths CPU (f64/GMP/perturbation
f64/exp/dd). ⚠️ Gaté `aa_jit.is_some()` → **hors AA bit-identique** (goldens
pixel-exact). Le sample `k=0` est aussi jitteré (fidèle F3), et le point
dégénéré `v=0.5` de `triangle` n'est plus jamais atteint. CLI
`--aa-samples N`/`--jitter-scale` (boucle dans `main.rs` via
`io::png::colorize_to_rgb` + `save_png_rgb_with_metadata`) ; GUI : dropdown **AA**
(accumulation après les passes, `RenderMessage::AaProgress`). CPU uniquement.

## Couleur

**27 palettes** (index 0-26) — Fire, Ocean, Forest, Violet, Rainbow, Sunset,
**Plasma** (défaut), Ice, Cosmic, Neon, Twilight, Emboss, Waves, SynthRed,
LightYears, Blues, Coffee, Classic, Dimensions, Earth, FireIce, Habs, Jays,
Slice, Stardust, Strobe, SynthBlue.

**3 espaces couleur** (`color_models.rs`) : RGB, HSB (circulaire),
LCH (CIE Lab, perceptuellement uniforme).

**15 modes coloring** (`OutColoringMode`) : Iter, IterPlus{Real,Imag,
RealImag,All}, BinaryDecomposition, Biomorphs, Potential,
ColorDecomposition, **Smooth** (défaut), OrbitTraps, Wings, Distance,
DistanceAO, Distance3D.

**4 orbit traps** (`orbit_traps.rs`) : Point, Line, Cross, Circle.

### ⚠️ Colorisation UNIQUE CLI ↔ GUI (invariant, 2026-08-13)

**Une seule** boucle de colorisation : `io::png::colorize_buffers(params,
iterations, zs, distances, orbits, width, height)`. `gui::colorize_buffer`
n'en est qu'une enveloppe (dimensions explicites : la GUI colorise aussi des
TUILES). Enveloppes de haut niveau : `colorize_to_rgb` (sans canaux annexes,
réservée à l'AA multi-sample qui n'en produit pas) et
`colorize_to_rgb_with_extras` (**à utiliser dès que le path dispose des
canaux**).

Le dispatcher renvoie 4 buffers : `(iterations, zs, orbits, distances)`. Les
deux derniers sont CONSOMMÉS par la colorisation — `distances` par
Distance/DistanceAO/Distance3D, `orbits` par OrbitTraps/Wings. Les jeter fait
retomber ces 5 modes **silencieusement sur Smooth** (aucune erreur, image
plausible mais fausse) : c'était le bug de la copie CLI, qui passait
`None, None, false` en dur. Même cause pour Anti-Buddhabrot, absent du test
`is_buddhabrot` côté CLI → colorisé comme un escape-time (densité maximale
⇒ `iter == iteration_max` ⇒ noir).

**Colorisation VÉRIFIÉE** (G5, 2026-08-24) : `io::png::colorize_output(
params, &RenderOutput) -> Result<Vec<u8>, String>` est le point d'entrée à
privilégier pour une sortie plein cadre — il vérifie via
`render::output::required_channels` que les canaux requis par
`out_coloring_mode` (Distance* → `distances`, OrbitTraps/Wings → `orbits`)
sont PRÉSENTS, et renvoie **Err** sinon (au lieu de l'image
plausible-mais-fausse). Câblé : CLI (PNG final + samples AA + `--from-map`),
HQ GUI, `colorize_keyframe` vidéo (+ refus au PLAN d'un manifest
`outcoloring distance` sans `distance_estimation`). Le CLI couple
mode→flags (`--outcoloring distance` ⇒ `enable_distance_estimation`, parité
GUI). Les paths d'affichage GUI (tuiles, recolor, previews) restent sur
`colorize_buffers` non-vérifié (tolérance interactive).

**Invariant mode → canaux UNIQUE** : `render::ensure_required_channels(
&mut params)` (`render/output.rs`) pose `channels ⊇ required_channels(params)`
— à appeler à CHAQUE frontière de rendu (CLI, GUI fenêtre, export HQ via la
fonction pure `gui::hq_render_state::hq_render_params`, preview Julia). Ne pas
le refaire à la main : la règle a existé en trois copies, et l'export HQ ne
l'appliquait pas (4K en mode Distance → colorisation vérifiée en erreur). Le
plan vidéo garde son REFUS explicite (canal `orbits` non persisté).

**Ne JAMAIS** réécrire une boucle de colorisation ailleurs ni appeler
`color_for_pixel_with_lut` directement depuis un path de sortie : une
divergence colorisation = un PNG qui ne ressemble pas à l'écran. Verrous :
`io::png::tests::{antibuddhabrot_colorizes_as_density_like_buddhabrot,
distance_channel_changes_distance_mode_output,
colorize_output_refuses_dropped_required_channel}`.

## Transformations de plan (XaoS-style)

| ID | Nom | Formule |
|----|-----|---------|
| 0 | Mu | c |
| 1 | Inversion | 1/c |
| 2 | InversionShifted | 1/(c + 0.25) |
| 3 | Lambda | 4·c·(1-c) |
| 4 | InversionLambda | 1/(4·c·(1-c)) |
| 5 | InversionLambdaMinus1 | 1/(4·c·(1-c)) - 1 |
| 6 | InversionSpecial | 1/(c - 1.40115) |

## Types de fractales (`--type N`)

| ID | Type | Algo |
|----|------|------|
| 1  | Von Koch | vectoriel |
| 2  | Dragon | vectoriel |
| 3  | Mandelbrot | **bytecode + perturbation** |
| 4  | Julia | **bytecode + perturbation** |
| 5  | Julia Sin | f64 / GMP |
| 6  | Newton | f64 / GMP |
| 7  | Phoenix | f64 / GMP |
| 8  | Buffalo | bytecode |
| 9–12 | Barnsley/Magnet (Julia + Mandelbrot) | f64 / GMP |
| 13 | Burning Ship | **bytecode + perturbation** |
| 14 | Tricorn | **bytecode + perturbation** |
| 15 | Mandelbulb | f64 / GMP (3D) |
| 16 | Buddhabrot | densité |
| 17 | Lyapunov | spécial (6 presets) |
| 18 | Perpendicular Burning Ship | bytecode |
| 19 | Celtic | bytecode |
| 20 | Alpha Mandelbrot | f64 / GMP |
| 21 | Pickover Stalks | f64 / GMP |
| 22 | Nova | f64 / GMP |
| 23 | Multibrot | bytecode (puiss. entières) sinon f64/GMP |
| 24 | Nebulabrot | densité |
| 25–30 | Julia variants (BS/Tricorn/Celtic/Buffalo/Multibrot/PerpBS) | bytecode |
| 31 | Alpha Mandelbrot Julia | f64 / GMP |
| 32 | Mandelbrot Sin | f64 / GMP |
| 33 | Anti-Buddhabrot | densité |

**Paires Mandelbrot ↔ Julia** (preview + touche `J`) : Mandelbrot↔Julia,
Barnsley↔Barnsley Julia, Magnet↔Magnet Julia, Burning Ship↔Burning Ship
Julia, Tricorn↔Tricorn Julia, Celtic↔Celtic Julia, Buffalo↔Buffalo Julia,
Multibrot↔Multibrot Julia, PerpBS↔PerpBS Julia, Alpha Mandelbrot↔Alpha
Mandelbrot Julia, Mandelbrot Sin↔Julia Sin.

## CLI (résumé — voir README pour la liste complète)

```
fractall-cli --type N --output FILE [OPTIONS]

  --type / --width / --height
  --center-x / --center-y (f64) ou --center-x-hp / --center-y-hp (string HP)
  --zoom EXPR (notation scientifique OK, ex: 1.41e219)
  --iterations N|auto     # ordre : N explicite → N ; `auto` explicite → sonde
                          #   (même sur une valeur --toml/--from-png) ; absent
                          #   + valeur de --toml/--from-png → cette valeur ;
                          #   absent sinon → sonde wisdom_iter si
                          #   `auto_iterations_applicable`, défaut du type
                          #   sinon. Sonde : ligne stderr « Iterations: auto →
                          #   N (…) », orbite réutilisée par le rendu, PNG =
                          #   valeur numérique finale (--from-png ne resonde pas)
  --algorithm auto|f64|perturbation|gmp
  --precision-bits N
  --palette 0-26 / --color-repeat N / --outcoloring MODE
  --aa-samples N          # AA multi-sample jitteré (1 = off), CPU only
  --jitter-scale F        # amplitude jitter sous-pixel (px, défaut 1.0)
  --plane N (0-6)
  --rotation DEG          # F3-style mat2 rotation, degrés CCW (CPU + GPU bytecode)
  --find-nucleus          # Mandelbrot nucleus refine (atom-domain)
  --dd-tier               # tier double-double ~106b (spirales deep sensibles)
  --no-bytecode           # désactive bytecode (debug)
  --gpu / --no-gpu        # override device ; sinon AUTO (wisdom::select_device,
                          #   G9.5 : benchmark machine + garde-fou correction —
                          #   GPU seulement plage deep both-perturbation)
  --enable-distance-estimation / --enable-interior-detection
  --multibrot-power F / --lyapunov-preset NAME
  --julia-re F / --julia-im F   # graine c des types Julia (ex. --julia-re=-0.8
                          #   --julia-im=0.156) ; ignorée ailleurs
  --toml FILE             # charge real/imag/zoom/iterations/rotate/phases/
                          # opcodes + blocs [[formula]] F3 natifs
                          # format léger rust-fractal-core
  --from-png FILE.png     # re-rend la vue EXACTE d'un PNG fractall (touche S
                          # GUI ou sortie CLI) ; --type non requis. Toute
                          # option EXPLICITE écrase la valeur du PNG (les
                          # options à défaut sont des Option<…>, les drapeaux
                          # bool n'agissent que posés). Resize : centre et
                          # span_y (chaînes HP) intacts, span_x(_hp) dérivé
                          # du nouveau ratio en HP ; une seule dimension ⇒
                          # l'autre suit le ratio du PNG. Logique pure :
                          # `io::png::params_for_rerender` +
                          # `ViewHp::resize_keep_vertical_span` ; overrides :
                          # `main.rs::resolve_params` (testé). AA non persisté.
  --opcodes "SQR ADD…"    # formule opcodes F3 (rot{DEG} incl.), prioritaire
                          # sur --phases ; ex. "sqr rot{30} add"
  --wisdom-bench          # bench machine (G9.2) : débits par technique →
                          # ~/.config/fractall/wisdom.toml (--output non requis)
```

## GPU (wgpu)

- `mandelbrot_f32.wgsl`, `julia_f32.wgsl`, `burning_ship_f32.wgsl` — paths
  legacy f32 par type
- `perturbation.wgsl` — **kernel perturbation F3-strict en f64 natif**
  (2026-07-15, G9.4) : port de `pixel_loop.rs::iterate_pixel_unified_
  mandelbrot` (n/m séparés, rebasing strict, anti-over-skip BLA, PAS de
  glitch detection), buffers zref/BLA f64, span/offset en paires hi/lo f32.
  Exige `Features::SHADER_F64` (NVIDIA/AMD/Intel/llvmpipe OK ; **Metal NON**
  → pipeline absente → fallback CPU). ⚠️ Le double-float 2×f32 est
  IMPOSSIBLE en WGSL : naga/drivers réassocient les EFT (two_sum → 0, fma
  non-fusionné) — diagnostic : `cargo run --release --bin df64_gpu_probe`.
  Gate host : `ref_len-1 ≥ iter_max` sinon fallback CPU (réfs tronquées).
- `bytecode_kernel.wgsl` — runtime bytecode unifié (P3.1 Task 7). Applique la
  matrice K (rotation/transform) au mapping pixel→c, parité CPU/F3.

- `perturbation_f32.wgsl` — **kernel perturbation à mantisse f32 + traqueur
  de fiabilité** (G9.6) pour les GPU SANS `SHADER_F64` (Metal/Apple Silicon)
  — forçable `FRACTALL_GPU_PERTURB_F32=1`. Mandelbrot/Julia, span ≥ 1e-28.
  Pixels `flags=2` (non fiables) recalculés sur le CPU
  (`perturbation::recompute_pixels_cpu`, mêmes plafonds que le plein cadre —
  sans `with_uncapped_perturb_limits` ils sortaient à 1024) ; frame entière
  CPU au-delà d'un seuil qui dépend du chemin CPU concurrent (0,45 s'il
  ferait de la perturbation, 0,25 s'il ferait du std f64). **Mesuré sur
  Metal (Apple M4, 2026-11-06)** : PASS vs GMP de 1e4 à 1e20 ; 4-13× le CPU
  sur les scènes à < 1 % de pixels renvoyés (e13, e17, minibrot e18),
  1,3-2,3× sur les spirales à 27-42 % ; perd sous 1e12 (CPU en std) ; le
  coût est dominé par la correction CPU, pas par le kernel. Sonde BLA
  ASCENDANTE dans le kernel (le rayon ne croît pas avec le niveau ; ×4 sur
  le temps kernel, sorties identiques au bit — non portée dans
  `perturbation.wgsl` f64, non testable sur Metal). `cabs_small` : modules
  mis à l'échelle 2⁵⁶ (|δ|² sous-débordait en f32 au-delà de ~1e19). Routage
  Auto : clés wisdom `gpu_perturb_f32` vs `cpu_perturb_f64_mid` (MÊME vue des
  deux côtés), donc seulement après `fractall-cli --wisdom-bench` sur la
  machine. Outil : `fractall-quality gpu-bench`. lavapipe vs GMP : **PASS** max_diff 1 à 1e4/3e5/1e6/
  e13 (kernel f64 : WARN max_diff 442 à 1e6), mais 11-16 % de pixels
  renvoyés au CPU sur le bord chaotique seahorse (0,2 % à e13 ; 1e8 → CPU
  entier).

**Rotation/transform** : seul le path bytecode applique K sur GPU. Les autres
paths GPU (perturbation, shaders f32 dédiés) retombent sur le CPU quand
`transform_matrix().is_some()` (garde-fou anti-sortie-non-tournée).

Le path perturbation GPU (f64) est vérifié par `gpu-suite` jusqu'à **1e30**
(G9.4b, 2026-07-15) : le kernel gère les réfs tronquées comme le CPU (wrap
périodique + rebase-at-end atom-domain + guard BLA `lands_on_ref_end`) ; seule
la réf tronquée par ESCAPE retombe en CPU (per-pixel GMP requis). Range borné
à zoom ≲ 4e37 par le transport span f32 hi/lo (gate `GPU_SPAN_F32_MIN`). Perf
e30 1024² : GPU ~2× plus lent que CPU f64 16t (GeForce f64 1:64) — l'intérêt
du kernel deep est la CORRECTION (div 3e-4 là où le CPU f64 fait 0.034 sur
scène ultra-sensible, cf. TODO 9.4/9.6), pas la vitesse. **Backend** : macOS
Metal (sans SHADER_F64 → perturbation sur CPU) ; Linux Vulkan (prioritaire),
puis OpenGL ; Windows DX12 / Vulkan.

## GUI (`FractallApp`)

### Menu Type
- Racine : Mandelbrot, Barnsley Mandelbrot, Magnet Mandelbrot, Burning
  Ship, Perp. Burning Ship, Tricorn, Celtic, Buffalo, Multibrot, Alpha
  Mandelbrot, Mandelbrot Sin.
- **Hybride (séquence)** (G4 jalon 5c) : éditeur de séquence multi-phase —
  ➕ par type (M/BS/Tricorn/Celtic/Buffalo/PerpBS/Multibrot), ⌫/🗑,
  « Appliquer » (≥ 2 phases) → `apply_hybrid_sequence()` (reset params
  convention Mandelbrot + `hybrid_phases` + `start_render`, GPU off). Le label
  du menu affiche « Hybride: M⊕BS » quand actif ; choisir un type normal
  désactive (default_params → `hybrid_phases: None`). Drag-and-drop d'un PNG
  hybride restaure la séquence via les métadonnées.
- **Julia all** (dossier) : toutes les variantes Julia.
- Mandelbulb, Julia Sin, Newton, Phoenix, Pickover Stalks, Nova après séparateur.
- **Densité** : Buddhabrot, Nebulabrot, Anti-Buddhabrot.
- **Lyapunov** : 6 presets.
- Von Koch / Dragon : CLI uniquement (`--type 1/2`).

### Fonctionnalités
- Rendu progressif multi-passes (preview → full).
- Recolorisation asynchrone (versioning pour ignorer les résultats obsolètes).
- Cache orbite + BLA entre re-rendus.
- **Réutilisation pixels inter-frame XaoS** (G10.4, `fractal/xaos.rs`) : en
  pan/zoom sans rotation, les colonnes/lignes de la frame précédente
  matchées à ≤ 0.5 px (positions vraies trackées `col_err`/`row_err`, aucune
  dérive cumulée) sont copiées au lieu d'être recalculées (~×40 en pan) ;
  param `xaos: Option<&XaosMap>` du dispatcher unique (CLI/quality/HQ/AA =
  `None`). **Zoom (2026-07-16)** : matching INJECTIF par axe (une colonne
  source → au plus une cible) — garantit ≥ (1−a)·n colonnes fraîches par axe
  en zoom-in (fin de l'« écho pur » du zoom ×2 aligné qui ne calculait RIEN
  et retardait l'image exacte), no-op en pan/zoom-out/previews. Raffinement
  exact silencieux à l'idle (400 ms, sans indicateur visible, déclenché seulement si
  erreur réelle > ε) via `build_refine_map` : map identité à tolérance
  exacte sur l'**erreur PAR PIXEL** — conserve tout pixel dont l'erreur ≤ ε,
  recalcule les approximations → cycle zoom ×2 écho+refine ≈ 99 % d'un
  rendu frais (image visible à ~0.8×, vs 148 % en refine total).
  ⚠️ **Erreur PAR PIXEL** (`XaosSourceFrame::err`, 2026-08-23) : un modèle
  par axe (`col_err` = « toute la colonne est décalée ») est FAUX sur une
  frame mixte — ce qu'est TOUTE frame d'écho zoom-in (colonnes fraîches par
  injectivité) : l'erreur cachée croissait géométriquement en molette
  continue (0,48 → 1,5 px en 6 crans). `col_err/row_err` ne sont plus que
  des hints de sélection ; `XaosMap` porte jusqu'à deux candidats par axe
  (hinté = pixels copiés, nominal = pixels frais), injectivité par variante,
  décision de copie PAR PIXEL dans `source_index` (verrou
  `continuous_anchored_zoom_keeps_true_error_bounded`). La transformée
  frame→frame est en précision DYNAMIQUE (`transform_precision`,
  −log2(span)+96 : à 256 b fixes, Δcentre d'un zoom ancré disparaissait
  au-delà de ~1e74, verrou `deep_anchored_zoom_transform_sees_off_center_
  delta`). **Plus de réutilisation grossière inter-passes côté GUI**
  (progressif ET HQ) : le pixel (i/r, j/r) copié a son centre décalé de
  (r−1)/2 px → treillis 1/16 de pixels faux (559/65536 mesurés) que XaoS
  marquait exacts. Une passe intermédiaire (1/16) ne remplace jamais une
  frame source plus résolue ; `glitch_ratio` est calculé sur les pixels
  CALCULÉS (les copies diluaient le ratio → escalade dd sautée).
  Frame source stockée par passe CPU uniquement (jamais GPU f32, jamais une
  passe écho-pur — aucune information nouvelle, dégraderait la source en
  copies de copies). Compatibilité = fingerprint JSON des params
  non-géométriques. Boucles pixel : point d'entrée unique
  `XaosMap::source_index(i, j)` (sémantique produit vs union).
  ⚠️ **Invariant : écho XaoS et reuse basse-résolution inter-passes sont
  mutuellement EXCLUSIFS** (dispatcher + `render_perturbation_with_cache`) :
  le `reuse` copie des centres décalés de (ratio−1)/2 px, ce qui
  contaminerait les pixels que le map déclare FRAIS/exacts (verrou
  `echo_pass_ignores_coarse_pass_reuse`) — et la GUI ne passe plus JAMAIS
  de `reuse` (cf. ci-dessus). Les passes
  intermédiaires écho-pur sont SAUTÉES (le warp G10.1 affiche déjà le même
  contenu, en plus net — supprime le pompage flou preview→full en
  navigation) ; la passe finale tourne toujours. Diagnostics :
  `xaos_pan_speedup_diagnostic`, `xaos_zoom_cycle_diagnostic` (`--ignored`).
- **File de tuiles priorité-centre** (G10.5, `render/tiles.rs`) : les 4
  boucles pixel CPU (f64/GMP/perturbation/perturbation-GMP) tournent sur une
  work-queue de tuiles (16/32/64 px, ≥ 8/thread) ordonnée par distance au
  curseur (`hover_norm`, centre sinon) — file atomique `fetch_add` (l'ordre
  rayon natif disperse les fronts). Param `tiles: Option<&TileOpts>` du
  dispatcher unique (CLI/quality/HQ/AA/refine = `None` ; l'ordre ne change
  AUCUN pixel, verrou `tiled_render_identical_across_priorities`). Streaming
  intra-passe GUI : le sink colorise chaque tuile, la blitte sur la passe
  précédente colorisée et envoie `TileProgress` throttlé 100 ms → la zone
  sous le curseur devient nette en premier. Sans `unsafe` : segments de
  lignes disjoints par tuile (`TileGrid::split`). Gates sink : refine
  silencieux, OrbitTraps/Wings, 1re passe (pas de base). Cancel : poll par
  tuile + par ligne (GMP/perturbation) + re-check final anti-buffer-troué.
- Coordonnées HP synchronisées vers `FractalParams`. ⚠️ L'arithmétique HP des
  zooms (`zoom_hp`/`zoom_anchored_hp`/`zoom_rect_hp`/`zoom_out_hp`) utilise `hp_arith_precision()`
  (≈ `-log2(span)+96` bits, **dynamique**), PAS le `HP_PRECISION` fixe (256 b) :
  sinon le centre est arrondi au zoom (à 1e235 il faut ~783 b) → vue fausse →
  **image uniforme** après zoom. Les sync HP↔f64 gardent 256 b (f64 = 53 b).
- Sélection rectangle pour zoom (méthode préférée), switch CPU/GPU, stats.
  Clic gauche = zoom in re-centré sur le point ; clic droit = zoom out re-centré
  sur le curseur (symétrique).
- Preview Julia au survol (touche `J` pour basculer).
- **Barres (haut, état, progression)** : `horizontal_wrapped` + groupes
  insécables `wrap_group` (largeur mémorisée de la frame précédente — un
  `ui.horizontal` ne connaît sa largeur qu'après dessin et déborde sinon).
  ⚠️ Une barre ne doit JAMAIS changer de hauteur avec l'état du rendu : la
  zone image change de taille ⇒ redimensionnement ⇒ nouveau rendu ⇒ boucle
  infinie (vécu 2026-11-18). D'où la barre de progression séparée tout en
  bas (une ligne fixe) et la hauteur « collante » `sticky_bar_min/record`
  (max par largeur de fenêtre). ⚠️ eframe est en `default-features = false` :
  `clipboard` (copie sous Wayland), `system_fonts` et winit par défaut
  (décorations adwaita : titre + croix sous Wayland) sont à garder.
- Rendu haute résolution asynchrone (Window / 4K / 8K).
- Drag-and-drop PNG pour restaurer l'état. Save (S) embed metadata JSON.
- **Historique de navigation + reprise de session** (`gui/view_history.rs`,
  logique pure testée ; `app.rs` = câblage). Entrée = snapshot
  `FractalParams` complet ; « même vue » = `view_key` : params sérialisés
  MOINS les clés des groupes de préférences (color/sampling/engine/
  perturbation/channels, frontière de `params_for_type_keeping_preferences`)
  et MOINS width/height/span_x(_hp)/iteration_max (fenêtre, sage des
  itérations) → ces changements mettent l'entrée courante à jour EN PLACE.
  Empilement : `observe` à chaque frame, vue empilée une fois stable
  `SETTLE` = 600 ms hors geste (bouton maintenu, sélection, inertie,
  `xaos_nav_dragging`) → une rafale de molette / un pan = UNE entrée ; rien
  en mode preview Julia ; changement de scène (type, preset Lyapunov,
  hybride, J, plan, PNG/TOML déposé, reset 0) → `on_new_scene` →
  `commit` immédiat (fige d'abord la vue en attente). Naviguer après un
  retour tronque l'avant ; borne 200. Restauration (`restore_params`) : même
  scène (type + formule) ⇒ géométrie/itérations de l'entrée, préférences
  COURANTES ; autre scène ⇒ entrée entière. Chaînes HP du centre intactes
  octet pour octet (taille de fenêtre différente : `ViewHp::resize`). Le
  rendu repart par `start_render` ; cache d'orbite gardé dans la même scène
  (le moteur juge), vidé sinon ; XaoS non contourné ; `on_new_view` remet
  à zéro `AutoGrowGuard` / shrink en attente. Redimensionner la fenêtre
  n'empile rien (`refresh_current`). **Session** : `on_exit` écrit
  historique + curseur + case Iter Auto + taille de fenêtre dans
  `~/.config/fractall/gui-session.json` (override
  `FRACTALL_GUI_SESSION_FILE`, écriture tmp + rename) ; relue au démarrage
  (`main_gui.rs`), fichier corrompu/illisible ⇒ vue par défaut sans panique ;
  `fractall-gui --fresh` l'ignore (et l'écrase à la sortie).

### Raccourcis clavier

| Touche | Action |
|--------|--------|
| F1–F12 | Type fractale (F1=Mandelbrot, F2=Julia, F3=JuliaSin, …, F12=Tricorn) |
| C | Cycler palette |
| R | Cycler color_repeat |
| J | Bascule Julia / preview |
| S | Screenshot PNG (avec metadata) |
| +/= | Zoom avant (×1.5) |
| - | Zoom arrière (×1.5) |
| 0 | Reset vue (ignoré si focus sur champ itérations) |
| Backspace / Alt+← | Vue précédente (historique ; bouton ◀) — inactif si un champ texte a le focus |
| Alt+→ | Vue suivante (bouton ▶) |
| Enter | Valider champ itérations |
| Molette | Zoom in/out continu ANCRÉ au curseur (`zoom_anchored_hp`, ≈×1.2/cran) |
| Clic gauche + drag | Sélection rectangle zoom |
| Clic droit | Zoom arrière |
| Clic milieu + drag | Pan (jamais de zoom molette tant qu'il est maintenu) |

+/=/-/0 désactivés en mode preview Julia.

## Threading

- **Rendu progressif** : thread dédié par passe, mpsc channels.
- **Recolorisation** : thread séparé + versioning pour annuler les résultats
  obsolètes au glissement rapide du slider.
- **Rendu HQ** : thread dédié, messages `Progress/Done/Error`.
- **Parallélisme pixel** : `rayon::par_chunks_mut`. `AtomicBool` pour
  l'annulation propre.

## Quality (`fractall-quality`)

Compare le chemin perturbation à un rendu GMP pur pixel-par-pixel. Garde-fou
de régression à chaque modification de `perturbation/`.

```bash
cargo run --release --bin fractall-quality -- list
cargo run --release --bin fractall-quality -- preset seahorse-valley
cargo run --release --bin fractall-quality -- suite --width 256 --height 256
cargo run --release --bin fractall-quality -- compare --type 3 \
    --center-x-hp "-0.7436..." --center-y-hp "0.1318..." --zoom 1e10
cargo run --release --bin fractall-quality -- gpu-suite   # parité GPU↔CPU (G9.5)
```

**Parité GPU↔CPU** (`gpu-suite` / `gpu-compare`, 2026-07-15) : rend la même
vue via `GpuRenderer::render_dispatch` et la juge contre le **GMP pur**
(⚠️ PAS le CPU Auto : le f64-std diverge lui-même de ~6 % de la vérité à
5000 iters sur bord chaotique), mêmes métriques/verdicts, rapports sous
`quality-reports/gpu/` (`pert.*` = GPU, `gmp.*` = juge). Presets
`GPU_PRESETS` (échelle seahorse 1e2→1e8). **État (kernel perturbation
F3-strict f64 natif, 2026-07-15) : WARN p99=0 div ≈ 0.001 sur tout le range
perturbation 1e4→1e8 (= niveau CPU-perturbation) ; FAIL 1e2/1e3 (shaders std
f32, gap restant G9.5)**. C'est le harnais d'acceptance du kernel deep G9.4.
Les tests unitaires du bin quality tournent en CI.

**Sorties** dans `quality-reports/<preset>/` :
- `pert.png` / `gmp.png` (metadata pour drag-and-drop fractall-gui).
- `diff.png` heatmap (noir=match, rouge-jaune=divergence).
- `report.md` métriques + top 10 pixels divergents.
- `suite-summary.md` global PASS/WARN/FAIL.

**Métriques** : `|iter_pert - iter_gmp|` (max, mean, rms, p50/p95/p99),
ratio divergence (>1), `|z_pert - z_gmp|`, erreur relative, désaccord
d'échappement, temps pert vs GMP.

**Seuils défaut (recalibrés G6, 2026-07-10 — robustes au bruit de bord)** :
- **PASS** : `max_iter_diff ≤ 1` ET `div_ratio ≤ 0.001` (quasi-exact, strict).
- **FAIL** : `p99_iter_diff > 1` (divergence LARGE : > 1 % des pixels divergent
  de > 1 → vrai bug) **OU** `div_ratio > 0.01` (SYSTÉMATIQUE : offset uniforme,
  signature over-skip BLA…).
- **WARN** : sinon (divergence éparse — quelques pixels de bord au plancher f64,
  cf. e13/e17/seahorse-valley : `max` grand mais `p99=0`, `div_ratio` minuscule).

⚠️ Le gate ne FAIL PLUS sur le `max` outlier seul (ancien comportement) : un `max`
élevé sur quelques pixels DISPERSÉS = bruit inhérent, pas une régression. `max`
reste rapporté dans `report.md`. Override : `--pass-max-iter-diff`,
`--fail-p99-iter-diff`, `--warn-divergence-ratio`.

**17 presets** (`src/quality/presets.rs`) : Mandelbrot (seahorse 1e8,
activation 1e13, GMP perturbation 1e17, Misiurewicz 1e12, minibrot 1e18,
spirales profondes e30/e50/e100), Julia (seed -0.8+0.156i à 1e10, et
`julia-edge-e20` : bord HORS CENTRE à 1e20, sentinelle du rebase relatif à
`Z[0]` et du traqueur Julia), Burning
Ship antenna 1e9 (non-conformal BLA), Tricorn spiral 1e8, et frontières
lisses hors axes de pliage Celtic/Buffalo/PerpBS 1e9 (G3 2026-07-13 — les
frontières hirsutes de ces familles sont à sensibilité de précision extrême,
GMP-256 non convergé, donc non comparables).

**Perf** : GMP pur O(1e3-1e4) plus lent que perturbation, d'où la résolution
défaut 256×256. La suite peut prendre plusieurs minutes au-delà de 1e15.

**Boucle d'auto-amélioration** :
1. Après toute modif `perturbation/` : `fractall-quality suite`.
2. Lire `suite-summary.md`, identifier FAIL.
3. Pour chaque FAIL, lire `report.md` + top 10 divergents.
4. Localiser via coordonnées pixel ↔ modules `bla/delta/nonconformal/orbit`.
5. Patcher, relancer, vérifier l'amélioration.

## Référence Fraktaler-3.1

Source de vérité algorithmique. Submodule présent dans
`fraktaler-3-3.1/src/`. Analyse complète dans `docs/fraktaler-3-analysis.md` ;
roadmap parité + perf dans `TODO.md` (goals G1/G2).

Fichiers F3 clés à consulter :
- `hybrid.h` / `hybrid.cc` — hybrid_period (atom-domain), hybrid_center
  (Newton), hybrid_size (matrice K), hybrid_render, opcodes
- `bla.h` / `bla.cc` — blaR2 multi-niveaux, merge F3, dual-number BLA
- `engine.cc` — pipeline complet newton_thread + render_thread
- `floatexp.h` / `softfloat.h` / `float128.h` — hiérarchie de types
- `wisdom.cc` / `wisdom.h` — sélection auto de précision (TOML persistant)
- `dual.h` — propagation Jacobienne par dual-numbers
