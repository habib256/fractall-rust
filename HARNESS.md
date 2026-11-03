# HARNESS.md — Protocole d'auto-amélioration du moteur fractall

> **Mission** : égaler Fraktaler-3, puis le dépasser, sur des axes mesurables :
> **vitesse**, **génération** (couverture/correction), **qualité d'image** —
> et, depuis 2026-11, **savoir OÙ va le temps** avant chaque optimisation.
> Ce document est le protocole ; l'outil est `scripts/harness.py` (package
> `scripts/harnesslib/`) ; la mémoire est `harness/` (scorecards versionnés).
> La boucle s'exécute via le skill `/improve` (`.claude/skills/improve/`).

## Vue d'ensemble

```
┌────────────── UNE ITÉRATION (skill /improve) ──────────────┐
│ 0. DOCTOR    scripts/harness.py doctor  (prérequis, axes   │
│              mesurables ICI, baseline/machine, crash en     │
│              suspens) — début de session / nouvelle machine │
│ 1. MESURER   scripts/harness.py score --tier quick         │
│              → harness/history/<date>-<sha>.json           │
│              → SCORECARD.md (tableaux + PROFIL + gaps)     │
│ 2. CHOISIR   gap #1 selon les priorités (voir §Priorités)  │
│ 3. DIAGNOSTIQUER  le PROFIL dit déjà quel path tourne et   │
│              quelle phase domine (leçon G2) ; affiner avec │
│              FRACTALL_PERTURB_STATS / trend --case         │
│ 4. PATCHER   Fraktaler-3.1 = source de vérité algorithmique│
│ 5. VÉRIFIER  unit + golden + quality suite + re-score      │
│              → AUCUNE régression sur les autres axes       │
│ 6. COMMITTER + archiver le scorecard + mettre à jour TODO  │
└────────────────────────────────────────────────────────────┘
```

## Les axes et leurs métriques

Deux familles. **Contre F3** (speed-ratio, parity) : exigent le binaire F3 et
les deps Python EXR — sans eux ces axes sont marqués `f3_unavailable` /
`f3_deps_missing` et **tout le reste tourne**. **Intrinsèques** (speed
auto-relative, quality, fuzz, goldens, profile, bench) : ne dépendent que du
repo et de cargo — ce sont eux qui font tourner la boucle sur une machine
neuve ou un conteneur CI, et eux qui détectent une régression de vitesse sans
F3.

### 1. Vitesse (`speed`) — ratio F3 **et** vitesse intrinsèque
Head-to-head **fractall vs F3**, mêmes cas, même taille, mêmes itérations,
même escape radius. Métrique par cas : `ratio = fractall_s / f3_s`
(< 1.0 = fractall plus rapide). Agrégats : **médiane géométrique des ratios**
(robuste aux outliers), pire ratio, nb de cas où fractall bat F3.
- Timing = wall-clock du process complet, N runs (défaut 1 en quick, 3 en
  standard, **médiane** retenue ; `fractall_s_min` et `fractall_s_spread` =
  (max−min)/médiane sont stockés : le spread est le **bruit de mesure**).
- ⚠️ Ne jamais comparer des temps pris sur des machines différentes ; le
  baseline est par-machine (`meta.machine`).
- **Vitesse intrinsèque (sans F3, 2026-11)** : chaque `fractall_s` est comparé
  au MÊME cas de la baseline (même machine, même tier) → `speed_self`
  (geomean cur/base, pire cas) et gaps « RÉGRESSION » par cas quand
  `ratio > max(1.15, 1 + spread)` ET `Δ > 50 ms` (plancher : sous 50 ms le
  chrono d'un process n'est pas fiable). Geomean > 1.10 ⇒ gap agrégé. C'est le
  détecteur de régression qui marche sur TOUTE machine ; le ratio F3 reste le
  juge « égaler/dépasser ».
- **Cible « égaler »** : geomean ≤ 2.0 (critère TODO pilier 2).
  **Cible « dépasser »** : geomean ≤ 1.0 (atteint : 0.19 quick, 80/80 full).

### 2. Génération / correction (`parity`)
Réutilise `scripts/compare_f3.py` (EXR N0/NF, Δsmooth-iter, détecteur
F3-dégénéré). Métriques : % de cas pixel-équivalents (< 0.01 % rel),
Δmean relatif par cas, inside_mismatch, statuts (ok / timeout / fail /
f3_degenerate). Les victoires fractall (F3 dégénéré) comptent POUR fractall.
- **Cible « égaler »** : 0 `fractall_fail`, 0 divergence non élucidée sur les
  84 `toml/`. **Cible « dépasser »** : rendre correctement les cas où F3 est
  dégénéré ou timeout (compteur `beyond_f3.wins`).
- Prérequis : F3 + `numpy`, `OpenEXR`, `Pillow` (`harness.py doctor` le dit).

### 3. Qualité d'image (`quality`)
`fractall-quality suite` : le chemin perturbation comparé au **ground truth
GMP pur** pixel par pixel (le juge n'est pas F3 ici, c'est la mathématique).
Métriques : verdicts PASS/WARN/FAIL par preset (16 presets), div_ratio,
max_iter_diff. Plus goldens (`tests/golden_images.rs`, pixel-exact) =
non-régression absolue.
- **Cible** : 0 FAIL sur la suite ; goldens toujours verts. État 2026-07-18 :
  16/16 PASS (seahorse-valley PASS à ≤ 96², WARN plancher-f64 au-delà — adjugé
  vs GMP : fractall ~600× plus fidèle que F3 sur ce cas).
- Au-delà : AA multi-sample, smooth coloring, modes hors-F3 — toute nouvelle
  feature de qualité DOIT arriver avec sa métrique dans la suite ou un golden.

### 4. Fuzz (`fuzz`, G8.2 2026-07-14)
N sondes **aléatoires déterministes** pert-vs-GMP (96², même juge que quality).
Les scènes (`scripts/fuzz_scenes.py`) sont des points frontière **stables**
(échappement invariant à un bump 2⁻⁴⁶ de c — ground truth bien conditionnée,
les frontières hirsutes sont filtrées) tirés sur 6 familles, zoom 1e5-1e11.
Vise les classes de divergence que les presets figés ratent (la classe
e13/dd-sensibilité avait été trouvée PAR ACCIDENT). Quick=3, standard=6,
full=8 sondes. La **seed est committée** (`FUZZ_SEED_DEFAULT`, affichée dans
le scorecard) → chaque score rejoue les mêmes scènes ; la faire **tourner au
rebaseline** (`--fuzz-seed`) pour élargir la couverture au fil du temps.
- Gap : FAIL fuzz = sévérité 1 (correction) ; WARN = 4 ; ERROR (rendu KO) = 2.

### 5. Profil — où va le temps (`profile`, 2026-11)
Rend chaque cas du tier **une fois** en capturant les diagnostics du moteur
(`[FRACTALL] path=… ns/iter=…`, `[PERTURB PERF] orbit= pixels= post=`,
`[WISDOM] tier= variants=`, `[DD-ESCALATION]`, `[RELIABILITY] flagged=`) et la
ligne CLI « Génération terminée en Xs (rendu: Ys, sauvegarde: Zs) ». Par cas :
**path exécuté**, ventilation du wall-clock en `orbit / pixels / post / save /
other` (part %), phase dominante, ns/iter effectif (skips BLA inclus), % de
pixels flaggés par le détecteur de fiabilité, escalade dd. Agrégats :
répartition cumulée + comptage des phases dominantes.
- C'est l'étape « **confirmer le path exécuté avant d'optimiser** » (leçon G2)
  rendue systématique et versionnée : la première ligne de tout diagnostic
  perf est désormais dans le SCORECARD, pas à reconstruire à la main.
- Gaps : `save_s > render_s` (sauvegarde qui domine — classe de bug prouvée,
  PNG niveau 6 2026-07-16) = sév. 3 ; `ns/iter` +15 % vs baseline **à path
  identique** (même machine) = sév. 3. L'escalade dd est affichée, pas
  gappée (elle peut être légitime) — mais c'est un candidat perf (voir
  TODO G14 : sur e50 le post-traitement fiabilité = +63 % de wall).
- Logs bruts : `bench/harness/profile/<cas>.log`.

### 6. Bench — débits par technique (`bench`, 2026-11)
`fractall-cli --wisdom-bench` lancé **isolé** (`FRACTALL_WISDOM_FILE` dans
`bench/harness/bench/` — le wisdom machine de l'utilisateur,
`~/.config/fractall/wisdom.toml`, n'est pas touché) → iters/s effectifs et
ns/iter par technique : `cpu_std_f64`, `cpu_perturb_f64`, `cpu_perturb_exp`,
`cpu_perturb_dd` (+ `gpu_*` si un GPU est là). Comparables d'un commit à
l'autre **sur la même machine** : débit < 85 % de la baseline = gap sév. 3.
C'est le micro-benchmark du moteur, sans le bruit du démarrage process ni de
l'orbite. (~30 s ; `--axes` sans `bench` pour l'omettre.)

### Axe wisdom-optimality (`wisdom-optimality`, G9)
Commande séparée (pas dans `score`) : pour un échantillon 2-régimes, le plan
CHOISI par le wisdom (`auto`) est chronométré contre les plans forcés (`bla`,
`lla`) ; FAIL = plus lent > 25 % **à sortie identique** ; ADJUDICATE = plus
lent mais l'alternative rend AUTRE CHOSE (tradeoff correction possible →
vérifier vs GMP, puis persister dans `harness/wisdom-adjudications.json` :
marqué `déjà-adjugé✓` tant que le régime ne change pas). Rapport
`bench/harness/wisdom-opt/wisdom-opt.md`.

### Arbitre 3-voies (`adjudicate`, G8.2 2026-07-14)
**F3 n'est PAS ground truth aux zooms modérés** (wisdom défaut = tier float
24 b, cf. TODO G3 diag 2026-07-13). Une divergence parité ne dit donc pas qui
a tort. `harness.py adjudicate <stem>` rend le cas en 3 voies (fractall-pert /
F3 / GMP per-pixel, via `scripts/three_way_gmp.py`) et **persiste** le verdict
(`fractall_wrong` / `f3_wrong` / `shared` / `both_match_gmp`) dans
`harness/adjudications.json` (versionné). Les scores suivants annotent les
gaps parité avec ce verdict — un cas `f3_wrong` est déclassé (sév. 2→4).
Fin du re-litige manuel. ⚠️ GMP per-pixel ≈ 1 µs/iter × size² : `--size 64`
sur les cas profonds. Adjugés au 2026-07-14 : test5 (`both_match_gmp`), rug
(`shared` — mais fractall big=156 vs F3 big=3852 : ~25× plus fidèle même là).

## L'outil : `scripts/harness.py`

```bash
python3 scripts/harness.py doctor            # prérequis, axes mesurables ICI, baseline
python3 scripts/harness.py score [--tier quick|standard|full] \
        [--axes speed,parity,quality,fuzz,goldens,profile,bench] [--cases a,b]
python3 scripts/harness.py render [history]  # re-génère SCORECARD.md sans re-mesurer
python3 scripts/harness.py trend [--case e50] [--bench] [--all-machines]
python3 scripts/harness.py baseline          # fige le dernier score comme référence
python3 scripts/harness.py gaps              # ré-affiche les gaps du dernier score
python3 scripts/harness.py adjudicate <stem> # arbitre 3-voies d'un cas parité
python3 scripts/harness.py wisdom-optimality # le wisdom n'est jamais battu >10 %
python3 scripts/harness.py preflight         # vet le corpus SOUS cap mémoire
python3 scripts/harness.py quarantine list   # cas exclus des sweeps (crash/OOM)
python3 scripts/harness.py slow-safe list    # cas corrects hors budget temps
python3 scripts/harness.py journal           # trace des incidents (OOM, crash, abort)
python3 scripts/test_harness_guard.py        # verrous du harness lui-même (CI)
```

- **Structure** : `harness.py` est une enveloppe ; la logique vit dans
  `scripts/harnesslib/` (`config` chemins/registres/tiers · `guard` garde-fou ·
  `axes_f3` speed+parity · `axes_truth` quality+fuzz+goldens · `axes_perf`
  profile+bench · `gaps` · `scorecard` · `commands*` · `wisdom_opt` · `cli`),
  chaque fichier < 800 lignes (pilier 5 — le harness respecte l'invariant
  qu'il impose). `compare_f3` (numpy/OpenEXR) est importé **paresseusement** :
  son absence éteint parity/adjudicate, pas le harness.
- **Tiers** : `quick` ≈ 10 cas représentatifs à 256², 1 run (~5-10 min, l'outil
  du cycle interne) ; `standard` ≈ 25 cas, 3 runs (validation avant commit
  perf) ; `full` = 84 cas (jalon, plusieurs heures — `preflight` d'abord).
- **Sorties** :
  - `harness/history/<UTC>-<sha>.json` — scorecard machine-readable complet
    (versionné : c'est la mémoire longue de la boucle ; schéma 2 depuis
    2026-11 : sections `profile`, `bench`, `speed_self` ; `trend` lit les deux).
  - `harness/baseline.json` — référence courante (mise à jour explicite via
    `baseline`, jamais implicite).
  - `SCORECARD.md` (racine) — dernier score lisible : tableaux par axe, delta
    vs baseline, temps fractall par cas, **profil** (où va le temps), bench,
    tendance (6 derniers scores même machine), **gaps triés** (la liste de
    travail).
  - Les rendus/artefacts lourds restent dans `bench/` (gitignoré).
- **Binaire F3** : auto-détection `fraktaler-3-3.1/fraktaler-3-3.1.linux`
  (préféré sur Linux) puis `fraktaler-3.macos`, override `F3_BIN=…` — filtrée
  par **format binaire** (ELF sous Linux, Mach-O sous macOS : le `.macos`
  versionné n'est plus « trouvé » sous Linux, où il produisait 10 `fail`
  silencieux). Sans F3, speed-ratio/parity sont `f3_unavailable` (le reste
  tourne).
  - **Build Linux** : `bash scripts/build_f3_linux.sh` compile un binaire F3
    **batch GUI-free** (le mode raw-EXR N0/NF passe par `batch()`, sans SDL/GL/
    imgui — submodules absents ici) et l'écrit à `fraktaler-3-3.1.linux`
    (gitignoré, par-machine). Installe les deps apt (gmp/mpfr/mpc/openexr/glm/
    sdl2/…) ; `SKIP_APT=1` si déjà présentes. macOS : `make SYSTEM=macos-batch`.

### Garde-fou crash / mémoire (sweeps étendus)

Un cas du corpus peut saturer la RAM et **faire tomber l'OS** pendant un sweep
`standard`/`full`. Trois mécanismes intégrés évitent que la boucle reste
coincée dessus :

- **Breadcrumb** `harness/inflight.json` (transient, gitignoré) : écrit avant
  chaque cas, effacé après. `score`/`preflight` le vérifient au démarrage — un
  résidu = crash non nettoyé du run précédent → journalisé + **quarantaine auto**.
  Une interruption GRACIEUSE (Ctrl-C / SIGTERM) est catchée et nettoie le
  breadcrumb (journal `interrupted`, PAS de quarantaine) : seule une mort NON
  catchable (SIGKILL, OOM-killer, panne OS) laisse le résidu → `died_uncleanly`.
- **Cap mémoire** RLIMIT_AS (défaut 85 % RAM ; `--mem-limit-mb N`, `0`/
  `--no-mem-limit` pour couper) : un runaway est tué (`aborted`/`killed_oom`,
  loggé) au lieu de planter l'OS. Faux positif éventuel = visible dans le
  journal, jamais un plantage silencieux.
- **Journal** `harness/crash-journal.jsonl` (gitignoré) + **quarantaine**
  `harness/quarantine.json` (versionné, commit délibéré comme la baseline ;
  les sweeps skippent ces cas, gap sévérité 2 dans le SCORECARD). Seuls les
  crashs du binaire **fractall** quarantainent ; un abort côté F3 (phase
  `speed_f3`) reste un simple `f3_aborted` de l'axe vitesse.
- **Registres de déconflation** : `harness/slow-safe.json` (cas CORRECTS et
  memory-safe mais hors budget temps — ex. e52465, 660 s d'orbite GMP ;
  skippés SANS gap) et `harness/f3-degenerate.json` (F3 rend un fast-path
  uniforme faux ; le ratio vitesse n'a pas de sens, exclu du geomean,
  entretenu par l'axe parité). `harness/resolved.json` = tombstones des cas
  sortis de quarantaine après fix vérifié.

**Avant tout sweep au-delà de `quick`**, passer `preflight` : il rend chaque
cas un-par-un sous cap mémoire, mesure le pic RSS (`/usr/bin/time -v`),
classe dans `bench/harness/preflight/preflight-report.md` et quarantaine
les gourmands/plantants — sans jamais risquer l'OS.

## Priorités de choix du gap (ordre strict)

1. **Correction** — toute sortie fausse silencieuse, golden rouge, FAIL
   quality/fuzz, divergence parité non élucidée. On ne travaille JAMAIS la
   vitesse avec une correction ouverte sur le même path.
2. **Robustesse** — artefacts (anneaux, blobs, uniformes), crashs, warnings
   de précision, cas quarantainés.
3. **Vitesse** — régression intrinsèque vs baseline d'abord (c'est un recul
   certain), puis pire ratio vs F3 (le geomean suit), puis la **phase
   dominante du profil** sur les cas les plus lents ; ne compte que si mesuré
   sur le path réel (colonne `path` du profil).
4. **Qualité/différenciation** — AA, coloring, features superset (G4
   hybrides…), maintenabilité (G5).

À priorité égale : le gap au **meilleur levier** (impact × certitude de
diagnostic / effort). En cas de doute, préférer le cas le plus petit qui
reproduit le problème. Pour la vitesse, le levier se lit dans le profil :
une phase à 60 % du wall sur 8 cas/10 vaut plus qu'un cas isolé.

## Invariants (ne JAMAIS transiger)

- **Zéro régression échangée** : un gain de vitesse qui dégrade parité,
  quality suite, fuzz ou goldens est un échec — on revert.
- **Goldens verts avant tout commit** ; s'ils changent légitimement,
  régénérer + **revue visuelle** obligatoire + justification dans le message
  de commit.
- **Confirmer le path exécuté avant d'optimiser** (leçon G2 : des heures
  perdues à optimiser la perturbation quand les cas rendaient en GMP).
  Instrument : colonne `path` + ventilation du **profil**, puis
  `FRACTALL_PERTURB_STATS`, compteurs par fonction.
- **F3 = source de vérité algorithmique** (`fraktaler-3-3.1/src/`,
  `fraktaler-3-3.1/fraktaler-3-analysis.md`) — sauf cas F3-dégénéré prouvé ;
  et **GMP = juge de correction** (F3 n'est pas ground truth aux zooms modérés).
- **Un seul chemin de rendu CLI ↔ GUI** (cf. CLAUDE.md, invariant absolu).
- **Mesures propres** : machine idle, tier `standard` (3 runs) pour toute
  affirmation de perf committée ; méfiance envers les A/B back-to-back
  (throttling thermique, leçon M4) ; un delta inférieur au `spread` mesuré
  n'est pas un résultat. Les ratios **ne se comparent qu'à machine égale**
  (`doctor` signale une baseline d'une autre machine).
- **Fichiers < ~800 lignes** (pilier 5) — le harness lui-même y compris
  (package `harnesslib/`).
- **Le harness est testé** : `scripts/test_harness_guard.py` (réconciliation
  journal↔quarantaine, gates de régression machine-sensibles, parseur du
  profil, seuils de vitesse intrinsèque) tourne en CI et DOIT rester vert.

## Dépasser F3 : définition opérationnelle

Le scorecard trace une section `beyond_f3` :
- `speed_wins` : cas où fractall < F3 en wall-clock.
- `correctness_wins` : cas F3-dégénéré/timeout rendus correctement par
  fractall (aujourd'hui : glitch_test_1, glitch_test_5).
- `feature_superset` : features mesurées absentes de F3 (27 palettes, 15
  coloring, 7 plane transforms, GUI, types non-escape-time, drag-drop PNG,
  vidéo, hybrides opcodes) — protégées par goldens/tests, listées
  statiquement dans TODO §« Ne pas régresser ».
- Franchir « dépasser » = `geomean_ratio ≤ 1.0` ET `correctness_wins ≥ 2`
  ET 0 régression du superset. **Atteint (2026-07-14, full 80/80).** La barre
  suivante n'est plus F3 : c'est notre propre baseline (`speed_self`,
  `bench`) et les moteurs GPU (FractalShark, cf. TODO G8.2/G14).

## Protocole de vérification (étape 5, obligatoire)

```bash
cargo test --release --lib                       # ~460 unit (cible lib)
cargo test --release --test golden_images        # 24 cas pixel-exact
cargo test --release --test deep_navigation      # types spéciaux e2e (~9 s)
cargo run --release --bin fractall-quality -- suite
python3 scripts/test_harness_guard.py            # si le harness a bougé
python3 scripts/harness.py score --tier quick    # delta vs baseline + profil
```
Un patch perf ⇒ ajouter `--tier standard` sur les cas touchés et lire
`trend --case <stem>` (le gain doit dépasser le spread). Un fix de
correction ⇒ ajouter un golden ou un preset quality qui fige le cas
(cf. G6 : chaque bug résolu laisse un verrou derrière lui).

## Démarrer sur une machine neuve (checklist)

1. `cargo build --release` (GMP/MPFR/MPC natifs ou bundle `gmp-mpfr-sys`).
2. `python3 scripts/harness.py doctor` → liste des axes mesurables ; suivre
   les « commandes recommandées » (build F3, deps EXR, baseline).
3. `python3 scripts/harness.py score --tier quick` — sans F3, le scorecard
   contient déjà : temps par cas, profil, bench, quality, fuzz, goldens.
4. `python3 scripts/harness.py baseline` → la baseline de CETTE machine
   (les régressions intrinsèques se mesurent dès le score suivant).
5. Si F3 est voulu : `bash scripts/build_f3_linux.sh` puis
   `pip3 install numpy OpenEXR Pillow`, re-`doctor`.

## Où regarder (rappels)

| Besoin | Où |
|--------|-----|
| Protocole détaillé parité | `scripts/compare_f3.py` (docstring) + TODO G1 |
| Presets quality | `src/quality/presets.rs` (ajouter = 1 struct literal) |
| Où va le temps (par cas) | `SCORECARD.md` §Profil, `bench/harness/profile/<cas>.log` |
| Timing interne par phase | `FRACTALL_PERTURB_STATS` (orbit/BLA/pixels/post), `[WISDOM]` via `FRACTALL_WISDOM=1` |
| Coût du détecteur de fiabilité | `FRACTALL_RELIABILITY=off|observe|escalate` (A/B), colonne `fiab. %` |
| Timing CLI parseable | `main.rs` « Génération terminée en Xs (rendu: Ys…) » |
| Débit par tier numérique | `SCORECARD.md` §Bench, `trend --bench` |
| Évolution d'un cas | `harness.py trend --case <stem>` |
| Algorithmes F3 | `fraktaler-3-3.1/fraktaler-3-analysis.md` + `src/` F3 |
| Algorithmes Imagina (post-BLA) | `docs/imagina-algorithms-analysis.md` |
| Roadmap / gaps connus | `TODO.md` (G1–G14 ; G14 = plan des optimisations) |
