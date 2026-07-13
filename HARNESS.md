# HARNESS.md — Protocole d'auto-amélioration du moteur fractall

> **Mission** : égaler Fraktaler-3, puis le dépasser, sur trois axes mesurables :
> **vitesse**, **génération** (couverture/correction), **qualité d'image**.
> Ce document est le protocole ; l'outil est `scripts/harness.py` ; la mémoire
> est `harness/` (scorecards versionnés). La boucle s'exécute via le skill
> `/improve` (`.claude/skills/improve/`).

## Vue d'ensemble

```
┌────────────── UNE ITÉRATION (skill /improve) ──────────────┐
│ 1. MESURER   scripts/harness.py score --tier quick         │
│              → harness/history/<date>-<sha>.json           │
│              → SCORECARD.md (tableau + gaps triés)         │
│ 2. CHOISIR   gap #1 selon les priorités (voir §Priorités)  │
│ 3. DIAGNOSTIQUER  confirmer QUEL path s'exécute (⚠ leçon   │
│              G2) avant toute analyse de perf               │
│ 4. PATCHER   Fraktaler-3.1 = source de vérité algorithmique│
│ 5. VÉRIFIER  unit + golden + quality suite + re-score      │
│              → AUCUNE régression sur les autres axes       │
│ 6. COMMITTER + archiver le scorecard + mettre à jour TODO  │
└────────────────────────────────────────────────────────────┘
```

## Les trois axes et leurs métriques

### 1. Vitesse (`speed`)
Head-to-head **fractall vs F3**, mêmes cas, même taille, mêmes itérations,
même escape radius. Métrique par cas : `ratio = fractall_s / f3_s`
(< 1.0 = fractall plus rapide). Agrégats : **médiane géométrique des ratios**
(robuste aux outliers), pire ratio, nb de cas où fractall bat F3.
- Timing = wall-clock du process complet, N runs (défaut 1 en quick, 3 en
  standard, médiane retenue), machine notée dans le scorecard (les ratios
  restent comparables entre machines, PAS les temps absolus).
- ⚠️ Ne jamais comparer des temps pris sur des machines différentes ; le
  baseline est par-machine (`meta.machine`).
- **Cible « égaler »** : geomean ≤ 2.0 (critère TODO pilier 2).
  **Cible « dépasser »** : geomean ≤ 1.0.

### 2. Génération / correction (`parity`)
Réutilise `scripts/compare_f3.py` (EXR N0/NF, Δsmooth-iter, détecteur
F3-dégénéré). Métriques : % de cas pixel-équivalents (< 0.01 % rel),
Δmean relatif par cas, inside_mismatch, statuts (ok / timeout / fail /
f3_degenerate). Les victoires fractall (F3 dégénéré) comptent POUR fractall.
- **Cible « égaler »** : 0 `fractall_fail`, 0 divergence non élucidée sur les
  84 `toml/`. **Cible « dépasser »** : rendre correctement les cas où F3 est
  dégénéré ou timeout (compteur `beyond_f3.wins`).

### 3. Qualité d'image (`quality`)
`fractall-quality suite` : le chemin perturbation comparé au **ground truth
GMP pur** pixel par pixel (le juge n'est pas F3 ici, c'est la mathématique).
Métriques : verdicts PASS/WARN/FAIL par preset, div_ratio, max_iter_diff.
Plus goldens (`tests/golden_images.rs`, pixel-exact) = non-régression absolue.
- **Cible** : 0 FAIL sur la suite (seahorse-valley est le FAIL connu, cf.
  TODO G3) ; goldens toujours verts.
- Au-delà : AA multi-sample, smooth coloring, modes hors-F3 — toute nouvelle
  feature de qualité DOIT arriver avec sa métrique dans la suite ou un golden.

## L'outil : `scripts/harness.py`

```bash
python3 scripts/harness.py score [--tier quick|standard|full] [--axes speed,parity,quality,goldens]
python3 scripts/harness.py baseline          # fige le score courant comme référence
python3 scripts/harness.py gaps              # ré-affiche les gaps du dernier score
python3 scripts/harness.py preflight         # vet le corpus SOUS cap mémoire (anti-crash-OS)
python3 scripts/harness.py quarantine list   # cas exclus des sweeps (crash/OOM connu)
python3 scripts/harness.py journal           # trace des incidents (OOM, crash, abort)
```

- **Tiers** : `quick` ≈ 10 cas représentatifs à 256², 1 run (~5 min, l'outil
  du cycle interne) ; `standard` ≈ 25 cas, 3 runs (validation avant commit
  perf) ; `full` = 84 cas (jalon, plusieurs heures).
- **Sorties** :
  - `harness/history/<UTC>-<sha>.json` — scorecard machine-readable complet
    (versionné : c'est la mémoire longue de la boucle).
  - `harness/baseline.json` — référence courante (mise à jour explicite via
    `baseline`, jamais implicite).
  - `SCORECARD.md` (racine) — dernier score lisible : tableau par axe,
    delta vs baseline, **gaps triés** (la liste de travail).
  - Les rendus/artefacts lourds restent dans `bench/` (gitignoré).
- **Binaire F3** : auto-détection `fraktaler-3-3.1/fraktaler-3-3.1.linux`
  (préféré sur Linux) puis `fraktaler-3.macos`, override `F3_BIN=…`. Sans F3,
  l'axe `speed`/`parity` est marqué `f3_unavailable` (le reste tourne).
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
  les sweeps skippent ces cas, gap sévérité 2 dans le SCORECARD).

**Avant tout sweep au-delà de `quick`**, passer `preflight` : il rend chaque
cas un-par-un sous cap mémoire, mesure le pic RSS (`/usr/bin/time -v`),
classe dans `bench/harness/preflight/preflight-report.md` et quarantaine
les gourmands/plantants — sans jamais risquer l'OS.

## Priorités de choix du gap (ordre strict)

1. **Correction** — toute sortie fausse silencieuse, golden rouge, FAIL
   quality, divergence parité non élucidée. On ne travaille JAMAIS la vitesse
   avec une correction ouverte sur le même path.
2. **Robustesse** — artefacts (anneaux, blobs, uniformes), crashs, warnings
   de précision.
3. **Vitesse** — pire ratio vs F3 d'abord (le geomean suit) ; ne compte que
   si mesuré sur le path perturbation réel (vérifier `FRACTALL_PERTURB_STATS`).
4. **Qualité/différenciation** — AA, coloring, features superset (G4
   hybrides…), maintenabilité (G5).

À priorité égale : le gap au **meilleur levier** (impact × certitude de
diagnostic / effort). En cas de doute, préférer le cas le plus petit qui
reproduit le problème.

## Invariants (ne JAMAIS transiger)

- **Zéro régression échangée** : un gain de vitesse qui dégrade parité,
  quality suite ou goldens est un échec — on revert.
- **Goldens verts avant tout commit** ; s'ils changent légitimement,
  régénérer + **revue visuelle** obligatoire + justification dans le message
  de commit.
- **Confirmer le path exécuté avant d'optimiser** (leçon G2 : des heures
  perdues à optimiser la perturbation quand les cas rendaient en GMP).
  Instrument : `FRACTALL_PERTURB_STATS`, compteurs par fonction.
- **F3 = source de vérité algorithmique** (`fraktaler-3-3.1/src/`,
  `fraktaler-3-3.1/fraktaler-3-analysis.md`) — sauf cas F3-dégénéré prouvé.
- **Un seul chemin de rendu CLI ↔ GUI** (cf. CLAUDE.md, invariant absolu).
- **Mesures propres** : machine idle, tier `standard` (3 runs) pour toute
  affirmation de perf committée ; méfiance envers les A/B back-to-back
  (throttling thermique, leçon M4).
- **Fichiers < ~800 lignes** (pilier 5) — le harness lui-même y compris.

## Dépasser F3 : définition opérationnelle

Le scorecard trace une section `beyond_f3` :
- `speed_wins` : cas où fractall < F3 en wall-clock.
- `correctness_wins` : cas F3-dégénéré/timeout rendus correctement par
  fractall (aujourd'hui : glitch_test_1, glitch_test_5).
- `feature_superset` : features mesurées absentes de F3 (27 palettes, 15
  coloring, 7 plane transforms, GUI, types non-escape-time, drag-drop PNG)
  — protégées par goldens/tests, listées statiquement.
- Franchir « dépasser » = `geomean_ratio ≤ 1.0` ET `correctness_wins ≥ 2`
  ET 0 régression du superset.

## Protocole de vérification (étape 5, obligatoire)

```bash
cargo test --release --bin fractall-cli          # ~178 unit
cargo test --release --test golden_images        # pixel-exact
cargo run --release --bin fractall-quality -- suite
python3 scripts/harness.py score --tier quick    # delta vs baseline
```
Un patch perf ⇒ ajouter `--tier standard` sur les cas touchés. Un fix de
correction ⇒ ajouter un golden ou un preset quality qui fige le cas
(cf. G6 : chaque bug résolu laisse un verrou derrière lui).

## Où regarder (rappels)

| Besoin | Où |
|--------|-----|
| Protocole détaillé parité | `scripts/compare_f3.py` (docstring) + TODO G1 |
| Presets quality | `src/quality/presets.rs` (ajouter = 1 struct literal) |
| Timing interne par phase | `FRACTALL_PERTURB_STATS` (orbit/BLA/pixels/post) |
| Timing CLI parseable | `main.rs:710` « Génération terminée en Xs (rendu: Ys…) » |
| Algorithmes F3 | `fraktaler-3-3.1/fraktaler-3-analysis.md` + `src/` F3 |
| Roadmap / gaps connus | `TODO.md` (G1–G7) |
