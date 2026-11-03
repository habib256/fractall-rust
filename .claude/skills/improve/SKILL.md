---
name: improve
description: Exécute UNE itération de la boucle d'auto-amélioration du moteur fractall (mesure → gap → diagnostic → patch → vérification → commit). Utiliser quand l'utilisateur demande d'améliorer le moteur, de continuer la boucle, ou invoque /improve. Argument optionnel = axe ou cas à cibler (ex: "speed", "seahorse-valley").
---

# /improve — une itération de la boucle d'auto-amélioration

Protocole complet : `HARNESS.md` (racine). Roadmap des gaps connus : `TODO.md`.
Ne JAMAIS sauter l'étape 5 (vérification). Une itération = un commit (ou un
revert propre + une note TODO si l'hypothèse échoue).

## Étape 0a — Doctor (début de session / machine neuve)

```bash
python3 scripts/harness.py doctor
```

Dit en 2 s ce qui est mesurable ICI : F3 présent (axes speed-ratio/parity),
deps Python EXR, binaires bâtis, baseline de CETTE machine ou d'une autre
(→ pas de delta vitesse tant qu'on n'a pas re-`baseline`), crash en suspens,
quarantaine/slow-safe. Suivre ses « commandes recommandées ». **Sans F3, la
boucle tourne quand même** : vitesse intrinsèque (temps fractall par cas vs
baseline même machine), quality, fuzz, goldens, profil, bench.

## Étape 0b — Garde-fou crash (TOUJOURS, avant tout sweep)

Un cas gourmand du corpus peut allouer toute la RAM et **faire tomber l'OS**
pendant un sweep étendu (`--tier standard`/`full` ou un `--cases` large). Au
redémarrage, sans trace, la boucle rejoue le même sweep et replante. Trois
mécanismes intégrés à `scripts/harness.py` évitent ça :

- **Breadcrumb** `harness/inflight.json` : écrit avant chaque cas, effacé
  après. `score`/`preflight` le vérifient au démarrage → un résidu = le cas qui
  a tué la machine ; il est journalisé et **mis en quarantaine automatiquement**.
- **Cap mémoire** (RLIMIT_AS, défaut 85 % RAM) : un runaway est tué proprement
  (`aborted`/`killed_oom`, loggé) au lieu de planter l'OS.
- **Journal** `harness/crash-journal.jsonl` + `harness/quarantine.json`.

**Réflexe au (re)démarrage de la session** — vérifier si un crash précédent
est resté en travers :

```bash
python3 scripts/harness.py journal          # incidents récents (OOM, crash…)
python3 scripts/harness.py quarantine list  # cas exclus des sweeps + raison
```

Si le journal montre un `died_uncleanly`/`killed_oom`/`aborted` : le cas est
déjà quarantainé (les sweeps le skippent). **Ne pas le réintégrer à l'aveugle** ;
l'étudier à part (petite taille, `--dd-tier`/précision, cf. TODO), corriger,
puis `quarantine remove <cas>` seulement une fois le fix vérifié.

**Avant un sweep étendu** (au-delà du tier quick — `standard`, `full`, ou un
`--cases` inhabituel), passer d'abord le corpus au crible SANS risque :

```bash
python3 scripts/harness.py preflight        # tout le corpus, sous cap mémoire
python3 scripts/harness.py preflight --tier standard   # ou un sous-ensemble
```

`preflight` rend chaque cas un-par-un à la résolution du sweep, sous cap
mémoire + breadcrumb, mesure le **pic RSS** (`/usr/bin/time -v`), classe les
cas dans `bench/harness/preflight/preflight-report.md` et quarantaine
automatiquement les gourmands (pic RSS ≥ seuil) et ceux qui plantent. Le tier
`quick` (10 cas légers) ne nécessite pas de preflight.

## Étape 1 — Mesurer

```bash
python3 scripts/harness.py score --tier quick
```

- Lire `SCORECARD.md` (généré) : agrégats par axe, **temps fractall par
  cas** (auto-ratio vs baseline), **Profil — où va le temps** (path exécuté,
  orbite/pixels/post/save %, ns/iter, % fiabilité, escalade dd), bench par
  technique, tendance, puis **gaps triés**.
- Si `$ARGUMENTS` désigne un axe/cas, filtrer les gaps dessus ; sinon prendre
  le gap #1 (priorités : correction > robustesse > vitesse > qualité —
  cf. HARNESS.md §Priorités). **Aucun gap ?** La liste de travail est alors
  `TODO.md` G14 (plan des optimisations) — choisir l'item dont le profil
  montre le plus gros levier (phase dominante × nombre de cas).
- Pas de baseline pour CETTE machine (`doctor` le dit) ? La créer d'abord :
  `python3 scripts/harness.py baseline` (→ `harness/baselines/<machine>.json`,
  la baseline de la machine de dev n'est pas touchée) puis committer — sinon
  aucune régression intrinsèque n'est détectable.
- Binaire F3 absent (axes `f3_unavailable`) ou deps EXR absentes
  (`f3_deps_missing`) ? `bash scripts/build_f3_linux.sh` + `pip3 install
  numpy OpenEXR Pillow` si la parité est le sujet ; sinon continuer, les
  axes intrinsèques suffisent au cycle vitesse/correction.

## Étape 2 — Reproduire petit

Réduire le gap au plus petit cas reproductible : un seul stem
(`--cases <stem>`), taille minimale qui montre le problème (48²–256²).
Vérifier que le gap n'est pas déjà élucidé dans `TODO.md` (G1–G7) — beaucoup
de divergences sont classées (bord chaotique, F3-dégénéré…) : ne pas
re-litiger un cas tranché.

## Étape 3 — Diagnostiquer (avant TOUT patch)

- **Confirmer QUEL path s'exécute** : colonne `path` du §Profil du SCORECARD
  (puis `FRACTALL_PERTURB_STATS=1` + compteurs pour le détail). Leçon G2 :
  des heures d'optimisation perdues sur un path qui ne tournait pas.
  Signature GMP par-pixel ≈ 1 µs/iter ; perturbation+BLA est 100-1000× plus
  rapide.
- Perf : la phase dominante est dans le profil (orbite référence / pixels /
  post / sauvegarde / autre) ; `post` élevé + `fiab. %` > 0 = coût de
  l'escalade de fiabilité (A/B : `FRACTALL_RELIABILITY=off`) ; `other` élevé
  sur un cas court = démarrage/colorisation. Suivre un cas dans le temps :
  `harness.py trend --case <stem>`. Méfiance thermique sur les A/B
  back-to-back ; un delta < `spread` (colonne bruit) n'est pas un résultat.
- Correction : comparer au ground truth GMP (`fractall-quality compare`),
  PAS seulement à F3 ; vérifier l'hypothèse F3-dégénéré (timing < 0.1 s +
  uniformité) avant d'accuser fractall.
- Consulter la référence : `fraktaler-3-3.1/fraktaler-3-analysis.md` puis le
  source F3 (`hybrid.cc`, `bla.cc`, `engine.cc`, `wisdom.cc`).

## Étape 4 — Patcher

- F3 = source de vérité algorithmique ; porter son approche, pas improviser.
- Respecter les invariants CLAUDE.md : chemin de rendu unique CLI↔GUI,
  un seul dispatcher, fichiers < ~800 lignes.
- Patch minimal et ciblé ; pas de refactor opportuniste dans le même commit.
- Toute hypothèse perf se teste d'abord en A/B isolé, AVANT de toucher le
  code quand une env/un flag le permet :
  `python3 scripts/harness.py ab --cases <stem> --env K=a,b --runs 3`
  (médiane, bruit, phases, sortie identique ou non). Une variante plus rapide
  à sortie ≠ est un tradeoff correction → `fractall-quality compare` d'abord.

## Étape 5 — Vérifier (obligatoire, dans cet ordre)

```bash
cargo test --release --lib                        # ~460 unit (cible lib, PAS --bin)
cargo test --release --test golden_images
cargo test --release --test deep_navigation       # si types spéciaux / dispatcher touchés
cargo run --release --bin fractall-quality -- suite
python3 scripts/test_harness_guard.py             # si scripts/harnesslib a bougé
python3 scripts/harness.py score --tier quick
```

- Comparer au baseline : le gap ciblé doit s'améliorer, **aucun autre axe ne
  doit régresser** (> 10 % = revert ou justification écrite).
- Les sweeps tournent sous le garde-fou de l'Étape 0 (cap mémoire + journal).
  Si un cas apparaît soudain **quarantainé** ou en `fractall_aborted`/
  `killed_oom` dans le SCORECARD (gap sévérité 2 « robustesse »), c'est une
  régression mémoire de ton patch : consulter `harness journal`, isoler,
  corriger avant de committer — ne pas laisser passer.
- Affirmation de perf → re-mesurer en `--tier standard` (3 runs, médiane)
  sur les cas touchés (`--cases a,b`), et vérifier que `ns/iter` (profil) et
  le débit `bench` bougent dans le bon sens — un gain de wall-clock sans gain
  de ns/iter est du bruit ou du démarrage.
- Goldens modifiés légitimement → `FRACTALL_UPDATE_GOLDENS=1` + **revue
  visuelle** de chaque PNG + justification dans le commit.
- Fix de correction → **poser un verrou** : nouveau golden
  (`tests/golden_images.rs`) ou preset quality (`src/quality/presets.rs`).

## Étape 6 — Committer et archiver

- Commit atomique : le fix + son verrou + `SCORECARD.md`/`harness/history/`
  régénérés. Message : cause → fix → chiffres avant/après (style des commits
  existants du repo).
- Mettre à jour `TODO.md` : cocher/annoter le goal touché ; si l'itération a
  révélé un nouveau gap, l'ajouter au bon goal.
- Si l'amélioration établit un nouveau niveau de référence :
  `python3 scripts/harness.py baseline` et committer le nouveau
  `harness/baseline.json` (jamais implicite). Un changement du FORMAT du
  scorecard se rejoue sans re-mesurer : `harness.py render`.
- Hypothèse infirmée ? Revert complet + noter l'impasse dans TODO.md (les
  fausses pistes documentées valent de l'or, cf. G3).
