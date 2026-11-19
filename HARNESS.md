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
auto-relative, quality, fuzz, precision, goldens, profile, bench) : ne dépendent que du
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
les frontières hirsutes sont filtrées) tirés sur 6 familles Mandelbrot-like
(+ Julia, cf. ci-dessous), zoom 1e5-1e11.
Vise les classes de divergence que les presets figés ratent (la classe
e13/dd-sensibilité avait été trouvée PAR ACCIDENT). Quick=3, standard=6,
full=8 sondes. La **seed est committée** (`FUZZ_SEED_DEFAULT`, affichée dans
le scorecard) → chaque score rejoue les mêmes scènes ; la faire **tourner au
rebaseline** (`--fuzz-seed`) pour élargir la couverture au fil du temps.
- **Sondes Julia** (2026-11-06, `fuzz_scenes.generate_julia`) : une pour trois
  sondes (au moins une), suite DISTINCTE ajoutée après les six familles — les
  scènes existantes d'une seed ne changent pas. La descente se fait dans le
  plan de z le long du bord d'un ensemble de Julia rempli (lapin, basilique,
  bord du cardioïde…), donc **le centre de la scène n'est jamais 0** : c'est
  ce qui manquait pour voir que la référence Julia part du centre de la vue
  (`Z[0] ≠ 0`) et que les boucles rebasaient comme si `Z[0] = 0`. Puissance
  vérifiée : 5 sondes sur 6 en FAIL (79-93 % de pixels faux) sur le binaire
  d'avant le correctif, 6 sur 6 en PASS (max_diff 0) après.
- Gap : FAIL fuzz = sévérité 1 (correction) ; WARN = 4 ; ERROR (rendu KO) = 2.

### 4 bis. Précision d'orbite — juge ×4 (`precision`, G14.3 2026-11-05)
**Angle mort que quality/fuzz ne peuvent PAS voir** : leur juge GMP est
calculé avec la MÊME formule de précision que le moteur, donc une précision
d'orbite insuffisante y passe inaperçue. Par cas (tier + 3 **sentinelles**
golden_spider/e227/evolution_trees) : rendu de production → `.fmap`, rendu
juge avec `--precision-bits 4·P` (P lu dans `[FRACTALL] prec=`) et
`FRACTALL_ORBIT_FAST_SQR=0` (arithmétique d'orbite indépendante, `mpc_sqr`),
puis comparaison des **itérations brutes** (lecteur `.fmap` stdlib, sans
numpy). Les paths sans orbite GMP (`std_f64`) sont `n/a`. Verdicts : PASS
≤ 0,05 % de pixels ≠ juge, FAIL ≥ 0,5 %, WARN entre. 96² (quick), 128²
(standard/full). Puissance vérifiée : le binaire d'AVANT la marge +64 b
(formule F3 nue) FAIL les 3 sentinelles (4-5 % de pixels faux), le binaire
corrigé les PASS (0-1 px). Juge convergé : ×2 == ×4 au pixel près.
- Gap : FAIL = sévérité 1 (correction) ; WARN = 4.

### 4 ter. Vérité — production vs tier dd (`truth`, 2026-11-06)
**Le juge qui manquait** : aucun autre axe ne compare la sortie PAR DÉFAUT à
la vérité sur les cas `toml/` (quality juge 16 presets, precision juge
l'orbite, fuzz des scènes shallow). Par cas (tier + 5 **sentinelles** e227 /
golden_spider / rug / olbaid2 / leaded_glass) : rendu de production →
`.fmap`, rendu juge `--dd-tier` (frame double-double, gardée par son traqueur
+ repli GMP par pixel, plafond de budget levé), comparaison des itérations
brutes. Mêmes seuils que precision (PASS ≤ 0,05 %, FAIL ≥ 0,5 %), même taille
(96² quick, 128² sinon). Paths `std_*` = `n/a` pour ce volet.
- **Volet path f64 standard** (`axes_stdtruth.py`, 2026-11-18, chantier de
  correction n° 1) : scènes peu profondes `std:<lieu>_<zoom>_<it>` —
  seahorse, vallée des éléphants, minibrot de l'axe réel × 1e4/1e7/1e10 ×
  1000/2500 it. (tier quick : 2500 seulement), 256×160 (la basse résolution
  reste en f64 standard en Auto), production Auto vs `--algorithm gmp` (GMP
  pur, convergé à 256 b). Mêmes seuils, cas fusionnés dans `truth.cases`.
  Avant le traqueur du path standard : seahorse 1e10/2500 **FAIL 7,76 %**,
  1e7 FAIL 1,08 %, éléphants 1e4/1e7/1e10 FAIL 1,7/1,3/3,4 %.
- Puissance vérifiée : le binaire d'AVANT les correctifs du 2026-11-06
  (troncature BLA comptée ε au lieu de l·ε, rayon harmonic 2⁻²⁴) donne e227
  **FAIL** (67 px, 0,73 % à 96²) et golden_spider / rug / olbaid2 /
  leaded_glass **WARN** (9-29 px) ; le binaire corrigé PASS les 14 cas
  (résidus : e113 1, e227 3, rug 4 px).
- Le dd n'est pas la vérité absolue (plancher à ~1,6 M itérations, cf.
  lethal_weapon) : un FAIL s'adjuge contre le GMP pur sur quelques pixels.
- Gap : FAIL = sévérité 1 (correction) ; WARN = 4.

### 5. Profil — où va le temps (`profile`, 2026-11)
Rend chaque cas du tier **une fois** en capturant les diagnostics du moteur
(`[FRACTALL] path=… ns/iter=…`, `[PERTURB PERF] orbit= pixels= post=`,
`[WISDOM] tier= variants=`, `[DD-ESCALATION]`, `[RELIABILITY] flagged=`) et la
ligne CLI « Done in Xs (render: Ys, save: Zs) » (ex-« Génération terminée en … », toujours lue). Par cas :
**path exécuté**, ventilation du wall-clock en `orbit / bla / pixels / post /
save / other` (part % ; `bla` = build de la table, chronométré à part depuis
2026-11-06), phase dominante, ns/iter effectif (skips BLA inclus) recalculé
par le harness et **normalisé par thread** (`ns/iter·thread`, la forme
comparable d'une machine à l'autre), % de
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
- **Pic RSS** par cas (colonne `RSS (MB)`, wrapper `getrusage` — sans
  dépendre de `/usr/bin/time`) : la mémoire est une métrique suivie
  (prérequis de la décision « compression de réf par défaut », G14.6).
- Les paths f64 standard et GMP par-pixel émettent aussi la ligne
  `[FRACTALL] path=std_f64|std_gmp` : tout cas du corpus a un path nommé.
- Logs bruts : `bench/harness/profile/<cas>.log`.

### 5 bis. Profil échantillonné — quelles fonctions (`hotspots`, 2026-11-06)
Le profil par phase dit où va le temps ENTRE les phases ; il ne voit rien de
ce qui se passe DANS une phase. `harness.py hotspots --case <stem>` rend le
cas avec un **binaire symbolisé** (bâti à part dans `bench/prof-target` : même
optimisation, non strippé — la release ne montre que des adresses),
échantillonne les piles (`sample` sous macOS, `perf` sous Linux) et classe les
fonctions par temps au sommet de pile, attente des workers exclue.
- **Quand** : dès qu'une phase domine sans raison évidente, et avant toute
  optimisation d'une boucle — c'est la question « la boucle est-elle vraiment
  le coût ? ». Trois réponses « non » le 2026-11-06 : tier exp → 70 % dans un
  parse GMP par pixel (`rug::float::big::parse` au milieu de la phase
  « pixels ») ; vues peu chères → clone d'`Arc` par pixel ; path f64 standard →
  formule recompilée par pixel. Aucun n'était visible au profil par phase.
- **Lire** : une fonction du moteur à > 90 % = la boucle est bien le coût
  (levier algorithmique) ; de l'allocateur (`malloc`/`free`), du GMP, des
  verrous (`__psynch_mutexwait`) ou `memcmp` dans une phase pixel = coût fixe
  par pixel à sortir de la boucle.
- `--size` doit faire durer le rendu quelques secondes (`sample` met ~1 s à
  s'attacher ; l'outil prévient sous 200 échantillons) ; `--delay` saute
  l'orbite. Sorties : `bench/harness/hotspots/<cas>.md` + rapport brut.
- Le chemin Linux (`perf record` + `perf report`) n'a pas été exercé sur la
  machine où l'outil a été écrit (macOS) : le premier usage sous Linux est à
  vérifier.

### 6. Bench — débits par technique (`bench`, 2026-11)
`fractall-cli --wisdom-bench` lancé **isolé** (`FRACTALL_WISDOM_FILE` dans
`bench/harness/bench/` — le wisdom machine de l'utilisateur,
`~/.config/fractall/wisdom.toml`, n'est pas touché) → iters/s effectifs et
ns/iter par technique : `cpu_std_f64`, `cpu_perturb_f64`, `cpu_perturb_exp`,
`cpu_perturb_dd` (+ `gpu_*` si un GPU est là). Comparables d'un commit à
l'autre **sur la même machine** : débit < 85 % de la baseline = gap sév. 3.
C'est le micro-benchmark du moteur, sans le bruit du démarrage process ni de
l'orbite. (~30 s ; `--axes` sans `bench` pour l'omettre.)

### A/B isolé (`ab`, 2026-11) — l'outil de la méthode G14
`harness.py ab --cases e50,dragon --env FRACTALL_RELIABILITY=off,observe,escalate
--runs 3` (ou `--flags '--dd-tier|'`) : pour chaque cas × variante, N runs
(médiane + spread), profil par phase (orbite/pixels/post), path routé, et
**identité de la sortie** (octets PNG — encodeur déterministe ⇒ mêmes octets ⇔
mêmes pixels). `vs A` = ratio au premier bras. Une variante plus rapide à
sortie **différente** n'est pas un gain : c'est un tradeoff correction à
adjuger vs GMP (`fractall-quality compare`) avant toute décision. Rapport
`bench/harness/ab/ab.md` + `ab.json`. C'est l'étape 2 de la méthode G14 (TODO),
avant la confirmation `score --tier standard --cases …`.

### 7. Wisdom — les décisions automatiques (`wisdom`, 2026-11-19)
Tous les autres axes jugent un RENDU ; celui-ci juge le CHOIX qui l'a
produit. Un seuil de routage mal calibré — ou devenu faux parce qu'un chemin
a été accéléré depuis — ne casse rien : il rend juste, plus lentement qu'il
ne pourrait. Deux volets (`scripts/harnesslib/axes_wisdom.py`), 256×160,
CPU :

- **Regret de routage** (`decision:<dimension>:<cas>`). Le plan `auto` est
  chronométré contre chaque plan FORCÉ de la dimension, en passes
  ENTRELACÉES (3, médiane). Dimensions : `harmonic` (LLA vs BLA, cas
  `toml/`), `algorithm` (f64 standard vs perturbation, de part et d'autre de
  la bascule Auto — y compris au plafond que choisit la sonde d'itérations).
  Le regret ne compte qu'à **sortie équivalente** (itérations brutes des
  `.fmap` égales à 0,05 % près) : WARN > ×1,10, FAIL > ×1,25, jamais sous
  60 ms d'écart. Une alternative plus rapide à sortie ≠ est un compromis de
  correction → `ADJUDICATE` ; sur une vue peu profonde l'axe **tranche
  lui-même contre le GMP pur** : production fausse ⇒ gap de CORRECTION
  (sév. 1) ; alternative juste ⇒ vrai regret (sév. 3) ; alternative fausse
  ⇒ PASS (le wisdom a raison de payer). Les cas `toml/` profonds gardent
  l'adjudication persistée (`harness/wisdom-adjudications.json`).
- **Itérations automatiques** (`iter:<scène>`). La scène est rendue SANS
  `--iterations` : c'est la sonde `wisdom_iter` qu'on juge, contre le même
  rendu à 4N. Sous-itération = pixels plafonnés à N qui s'évadent avant 4N
  (PASS ≤ 0,5 % — le seuil de croissance du moteur —, FAIL > 2 % : sév. 1).
  Sur-provision = plus petit N/4^k qui satisfait encore le critère du moteur,
  et ce qu'il aurait coûté (`WASTE` à partir de ×16 et ×1,25 en temps : 4×
  de marge est voulu). Rapporte aussi la durée de la sonde et la part de la
  correction de fiabilité au plafond choisi → alimentent les leviers.

Ajouter une dimension = une entrée de `DIMENSIONS` (variantes par variable
d'environnement ou option CLI). Cycle ciblé :
`harness.py wisdom-optimality [--only harmonic|algorithm|decision|iter]
[--quick]` (rapport `bench/harness/wisdom/`).

⚠️ Premier passage (2026-11-19) : 0 regret de routage, mais l'adjudication a
trouvé **1,04 % de pixels faux en production** sur éléphants 1e7 au plafond
auto (60 000) — la scène est depuis jugée par `truth`
(`axes_stdtruth.AUTO_CAP_SCENES`). Leçon : un axe de vérité à plafond fixe
(2500) ne juge pas le régime « Iter: Auto ».

### Leviers — où gagner quand aucun gap ne le dit (`levers`, 2026-11-19)
Un gap est un DÉFAUT ; « aucun gap » veut dire que rien n'a régressé, pas que
le moteur est optimal. La section **Leviers** du scorecard (et
`harness.py levers`) classe le temps MESURÉ par ce qui le consomme, tous axes
confondus : secondes par (cas, phase) du profil, correction de fiabilité et
sonde d'itérations des scènes en itérations auto, regret de routage, rendus
de production du volet standard de `truth`. Chaque ligne porte la commande
qui l'ouvre (`ab`, `hotspots`, item TODO). C'est un classement de pistes, pas
un budget (les scènes n'ont pas la même taille) ; planchers : 50 ms et 10 %
du cas. C'est la liste de travail de la boucle quand `gaps` est vide.

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
        [--axes speed,parity,quality,fuzz,precision,truth,wisdom,goldens,profile,bench] [--cases a,b]
python3 scripts/harness.py ab --cases e50 --env FRACTALL_RELIABILITY=off,escalate --runs 3
                                             # A/B isolé (méthode G14) : médiane, bruit, phases, sortie ≡
python3 scripts/harness.py hotspots --case e1000 --size 1200
                                             # profil échantillonné : quelles FONCTIONS (binaire symbolisé)
python3 scripts/harness.py render [history]  # re-génère SCORECARD.md sans re-mesurer
python3 scripts/harness.py trend [--case e50] [--bench] [--all-machines]
python3 scripts/harness.py baseline          # fige le dernier score comme référence
python3 scripts/harness.py gaps              # ré-affiche les gaps du dernier score
python3 scripts/harness.py adjudicate <stem> # arbitre 3-voies d'un cas parité
python3 scripts/harness.py levers            # où gagner : temps mesuré, trié (même sans gap)
python3 scripts/harness.py wisdom-optimality [--only harmonic|algorithm|iter]
                                             # axe wisdom seul : regret des décisions auto
python3 scripts/harness.py preflight         # vet le corpus SOUS cap mémoire
python3 scripts/harness.py quarantine list   # cas exclus des sweeps (crash/OOM)
python3 scripts/harness.py slow-safe list    # cas corrects hors budget temps
python3 scripts/harness.py journal           # trace des incidents (OOM, crash, abort)
python3 scripts/test_harness_guard.py        # verrous du harness lui-même (CI)
```

- **Structure** : `harness.py` est une enveloppe ; la logique vit dans
  `scripts/harnesslib/` (`config` chemins/registres/tiers · `guard` garde-fou ·
  `axes_f3` speed+parity · `axes_truth` quality+fuzz+goldens ·
  `axes_precision` juge orbite ×4 · `axes_perf`
  profile+bench · `gaps` · `scorecard` · `commands*` · `wisdom_opt` · `cli`),
  chaque fichier < 800 lignes (pilier 5 — le harness respecte l'invariant
  qu'il impose). `compare_f3` (numpy/OpenEXR) est importé **paresseusement** :
  son absence éteint parity/adjudicate, pas le harness.
- **Tiers** : `quick` ≈ 10 cas représentatifs à 256², 1 run (~5-10 min, l'outil
  du cycle interne) ; `standard` ≈ 25 cas, 3 runs (validation avant commit
  perf) ; `full` = 84 cas (jalon, plusieurs heures — `preflight` d'abord, et
  `--parity-iterations N` pour plafonner les itérations des deux moteurs sur
  l'axe parité : à itérations natives trois cas coûtent 600 s chacun).
- **Sorties** :
  - `harness/history/<UTC>-<sha>.json` — scorecard machine-readable complet
    (versionné : c'est la mémoire longue de la boucle ; schéma 2 depuis
    2026-11 : sections `profile`, `bench`, `speed_self` ; `trend` lit les deux).
  - `harness/baselines/<machine>.json` — référence **par machine** (slug
    cpu+threads), écrite par `baseline` depuis le dernier score de CETTE
    machine ; `harness/baseline.json` reste le repli historique (machine de
    dev, `baseline --legacy`). Le scorecard nomme la baseline utilisée. Jamais
    mise à jour implicitement.
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
reproduit le problème. Pour la vitesse, le levier se lit dans la section
**Leviers** du scorecard (`harness.py levers`) : une phase à 60 % du wall
sur 8 cas/10 vaut plus qu'un cas isolé.

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
4. `python3 scripts/harness.py baseline` → `harness/baselines/<machine>.json`
   (les régressions intrinsèques se mesurent dès le score suivant, sans
   toucher la baseline de la machine de dev).
5. Si F3 est voulu : `bash scripts/build_f3_linux.sh` puis
   `pip3 install numpy OpenEXR Pillow`, re-`doctor`.
6. **macOS** : le Python système (3.9) n'a pas `tomllib` → lancer le harness
   avec un Python ≥ 3.11 (ex. venv `bench/venv` : `python3.14 -m venv
   bench/venv && bench/venv/bin/pip install numpy OpenEXR Pillow`). Le cap
   mémoire (RLIMIT_AS) n'existe pas sous macOS : le garde-fou crash repose
   sur le breadcrumb + la quarantaine. Le binaire `fraktaler-3.macos`
   versionné dépend des dylibs Homebrew d'OpenEXR/Imath : `doctor` signale
   s'il n'est pas lançable.

## Où regarder (rappels)

| Besoin | Où |
|--------|-----|
| Protocole détaillé parité | `scripts/compare_f3.py` (docstring) + TODO G1 |
| Presets quality | `src/quality/presets.rs` (ajouter = 1 struct literal) |
| Où va le temps (par cas) | `SCORECARD.md` §Profil, `bench/harness/profile/<cas>.log` |
| Timing interne par phase | `FRACTALL_PERTURB_STATS` (orbit/BLA/pixels/post), `[WISDOM]` via `FRACTALL_WISDOM=1` |
| Coût du détecteur de fiabilité | `FRACTALL_RELIABILITY=off|observe|escalate` (A/B), colonne `fiab. %` |
| Timing CLI parseable | `main.rs` « Done in Xs (render: Ys…) » (ex-« Génération terminée en … ») |
| Débit par tier numérique | `SCORECARD.md` §Bench, `trend --bench` |
| Évolution d'un cas | `harness.py trend --case <stem>` |
| Algorithmes F3 | `fraktaler-3-3.1/fraktaler-3-analysis.md` + `src/` F3 |
| Algorithmes Imagina (post-BLA) | `docs/imagina-algorithms-analysis.md` |
| Roadmap / gaps connus | `TODO.md` (G1–G14 ; G14 = plan des optimisations) |
