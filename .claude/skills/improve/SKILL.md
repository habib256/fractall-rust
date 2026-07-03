---
name: improve
description: Exécute UNE itération de la boucle d'auto-amélioration du moteur fractall (mesure → gap → diagnostic → patch → vérification → commit). Utiliser quand l'utilisateur demande d'améliorer le moteur, de continuer la boucle, ou invoque /improve. Argument optionnel = axe ou cas à cibler (ex: "speed", "seahorse-valley").
---

# /improve — une itération de la boucle d'auto-amélioration

Protocole complet : `HARNESS.md` (racine). Roadmap des gaps connus : `TODO.md`.
Ne JAMAIS sauter l'étape 5 (vérification). Une itération = un commit (ou un
revert propre + une note TODO si l'hypothèse échoue).

## Étape 1 — Mesurer

```bash
python3 scripts/harness.py score --tier quick
```

- Lire `SCORECARD.md` (généré) : agrégats par axe + **gaps triés**.
- Si `$ARGUMENTS` désigne un axe/cas, filtrer les gaps dessus ; sinon prendre
  le gap #1 (priorités : correction > robustesse > vitesse > qualité —
  cf. HARNESS.md §Priorités).
- Pas de baseline (`harness/baseline.json` absent) ? La créer d'abord :
  `python3 scripts/harness.py baseline` puis committer.
- Binaire F3 absent (axes marqués `f3_unavailable`) ? Le (re)builder d'abord
  (voir `fraktaler-3-3.1/`, cible batch du Makefile) — sans F3, seuls les
  axes quality/goldens sont mesurables.

## Étape 2 — Reproduire petit

Réduire le gap au plus petit cas reproductible : un seul stem
(`--cases <stem>`), taille minimale qui montre le problème (48²–256²).
Vérifier que le gap n'est pas déjà élucidé dans `TODO.md` (G1–G7) — beaucoup
de divergences sont classées (bord chaotique, F3-dégénéré…) : ne pas
re-litiger un cas tranché.

## Étape 3 — Diagnostiquer (avant TOUT patch)

- **Confirmer QUEL path s'exécute** : `FRACTALL_PERTURB_STATS=1` + compteurs.
  Leçon G2 : des heures d'optimisation perdues sur un path qui ne tournait
  pas. Signature GMP par-pixel ≈ 1 µs/iter ; perturbation+BLA est 100-1000×
  plus rapide.
- Perf : isoler la phase (orbite référence / BLA build / pixels / post) via
  les stats internes ; méfiance thermique sur les A/B back-to-back.
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

## Étape 5 — Vérifier (obligatoire, dans cet ordre)

```bash
cargo test --release --bin fractall-cli
cargo test --release --test golden_images
cargo run --release --bin fractall-quality -- suite
python3 scripts/harness.py score --tier quick
```

- Comparer au baseline : le gap ciblé doit s'améliorer, **aucun autre axe ne
  doit régresser** (> 10 % = revert ou justification écrite).
- Affirmation de perf → re-mesurer en `--tier standard` (3 runs, médiane)
  sur les cas touchés.
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
  `harness/baseline.json` (jamais implicite).
- Hypothèse infirmée ? Revert complet + noter l'impasse dans TODO.md (les
  fausses pistes documentées valent de l'or, cf. G3).
