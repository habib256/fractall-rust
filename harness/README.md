# harness/ — mémoire de la boucle d'auto-amélioration

Protocole : `../HARNESS.md`. Outil : `../scripts/harness.py` (package
`../scripts/harnesslib/`). Dernier score lisible : `../SCORECARD.md`.

| Fichier | Versionné | Rôle | Écrit par |
|---|---|---|---|
| `history/<UTC>-<sha>.json` | oui | scorecard complet (mémoire longue ; schéma 1 = axes speed/parity/quality/fuzz/goldens, schéma 2 (2026-11) + `profile`, `bench`, `speed_self`) | `harness.py score` |
| `baselines/<machine>.json` | oui | référence **par machine** (slug cpu+threads) — lue en priorité par `score`/`render`/`doctor` | `harness.py baseline` |
| `baseline.json` | oui | repli historique (machine de dev i7) quand la machine n'a pas sa baseline | `harness.py baseline --legacy` |
| `adjudications.json` | oui | verdicts 3-voies fractall/F3/GMP par cas parité (`f3_wrong` déclasse le gap) | `harness.py adjudicate` |
| `wisdom-adjudications.json` | oui | tradeoffs correction tranchés de l'axe wisdom-optimality (évite le re-litige tant que le régime est stable) | manuel, après vérification GMP |
| `f3-degenerate.json` | oui | cas où F3 rend un fast-path uniforme FAUX → exclus du geomean vitesse | axe parité (auto-entretenu) |
| `quarantine.json` | oui | cas exclus des sweeps (crash/OOM prouvé) — gap robustesse sév. 2 | `preflight`, `score`, réconciliation journal, `quarantine add` |
| `resolved.json` | oui | tombstones : cas sortis de quarantaine après fix VÉRIFIÉ (non re-quarantainés sauf incident postérieur) | `quarantine remove` |
| `slow-safe.json` | oui | cas corrects + memory-safe mais hors budget temps (skip SANS gap) — attesté par l'opérateur après mesure | `slow-safe add` |
| `inflight.json` | non | breadcrumb du cas en cours ; un résidu au démarrage = crash du run précédent | `score`/`preflight` |
| `crash-journal.jsonl` | non | trace append-only des incidents (OOM, abort, timeout, interruption) de CETTE machine | garde-fou |

Lecture rapide : `harness.py doctor` (état des registres), `harness.py trend`
(évolution des scores de cette machine), `harness.py render` (re-génère le
SCORECARD depuis un history sans re-mesurer).

Invariant « crash journalisé ⇒ quarantainé » : `quarantine.json` peut dériver
(revert, checkout) mais le journal fait foi — la réconciliation au démarrage
re-quarantaine tout cas hard-crash non couvert (verrou
`scripts/test_harness_guard.py`).
