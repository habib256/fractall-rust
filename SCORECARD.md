<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-07-03T16:00:31+00:00
- **Commit** : `fb8de02`  ⚠️ arbre modifié (dirty)
- **Machine** : Intel(R) Core(TM) i7-10700F CPU @ 2.90GHz · 16 threads · Linux 6.14.0-37-generic
- **Tier** : quick · 256×256 · runs=1 · axes=speed
- **F3** : /home/gistarcade/src/fractall-rust/fraktaler-3-3.1/fraktaler-3-3.1.linux

## Vitesse (ratio fractall/F3, <1 = fractall gagne)

| Métrique | Valeur | vs baseline |
|---|---:|---|
| geomean ratio | 2.625 |  |
| pire ratio | 33.006 (test5) | |
| wins (ratio<1) | 1 | |
| timeouts | 0 | |
| cas comparés | 2/2 | |

## Au-delà de F3

- **speed_wins** : spiral

## Gaps (top 10 — sévérité asc, magnitude desc)

| # | Sévérité | Axe | Cas | Métrique | Valeur | Note |
|---:|---|---|---|---|---:|---|
| 1 | 3 vitesse | speed | `test5` | ratio | 33.0064 | 33.01× plus lent que F3 |

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
