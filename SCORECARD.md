<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-07-04T12:57:57+00:00
- **Commit** : `2acf2bd`  ⚠️ arbre modifié (dirty)
- **Machine** : Intel(R) Core(TM) i7-10700F CPU @ 2.90GHz · 16 threads · Linux 6.14.0-37-generic
- **Tier** : quick · 256×256 · quality 96×96 · runs=1 · axes=speed,parity,quality,goldens
- **F3** : /home/gistarcade/src/fractall-rust/fraktaler-3-3.1/fraktaler-3-3.1.linux

## Vitesse (ratio fractall/F3, <1 = fractall gagne)

| Métrique | Valeur | vs baseline |
|---|---:|---|
| geomean ratio | 0.837 |  (↓0.0525 ✅) |
| pire ratio | 3.161 (glitch_test_2) | |
| wins (ratio<1) | 4 | |
| timeouts | 0 | |
| cas comparés | 10/10 | |

## Parité (compare_f3 — Δsmooth-iter vs F3)

| Métrique | Valeur | vs baseline |
|---|---:|---|
| n_ok | 10 |  (=) |
| pixel-équivalents (<0.01%) | 5 |  (=) |
| échecs | 0 |  (=) |
| timeouts | 0 |  (=) |
| F3-dégénéré (win fractall) | 0 |  (=) |

## Qualité (fractall-quality suite — perturbation vs GMP)

| Verdict | Nombre | vs baseline |
|---|---:|---|
| PASS | 11 |  (↑6 ✅) |
| WARN | 0 |  (↓1 ✅) |
| FAIL | 0 |  (↓5 ✅) |

## Goldens (pixel-exact)

- 🟢 VERT

## Au-delà de F3

- **speed_wins** : test5, spiral, e401, e1000

## Gaps (top 10 — sévérité asc, magnitude desc)

| # | Sévérité | Axe | Cas | Métrique | Valeur | Note |
|---:|---|---|---|---|---:|---|
| 1 | 3 vitesse | speed | `glitch_test_2` | ratio | 3.1609 | 3.16× plus lent que F3 |
| 2 | 3 vitesse | speed | `dragon` | ratio | 2.6006 | 2.60× plus lent que F3 |
| 3 | 3 vitesse | speed | `e113` | ratio | 2.153 | 2.15× plus lent que F3 |

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
