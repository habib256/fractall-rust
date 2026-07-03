<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-07-03T18:10:08+00:00
- **Commit** : `3d8adc2`  ⚠️ arbre modifié (dirty)
- **Machine** : Intel(R) Core(TM) i7-10700F CPU @ 2.90GHz · 16 threads · Linux 6.14.0-37-generic
- **Tier** : quick · 256×256 · quality 96×96 · runs=1 · axes=speed,parity,quality,goldens
- **F3** : /home/gistarcade/src/fractall-rust/fraktaler-3-3.1/fraktaler-3-3.1.linux

## Vitesse (ratio fractall/F3, <1 = fractall gagne)

| Métrique | Valeur | vs baseline |
|---|---:|---|
| geomean ratio | 1.662 |  (↓0.05 ✅) |
| pire ratio | 16.138 (test5) | |
| wins (ratio<1) | 3 | |
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
| PASS | 5 |  (↑3 ✅) |
| WARN | 1 |  (↑1 ⚠️) |
| FAIL | 5 |  (↓4 ✅) |

FAIL : seahorse-valley, misiurewicz-m32, mandelbrot-e30, mandelbrot-e50, mandelbrot-e100

## Goldens (pixel-exact)

- 🟢 VERT

## Au-delà de F3

- **speed_wins** : spiral, e401, e1000

## Gaps (top 10 — sévérité asc, magnitude desc)

| # | Sévérité | Axe | Cas | Métrique | Valeur | Note |
|---:|---|---|---|---|---:|---|
| 1 | 1 correction | quality | `seahorse-valley` | verdict | FAIL | quality suite FAIL |
| 2 | 1 correction | quality | `misiurewicz-m32` | verdict | FAIL | quality suite FAIL |
| 3 | 1 correction | quality | `mandelbrot-e30` | verdict | FAIL | quality suite FAIL |
| 4 | 1 correction | quality | `mandelbrot-e50` | verdict | FAIL | quality suite FAIL |
| 5 | 1 correction | quality | `mandelbrot-e100` | verdict | FAIL | quality suite FAIL |
| 6 | 3 vitesse | speed | `test5` | ratio | 16.1379 | 16.14× plus lent que F3 |
| 7 | 3 vitesse | speed | `glitch_test_2` | ratio | 6.8109 | 6.81× plus lent que F3 |
| 8 | 3 vitesse | speed | `dragon` | ratio | 3.6743 | 3.67× plus lent que F3 |
| 9 | 3 vitesse | speed | `floral_fantasy` | ratio | 3.457 | 3.46× plus lent que F3 |
| 10 | 3 vitesse | speed | `flake` | ratio | 2.6534 | 2.65× plus lent que F3 |

_… 3 gap(s) supplémentaire(s) dans le history JSON._

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
