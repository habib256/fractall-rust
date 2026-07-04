<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-07-04T08:41:07+00:00
- **Commit** : `3bf7749`  ⚠️ arbre modifié (dirty)
- **Machine** : Intel(R) Core(TM) i7-10700F CPU @ 2.90GHz · 16 threads · Linux 6.14.0-37-generic
- **Tier** : quick · 256×256 · quality 96×96 · runs=1 · axes=speed,parity,quality,goldens
- **F3** : /home/gistarcade/src/fractall-rust/fraktaler-3-3.1/fraktaler-3-3.1.linux

## Vitesse (ratio fractall/F3, <1 = fractall gagne)

| Métrique | Valeur | vs baseline |
|---|---:|---|
| geomean ratio | 0.759 |  (↓0.131 ✅) |
| pire ratio | 3.449 (glitch_test_2) | |
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
| PASS | 5 |  (=) |
| WARN | 1 |  (=) |
| FAIL | 5 |  (=) |

FAIL : seahorse-valley, misiurewicz-m32, mandelbrot-e30, mandelbrot-e50, mandelbrot-e100

## Goldens (pixel-exact)

- 🟢 VERT

## Au-delà de F3

- **speed_wins** : test5, spiral, e401, e1000

## Gaps (top 10 — sévérité asc, magnitude desc)

| # | Sévérité | Axe | Cas | Métrique | Valeur | Note |
|---:|---|---|---|---|---:|---|
| 1 | 1 correction | quality | `seahorse-valley` | verdict | FAIL | quality suite FAIL |
| 2 | 1 correction | quality | `misiurewicz-m32` | verdict | FAIL | quality suite FAIL |
| 3 | 1 correction | quality | `mandelbrot-e30` | verdict | FAIL | quality suite FAIL |
| 4 | 1 correction | quality | `mandelbrot-e50` | verdict | FAIL | quality suite FAIL |
| 5 | 1 correction | quality | `mandelbrot-e100` | verdict | FAIL | quality suite FAIL |
| 6 | 3 vitesse | speed | `glitch_test_2` | ratio | 3.4488 | 3.45× plus lent que F3 |
| 7 | 3 vitesse | speed | `dragon` | ratio | 2.9387 | 2.94× plus lent que F3 |
| 8 | 4 qualité | quality | `mandelbrot-e18-minibrot` | verdict | WARN | quality suite WARN |

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
