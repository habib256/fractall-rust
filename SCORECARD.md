<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-07-03T16:28:41+00:00
- **Commit** : `6b8717d`
- **Machine** : Intel(R) Core(TM) i7-10700F CPU @ 2.90GHz · 16 threads · Linux 6.14.0-37-generic
- **Tier** : quick · 256×256 · runs=1 · axes=speed,parity,quality,goldens
- **F3** : /home/gistarcade/src/fractall-rust/fraktaler-3-3.1/fraktaler-3-3.1.linux

## Vitesse (ratio fractall/F3, <1 = fractall gagne)

| Métrique | Valeur | vs baseline |
|---|---:|---|
| geomean ratio | 1.187 |  (↓0.525 ✅) |
| pire ratio | 8.049 (test5) | |
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
| PASS | 2 |  (=) |
| WARN | 0 |  (=) |
| FAIL | 9 |  (=) |

FAIL : seahorse-valley, mandelbrot-e13, mandelbrot-e17, misiurewicz-m32, julia-siegel-disk, mandelbrot-e18-minibrot, mandelbrot-e30, mandelbrot-e50, mandelbrot-e100

## Goldens (pixel-exact)

- 🟢 VERT

## Au-delà de F3

- **speed_wins** : spiral, e401, e1000

## Gaps (top 10 — sévérité asc, magnitude desc)

| # | Sévérité | Axe | Cas | Métrique | Valeur | Note |
|---:|---|---|---|---|---:|---|
| 1 | 1 correction | quality | `seahorse-valley` | verdict | FAIL | quality suite FAIL |
| 2 | 1 correction | quality | `mandelbrot-e13` | verdict | FAIL | quality suite FAIL |
| 3 | 1 correction | quality | `mandelbrot-e17` | verdict | FAIL | quality suite FAIL |
| 4 | 1 correction | quality | `misiurewicz-m32` | verdict | FAIL | quality suite FAIL |
| 5 | 1 correction | quality | `julia-siegel-disk` | verdict | FAIL | quality suite FAIL |
| 6 | 1 correction | quality | `mandelbrot-e18-minibrot` | verdict | FAIL | quality suite FAIL |
| 7 | 1 correction | quality | `mandelbrot-e30` | verdict | FAIL | quality suite FAIL |
| 8 | 1 correction | quality | `mandelbrot-e50` | verdict | FAIL | quality suite FAIL |
| 9 | 1 correction | quality | `mandelbrot-e100` | verdict | FAIL | quality suite FAIL |
| 10 | 3 vitesse | speed | `test5` | ratio | 8.0494 | 8.05× plus lent que F3 |

_… 4 gap(s) supplémentaire(s) dans le history JSON._

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
