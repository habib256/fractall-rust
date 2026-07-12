<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-07-12T12:00:28+00:00
- **Commit** : `cd05f6a`
- **Machine** : Intel(R) Core(TM) i7-10700F CPU @ 2.90GHz · 16 threads · Linux 6.14.0-37-generic
- **Tier** : standard · 256×256 · runs=3 · axes=speed,parity,quality,goldens
- **F3** : /home/gistarcade/src/fractall-rust/fraktaler-3-3.1/fraktaler-3-3.1.linux
- _baseline présente mais tier différent (quick) — pas de delta._

## Vitesse (ratio fractall/F3, <1 = fractall gagne)

| Métrique | Valeur | vs baseline |
|---|---:|---|
| geomean ratio | 0.223 |  |
| pire ratio | 0.580 (test5) | |
| wins (ratio<1) | 25 | |
| timeouts | 0 | |
| cas comparés | 25/25 | |

## Parité (compare_f3 — Δsmooth-iter vs F3)

| Métrique | Valeur | vs baseline |
|---|---:|---|
| n_ok | 25 |  |
| pixel-équivalents (<0.01%) | 10 |  |
| échecs | 0 |  |
| timeouts | 0 |  |
| F3-dégénéré (win fractall) | 0 |  |

## Qualité (fractall-quality suite — perturbation vs GMP)

| Verdict | Nombre | vs baseline |
|---|---:|---|
| PASS | 8 |  |
| WARN | 3 |  |
| FAIL | 0 |  |

## Goldens (pixel-exact)

- 🟢 VERT

## Au-delà de F3

- **speed_wins** : test5, spiral, flake, glitch_test_2, e50, e113, e401, e1000, floral_fantasy, dragon, e318, e1121, e1200, glitch_test_3, glitch_test_4, heaven, integral_of_ex2, windmill, mitosis, golden_spider, leaded_glass, magic, tick_tock, virus, x

## Gaps (top 10 — sévérité asc, magnitude desc)

| # | Sévérité | Axe | Cas | Métrique | Valeur | Note |
|---:|---|---|---|---|---:|---|
| 1 | 4 qualité | quality | `seahorse-valley` | verdict | WARN | quality suite WARN |
| 2 | 4 qualité | quality | `mandelbrot-e13` | verdict | WARN | quality suite WARN |
| 3 | 4 qualité | quality | `mandelbrot-e17` | verdict | WARN | quality suite WARN |

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
