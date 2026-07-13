<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-07-13T23:01:34+00:00
- **Commit** : `7fdfd61`  ⚠️ arbre modifié (dirty)
- **Machine** : Intel(R) Core(TM) i7-10700F CPU @ 2.90GHz · 16 threads · Linux 6.14.0-37-generic
- **Tier** : quick · 256×256 · quality 96×96 · runs=1 · axes=speed,parity,quality,fuzz,goldens
- **F3** : /home/gistarcade/src/fractall-rust/fraktaler-3-3.1/fraktaler-3-3.1.linux
- _baseline présente mais tier différent (standard) — pas de delta._

## Vitesse (ratio fractall/F3, <1 = fractall gagne)

| Métrique | Valeur | vs baseline |
|---|---:|---|
| geomean ratio | 0.237 |  |
| pire ratio | 0.611 (test5) | |
| wins (ratio<1) | 10 | |
| timeouts | 0 | |
| cas comparés | 10/10 | |

## Parité (compare_f3 — Δsmooth-iter vs F3)

| Métrique | Valeur | vs baseline |
|---|---:|---|
| n_ok | 10 |  |
| pixel-équivalents (<0.01%) | 5 |  |
| échecs | 0 |  |
| timeouts | 0 |  |
| F3-dégénéré (win fractall) | 0 |  |

## Qualité (fractall-quality suite — perturbation vs GMP)

| Verdict | Nombre | vs baseline |
|---|---:|---|
| PASS | 14 |  |
| WARN | 0 |  |
| FAIL | 0 |  |

## Fuzz (sondes aléatoires pert vs GMP)

- seed `20260714` · 3 sondes → **2 PASS · 1 WARN · 0 FAIL**
  - `fuzz-20260714-1-mandelbrot` **WARN** — c=(-0.6152286330858866, 0.40106525023778294) zoom 6.023813e+07 iters 2048

## Goldens (pixel-exact)

- 🟢 VERT

## Au-delà de F3

- **speed_wins** : test5, spiral, flake, glitch_test_2, e50, e113, e401, e1000, floral_fantasy, dragon

## Gaps (top 10 — sévérité asc, magnitude desc)

| # | Sévérité | Axe | Cas | Métrique | Valeur | Note |
|---:|---|---|---|---|---:|---|
| 1 | 4 qualité | fuzz | `fuzz-20260714-1-mandelbrot` | verdict | WARN | fuzz WARN — divergence éparse (seed 20260714) |

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
