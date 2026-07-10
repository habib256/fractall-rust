<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-07-10T20:17:02+00:00
- **Commit** : `f5b7c90`  ⚠️ arbre modifié (dirty)
- **Machine** : Intel(R) Core(TM) i7-10700F CPU @ 2.90GHz · 16 threads · Linux 6.14.0-37-generic
- **Tier** : quick · 256×256 · quality 96×96 · runs=1 · axes=speed,parity,quality,goldens
- **F3** : /home/gistarcade/src/fractall-rust/fraktaler-3-3.1/fraktaler-3-3.1.linux

## Vitesse (ratio fractall/F3, <1 = fractall gagne)

| Métrique | Valeur | vs baseline |
|---|---:|---|
| geomean ratio | 0.285 |  (↓0.0441 ✅) |
| pire ratio | 1.302 (glitch_test_2) | |
| wins (ratio<1) | 9 | |
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
| PASS | 11 |  (=) |
| WARN | 0 |  (=) |
| FAIL | 0 |  (=) |

## Goldens (pixel-exact)

- 🟢 VERT

## Au-delà de F3

- **speed_wins** : test5, spiral, flake, e50, e113, e401, e1000, floral_fantasy, dragon

## Gaps (top 10 — sévérité asc, magnitude desc)

_aucun gap détecté 🎉_

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
