<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-11-03T04:41:42+00:00
- **Commit** : `922ddf0`  ⚠️ arbre modifié (dirty)
- **Machine** : Intel(R) Xeon(R) Processor · 4 threads · Linux 6.18.44-fc-v230
- **Tier** : quick · 256×256 · quality 96×96 · runs=1 · axes=speed,parity,quality,fuzz,goldens,profile,bench
- **F3** : — (indisponible)
- _baseline présente mais tier différent (standard) — pas de delta._

## Vitesse (ratio fractall/F3, <1 = fractall gagne)

_F3 indisponible — timings fractall seuls (voir history JSON)._

| Métrique | Valeur | vs baseline |
|---|---:|---|
| geomean ratio | — |  |
| pire ratio | — | |
| wins (ratio<1) | 0 | |
| timeouts | 0 | |
| cas comparés | 0/10 | |

### Temps fractall par cas (vitesse intrinsèque)

- _pas d'auto-ratio : baseline d'une autre machine._
- temps cumulé : **8.915 s** sur 10 cas mesurés

| Cas | fractall (s) | min (s) | bruit | vs baseline | ratio F3 |
|---|---:|---:|---:|---:|---:|
| `test5` | 0.026 | 0.026 | — | — | — |
| `spiral` | 0.033 | 0.033 | — | — | — |
| `flake` | 0.035 | 0.035 | — | — | — |
| `glitch_test_2` | 0.065 | 0.065 | — | — | — |
| `e50` | 1.927 | 1.927 | — | — | — |
| `e113` | 0.219 | 0.219 | — | — | — |
| `e401` | 0.820 | 0.820 | — | — | — |
| `e1000` | 0.770 | 0.770 | — | — | — |
| `floral_fantasy` | 0.036 | 0.036 | — | — | — |
| `dragon` | 4.984 | 4.984 | — | — | — |

## Parité (compare_f3 — Δsmooth-iter vs F3)

_F3 indisponible — axe non exécuté._

## Qualité (fractall-quality suite — perturbation vs GMP)

| Verdict | Nombre | vs baseline |
|---|---:|---|
| PASS | 16 |  |
| WARN | 0 |  |
| FAIL | 0 |  |

## Fuzz (sondes aléatoires pert vs GMP)

- seed `20260714` · 3 sondes → **3 PASS · 0 WARN · 0 FAIL**

## Goldens (pixel-exact)

- 🟢 VERT

## Profil — où va le temps (sans F3, 1 run par cas)

- **répartition cumulée** : orbit 15.6 % · pixels 50.6 % · post 32.1 % · save 0.0 % · other 1.7 % (sur 10 cas)
- **phase dominante** : pixels ×9, post ×1
- paths : bytecode_exp, bytecode_f64, bytecode_f64_harmonic_lla

| Cas | path | variantes | wall (s) | orbite % | pixels % | post % | save % | autre % | ns/iter | iter/px | esc.dd | fiab. % |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `test5` | bytecode_f64 | harmonic_lla | 0.019 | 5.2 | 61.8 | 0.0 | 0.0 | 33.0 | 31.700 | 6 |  | 0.00 |
| `spiral` | ? | — | 0.025 | 0.0 | 78.7 | 0.0 | 0.0 | 21.3 | — | — |  | — |
| `flake` | bytecode_f64_harmonic_lla | harmonic_lla | 0.028 | 18.1 | 57.9 | 0.0 | 0.0 | 23.9 | 0.000 | 30921 |  | 0.00 |
| `glitch_test_2` | bytecode_f64_harmonic_lla | harmonic_lla | 0.035 | 2.9 | 80.7 | 0.0 | 0.0 | 16.4 | 0.100 | 7779 |  | 0.00 |
| `e50` | bytecode_f64 | harmonic_lla | 1.945 | 2.0 | 74.0 | 23.3 | 0.0 | 0.7 | 0.100 | 172455 |  | 0.38 |
| `e113` | bytecode_f64 | harmonic_lla | 0.188 | 3.2 | 92.9 | 0.0 | 0.0 | 3.9 | 0.100 | 22729 |  | 0.00 |
| `e401` | bytecode_exp | bla | 0.743 | 4.6 | 93.8 | 0.0 | 0.0 | 1.7 | 0.400 | 27584 |  | 0.00 |
| `e1000` | bytecode_exp | bla | 0.747 | 2.3 | 96.6 | 0.0 | 0.0 | 1.1 | 0.700 | 16634 |  | 0.00 |
| `floral_fantasy` | bytecode_f64_harmonic_lla | harmonic_lla | 0.025 | 4.0 | 68.7 | 0.0 | 0.0 | 27.3 | 0.100 | 2156 |  | 0.00 |
| `dragon` | bytecode_f64 | harmonic_lla | 4.963 | 25.3 | 25.9 | 47.3 | 0.0 | 1.5 | 0.000 | 3921441 |  | 0.12 |

## Bench — débit par technique (`--wisdom-bench` isolé)

| Technique | iters/s | ns/iter | taille | vs baseline |
|---|---:|---:|---:|---:|
| cpu_std_f64 | 1.074e+09 | 0.931 | 1024² | — |
| cpu_perturb_f64 | 5.765e+09 | 0.173 | 256² | — |
| cpu_perturb_exp | 1.436e+10 | 0.070 | 512² | — |
| cpu_perturb_dd | 1.311e+08 | 7.626 | 128² | — |

## Gaps (top 10 — sévérité asc, magnitude desc)

_aucun gap détecté 🎉_

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
