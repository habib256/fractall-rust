<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-11-06T06:14:02+00:00
- **Commit** : `f7db05f`  ⚠️ arbre modifié (dirty)
- **Machine** : Apple M4 · 10 threads · Darwin 27.2.0
- **Tier** : standard · 256×256 · runs=3 · axes=speed,parity,quality,fuzz,precision,truth,goldens,profile,bench
- **F3** : — (indisponible)
- **Baseline** : `harness/baselines/apple-m4-10t.json` (e3ff8be · tier quick · 2026-11-06)
- _baseline présente mais tier différent (quick) — pas de delta._

## Vitesse (ratio fractall/F3, <1 = fractall gagne)

_F3 indisponible — timings fractall seuls (voir history JSON)._

| Métrique | Valeur | vs baseline |
|---|---:|---|
| geomean ratio | — |  |
| pire ratio | — | |
| wins (ratio<1) | 0 | |
| timeouts | 0 | |
| cas comparés | 0/25 | |

### Temps fractall par cas (vitesse intrinsèque)

- _pas d'auto-ratio : baseline d'une autre machine._
- temps cumulé : **4.594 s** sur 25 cas mesurés

| Cas | fractall (s) | min (s) | bruit | vs baseline | ratio F3 |
|---|---:|---:|---:|---:|---:|
| `test5` | 0.012 | 0.010 | 24 % | — | — |
| `spiral` | 0.012 | 0.011 | 11 % | — | — |
| `flake` | 0.046 | 0.045 | 10 % | — | — |
| `glitch_test_2` | 0.047 | 0.026 | 47 % | — | — |
| `e50` | 0.620 | 0.605 | 7 % | — | — |
| `e113` | 0.084 | 0.080 | 16 % | — | — |
| `e401` | 0.088 | 0.088 | 1 % | — | — |
| `e1000` | 0.055 | 0.049 | 60 % | — | — |
| `floral_fantasy` | 0.025 | 0.024 | 8 % | — | — |
| `dragon` | 1.294 | 1.259 | 3 % | — | — |
| `e318` | 0.195 | 0.194 | 6 % | — | — |
| `e1121` | 0.083 | 0.080 | 14 % | — | — |
| `e1200` | 0.084 | 0.083 | 8 % | — | — |
| `glitch_test_3` | 0.092 | 0.090 | 8 % | — | — |
| `glitch_test_4` | 0.025 | 0.025 | 1 % | — | — |
| `heaven` | 0.084 | 0.080 | 10 % | — | — |
| `integral_of_ex2` | 0.149 | 0.139 | 8 % | — | — |
| `windmill` | 0.146 | 0.095 | 35 % | — | — |
| `mitosis` | 0.041 | 0.040 | 17 % | — | — |
| `golden_spider` | 0.603 | 0.601 | 2 % | — | — |
| `leaded_glass` | 0.206 | 0.196 | 8 % | — | — |
| `magic` | 0.320 | 0.308 | 4 % | — | — |
| `tick_tock` | 0.144 | 0.136 | 9 % | — | — |
| `virus` | 0.091 | 0.091 | 8 % | — | — |
| `x` | 0.049 | 0.041 | 21 % | — | — |

## Parité (compare_f3 — Δsmooth-iter vs F3)

_F3 indisponible — axe non exécuté._

## Qualité (fractall-quality suite — perturbation vs GMP)

| Verdict | Nombre | vs baseline |
|---|---:|---|
| PASS | 16 |  |
| WARN | 1 |  |
| FAIL | 0 |  |

## Fuzz (sondes aléatoires pert vs GMP)

- seed `20260714` · 8 sondes → **8 PASS · 0 WARN · 0 FAIL**

## Précision d'orbite (juge ×4)

- 26 cas à 128² → **26 PASS · 0 WARN · 0 FAIL** (itérations brutes vs orbite à précision ×4)

| Cas | prec | juge | pixels ≠ | % | verdict |
|---|---:|---:|---:|---:|---|
| `test5` | 256 | 1024 | 0 | 0.000 | PASS |
| `flake` | 624 | 2496 | 0 | 0.000 | PASS |
| `glitch_test_2` | 476 | 1904 | 0 | 0.000 | PASS |
| `e50` | 268 | 1072 | 0 | 0.000 | PASS |
| `e113` | 479 | 1916 | 0 | 0.000 | PASS |
| `e401` | 1434 | 5736 | 0 | 0.000 | PASS |
| `e1000` | 3423 | 13692 | 0 | 0.000 | PASS |
| `floral_fantasy` | 384 | 1536 | 0 | 0.000 | PASS |
| `dragon` | 738 | 2952 | 0 | 0.000 | PASS |
| `e318` | 1160 | 4640 | 0 | 0.000 | PASS |
| `e1121` | 3827 | 15308 | 0 | 0.000 | PASS |
| `e1200` | 4091 | 16364 | 0 | 0.000 | PASS |
| `glitch_test_3` | 628 | 2512 | 0 | 0.000 | PASS |
| `glitch_test_4` | 385 | 1540 | 0 | 0.000 | PASS |
| `heaven` | 594 | 2376 | 0 | 0.000 | PASS |
| `integral_of_ex2` | 774 | 3096 | 0 | 0.000 | PASS |
| `windmill` | 633 | 2532 | 0 | 0.000 | PASS |
| `mitosis` | 1001 | 4004 | 0 | 0.000 | PASS |
| `golden_spider` ⚑ | 605 | 2420 | 2 | 0.012 | PASS |
| `leaded_glass` | 413 | 1652 | 0 | 0.000 | PASS |
| `magic` | 521 | 2084 | 0 | 0.000 | PASS |
| `tick_tock` | 572 | 2288 | 0 | 0.000 | PASS |
| `virus` | 846 | 3384 | 0 | 0.000 | PASS |
| `x` | 884 | 3536 | 0 | 0.000 | PASS |
| `e227` ⚑ | 858 | 3432 | 0 | 0.000 | PASS |
| `evolution_trees` ⚑ | 858 | 3432 | 2 | 0.012 | PASS |

_⚑ = sentinelle (précision F3 nue : 2-4 % faux avant la marge +64 b)._

## Vérité (rendu de production vs tier dd)

- 27 cas à 128² → **27 PASS · 0 WARN · 0 FAIL** (itérations brutes du rendu par défaut vs frame `--dd-tier`)

| Cas | path | pixels ≠ | % | verdict |
|---|---|---:|---:|---|
| `test5` | bytecode_f64 | 0 | 0.000 | PASS |
| `flake` | bytecode_f64_harmonic_lla | 0 | 0.000 | PASS |
| `glitch_test_2` | bytecode_f64_harmonic_lla | 0 | 0.000 | PASS |
| `e50` | bytecode_f64 | 1 | 0.006 | PASS |
| `e113` | bytecode_f64 | 0 | 0.000 | PASS |
| `e401` | bytecode_exp | 0 | 0.000 | PASS |
| `e1000` | bytecode_exp | 0 | 0.000 | PASS |
| `floral_fantasy` | bytecode_f64_harmonic_lla | 0 | 0.000 | PASS |
| `dragon` | bytecode_f64 | 3 | 0.018 | PASS |
| `e318` | bytecode_exp | 0 | 0.000 | PASS |
| `e1121` | bytecode_exp | 0 | 0.000 | PASS |
| `e1200` | bytecode_exp | 0 | 0.000 | PASS |
| `glitch_test_3` | bytecode_f64_harmonic_lla | 0 | 0.000 | PASS |
| `glitch_test_4` | bytecode_f64_harmonic_lla | 0 | 0.000 | PASS |
| `heaven` | bytecode_f64 | 0 | 0.000 | PASS |
| `integral_of_ex2` | bytecode_f64 | 0 | 0.000 | PASS |
| `windmill` | bytecode_f64 | 0 | 0.000 | PASS |
| `mitosis` | bytecode_f64_harmonic_lla | 0 | 0.000 | PASS |
| `golden_spider` ⚑ | bytecode_f64 | 0 | 0.000 | PASS |
| `leaded_glass` ⚑ | bytecode_f64_harmonic_lla | 0 | 0.000 | PASS |
| `magic` | bytecode_f64 | 0 | 0.000 | PASS |
| `tick_tock` | bytecode_f64 | 0 | 0.000 | PASS |
| `virus` | bytecode_f64 | 0 | 0.000 | PASS |
| `x` | bytecode_f64 | 0 | 0.000 | PASS |
| `e227` ⚑ | bytecode_f64 | 0 | 0.000 | PASS |
| `rug` ⚑ | bytecode_f64 | 0 | 0.000 | PASS |
| `olbaid2` ⚑ | bytecode_exp | 0 | 0.000 | PASS |

_⚑ = sentinelle (rendu par défaut faux avant les correctifs du 2026-11-06). Un FAIL se juge contre le GMP pur : le dd a son propre plancher._

## Goldens (pixel-exact)

- 🟢 VERT

## Profil — où va le temps (sans F3, 1 run par cas)

- **répartition cumulée** : orbit 14.5 % · bla 0.7 % · pixels 51.4 % · post 19.7 % · save 0.0 % · other 13.7 % (sur 25 cas, 10 threads)
- **phase dominante** : pixels ×15, other ×6, orbit ×3, post ×1
- paths : bytecode_exp, bytecode_f64, bytecode_f64_harmonic_lla, std_f64 · pic RSS max 329 MB

| Cas | path | variantes | wall (s) | orbite % | BLA % | pixels % | post % | save % | autre % | ns/iter | ns/iter·thread | iter/px | esc.dd | fiab. % | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `test5` | bytecode_f64 | harmonic_lla | 0.025 | 0.0 | 0.0 | 4.0 | 0.0 | 0.0 | 96.0 | 2.543 | 25.431 | 6 |  | 0.00 | 12 |
| `spiral` | std_f64 | — | 0.028 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.100 | — | 251 |  | — | 16 |
| `flake` | bytecode_f64_harmonic_lla | harmonic_lla | 0.053 | 3.8 | 0.0 | 50.8 | 0.0 | 0.0 | 45.4 | 0.013 | 0.133 | 30921 |  | 0.00 | 13 |
| `glitch_test_2` | bytecode_f64_harmonic_lla | harmonic_lla | 0.044 | 0.0 | 0.0 | 45.2 | 2.3 | 0.0 | 52.5 | 0.039 | 0.392 | 7779 |  | 0.00 | 12 |
| `e50` | bytecode_f64 | harmonic_lla | 0.617 | 2.3 | 0.2 | 78.0 | 15.6 | 0.0 | 4.0 | 0.043 | 0.426 | 172455 |  | 0.69 | 30 |
| `e113` | bytecode_f64 | harmonic_lla | 0.084 | 2.4 | 0.0 | 67.6 | 2.4 | 0.0 | 27.7 | 0.038 | 0.383 | 22729 |  | 0.01 | 14 |
| `e401` | bytecode_exp | bla | 0.089 | 10.1 | 2.2 | 57.0 | 3.4 | 0.0 | 27.4 | 0.028 | 0.282 | 27584 |  | 0.04 | 52 |
| `e1000` | bytecode_exp | bla | 0.060 | 11.7 | 1.7 | 43.6 | 0.0 | 0.0 | 43.0 | 0.024 | 0.238 | 16634 |  | 0.00 | 24 |
| `floral_fantasy` | bytecode_f64_harmonic_lla | harmonic_lla | 0.034 | 0.0 | 0.0 | 26.2 | 0.0 | 0.0 | 73.8 | 0.064 | 0.637 | 2156 |  | 0.00 | 12 |
| `dragon` | bytecode_f64 | harmonic_lla | 1.305 | 34.9 | 1.0 | 32.7 | 29.3 | 0.0 | 2.2 | 0.002 | 0.017 | 3921444 |  | 0.72 | 329 |
| `e318` | bytecode_exp | bla | 0.186 | 10.8 | 3.8 | 67.8 | 4.3 | 0.0 | 13.3 | 0.018 | 0.179 | 107483 |  | 0.07 | 35 |
| `e1121` | bytecode_exp | bla | 0.086 | 38.6 | 3.5 | 26.9 | 0.0 | 0.0 | 31.0 | 0.018 | 0.183 | 19218 |  | 0.00 | 68 |
| `e1200` | bytecode_exp | bla | 0.081 | 38.2 | 3.7 | 25.9 | 0.0 | 0.0 | 32.3 | 0.015 | 0.152 | 21119 |  | 0.00 | 69 |
| `glitch_test_3` | bytecode_f64_harmonic_lla | harmonic_lla | 0.078 | 6.4 | 0.0 | 61.2 | 0.0 | 0.0 | 32.4 | 0.023 | 0.231 | 31683 |  | 0.00 | 14 |
| `glitch_test_4` | bytecode_f64_harmonic_lla | harmonic_lla | 0.034 | 0.0 | 0.0 | 26.7 | 0.0 | 0.0 | 73.3 | 0.063 | 0.633 | 2171 |  | 0.00 | 12 |
| `heaven` | bytecode_f64 | harmonic_lla | 0.072 | 2.8 | 0.0 | 63.4 | 0.0 | 0.0 | 33.8 | 0.032 | 0.322 | 21784 |  | 0.00 | 13 |
| `integral_of_ex2` | bytecode_f64 | harmonic_lla | 0.141 | 9.2 | 0.7 | 68.8 | 4.3 | 0.0 | 17.0 | 0.015 | 0.146 | 101291 |  | 0.13 | 20 |
| `windmill` | bytecode_f64 | harmonic_lla | 0.105 | 3.8 | 0.0 | 72.2 | 1.9 | 0.0 | 22.1 | 0.022 | 0.220 | 52681 |  | 0.01 | 15 |
| `mitosis` | bytecode_f64_harmonic_lla | harmonic_lla | 0.051 | 13.8 | 0.0 | 33.4 | 5.9 | 0.0 | 47.0 | 0.006 | 0.059 | 43913 |  | 0.00 | 16 |
| `golden_spider` | bytecode_f64 | harmonic_lla | 0.610 | 2.6 | 0.2 | 33.8 | 59.2 | 0.0 | 4.2 | 0.021 | 0.205 | 152927 |  | 4.84 | 29 |
| `leaded_glass` | bytecode_f64_harmonic_lla | harmonic_lla | 0.206 | 1.9 | 0.0 | 85.9 | 0.0 | 0.0 | 12.2 | 0.037 | 0.367 | 73555 |  | 0.00 | 15 |
| `magic` | bytecode_f64 | harmonic_lla | 0.286 | 5.6 | 0.4 | 78.8 | 6.7 | 0.0 | 8.6 | 0.021 | 0.206 | 166590 |  | 0.32 | 29 |
| `tick_tock` | bytecode_f64 | harmonic_lla | 0.117 | 4.3 | 0.0 | 73.5 | 1.7 | 0.0 | 20.5 | 0.030 | 0.299 | 43825 |  | 0.02 | 15 |
| `virus` | bytecode_f64 | harmonic_lla | 0.091 | 13.2 | 1.1 | 52.8 | 7.7 | 0.0 | 25.2 | 0.009 | 0.093 | 78865 |  | 0.13 | 20 |
| `x` | bytecode_f64 | harmonic_lla | 0.059 | 5.1 | 0.0 | 51.2 | 3.4 | 0.0 | 40.3 | 0.017 | 0.169 | 27132 |  | 0.01 | 13 |

## Bench — débit par technique (`--wisdom-bench` isolé)

| Technique | iters/s | ns/iter | taille | vs baseline |
|---|---:|---:|---:|---:|
| cpu_std_f64 | 1.577e+10 | 0.063 | 1024² | — |
| cpu_perturb_f64 | 1.985e+10 | 0.050 | 512² | — |
| cpu_perturb_exp | 5.633e+10 | 0.018 | 1024² | — |
| cpu_perturb_dd | 8.512e+08 | 1.175 | 128² | — |
| cpu_perturb_f64_mid | 1.997e+09 | 0.501 | 1024² | — |
| gpu_std_f32 | 1.053e+10 | 0.095 | 2048² | — |
| gpu_perturb_f32 | 5.122e+09 | 0.195 | 1024² | — |

## Tendance (même machine, 6 derniers scores)

| Date (UTC) | Commit | Tier | geomean/F3 | wins | fractall Σs | auto-ratio | parité ok/équiv | Q P/W/F | fuzz P/W/F | goldens | px % | gaps |
|---|---|---|---:|---:|---:|---:|---|---|---|---|---:|---:|
| 2026-11-06 04:39 | `0473296` | quick | — | 0 | 2.261 | 0.981 | —/— | 17/0/0 | 3/0/0 | 🟢 | 47.1 | 0 |
| 2026-11-06 05:10 | `b3ed210` | standard | — | 0 | 4.661 | — | —/— | 16/1/0 | 6/0/0 | 🟢 | 51.4 | 1 |
| 2026-11-06 05:37 | `b5cdf1a` | quick | — | 0 | 2.353 | 1.040 | —/— | 17/0/0 | 3/0/0 | 🟢 | 47.5 | 0 |
| 2026-11-06 06:09 | `376c59c` | quick | — | 0 | 2.360 | 0.989 | —/— | 17/0/0 | 4/0/0 | 🟢 | 46.8 | 0 |
| 2026-11-06 06:11 | `e3ff8be` | quick | — | 0 | 2.292 | 0.957 | —/— | 17/0/0 | 4/0/0 | 🟢 | 48.1 | 0 |
| 2026-11-06 06:14 | `f7db05f` | standard | — | 0 | 4.594 | — | —/— | 16/1/0 | 8/0/0 | 🟢 | 51.4 | 1 |

_`python3 scripts/harness.py trend` pour l'historique complet (`--case <stem>` pour un cas)._

## Gaps (top 10 — sévérité asc, magnitude desc)

| # | Sévérité | Axe | Cas | Métrique | Valeur | Note |
|---:|---|---|---|---|---:|---|
| 1 | 4 qualité | quality | `seahorse-valley` | verdict | WARN | quality suite WARN |

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
