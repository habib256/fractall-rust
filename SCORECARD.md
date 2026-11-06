<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-11-06T05:10:51+00:00
- **Commit** : `b3ed210`  ⚠️ arbre modifié (dirty)
- **Machine** : Apple M4 · 10 threads · Darwin 27.2.0
- **Tier** : standard · 256×256 · runs=3 · axes=speed,parity,quality,fuzz,precision,truth,goldens,profile,bench
- **F3** : — (indisponible)
- **Baseline** : `harness/baselines/apple-m4-10t.json` (4bfbdba · tier quick · 2026-11-06)
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
- temps cumulé : **4.661 s** sur 25 cas mesurés

| Cas | fractall (s) | min (s) | bruit | vs baseline | ratio F3 |
|---|---:|---:|---:|---:|---:|
| `test5` | 0.013 | 0.013 | 3 % | — | — |
| `spiral` | 0.012 | 0.010 | 17 % | — | — |
| `flake` | 0.050 | 0.045 | 13 % | — | — |
| `glitch_test_2` | 0.048 | 0.026 | 47 % | — | — |
| `e50` | 0.652 | 0.616 | 7 % | — | — |
| `e113` | 0.084 | 0.079 | 23 % | — | — |
| `e401` | 0.093 | 0.090 | 6 % | — | — |
| `e1000` | 0.049 | 0.045 | 16 % | — | — |
| `floral_fantasy` | 0.025 | 0.023 | 9 % | — | — |
| `dragon` | 1.309 | 1.292 | 2 % | — | — |
| `e318` | 0.204 | 0.194 | 11 % | — | — |
| `e1121` | 0.092 | 0.091 | 1 % | — | — |
| `e1200` | 0.085 | 0.084 | 11 % | — | — |
| `glitch_test_3` | 0.089 | 0.083 | 11 % | — | — |
| `glitch_test_4` | 0.024 | 0.024 | 5 % | — | — |
| `heaven` | 0.085 | 0.085 | 10 % | — | — |
| `integral_of_ex2` | 0.143 | 0.134 | 8 % | — | — |
| `windmill` | 0.090 | 0.088 | 64 % | — | — |
| `mitosis` | 0.049 | 0.043 | 17 % | — | — |
| `golden_spider` | 0.656 | 0.636 | 5 % | — | — |
| `leaded_glass` | 0.208 | 0.205 | 2 % | — | — |
| `magic` | 0.312 | 0.268 | 19 % | — | — |
| `tick_tock` | 0.150 | 0.134 | 13 % | — | — |
| `virus` | 0.090 | 0.082 | 15 % | — | — |
| `x` | 0.049 | 0.045 | 13 % | — | — |

## Parité (compare_f3 — Δsmooth-iter vs F3)

_F3 indisponible — axe non exécuté._

## Qualité (fractall-quality suite — perturbation vs GMP)

| Verdict | Nombre | vs baseline |
|---|---:|---|
| PASS | 16 |  |
| WARN | 1 |  |
| FAIL | 0 |  |

## Fuzz (sondes aléatoires pert vs GMP)

- seed `20260714` · 6 sondes → **6 PASS · 0 WARN · 0 FAIL**

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

- **répartition cumulée** : orbit 14.6 % · bla 0.8 % · pixels 51.4 % · post 19.5 % · save 0.0 % · other 13.7 % (sur 25 cas, 10 threads)
- **phase dominante** : pixels ×15, other ×6, orbit ×3, post ×1
- paths : bytecode_exp, bytecode_f64, bytecode_f64_harmonic_lla, std_f64 · pic RSS max 329 MB

| Cas | path | variantes | wall (s) | orbite % | BLA % | pixels % | post % | save % | autre % | ns/iter | ns/iter·thread | iter/px | esc.dd | fiab. % | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `test5` | bytecode_f64 | harmonic_lla | 0.025 | 0.0 | 0.0 | 4.0 | 0.0 | 0.0 | 96.0 | 2.543 | 25.431 | 6 |  | 0.00 | 12 |
| `spiral` | std_f64 | — | 0.025 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.100 | — | 251 |  | — | 15 |
| `flake` | bytecode_f64_harmonic_lla | harmonic_lla | 0.052 | 3.8 | 0.0 | 51.8 | 0.0 | 0.0 | 44.4 | 0.013 | 0.133 | 30921 |  | 0.00 | 12 |
| `glitch_test_2` | bytecode_f64_harmonic_lla | harmonic_lla | 0.044 | 0.0 | 0.0 | 45.9 | 2.3 | 0.0 | 51.8 | 0.039 | 0.392 | 7779 |  | 0.00 | 12 |
| `e50` | bytecode_f64 | harmonic_lla | 0.614 | 2.3 | 0.2 | 77.3 | 16.1 | 0.0 | 4.1 | 0.042 | 0.420 | 172455 |  | 0.69 | 30 |
| `e113` | bytecode_f64 | harmonic_lla | 0.097 | 2.1 | 0.0 | 68.1 | 2.1 | 0.0 | 27.8 | 0.044 | 0.443 | 22729 |  | 0.01 | 14 |
| `e401` | bytecode_exp | bla | 0.088 | 10.2 | 2.3 | 57.8 | 3.4 | 0.0 | 26.3 | 0.028 | 0.282 | 27584 |  | 0.04 | 51 |
| `e1000` | bytecode_exp | bla | 0.060 | 11.6 | 1.7 | 44.7 | 0.0 | 0.0 | 42.1 | 0.025 | 0.248 | 16634 |  | 0.00 | 24 |
| `floral_fantasy` | bytecode_f64_harmonic_lla | harmonic_lla | 0.035 | 0.0 | 0.0 | 25.6 | 0.0 | 0.0 | 74.4 | 0.064 | 0.637 | 2156 |  | 0.00 | 12 |
| `dragon` | bytecode_f64 | harmonic_lla | 1.252 | 36.6 | 1.1 | 33.4 | 26.7 | 0.0 | 2.2 | 0.002 | 0.016 | 3921444 |  | 0.72 | 329 |
| `e318` | bytecode_exp | bla | 0.185 | 10.8 | 3.8 | 67.7 | 4.3 | 0.0 | 13.3 | 0.018 | 0.177 | 107483 |  | 0.07 | 34 |
| `e1121` | bytecode_exp | bla | 0.085 | 38.7 | 3.5 | 27.0 | 0.0 | 0.0 | 30.8 | 0.018 | 0.183 | 19218 |  | 0.00 | 68 |
| `e1200` | bytecode_exp | bla | 0.081 | 38.4 | 3.7 | 26.0 | 0.0 | 0.0 | 31.8 | 0.015 | 0.152 | 21119 |  | 0.00 | 69 |
| `glitch_test_3` | bytecode_f64_harmonic_lla | harmonic_lla | 0.083 | 6.0 | 0.0 | 61.4 | 0.0 | 0.0 | 32.5 | 0.025 | 0.246 | 31683 |  | 0.00 | 14 |
| `glitch_test_4` | bytecode_f64_harmonic_lla | harmonic_lla | 0.035 | 0.0 | 0.0 | 25.5 | 0.0 | 0.0 | 74.5 | 0.063 | 0.633 | 2171 |  | 0.00 | 12 |
| `heaven` | bytecode_f64 | harmonic_lla | 0.074 | 2.7 | 0.0 | 63.7 | 0.0 | 0.0 | 33.6 | 0.033 | 0.329 | 21784 |  | 0.00 | 13 |
| `integral_of_ex2` | bytecode_f64 | harmonic_lla | 0.141 | 9.2 | 0.7 | 68.7 | 4.2 | 0.0 | 17.2 | 0.015 | 0.146 | 101291 |  | 0.13 | 20 |
| `windmill` | bytecode_f64 | harmonic_lla | 0.106 | 3.8 | 0.0 | 72.0 | 1.9 | 0.0 | 22.3 | 0.022 | 0.220 | 52681 |  | 0.01 | 15 |
| `mitosis` | bytecode_f64_harmonic_lla | harmonic_lla | 0.051 | 13.8 | 0.0 | 33.4 | 5.9 | 0.0 | 46.9 | 0.006 | 0.059 | 43913 |  | 0.00 | 16 |
| `golden_spider` | bytecode_f64 | harmonic_lla | 0.649 | 2.5 | 0.2 | 32.1 | 61.5 | 0.0 | 3.8 | 0.021 | 0.207 | 152927 |  | 4.84 | 29 |
| `leaded_glass` | bytecode_f64_harmonic_lla | harmonic_lla | 0.206 | 1.9 | 0.0 | 86.0 | 0.0 | 0.0 | 12.1 | 0.037 | 0.367 | 73555 |  | 0.00 | 15 |
| `magic` | bytecode_f64 | harmonic_lla | 0.286 | 5.6 | 0.3 | 78.7 | 6.6 | 0.0 | 8.7 | 0.021 | 0.206 | 166590 |  | 0.32 | 29 |
| `tick_tock` | bytecode_f64 | harmonic_lla | 0.117 | 4.3 | 0.0 | 73.3 | 1.7 | 0.0 | 20.7 | 0.030 | 0.299 | 43825 |  | 0.02 | 15 |
| `virus` | bytecode_f64 | harmonic_lla | 0.089 | 13.5 | 1.1 | 51.6 | 6.7 | 0.0 | 27.1 | 0.009 | 0.089 | 78865 |  | 0.13 | 20 |
| `x` | bytecode_f64 | harmonic_lla | 0.058 | 5.1 | 0.0 | 51.5 | 1.7 | 0.0 | 41.7 | 0.017 | 0.169 | 27132 |  | 0.01 | 13 |

## Bench — débit par technique (`--wisdom-bench` isolé)

| Technique | iters/s | ns/iter | taille | vs baseline |
|---|---:|---:|---:|---:|
| cpu_std_f64 | 1.593e+10 | 0.063 | 1024² | — |
| cpu_perturb_f64 | 1.986e+10 | 0.050 | 512² | — |
| cpu_perturb_exp | 5.605e+10 | 0.018 | 1024² | — |
| cpu_perturb_dd | 8.538e+08 | 1.171 | 128² | — |
| cpu_perturb_f64_mid | 1.996e+09 | 0.501 | 1024² | — |
| gpu_std_f32 | 1.058e+10 | 0.095 | 2048² | — |
| gpu_perturb_f32 | 5.128e+09 | 0.195 | 1024² | — |

## Tendance (même machine, 6 derniers scores)

| Date (UTC) | Commit | Tier | geomean/F3 | wins | fractall Σs | auto-ratio | parité ok/équiv | Q P/W/F | fuzz P/W/F | goldens | px % | gaps |
|---|---|---|---:|---:|---:|---:|---|---|---|---|---:|---:|
| 2026-11-06 01:01 | `4bfbdba` | quick | — | 0 | 2.280 | 0.695 | —/— | 16/0/0 | 3/0/0 | 🟢 | 46.9 | 0 |
| 2026-11-06 01:52 | `6bb4e9e` | quick | — | 0 | 2.268 | 0.974 | —/— | 16/0/0 | 3/0/0 | 🟢 | 46.9 | 0 |
| 2026-11-06 02:11 | `1b1a394` | quick | — | 0 | 2.306 | 0.998 | —/— | 16/0/0 | 3/0/0 | 🟢 | 47.4 | 0 |
| 2026-11-06 02:15 | `791fec8` | standard | — | 0 | 4.709 | — | —/— | 15/1/0 | 6/0/0 | 🟢 | 50.3 | 1 |
| 2026-11-06 04:39 | `0473296` | quick | — | 0 | 2.261 | 0.981 | —/— | 17/0/0 | 3/0/0 | 🟢 | 47.1 | 0 |
| 2026-11-06 05:10 | `b3ed210` | standard | — | 0 | 4.661 | — | —/— | 16/1/0 | 6/0/0 | 🟢 | 51.4 | 1 |

_`python3 scripts/harness.py trend` pour l'historique complet (`--case <stem>` pour un cas)._

## Gaps (top 10 — sévérité asc, magnitude desc)

| # | Sévérité | Axe | Cas | Métrique | Valeur | Note |
|---:|---|---|---|---|---:|---|
| 1 | 4 qualité | quality | `seahorse-valley` | verdict | WARN | quality suite WARN |

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
