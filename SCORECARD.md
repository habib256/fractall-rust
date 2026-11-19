<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-11-19T12:52:07+00:00
- **Commit** : `f932132`  ⚠️ arbre modifié (dirty)
- **Machine** : Intel(R) Xeon(R) Processor @ 2.10GHz · 4 threads · Linux 6.18.44-fc-v653
- **Tier** : micro · 128×128 · quality 48×48 · runs=1 · axes=speed,truth,wisdom,types,profile
- **F3** : — (indisponible)
- **Baseline** : `harness/baseline.json` (cd05f6a · tier standard · 2026-07-12) · **autre machine**
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
- temps cumulé : **3.822 s** sur 10 cas mesurés

| Cas | fractall (s) | min (s) | bruit | vs baseline | ratio F3 |
|---|---:|---:|---:|---:|---:|
| `test5` | 0.009 | 0.009 | — | — | — |
| `spiral` | 0.018 | 0.018 | — | — | — |
| `flake` | 0.068 | 0.068 | — | — | — |
| `glitch_test_2` | 0.037 | 0.037 | — | — | — |
| `e50` | 0.720 | 0.720 | — | — | — |
| `e113` | 0.120 | 0.120 | — | — | — |
| `e401` | 0.166 | 0.166 | — | — | — |
| `e1000` | 0.070 | 0.070 | — | — | — |
| `floral_fantasy` | 0.034 | 0.034 | — | — | — |
| `dragon` | 2.581 | 2.581 | — | — | — |

## Vérité (rendu de production vs tier dd)

- 24 cas à 48² → **24 PASS · 0 WARN · 0 FAIL** (itérations brutes du rendu par défaut vs frame `--dd-tier`)

| Cas | path | pixels ≠ | % | verdict |
|---|---|---:|---:|---|
| `test5` | bytecode_f64 | 0 | 0.000 | PASS |
| `flake` | bytecode_f64_harmonic_lla | 0 | 0.000 | PASS |
| `glitch_test_2` | bytecode_f64_harmonic_lla | 0 | 0.000 | PASS |
| `e50` | bytecode_f64 | 0 | 0.000 | PASS |
| `e113` | bytecode_f64 | 0 | 0.000 | PASS |
| `e401` | bytecode_exp | 0 | 0.000 | PASS |
| `e1000` | bytecode_exp | 0 | 0.000 | PASS |
| `floral_fantasy` | bytecode_f64_harmonic_lla | 0 | 0.000 | PASS |
| `dragon` | bytecode_f64 | 0 | 0.000 | PASS |
| `e227` ⚑ | bytecode_f64 | 0 | 0.000 | PASS |
| `golden_spider` ⚑ | bytecode_f64 | 0 | 0.000 | PASS |
| `rug` ⚑ | bytecode_f64 | 0 | 0.000 | PASS |
| `olbaid2` ⚑ | bytecode_exp | 0 | 0.000 | PASS |
| `leaded_glass` ⚑ | bytecode_f64_harmonic_lla | 0 | 0.000 | PASS |
| `std:seahorse_1e4_2500` | std_f64 | 0 | 0.000 | PASS |
| `std:seahorse_1e7_2500` | std_f64 | 0 | 0.000 | PASS |
| `std:seahorse_1e10_2500` | std_f64 | 0 | 0.000 | PASS |
| `std:elephants_1e4_2500` | std_f64 | 2 | 0.019 | PASS |
| `std:elephants_1e7_2500` | std_f64 | 2 | 0.019 | PASS |
| `std:elephants_1e10_2500` | std_f64 | 0 | 0.000 | PASS |
| `std:minibrot_1e4_2500` | std_f64 | 0 | 0.000 | PASS |
| `std:minibrot_1e7_2500` | std_f64 | 0 | 0.000 | PASS |
| `std:minibrot_1e10_2500` | std_f64 | 0 | 0.000 | PASS |
| `std:elephants_1e7_60000` | std_f64 | 5 | 0.049 | PASS |

_⚑ = sentinelle (rendu par défaut faux avant les correctifs du 2026-11-06). Un FAIL se juge contre le GMP pur : le dd a son propre plancher._

## Wisdom (décisions automatiques : regret et itérations auto)

- **16 PASS** — 128×80, 2 passes entrelacées (médiane)

**Regret de routage** — `auto` contre chaque plan forcé ; le regret ne compte qu'à sortie équivalente.

| Décision | path auto | auto (s) | meilleure équivalente | regret | plus rapide à sortie ≠ | verdict |
|---|---|---:|---|---:|---|---|
| `harmonic:flake` | bytecode_f64_harmonic_lla | 0.035 | `lla` 0.036 s | 0.97 | — | PASS |
| `harmonic:mitosis` | bytecode_f64_harmonic_lla | 0.048 | `lla` 0.049 s | 0.976 | — | PASS |
| `harmonic:dinosaur_fossils` | bytecode_f64_harmonic_lla | 0.123 | `lla` 0.100 s | 1.226 | — | PASS |
| `harmonic:e50` | bytecode_f64 | 0.441 | `bla` 0.503 s | 0.877 | — | PASS |
| `harmonic:e113` | bytecode_f64 | 0.062 | `bla` 0.064 s | 0.977 | — | PASS |
| `algorithm:seahorse_1e7` | std_f64 | 0.031 | `f64` 0.029 s | 1.075 | — | PASS |
| `algorithm:seahorse_1e10` | std_f64 | 0.034 | `f64` 0.039 s | 0.864 | — | PASS |
| `algorithm:seahorse_1e12` | bytecode_f64 | 0.088 | `perturbation` 0.074 s | 1.193 | — | PASS |
| `algorithm:elephants_1e10` | std_f64 | 0.039 | `f64` 0.038 s | 1.017 | — | PASS |

**Itérations automatiques** — juge : le même rendu à 4N.

| Scène | N auto | sous-itération % | suffisant | surcoût temps | sonde (s) | rendu (s) | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `mandelbrot_home` | 750 | 0.000 | 187 | ×1.032 | 0.01 | 0.01 | PASS |
| `julia_home` | 100 | 0.078 | 100 | — | 0.0 | 0.0 | PASS |
| `seahorse_1e4` | 960 | 0.010 | 960 | — | 0.01 | 0.01 | PASS |
| `seahorse_1e7` | 90000 | 0.215 | 90000 | — | 0.69 | 1.27 | PASS |
| `elephants_1e7` | 60000 | 0.019 | 60000 | — | 2.5 | 4.99 | PASS |
| `barnsley_m_zoom` | 60000 | 0.361 | 60000 | — | 0.32 | 0.86 | PASS |
| `multibrot_zoom` | 8900 | 0.244 | 8900 | — | 0.16 | 0.32 | PASS |

## Types (tous les types — vitesse et juge GMP)

- **9 CHAOS** · **2 FAIL** · **10 N/A** · **39 PASS** · **4 WARN** — vitesse à 128×80, juge GMP à 64×40 ; temps cumulé **4.053 s**

| Type | Scène | path | s | ns/px·it | it. moy. | juge | verdict |
|---|---|---|---:|---:|---:|---|---|
| 1 VonKoch | home | — | 0.016 | 10355.62 | 0.6 | determinism | PASS |
| 2 Dragon | home | — | 0.039 | 1931.297 | 7.9 | determinism | PASS |
| 3 Mandelbrot | home | std_f64 | 0.010 | 5.946 | 632.9 | 0 px (0.00 %) | PASS |
| 3 Mandelbrot | edge | std_f64 | 0.172 | 23.957 | 2799.4 | 0 px (0.00 %) | PASS |
| 4 Julia | home | std_f64 | 0.011 | 499.336 | 8.4 | 0 px (0.00 %) | PASS |
| 4 Julia | edge | std_f64 | 0.011 | 37.908 | 117.5 | 0 px (0.00 %) | PASS |
| 5 JuliaSin | home | std_f64 | 0.019 | 129.746 | 58.5 | 0 px (0.00 %) | PASS |
| 5 JuliaSin | edge | std_f64 | 0.229 | 58.618 | 1526.9 | gmp:timeout | N/A |
| 6 Newton | home | std_f64 | 0.011 | 1.053 | 4145.5 | 0 px (0.00 %) | PASS |
| 6 Newton | edge | std_f64 | 0.011 | 1.076 | 3876.2 | 0 px (0.00 %) | PASS |
| 7 Phoenix | home | std_f64 | 0.013 | 32.124 | 160.4 | 0 px (0.00 %) | PASS |
| 7 Phoenix | edge | std_f64 | 0.074 | 6.191 | 4660.6 | 2 px (0.08 %) — juge convergé | WARN |
| 8 Buffalo | home | std_f64 | 0.011 | 5.877 | 747.9 | 53 px (2.07 %) — juge NON convergé (1 % entre 256 et 512 b) | CHAOS |
| 8 Buffalo | edge | std_f64 | 0.043 | 5.488 | 3091.4 | 2194 px (85.70 %) — juge NON convergé (81 % entre 256 et 512 b) | CHAOS |
| 9 BarnsleyJulia | home | std_f64 | 0.011 | 166.91 | 25.2 | 0 px (0.00 %) | PASS |
| 9 BarnsleyJulia | edge | std_f64 | 0.017 | 89.447 | 74.2 | 0 px (0.00 %) | PASS |
| 10 BarnsleyMandelbrot | home | std_f64 | 0.025 | 10.164 | 977.7 | 8 px (0.31 %) — juge NON convergé (0 % entre 256 et 512 b) | CHAOS |
| 10 BarnsleyMandelbrot | edge | std_f64 | 0.095 | 11.548 | 3200.9 | 2098 px (81.95 %) — juge NON convergé (73 % entre 256 et 512 b) | CHAOS |
| 11 MagnetJulia | home | std_f64 | 0.010 | 1.375 | 2962.9 | 1 px (0.04 %) | PASS |
| 11 MagnetJulia | edge | std_f64 | 0.011 | 2.069 | 2103.7 | 8 px (0.31 %) — juge convergé | WARN |
| 12 MagnetMandelbrot | home | std_f64 | 0.015 | 1.587 | 3613.3 | gmp:timeout | N/A |
| 12 MagnetMandelbrot | edge | std_f64 | 0.148 | 15.961 | 3620.0 | gmp:timeout | N/A |
| 13 BurningShip | home | std_f64 | 0.011 | 8.326 | 531.3 | 36 px (1.41 %) — juge NON convergé (0 % entre 256 et 512 b) | CHAOS |
| 13 BurningShip | edge | std_f64 | 0.033 | 3.77 | 3438.2 | 1947 px (76.05 %) — juge NON convergé (67 % entre 256 et 512 b) | CHAOS |
| 14 Tricorn | home | std_f64 | 0.013 | 15.831 | 312.2 | 0 px (0.00 %) | PASS |
| 14 Tricorn | edge | std_f64 | 0.052 | 6.251 | 3248.5 | 0 px (0.00 %) | PASS |
| 15 Mandelbulb | home | std_f64 | 0.010 | 3.062 | 1316.8 | 0 px (0.00 %) | PASS |
| 15 Mandelbulb | edge | std_f64 | 0.079 | 11.871 | 2601.7 | 101 px (3.95 %) — juge convergé | FAIL |
| 16 Buddhabrot | home | — | 0.017 | 69.738 | 97.4 | determinism | PASS |
| 16 Buddhabrot | edge | — | 0.090 | 515.904 | 68.4 | determinism | PASS |
| 17 Lyapunov | home | — | 0.051 | 27.714 | 711.8 | determinism | PASS |
| 17 Lyapunov | edge | — | 0.048 | 19.322 | 970.7 | determinism | PASS |
| 18 PerpendicularBurningShip | home | std_f64 | 0.013 | 10.011 | 496.0 | 2 px (0.08 %) — juge convergé | WARN |
| 18 PerpendicularBurningShip | edge | std_f64 | 0.036 | 4.204 | 3316.1 | 2072 px (80.94 %) — juge NON convergé (63 % entre 256 et 512 b) | CHAOS |
| 19 Celtic | home | std_f64 | 0.014 | 6.543 | 818.4 | 28 px (1.09 %) — juge NON convergé (1 % entre 256 et 512 b) | CHAOS |
| 19 Celtic | edge | std_f64 | 0.033 | 3.551 | 3646.4 | 1751 px (68.40 %) — juge NON convergé (51 % entre 256 et 512 b) | CHAOS |
| 20 AlphaMandelbrot | home | std_f64 | 0.012 | 6.3 | 749.7 | 0 px (0.00 %) | PASS |
| 20 AlphaMandelbrot | edge | std_f64 | 0.081 | 10.691 | 2949.3 | 116 px (4.53 %) — juge convergé | FAIL |
| 21 PickoverStalks | home | std_f64 | 0.012 | 153.167 | 29.7 | 0 px (0.00 %) | PASS |
| 21 PickoverStalks | edge | std_f64 | 0.073 | 22.176 | 1284.5 | 0 px (0.00 %) | PASS |
| 22 Nova | home | std_f64 | 0.011 | 99.042 | 45.2 | 0 px (0.00 %) | PASS |
| 22 Nova | edge | std_f64 | 0.268 | 25.223 | 4155.4 | gmp:timeout | N/A |
| 23 Multibrot | home | std_f64 | 0.033 | 16.425 | 782.2 | gmp:timeout | N/A |
| 23 Multibrot | edge | std_f64 | 0.662 | 87.32 | 2963.7 | gmp:timeout | N/A |
| 24 Nebulabrot | home | — | 0.035 | 0.001 | 9338865.8 | determinism | PASS |
| 24 Nebulabrot | edge | — | 0.157 | 0.014 | 4486589.6 | determinism | PASS |
| 25 BurningShipJulia | home | std_f64 | 0.010 | 831.061 | 4.9 | 0 px (0.00 %) | PASS |
| 25 BurningShipJulia | edge | std_f64 | 0.010 | 155.678 | 24.8 | 0 px (0.00 %) | PASS |
| 26 TricornJulia | home | std_f64 | 0.009 | 600.167 | 5.8 | 0 px (0.00 %) | PASS |
| 26 TricornJulia | edge | std_f64 | 0.011 | 148.301 | 29.5 | 0 px (0.00 %) | PASS |
| 27 CelticJulia | home | std_f64 | 0.010 | 77.69 | 51.1 | 0 px (0.00 %) | PASS |
| 27 CelticJulia | edge | std_f64 | 0.028 | 4.105 | 2620.4 | 0 px (0.00 %) | PASS |
| 28 BuffaloJulia | home | std_f64 | 0.009 | 874.311 | 4.2 | 0 px (0.00 %) | PASS |
| 28 BuffaloJulia | edge | std_f64 | 0.011 | 184.152 | 23.1 | 0 px (0.00 %) | PASS |
| 29 MultibrotJulia | home | std_f64 | 0.021 | 7.575 | 1089.3 | gmp:timeout | N/A |
| 29 MultibrotJulia | edge | std_f64 | 0.033 | 5.259 | 2419.0 | gmp:timeout | N/A |
| 30 PerpendicularBurningShipJulia | home | std_f64 | 0.011 | 373.351 | 12.1 | 0 px (0.00 %) | PASS |
| 30 PerpendicularBurningShipJulia | edge | std_f64 | 0.013 | 37.387 | 134.1 | 9 px (0.35 %) — juge convergé | WARN |
| 31 AlphaMandelbrotJulia | home | std_f64 | 0.011 | 1933.971 | 2.2 | 0 px (0.00 %) | PASS |
| 31 AlphaMandelbrotJulia | edge | std_f64 | 0.011 | 251.039 | 17.4 | 0 px (0.00 %) | PASS |
| 32 MandelbrotSin | home | std_f64 | 0.079 | 25.563 | 1209.9 | gmp:timeout | N/A |
| 32 MandelbrotSin | edge | std_f64 | 0.428 | 56.058 | 2985.7 | gmp:timeout | N/A |
| 33 AntiBuddhabrot | home | — | 0.063 | 909.001 | 27.0 | determinism | PASS |
| 33 AntiBuddhabrot | edge | — | 0.437 | 758.199 | 225.0 | determinism | PASS |

## Profil — où va le temps (sans F3, 1 run par cas)

- **répartition cumulée** : orbit 34.5 % · bla 2.2 % · pixels 28.7 % · post 24.1 % · save 0.0 % · other 10.5 % (sur 10 cas, 4 threads)
- **phase dominante** : other ×5, pixels ×3, orbit ×2
- paths : bytecode_exp, bytecode_f64, bytecode_f64_harmonic_lla, std_f64 · pic RSS max 312 MB

| Cas | path | variantes | wall (s) | orbite % | BLA % | pixels % | post % | save % | autre % | ns/iter | ns/iter·thread | iter/px | esc.dd | fiab. % | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `test5` | bytecode_f64 | harmonic_lla | 0.047 | 2.1 | 0.0 | 2.1 | 0.0 | 0.0 | 95.7 | 10.172 | 40.690 | 6 |  | 0.00 | 12 |
| `spiral` | std_f64 | — | 0.049 | 0.0 | 0.0 | 20.3 | 0.0 | 0.0 | 79.7 | 1.200 | — | 251 |  | — | 12 |
| `flake` | bytecode_f64_harmonic_lla | harmonic_lla | 0.082 | 8.6 | 1.2 | 39.2 | 0.0 | 0.0 | 51.0 | 0.063 | 0.253 | 30921 |  | 0.00 | 12 |
| `glitch_test_2` | bytecode_f64_harmonic_lla | harmonic_lla | 0.069 | 1.5 | 0.0 | 33.5 | 0.0 | 0.0 | 65.0 | 0.180 | 0.721 | 7789 |  | 0.00 | 12 |
| `e50` | bytecode_f64 | harmonic_lla | 0.684 | 5.0 | 1.2 | 71.1 | 17.1 | 0.0 | 5.6 | 0.172 | 0.689 | 172456 |  | 0.62 | 20 |
| `e113` | bytecode_f64 | harmonic_lla | 0.105 | 4.8 | 1.0 | 57.4 | 0.0 | 0.0 | 36.9 | 0.161 | 0.644 | 22729 |  | 0.00 | 12 |
| `e401` | bytecode_exp | bla | 0.154 | 33.7 | 6.5 | 31.8 | 5.2 | 0.0 | 22.8 | 0.108 | 0.434 | 27584 |  | 0.07 | 43 |
| `e1000` | bytecode_exp | bla | 0.092 | 20.6 | 2.2 | 40.2 | 0.0 | 0.0 | 37.0 | 0.136 | 0.543 | 16634 |  | 0.00 | 17 |
| `floral_fantasy` | bytecode_f64_harmonic_lla | harmonic_lla | 0.057 | 1.8 | 0.0 | 19.3 | 0.0 | 0.0 | 78.9 | 0.311 | 1.246 | 2156 |  | 0.00 | 12 |
| `dragon` | bytecode_f64 | harmonic_lla | 2.578 | 47.8 | 2.5 | 16.1 | 31.7 | 0.0 | 1.9 | 0.006 | 0.026 | 3921951 |  | 0.70 | 312 |

## Tendance (même machine, 6 derniers scores)

| Date (UTC) | Commit | Tier | geomean/F3 | wins | fractall Σs | auto-ratio | parité ok/équiv | Q P/W/F | fuzz P/W/F | goldens | px % | gaps |
|---|---|---|---:|---:|---:|---:|---|---|---|---|---:|---:|
| 2026-11-05 13:49 | `3cbed34` | quick | — | — | — | — | —/— | — | — | — | — | 0 |
| 2026-11-19 11:16 | `c35f0e7` | micro | — | 0 | 4.458 | — | —/— | — | — | — | 27.2 | 11 |
| 2026-11-19 11:46 | `f16395a` | micro | — | 0 | 3.595 | — | —/— | — | — | — | 27.7 | 10 |
| 2026-11-19 12:01 | `b7bdab2` | micro | — | 0 | 3.655 | — | —/— | — | — | — | 28.3 | 9 |
| 2026-11-19 12:31 | `4ffcf4c` | micro | — | 0 | 4.230 | — | —/— | — | — | — | 28.0 | 8 |
| 2026-11-19 12:52 | `f932132` | micro | — | 0 | 3.822 | — | —/— | — | — | — | 28.7 | 6 |

_`python3 scripts/harness.py trend` pour l'historique complet (`--case <stem>` pour un cas)._

## Leviers (où gagner — temps mesuré, trié)

_Classement de pistes, pas un budget : les scènes n'ont pas toutes la même taille. Un levier n'est pas un défaut — c'est là que le temps va._

| # | s | Source | Cas | Ce qui coûte | Pour l'ouvrir |
|---:|---:|---|---|---|---|
| 1 | 2.52 | truth | `std:elephants_1e7_60000` | rendu de production `std_f64` (256×160) | `FRACTALL_PERTURB_STATS=1` : part du traqueur et de la correction |
| 2 | 2.50 | wisdom | `iter:elephants_1e7` | sonde d'itérations automatiques (50 % du rendu) | `fractal/wisdom_iter.rs` (tours, taille de sonde) |
| 3 | 2.47 | wisdom | `iter:elephants_1e7` | correction de fiabilité du path `std_f64` à N auto = 60000 (43.535 % de pixels signalés, 49 % du rendu) | prix de la correction par pixel — TODO G14.1 |
| 4 | 1.23 | profile | `dragon` | orbite de référence GMP (48 % du cas) | séquentielle — TODO G14.3 (`bench_orbit_join`) |
| 5 | 0.82 | profile | `dragon` | correction de fiabilité (0.7019 % de pixels signalés, 32 % du cas) | `harness.py ab --cases dragon --env FRACTALL_RELIABILITY=off,escalate` — TODO G14.1 |
| 6 | 0.69 | wisdom | `iter:seahorse_1e7` | sonde d'itérations automatiques (54 % du rendu) | `fractal/wisdom_iter.rs` (tours, taille de sonde) |
| 7 | 0.66 | types | `23:Multibrot:edge` | rendu `std_f64` — 87.32 ns/pixel·itération·thread (2963.7 it. moy.) | comparer aux autres types du tableau Types : un ns/px·it hors norme = boucle scalaire à porter (lots, raccourci de cycle) |
| 8 | 0.49 | profile | `e50` | boucle pixel `bytecode_f64` (71 % du cas, 0.6894 ns/iter·thread) | `harness.py hotspots --case e50` |

_… 24 autre(s) dans le history JSON (`harness.py levers`)._

## Gaps (top 10 — sévérité asc, magnitude desc)

| # | Sévérité | Axe | Cas | Métrique | Valeur | Note |
|---:|---|---|---|---|---:|---|
| 1 | 1 correction | types | `20:AlphaMandelbrot:edge` | frac | 0.045312 | 116 px ≠ GMP pur (4.53 %, path std_f64) — juge convergé (256 b ≡ 512 b) |
| 2 | 1 correction | types | `15:Mandelbulb:edge` | frac | 0.039453 | 101 px ≠ GMP pur (3.95 %, path std_f64) — juge convergé (256 b ≡ 512 b) |
| 3 | 4 qualité | types | `30:PerpendicularBurningShipJulia:edge` | frac | 0.003516 | 9 px ≠ GMP pur (0.35 %, path std_f64) — juge convergé (256 b ≡ 512 b) |
| 4 | 4 qualité | types | `11:MagnetJulia:edge` | frac | 0.003125 | 8 px ≠ GMP pur (0.31 %, path std_f64) — juge convergé (256 b ≡ 512 b) |
| 5 | 4 qualité | types | `7:Phoenix:edge` | frac | 0.000781 | 2 px ≠ GMP pur (0.08 %, path std_f64) — juge convergé (256 b ≡ 512 b) |
| 6 | 4 qualité | types | `18:PerpendicularBurningShip:home` | frac | 0.000781 | 2 px ≠ GMP pur (0.08 %, path std_f64) — juge convergé (256 b ≡ 512 b) |

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
