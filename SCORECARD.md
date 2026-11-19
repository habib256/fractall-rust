<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-11-19T12:01:23+00:00
- **Commit** : `b7bdab2`  ⚠️ arbre modifié (dirty)
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
- temps cumulé : **3.655 s** sur 10 cas mesurés

| Cas | fractall (s) | min (s) | bruit | vs baseline | ratio F3 |
|---|---:|---:|---:|---:|---:|
| `test5` | 0.010 | 0.010 | — | — | — |
| `spiral` | 0.011 | 0.011 | — | — | — |
| `flake` | 0.069 | 0.069 | — | — | — |
| `glitch_test_2` | 0.037 | 0.037 | — | — | — |
| `e50` | 0.672 | 0.672 | — | — | — |
| `e113` | 0.119 | 0.119 | — | — | — |
| `e401` | 0.116 | 0.116 | — | — | — |
| `e1000` | 0.066 | 0.066 | — | — | — |
| `floral_fantasy` | 0.023 | 0.023 | — | — | — |
| `dragon` | 2.531 | 2.531 | — | — | — |

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
| `harmonic:flake` | bytecode_f64_harmonic_lla | 0.040 | `lla` 0.041 s | 0.978 | — | PASS |
| `harmonic:mitosis` | bytecode_f64_harmonic_lla | 0.050 | `lla` 0.042 s | 1.198 | — | PASS |
| `harmonic:dinosaur_fossils` | bytecode_f64_harmonic_lla | 0.081 | `bla` 0.079 s | 1.028 | — | PASS |
| `harmonic:e50` | bytecode_f64 | 0.475 | `bla` 0.445 s | 1.068 | — | PASS |
| `harmonic:e113` | bytecode_f64 | 0.062 | `bla` 0.060 s | 1.028 | — | PASS |
| `algorithm:seahorse_1e7` | std_f64 | 0.029 | `f64` 0.029 s | 0.977 | — | PASS |
| `algorithm:seahorse_1e10` | std_f64 | 0.038 | `f64` 0.033 s | 1.149 | — | PASS |
| `algorithm:seahorse_1e12` | bytecode_f64 | 0.082 | `perturbation` 0.073 s | 1.126 | — | PASS |
| `algorithm:elephants_1e10` | std_f64 | 0.035 | `f64` 0.036 s | 0.969 | — | PASS |

**Itérations automatiques** — juge : le même rendu à 4N.

| Scène | N auto | sous-itération % | suffisant | surcoût temps | sonde (s) | rendu (s) | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `mandelbrot_home` | 750 | 0.000 | 187 | ×0.945 | 0.01 | 0.01 | PASS |
| `julia_home` | 100 | 0.078 | 100 | — | 0.0 | 0.0 | PASS |
| `seahorse_1e4` | 960 | 0.010 | 960 | — | 0.01 | 0.01 | PASS |
| `seahorse_1e7` | 90000 | 0.215 | 90000 | — | 0.69 | 1.27 | PASS |
| `elephants_1e7` | 60000 | 0.019 | 60000 | — | 1.93 | 4.61 | PASS |
| `barnsley_m_zoom` | 52000 | 0.391 | 52000 | — | 0.23 | 0.66 | PASS |
| `multibrot_zoom` | 8900 | 0.244 | 8900 | — | 0.14 | 0.31 | PASS |

## Types (tous les types — vitesse et juge GMP)

- **9 CHAOS** · **5 FAIL** · **11 N/A** · **35 PASS** · **4 WARN** — vitesse à 128×80, juge GMP à 64×40 ; temps cumulé **4.754 s**

| Type | Scène | path | s | ns/px·it | it. moy. | juge | verdict |
|---|---|---|---:|---:|---:|---|---|
| 1 VonKoch | home | — | 0.013 | 8082.941 | 0.6 | determinism | PASS |
| 2 Dragon | home | — | 0.043 | 2134.729 | 7.9 | determinism | PASS |
| 3 Mandelbrot | home | std_f64 | 0.010 | 6.278 | 632.9 | 0 px (0.00 %) | PASS |
| 3 Mandelbrot | edge | std_f64 | 0.154 | 21.547 | 2799.4 | 0 px (0.00 %) | PASS |
| 4 Julia | home | std_f64 | 0.009 | 435.047 | 8.4 | 0 px (0.00 %) | PASS |
| 4 Julia | edge | std_f64 | 0.011 | 36.119 | 117.5 | 0 px (0.00 %) | PASS |
| 5 JuliaSin | home | std_f64 | 0.021 | 137.157 | 58.5 | 0 px (0.00 %) | PASS |
| 5 JuliaSin | edge | std_f64 | 0.211 | 53.966 | 1526.9 | gmp:timeout | N/A |
| 6 Newton | home | std_f64 | 0.205 | 19.318 | 4145.5 | gmp:timeout | N/A |
| 6 Newton | edge | std_f64 | 0.845 | 85.199 | 3876.2 | 0 px (0.00 %) | PASS |
| 7 Phoenix | home | std_f64 | 0.014 | 34.726 | 160.4 | 0 px (0.00 %) | PASS |
| 7 Phoenix | edge | std_f64 | 0.075 | 6.319 | 4660.6 | 2 px (0.08 %) — juge convergé | WARN |
| 8 Buffalo | home | std_f64 | 0.016 | 8.289 | 747.9 | 53 px (2.07 %) — juge NON convergé (1 % entre 256 et 512 b) | CHAOS |
| 8 Buffalo | edge | std_f64 | 0.029 | 3.729 | 3091.4 | 2194 px (85.70 %) — juge NON convergé (81 % entre 256 et 512 b) | CHAOS |
| 9 BarnsleyJulia | home | std_f64 | 0.009 | 131.747 | 25.3 | 27 px (1.05 %) — juge convergé | FAIL |
| 9 BarnsleyJulia | edge | std_f64 | 0.011 | 56.745 | 74.2 | 129 px (5.04 %) — juge convergé | FAIL |
| 10 BarnsleyMandelbrot | home | std_f64 | 0.017 | 6.75 | 978.4 | 12 px (0.47 %) — juge NON convergé (0 % entre 256 et 512 b) | CHAOS |
| 10 BarnsleyMandelbrot | edge | std_f64 | 0.080 | 9.573 | 3249.4 | 2141 px (83.63 %) — juge NON convergé (73 % entre 256 et 512 b) | CHAOS |
| 11 MagnetJulia | home | std_f64 | 0.010 | 1.315 | 2962.9 | 1 px (0.04 %) | PASS |
| 11 MagnetJulia | edge | std_f64 | 0.011 | 2.045 | 2103.7 | 8 px (0.31 %) — juge convergé | WARN |
| 12 MagnetMandelbrot | home | std_f64 | 0.016 | 1.76 | 3613.3 | gmp:timeout | N/A |
| 12 MagnetMandelbrot | edge | std_f64 | 0.152 | 16.423 | 3620.0 | gmp:timeout | N/A |
| 13 BurningShip | home | std_f64 | 0.015 | 11.404 | 531.3 | 36 px (1.41 %) — juge NON convergé (0 % entre 256 et 512 b) | CHAOS |
| 13 BurningShip | edge | std_f64 | 0.031 | 3.555 | 3438.2 | 1947 px (76.05 %) — juge NON convergé (67 % entre 256 et 512 b) | CHAOS |
| 14 Tricorn | home | std_f64 | 0.013 | 15.841 | 312.3 | 0 px (0.00 %) | PASS |
| 14 Tricorn | edge | std_f64 | 0.028 | 3.364 | 3248.7 | 65 px (2.54 %) — juge convergé | FAIL |
| 15 Mandelbulb | home | std_f64 | 0.012 | 3.612 | 1316.8 | 0 px (0.00 %) | PASS |
| 15 Mandelbulb | edge | std_f64 | 0.073 | 10.962 | 2601.7 | 101 px (3.95 %) — juge convergé | FAIL |
| 16 Buddhabrot | home | — | 0.016 | 65.367 | 97.4 | determinism | PASS |
| 16 Buddhabrot | edge | — | 0.080 | 455.265 | 68.4 | determinism | PASS |
| 17 Lyapunov | home | — | 0.045 | 24.954 | 711.8 | determinism | PASS |
| 17 Lyapunov | edge | — | 0.046 | 18.341 | 970.7 | determinism | PASS |
| 18 PerpendicularBurningShip | home | std_f64 | 0.011 | 8.55 | 496.0 | 2 px (0.08 %) — juge convergé | WARN |
| 18 PerpendicularBurningShip | edge | std_f64 | 0.032 | 3.773 | 3316.1 | 2072 px (80.94 %) — juge NON convergé (63 % entre 256 et 512 b) | CHAOS |
| 19 Celtic | home | std_f64 | 0.011 | 5.064 | 818.4 | 28 px (1.09 %) — juge NON convergé (1 % entre 256 et 512 b) | CHAOS |
| 19 Celtic | edge | std_f64 | 0.026 | 2.779 | 3646.4 | 1751 px (68.40 %) — juge NON convergé (51 % entre 256 et 512 b) | CHAOS |
| 20 AlphaMandelbrot | home | std_f64 | 0.010 | 5.284 | 749.7 | 0 px (0.00 %) | PASS |
| 20 AlphaMandelbrot | edge | std_f64 | 0.077 | 10.157 | 2949.3 | 116 px (4.53 %) — juge convergé | FAIL |
| 21 PickoverStalks | home | std_f64 | 0.012 | 157.96 | 29.7 | 0 px (0.00 %) | PASS |
| 21 PickoverStalks | edge | std_f64 | 0.066 | 20.193 | 1284.5 | 0 px (0.00 %) | PASS |
| 22 Nova | home | std_f64 | 0.011 | 95.606 | 45.2 | 0 px (0.00 %) | PASS |
| 22 Nova | edge | std_f64 | 0.234 | 21.998 | 4155.4 | gmp:timeout | N/A |
| 23 Multibrot | home | std_f64 | 0.033 | 16.69 | 782.2 | gmp:timeout | N/A |
| 23 Multibrot | edge | std_f64 | 0.621 | 81.85 | 2963.7 | gmp:timeout | N/A |
| 24 Nebulabrot | home | — | 0.034 | 0.001 | 9338865.8 | determinism | PASS |
| 24 Nebulabrot | edge | — | 0.123 | 0.011 | 4486589.6 | determinism | PASS |
| 25 BurningShipJulia | home | std_f64 | 0.007 | 594.058 | 4.9 | 0 px (0.00 %) | PASS |
| 25 BurningShipJulia | edge | std_f64 | 0.009 | 138.73 | 24.8 | 0 px (0.00 %) | PASS |
| 26 TricornJulia | home | std_f64 | 0.009 | 584.495 | 5.8 | 0 px (0.00 %) | PASS |
| 26 TricornJulia | edge | std_f64 | 0.008 | 106.52 | 29.5 | 0 px (0.00 %) | PASS |
| 27 CelticJulia | home | std_f64 | 0.008 | 61.972 | 51.1 | 0 px (0.00 %) | PASS |
| 27 CelticJulia | edge | std_f64 | 0.022 | 3.303 | 2620.4 | 0 px (0.00 %) | PASS |
| 28 BuffaloJulia | home | std_f64 | 0.008 | 748.125 | 4.2 | 0 px (0.00 %) | PASS |
| 28 BuffaloJulia | edge | std_f64 | 0.009 | 147.943 | 23.1 | 0 px (0.00 %) | PASS |
| 29 MultibrotJulia | home | std_f64 | 0.019 | 6.764 | 1089.3 | gmp:timeout | N/A |
| 29 MultibrotJulia | edge | std_f64 | 0.034 | 5.506 | 2419.0 | gmp:timeout | N/A |
| 30 PerpendicularBurningShipJulia | home | std_f64 | 0.009 | 278.761 | 12.1 | 0 px (0.00 %) | PASS |
| 30 PerpendicularBurningShipJulia | edge | std_f64 | 0.010 | 30.369 | 134.1 | 9 px (0.35 %) — juge convergé | WARN |
| 31 AlphaMandelbrotJulia | home | std_f64 | 0.008 | 1480.405 | 2.2 | 0 px (0.00 %) | PASS |
| 31 AlphaMandelbrotJulia | edge | std_f64 | 0.008 | 177.398 | 17.4 | 0 px (0.00 %) | PASS |
| 32 MandelbrotSin | home | std_f64 | 0.074 | 23.934 | 1209.9 | gmp:timeout | N/A |
| 32 MandelbrotSin | edge | std_f64 | 0.412 | 53.876 | 2985.7 | gmp:timeout | N/A |
| 33 AntiBuddhabrot | home | — | 0.067 | 966.391 | 27.0 | determinism | PASS |
| 33 AntiBuddhabrot | edge | — | 0.408 | 708.32 | 225.0 | determinism | PASS |

## Profil — où va le temps (sans F3, 1 run par cas)

- **répartition cumulée** : orbit 33.7 % · bla 2.3 % · pixels 28.3 % · post 26.2 % · save 0.0 % · other 9.5 % (sur 10 cas, 4 threads)
- **phase dominante** : other ×5, pixels ×4, orbit ×1
- paths : bytecode_exp, bytecode_f64, bytecode_f64_harmonic_lla, std_f64 · pic RSS max 312 MB

| Cas | path | variantes | wall (s) | orbite % | BLA % | pixels % | post % | save % | autre % | ns/iter | ns/iter·thread | iter/px | esc.dd | fiab. % | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `test5` | bytecode_f64 | harmonic_lla | 0.034 | 3.0 | 0.0 | 3.0 | 0.0 | 0.0 | 94.0 | 10.172 | 40.690 | 6 |  | 0.00 | 13 |
| `spiral` | std_f64 | — | 0.036 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.900 | — | 251 |  | — | 13 |
| `flake` | bytecode_f64_harmonic_lla | harmonic_lla | 0.067 | 6.0 | 1.5 | 44.8 | 0.0 | 0.0 | 47.7 | 0.059 | 0.237 | 30921 |  | 0.00 | 13 |
| `glitch_test_2` | bytecode_f64_harmonic_lla | harmonic_lla | 0.058 | 1.7 | 0.0 | 36.2 | 0.0 | 0.0 | 62.0 | 0.165 | 0.658 | 7789 |  | 0.00 | 13 |
| `e50` | bytecode_f64 | harmonic_lla | 0.667 | 5.1 | 1.2 | 69.8 | 18.6 | 0.0 | 5.3 | 0.165 | 0.660 | 172456 |  | 0.62 | 20 |
| `e113` | bytecode_f64 | harmonic_lla | 0.107 | 4.7 | 0.9 | 55.3 | 0.0 | 0.0 | 39.1 | 0.158 | 0.634 | 22729 |  | 0.00 | 13 |
| `e401` | bytecode_exp | bla | 0.138 | 28.2 | 4.3 | 37.7 | 5.1 | 0.0 | 24.7 | 0.115 | 0.460 | 27584 |  | 0.07 | 43 |
| `e1000` | bytecode_exp | bla | 0.103 | 24.4 | 1.9 | 37.0 | 0.0 | 0.0 | 36.7 | 0.139 | 0.558 | 16634 |  | 0.00 | 17 |
| `floral_fantasy` | bytecode_f64_harmonic_lla | harmonic_lla | 0.050 | 2.0 | 0.0 | 24.1 | 0.0 | 0.0 | 73.9 | 0.340 | 1.359 | 2156 |  | 0.00 | 13 |
| `dragon` | bytecode_f64 | harmonic_lla | 2.589 | 45.8 | 2.7 | 15.8 | 33.9 | 0.0 | 1.8 | 0.006 | 0.025 | 3921951 |  | 0.70 | 312 |

## Tendance (même machine, 6 derniers scores)

| Date (UTC) | Commit | Tier | geomean/F3 | wins | fractall Σs | auto-ratio | parité ok/équiv | Q P/W/F | fuzz P/W/F | goldens | px % | gaps |
|---|---|---|---:|---:|---:|---:|---|---|---|---|---:|---:|
| 2026-11-04 08:07 | `fbbbb52` | quick | — | — | — | — | —/— | 16/0/0 | 3/0/0 | 🟢 | 57.8 | 0 |
| 2026-11-05 13:32 | `ed33ef3` | quick | — | — | — | — | —/— | 16/0/0 | 3/0/0 | — | 60.5 | 0 |
| 2026-11-05 13:49 | `3cbed34` | quick | — | — | — | — | —/— | — | — | — | — | 0 |
| 2026-11-19 11:16 | `c35f0e7` | micro | — | 0 | 4.458 | — | —/— | — | — | — | 27.2 | 11 |
| 2026-11-19 11:46 | `f16395a` | micro | — | 0 | 3.595 | — | —/— | — | — | — | 27.7 | 10 |
| 2026-11-19 12:01 | `b7bdab2` | micro | — | 0 | 3.655 | — | —/— | — | — | — | 28.3 | 9 |

_`python3 scripts/harness.py trend` pour l'historique complet (`--case <stem>` pour un cas)._

## Leviers (où gagner — temps mesuré, trié)

_Classement de pistes, pas un budget : les scènes n'ont pas toutes la même taille. Un levier n'est pas un défaut — c'est là que le temps va._

| # | s | Source | Cas | Ce qui coûte | Pour l'ouvrir |
|---:|---:|---|---|---|---|
| 1 | 2.65 | wisdom | `iter:elephants_1e7` | correction de fiabilité du path `std_f64` à N auto = 60000 (43.535 % de pixels signalés, 57 % du rendu) | prix de la correction par pixel — TODO G14.1 |
| 2 | 2.44 | truth | `std:elephants_1e7_60000` | rendu de production `std_f64` (256×160) | `FRACTALL_PERTURB_STATS=1` : part du traqueur et de la correction |
| 3 | 1.93 | wisdom | `iter:elephants_1e7` | sonde d'itérations automatiques (42 % du rendu) | `fractal/wisdom_iter.rs` (tours, taille de sonde) |
| 4 | 1.19 | profile | `dragon` | orbite de référence GMP (46 % du cas) | séquentielle — TODO G14.3 (`bench_orbit_join`) |
| 5 | 0.88 | profile | `dragon` | correction de fiabilité (0.7019 % de pixels signalés, 34 % du cas) | `harness.py ab --cases dragon --env FRACTALL_RELIABILITY=off,escalate` — TODO G14.1 |
| 6 | 0.84 | types | `6:Newton:edge` | rendu `std_f64` — 85.199 ns/pixel·itération·thread (3876.2 it. moy.) | comparer aux autres types du tableau Types : un ns/px·it hors norme = boucle scalaire à porter (lots, raccourci de cycle) |
| 7 | 0.69 | wisdom | `iter:seahorse_1e7` | sonde d'itérations automatiques (54 % du rendu) | `fractal/wisdom_iter.rs` (tours, taille de sonde) |
| 8 | 0.62 | types | `23:Multibrot:edge` | rendu `std_f64` — 81.85 ns/pixel·itération·thread (2963.7 it. moy.) | comparer aux autres types du tableau Types : un ns/px·it hors norme = boucle scalaire à porter (lots, raccourci de cycle) |

_… 24 autre(s) dans le history JSON (`harness.py levers`)._

## Gaps (top 10 — sévérité asc, magnitude desc)

| # | Sévérité | Axe | Cas | Métrique | Valeur | Note |
|---:|---|---|---|---|---:|---|
| 1 | 1 correction | types | `9:BarnsleyJulia:edge` | frac | 0.050391 | 129 px ≠ GMP pur (5.04 %, path std_f64) — juge convergé (256 b ≡ 512 b) |
| 2 | 1 correction | types | `20:AlphaMandelbrot:edge` | frac | 0.045312 | 116 px ≠ GMP pur (4.53 %, path std_f64) — juge convergé (256 b ≡ 512 b) |
| 3 | 1 correction | types | `15:Mandelbulb:edge` | frac | 0.039453 | 101 px ≠ GMP pur (3.95 %, path std_f64) — juge convergé (256 b ≡ 512 b) |
| 4 | 1 correction | types | `14:Tricorn:edge` | frac | 0.025391 | 65 px ≠ GMP pur (2.54 %, path std_f64) — juge convergé (256 b ≡ 512 b) |
| 5 | 1 correction | types | `9:BarnsleyJulia:home` | frac | 0.010547 | 27 px ≠ GMP pur (1.05 %, path std_f64) — juge convergé (256 b ≡ 512 b) |
| 6 | 4 qualité | types | `30:PerpendicularBurningShipJulia:edge` | frac | 0.003516 | 9 px ≠ GMP pur (0.35 %, path std_f64) — juge convergé (256 b ≡ 512 b) |
| 7 | 4 qualité | types | `11:MagnetJulia:edge` | frac | 0.003125 | 8 px ≠ GMP pur (0.31 %, path std_f64) — juge convergé (256 b ≡ 512 b) |
| 8 | 4 qualité | types | `7:Phoenix:edge` | frac | 0.000781 | 2 px ≠ GMP pur (0.08 %, path std_f64) — juge convergé (256 b ≡ 512 b) |
| 9 | 4 qualité | types | `18:PerpendicularBurningShip:home` | frac | 0.000781 | 2 px ≠ GMP pur (0.08 %, path std_f64) — juge convergé (256 b ≡ 512 b) |

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
