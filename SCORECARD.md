<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-11-20T00:19:16+00:00
- **Commit** : `d2f0f54`
- **Machine** : Intel(R) Xeon(R) Processor @ 2.10GHz · 4 threads · Linux 6.18.44-fc-v726
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
- temps cumulé : **4.929 s** sur 10 cas mesurés

| Cas | fractall (s) | min (s) | bruit | vs baseline | ratio F3 |
|---|---:|---:|---:|---:|---:|
| `test5` | 0.019 | 0.019 | — | — | — |
| `spiral` | 0.011 | 0.011 | — | — | — |
| `flake` | 0.120 | 0.120 | — | — | — |
| `glitch_test_2` | 0.067 | 0.067 | — | — | — |
| `e50` | 1.257 | 1.257 | — | — | — |
| `e113` | 0.118 | 0.118 | — | — | — |
| `e401` | 0.121 | 0.121 | — | — | — |
| `e1000` | 0.122 | 0.122 | — | — | — |
| `floral_fantasy` | 0.068 | 0.068 | — | — | — |
| `dragon` | 3.026 | 3.026 | — | — | — |

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
| `std:elephants_1e7_60000` | std_f64 | 2 | 0.019 | PASS |

_⚑ = sentinelle (rendu par défaut faux avant les correctifs du 2026-11-06). Un FAIL se juge contre le GMP pur : le dd a son propre plancher._

## Wisdom (décisions automatiques : regret et itérations auto)

- **16 PASS** — 128×80, 2 passes entrelacées (médiane)

**Regret de routage** — `auto` contre chaque plan forcé ; le regret ne compte qu'à sortie équivalente.

| Décision | path auto | auto (s) | meilleure équivalente | regret | plus rapide à sortie ≠ | verdict |
|---|---|---:|---|---:|---|---|
| `harmonic:flake` | bytecode_f64_harmonic_lla | 0.053 | `lla` 0.053 s | 0.996 | — | PASS |
| `harmonic:mitosis` | bytecode_f64_harmonic_lla | 0.040 | `lla` 0.035 s | 1.135 | — | PASS |
| `harmonic:dinosaur_fossils` | bytecode_f64_harmonic_lla | 0.098 | `lla` 0.103 s | 0.95 | — | PASS |
| `harmonic:e50` | bytecode_f64 | 0.641 | `bla` 0.611 s | 1.048 | — | PASS |
| `harmonic:e113` | bytecode_f64 | 0.071 | `bla` 0.076 s | 0.93 | — | PASS |
| `algorithm:seahorse_1e7` | std_f64 | 0.035 | `f64` 0.037 s | 0.945 | — | PASS |
| `algorithm:seahorse_1e10` | std_f64 | 0.041 | `f64` 0.044 s | 0.947 | — | PASS |
| `algorithm:seahorse_1e12` | bytecode_f64 | 0.131 | `perturbation` 0.104 s | 1.252 | — | PASS |
| `algorithm:elephants_1e10` | std_f64 | 0.043 | `f64` 0.041 s | 1.031 | — | PASS |

**Itérations automatiques** — juge : le même rendu à 4N.

| Scène | N auto | sous-itération % | suffisant | surcoût temps | sonde (s) | rendu (s) | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `mandelbrot_home` | 750 | 0.000 | 187 | ×0.914 | 0.0 | 0.0 | PASS |
| `julia_home` | 100 | 0.078 | 100 | — | 0.0 | 0.0 | PASS |
| `seahorse_1e4` | 960 | 0.010 | 960 | — | 0.0 | 0.0 | PASS |
| `seahorse_1e7` | 90000 | 0.215 | 90000 | — | 0.21 | 0.49 | PASS |
| `elephants_1e7` | 60000 | 0.019 | 60000 | — | 0.4 | 1.18 | PASS |
| `barnsley_m_zoom` | 60000 | 0.361 | 60000 | — | 0.38 | 1.09 | PASS |
| `multibrot_zoom` | 25000 | 0.185 | 25000 | — | 0.14 | 0.3 | PASS |

## Types (tous les types — vitesse et juge GMP)

- **11 CHAOS** · **1 N/A** · **51 PASS** · **1 WARN** — vitesse à 128×80, juge GMP à 64×40 ; temps cumulé **11.491 s**

| Type | Scène | path | s | ns/px·it | it. moy. | juge | verdict |
|---|---|---|---:|---:|---:|---|---|
| 1 VonKoch | home | — | 0.018 | 11745.113 | 0.6 | determinism | PASS |
| 2 Dragon | home | — | 0.076 | 3720.173 | 7.9 | determinism | PASS |
| 3 Mandelbrot | home | std_f64 | 0.013 | 8.119 | 632.9 | 0 px (0.00 %) | PASS |
| 3 Mandelbrot | edge | std_f64 | 0.066 | 9.248 | 2799.4 | 0 px (0.00 %) | PASS |
| 4 Julia | home | std_f64 | 0.012 | 570.792 | 8.4 | 0 px (0.00 %) | PASS |
| 4 Julia | edge | std_f64 | 0.011 | 36.136 | 117.5 | 0 px (0.00 %) | PASS |
| 5 JuliaSin | home | std_f64 | 0.027 | 177.563 | 58.5 | 0 px (0.00 %) | PASS |
| 5 JuliaSin | edge | std_f64 | 0.261 | 66.648 | 1526.9 | 0 px (0.00 %) | PASS |
| 6 Newton | home | std_f64 | 0.013 | 1.181 | 4145.5 | 0 px (0.00 %) | PASS |
| 6 Newton | edge | std_f64 | 0.019 | 1.884 | 3876.2 | 0 px (0.00 %) | PASS |
| 7 Phoenix | home | std_f64 | 0.018 | 44.625 | 160.4 | 0 px (0.00 %) | PASS |
| 7 Phoenix | edge | std_f64 | 0.220 | 18.469 | 4660.6 | 2 px (0.08 %) — juge convergé | WARN |
| 8 Buffalo | home | std_f64 | 0.013 | 6.98 | 747.9 | 53 px (2.07 %) — juge NON convergé (1 % entre 256 et 512 b) | CHAOS |
| 8 Buffalo | edge | std_f64 | 0.041 | 5.22 | 3091.4 | 2194 px (85.70 %) — juge NON convergé (81 % entre 256 et 512 b) | CHAOS |
| 9 BarnsleyJulia | home | std_f64 | 0.019 | 291.326 | 25.2 | 0 px (0.00 %) | PASS |
| 9 BarnsleyJulia | edge | std_f64 | 0.024 | 123.964 | 74.2 | 0 px (0.00 %) | PASS |
| 10 BarnsleyMandelbrot | home | std_f64 | 0.024 | 9.74 | 977.7 | 8 px (0.31 %) — juge NON convergé (0 % entre 256 et 512 b) | CHAOS |
| 10 BarnsleyMandelbrot | edge | std_f64 | 0.181 | 22.14 | 3201.6 | 2099 px (81.99 %) — juge NON convergé (73 % entre 256 et 512 b) | CHAOS |
| 11 MagnetJulia | home | std_f64 | 0.019 | 2.504 | 2961.0 | 0 px (0.00 %) | PASS |
| 11 MagnetJulia | edge | std_f64 | 0.012 | 2.277 | 2099.4 | 0 px (0.00 %) | PASS |
| 12 MagnetMandelbrot | home | std_f64 | 0.022 | 2.425 | 3612.4 | 0 px (0.00 %) | PASS |
| 12 MagnetMandelbrot | edge | std_f64 | 1.119 | 120.737 | 3619.4 | 0 px (0.00 %) | PASS |
| 13 BurningShip | home | std_f64 | 0.012 | 8.589 | 531.3 | 36 px (1.41 %) — juge NON convergé (0 % entre 256 et 512 b) | CHAOS |
| 13 BurningShip | edge | std_f64 | 0.051 | 5.765 | 3438.2 | 1947 px (76.05 %) — juge NON convergé (67 % entre 256 et 512 b) | CHAOS |
| 14 Tricorn | home | std_f64 | 0.011 | 13.333 | 312.2 | 0 px (0.00 %) | PASS |
| 14 Tricorn | edge | std_f64 | 0.084 | 10.122 | 3248.5 | 0 px (0.00 %) | PASS |
| 15 Mandelbulb | home | std_f64 | 0.016 | 4.65 | 1316.8 | 0 px (0.00 %) | PASS |
| 15 Mandelbulb | edge | std_f64 | 0.281 | 42.205 | 2602.0 | 0 px (0.00 %) | PASS |
| 16 Buddhabrot | home | — | 0.033 | 132.761 | 97.4 | determinism | PASS |
| 16 Buddhabrot | edge | — | 0.124 | 708.046 | 68.4 | determinism | PASS |
| 17 Lyapunov | home | — | 0.109 | 59.768 | 711.8 | determinism | PASS |
| 17 Lyapunov | edge | — | 0.085 | 34.324 | 970.7 | determinism | PASS |
| 18 PerpendicularBurningShip | home | std_f64 | 0.018 | 14.358 | 496.4 | 0 px (0.00 %) | PASS |
| 18 PerpendicularBurningShip | edge | std_f64 | 0.036 | 4.303 | 3316.1 | 2072 px (80.94 %) — juge NON convergé (63 % entre 256 et 512 b) | CHAOS |
| 19 Celtic | home | std_f64 | 0.012 | 5.69 | 818.4 | 28 px (1.09 %) — juge NON convergé (1 % entre 256 et 512 b) | CHAOS |
| 19 Celtic | edge | std_f64 | 0.042 | 4.498 | 3646.4 | 1751 px (68.40 %) — juge NON convergé (51 % entre 256 et 512 b) | CHAOS |
| 20 AlphaMandelbrot | home | std_f64 | 0.012 | 6.449 | 749.7 | 0 px (0.00 %) | PASS |
| 20 AlphaMandelbrot | edge | std_f64 | 0.393 | 51.96 | 2950.4 | 0 px (0.00 %) | PASS |
| 21 PickoverStalks | home | std_f64 | 0.014 | 187.867 | 29.7 | 0 px (0.00 %) | PASS |
| 21 PickoverStalks | edge | std_f64 | 0.159 | 48.226 | 1284.5 | 0 px (0.00 %) | PASS |
| 22 Nova | home | std_f64 | 0.023 | 197.82 | 45.2 | 0 px (0.00 %) | PASS |
| 22 Nova | edge | std_f64 | 0.441 | 41.45 | 4155.4 | 0 px (0.00 %) | PASS |
| 23 Multibrot | home | std_f64 | 0.024 | 11.922 | 781.6 | 6 px (0.23 %) — juge NON convergé (0 % entre 256 et 512 b) | CHAOS |
| 23 Multibrot | edge | std_f64 | 0.514 | 68.767 | 2921.9 | 1848 px (72.19 %) — juge NON convergé (15 % entre 256 et 512 b) | CHAOS |
| 24 Nebulabrot | home | — | 0.053 | 0.002 | 9338865.8 | determinism | PASS |
| 24 Nebulabrot | edge | — | 0.171 | 0.015 | 4486589.6 | determinism | PASS |
| 25 BurningShipJulia | home | std_f64 | 0.009 | 760.379 | 4.9 | 0 px (0.00 %) | PASS |
| 25 BurningShipJulia | edge | std_f64 | 0.007 | 116.557 | 24.8 | 0 px (0.00 %) | PASS |
| 26 TricornJulia | home | std_f64 | 0.007 | 482.899 | 5.8 | 0 px (0.00 %) | PASS |
| 26 TricornJulia | edge | std_f64 | 0.010 | 136.662 | 29.5 | 0 px (0.00 %) | PASS |
| 27 CelticJulia | home | std_f64 | 0.009 | 65.637 | 51.1 | 0 px (0.00 %) | PASS |
| 27 CelticJulia | edge | std_f64 | 0.035 | 5.177 | 2620.4 | 0 px (0.00 %) | PASS |
| 28 BuffaloJulia | home | std_f64 | 0.015 | 1344.976 | 4.2 | 0 px (0.00 %) | PASS |
| 28 BuffaloJulia | edge | std_f64 | 0.009 | 158.298 | 23.1 | 0 px (0.00 %) | PASS |
| 29 MultibrotJulia | home | std_f64 | 0.011 | 3.986 | 1089.3 | 0 px (0.00 %) | PASS |
| 29 MultibrotJulia | edge | std_f64 | 0.023 | 3.698 | 2419.0 | 0 px (0.00 %) | PASS |
| 30 PerpendicularBurningShipJulia | home | std_f64 | 0.010 | 329.633 | 12.1 | 0 px (0.00 %) | PASS |
| 30 PerpendicularBurningShipJulia | edge | std_f64 | 0.010 | 28.733 | 134.1 | 0 px (0.00 %) | PASS |
| 31 AlphaMandelbrotJulia | home | std_f64 | 0.009 | 1635.894 | 2.2 | 0 px (0.00 %) | PASS |
| 31 AlphaMandelbrotJulia | edge | std_f64 | 0.017 | 379.677 | 17.4 | 0 px (0.00 %) | PASS |
| 32 MandelbrotSin | home | std_f64 | 0.195 | 62.835 | 1209.9 | 0 px (0.00 %) | PASS |
| 32 MandelbrotSin | edge | std_f64 | 5.279 | 691.619 | 2981.4 | gmp:timeout | N/A |
| 33 AntiBuddhabrot | home | — | 0.119 | 1720.653 | 27.0 | determinism | PASS |
| 33 AntiBuddhabrot | edge | — | 0.771 | 1338.575 | 225.0 | determinism | PASS |

## Profil — où va le temps (sans F3, 1 run par cas)

- **répartition cumulée** : orbit 20.9 % · bla 2.3 % · pixels 37.8 % · post 31.8 % · save 0.0 % · other 7.2 % (sur 10 cas, 4 threads)
- **phase dominante** : pixels ×5, other ×4, post ×1
- paths : bytecode_exp, bytecode_f64, bytecode_f64_harmonic_lla, std_f64 · pic RSS max 312 MB

| Cas | path | variantes | wall (s) | orbite % | BLA % | pixels % | post % | save % | autre % | ns/iter | ns/iter·thread | iter/px | esc.dd | fiab. % | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `test5` | bytecode_f64 | harmonic_lla | 0.047 | 0.0 | 0.0 | 4.2 | 0.0 | 0.0 | 95.8 | 20.345 | 81.380 | 6 |  | 0.00 | 13 |
| `spiral` | std_f64 | — | 0.034 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 1.000 | — | 251 |  | — | 12 |
| `flake` | bytecode_f64_harmonic_lla | harmonic_lla | 0.088 | 4.6 | 1.1 | 56.9 | 0.0 | 0.0 | 37.4 | 0.099 | 0.395 | 30921 |  | 0.00 | 13 |
| `glitch_test_2` | bytecode_f64_harmonic_lla | harmonic_lla | 0.089 | 1.1 | 0.0 | 38.4 | 0.0 | 0.0 | 60.5 | 0.266 | 1.066 | 7789 |  | 0.00 | 13 |
| `e50` | bytecode_f64 | harmonic_lla | 1.243 | 1.9 | 0.4 | 75.4 | 18.4 | 0.0 | 3.9 | 0.332 | 1.327 | 172456 |  | 0.62 | 21 |
| `e113` | bytecode_f64 | harmonic_lla | 0.153 | 2.6 | 0.7 | 71.4 | 0.0 | 0.0 | 25.3 | 0.293 | 1.171 | 22729 |  | 0.00 | 13 |
| `e401` | bytecode_exp | bla | 0.145 | 6.9 | 3.4 | 47.4 | 17.9 | 0.0 | 24.4 | 0.153 | 0.611 | 27584 |  | 0.07 | 13 |
| `e1000` | bytecode_exp | bla | 0.128 | 11.0 | 1.6 | 58.8 | 0.0 | 0.0 | 28.7 | 0.275 | 1.101 | 16634 |  | 0.00 | 13 |
| `floral_fantasy` | bytecode_f64_harmonic_lla | harmonic_lla | 0.076 | 1.3 | 0.0 | 38.1 | 5.2 | 0.0 | 55.4 | 0.821 | 3.284 | 2156 |  | 0.00 | 12 |
| `dragon` | bytecode_f64 | harmonic_lla | 3.755 | 30.5 | 3.1 | 23.2 | 41.9 | 0.0 | 1.3 | 0.013 | 0.054 | 3921951 |  | 0.70 | 312 |

## Tendance (même machine, 6 derniers scores)

| Date (UTC) | Commit | Tier | geomean/F3 | wins | fractall Σs | auto-ratio | parité ok/équiv | Q P/W/F | fuzz P/W/F | goldens | px % | gaps |
|---|---|---|---:|---:|---:|---:|---|---|---|---|---:|---:|
| 2026-11-19 11:16 | `c35f0e7` | micro | — | 0 | 4.458 | — | —/— | — | — | — | 27.2 | 11 |
| 2026-11-19 11:46 | `f16395a` | micro | — | 0 | 3.595 | — | —/— | — | — | — | 27.7 | 10 |
| 2026-11-19 12:01 | `b7bdab2` | micro | — | 0 | 3.655 | — | —/— | — | — | — | 28.3 | 9 |
| 2026-11-19 12:31 | `4ffcf4c` | micro | — | 0 | 4.230 | — | —/— | — | — | — | 28.0 | 8 |
| 2026-11-19 12:52 | `f932132` | micro | — | 0 | 3.822 | — | —/— | — | — | — | 28.7 | 6 |
| 2026-11-20 00:19 | `d2f0f54` | micro | — | 0 | 4.929 | — | —/— | — | — | — | 37.8 | 1 |

_`python3 scripts/harness.py trend` pour l'historique complet (`--case <stem>` pour un cas)._

## Leviers (où gagner — temps mesuré, trié)

_Classement de pistes, pas un budget : les scènes n'ont pas toutes la même taille. Un levier n'est pas un défaut — c'est là que le temps va._

| # | s | Source | Cas | Ce qui coûte | Pour l'ouvrir |
|---:|---:|---|---|---|---|
| 1 | 5.28 | types | `32:MandelbrotSin:edge` | rendu `std_f64` — 691.619 ns/pixel·itération·thread (2981.4 it. moy.) | comparer aux autres types du tableau Types : un ns/px·it hors norme = boucle scalaire à porter (lots, raccourci de cycle) |
| 2 | 1.57 | profile | `dragon` | correction de fiabilité (0.7019 % de pixels signalés, 42 % du cas) | `harness.py ab --cases dragon --env FRACTALL_RELIABILITY=off,escalate` — TODO G14.1 |
| 3 | 1.15 | profile | `dragon` | orbite de référence GMP (31 % du cas) | séquentielle — TODO G14.3 (`bench_orbit_join`) |
| 4 | 1.12 | types | `12:MagnetMandelbrot:edge` | rendu `std_f64` — 120.737 ns/pixel·itération·thread (3619.4 it. moy.) | comparer aux autres types du tableau Types : un ns/px·it hors norme = boucle scalaire à porter (lots, raccourci de cycle) |
| 5 | 0.94 | profile | `e50` | boucle pixel `bytecode_f64` (75 % du cas, 1.3265 ns/iter·thread) | `harness.py hotspots --case e50` |
| 6 | 0.87 | profile | `dragon` | boucle pixel `bytecode_f64` (23 % du cas, 0.0542 ns/iter·thread) | `harness.py hotspots --case dragon` |
| 7 | 0.77 | types | `33:AntiBuddhabrot:edge` | rendu `None` — 1338.575 ns/pixel·itération·thread (225.0 it. moy.) | comparer aux autres types du tableau Types : un ns/px·it hors norme = boucle scalaire à porter (lots, raccourci de cycle) |
| 8 | 0.76 | wisdom | `iter:elephants_1e7` | correction de fiabilité du path `std_f64` à N auto = 60000 (43.535 % de pixels signalés, 65 % du rendu) | prix de la correction par pixel — TODO G14.1 |

_… 31 autre(s) dans le history JSON (`harness.py levers`)._

## Gaps (top 10 — sévérité asc, magnitude desc)

| # | Sévérité | Axe | Cas | Métrique | Valeur | Note |
|---:|---|---|---|---|---:|---|
| 1 | 4 qualité | types | `7:Phoenix:edge` | frac | 0.000781 | 2 px ≠ GMP pur (0.08 %, path std_f64) — juge convergé (256 b ≡ 512 b) |

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
