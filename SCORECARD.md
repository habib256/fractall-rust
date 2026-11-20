<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-11-20T08:33:39+00:00
- **Commit** : `7ecd1b7`
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
- temps cumulé : **2.908 s** sur 10 cas mesurés

| Cas | fractall (s) | min (s) | bruit | vs baseline | ratio F3 |
|---|---:|---:|---:|---:|---:|
| `test5` | 0.010 | 0.010 | — | — | — |
| `spiral` | 0.018 | 0.018 | — | — | — |
| `flake` | 0.066 | 0.066 | — | — | — |
| `glitch_test_2` | 0.038 | 0.038 | — | — | — |
| `e50` | 0.571 | 0.571 | — | — | — |
| `e113` | 0.068 | 0.068 | — | — | — |
| `e401` | 0.069 | 0.069 | — | — | — |
| `e1000` | 0.069 | 0.069 | — | — | — |
| `floral_fantasy` | 0.021 | 0.021 | — | — | — |
| `dragon` | 1.979 | 1.979 | — | — | — |

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
| `harmonic:flake` | bytecode_f64_harmonic_lla | 0.030 | `lla` 0.030 s | 1.005 | — | PASS |
| `harmonic:mitosis` | bytecode_f64_harmonic_lla | 0.033 | `lla` 0.031 s | 1.076 | — | PASS |
| `harmonic:dinosaur_fossils` | bytecode_f64_harmonic_lla | 0.069 | `bla` 0.065 s | 1.062 | — | PASS |
| `harmonic:e50` | bytecode_f64 | 0.382 | `bla` 0.346 s | 1.105 | — | PASS |
| `harmonic:e113` | bytecode_f64 | 0.043 | `bla` 0.041 s | 1.048 | — | PASS |
| `algorithm:seahorse_1e7` | std_f64 | 0.018 | `f64` 0.022 s | 0.841 | — | PASS |
| `algorithm:seahorse_1e10` | std_f64 | 0.023 | `f64` 0.023 s | 1.003 | — | PASS |
| `algorithm:seahorse_1e12` | bytecode_f64 | 0.052 | `perturbation` 0.056 s | 0.938 | — | PASS |
| `algorithm:elephants_1e10` | std_f64 | 0.025 | `f64` 0.026 s | 0.961 | — | PASS |

**Itérations automatiques** — juge : le même rendu à 4N.

| Scène | N auto | sous-itération % | suffisant | surcoût temps | sonde (s) | rendu (s) | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `mandelbrot_home` | 750 | 0.000 | 187 | ×0.734 | 0.01 | 0.02 | PASS |
| `julia_home` | 100 | 0.078 | 100 | — | 0.0 | 0.0 | PASS |
| `seahorse_1e4` | 960 | 0.010 | 960 | — | 0.0 | 0.0 | PASS |
| `seahorse_1e7` | 90000 | 0.215 | 90000 | — | 0.14 | 0.3 | PASS |
| `elephants_1e7` | 60000 | 0.019 | 60000 | — | 0.2 | 0.56 | PASS |
| `barnsley_m_zoom` | 60000 | 0.361 | 60000 | — | 0.25 | 0.66 | PASS |
| `multibrot_zoom` | 25000 | 0.185 | 25000 | — | 0.09 | 0.17 | PASS |

## Types (tous les types — vitesse et juge GMP)

- **11 CHAOS** · **53 PASS** — vitesse à 128×80, juge GMP à 64×40 ; temps cumulé **3.303 s**

| Type | Scène | path | s | ns/px·it | it. moy. | juge | verdict |
|---|---|---|---:|---:|---:|---|---|
| 1 VonKoch | home | — | 0.014 | 9219.307 | 0.6 | determinism | PASS |
| 2 Dragon | home | — | 0.045 | 2193.006 | 7.9 | determinism | PASS |
| 3 Mandelbrot | home | std_f64 | 0.010 | 6.273 | 632.9 | 0 px (0.00 %) | PASS |
| 3 Mandelbrot | edge | std_f64 | 0.043 | 5.934 | 2799.4 | 0 px (0.00 %) | PASS |
| 4 Julia | home | std_f64 | 0.008 | 386.122 | 8.4 | 0 px (0.00 %) | PASS |
| 4 Julia | edge | std_f64 | 0.009 | 30.883 | 117.5 | 0 px (0.00 %) | PASS |
| 5 JuliaSin | home | std_f64 | 0.014 | 91.456 | 58.5 | 0 px (0.00 %) | PASS |
| 5 JuliaSin | edge | std_f64 | 0.141 | 36.066 | 1526.9 | 0 px (0.00 %) | PASS |
| 6 Newton | home | std_f64 | 0.008 | 0.762 | 4145.5 | 0 px (0.00 %) | PASS |
| 6 Newton | edge | std_f64 | 0.010 | 1.046 | 3876.2 | 0 px (0.00 %) | PASS |
| 7 Phoenix | home | std_f64 | 0.013 | 31.029 | 160.4 | 0 px (0.00 %) | PASS |
| 7 Phoenix | edge | std_f64 | 0.090 | 7.565 | 4660.7 | 0 px (0.00 %) | PASS |
| 8 Buffalo | home | std_f64 | 0.011 | 5.851 | 747.9 | 53 px (2.07 %) — juge NON convergé (1 % entre 256 et 512 b) | CHAOS |
| 8 Buffalo | edge | std_f64 | 0.024 | 2.989 | 3091.4 | 2194 px (85.70 %) — juge NON convergé (81 % entre 256 et 512 b) | CHAOS |
| 9 BarnsleyJulia | home | std_f64 | 0.015 | 234.161 | 25.2 | 0 px (0.00 %) | PASS |
| 9 BarnsleyJulia | edge | std_f64 | 0.014 | 75.573 | 74.2 | 0 px (0.00 %) | PASS |
| 10 BarnsleyMandelbrot | home | std_f64 | 0.022 | 8.984 | 977.7 | 8 px (0.31 %) — juge NON convergé (0 % entre 256 et 512 b) | CHAOS |
| 10 BarnsleyMandelbrot | edge | std_f64 | 0.072 | 8.829 | 3201.6 | 2099 px (81.99 %) — juge NON convergé (73 % entre 256 et 512 b) | CHAOS |
| 11 MagnetJulia | home | std_f64 | 0.009 | 1.204 | 2961.0 | 0 px (0.00 %) | PASS |
| 11 MagnetJulia | edge | std_f64 | 0.012 | 2.176 | 2099.4 | 0 px (0.00 %) | PASS |
| 12 MagnetMandelbrot | home | std_f64 | 0.020 | 2.142 | 3612.4 | 0 px (0.00 %) | PASS |
| 12 MagnetMandelbrot | edge | std_f64 | 0.344 | 37.073 | 3619.4 | 0 px (0.00 %) | PASS |
| 13 BurningShip | home | std_f64 | 0.012 | 8.579 | 531.3 | 36 px (1.41 %) — juge NON convergé (0 % entre 256 et 512 b) | CHAOS |
| 13 BurningShip | edge | std_f64 | 0.027 | 3.032 | 3438.2 | 1947 px (76.05 %) — juge NON convergé (67 % entre 256 et 512 b) | CHAOS |
| 14 Tricorn | home | std_f64 | 0.012 | 14.732 | 312.2 | 0 px (0.00 %) | PASS |
| 14 Tricorn | edge | std_f64 | 0.032 | 3.791 | 3248.5 | 0 px (0.00 %) | PASS |
| 15 Mandelbulb | home | std_f64 | 0.012 | 3.567 | 1316.8 | 0 px (0.00 %) | PASS |
| 15 Mandelbulb | edge | std_f64 | 0.101 | 15.151 | 2602.0 | 0 px (0.00 %) | PASS |
| 16 Buddhabrot | home | — | 0.016 | 63.332 | 97.4 | determinism | PASS |
| 16 Buddhabrot | edge | — | 0.058 | 331.702 | 68.4 | determinism | PASS |
| 17 Lyapunov | home | — | 0.041 | 22.385 | 711.8 | determinism | PASS |
| 17 Lyapunov | edge | — | 0.041 | 16.379 | 970.7 | determinism | PASS |
| 18 PerpendicularBurningShip | home | std_f64 | 0.013 | 10.349 | 496.4 | 0 px (0.00 %) | PASS |
| 18 PerpendicularBurningShip | edge | std_f64 | 0.024 | 2.816 | 3316.1 | 2072 px (80.94 %) — juge NON convergé (63 % entre 256 et 512 b) | CHAOS |
| 19 Celtic | home | std_f64 | 0.013 | 6.223 | 818.4 | 28 px (1.09 %) — juge NON convergé (1 % entre 256 et 512 b) | CHAOS |
| 19 Celtic | edge | std_f64 | 0.026 | 2.767 | 3646.4 | 1751 px (68.40 %) — juge NON convergé (51 % entre 256 et 512 b) | CHAOS |
| 20 AlphaMandelbrot | home | std_f64 | 0.011 | 5.572 | 749.7 | 0 px (0.00 %) | PASS |
| 20 AlphaMandelbrot | edge | std_f64 | 0.105 | 13.928 | 2950.4 | 0 px (0.00 %) | PASS |
| 21 PickoverStalks | home | std_f64 | 0.011 | 137.737 | 29.7 | 0 px (0.00 %) | PASS |
| 21 PickoverStalks | edge | std_f64 | 0.055 | 16.653 | 1284.5 | 0 px (0.00 %) | PASS |
| 22 Nova | home | std_f64 | 0.011 | 91.785 | 45.2 | 0 px (0.00 %) | PASS |
| 22 Nova | edge | std_f64 | 0.169 | 15.881 | 4155.4 | 0 px (0.00 %) | PASS |
| 23 Multibrot | home | std_f64 | 0.015 | 7.283 | 781.6 | 6 px (0.23 %) — juge NON convergé (0 % entre 256 et 512 b) | CHAOS |
| 23 Multibrot | edge | std_f64 | 0.185 | 24.801 | 2921.9 | 1848 px (72.19 %) — juge NON convergé (15 % entre 256 et 512 b) | CHAOS |
| 24 Nebulabrot | home | — | 0.030 | 0.001 | 9338865.8 | determinism | PASS |
| 24 Nebulabrot | edge | — | 0.101 | 0.009 | 4486589.6 | determinism | PASS |
| 25 BurningShipJulia | home | std_f64 | 0.008 | 603.597 | 4.9 | 0 px (0.00 %) | PASS |
| 25 BurningShipJulia | edge | std_f64 | 0.007 | 117.768 | 24.8 | 0 px (0.00 %) | PASS |
| 26 TricornJulia | home | std_f64 | 0.007 | 486.871 | 5.8 | 0 px (0.00 %) | PASS |
| 26 TricornJulia | edge | std_f64 | 0.008 | 109.39 | 29.5 | 0 px (0.00 %) | PASS |
| 27 CelticJulia | home | std_f64 | 0.008 | 61.193 | 51.1 | 0 px (0.00 %) | PASS |
| 27 CelticJulia | edge | std_f64 | 0.021 | 3.102 | 2620.4 | 0 px (0.00 %) | PASS |
| 28 BuffaloJulia | home | std_f64 | 0.009 | 788.496 | 4.2 | 0 px (0.00 %) | PASS |
| 28 BuffaloJulia | edge | std_f64 | 0.008 | 138.393 | 23.1 | 0 px (0.00 %) | PASS |
| 29 MultibrotJulia | home | std_f64 | 0.011 | 4.123 | 1089.3 | 0 px (0.00 %) | PASS |
| 29 MultibrotJulia | edge | std_f64 | 0.017 | 2.678 | 2419.0 | 0 px (0.00 %) | PASS |
| 30 PerpendicularBurningShipJulia | home | std_f64 | 0.009 | 290.981 | 12.1 | 0 px (0.00 %) | PASS |
| 30 PerpendicularBurningShipJulia | edge | std_f64 | 0.012 | 35.699 | 134.1 | 0 px (0.00 %) | PASS |
| 31 AlphaMandelbrotJulia | home | std_f64 | 0.010 | 1851.671 | 2.2 | 0 px (0.00 %) | PASS |
| 31 AlphaMandelbrotJulia | edge | std_f64 | 0.009 | 194.288 | 17.4 | 0 px (0.00 %) | PASS |
| 32 MandelbrotSin | home | std_f64 | 0.068 | 21.905 | 1209.9 | 0 px (0.00 %) | PASS |
| 32 MandelbrotSin | edge | std_f64 | 0.630 | 82.527 | 2981.4 | 0 px (0.00 %) | PASS |
| 33 AntiBuddhabrot | home | — | 0.060 | 863.784 | 27.0 | determinism | PASS |
| 33 AntiBuddhabrot | edge | — | 0.335 | 582.089 | 225.0 | determinism | PASS |

## Profil — où va le temps (sans F3, 1 run par cas)

- **répartition cumulée** : orbit 33.7 % · bla 2.4 % · pixels 28.2 % · post 25.0 % · save 0.0 % · other 10.7 % (sur 10 cas, 4 threads)
- **phase dominante** : other ×6, pixels ×3, orbit ×1
- paths : bytecode_exp, bytecode_f64, bytecode_f64_harmonic_lla, std_f64 · pic RSS max 312 MB

| Cas | path | variantes | wall (s) | orbite % | BLA % | pixels % | post % | save % | autre % | ns/iter | ns/iter·thread | iter/px | esc.dd | fiab. % | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `test5` | bytecode_f64 | harmonic_lla | 0.034 | 0.0 | 0.0 | 3.0 | 0.0 | 0.0 | 97.0 | 10.172 | 40.690 | 6 |  | 0.00 | 13 |
| `spiral` | std_f64 | — | 0.032 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.700 | — | 251 |  | — | 13 |
| `flake` | bytecode_f64_harmonic_lla | harmonic_lla | 0.056 | 5.4 | 0.0 | 35.9 | 0.0 | 0.0 | 58.7 | 0.040 | 0.158 | 30921 |  | 0.00 | 13 |
| `glitch_test_2` | bytecode_f64_harmonic_lla | harmonic_lla | 0.049 | 2.1 | 0.0 | 32.9 | 0.0 | 0.0 | 65.0 | 0.125 | 0.501 | 7789 |  | 0.00 | 13 |
| `e50` | bytecode_f64 | harmonic_lla | 0.541 | 4.6 | 1.3 | 72.1 | 16.1 | 0.0 | 5.9 | 0.138 | 0.552 | 172456 |  | 0.62 | 20 |
| `e113` | bytecode_f64 | harmonic_lla | 0.082 | 4.9 | 1.2 | 54.8 | 0.0 | 0.0 | 39.1 | 0.121 | 0.483 | 22729 |  | 0.00 | 13 |
| `e401` | bytecode_exp | bla | 0.085 | 10.5 | 5.9 | 38.7 | 7.0 | 0.0 | 37.9 | 0.073 | 0.292 | 27584 |  | 0.07 | 13 |
| `e1000` | bytecode_exp | bla | 0.067 | 19.3 | 3.0 | 29.7 | 0.0 | 0.0 | 48.0 | 0.073 | 0.293 | 16634 |  | 0.00 | 13 |
| `floral_fantasy` | bytecode_f64_harmonic_lla | harmonic_lla | 0.038 | 2.7 | 0.0 | 21.2 | 0.0 | 0.0 | 76.1 | 0.227 | 0.906 | 2156 |  | 0.00 | 13 |
| `dragon` | bytecode_f64 | harmonic_lla | 2.122 | 46.6 | 2.9 | 16.2 | 32.2 | 0.0 | 2.2 | 0.005 | 0.021 | 3921951 |  | 0.70 | 312 |

## Tendance (même machine, 6 derniers scores)

| Date (UTC) | Commit | Tier | geomean/F3 | wins | fractall Σs | auto-ratio | parité ok/équiv | Q P/W/F | fuzz P/W/F | goldens | px % | gaps |
|---|---|---|---:|---:|---:|---:|---|---|---|---|---:|---:|
| 2026-11-19 12:31 | `4ffcf4c` | micro | — | 0 | 4.230 | — | —/— | — | — | — | 28.0 | 8 |
| 2026-11-19 12:52 | `f932132` | micro | — | 0 | 3.822 | — | —/— | — | — | — | 28.7 | 6 |
| 2026-11-20 00:19 | `d2f0f54` | micro | — | 0 | 4.929 | — | —/— | — | — | — | 37.8 | 1 |
| 2026-11-20 05:53 | `c25f508` | micro | — | 0 | 2.761 | — | —/— | — | — | — | 26.8 | 0 |
| 2026-11-20 07:44 | `abf534e` | micro | — | 0 | 3.199 | — | —/— | — | — | — | 29.3 | 0 |
| 2026-11-20 08:33 | `7ecd1b7` | micro | — | 0 | 2.908 | — | —/— | — | — | — | 28.2 | 0 |

_`python3 scripts/harness.py trend` pour l'historique complet (`--case <stem>` pour un cas)._

## Leviers (où gagner — temps mesuré, trié)

_Classement de pistes, pas un budget : les scènes n'ont pas toutes la même taille. Un levier n'est pas un défaut — c'est là que le temps va._

| # | s | Source | Cas | Ce qui coûte | Pour l'ouvrir |
|---:|---:|---|---|---|---|
| 1 | 0.99 | profile | `dragon` | orbite de référence GMP (47 % du cas) | séquentielle — TODO G14.3 (`bench_orbit_join`) |
| 2 | 0.68 | profile | `dragon` | correction de fiabilité (0.7019 % de pixels signalés, 32 % du cas) | `harness.py ab --cases dragon --env FRACTALL_RELIABILITY=off,escalate` — TODO G14.1 |
| 3 | 0.63 | types | `32:MandelbrotSin:edge` | rendu `std_f64` — 82.527 ns/pixel·itération·thread (2981.4 it. moy.) | comparer aux autres types du tableau Types : un ns/px·it hors norme = boucle scalaire à porter (lots, raccourci de cycle) |
| 4 | 0.40 | truth | `std:elephants_1e7_60000` | rendu de production `std_f64` (256×160) | `FRACTALL_PERTURB_STATS=1` : part du traqueur et de la correction |
| 5 | 0.39 | profile | `e50` | boucle pixel `bytecode_f64` (72 % du cas, 0.5521 ns/iter·thread) | `harness.py hotspots --case e50` |
| 6 | 0.34 | types | `12:MagnetMandelbrot:edge` | rendu `std_f64` — 37.073 ns/pixel·itération·thread (3619.4 it. moy.) | comparer aux autres types du tableau Types : un ns/px·it hors norme = boucle scalaire à porter (lots, raccourci de cycle) |
| 7 | 0.34 | profile | `dragon` | boucle pixel `bytecode_f64` (16 % du cas, 0.0214 ns/iter·thread) | `harness.py hotspots --case dragon` |
| 8 | 0.34 | wisdom | `iter:elephants_1e7` | correction de fiabilité du path `std_f64` à N auto = 60000 (43.535 % de pixels signalés, 60 % du rendu) | prix de la correction par pixel — TODO G14.1 |

_… 19 autre(s) dans le history JSON (`harness.py levers`)._

## Gaps (top 10 — sévérité asc, magnitude desc)

_aucun gap détecté 🎉_

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
