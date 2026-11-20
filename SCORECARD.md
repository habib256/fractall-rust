<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-11-20T07:44:55+00:00
- **Commit** : `abf534e`
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
- temps cumulé : **3.199 s** sur 10 cas mesurés

| Cas | fractall (s) | min (s) | bruit | vs baseline | ratio F3 |
|---|---:|---:|---:|---:|---:|
| `test5` | 0.010 | 0.010 | — | — | — |
| `spiral` | 0.010 | 0.010 | — | — | — |
| `flake` | 0.068 | 0.068 | — | — | — |
| `glitch_test_2` | 0.037 | 0.037 | — | — | — |
| `e50` | 0.520 | 0.520 | — | — | — |
| `e113` | 0.116 | 0.116 | — | — | — |
| `e401` | 0.121 | 0.121 | — | — | — |
| `e1000` | 0.070 | 0.070 | — | — | — |
| `floral_fantasy` | 0.019 | 0.019 | — | — | — |
| `dragon` | 2.227 | 2.227 | — | — | — |

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
| `harmonic:flake` | bytecode_f64_harmonic_lla | 0.039 | `lla` 0.038 s | 1.041 | — | PASS |
| `harmonic:mitosis` | bytecode_f64_harmonic_lla | 0.036 | `lla` 0.034 s | 1.05 | — | PASS |
| `harmonic:dinosaur_fossils` | bytecode_f64_harmonic_lla | 0.081 | `lla` 0.082 s | 0.99 | — | PASS |
| `harmonic:e50` | bytecode_f64 | 0.378 | `bla` 0.380 s | 0.996 | — | PASS |
| `harmonic:e113` | bytecode_f64 | 0.047 | `bla` 0.053 s | 0.887 | — | PASS |
| `algorithm:seahorse_1e7` | std_f64 | 0.026 | `f64` 0.025 s | 1.037 | — | PASS |
| `algorithm:seahorse_1e10` | std_f64 | 0.027 | `f64` 0.027 s | 1.0 | — | PASS |
| `algorithm:seahorse_1e12` | bytecode_f64 | 0.056 | `perturbation` 0.059 s | 0.954 | — | PASS |
| `algorithm:elephants_1e10` | std_f64 | 0.030 | `f64` 0.029 s | 1.051 | — | PASS |

**Itérations automatiques** — juge : le même rendu à 4N.

| Scène | N auto | sous-itération % | suffisant | surcoût temps | sonde (s) | rendu (s) | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `mandelbrot_home` | 750 | 0.000 | 187 | ×1.008 | 0.0 | 0.0 | PASS |
| `julia_home` | 100 | 0.078 | 100 | — | 0.0 | 0.0 | PASS |
| `seahorse_1e4` | 960 | 0.010 | 960 | — | 0.0 | 0.01 | PASS |
| `seahorse_1e7` | 90000 | 0.215 | 90000 | — | 0.14 | 0.33 | PASS |
| `elephants_1e7` | 60000 | 0.019 | 60000 | — | 0.22 | 0.6 | PASS |
| `barnsley_m_zoom` | 60000 | 0.361 | 60000 | — | 0.24 | 0.62 | PASS |
| `multibrot_zoom` | 25000 | 0.185 | 25000 | — | 0.1 | 0.18 | PASS |

## Types (tous les types — vitesse et juge GMP)

- **11 CHAOS** · **53 PASS** — vitesse à 128×80, juge GMP à 64×40 ; temps cumulé **3.482 s**

| Type | Scène | path | s | ns/px·it | it. moy. | juge | verdict |
|---|---|---|---:|---:|---:|---|---|
| 1 VonKoch | home | — | 0.014 | 9179.981 | 0.6 | determinism | PASS |
| 2 Dragon | home | — | 0.049 | 2387.204 | 7.9 | determinism | PASS |
| 3 Mandelbrot | home | std_f64 | 0.011 | 6.787 | 632.9 | 0 px (0.00 %) | PASS |
| 3 Mandelbrot | edge | std_f64 | 0.042 | 5.839 | 2799.4 | 0 px (0.00 %) | PASS |
| 4 Julia | home | std_f64 | 0.009 | 422.158 | 8.4 | 0 px (0.00 %) | PASS |
| 4 Julia | edge | std_f64 | 0.009 | 30.046 | 117.5 | 0 px (0.00 %) | PASS |
| 5 JuliaSin | home | std_f64 | 0.017 | 115.33 | 58.5 | 0 px (0.00 %) | PASS |
| 5 JuliaSin | edge | std_f64 | 0.147 | 37.508 | 1526.9 | 0 px (0.00 %) | PASS |
| 6 Newton | home | std_f64 | 0.014 | 1.306 | 4145.5 | 0 px (0.00 %) | PASS |
| 6 Newton | edge | std_f64 | 0.010 | 1.013 | 3876.2 | 0 px (0.00 %) | PASS |
| 7 Phoenix | home | std_f64 | 0.018 | 43.552 | 160.4 | 0 px (0.00 %) | PASS |
| 7 Phoenix | edge | std_f64 | 0.093 | 7.795 | 4660.7 | 0 px (0.00 %) | PASS |
| 8 Buffalo | home | std_f64 | 0.012 | 6.094 | 747.9 | 53 px (2.07 %) — juge NON convergé (1 % entre 256 et 512 b) | CHAOS |
| 8 Buffalo | edge | std_f64 | 0.027 | 3.451 | 3091.4 | 2194 px (85.70 %) — juge NON convergé (81 % entre 256 et 512 b) | CHAOS |
| 9 BarnsleyJulia | home | std_f64 | 0.010 | 157.9 | 25.2 | 0 px (0.00 %) | PASS |
| 9 BarnsleyJulia | edge | std_f64 | 0.013 | 71.178 | 74.2 | 0 px (0.00 %) | PASS |
| 10 BarnsleyMandelbrot | home | std_f64 | 0.019 | 7.414 | 977.7 | 8 px (0.31 %) — juge NON convergé (0 % entre 256 et 512 b) | CHAOS |
| 10 BarnsleyMandelbrot | edge | std_f64 | 0.081 | 9.854 | 3201.6 | 2099 px (81.99 %) — juge NON convergé (73 % entre 256 et 512 b) | CHAOS |
| 11 MagnetJulia | home | std_f64 | 0.010 | 1.262 | 2961.0 | 0 px (0.00 %) | PASS |
| 11 MagnetJulia | edge | std_f64 | 0.011 | 2.138 | 2099.4 | 0 px (0.00 %) | PASS |
| 12 MagnetMandelbrot | home | std_f64 | 0.025 | 2.72 | 3612.4 | 0 px (0.00 %) | PASS |
| 12 MagnetMandelbrot | edge | std_f64 | 0.359 | 38.767 | 3619.4 | 0 px (0.00 %) | PASS |
| 13 BurningShip | home | std_f64 | 0.011 | 8.378 | 531.3 | 36 px (1.41 %) — juge NON convergé (0 % entre 256 et 512 b) | CHAOS |
| 13 BurningShip | edge | std_f64 | 0.028 | 3.169 | 3438.2 | 1947 px (76.05 %) — juge NON convergé (67 % entre 256 et 512 b) | CHAOS |
| 14 Tricorn | home | std_f64 | 0.013 | 15.654 | 312.2 | 0 px (0.00 %) | PASS |
| 14 Tricorn | edge | std_f64 | 0.029 | 3.511 | 3248.5 | 0 px (0.00 %) | PASS |
| 15 Mandelbulb | home | std_f64 | 0.013 | 3.715 | 1316.8 | 0 px (0.00 %) | PASS |
| 15 Mandelbulb | edge | std_f64 | 0.100 | 14.996 | 2602.0 | 0 px (0.00 %) | PASS |
| 16 Buddhabrot | home | — | 0.021 | 82.727 | 97.4 | determinism | PASS |
| 16 Buddhabrot | edge | — | 0.061 | 348.621 | 68.4 | determinism | PASS |
| 17 Lyapunov | home | — | 0.040 | 21.735 | 711.8 | determinism | PASS |
| 17 Lyapunov | edge | — | 0.037 | 14.877 | 970.7 | determinism | PASS |
| 18 PerpendicularBurningShip | home | std_f64 | 0.016 | 12.369 | 496.4 | 0 px (0.00 %) | PASS |
| 18 PerpendicularBurningShip | edge | std_f64 | 0.023 | 2.734 | 3316.1 | 2072 px (80.94 %) — juge NON convergé (63 % entre 256 et 512 b) | CHAOS |
| 19 Celtic | home | std_f64 | 0.010 | 4.89 | 818.4 | 28 px (1.09 %) — juge NON convergé (1 % entre 256 et 512 b) | CHAOS |
| 19 Celtic | edge | std_f64 | 0.029 | 3.148 | 3646.4 | 1751 px (68.40 %) — juge NON convergé (51 % entre 256 et 512 b) | CHAOS |
| 20 AlphaMandelbrot | home | std_f64 | 0.011 | 5.558 | 749.7 | 0 px (0.00 %) | PASS |
| 20 AlphaMandelbrot | edge | std_f64 | 0.103 | 13.693 | 2950.4 | 0 px (0.00 %) | PASS |
| 21 PickoverStalks | home | std_f64 | 0.012 | 155.557 | 29.7 | 0 px (0.00 %) | PASS |
| 21 PickoverStalks | edge | std_f64 | 0.055 | 16.59 | 1284.5 | 0 px (0.00 %) | PASS |
| 22 Nova | home | std_f64 | 0.014 | 118.993 | 45.2 | 0 px (0.00 %) | PASS |
| 22 Nova | edge | std_f64 | 0.182 | 17.111 | 4155.4 | 0 px (0.00 %) | PASS |
| 23 Multibrot | home | std_f64 | 0.023 | 11.345 | 781.6 | 6 px (0.23 %) — juge NON convergé (0 % entre 256 et 512 b) | CHAOS |
| 23 Multibrot | edge | std_f64 | 0.206 | 27.564 | 2921.9 | 1848 px (72.19 %) — juge NON convergé (15 % entre 256 et 512 b) | CHAOS |
| 24 Nebulabrot | home | — | 0.031 | 0.001 | 9338865.8 | determinism | PASS |
| 24 Nebulabrot | edge | — | 0.090 | 0.008 | 4486589.6 | determinism | PASS |
| 25 BurningShipJulia | home | std_f64 | 0.009 | 691.279 | 4.9 | 0 px (0.00 %) | PASS |
| 25 BurningShipJulia | edge | std_f64 | 0.009 | 136.897 | 24.8 | 0 px (0.00 %) | PASS |
| 26 TricornJulia | home | std_f64 | 0.008 | 557.241 | 5.8 | 0 px (0.00 %) | PASS |
| 26 TricornJulia | edge | std_f64 | 0.009 | 118.978 | 29.5 | 0 px (0.00 %) | PASS |
| 27 CelticJulia | home | std_f64 | 0.008 | 63.365 | 51.1 | 0 px (0.00 %) | PASS |
| 27 CelticJulia | edge | std_f64 | 0.023 | 3.429 | 2620.4 | 0 px (0.00 %) | PASS |
| 28 BuffaloJulia | home | std_f64 | 0.009 | 856.673 | 4.2 | 0 px (0.00 %) | PASS |
| 28 BuffaloJulia | edge | std_f64 | 0.011 | 180.938 | 23.1 | 0 px (0.00 %) | PASS |
| 29 MultibrotJulia | home | std_f64 | 0.012 | 4.409 | 1089.3 | 0 px (0.00 %) | PASS |
| 29 MultibrotJulia | edge | std_f64 | 0.018 | 2.987 | 2419.0 | 0 px (0.00 %) | PASS |
| 30 PerpendicularBurningShipJulia | home | std_f64 | 0.010 | 336.012 | 12.1 | 0 px (0.00 %) | PASS |
| 30 PerpendicularBurningShipJulia | edge | std_f64 | 0.011 | 32.348 | 134.1 | 0 px (0.00 %) | PASS |
| 31 AlphaMandelbrotJulia | home | std_f64 | 0.011 | 1980.112 | 2.2 | 0 px (0.00 %) | PASS |
| 31 AlphaMandelbrotJulia | edge | std_f64 | 0.010 | 232.784 | 17.4 | 0 px (0.00 %) | PASS |
| 32 MandelbrotSin | home | std_f64 | 0.067 | 21.641 | 1209.9 | 0 px (0.00 %) | PASS |
| 32 MandelbrotSin | edge | std_f64 | 0.698 | 91.41 | 2981.4 | 0 px (0.00 %) | PASS |
| 33 AntiBuddhabrot | home | — | 0.060 | 873.741 | 27.0 | determinism | PASS |
| 33 AntiBuddhabrot | edge | — | 0.340 | 590.935 | 225.0 | determinism | PASS |

## Profil — où va le temps (sans F3, 1 run par cas)

- **répartition cumulée** : orbit 30.3 % · bla 2.7 % · pixels 29.3 % · post 25.9 % · save 0.0 % · other 11.8 % (sur 10 cas, 4 threads)
- **phase dominante** : other ×6, pixels ×3, orbit ×1
- paths : bytecode_exp, bytecode_f64, bytecode_f64_harmonic_lla, std_f64 · pic RSS max 312 MB

| Cas | path | variantes | wall (s) | orbite % | BLA % | pixels % | post % | save % | autre % | ns/iter | ns/iter·thread | iter/px | esc.dd | fiab. % | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `test5` | bytecode_f64 | harmonic_lla | 0.044 | 0.0 | 0.0 | 4.5 | 0.0 | 0.0 | 95.5 | 20.345 | 81.380 | 6 |  | 0.00 | 13 |
| `spiral` | std_f64 | — | 0.038 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.900 | — | 251 |  | — | 13 |
| `flake` | bytecode_f64_harmonic_lla | harmonic_lla | 0.065 | 6.2 | 1.5 | 36.9 | 0.0 | 0.0 | 55.4 | 0.047 | 0.190 | 30921 |  | 0.00 | 13 |
| `glitch_test_2` | bytecode_f64_harmonic_lla | harmonic_lla | 0.060 | 1.7 | 0.0 | 40.2 | 0.0 | 0.0 | 58.1 | 0.188 | 0.752 | 7789 |  | 0.00 | 13 |
| `e50` | bytecode_f64 | harmonic_lla | 0.630 | 4.8 | 1.1 | 71.2 | 16.2 | 0.0 | 6.7 | 0.159 | 0.636 | 172456 |  | 0.62 | 20 |
| `e113` | bytecode_f64 | harmonic_lla | 0.108 | 6.5 | 0.9 | 50.8 | 0.0 | 0.0 | 41.9 | 0.148 | 0.591 | 22729 |  | 0.00 | 13 |
| `e401` | bytecode_exp | bla | 0.131 | 10.0 | 6.1 | 40.6 | 6.1 | 0.0 | 37.2 | 0.117 | 0.469 | 27584 |  | 0.07 | 13 |
| `e1000` | bytecode_exp | bla | 0.090 | 17.8 | 2.2 | 27.9 | 0.0 | 0.0 | 52.0 | 0.092 | 0.367 | 16634 |  | 0.00 | 13 |
| `floral_fantasy` | bytecode_f64_harmonic_lla | harmonic_lla | 0.043 | 2.3 | 0.0 | 21.1 | 0.0 | 0.0 | 76.6 | 0.255 | 1.019 | 2156 |  | 0.00 | 13 |
| `dragon` | bytecode_f64 | harmonic_lla | 2.369 | 42.7 | 3.2 | 17.2 | 34.5 | 0.0 | 2.4 | 0.006 | 0.025 | 3921951 |  | 0.70 | 312 |

## Tendance (même machine, 6 derniers scores)

| Date (UTC) | Commit | Tier | geomean/F3 | wins | fractall Σs | auto-ratio | parité ok/équiv | Q P/W/F | fuzz P/W/F | goldens | px % | gaps |
|---|---|---|---:|---:|---:|---:|---|---|---|---|---:|---:|
| 2026-11-19 12:01 | `b7bdab2` | micro | — | 0 | 3.655 | — | —/— | — | — | — | 28.3 | 9 |
| 2026-11-19 12:31 | `4ffcf4c` | micro | — | 0 | 4.230 | — | —/— | — | — | — | 28.0 | 8 |
| 2026-11-19 12:52 | `f932132` | micro | — | 0 | 3.822 | — | —/— | — | — | — | 28.7 | 6 |
| 2026-11-20 00:19 | `d2f0f54` | micro | — | 0 | 4.929 | — | —/— | — | — | — | 37.8 | 1 |
| 2026-11-20 05:53 | `c25f508` | micro | — | 0 | 2.761 | — | —/— | — | — | — | 26.8 | 0 |
| 2026-11-20 07:44 | `abf534e` | micro | — | 0 | 3.199 | — | —/— | — | — | — | 29.3 | 0 |

_`python3 scripts/harness.py trend` pour l'historique complet (`--case <stem>` pour un cas)._

## Leviers (où gagner — temps mesuré, trié)

_Classement de pistes, pas un budget : les scènes n'ont pas toutes la même taille. Un levier n'est pas un défaut — c'est là que le temps va._

| # | s | Source | Cas | Ce qui coûte | Pour l'ouvrir |
|---:|---:|---|---|---|---|
| 1 | 1.01 | profile | `dragon` | orbite de référence GMP (43 % du cas) | séquentielle — TODO G14.3 (`bench_orbit_join`) |
| 2 | 0.82 | profile | `dragon` | correction de fiabilité (0.7019 % de pixels signalés, 34 % du cas) | `harness.py ab --cases dragon --env FRACTALL_RELIABILITY=off,escalate` — TODO G14.1 |
| 3 | 0.70 | types | `32:MandelbrotSin:edge` | rendu `std_f64` — 91.41 ns/pixel·itération·thread (2981.4 it. moy.) | comparer aux autres types du tableau Types : un ns/px·it hors norme = boucle scalaire à porter (lots, raccourci de cycle) |
| 4 | 0.45 | profile | `e50` | boucle pixel `bytecode_f64` (71 % du cas, 0.6356 ns/iter·thread) | `harness.py hotspots --case e50` |
| 5 | 0.41 | profile | `dragon` | boucle pixel `bytecode_f64` (17 % du cas, 0.0254 ns/iter·thread) | `harness.py hotspots --case dragon` |
| 6 | 0.40 | truth | `std:elephants_1e7_60000` | rendu de production `std_f64` (256×160) | `FRACTALL_PERTURB_STATS=1` : part du traqueur et de la correction |
| 7 | 0.36 | types | `12:MagnetMandelbrot:edge` | rendu `std_f64` — 38.767 ns/pixel·itération·thread (3619.4 it. moy.) | comparer aux autres types du tableau Types : un ns/px·it hors norme = boucle scalaire à porter (lots, raccourci de cycle) |
| 8 | 0.35 | wisdom | `iter:elephants_1e7` | correction de fiabilité du path `std_f64` à N auto = 60000 (43.535 % de pixels signalés, 59 % du rendu) | prix de la correction par pixel — TODO G14.1 |

_… 21 autre(s) dans le history JSON (`harness.py levers`)._

## Gaps (top 10 — sévérité asc, magnitude desc)

_aucun gap détecté 🎉_

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
