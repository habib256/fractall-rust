<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-11-20T05:53:10+00:00
- **Commit** : `c25f508`
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
- temps cumulé : **2.761 s** sur 10 cas mesurés

| Cas | fractall (s) | min (s) | bruit | vs baseline | ratio F3 |
|---|---:|---:|---:|---:|---:|
| `test5` | 0.011 | 0.011 | — | — | — |
| `spiral` | 0.009 | 0.009 | — | — | — |
| `flake` | 0.033 | 0.033 | — | — | — |
| `glitch_test_2` | 0.033 | 0.033 | — | — | — |
| `e50` | 0.472 | 0.472 | — | — | — |
| `e113` | 0.069 | 0.069 | — | — | — |
| `e401` | 0.069 | 0.069 | — | — | — |
| `e1000` | 0.066 | 0.066 | — | — | — |
| `floral_fantasy` | 0.020 | 0.020 | — | — | — |
| `dragon` | 1.978 | 1.978 | — | — | — |

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
| `harmonic:flake` | bytecode_f64_harmonic_lla | 0.031 | `lla` 0.034 s | 0.929 | — | PASS |
| `harmonic:mitosis` | bytecode_f64_harmonic_lla | 0.029 | `lla` 0.027 s | 1.11 | — | PASS |
| `harmonic:dinosaur_fossils` | bytecode_f64_harmonic_lla | 0.060 | `bla` 0.060 s | 0.989 | — | PASS |
| `harmonic:e50` | bytecode_f64 | 0.343 | `bla` 0.341 s | 1.007 | — | PASS |
| `harmonic:e113` | bytecode_f64 | 0.046 | `bla` 0.043 s | 1.077 | — | PASS |
| `algorithm:seahorse_1e7` | std_f64 | 0.018 | `f64` 0.021 s | 0.88 | — | PASS |
| `algorithm:seahorse_1e10` | std_f64 | 0.022 | `f64` 0.021 s | 1.038 | — | PASS |
| `algorithm:seahorse_1e12` | bytecode_f64 | 0.053 | `perturbation` 0.055 s | 0.967 | — | PASS |
| `algorithm:elephants_1e10` | std_f64 | 0.040 | `f64` 0.030 s | 1.343 | — | PASS |

**Itérations automatiques** — juge : le même rendu à 4N.

| Scène | N auto | sous-itération % | suffisant | surcoût temps | sonde (s) | rendu (s) | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `mandelbrot_home` | 750 | 0.000 | 187 | ×0.986 | 0.0 | 0.01 | PASS |
| `julia_home` | 100 | 0.078 | 100 | — | 0.0 | 0.0 | PASS |
| `seahorse_1e4` | 960 | 0.010 | 960 | — | 0.0 | 0.0 | PASS |
| `seahorse_1e7` | 90000 | 0.215 | 90000 | — | 0.13 | 0.28 | PASS |
| `elephants_1e7` | 60000 | 0.019 | 60000 | — | 0.19 | 0.54 | PASS |
| `barnsley_m_zoom` | 60000 | 0.361 | 60000 | — | 0.21 | 0.58 | PASS |
| `multibrot_zoom` | 25000 | 0.185 | 25000 | — | 0.09 | 0.17 | PASS |

## Types (tous les types — vitesse et juge GMP)

- **11 CHAOS** · **53 PASS** — vitesse à 128×80, juge GMP à 64×40 ; temps cumulé **3.116 s**

| Type | Scène | path | s | ns/px·it | it. moy. | juge | verdict |
|---|---|---|---:|---:|---:|---|---|
| 1 VonKoch | home | — | 0.012 | 7862.573 | 0.6 | determinism | PASS |
| 2 Dragon | home | — | 0.039 | 1940.6 | 7.9 | determinism | PASS |
| 3 Mandelbrot | home | std_f64 | 0.011 | 6.561 | 632.9 | 0 px (0.00 %) | PASS |
| 3 Mandelbrot | edge | std_f64 | 0.038 | 5.262 | 2799.4 | 0 px (0.00 %) | PASS |
| 4 Julia | home | std_f64 | 0.018 | 850.297 | 8.4 | 0 px (0.00 %) | PASS |
| 4 Julia | edge | std_f64 | 0.009 | 28.385 | 117.5 | 0 px (0.00 %) | PASS |
| 5 JuliaSin | home | std_f64 | 0.013 | 90.104 | 58.5 | 0 px (0.00 %) | PASS |
| 5 JuliaSin | edge | std_f64 | 0.144 | 36.84 | 1526.9 | 0 px (0.00 %) | PASS |
| 6 Newton | home | std_f64 | 0.008 | 0.79 | 4145.5 | 0 px (0.00 %) | PASS |
| 6 Newton | edge | std_f64 | 0.009 | 0.951 | 3876.2 | 0 px (0.00 %) | PASS |
| 7 Phoenix | home | std_f64 | 0.011 | 27.655 | 160.4 | 0 px (0.00 %) | PASS |
| 7 Phoenix | edge | std_f64 | 0.086 | 7.173 | 4660.7 | 0 px (0.00 %) | PASS |
| 8 Buffalo | home | std_f64 | 0.011 | 5.622 | 747.9 | 53 px (2.07 %) — juge NON convergé (1 % entre 256 et 512 b) | CHAOS |
| 8 Buffalo | edge | std_f64 | 0.024 | 2.989 | 3091.4 | 2194 px (85.70 %) — juge NON convergé (81 % entre 256 et 512 b) | CHAOS |
| 9 BarnsleyJulia | home | std_f64 | 0.010 | 159.067 | 25.2 | 0 px (0.00 %) | PASS |
| 9 BarnsleyJulia | edge | std_f64 | 0.013 | 67.857 | 74.2 | 0 px (0.00 %) | PASS |
| 10 BarnsleyMandelbrot | home | std_f64 | 0.020 | 8.064 | 977.7 | 8 px (0.31 %) — juge NON convergé (0 % entre 256 et 512 b) | CHAOS |
| 10 BarnsleyMandelbrot | edge | std_f64 | 0.064 | 7.797 | 3201.6 | 2099 px (81.99 %) — juge NON convergé (73 % entre 256 et 512 b) | CHAOS |
| 11 MagnetJulia | home | std_f64 | 0.010 | 1.301 | 2961.0 | 0 px (0.00 %) | PASS |
| 11 MagnetJulia | edge | std_f64 | 0.014 | 2.568 | 2099.4 | 0 px (0.00 %) | PASS |
| 12 MagnetMandelbrot | home | std_f64 | 0.020 | 2.197 | 3612.4 | 0 px (0.00 %) | PASS |
| 12 MagnetMandelbrot | edge | std_f64 | 0.308 | 33.225 | 3619.4 | 0 px (0.00 %) | PASS |
| 13 BurningShip | home | std_f64 | 0.010 | 7.059 | 531.3 | 36 px (1.41 %) — juge NON convergé (0 % entre 256 et 512 b) | CHAOS |
| 13 BurningShip | edge | std_f64 | 0.028 | 3.144 | 3438.2 | 1947 px (76.05 %) — juge NON convergé (67 % entre 256 et 512 b) | CHAOS |
| 14 Tricorn | home | std_f64 | 0.012 | 14.81 | 312.2 | 0 px (0.00 %) | PASS |
| 14 Tricorn | edge | std_f64 | 0.027 | 3.271 | 3248.5 | 0 px (0.00 %) | PASS |
| 15 Mandelbulb | home | std_f64 | 0.010 | 2.895 | 1316.8 | 0 px (0.00 %) | PASS |
| 15 Mandelbulb | edge | std_f64 | 0.099 | 14.915 | 2602.0 | 0 px (0.00 %) | PASS |
| 16 Buddhabrot | home | — | 0.014 | 57.268 | 97.4 | determinism | PASS |
| 16 Buddhabrot | edge | — | 0.052 | 298.653 | 68.4 | determinism | PASS |
| 17 Lyapunov | home | — | 0.038 | 20.779 | 711.8 | determinism | PASS |
| 17 Lyapunov | edge | — | 0.039 | 15.604 | 970.7 | determinism | PASS |
| 18 PerpendicularBurningShip | home | std_f64 | 0.015 | 11.661 | 496.4 | 0 px (0.00 %) | PASS |
| 18 PerpendicularBurningShip | edge | std_f64 | 0.021 | 2.504 | 3316.1 | 2072 px (80.94 %) — juge NON convergé (63 % entre 256 et 512 b) | CHAOS |
| 19 Celtic | home | std_f64 | 0.009 | 4.378 | 818.4 | 28 px (1.09 %) — juge NON convergé (1 % entre 256 et 512 b) | CHAOS |
| 19 Celtic | edge | std_f64 | 0.022 | 2.406 | 3646.4 | 1751 px (68.40 %) — juge NON convergé (51 % entre 256 et 512 b) | CHAOS |
| 20 AlphaMandelbrot | home | std_f64 | 0.011 | 5.479 | 749.7 | 0 px (0.00 %) | PASS |
| 20 AlphaMandelbrot | edge | std_f64 | 0.115 | 15.231 | 2950.4 | 0 px (0.00 %) | PASS |
| 21 PickoverStalks | home | std_f64 | 0.010 | 133.657 | 29.7 | 0 px (0.00 %) | PASS |
| 21 PickoverStalks | edge | std_f64 | 0.051 | 15.591 | 1284.5 | 0 px (0.00 %) | PASS |
| 22 Nova | home | std_f64 | 0.010 | 88.061 | 45.2 | 0 px (0.00 %) | PASS |
| 22 Nova | edge | std_f64 | 0.172 | 16.2 | 4155.4 | 0 px (0.00 %) | PASS |
| 23 Multibrot | home | std_f64 | 0.016 | 8.167 | 781.6 | 6 px (0.23 %) — juge NON convergé (0 % entre 256 et 512 b) | CHAOS |
| 23 Multibrot | edge | std_f64 | 0.189 | 25.309 | 2921.9 | 1848 px (72.19 %) — juge NON convergé (15 % entre 256 et 512 b) | CHAOS |
| 24 Nebulabrot | home | — | 0.032 | 0.001 | 9338865.8 | determinism | PASS |
| 24 Nebulabrot | edge | — | 0.095 | 0.008 | 4486589.6 | determinism | PASS |
| 25 BurningShipJulia | home | std_f64 | 0.007 | 583.517 | 4.9 | 0 px (0.00 %) | PASS |
| 25 BurningShipJulia | edge | std_f64 | 0.007 | 106.796 | 24.8 | 0 px (0.00 %) | PASS |
| 26 TricornJulia | home | std_f64 | 0.007 | 489.747 | 5.8 | 0 px (0.00 %) | PASS |
| 26 TricornJulia | edge | std_f64 | 0.008 | 105.554 | 29.5 | 0 px (0.00 %) | PASS |
| 27 CelticJulia | home | std_f64 | 0.008 | 58.399 | 51.1 | 0 px (0.00 %) | PASS |
| 27 CelticJulia | edge | std_f64 | 0.020 | 3.007 | 2620.4 | 0 px (0.00 %) | PASS |
| 28 BuffaloJulia | home | std_f64 | 0.009 | 822.446 | 4.2 | 0 px (0.00 %) | PASS |
| 28 BuffaloJulia | edge | std_f64 | 0.009 | 143.164 | 23.1 | 0 px (0.00 %) | PASS |
| 29 MultibrotJulia | home | std_f64 | 0.011 | 4.052 | 1089.3 | 0 px (0.00 %) | PASS |
| 29 MultibrotJulia | edge | std_f64 | 0.018 | 2.861 | 2419.0 | 0 px (0.00 %) | PASS |
| 30 PerpendicularBurningShipJulia | home | std_f64 | 0.008 | 267.476 | 12.1 | 0 px (0.00 %) | PASS |
| 30 PerpendicularBurningShipJulia | edge | std_f64 | 0.009 | 24.788 | 134.1 | 0 px (0.00 %) | PASS |
| 31 AlphaMandelbrotJulia | home | std_f64 | 0.008 | 1455.893 | 2.2 | 0 px (0.00 %) | PASS |
| 31 AlphaMandelbrotJulia | edge | std_f64 | 0.007 | 167.8 | 17.4 | 0 px (0.00 %) | PASS |
| 32 MandelbrotSin | home | std_f64 | 0.063 | 20.201 | 1209.9 | 0 px (0.00 %) | PASS |
| 32 MandelbrotSin | edge | std_f64 | 0.537 | 70.429 | 2981.4 | 0 px (0.00 %) | PASS |
| 33 AntiBuddhabrot | home | — | 0.054 | 787.944 | 27.0 | determinism | PASS |
| 33 AntiBuddhabrot | edge | — | 0.335 | 581.749 | 225.0 | determinism | PASS |

## Profil — où va le temps (sans F3, 1 run par cas)

- **répartition cumulée** : orbit 33.3 % · bla 2.6 % · pixels 26.8 % · post 26.2 % · save 0.0 % · other 11.1 % (sur 10 cas, 4 threads)
- **phase dominante** : other ×7, pixels ×2, orbit ×1
- paths : bytecode_exp, bytecode_f64, bytecode_f64_harmonic_lla, std_f64 · pic RSS max 312 MB

| Cas | path | variantes | wall (s) | orbite % | BLA % | pixels % | post % | save % | autre % | ns/iter | ns/iter·thread | iter/px | esc.dd | fiab. % | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `test5` | bytecode_f64 | harmonic_lla | 0.034 | 0.0 | 0.0 | 3.0 | 0.0 | 0.0 | 97.0 | 10.172 | 40.690 | 6 |  | 0.00 | 13 |
| `spiral` | std_f64 | — | 0.030 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.600 | — | 251 |  | — | 13 |
| `flake` | bytecode_f64_harmonic_lla | harmonic_lla | 0.054 | 5.6 | 1.9 | 37.2 | 0.0 | 0.0 | 55.4 | 0.040 | 0.158 | 30921 |  | 0.00 | 13 |
| `glitch_test_2` | bytecode_f64_harmonic_lla | harmonic_lla | 0.043 | 2.3 | 0.0 | 32.3 | 0.0 | 0.0 | 65.4 | 0.110 | 0.439 | 7789 |  | 0.00 | 13 |
| `e50` | bytecode_f64 | harmonic_lla | 0.492 | 4.9 | 1.2 | 68.1 | 19.7 | 0.0 | 6.1 | 0.119 | 0.474 | 172456 |  | 0.62 | 21 |
| `e113` | bytecode_f64 | harmonic_lla | 0.077 | 5.2 | 1.3 | 51.9 | 0.0 | 0.0 | 41.6 | 0.107 | 0.430 | 22729 |  | 0.00 | 13 |
| `e401` | bytecode_exp | bla | 0.081 | 12.3 | 6.1 | 36.9 | 6.1 | 0.0 | 38.5 | 0.066 | 0.266 | 27584 |  | 0.07 | 13 |
| `e1000` | bytecode_exp | bla | 0.077 | 20.8 | 3.9 | 28.6 | 0.0 | 0.0 | 46.7 | 0.081 | 0.323 | 16634 |  | 0.00 | 13 |
| `floral_fantasy` | bytecode_f64_harmonic_lla | harmonic_lla | 0.040 | 2.5 | 0.0 | 19.9 | 0.0 | 0.0 | 77.6 | 0.227 | 0.906 | 2156 |  | 0.00 | 13 |
| `dragon` | bytecode_f64 | harmonic_lla | 1.972 | 46.0 | 3.0 | 15.5 | 33.3 | 0.0 | 2.1 | 0.005 | 0.019 | 3921951 |  | 0.70 | 312 |

## Tendance (même machine, 6 derniers scores)

| Date (UTC) | Commit | Tier | geomean/F3 | wins | fractall Σs | auto-ratio | parité ok/équiv | Q P/W/F | fuzz P/W/F | goldens | px % | gaps |
|---|---|---|---:|---:|---:|---:|---|---|---|---|---:|---:|
| 2026-11-19 11:46 | `f16395a` | micro | — | 0 | 3.595 | — | —/— | — | — | — | 27.7 | 10 |
| 2026-11-19 12:01 | `b7bdab2` | micro | — | 0 | 3.655 | — | —/— | — | — | — | 28.3 | 9 |
| 2026-11-19 12:31 | `4ffcf4c` | micro | — | 0 | 4.230 | — | —/— | — | — | — | 28.0 | 8 |
| 2026-11-19 12:52 | `f932132` | micro | — | 0 | 3.822 | — | —/— | — | — | — | 28.7 | 6 |
| 2026-11-20 00:19 | `d2f0f54` | micro | — | 0 | 4.929 | — | —/— | — | — | — | 37.8 | 1 |
| 2026-11-20 05:53 | `c25f508` | micro | — | 0 | 2.761 | — | —/— | — | — | — | 26.8 | 0 |

_`python3 scripts/harness.py trend` pour l'historique complet (`--case <stem>` pour un cas)._

## Leviers (où gagner — temps mesuré, trié)

_Classement de pistes, pas un budget : les scènes n'ont pas toutes la même taille. Un levier n'est pas un défaut — c'est là que le temps va._

| # | s | Source | Cas | Ce qui coûte | Pour l'ouvrir |
|---:|---:|---|---|---|---|
| 1 | 0.91 | profile | `dragon` | orbite de référence GMP (46 % du cas) | séquentielle — TODO G14.3 (`bench_orbit_join`) |
| 2 | 0.66 | profile | `dragon` | correction de fiabilité (0.7019 % de pixels signalés, 33 % du cas) | `harness.py ab --cases dragon --env FRACTALL_RELIABILITY=off,escalate` — TODO G14.1 |
| 3 | 0.54 | types | `32:MandelbrotSin:edge` | rendu `std_f64` — 70.429 ns/pixel·itération·thread (2981.4 it. moy.) | comparer aux autres types du tableau Types : un ns/px·it hors norme = boucle scalaire à porter (lots, raccourci de cycle) |
| 4 | 0.42 | truth | `std:elephants_1e7_60000` | rendu de production `std_f64` (256×160) | `FRACTALL_PERTURB_STATS=1` : part du traqueur et de la correction |
| 5 | 0.34 | profile | `e50` | boucle pixel `bytecode_f64` (68 % du cas, 0.4742 ns/iter·thread) | `harness.py hotspots --case e50` |
| 6 | 0.34 | types | `33:AntiBuddhabrot:edge` | rendu `None` — 581.749 ns/pixel·itération·thread (225.0 it. moy.) | comparer aux autres types du tableau Types : un ns/px·it hors norme = boucle scalaire à porter (lots, raccourci de cycle) |
| 7 | 0.33 | wisdom | `iter:elephants_1e7` | correction de fiabilité du path `std_f64` à N auto = 60000 (43.535 % de pixels signalés, 60 % du rendu) | prix de la correction par pixel — TODO G14.1 |
| 8 | 0.31 | types | `12:MagnetMandelbrot:edge` | rendu `std_f64` — 33.225 ns/pixel·itération·thread (3619.4 it. moy.) | comparer aux autres types du tableau Types : un ns/px·it hors norme = boucle scalaire à porter (lots, raccourci de cycle) |

_… 19 autre(s) dans le history JSON (`harness.py levers`)._

## Gaps (top 10 — sévérité asc, magnitude desc)

_aucun gap détecté 🎉_

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
