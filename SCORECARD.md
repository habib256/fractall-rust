<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-11-19T11:46:04+00:00
- **Commit** : `f16395a`  ⚠️ arbre modifié (dirty)
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
- temps cumulé : **3.595 s** sur 10 cas mesurés

| Cas | fractall (s) | min (s) | bruit | vs baseline | ratio F3 |
|---|---:|---:|---:|---:|---:|
| `test5` | 0.010 | 0.010 | — | — | — |
| `spiral` | 0.011 | 0.011 | — | — | — |
| `flake` | 0.070 | 0.070 | — | — | — |
| `glitch_test_2` | 0.037 | 0.037 | — | — | — |
| `e50` | 0.671 | 0.671 | — | — | — |
| `e113` | 0.116 | 0.116 | — | — | — |
| `e401` | 0.119 | 0.119 | — | — | — |
| `e1000` | 0.066 | 0.066 | — | — | — |
| `floral_fantasy` | 0.019 | 0.019 | — | — | — |
| `dragon` | 2.477 | 2.477 | — | — | — |

## Vérité (rendu de production vs tier dd)

- 24 cas à 48² → **23 PASS · 0 WARN · 1 FAIL** (itérations brutes du rendu par défaut vs frame `--dd-tier`)

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
| `std:elephants_1e7_60000` | std_f64 | 120 | 1.172 | FAIL |

_⚑ = sentinelle (rendu par défaut faux avant les correctifs du 2026-11-06). Un FAIL se juge contre le GMP pur : le dd a son propre plancher._

## Wisdom (décisions automatiques : regret et itérations auto)

- **16 PASS** — 128×80, 2 passes entrelacées (médiane)

**Regret de routage** — `auto` contre chaque plan forcé ; le regret ne compte qu'à sortie équivalente.

| Décision | path auto | auto (s) | meilleure équivalente | regret | plus rapide à sortie ≠ | verdict |
|---|---|---:|---|---:|---|---|
| `harmonic:flake` | bytecode_f64_harmonic_lla | 0.034 | `lla` 0.035 s | 0.971 | — | PASS |
| `harmonic:mitosis` | bytecode_f64_harmonic_lla | 0.036 | `lla` 0.042 s | 0.875 | — | PASS |
| `harmonic:dinosaur_fossils` | bytecode_f64_harmonic_lla | 0.080 | `bla` 0.080 s | 1.0 | — | PASS |
| `harmonic:e50` | bytecode_f64 | 0.445 | `bla` 0.432 s | 1.031 | — | PASS |
| `harmonic:e113` | bytecode_f64 | 0.051 | `bla` 0.052 s | 0.965 | — | PASS |
| `algorithm:seahorse_1e7` | std_f64 | 0.025 | `f64` 0.024 s | 1.032 | — | PASS |
| `algorithm:seahorse_1e10` | std_f64 | 0.029 | `f64` 0.030 s | 0.96 | — | PASS |
| `algorithm:seahorse_1e12` | bytecode_f64 | 0.069 | `perturbation` 0.068 s | 1.011 | — | PASS |
| `algorithm:elephants_1e10` | std_f64 | 0.039 | `f64` 0.040 s | 0.972 | — | PASS |

**Itérations automatiques** — juge : le même rendu à 4N.

| Scène | N auto | sous-itération % | suffisant | surcoût temps | sonde (s) | rendu (s) | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `mandelbrot_home` | 750 | 0.000 | 187 | ×1.317 | 0.01 | 0.01 | PASS |
| `julia_home` | 100 | 0.078 | 100 | — | 0.0 | 0.0 | PASS |
| `seahorse_1e4` | 960 | 0.010 | 960 | — | 0.01 | 0.01 | PASS |
| `seahorse_1e7` | 90000 | 0.215 | 90000 | — | 0.55 | 0.99 | PASS |
| `elephants_1e7` | 60000 | 0.019 | 60000 | — | 1.63 | 3.6 | PASS |
| `barnsley_m_zoom` | 52000 | 0.391 | 52000 | — | 0.23 | 0.66 | PASS |
| `multibrot_zoom` | 8900 | 0.244 | 8900 | — | 0.15 | 0.33 | PASS |

## Types (tous les types — vitesse et juge GMP)

- **9 CHAOS** · **5 FAIL** · **10 N/A** · **36 PASS** · **4 WARN** — vitesse à 128×80, juge GMP à 64×40 ; temps cumulé **4.77 s**

| Type | Scène | path | s | ns/px·it | it. moy. | juge | verdict |
|---|---|---|---:|---:|---:|---|---|
| 1 VonKoch | home | — | 0.012 | 7642.382 | 0.6 | determinism | PASS |
| 2 Dragon | home | — | 0.037 | 1827.434 | 7.9 | determinism | PASS |
| 3 Mandelbrot | home | std_f64 | 0.011 | 7.002 | 632.9 | 0 px (0.00 %) | PASS |
| 3 Mandelbrot | edge | std_f64 | 0.142 | 19.886 | 2799.4 | 0 px (0.00 %) | PASS |
| 4 Julia | home | std_f64 | 0.010 | 445.004 | 8.4 | 0 px (0.00 %) | PASS |
| 4 Julia | edge | std_f64 | 0.009 | 30.418 | 117.5 | 0 px (0.00 %) | PASS |
| 5 JuliaSin | home | std_f64 | 0.018 | 120.816 | 58.5 | 0 px (0.00 %) | PASS |
| 5 JuliaSin | edge | std_f64 | 0.196 | 50.074 | 1526.9 | gmp:timeout | N/A |
| 6 Newton | home | std_f64 | 0.192 | 18.059 | 4145.5 | 0 px (0.00 %) | PASS |
| 6 Newton | edge | std_f64 | 0.878 | 88.454 | 3876.2 | 0 px (0.00 %) | PASS |
| 7 Phoenix | home | std_f64 | 0.011 | 27.263 | 160.4 | 0 px (0.00 %) | PASS |
| 7 Phoenix | edge | std_f64 | 0.085 | 7.091 | 4660.6 | 2 px (0.08 %) — juge convergé | WARN |
| 8 Buffalo | home | std_f64 | 0.011 | 5.653 | 747.9 | 53 px (2.07 %) — juge NON convergé (1 % entre 256 et 512 b) | CHAOS |
| 8 Buffalo | edge | std_f64 | 0.026 | 3.331 | 3091.4 | 2194 px (85.70 %) — juge NON convergé (81 % entre 256 et 512 b) | CHAOS |
| 9 BarnsleyJulia | home | std_f64 | 0.009 | 138.045 | 25.3 | 27 px (1.05 %) — juge convergé | FAIL |
| 9 BarnsleyJulia | edge | std_f64 | 0.010 | 53.503 | 74.2 | 129 px (5.04 %) — juge convergé | FAIL |
| 10 BarnsleyMandelbrot | home | std_f64 | 0.015 | 5.996 | 978.4 | 12 px (0.47 %) — juge NON convergé (0 % entre 256 et 512 b) | CHAOS |
| 10 BarnsleyMandelbrot | edge | std_f64 | 0.069 | 8.239 | 3249.4 | 2141 px (83.63 %) — juge NON convergé (73 % entre 256 et 512 b) | CHAOS |
| 11 MagnetJulia | home | std_f64 | 0.008 | 1.11 | 2962.9 | 1 px (0.04 %) | PASS |
| 11 MagnetJulia | edge | std_f64 | 0.012 | 2.274 | 2103.7 | 8 px (0.31 %) — juge convergé | WARN |
| 12 MagnetMandelbrot | home | std_f64 | 0.015 | 1.623 | 3613.3 | gmp:timeout | N/A |
| 12 MagnetMandelbrot | edge | std_f64 | 0.139 | 15.044 | 3620.0 | gmp:timeout | N/A |
| 13 BurningShip | home | std_f64 | 0.013 | 9.502 | 531.3 | 36 px (1.41 %) — juge NON convergé (0 % entre 256 et 512 b) | CHAOS |
| 13 BurningShip | edge | std_f64 | 0.031 | 3.505 | 3438.2 | 1947 px (76.05 %) — juge NON convergé (67 % entre 256 et 512 b) | CHAOS |
| 14 Tricorn | home | std_f64 | 0.010 | 12.471 | 312.3 | 0 px (0.00 %) | PASS |
| 14 Tricorn | edge | std_f64 | 0.026 | 3.172 | 3248.7 | 65 px (2.54 %) — juge convergé | FAIL |
| 15 Mandelbulb | home | std_f64 | 0.009 | 2.829 | 1316.8 | 0 px (0.00 %) | PASS |
| 15 Mandelbulb | edge | std_f64 | 0.074 | 11.183 | 2601.7 | 101 px (3.95 %) — juge convergé | FAIL |
| 16 Buddhabrot | home | — | 0.015 | 60.209 | 97.4 | determinism | PASS |
| 16 Buddhabrot | edge | — | 0.069 | 391.511 | 68.4 | determinism | PASS |
| 17 Lyapunov | home | — | 0.048 | 26.38 | 711.8 | determinism | PASS |
| 17 Lyapunov | edge | — | 0.046 | 18.423 | 970.7 | determinism | PASS |
| 18 PerpendicularBurningShip | home | std_f64 | 0.009 | 6.938 | 496.0 | 2 px (0.08 %) — juge convergé | WARN |
| 18 PerpendicularBurningShip | edge | std_f64 | 0.027 | 3.151 | 3316.1 | 2072 px (80.94 %) — juge NON convergé (63 % entre 256 et 512 b) | CHAOS |
| 19 Celtic | home | std_f64 | 0.010 | 4.829 | 818.4 | 28 px (1.09 %) — juge NON convergé (1 % entre 256 et 512 b) | CHAOS |
| 19 Celtic | edge | std_f64 | 0.026 | 2.777 | 3646.4 | 1751 px (68.40 %) — juge NON convergé (51 % entre 256 et 512 b) | CHAOS |
| 20 AlphaMandelbrot | home | std_f64 | 0.009 | 4.954 | 749.7 | 0 px (0.00 %) | PASS |
| 20 AlphaMandelbrot | edge | std_f64 | 0.075 | 9.993 | 2949.3 | 116 px (4.53 %) — juge convergé | FAIL |
| 21 PickoverStalks | home | std_f64 | 0.011 | 149.839 | 29.7 | 0 px (0.00 %) | PASS |
| 21 PickoverStalks | edge | std_f64 | 0.071 | 21.465 | 1284.5 | 0 px (0.00 %) | PASS |
| 22 Nova | home | std_f64 | 0.012 | 105.865 | 45.2 | 0 px (0.00 %) | PASS |
| 22 Nova | edge | std_f64 | 0.230 | 21.614 | 4155.4 | gmp:timeout | N/A |
| 23 Multibrot | home | std_f64 | 0.027 | 13.436 | 782.2 | gmp:timeout | N/A |
| 23 Multibrot | edge | std_f64 | 0.637 | 83.902 | 2963.7 | gmp:timeout | N/A |
| 24 Nebulabrot | home | — | 0.035 | 0.001 | 9338865.8 | determinism | PASS |
| 24 Nebulabrot | edge | — | 0.164 | 0.014 | 4486589.6 | determinism | PASS |
| 25 BurningShipJulia | home | std_f64 | 0.008 | 630.889 | 4.9 | 0 px (0.00 %) | PASS |
| 25 BurningShipJulia | edge | std_f64 | 0.008 | 132.97 | 24.8 | 0 px (0.00 %) | PASS |
| 26 TricornJulia | home | std_f64 | 0.008 | 549.943 | 5.8 | 0 px (0.00 %) | PASS |
| 26 TricornJulia | edge | std_f64 | 0.008 | 108.538 | 29.5 | 0 px (0.00 %) | PASS |
| 27 CelticJulia | home | std_f64 | 0.008 | 62.945 | 51.1 | 0 px (0.00 %) | PASS |
| 27 CelticJulia | edge | std_f64 | 0.022 | 3.231 | 2620.4 | 0 px (0.00 %) | PASS |
| 28 BuffaloJulia | home | std_f64 | 0.010 | 945.79 | 4.2 | 0 px (0.00 %) | PASS |
| 28 BuffaloJulia | edge | std_f64 | 0.010 | 169.906 | 23.1 | 0 px (0.00 %) | PASS |
| 29 MultibrotJulia | home | std_f64 | 0.021 | 7.452 | 1089.3 | gmp:timeout | N/A |
| 29 MultibrotJulia | edge | std_f64 | 0.034 | 5.472 | 2419.0 | gmp:timeout | N/A |
| 30 PerpendicularBurningShipJulia | home | std_f64 | 0.009 | 274.439 | 12.1 | 0 px (0.00 %) | PASS |
| 30 PerpendicularBurningShipJulia | edge | std_f64 | 0.011 | 32.771 | 134.1 | 9 px (0.35 %) — juge convergé | WARN |
| 31 AlphaMandelbrotJulia | home | std_f64 | 0.009 | 1657.858 | 2.2 | 0 px (0.00 %) | PASS |
| 31 AlphaMandelbrotJulia | edge | std_f64 | 0.011 | 238.144 | 17.4 | 0 px (0.00 %) | PASS |
| 32 MandelbrotSin | home | std_f64 | 0.085 | 27.308 | 1209.9 | gmp:timeout | N/A |
| 32 MandelbrotSin | edge | std_f64 | 0.409 | 53.503 | 2985.7 | gmp:timeout | N/A |
| 33 AntiBuddhabrot | home | — | 0.062 | 896.046 | 27.0 | determinism | PASS |
| 33 AntiBuddhabrot | edge | — | 0.439 | 761.997 | 225.0 | determinism | PASS |

## Profil — où va le temps (sans F3, 1 run par cas)

- **répartition cumulée** : orbit 35.4 % · bla 2.2 % · pixels 27.7 % · post 25.6 % · save 0.0 % · other 9.2 % (sur 10 cas, 4 threads)
- **phase dominante** : other ×6, pixels ×3, orbit ×1
- paths : bytecode_exp, bytecode_f64, bytecode_f64_harmonic_lla, std_f64 · pic RSS max 312 MB

| Cas | path | variantes | wall (s) | orbite % | BLA % | pixels % | post % | save % | autre % | ns/iter | ns/iter·thread | iter/px | esc.dd | fiab. % | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `test5` | bytecode_f64 | harmonic_lla | 0.035 | 2.9 | 0.0 | 5.7 | 0.0 | 0.0 | 91.4 | 20.345 | 81.380 | 6 |  | 0.00 | 13 |
| `spiral` | std_f64 | — | 0.047 | 0.0 | 0.0 | 21.2 | 0.0 | 0.0 | 78.8 | 1.000 | — | 251 |  | — | 13 |
| `flake` | bytecode_f64_harmonic_lla | harmonic_lla | 0.066 | 6.1 | 1.5 | 41.1 | 0.0 | 0.0 | 51.3 | 0.053 | 0.213 | 30921 |  | 0.00 | 13 |
| `glitch_test_2` | bytecode_f64_harmonic_lla | harmonic_lla | 0.059 | 1.7 | 0.0 | 37.1 | 0.0 | 0.0 | 61.3 | 0.172 | 0.690 | 7789 |  | 0.00 | 13 |
| `e50` | bytecode_f64 | harmonic_lla | 0.629 | 5.2 | 1.1 | 70.9 | 17.5 | 0.0 | 5.2 | 0.158 | 0.631 | 172456 |  | 0.62 | 20 |
| `e113` | bytecode_f64 | harmonic_lla | 0.102 | 4.9 | 1.0 | 54.8 | 0.0 | 0.0 | 39.3 | 0.150 | 0.602 | 22729 |  | 0.00 | 12 |
| `e401` | bytecode_exp | bla | 0.138 | 26.9 | 5.1 | 37.0 | 5.8 | 0.0 | 25.2 | 0.113 | 0.451 | 27584 |  | 0.07 | 43 |
| `e1000` | bytecode_exp | bla | 0.097 | 28.0 | 2.1 | 33.1 | 0.0 | 0.0 | 36.8 | 0.117 | 0.470 | 16634 |  | 0.00 | 17 |
| `floral_fantasy` | bytecode_f64_harmonic_lla | harmonic_lla | 0.048 | 2.1 | 0.0 | 25.3 | 0.0 | 0.0 | 72.6 | 0.340 | 1.359 | 2156 |  | 0.00 | 13 |
| `dragon` | bytecode_f64 | harmonic_lla | 2.724 | 47.2 | 2.5 | 16.0 | 32.7 | 0.0 | 1.6 | 0.007 | 0.027 | 3921951 |  | 0.70 | 312 |

## Tendance (même machine, 6 derniers scores)

| Date (UTC) | Commit | Tier | geomean/F3 | wins | fractall Σs | auto-ratio | parité ok/équiv | Q P/W/F | fuzz P/W/F | goldens | px % | gaps |
|---|---|---|---:|---:|---:|---:|---|---|---|---|---:|---:|
| 2026-11-04 08:07 | `fbbbb52` | quick | — | — | — | — | —/— | 16/0/0 | 3/0/0 | 🟢 | 57.8 | 0 |
| 2026-11-05 13:32 | `ed33ef3` | quick | — | — | — | — | —/— | 16/0/0 | 3/0/0 | — | 60.5 | 0 |
| 2026-11-05 13:49 | `3cbed34` | quick | — | — | — | — | —/— | — | — | — | — | 0 |
| 2026-11-19 11:16 | `c35f0e7` | micro | — | 0 | 4.458 | — | —/— | — | — | — | 27.2 | 11 |
| 2026-11-19 11:46 | `f16395a` | micro | — | 0 | 3.595 | — | —/— | — | — | — | 27.7 | 10 |

_`python3 scripts/harness.py trend` pour l'historique complet (`--case <stem>` pour un cas)._

## Leviers (où gagner — temps mesuré, trié)

_Classement de pistes, pas un budget : les scènes n'ont pas toutes la même taille. Un levier n'est pas un défaut — c'est là que le temps va._

| # | s | Source | Cas | Ce qui coûte | Pour l'ouvrir |
|---:|---:|---|---|---|---|
| 1 | 1.96 | truth | `std:elephants_1e7_60000` | rendu de production `std_f64` (256×160) | `FRACTALL_PERTURB_STATS=1` : part du traqueur et de la correction |
| 2 | 1.95 | wisdom | `iter:elephants_1e7` | correction de fiabilité du path `std_f64` à N auto = 60000 (43.535 % de pixels signalés, 54 % du rendu) | prix de la correction par pixel — TODO G14.1 |
| 3 | 1.63 | wisdom | `iter:elephants_1e7` | sonde d'itérations automatiques (45 % du rendu) | `fractal/wisdom_iter.rs` (tours, taille de sonde) |
| 4 | 1.29 | profile | `dragon` | orbite de référence GMP (47 % du cas) | séquentielle — TODO G14.3 (`bench_orbit_join`) |
| 5 | 0.89 | profile | `dragon` | correction de fiabilité (0.7019 % de pixels signalés, 33 % du cas) | `harness.py ab --cases dragon --env FRACTALL_RELIABILITY=off,escalate` — TODO G14.1 |
| 6 | 0.88 | types | `6:Newton:edge` | rendu `std_f64` — 88.454 ns/pixel·itération·thread (3876.2 it. moy.) | comparer aux autres types du tableau Types : un ns/px·it hors norme = boucle scalaire à porter (lots, raccourci de cycle) |
| 7 | 0.64 | types | `23:Multibrot:edge` | rendu `std_f64` — 83.902 ns/pixel·itération·thread (2963.7 it. moy.) | comparer aux autres types du tableau Types : un ns/px·it hors norme = boucle scalaire à porter (lots, raccourci de cycle) |
| 8 | 0.55 | wisdom | `iter:seahorse_1e7` | sonde d'itérations automatiques (56 % du rendu) | `fractal/wisdom_iter.rs` (tours, taille de sonde) |

_… 24 autre(s) dans le history JSON (`harness.py levers`)._

## Gaps (top 10 — sévérité asc, magnitude desc)

| # | Sévérité | Axe | Cas | Métrique | Valeur | Note |
|---:|---|---|---|---|---:|---|
| 1 | 1 correction | truth | `std:elephants_1e7_60000` | diff_px | 120 | pixels du rendu par défaut ≠ GMP pur (1.17 %, path std_f64) |
| 2 | 1 correction | types | `9:BarnsleyJulia:edge` | frac | 0.050391 | 129 px ≠ GMP pur (5.04 %, path std_f64) — juge convergé (256 b ≡ 512 b) |
| 3 | 1 correction | types | `20:AlphaMandelbrot:edge` | frac | 0.045312 | 116 px ≠ GMP pur (4.53 %, path std_f64) — juge convergé (256 b ≡ 512 b) |
| 4 | 1 correction | types | `15:Mandelbulb:edge` | frac | 0.039453 | 101 px ≠ GMP pur (3.95 %, path std_f64) — juge convergé (256 b ≡ 512 b) |
| 5 | 1 correction | types | `14:Tricorn:edge` | frac | 0.025391 | 65 px ≠ GMP pur (2.54 %, path std_f64) — juge convergé (256 b ≡ 512 b) |
| 6 | 1 correction | types | `9:BarnsleyJulia:home` | frac | 0.010547 | 27 px ≠ GMP pur (1.05 %, path std_f64) — juge convergé (256 b ≡ 512 b) |
| 7 | 4 qualité | types | `30:PerpendicularBurningShipJulia:edge` | frac | 0.003516 | 9 px ≠ GMP pur (0.35 %, path std_f64) — juge convergé (256 b ≡ 512 b) |
| 8 | 4 qualité | types | `11:MagnetJulia:edge` | frac | 0.003125 | 8 px ≠ GMP pur (0.31 %, path std_f64) — juge convergé (256 b ≡ 512 b) |
| 9 | 4 qualité | types | `7:Phoenix:edge` | frac | 0.000781 | 2 px ≠ GMP pur (0.08 %, path std_f64) — juge convergé (256 b ≡ 512 b) |
| 10 | 4 qualité | types | `18:PerpendicularBurningShip:home` | frac | 0.000781 | 2 px ≠ GMP pur (0.08 %, path std_f64) — juge convergé (256 b ≡ 512 b) |

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
