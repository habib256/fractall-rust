<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-11-18T02:49:10+00:00
- **Commit** : `372236f`
- **Machine** : Intel(R) Core(TM) i7-10700F CPU @ 2.90GHz · 16 threads · Linux 7.0.0-43-generic
- **Tier** : quick · 256×256 · quality 96×96 · runs=1 · axes=speed,parity,quality,fuzz,precision,truth,goldens,profile,bench
- **F3** : — (indisponible)
- **Baseline** : `harness/baseline.json` (cd05f6a · tier standard · 2026-07-12)
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
- temps cumulé : **3.850 s** sur 10 cas mesurés

| Cas | fractall (s) | min (s) | bruit | vs baseline | ratio F3 |
|---|---:|---:|---:|---:|---:|
| `test5` | 0.009 | 0.009 | — | — | — |
| `spiral` | 0.010 | 0.010 | — | — | — |
| `flake` | 0.070 | 0.070 | — | — | — |
| `glitch_test_2` | 0.035 | 0.035 | — | — | — |
| `e50` | 0.817 | 0.817 | — | — | — |
| `e113` | 0.118 | 0.118 | — | — | — |
| `e401` | 0.167 | 0.167 | — | — | — |
| `e1000` | 0.115 | 0.115 | — | — | — |
| `floral_fantasy` | 0.034 | 0.034 | — | — | — |
| `dragon` | 2.474 | 2.474 | — | — | — |

## Parité (compare_f3 — Δsmooth-iter vs F3)

_F3 indisponible — axe non exécuté._

## Qualité (fractall-quality suite — perturbation vs GMP)

| Verdict | Nombre | vs baseline |
|---|---:|---|
| PASS | 17 |  |
| WARN | 0 |  |
| FAIL | 0 |  |

## Fuzz (sondes aléatoires pert vs GMP)

- seed `20260714` · 4 sondes → **4 PASS · 0 WARN · 0 FAIL**

## Précision d'orbite (juge ×4)

- 12 cas à 96² → **12 PASS · 0 WARN · 0 FAIL** (itérations brutes vs orbite à précision ×4)

| Cas | prec | juge | pixels ≠ | % | verdict |
|---|---:|---:|---:|---:|---|
| `test5` | 256 | 1024 | 0 | 0.000 | PASS |
| `flake` | 624 | 2496 | 0 | 0.000 | PASS |
| `glitch_test_2` | 475 | 1900 | 0 | 0.000 | PASS |
| `e50` | 267 | 1068 | 0 | 0.000 | PASS |
| `e113` | 478 | 1912 | 0 | 0.000 | PASS |
| `e401` | 1433 | 5732 | 0 | 0.000 | PASS |
| `e1000` | 3423 | 13692 | 0 | 0.000 | PASS |
| `floral_fantasy` | 384 | 1536 | 0 | 0.000 | PASS |
| `dragon` | 737 | 2948 | 0 | 0.000 | PASS |
| `golden_spider` ⚑ | 605 | 2420 | 1 | 0.011 | PASS |
| `e227` ⚑ | 857 | 3428 | 0 | 0.000 | PASS |
| `evolution_trees` ⚑ | 858 | 3432 | 1 | 0.011 | PASS |

_⚑ = sentinelle (précision F3 nue : 2-4 % faux avant la marge +64 b)._

## Vérité (rendu de production vs tier dd)

- 23 cas à 96² → **23 PASS · 0 WARN · 0 FAIL** (itérations brutes du rendu par défaut vs frame `--dd-tier`)

| Cas | path | pixels ≠ | % | verdict |
|---|---|---:|---:|---|
| `test5` | bytecode_f64 | 0 | 0.000 | PASS |
| `flake` | bytecode_f64_harmonic_lla | 0 | 0.000 | PASS |
| `glitch_test_2` | bytecode_f64_harmonic_lla | 0 | 0.000 | PASS |
| `e50` | bytecode_f64 | 0 | 0.000 | PASS |
| `e113` | bytecode_f64 | 1 | 0.011 | PASS |
| `e401` | bytecode_exp | 0 | 0.000 | PASS |
| `e1000` | bytecode_exp | 0 | 0.000 | PASS |
| `floral_fantasy` | bytecode_f64_harmonic_lla | 0 | 0.000 | PASS |
| `dragon` | bytecode_f64 | 0 | 0.000 | PASS |
| `e227` ⚑ | bytecode_f64 | 3 | 0.033 | PASS |
| `golden_spider` ⚑ | bytecode_f64 | 0 | 0.000 | PASS |
| `rug` ⚑ | bytecode_f64 | 4 | 0.043 | PASS |
| `olbaid2` ⚑ | bytecode_exp | 0 | 0.000 | PASS |
| `leaded_glass` ⚑ | bytecode_f64_harmonic_lla | 0 | 0.000 | PASS |
| `std:seahorse_1e4_2500` | std_f64 | 0 | 0.000 | PASS |
| `std:seahorse_1e7_2500` | std_f64 | 1 | 0.002 | PASS |
| `std:seahorse_1e10_2500` | std_f64 | 3 | 0.007 | PASS |
| `std:elephants_1e4_2500` | std_f64 | 1 | 0.002 | PASS |
| `std:elephants_1e7_2500` | std_f64 | 0 | 0.000 | PASS |
| `std:elephants_1e10_2500` | std_f64 | 2 | 0.005 | PASS |
| `std:minibrot_1e4_2500` | std_f64 | 0 | 0.000 | PASS |
| `std:minibrot_1e7_2500` | std_f64 | 0 | 0.000 | PASS |
| `std:minibrot_1e10_2500` | std_f64 | 0 | 0.000 | PASS |

_⚑ = sentinelle (rendu par défaut faux avant les correctifs du 2026-11-06). Un FAIL se juge contre le GMP pur : le dd a son propre plancher._

## Goldens (pixel-exact)

- 🟢 VERT

## Profil — où va le temps (sans F3, 1 run par cas)

- **répartition cumulée** : orbit 25.4 % · bla 1.2 % · pixels 41.5 % · post 26.0 % · save 0.0 % · other 5.9 % (sur 10 cas, 16 threads)
- **phase dominante** : pixels ×6, other ×3, orbit ×1
- paths : bytecode_exp, bytecode_f64, bytecode_f64_harmonic_lla, std_f64 · pic RSS max 314 MB

| Cas | path | variantes | wall (s) | orbite % | BLA % | pixels % | post % | save % | autre % | ns/iter | ns/iter·thread | iter/px | esc.dd | fiab. % | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `test5` | bytecode_f64 | harmonic_lla | 0.023 | 4.3 | 0.0 | 4.3 | 0.0 | 0.0 | 91.4 | 2.543 | 40.690 | 6 |  | 0.00 | 12 |
| `spiral` | std_f64 | — | 0.027 | 0.0 | 0.0 | 37.6 | 0.0 | 0.0 | 62.4 | 0.300 | — | 251 |  | — | 13 |
| `flake` | bytecode_f64_harmonic_lla | harmonic_lla | 0.065 | 6.2 | 1.5 | 58.7 | 0.0 | 0.0 | 33.6 | 0.019 | 0.300 | 30921 |  | 0.00 | 12 |
| `glitch_test_2` | bytecode_f64_harmonic_lla | harmonic_lla | 0.063 | 1.6 | 0.0 | 52.6 | 3.2 | 0.0 | 42.7 | 0.065 | 1.036 | 7779 |  | 0.00 | 12 |
| `e50` | bytecode_f64 | harmonic_lla | 1.077 | 3.1 | 0.5 | 74.3 | 19.5 | 0.0 | 2.7 | 0.071 | 1.133 | 172455 |  | 0.69 | 22 |
| `e113` | bytecode_f64 | harmonic_lla | 0.126 | 4.0 | 0.8 | 72.2 | 3.2 | 0.0 | 19.8 | 0.061 | 0.978 | 22729 |  | 0.01 | 12 |
| `e401` | bytecode_exp | bla | 0.167 | 17.9 | 3.6 | 59.7 | 3.0 | 0.0 | 15.8 | 0.055 | 0.885 | 27584 |  | 0.04 | 42 |
| `e1000` | bytecode_exp | bla | 0.097 | 15.5 | 2.1 | 55.9 | 0.0 | 0.0 | 26.4 | 0.050 | 0.793 | 16634 |  | 0.00 | 17 |
| `floral_fantasy` | bytecode_f64_harmonic_lla | harmonic_lla | 0.047 | 2.1 | 0.0 | 34.2 | 0.0 | 0.0 | 63.7 | 0.113 | 1.812 | 2156 |  | 0.00 | 12 |
| `dragon` | bytecode_f64 | harmonic_lla | 2.782 | 37.5 | 1.4 | 25.7 | 33.9 | 0.0 | 1.4 | 0.003 | 0.044 | 3921444 |  | 0.72 | 314 |

## Bench — débit par technique (`--wisdom-bench` isolé)

| Technique | iters/s | ns/iter | taille | vs baseline |
|---|---:|---:|---:|---:|
| cpu_std_f64 | 5.403e+09 | 0.185 | 1024² | — |
| cpu_perturb_f64 | 1.114e+10 | 0.090 | 512² | — |
| cpu_perturb_exp | 2.414e+10 | 0.041 | 1024² | — |
| cpu_perturb_dd | 3.984e+08 | 2.510 | 128² | — |
| cpu_perturb_f64_mid | 1.198e+09 | 0.835 | 512² | — |
| gpu_std_f32 | 3.270e+09 | 0.306 | 2048² | — |
| gpu_perturb_f64 | 6.356e+09 | 0.157 | 1024² | — |
| gpu_perturb_f32 | 6.301e+09 | 0.159 | 2048² | — |

## Tendance (même machine, 6 derniers scores)

| Date (UTC) | Commit | Tier | geomean/F3 | wins | fractall Σs | auto-ratio | parité ok/équiv | Q P/W/F | fuzz P/W/F | goldens | px % | gaps |
|---|---|---|---:|---:|---:|---:|---|---|---|---|---:|---:|
| 2026-07-17 20:01 | `2a9d357` | quick | 0.164 | 10 | — | — | 10/5 | 16/0/0 | 3/0/0 | 🟢 | — | 0 |
| 2026-07-17 21:03 | `2467891` | quick | 0.193 | 10 | — | — | 10/5 | 16/0/0 | 3/0/0 | 🟢 | — | 0 |
| 2026-07-17 21:57 | `5f53b08` | quick | 0.196 | 10 | — | — | 10/5 | 16/0/0 | 3/0/0 | 🟢 | — | 0 |
| 2026-07-18 15:16 | `cf11b59` | quick | 0.191 | 10 | — | — | 10/5 | 16/0/0 | 3/0/0 | 🟢 | — | 0 |
| 2026-07-18 17:18 | `4b7a065` | quick | 0.195 | 10 | — | — | 10/5 | 16/0/0 | 3/0/0 | 🟢 | — | 0 |
| 2026-11-18 02:49 | `372236f` | quick | — | 0 | 3.850 | — | —/— | 17/0/0 | 4/0/0 | 🟢 | 41.5 | 0 |

_`python3 scripts/harness.py trend` pour l'historique complet (`--case <stem>` pour un cas)._

## Gaps (top 10 — sévérité asc, magnitude desc)

_aucun gap détecté 🎉_

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
