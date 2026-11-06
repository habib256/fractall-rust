<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-11-06T05:37:07+00:00
- **Commit** : `b5cdf1a`  ⚠️ arbre modifié (dirty)
- **Machine** : Apple M4 · 10 threads · Darwin 27.2.0
- **Tier** : quick · 256×256 · quality 96×96 · runs=1 · axes=speed,parity,quality,fuzz,precision,truth,goldens,profile,bench
- **F3** : — (indisponible)
- **Baseline** : `harness/baselines/apple-m4-10t.json` (4bfbdba · tier quick · 2026-11-06)

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

- **auto-ratio vs baseline `4bfbdba`** (même machine, geomean cur/base sur 10 cas, <1 = plus rapide) : **1.040** · pire 1.123 (e1000)
- temps cumulé : **2.353 s** sur 10 cas mesurés

| Cas | fractall (s) | min (s) | bruit | vs baseline | ratio F3 |
|---|---:|---:|---:|---:|---:|
| `test5` | 0.013 | 0.013 | — | 1.07× | — |
| `spiral` | 0.012 | 0.012 | — | 0.99× | — |
| `flake` | 0.050 | 0.050 | — | 1.01× | — |
| `glitch_test_2` | 0.048 | 0.048 | — | 1.08× | — |
| `e50` | 0.649 | 0.649 | — | 1.07× | — |
| `e113` | 0.091 | 0.091 | — | 1.04× | — |
| `e401` | 0.091 | 0.091 | — | 1.06× | — |
| `e1000` | 0.055 | 0.055 | — | 1.12× | — |
| `floral_fantasy` | 0.024 | 0.024 | — | 0.97× | — |
| `dragon` | 1.319 | 1.319 | — | 1.01× | — |

## Parité (compare_f3 — Δsmooth-iter vs F3)

_F3 indisponible — axe non exécuté._

## Qualité (fractall-quality suite — perturbation vs GMP)

| Verdict | Nombre | vs baseline |
|---|---:|---|
| PASS | 17 |  (↑1 ✅) |
| WARN | 0 |  (=) |
| FAIL | 0 |  (=) |

## Fuzz (sondes aléatoires pert vs GMP)

- seed `20260714` · 3 sondes → **3 PASS · 0 WARN · 0 FAIL** (baseline : 3P/0W/0F)

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

- 14 cas à 96² → **14 PASS · 0 WARN · 0 FAIL** (itérations brutes du rendu par défaut vs frame `--dd-tier`)

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

_⚑ = sentinelle (rendu par défaut faux avant les correctifs du 2026-11-06). Un FAIL se juge contre le GMP pur : le dd a son propre plancher._

## Goldens (pixel-exact)

- 🟢 VERT

## Profil — où va le temps (sans F3, 1 run par cas)

- **répartition cumulée** : orbit 21.5 % · bla 0.7 % · pixels 47.5 % · post 19.4 % · save 0.0 % · other 10.9 % (sur 10 cas, 10 threads)
- **phase dominante** : pixels ×5, other ×4, orbit ×1
- paths : bytecode_exp, bytecode_f64, bytecode_f64_harmonic_lla, std_f64 · pic RSS max 329 MB

| Cas | path | variantes | wall (s) | orbite % | BLA % | pixels % | post % | save % | autre % | ns/iter | ns/iter·thread | iter/px | esc.dd | fiab. % | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `test5` | bytecode_f64 | harmonic_lla | 0.026 | 0.0 | 0.0 | 3.9 | 0.0 | 0.0 | 96.1 | 2.543 | 25.431 | 6 |  | 0.00 | 12 |
| `spiral` | std_f64 | — | 0.026 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.100 | — | 251 |  | — | 15 |
| `flake` | bytecode_f64_harmonic_lla | harmonic_lla | 0.053 | 3.8 | 0.0 | 51.3 | 0.0 | 0.0 | 44.9 | 0.013 | 0.133 | 30921 |  | 0.00 | 13 |
| `glitch_test_2` | bytecode_f64_harmonic_lla | harmonic_lla | 0.044 | 0.0 | 0.0 | 45.7 | 2.3 | 0.0 | 52.1 | 0.039 | 0.392 | 7779 |  | 0.00 | 12 |
| `e50` | bytecode_f64 | harmonic_lla | 0.611 | 2.3 | 0.2 | 77.7 | 15.7 | 0.0 | 4.1 | 0.042 | 0.420 | 172455 |  | 0.69 | 30 |
| `e113` | bytecode_f64 | harmonic_lla | 0.085 | 2.4 | 0.0 | 67.5 | 1.2 | 0.0 | 29.0 | 0.038 | 0.383 | 22729 |  | 0.01 | 14 |
| `e401` | bytecode_exp | bla | 0.088 | 10.2 | 2.3 | 56.8 | 3.4 | 0.0 | 27.3 | 0.028 | 0.277 | 27584 |  | 0.04 | 52 |
| `e1000` | bytecode_exp | bla | 0.060 | 11.7 | 1.7 | 45.1 | 0.0 | 0.0 | 41.5 | 0.025 | 0.248 | 16634 |  | 0.00 | 24 |
| `floral_fantasy` | bytecode_f64_harmonic_lla | harmonic_lla | 0.035 | 0.0 | 0.0 | 25.8 | 0.0 | 0.0 | 74.2 | 0.064 | 0.637 | 2156 |  | 0.00 | 12 |
| `dragon` | bytecode_f64 | harmonic_lla | 1.260 | 36.4 | 1.0 | 33.3 | 27.2 | 0.0 | 2.2 | 0.002 | 0.016 | 3921444 |  | 0.72 | 329 |

## Bench — débit par technique (`--wisdom-bench` isolé)

| Technique | iters/s | ns/iter | taille | vs baseline |
|---|---:|---:|---:|---:|
| cpu_std_f64 | 1.548e+10 | 0.065 | 1024² | 1.80× |
| cpu_perturb_f64 | 1.982e+10 | 0.050 | 512² | 1.01× |
| cpu_perturb_exp | 5.635e+10 | 0.018 | 1024² | 1.15× |
| cpu_perturb_dd | 8.467e+08 | 1.181 | 128² | 1.00× |
| cpu_perturb_f64_mid | 1.979e+09 | 0.505 | 1024² | 0.99× |
| gpu_std_f32 | 1.066e+10 | 0.094 | 2048² | 1.01× |
| gpu_perturb_f32 | 5.109e+09 | 0.196 | 1024² | 1.00× |

## Tendance (même machine, 6 derniers scores)

| Date (UTC) | Commit | Tier | geomean/F3 | wins | fractall Σs | auto-ratio | parité ok/équiv | Q P/W/F | fuzz P/W/F | goldens | px % | gaps |
|---|---|---|---:|---:|---:|---:|---|---|---|---|---:|---:|
| 2026-11-06 01:52 | `6bb4e9e` | quick | — | 0 | 2.268 | 0.974 | —/— | 16/0/0 | 3/0/0 | 🟢 | 46.9 | 0 |
| 2026-11-06 02:11 | `1b1a394` | quick | — | 0 | 2.306 | 0.998 | —/— | 16/0/0 | 3/0/0 | 🟢 | 47.4 | 0 |
| 2026-11-06 02:15 | `791fec8` | standard | — | 0 | 4.709 | — | —/— | 15/1/0 | 6/0/0 | 🟢 | 50.3 | 1 |
| 2026-11-06 04:39 | `0473296` | quick | — | 0 | 2.261 | 0.981 | —/— | 17/0/0 | 3/0/0 | 🟢 | 47.1 | 0 |
| 2026-11-06 05:10 | `b3ed210` | standard | — | 0 | 4.661 | — | —/— | 16/1/0 | 6/0/0 | 🟢 | 51.4 | 1 |
| 2026-11-06 05:37 | `b5cdf1a` | quick | — | 0 | 2.353 | 1.040 | —/— | 17/0/0 | 3/0/0 | 🟢 | 47.5 | 0 |

_`python3 scripts/harness.py trend` pour l'historique complet (`--case <stem>` pour un cas)._

## Gaps (top 10 — sévérité asc, magnitude desc)

_aucun gap détecté 🎉_

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
