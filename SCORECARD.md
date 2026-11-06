<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-11-06T06:11:44+00:00
- **Commit** : `e3ff8be`  ⚠️ arbre modifié (dirty)
- **Machine** : Apple M4 · 10 threads · Darwin 27.2.0
- **Tier** : quick · 256×256 · quality 96×96 · runs=3 · axes=speed,parity,quality,fuzz,precision,truth,goldens,profile,bench
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

- **auto-ratio vs baseline `4bfbdba`** (même machine, geomean cur/base sur 10 cas, <1 = plus rapide) : **0.957** · pire 1.065 (e113)
- temps cumulé : **2.292 s** sur 10 cas mesurés

| Cas | fractall (s) | min (s) | bruit | vs baseline | ratio F3 |
|---|---:|---:|---:|---:|---:|
| `test5` | 0.012 | 0.012 | 89 % | 1.04× | — |
| `spiral` | 0.011 | 0.011 | 11 % | 0.92× | — |
| `flake` | 0.050 | 0.049 | 1 % | 1.00× | — |
| `glitch_test_2` | 0.026 | 0.026 | 2 % | 0.59× | — |
| `e50` | 0.617 | 0.605 | 8 % | 1.01× | — |
| `e113` | 0.094 | 0.086 | 10 % | 1.06× | — |
| `e401` | 0.090 | 0.079 | 16 % | 1.04× | — |
| `e1000` | 0.050 | 0.049 | 13 % | 1.03× | — |
| `floral_fantasy` | 0.025 | 0.024 | 5 % | 0.98× | — |
| `dragon` | 1.318 | 1.317 | 0 % | 1.01× | — |

## Parité (compare_f3 — Δsmooth-iter vs F3)

_F3 indisponible — axe non exécuté._

## Qualité (fractall-quality suite — perturbation vs GMP)

| Verdict | Nombre | vs baseline |
|---|---:|---|
| PASS | 17 |  (↑1 ✅) |
| WARN | 0 |  (=) |
| FAIL | 0 |  (=) |

## Fuzz (sondes aléatoires pert vs GMP)

- seed `20260714` · 4 sondes → **4 PASS · 0 WARN · 0 FAIL** (baseline : 3P/0W/0F)

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

- **répartition cumulée** : orbit 21.5 % · bla 0.7 % · pixels 48.1 % · post 18.9 % · save 0.0 % · other 10.8 % (sur 10 cas, 10 threads)
- **phase dominante** : pixels ×5, other ×4, orbit ×1
- paths : bytecode_exp, bytecode_f64, bytecode_f64_harmonic_lla, std_f64 · pic RSS max 329 MB

| Cas | path | variantes | wall (s) | orbite % | BLA % | pixels % | post % | save % | autre % | ns/iter | ns/iter·thread | iter/px | esc.dd | fiab. % | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `test5` | bytecode_f64 | harmonic_lla | 0.025 | 0.0 | 0.0 | 4.0 | 0.0 | 0.0 | 96.0 | 2.543 | 25.431 | 6 |  | 0.00 | 12 |
| `spiral` | std_f64 | — | 0.026 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.100 | — | 251 |  | — | 16 |
| `flake` | bytecode_f64_harmonic_lla | harmonic_lla | 0.052 | 3.9 | 0.0 | 52.0 | 0.0 | 0.0 | 44.1 | 0.013 | 0.133 | 30921 |  | 0.00 | 13 |
| `glitch_test_2` | bytecode_f64_harmonic_lla | harmonic_lla | 0.044 | 0.0 | 0.0 | 45.2 | 2.3 | 0.0 | 52.5 | 0.039 | 0.392 | 7779 |  | 0.00 | 12 |
| `e50` | bytecode_f64 | harmonic_lla | 0.618 | 2.3 | 0.2 | 77.6 | 16.0 | 0.0 | 4.0 | 0.042 | 0.424 | 172455 |  | 0.69 | 30 |
| `e113` | bytecode_f64 | harmonic_lla | 0.085 | 2.4 | 0.0 | 67.1 | 2.4 | 0.0 | 28.1 | 0.038 | 0.383 | 22729 |  | 0.01 | 14 |
| `e401` | bytecode_exp | bla | 0.088 | 10.3 | 2.3 | 57.1 | 3.4 | 0.0 | 27.0 | 0.028 | 0.277 | 27584 |  | 0.04 | 51 |
| `e1000` | bytecode_exp | bla | 0.060 | 11.6 | 1.7 | 44.7 | 0.0 | 0.0 | 42.0 | 0.025 | 0.248 | 16634 |  | 0.00 | 24 |
| `floral_fantasy` | bytecode_f64_harmonic_lla | harmonic_lla | 0.034 | 0.0 | 0.0 | 26.2 | 0.0 | 0.0 | 73.8 | 0.064 | 0.637 | 2156 |  | 0.00 | 12 |
| `dragon` | bytecode_f64 | harmonic_lla | 1.246 | 36.5 | 1.0 | 34.2 | 26.2 | 0.0 | 2.1 | 0.002 | 0.017 | 3921444 |  | 0.72 | 329 |

## Bench — débit par technique (`--wisdom-bench` isolé)

| Technique | iters/s | ns/iter | taille | vs baseline |
|---|---:|---:|---:|---:|
| cpu_std_f64 | 1.591e+10 | 0.063 | 1024² | 1.84× |
| cpu_perturb_f64 | 1.975e+10 | 0.051 | 512² | 1.00× |
| cpu_perturb_exp | 5.615e+10 | 0.018 | 1024² | 1.15× |
| cpu_perturb_dd | 8.518e+08 | 1.174 | 128² | 1.00× |
| cpu_perturb_f64_mid | 1.994e+09 | 0.502 | 1024² | 1.00× |
| gpu_std_f32 | 1.065e+10 | 0.094 | 2048² | 1.01× |
| gpu_perturb_f32 | 5.120e+09 | 0.195 | 1024² | 1.00× |

## Tendance (même machine, 6 derniers scores)

| Date (UTC) | Commit | Tier | geomean/F3 | wins | fractall Σs | auto-ratio | parité ok/équiv | Q P/W/F | fuzz P/W/F | goldens | px % | gaps |
|---|---|---|---:|---:|---:|---:|---|---|---|---|---:|---:|
| 2026-11-06 02:15 | `791fec8` | standard | — | 0 | 4.709 | — | —/— | 15/1/0 | 6/0/0 | 🟢 | 50.3 | 1 |
| 2026-11-06 04:39 | `0473296` | quick | — | 0 | 2.261 | 0.981 | —/— | 17/0/0 | 3/0/0 | 🟢 | 47.1 | 0 |
| 2026-11-06 05:10 | `b3ed210` | standard | — | 0 | 4.661 | — | —/— | 16/1/0 | 6/0/0 | 🟢 | 51.4 | 1 |
| 2026-11-06 05:37 | `b5cdf1a` | quick | — | 0 | 2.353 | 1.040 | —/— | 17/0/0 | 3/0/0 | 🟢 | 47.5 | 0 |
| 2026-11-06 06:09 | `376c59c` | quick | — | 0 | 2.360 | 0.989 | —/— | 17/0/0 | 4/0/0 | 🟢 | 46.8 | 0 |
| 2026-11-06 06:11 | `e3ff8be` | quick | — | 0 | 2.292 | 0.957 | —/— | 17/0/0 | 4/0/0 | 🟢 | 48.1 | 0 |

_`python3 scripts/harness.py trend` pour l'historique complet (`--case <stem>` pour un cas)._

## Gaps (top 10 — sévérité asc, magnitude desc)

_aucun gap détecté 🎉_

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
