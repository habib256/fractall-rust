<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-11-06T04:39:55+00:00
- **Commit** : `0473296`  ⚠️ arbre modifié (dirty)
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

- **auto-ratio vs baseline `4bfbdba`** (même machine, geomean cur/base sur 10 cas, <1 = plus rapide) : **0.981** · pire 1.100 (spiral)
- temps cumulé : **2.261 s** sur 10 cas mesurés

| Cas | fractall (s) | min (s) | bruit | vs baseline | ratio F3 |
|---|---:|---:|---:|---:|---:|
| `test5` | 0.013 | 0.013 | — | 1.08× | — |
| `spiral` | 0.013 | 0.013 | — | 1.10× | — |
| `flake` | 0.042 | 0.042 | — | 0.86× | — |
| `glitch_test_2` | 0.048 | 0.048 | — | 1.08× | — |
| `e50` | 0.600 | 0.600 | — | 0.99× | — |
| `e113` | 0.084 | 0.084 | — | 0.95× | — |
| `e401` | 0.081 | 0.081 | — | 0.94× | — |
| `e1000` | 0.043 | 0.043 | — | 0.88× | — |
| `floral_fantasy` | 0.024 | 0.024 | — | 0.95× | — |
| `dragon` | 1.312 | 1.312 | — | 1.01× | — |

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

- **répartition cumulée** : orbit 21.2 % · bla 0.7 % · pixels 47.1 % · post 20.3 % · save 0.0 % · other 10.7 % (sur 10 cas, 10 threads)
- **phase dominante** : pixels ×5, other ×4, orbit ×1
- paths : bytecode_exp, bytecode_f64, bytecode_f64_harmonic_lla, std_f64 · pic RSS max 329 MB

| Cas | path | variantes | wall (s) | orbite % | BLA % | pixels % | post % | save % | autre % | ns/iter | ns/iter·thread | iter/px | esc.dd | fiab. % | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `test5` | bytecode_f64 | harmonic_lla | 0.025 | 0.0 | 0.0 | 4.1 | 0.0 | 0.0 | 95.9 | 2.543 | 25.431 | 6 |  | 0.00 | 12 |
| `spiral` | std_f64 | — | 0.026 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.100 | — | 251 |  | — | 15 |
| `flake` | bytecode_f64_harmonic_lla | harmonic_lla | 0.053 | 3.7 | 0.0 | 50.6 | 0.0 | 0.0 | 45.7 | 0.013 | 0.133 | 30921 |  | 0.00 | 12 |
| `glitch_test_2` | bytecode_f64_harmonic_lla | harmonic_lla | 0.044 | 0.0 | 0.0 | 45.2 | 2.3 | 0.0 | 52.6 | 0.039 | 0.392 | 7779 |  | 0.00 | 12 |
| `e50` | bytecode_f64 | harmonic_lla | 0.613 | 2.3 | 0.2 | 78.0 | 15.7 | 0.0 | 3.9 | 0.042 | 0.423 | 172455 |  | 0.69 | 30 |
| `e113` | bytecode_f64 | harmonic_lla | 0.085 | 2.4 | 0.0 | 67.4 | 2.4 | 0.0 | 27.9 | 0.038 | 0.383 | 22729 |  | 0.01 | 14 |
| `e401` | bytecode_exp | bla | 0.087 | 10.3 | 2.3 | 57.2 | 2.3 | 0.0 | 27.9 | 0.028 | 0.277 | 27584 |  | 0.04 | 51 |
| `e1000` | bytecode_exp | bla | 0.059 | 10.1 | 1.7 | 45.5 | 0.0 | 0.0 | 42.7 | 0.025 | 0.248 | 16634 |  | 0.00 | 24 |
| `floral_fantasy` | bytecode_f64_harmonic_lla | harmonic_lla | 0.036 | 0.0 | 0.0 | 27.9 | 0.0 | 0.0 | 72.1 | 0.071 | 0.708 | 2156 |  | 0.00 | 12 |
| `dragon` | bytecode_f64 | harmonic_lla | 1.291 | 35.5 | 1.0 | 32.7 | 28.7 | 0.0 | 2.1 | 0.002 | 0.016 | 3921444 |  | 0.72 | 329 |

## Bench — débit par technique (`--wisdom-bench` isolé)

| Technique | iters/s | ns/iter | taille | vs baseline |
|---|---:|---:|---:|---:|
| cpu_std_f64 | 1.597e+10 | 0.063 | 1024² | 1.85× |
| cpu_perturb_f64 | 1.980e+10 | 0.050 | 512² | 1.00× |
| cpu_perturb_exp | 5.587e+10 | 0.018 | 1024² | 1.14× |
| cpu_perturb_dd | 8.529e+08 | 1.173 | 128² | 1.00× |
| cpu_perturb_f64_mid | 1.992e+09 | 0.502 | 1024² | 1.00× |
| gpu_std_f32 | 1.059e+10 | 0.094 | 2048² | 1.00× |
| gpu_perturb_f32 | 5.122e+09 | 0.195 | 1024² | 1.00× |

## Tendance (même machine, 6 derniers scores)

| Date (UTC) | Commit | Tier | geomean/F3 | wins | fractall Σs | auto-ratio | parité ok/équiv | Q P/W/F | fuzz P/W/F | goldens | px % | gaps |
|---|---|---|---:|---:|---:|---:|---|---|---|---|---:|---:|
| 2026-11-06 00:45 | `61df5f3` | standard | — | 0 | 4.761 | — | —/— | 15/1/0 | 6/0/0 | 🟢 | 49.8 | 1 |
| 2026-11-06 01:01 | `4bfbdba` | quick | — | 0 | 2.280 | 0.695 | —/— | 16/0/0 | 3/0/0 | 🟢 | 46.9 | 0 |
| 2026-11-06 01:52 | `6bb4e9e` | quick | — | 0 | 2.268 | 0.974 | —/— | 16/0/0 | 3/0/0 | 🟢 | 46.9 | 0 |
| 2026-11-06 02:11 | `1b1a394` | quick | — | 0 | 2.306 | 0.998 | —/— | 16/0/0 | 3/0/0 | 🟢 | 47.4 | 0 |
| 2026-11-06 02:15 | `791fec8` | standard | — | 0 | 4.709 | — | —/— | 15/1/0 | 6/0/0 | 🟢 | 50.3 | 1 |
| 2026-11-06 04:39 | `0473296` | quick | — | 0 | 2.261 | 0.981 | —/— | 17/0/0 | 3/0/0 | 🟢 | 47.1 | 0 |

_`python3 scripts/harness.py trend` pour l'historique complet (`--case <stem>` pour un cas)._

## Gaps (top 10 — sévérité asc, magnitude desc)

_aucun gap détecté 🎉_

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
