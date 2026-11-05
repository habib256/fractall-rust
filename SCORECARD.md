<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-11-05T19:28:27+00:00
- **Commit** : `98dc47a`  ⚠️ arbre modifié (dirty)
- **Machine** : Apple M4 · 10 threads · Darwin 27.2.0
- **Tier** : quick · 256×256 · quality 96×96 · runs=1 · axes=speed,parity,quality,fuzz,precision,goldens,profile,bench
- **F3** : /Users/mercure/src/fractall-rust/fraktaler-3-3.1/fraktaler-3.macos
- **Baseline** : `harness/baseline.json` (cd05f6a · tier standard · 2026-07-12) · **autre machine**
- _baseline présente mais tier différent (standard) — pas de delta._

## Vitesse (ratio fractall/F3, <1 = fractall gagne)

| Métrique | Valeur | vs baseline |
|---|---:|---|
| geomean ratio | — |  |
| pire ratio | — | |
| wins (ratio<1) | 0 | |
| timeouts | 0 | |
| cas comparés | 0/10 | |

### Temps fractall par cas (vitesse intrinsèque)

- _pas d'auto-ratio : baseline d'une autre machine._
- temps cumulé : **3.060 s** sur 10 cas mesurés

| Cas | fractall (s) | min (s) | bruit | vs baseline | ratio F3 |
|---|---:|---:|---:|---:|---:|
| `test5` | 0.018 | 0.018 | — | — | — |
| `spiral` | 0.025 | 0.025 | — | — | — |
| `flake` | 0.023 | 0.023 | — | — | — |
| `glitch_test_2` | 0.026 | 0.026 | — | — | — |
| `e50` | 0.673 | 0.673 | — | — | — |
| `e113` | 0.085 | 0.085 | — | — | — |
| `e401` | 0.460 | 0.460 | — | — | — |
| `e1000` | 0.395 | 0.395 | — | — | — |
| `floral_fantasy` | 0.024 | 0.024 | — | — | — |
| `dragon` | 1.332 | 1.332 | — | — | — |

## Parité (compare_f3 — Δsmooth-iter vs F3)

_F3 build Linux sans support EXR (HAVE_EXR=0) — parité impossible tant que F3 n'est pas rebuilé avec OpenEXR._

## Qualité (fractall-quality suite — perturbation vs GMP)

| Verdict | Nombre | vs baseline |
|---|---:|---|
| PASS | 16 |  |
| WARN | 0 |  |
| FAIL | 0 |  |

## Fuzz (sondes aléatoires pert vs GMP)

- seed `20260714` · 3 sondes → **3 PASS · 0 WARN · 0 FAIL**

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
| `golden_spider` ⚑ | 605 | 2420 | 0 | 0.000 | PASS |
| `e227` ⚑ | 857 | 3428 | 1 | 0.011 | PASS |
| `evolution_trees` ⚑ | 858 | 3432 | 0 | 0.000 | PASS |

_⚑ = sentinelle (précision F3 nue : 2-4 % faux avant la marge +64 b)._

## Goldens (pixel-exact)

- 🟢 VERT

## Profil — où va le temps (sans F3, 1 run par cas)

- **répartition cumulée** : orbit 22.4 % · pixels 56.8 % · post 11.8 % · save 0.0 % · other 9.0 % (sur 10 cas)
- **phase dominante** : other ×5, pixels ×4, orbit ×1
- paths : bytecode_exp, bytecode_f64, bytecode_f64_harmonic_lla, std_f64 · pic RSS max 1063 MB

| Cas | path | variantes | wall (s) | orbite % | pixels % | post % | save % | autre % | ns/iter | iter/px | esc.dd | fiab. % | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `test5` | bytecode_f64 | harmonic_lla | 0.033 | 0.0 | 27.3 | 0.0 | 0.0 | 72.7 | 22.300 | 6 |  | 0.00 | 12 |
| `spiral` | std_f64 | — | 0.035 | 0.0 | 29.0 | 0.0 | 0.0 | 71.0 | 0.600 | 251 |  | — | 16 |
| `flake` | bytecode_f64_harmonic_lla | harmonic_lla | 0.035 | 5.6 | 25.4 | 0.0 | 0.0 | 69.0 | 0.000 | 30921 |  | 0.00 | 13 |
| `glitch_test_2` | bytecode_f64_harmonic_lla | harmonic_lla | 0.035 | 0.0 | 31.1 | 0.0 | 0.0 | 68.9 | 0.000 | 7779 |  | 0.00 | 12 |
| `e50` | bytecode_f64 | harmonic_lla | 0.649 | 3.4 | 73.0 | 19.6 | 0.0 | 4.0 | 0.000 | 172455 |  | 0.38 | 58 |
| `e113` | bytecode_f64 | harmonic_lla | 0.098 | 3.1 | 70.3 | 0.0 | 0.0 | 26.6 | 0.000 | 22729 |  | 0.00 | 13 |
| `e401` | bytecode_exp | bla | 0.414 | 2.2 | 91.3 | 0.0 | 0.0 | 6.5 | 0.200 | 27584 |  | 0.00 | 49 |
| `e1000` | bytecode_exp | bla | 0.368 | 1.9 | 91.0 | 0.0 | 0.0 | 7.1 | 0.300 | 16634 |  | 0.00 | 24 |
| `floral_fantasy` | bytecode_f64_harmonic_lla | harmonic_lla | 0.035 | 2.9 | 25.9 | 0.0 | 0.0 | 71.2 | 0.100 | 2156 |  | 0.00 | 12 |
| `dragon` | bytecode_f64 | harmonic_lla | 1.329 | 47.8 | 31.5 | 17.4 | 0.0 | 3.3 | 0.000 | 3921441 |  | 0.12 | 1063 |

## Bench — débit par technique (`--wisdom-bench` isolé)

| Technique | iters/s | ns/iter | taille | vs baseline |
|---|---:|---:|---:|---:|
| cpu_std_f64 | 1.917e+09 | 0.522 | 1024² | — |
| cpu_perturb_f64 | 1.897e+10 | 0.053 | 512² | — |
| cpu_perturb_exp | 1.473e+10 | 0.068 | 512² | — |
| cpu_perturb_dd | 3.490e+08 | 2.865 | 128² | — |
| gpu_std_f32 | 1.148e+10 | 0.087 | 2048² | — |

## Gaps (top 10 — sévérité asc, magnitude desc)

_aucun gap détecté 🎉_

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
