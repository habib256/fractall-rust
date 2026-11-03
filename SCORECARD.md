<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-11-03T04:52:27+00:00
- **Commit** : `169a8de`  ⚠️ arbre modifié (dirty)
- **Machine** : Intel(R) Xeon(R) Processor · 4 threads · Linux 6.18.44-fc-v230
- **Tier** : quick · 256×256 · quality 96×96 · runs=1 · axes=speed,parity,quality,fuzz,goldens,profile,bench
- **F3** : /home/user/fractall-rust/fraktaler-3-3.1/fraktaler-3-3.1.linux
- **Baseline** : `harness/baselines/intel-r-xeon-r-processor-4t.json` (169a8de · tier quick · 2026-11-03)

## Vitesse (ratio fractall/F3, <1 = fractall gagne)

| Métrique | Valeur | vs baseline |
|---|---:|---|
| geomean ratio | 0.265 |  (=) |
| pire ratio | 1.101 (test5) | |
| wins (ratio<1) | 9 | |
| timeouts | 0 | |
| cas comparés | 10/10 | |

### Temps fractall par cas (vitesse intrinsèque)

- **auto-ratio vs baseline `169a8de`** (même machine, geomean cur/base sur 10 cas, <1 = plus rapide) : **1.000** · pire 1.000 (test5)
- temps cumulé : **8.778 s** sur 10 cas mesurés

| Cas | fractall (s) | min (s) | bruit | vs baseline | ratio F3 |
|---|---:|---:|---:|---:|---:|
| `test5` | 0.037 | 0.037 | — | 1.00× | 1.101 |
| `spiral` | 0.037 | 0.037 | — | 1.00× | 0.101 |
| `flake` | 0.037 | 0.037 | — | 1.00× | 0.088 |
| `glitch_test_2` | 0.066 | 0.066 | — | 1.00× | 0.246 |
| `e50` | 1.926 | 1.926 | — | 1.00× | 0.795 |
| `e113` | 0.219 | 0.219 | — | 1.00× | 0.384 |
| `e401` | 0.769 | 0.769 | — | 1.00× | 0.125 |
| `e1000` | 0.719 | 0.719 | — | 1.00× | 0.251 |
| `floral_fantasy` | 0.034 | 0.034 | — | 1.00× | 0.107 |
| `dragon` | 4.934 | 4.934 | — | 1.00× | 0.687 |

## Parité (compare_f3 — Δsmooth-iter vs F3)

| Métrique | Valeur | vs baseline |
|---|---:|---|
| n_ok | 10 |  (=) |
| pixel-équivalents (<0.01%) | 5 |  (=) |
| échecs | 0 |  (=) |
| timeouts | 0 |  (=) |
| F3-dégénéré (win fractall) | 0 |  (=) |

## Qualité (fractall-quality suite — perturbation vs GMP)

| Verdict | Nombre | vs baseline |
|---|---:|---|
| PASS | 16 |  (=) |
| WARN | 0 |  (=) |
| FAIL | 0 |  (=) |

## Fuzz (sondes aléatoires pert vs GMP)

- seed `20260714` · 3 sondes → **3 PASS · 0 WARN · 0 FAIL** (baseline : 3P/0W/0F)

## Goldens (pixel-exact)

- 🟢 VERT

## Profil — où va le temps (sans F3, 1 run par cas)

- **répartition cumulée** : orbit 14.7 % · pixels 51.3 % · post 32.1 % · save 0.0 % · other 1.8 % (sur 10 cas)
- **phase dominante** : pixels ×9, post ×1
- paths : bytecode_exp, bytecode_f64, bytecode_f64_harmonic_lla

| Cas | path | variantes | wall (s) | orbite % | pixels % | post % | save % | autre % | ns/iter | iter/px | esc.dd | fiab. % |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `test5` | bytecode_f64 | harmonic_lla | 0.022 | 4.6 | 64.8 | 0.0 | 0.0 | 30.6 | 34.600 | 6 |  | 0.00 |
| `spiral` | ? | — | 0.028 | 0.0 | 70.7 | 0.0 | 0.0 | 29.3 | — | — |  | — |
| `flake` | bytecode_f64_harmonic_lla | harmonic_lla | 0.028 | 17.7 | 56.6 | 0.0 | 0.0 | 25.7 | 0.000 | 30921 |  | 0.00 |
| `glitch_test_2` | bytecode_f64_harmonic_lla | harmonic_lla | 0.037 | 2.7 | 78.9 | 0.0 | 0.0 | 18.4 | 0.100 | 7779 |  | 0.00 |
| `e50` | bytecode_f64 | harmonic_lla | 1.919 | 2.2 | 75.8 | 21.2 | 0.0 | 0.8 | 0.100 | 172455 |  | 0.38 |
| `e113` | bytecode_f64 | harmonic_lla | 0.194 | 3.1 | 92.6 | 0.0 | 0.0 | 4.3 | 0.100 | 22729 |  | 0.00 |
| `e401` | bytecode_exp | bla | 0.783 | 4.9 | 92.9 | 0.0 | 0.0 | 2.3 | 0.400 | 27584 |  | 0.00 |
| `e1000` | bytecode_exp | bla | 0.774 | 2.2 | 96.5 | 0.0 | 0.0 | 1.3 | 0.700 | 16634 |  | 0.00 |
| `floral_fantasy` | bytecode_f64_harmonic_lla | harmonic_lla | 0.026 | 3.8 | 68.3 | 0.0 | 0.0 | 27.9 | 0.100 | 2156 |  | 0.00 |
| `dragon` | bytecode_f64 | harmonic_lla | 4.913 | 23.9 | 25.9 | 48.8 | 0.0 | 1.5 | 0.000 | 3921441 |  | 0.12 |

## Bench — débit par technique (`--wisdom-bench` isolé)

| Technique | iters/s | ns/iter | taille | vs baseline |
|---|---:|---:|---:|---:|
| cpu_std_f64 | 1.028e+09 | 0.973 | 1024² | 1.00× |
| cpu_perturb_f64 | 6.050e+09 | 0.165 | 256² | 1.00× |
| cpu_perturb_exp | 1.456e+10 | 0.069 | 512² | 1.00× |
| cpu_perturb_dd | 1.339e+08 | 7.467 | 128² | 1.00× |
| gpu_std_f32 | 3.078e+08 | 3.249 | 2048² | 1.00× |

## Tendance (même machine, 6 derniers scores)

| Date (UTC) | Commit | Tier | geomean/F3 | wins | fractall Σs | auto-ratio | parité ok/équiv | Q P/W/F | fuzz P/W/F | goldens | px % | gaps |
|---|---|---|---:|---:|---:|---:|---|---|---|---|---:|---:|
| 2026-11-03 04:41 | `922ddf0` | quick | — | 0 | 8.915 | — | —/— | 16/0/0 | 3/0/0 | 🟢 | 50.6 | 0 |
| 2026-11-03 04:52 | `169a8de` | quick | 0.265 | 9 | 8.778 | — | 10/5 | 16/0/0 | 3/0/0 | 🟢 | 51.3 | 0 |

_`python3 scripts/harness.py trend` pour l'historique complet (`--case <stem>` pour un cas)._

## Au-delà de F3

- **speed_wins** : spiral, flake, glitch_test_2, e50, e113, e401, e1000, floral_fantasy, dragon

## Gaps (top 10 — sévérité asc, magnitude desc)

_aucun gap détecté 🎉_

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
