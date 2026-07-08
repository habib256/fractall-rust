<!-- généré par scripts/harness.py — ne pas éditer à la main -->

# SCORECARD — fractall vs Fraktaler-3

- **Date** : 2026-07-07T15:31:58+00:00
- **Commit** : `6ed5fa9`
- **Machine** : Intel(R) Core(TM) i7-10700F CPU @ 2.90GHz · 16 threads · Linux 6.14.0-37-generic
- **Tier** : full · 256×256 · runs=1 · axes=speed
- **F3** : /home/gistarcade/src/fractall-rust/fraktaler-3-3.1/fraktaler-3-3.1.linux
- _baseline présente mais tier différent (quick) — pas de delta._

## Vitesse (ratio fractall/F3, <1 = fractall gagne)

| Métrique | Valeur | vs baseline |
|---|---:|---|
| geomean ratio | 0.294 |  |
| pire ratio | 1.983 (glitch_test_5) | |
| wins (ratio<1) | 72 | |
| timeouts | 2 | |
| cas comparés | 77/84 | |

## Au-delà de F3

- **speed_wins** : 11_dimensions, adventurous_forest, all_seeing_eye, dinosaur_fossils, dragon, dragon_detail, e1000, e1016, e1086, e1121, e113, e1200, e1298, e227, e318, e401, e50, e533, e634, e890, evolution_trees, flake, floral_fantasy, glitch_test_1, glitch_test_3, glitch_test_4, glitch_test_6, glitch_test_7, golden_spider, hard, heaven, infinity, integral_of_ex2, layers, leaded_glass, lethal_weapon, liiiines, line, long, lya, magic, mitosis, mitosis2, nr_fail, olbaid1, olbaid2, olbaid3, olbaid4, olbaid5, opus, peanuts, rug, safari, spiral, ssssss, test, test2, test3, test4, test5, test6, the_complexity_of_a_line, threads_colour, tick_tock, triangle, uranium, verstoppertje, virus, wfs, wfs2, windmill, x

## Gaps (top 10 — sévérité asc, magnitude desc)

| # | Sévérité | Axe | Cas | Métrique | Valeur | Note |
|---:|---|---|---|---|---:|---|
| 1 | 2 robustesse | speed | `e22522` | status | quarantined | cas en QUARANTAINE — crash/OOM connu, voir harness/crash-journal.jsonl |
| 2 | 2 robustesse | speed | `opus2` | status | quarantined | cas en QUARANTAINE — crash/OOM connu, voir harness/crash-journal.jsonl |
| 3 | 2 robustesse | speed | `orion` | status | quarantined | cas en QUARANTAINE — crash/OOM connu, voir harness/crash-journal.jsonl |
| 4 | 2 robustesse | speed | `seahorse` | status | quarantined | cas en QUARANTAINE — crash/OOM connu, voir harness/crash-journal.jsonl |
| 5 | 2 robustesse | speed | `wfs_mb` | status | quarantined | cas en QUARANTAINE — crash/OOM connu, voir harness/crash-journal.jsonl |

---
_Scorecards versionnés : `harness/history/` · baseline : `harness/baseline.json`. Généré par `scripts/harness.py`._
