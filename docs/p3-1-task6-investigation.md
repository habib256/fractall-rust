# P3.1 #6 — Investigation rebasing F3 pur

État au 2026-05-17 sur branche `bytecode`.

## Mesure du gap legacy vs F3-pure

Pour chaque cas perturbation, deux renders 160×100 :
- legacy : `--algorithm perturbation` (defaut, `use_legacy_glitch_detection=true`)
- F3-pure : `--algorithm perturbation --no-legacy-glitch-detection`

| Cas | Type | Zoom | % pixels diff | mean \|d\| (RGB) | px lourdement diff (max canal > 50) |
|-----|------|------|---------------|------------------|--------------------------------------|
| m_1e6 | Mandelbrot | 1e6 | 97.6 % | 63.31 | 78.31 % |
| m_1e8 | Mandelbrot | 1e8 | 95.8 % | 90.79 | 83.94 % |
| bs_1e3 | Burning Ship | 1e3 | 93.1 % | 99.66 | 80.94 % |
| bs_1e5 | Burning Ship | 1e5 | 99.5 % | 121.13 | 94.34 % |
| tc_1e2 | Tricorn | 1e2 | 58.9 % | 143.34 | 58.77 % |

→ Le path "rebasing seul" (style F3) produit aujourd'hui **des images
fondamentalement différentes** des goldens (qui utilisent legacy). Confirmé
ce qu'indiquait le commentaire dans `types.rs::use_legacy_glitch_detection`.

## Code en place vs F3 référence

`delta.rs::should_rebase` implémente la condition F3 (`z_curr² < delta²`) avec
une exception : `if z_ref_norm_sqr < 1e-20 { return false }`. Cette exception
diffère subtilement de F3 (qui laisse la condition `Zz2 < z2` se trancher
naturellement quand Z=0).

Le déclencheur F3 secondaire (`m + 1 == ref_len`) est lui présent dans le
code (cf. ligne 1261 « Reached end of effective orbit: rebase »).

## Pourquoi le path F3-pure rate aujourd'hui

Hypothèses (à valider) :
1. **Interaction BLA + rebasing**. F3 fait `if (b) { … continue }` puis check
   rebasing **après** le pas perturbation. Notre code a un check rebase
   post-BLA mais l'ordre/condition pourrait diverger.
2. **Référence orbit pas idéale**. F3 fait du nucleus-snapping agressif via
   `optimize_reference_center` (qui existe chez nous mais seulement pour
   Mandelbrot, et seulement quand `zoom > seuil`).
3. **Rebasing post-BLA absent**. Quand un step BLA saute plusieurs itérations,
   F3 met à jour `m` mais n'évalue pas `Zz2 < z2` à chaque sub-itération
   sautée — il évalue juste à la fin. Le legacy code peut faire mieux ou pire.
4. **Pas de mécanisme phase-aware**. F3 cycle `phase` à chaque rebase ; en
   mono-phase c'est no-op. Mais notre code a un `current_phase` dans
   `iterate_pixel` qui pourrait être incorrect quand `use_legacy=false`.

## Recommandation pour la suite

#6 dans toute son ampleur (= rebasing F3 complet remplaçant glitch detection)
nécessite probablement 2-3 sessions dédiées :

1. **Session A : isoler le path rebasing-only et debug**.
   - Comparer step-par-step un pixel précis entre le path actuel
     (`use_legacy=false`) et un re-implementation minimaliste exactement
     calquée sur `hybrid_render` de F3 (sans BLA).
   - Trouver d'où vient la divergence.
2. **Session B : intégrer le BLA mat2 unifié (`bytecode/bla_dual.rs`) au
   pixel loop sans glitch detection**.
   - Réutilise le builder de #5 → produit Table mat2.
   - Le pixel loop évalue `b->A * z + b->B * c` puis check rebase comme F3.
3. **Session C : validation goldens + nettoyage**.
   - Tous les goldens passent.
   - Suppression `glitch.rs`, `nonconformal.rs`, params glitch obsolètes.

Sans P2.1 (golden harness) ce travail aurait été irresponsable. Maintenant
il est à risque mais traçable.

## Goldens utiles pour la suite

Goldens deep-zoom ajoutées qui exerceront fortement le pixel loop refactor :

- `mandelbrot_minibrot_1e8.png` — minibrot zoom 1e8
- `burning_ship_zoom_1e5.png` — BS zoom 1e5
- (déjà présents) `mandelbrot_perturb_1e6.png`, `burning_ship_perturb_1e3.png`,
  `tricorn_perturb_1e2.png`

Si une futur tentative #6 fait passer ces 5 goldens en mode F3-pure : objectif
atteint.
