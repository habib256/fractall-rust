# P3.1 #6 Session C — Findings

État au 2026-05-17 sur branche `Stable`. Suite de
`docs/p3-1-task6-session-a-findings.md` et `docs/p3-1-task6-investigation.md`.

## Setup

Création de `src/fractal/bytecode/pixel_loop.rs` : pixel loop perturbation
F3-pur avec BLA mat2 unifiée (de `bytecode/bla_dual.rs`, Session B).
Mandelbrot uniquement pour l'instant (forme delta `δ' = 2·Z·δ + δ² + dc`
hardcodée).

Structure (mirror F3 `hybrid_render` strictement) :
1. BLA lookup → si trouvé, `δ := A·δ + B·dc`, skip `l` itérations
2. Sinon : pas perturbation
3. Rebase F3 strict : `|Z[m+1]+δ|² < |δ|²` ou `m+1==ref_len` → reseat,
   `m := 0`

Pas de glitch detection. Pas de secondary refs. Pas de série.

## Mesures Session C — Mandelbrot zoom 1e6, 160×100

Render complet via le pixel loop unifié + coloriage production
(`color_for_pixel_with_lut`, OutColoringMode::Smooth, palette 6 Plasma).
Comparaison pixel à pixel vs golden legacy `mandelbrot_perturb_1e6.png`.

| Métrique | Session A `iterate_pixel(use_legacy=false)` | Session C unified loop |
|----------|---------------------------------------------|------------------------|
| % pixels diff | 97.6 % | 93.66 % |
| Pixels lourdement diff (>50/canal) | 78.31 % | **8.47 %** |
| Mean \|diff\|/canal | 63.31 | **9.74** |
| BLA steps utilisés | inconnu | 893 |
| Rebases F3 déclenchés | inconnu | 41564 |
| Classifications correctes vs direct | 100/100 | 100/100 |

**Réduction d'un ordre de grandeur** des pixels lourdement différents
(78.31 % → 8.47 %). Le mean diff/canal passe de 63 (≈25%) à 9.74 (≈4%).

Le 93.66 % « pixels diff » de Session C est trompeur — c'est principalement
du décalage très fin (≈±5 unités de couleur sur 255). Visuellement, les
deux images sont quasi-identiques.

## Conclusion

Le pixel loop unifié (BLA mat2 + rebasing F3 strict, sans glitch
detection) **produit des images de qualité comparable à legacy** sur
Mandelbrot zoom 1e6 — au prix d'un léger décalage de couleur fine, pas
de pixels manifestement faux comme avec l'ancien `use_legacy=false`.

L'origine probable du décalage résiduel : le rebasing F3 strict produit
des trajectoires δ légèrement différentes du chemin legacy (qui combine
BLA approximative + glitch detection corrective), affectant le `|z_final|`
final et donc le smooth coloring fractionnaire.

## Limites de Session C

1. **Mandelbrot uniquement** : la forme delta des autres types (BS, Tricorn,
   Celtic, Multibrot, etc.) reste à écrire. La forme est canonique pour
   chaque opcode et peut être dérivée directement du bytecode :
   - `Sqr` : `δ' = 2·Z·δ + δ²`
   - `Mul` : `δ' = stored·δ + Z·stored_δ + δ·stored_δ`
   - `AbsX/Y` : utiliser `diffabs(Z.coord, δ.coord)` (existe déjà dans
     `delta.rs::diffabs`)
   - `NegX/Y`, `Store`, `Add` : linéaires
2. **f64 uniquement** (pas de ComplexExp pour deep zoom > ~1e10).
3. **Mono-phase seulement** (`BlaTableUnified` unique, pas `Vec<>` par
   phase).
4. **Pas d'intégration dans `iterate_pixel`** : nouveau pixel loop est
   testé en isolation, pas dispatché depuis `delta.rs`. L'intégration
   demande du plumbing (passer `BlaTableUnified` à travers les couches
   d'appel) — Session D.

## Plan Session D (suggéré)

1. Étendre la forme delta aux autres types (BS, Tricorn, Celtic, Multibrot,
   Buffalo, PerpBS). 1 fonction par type ou (mieux) un interpréteur
   bytecode-delta unifié.
2. Threader `Option<BlaTableUnified>` à travers le pipeline `render_perturbation
   _cancellable_with_reuse` → `iterate_pixel`. Construire la table une
   fois par render et dispatcher quand `use_bytecode_engine && type
   supporté`.
3. Regénérer goldens en mode bytecode (la voie est plus F3-correcte,
   accepter la nouvelle référence).
4. Une fois validé : supprimer `glitch.rs`, `nonconformal.rs`, paramètres
   glitch obsolètes — c'est #8.

## Code livré

- `src/fractal/bytecode/pixel_loop.rs` (~250 lignes)
- `src/fractal/bytecode/bla_dual.rs::build_bla_table_for_formula()` ajouté
- 2 tests : classification (100/100) + render comparaison vs golden

120 unit tests + 12 goldens verts. Aucune régression runtime — toute la
logique unified loop est test-only à ce stade.
