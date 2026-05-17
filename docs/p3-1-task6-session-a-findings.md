# P3.1 #6 Session A — Findings

État au 2026-05-17 sur branche `bytecode`. Suite de
`docs/p3-1-task6-investigation.md`.

## Setup

Création d'un module `src/fractal/perturbation/debug_pure_f3.rs` (test-only)
qui implémente une perturbation Mandelbrot **strictement F3-pure** : rebasing
seul, *aucune* BLA, *aucune* glitch detection, *aucune* feature additionnelle.
~100 lignes. Validation par 3 tests :
- intérieur/extérieur correctement classifiés
- rebasing déclenché en deep zoom
- consistance avec itération Mandelbrot directe à zoom=1

## Test critique : F3-pure vs `iterate_pixel(use_legacy=false)`

Sur la location Mandelbrot zoom 1e6 (`mandelbrot_perturb_1e6` du corpus
golden), 100 pixels échantillonnés en grille 10×10.

| Métrique | F3-pure vs direct f64 | F3-pure vs iterate_pixel(no_legacy) |
|----------|----------------------|-------------------------------------|
| Classifications matchent | 100/100 | 100/100 |
| Iter counts identiques | 54/100 | **8/100** |
| Mean \|diff\| iter count | 3.27 | **67.26** |
| Max \|diff\| iter count | 80 | **199** |

## Conclusions

1. **Le rebasing F3 est algorithmiquement correct**. La classification
   escaped/interior matche à 100% avec une itération directe f64 — preuve
   que l'algorithme rebasing seul (sans BLA, sans glitch detection) classifie
   bien chaque pixel à zoom 1e6.

2. **`iterate_pixel(use_legacy=false)` classifie correctement aussi** —
   100/100 vs F3-pure. Donc la branche "F3-pure" de production n'a pas
   de bug algorithmique de classification.

3. **MAIS** : les iteration counts sont décalés de **~67 itérations en
   moyenne** entre F3-pure (sans BLA) et iterate_pixel (avec BLA). Avec
   un max de 199 itérations d'écart.

4. **C'est la source du gap 95% pixels diff vs legacy** dans les goldens.
   Le smooth coloring `n + 1 - log2(log2(|z|))` est très sensible au n
   exact : un shift de 67 itérations sur une palette à 40 répétitions
   produit des couleurs complètement différentes.

## Origine probable du shift

La BLA saute des blocs d'itérations (l = 2^level) puis check bailout à la
fin du bloc. Quand le pixel échappe au milieu du bloc, l'itération
rapportée est celle de la fin du bloc, pas la vraie itération d'évasion.
Plus de BLA = plus de shift cumulé.

La glitch detection legacy semble *corriger* ce shift (peut-être en
forçant des micro-rebases qui recalent le compte) — ce qui expliquerait
pourquoi les goldens (legacy) ont des iter counts cohérents alors que
no-legacy a un drift.

## Implications pour la suite de #6

**Le refactor F3-pure n'a pas besoin de toucher au rebasing** (déjà
correct). Le vrai travail est sur la BLA :

- **Option A** : remplacer la BLA actuelle par le builder mat2 unifié
  de `bytecode/bla_dual.rs` (déjà construit en #5). Le rayon de validité
  F3 (`r = ε·|W|/sup(A)`) est plus strict que le nôtre actuel, ce qui
  devrait réduire le shift.
- **Option B** : changer la stratégie BLA pour ne **jamais sauter d'itérations
  près du bailout** (check à mi-bloc).
- **Option C** : accepter que F3-pure = goldens nouveaux. Régénérer les
  goldens en mode F3-pure et les figer comme la nouvelle référence.

Option A est la plus alignée avec l'objectif F3 et débloque le cleanup
(#8 : retirer `nonconformal.rs`). C'est le chemin pour Session B.

## Code livré

- `src/fractal/perturbation/debug_pure_f3.rs` (~280 lignes test-only)
- 5 tests verts dans `cargo test --release --bin fractall-cli debug_pure_f3`

Aucun changement runtime, aucune régression.
