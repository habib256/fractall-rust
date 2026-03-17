# Perturbation Theory for Deep Zoom Fractals

Documentation technique basee sur les travaux de Fraktaler-3 (Claude Heiland-Allen),
K.I. Martin (SuperFractalThing), Pauldelbrot (glitch detection), Zhuoran (rebasing + BLA),
et rust-fractal-core (implementation Rust de reference).

## References

- [Fraktaler-3](https://fraktaler.mathr.co.uk/) - Implementation de reference (C++)
- [rust-fractal-core](https://github.com/rust-fractal/rust-fractal-core) - Implementation Rust de reference (verifie jusqu'a E50000)
- [Deep Zoom Theory and Practice](https://mathr.co.uk/blog/2021-05-14_deep_zoom_theory_and_practice.html) - Article fondateur
- [Deep Zoom Theory and Practice (again)](https://mathr.co.uk/blog/2022-02-21_deep_zoom_theory_and_practice_again.html) - BLA + rebasing
- [At the Helm of the Burning Ship](https://mathr.co.uk/web/helm.html) - Paper EVA 2019
- [The Burning Ship](https://mathr.co.uk/blog/2018-01-04_the_burning_ship.html) - Perturbation pour Burning Ship
- [Fractal Wiki - Perturbation theory](https://fractalwiki.org/wiki/Perturbation_theory)
- [Wikibooks - Fractals/perturbation](https://en.wikibooks.org/wiki/Fractals/perturbation)

---

## 1. Principe fondamental

### Probleme

Pour un zoom de facteur `10^N`, il faut `N * log2(10) ≈ 3.32N` bits de precision.
A `10^300`, cela represente ~1000 bits par nombre complexe, rendant le calcul classique
extremement lent (chaque multiplication est O(bits^2) ou O(bits·log(bits))).

### Solution: Perturbation

Au lieu de calculer chaque pixel en haute precision, on calcule **une seule orbite de reference**
en haute precision au centre de l'image, puis chaque pixel est exprime comme un petit delta
par rapport a cette reference.

**Notations**:
- `Z_n` : orbite de reference haute precision (calculee en GMP/MPFR)
- `z_n` : delta de perturbation basse precision (f64 ou FloatExp)
- `C` : point de reference (centre de l'image)
- `c` : offset du pixel par rapport au centre (dc)

L'orbite reelle du pixel est `Z_n + z_n`, et le point C du pixel est `C + c`.

### Derivation pour Mandelbrot (z^2 + c)

L'iteration complete:
```
(Z_n + z_n)^2 + (C + c) = Z_n^2 + 2·Z_n·z_n + z_n^2 + C + c
```

L'orbite de reference satisfait: `Z_{n+1} = Z_n^2 + C`

En soustrayant:
```
z_{n+1} = (Z_n + z_n)^2 + (C + c) - (Z_n^2 + C)
        = 2·Z_n·z_n + z_n^2 + c
```

**Formule de perturbation**:
```
z_{n+1} = 2·Z_n·z_n + z_n^2 + c
```

- `2·Z_n·z_n` : terme lineaire (dominant quand z_n est petit)
- `z_n^2` : terme non-lineaire (negligeable quand z_n << Z_n)
- `c` : contribution du pixel

### Initialisation

- **Mandelbrot**: `z_0 = 0` (car l'orbite commence a 0, delta = 0), `c = dc` (offset du pixel)
- **Julia**: `z_0 = c` (car le point de depart est C + c), pas de terme `c` dans l'iteration

---

## 2. Representation en FloatExp / ComplexExp

Pour les zooms intermediaires (~10^13 a ~10^15), les deltas `z_n` peuvent etre trop petits
pour f64 (denormalisations). La representation `FloatExp` utilise une mantisse f64 normalisee
et un exposant entier separe:

```
valeur = mantissa * 2^exponent
```

Ou `mantissa ∈ [0.5, 1.0)` (normalise). Cela permet de representer des nombres aussi petits
que `2^{-2^31}` sans perte de precision sur la mantisse.

**Operations**: Addition, multiplication et norme sont implementees en manipulant separement
mantisse et exposant, avec renormalisation apres chaque operation.
La methode `reduce()` est appelee periodiquement (toutes les 256 iterations) pour
renormaliser la mantisse et prevenir la derive de precision.

**ComplexExp** associe deux `FloatExp` (partie reelle et imaginaire). La methode
`mul_signed(sign_re, sign_im)` applique la transformation de signe pour le Burning Ship.

### Comparaison avec rust-fractal-core

rust-fractal-core utilise une approche similaire mais structurellement differente:

| Aspect | fractall-rust | rust-fractal-core |
|--------|---------------|-------------------|
| Type | `ComplexExp` (re: FloatExp, im: FloatExp) | `ComplexExtended` (Complex<f64> mantissa + i32 exponent) |
| Stockage | Exposant separe par composante (re, im) | Exposant unique pour le nombre complexe |
| Iteration rapide | Bascule f64 → ComplexExp selon extended_iterations | "Scaled doubles": mantisse `ComplexFixed<f64>` + `scale_factor_delta` via `ldexp` |
| Seuil extended | `|re|` ou `|im| < 1e-300` → marque comme extended iteration | `|re|` ou `|im| < 1e-300` → marque comme extended iteration |

L'approche "scaled doubles" de rust-fractal-core pre-calcule `scale_factor_delta = 1.0.ldexp(exponent)` et travaille
avec des `Complex<f64>` multiplies par ce facteur. Cela evite le surcout de l'arithmetique FloatExp pour la majorite
des iterations, ne basculant en `ComplexExtended` complet qu'aux iterations marquees comme "extended".

---

## 3. Orbite de reference (GMP/MPFR)

### Calcul de precision

La precision necessaire (en bits) est calculee par la formule de Fraktaler-3:
```
bits = max(24, 24 + floor(log2(zoom * height)))
```
Clampee dans l'intervalle `[128, 8192]`.

rust-fractal-core utilise une formule plus simple: `bits = max(64, -radius.exponent + 64)`.

### Stockage

L'orbite de reference est stockee sous plusieurs formes:
- **GMP** (`Complex` rug): haute precision pour les calculs perturbation GMP
- **f64** (`Complex64`): basse precision pour les calculs perturbation f64/BLA
- **ComplexExp**: precision etendue pour les zooms intermediaires
- **extended_iterations** (`Vec<u32>`): indices des iterations ou f64 sous-depasse (`< 1e-300`)

rust-fractal-core stocke de meme `reference_data` (f64), `reference_data_extended` (ComplexExtended),
et `extended_iterations` (Vec<usize>), avec en plus `high_precision_data` (snapshots GMP a intervalles)
pour la correction de glitchs econome en memoire.

### Stockage haute precision a intervalles

Pour les orbites longues (>1M iterations), stocker toutes les valeurs GMP est prohibitif en memoire.
L'orbite peut stocker les donnees haute precision a intervalles (`data_storage_interval`):
- `< 1M iterations`: interval = 1 (toutes les iterations)
- `1M - 10M iterations`: interval = 10
- `> 10M iterations`: interval = 100

Lors de la correction de glitch, le point de depart le plus proche est retrouve par
`iteration / interval`. rust-fractal-core utilise cette meme technique dans
`get_glitch_resolving_reference()`.

### Cache

L'orbite est mise en cache (`ReferenceOrbitCache`) entre les rendus pour eviter de la recalculer
si le centre et le type n'ont pas change.

---

## 4. Bilinear Approximation (BLA)

### Principe

Parfois, `l` iterations a partir de `n` peuvent etre approximees par une fonction bilineaire:
```
z_{n+l} ≈ A_{n,l}·z_n + B_{n,l}·c
```

C'est valide quand la partie non-lineaire (z_n^2 et termes superieurs) est negligeable
par rapport a la precision du type de donnees basse precision.

**Note**: rust-fractal-core n'utilise **pas** de BLA. Il s'appuie uniquement sur l'approximation
par series pour sauter des iterations. fractall-rust utilise les deux techniques (BLA + series).

### Representation des coefficients

**Fraktaler-3 utilise des matrices 2×2 reelles** pour A et B, meme pour le Mandelbrot conforme.
Cela permet de gerer uniformement les formules conformes et non-conformes.

Dans fractall-rust:
- **Conforme** (Mandelbrot, Julia, Multibrot): nombres complexes pour A, B (equivalent a [[a,-b],[b,a]])
- **Non-conforme** (Tricorn, Burning Ship): matrices 2×2 reelles (`Matrix2x2`)

### Coefficients single-step (niveau 0)

Pour Mandelbrot (`z^2 + c`):
```
A_{n,1} = 2·Z_n     (derivee de l'iteration)
B_{n,1} = 1          (coefficient du terme c)
```

Pour Multibrot (`z^d + c`):
```
A_{n,1} = d·Z_n^{d-1}
B_{n,1} = 1
```

### Rayon de validite

Le BLA est valide quand `|z_n| < R_{n,l}`.

**Formule single-step (conforme)**:
```
R_{n,1} = ε·|A_{n,1}|
```
ou `ε` est le seuil BLA (typiquement 1e-8).

**Formule single-step (non-conforme)**:
```
R_{n,1} = ε·inf|A_{n,1}| - sup|B_{n,1}|·|c| / inf|A_{n,1}|
```
ou `inf|M|` et `sup|M|` sont les plus petite et plus grande valeurs singulieres de la matrice M.

### Fusion (merging)

Deux BLA consecutifs T_x (saute l_x iterations) et T_y (saute l_y iterations) peuvent etre
fusionnes en T_z (saute l_x + l_y iterations):

**Coefficients**:
```
A_z = A_y · A_x       (multiplication matricielle ou complexe)
B_z = A_y · B_x + B_y
```

**Rayon de validite**:
```
R_z = max{0, min{R_x, R_y - sup|B_x|·|c| / sup|A_x|}}
```

Pour les formules conformes, `sup|M| = |M|` (norme complexe) et la formule se simplifie.

### Table BLA multi-niveaux

La table BLA est construite de maniere hierarchique:
- **Niveau 0**: M noeuds, chacun saute 1 iteration
- **Niveau 1**: M/2 noeuds, chacun saute 2 iterations (fusion de paires adjacentes)
- **Niveau k**: M/2^k noeuds, chacun saute 2^k iterations
- **Total**: O(M) noeuds (somme geometrique)

Un noeud dummy est insere a l'index 0 du niveau 1 pour aligner `level_nodes[n]` avec l'iteration `n`.

### Lookup

Lors du calcul d'un pixel, a chaque iteration `n`:
1. Chercher le BLA de plus haut niveau valide (`|z_n| < R_{n,l}`)
2. Appliquer: `z_{n+l} = A_{n,l}·z_n + B_{n,l}·c`
3. Sauter `l = 2^level` iterations
4. Si aucun BLA valide, faire une iteration de perturbation standard

---

## 5. Rebasing

### Probleme

Quand l'orbite du pixel (`Z_n + z_n`) passe pres d'un point critique (0 pour Mandelbrot),
le delta `z_n` peut devenir aussi grand que `Z_n`, rendant la perturbation imprecise (glitch).

### Solution

**Condition de rebasing**: Quand `|Z_m + z_n|^2 < |z_n|^2 * H`
(le point pixel est plus pres de l'origine que son delta par rapport a la reference).

**Hysteresis** (`H`): fractall-rust utilise `H = 0.5` (REBASE_HYSTERESIS) pour eviter
les rebases oscillants. L'iteration n'est rebasee que si z_curr² < delta² * 0.5.
De plus, le rebasing est desactive quand `z_ref² < 1e-20` (pres des zeros de l'orbite reference).

rust-fractal-core utilise une condition plus simple: `z_norm < delta_norm_sqr` sans hysteresis,
et rebase toujours a l'iteration 0.

**Procedure**:
1. Remplacer `z_n` par `Z_m + z_n` (delta absolu = position reelle du pixel)
2. Reset `m = 0` (recommencer depuis le debut de l'orbite de reference)

Avec le rebasing, une seule orbite de reference suffit (au lieu de multiples references).
Les glitchs sont **evites** plutot que **detectes**.

### Rebasing stride

Le stride de rebasing (frequence de verification) est configurable via la variable
d'environnement `FRACTALL_PERTURB_REBASE_STRIDE` (defaut 1, plage 1-64).
Un stride > 1 reduit le surcout de verification au prix d'une detection retardee.

### Rebasing pour formules hybrides

Pour les formules hybrides (combinaison de plusieurs iterations en boucle, ex: "M,BS,M,M"),
le rebasing necessite une orbite de reference par phase dans la boucle. Lors du rebasing,
la phase est mise a jour: `phase = (phase + n) % cycle_period`.

---

## 6. Detection de glitchs (Pauldelbrot)

### Critere

Malgre le rebasing, certains pixels peuvent encore etre imprecis. Le critere de Pauldelbrot
detecte ces "glitchs":

```
|z_n|^2 > G^2 · max(|Z_n|^2, 1e-6)
```

ou `G` est la tolerance de glitch (typiquement 1e-4).

Le denominateur utilise `max(|Z_n|^2, 1e-6)` au lieu de `|Z_n|^2 + 1.0` pour eviter
de masquer les glitchs pres de l'origine (bug corrige).

### Comparaison avec rust-fractal-core

rust-fractal-core utilise une detection de glitch differente:
- **Inline rebasing**: `z_norm < delta_norm_sqr` → rebase a iteration 0 immediatement
- **Post-render resolution**: les pixels encore glitches apres le rendu sont regroupes par iteration,
  et de nouvelles references sont calculees a partir des snapshots haute precision
- **Pas de seuil Pauldelbrot**: la detection est implicite via le rebasing

fractall-rust combine rebasing (prevention) + Pauldelbrot (detection) + clusters (correction).

### Tolerance adaptative

La tolerance est calculee en continu selon le niveau de zoom:
```
tolerance = 10^(-5 + zoom_level * 0.1)    // clampee a [1e-6, 1e-1]
zoom_level = log10(4.0 / pixel_size)
```

Valeurs typiques:
- Zoom peu profond (zoom_level 0-10): `G ≈ 1e-5`
- Moyen (10-20): `G ≈ 1e-4 a 1e-3`
- Profond (20-40): `G ≈ 1e-3 a 1e-1`
- Extreme (>40): `G = 1e-1`

### Correction des glitchs

Les pixels detectes comme glitches sont regroupes en clusters (flood-fill).
Pour chaque cluster suffisamment grand (> `min_cluster_size`, defaut 100), une **reference secondaire**
est calculee au centre du cluster, et les pixels sont recalcules avec cette nouvelle reference.

Le processus est recursif (jusqu'a `max_secondary_refs` iterations, defaut 3).
Si plus de 30% des pixels sont encore glitches apres correction, l'image entiere est
recalculee en GMP complet (fallback).

rust-fractal-core utilise une approche similaire via `get_glitch_resolving_reference()`:
une nouvelle reference est creee en ajoutant le delta du pixel glitche au snapshot haute precision
le plus proche. Le code actuel de rust-fractal-core a la resolution de glitch commentee.

---

## 7. Non-conforme: Burning Ship et Tricorn

### Burning Ship: z' = (|Re(z)| + i|Im(z)|)^2 + c

La fonction `|·|` (valeur absolue) rend l'iteration **non-analytique** (non-conforme).
Les derivees ne sont plus representables par un seul nombre complexe.

**Jacobienne** au point de reference Z = X + iY:
```
J = | 2X              -2Y            |
    | 2·sign(X)·|Y|   2·|X|·sign(Y) |
```

**Comportement par quadrant**:
- 1er quadrant (X>=0, Y>=0): J = [[2X,-2Y],[2Y,2X]] -- **conforme** (= mult. complexe par 2Z)
- 3eme quadrant (X<0, Y<0): J = [[2X,-2Y],[2Y,2X]] -- **conforme** (meme structure)
- 2eme quadrant (X<0, Y>=0): J = [[2X,-2Y],[-2Y,-2X]] -- **anti-conforme**
- 4eme quadrant (X>=0, Y<0): J = [[2X,-2Y],[-2Y,-2X]] -- **anti-conforme**

Pour le BLA, il est **imperatif** d'utiliser des matrices 2×2 (et non des nombres complexes)
car la fusion de coefficients par multiplication complexe est **incorrecte** dans les quadrants
anti-conformes. Fraktaler-3 utilise `mat2<real>` pour tous les BLA.

**Fonction diffabs** (credit: laser blaster, fractalforums.com):
```
diffabs(c, d) = |c + d| - |c|
```
Avec analyse de cas pour eviter l'annulation catastrophique:
```
diffabs(c, d) = {
  d         si c >= 0 et c + d >= 0
  -(2c + d) si c >= 0 et c + d < 0
  2c + d    si c < 0 et c + d > 0
  -d        si c < 0 et c + d <= 0
}
```

rust-fractal-core utilise la meme fonction `diff_abs()` pour le Burning Ship, avec
le const generic `FRACTAL_TYPE = 1` pour activer le path non-conforme.

**Contrainte de pliage** (folding lines): Le rayon de validite BLA est contraint par la distance
aux axes (Re=0, Im=0) car la valeur absolue "plie" le plan a ces endroits:
```
R = min(R_nonlineaire, |Re(Z)|/2, |Im(Z)|/2)
```
Le facteur `/2` est un facteur de securite (fudge factor, utilise dans Fraktaler-3).

### Tricorn (Mandelbar): z' = conj(z)^2 + c

La conjugaison rend l'iteration non-conforme. L'expansion:
```
z' = (X - iY)^2 + C = (X^2 - Y^2) + i(-2XY) + C
```

**Jacobienne**:
```
J = | 2X   -2Y |
    | -2Y  -2X |
```

Toujours anti-conforme (pas de dependance au quadrant, contrairement au Burning Ship).

---

## 8. Approximation par series (Taylor)

### Principe

Au lieu de calculer les iterations une par une, on approxime les premiers N iterations
par un polynome de Taylor en `c` (l'offset du pixel):

```
z_N(c) ≈ a_N·c + b_N·c^2 + c_N·c^3 + d_N·c^4
```

fractall-rust supporte des ordres elevees (jusqu'a MAX_SERIES_ORDER = 32 termes),
avec evaluation par la methode de Horner.

### Ordre adaptatif

L'ordre de la serie est choisi automatiquement (`compute_adaptive_series_order`) selon la profondeur:
- Shallow (`< 10^8`): ordre 4
- Medium (`10^8 - 10^15`): ordre 6-8
- Deep (`10^15 - 10^30`): ordre 10-16
- Ultra-deep (`> 10^30`): ordre 16-24
- Bonus iterations: +4 pour >100k iterations, +2 pour >10k

### Relations de recurrence (Mandelbrot)

```
a_{n+1} = 2·Z_n·a_n + 1         (a_0 = 0)
b_{n+1} = 2·Z_n·b_n + a_n^2     (b_0 = 0)
c_{n+1} = 2·Z_n·c_n + 2·a_n·b_n (c_0 = 0)
d_{n+1} = 2·Z_n·d_n + 2·a_n·c_n + b_n^2 (d_0 = 0)
```

### Relations de recurrence (Julia)

```
a_{n+1} = 2·Z_n·a_n             (a_0 = 1)
b_{n+1} = 2·Z_n·b_n + a_n^2     (b_0 = 0)
c_{n+1} = 2·Z_n·c_n + 2·a_n·b_n (c_0 = 0)
d_{n+1} = 2·Z_n·d_n + 2·a_n·c_n + b_n^2 (d_0 = 0)
```

Difference: pas de `+1` pour Julia, et `a_0 = 1` au lieu de 0.

### Estimation d'erreur et terminaison

L'erreur de troncature est estimee heuristiquement:
```
erreur ≈ |d|^2/|c| · |dc|^5
```
Le skip est accepte si `erreur < tolerance` et si le dernier terme est petit
par rapport aux precedents (`term_ratio < 0.5`).

La verification d'overflow/NaN porte sur **tous** les coefficients (a, b, c, d), pas seulement `a`.

### Validation par sondes (probe-based)

**fractall-rust et rust-fractal-core** utilisent tous deux une validation par sondes pour
determiner le nombre d'iterations que la serie peut fiablement sauter.

**Principe**: On place une grille de points-sonde sur l'image (configurable, 2-8 par axe),
on calcule leur orbite par perturbation classique, puis on compare avec le resultat de la serie.
Le skip valide est le maximum d'iterations ou la serie reste fidele a la perturbation.

**Mode tile** (`validate_series_with_probes_tiled`): Chaque pixel peut sauter un nombre
different d'iterations, interpole a partir des sondes les plus proches:
- Pour chaque cellule 2×2 de sondes, le minimum des skips valides est utilise (conservateur)
- Validation en deux phases: coins d'abord (tolerance large), puis sondes restantes
- Reduction periodique (~250 iterations) pour maintenir la precision

rust-fractal-core utilise une approche equivalente avec `probe_sampling` configurable
et support du mode tile per-pixel.

### Limitation pour Burning Ship

Les series de Taylor traditionnelles ne fonctionnent **pas** pour le Burning Ship
car la valeur absolue cree des discontinuites dans les derivees.
Pour les "abs variations", on utilise deux series bivariees reelles en Re(c) et Im(c)
(pas des series complexes). En pratique, le BLA est prefere car plus general.

---

## 9. Estimation de distance (Dual Numbers)

### Principe

On calcule simultanement l'orbite et sa derivee par rapport au pixel:
```
dz_{n+1}/dc = 2·Z_n·(dz_n/dc) + 2·z_n·(dz_n/dc) + 1
```

La distance au bord de l'ensemble est estimee par:
```
distance ≈ |z_n| · log|z_n| / |dz_n/dc|
```

rust-fractal-core supporte l'estimation de distance via un tracking de Jacobien
(DATA_TYPE 1 ou 3 via const generics), integre directement dans la boucle d'iteration.
fractall-rust utilise des types dedies (DualComplex, ExtendedDualComplex).

### DualComplex

Pour les formules conformes (Mandelbrot), on utilise `DualComplex`:
- `value`: la valeur z_n (Complex64)
- `dual`: la derivee dz/dc (Complex64)

### ExtendedDualComplex

Pour les formules non-conformes et la detection d'interieur, on utilise `ExtendedDualComplex`
avec 5 composantes:
- `value`: z_n
- `dual_re`: dz/d(Re(c))
- `dual_im`: dz/d(Im(c))
- `dual_z1_re`: d^2z/d(Re(c))dz pour l'interieur
- `dual_z1_im`: d^2z/d(Im(c))dz pour l'interieur

La fonction `mul_signed(sign_re, sign_im)` applique la transformation de signe
a toutes les composantes simultanement (necessaire pour le Burning Ship).

---

## 10. Detection d'interieur

### Principe

Un pixel est a l'interieur de l'ensemble si son orbite converge vers un cycle attractif.
On detecte cela en cherchant un point fixe du second ordre:

```
|dz/dz_0| < 1
```

Si le multiplicateur (norme de la derivee de l'iteration par rapport au point de depart)
est inferieur a 1, l'orbite est dans un bassin d'attraction.

### Seuil

Le seuil par defaut est `0.001`. Un pixel avec `|dz/dz_0| < seuil` est marque comme interieur
et colore en noir (ou selon le mode de coloration).

---

## 11. Optimisations de la boucle d'iteration

### Iterations par lot (batch)

fractall-rust utilise des lots de `BATCH_SIZE = 256` iterations pour le path f64 rapide,
amortissant le surcout des verifications (bailout, glitch, rebasing).

rust-fractal-core utilise une approche similaire: `iterations_before_check = 400 / power`
(soit 200 pour Mandelbrot standard). De plus, quand `delta_current.exponent > -500`,
les checks d'escape sont effectues; sinon ils sont sautes (valeurs trop petites pour s'echapper).

### Extended iterations

Les deux implementations tracent les iterations ou les valeurs f64 sous-depassent (< 1e-300).
A ces iterations, l'arithmetique bascule de f64 pur vers la representation etendue
(ComplexExp pour fractall-rust, ComplexExtended pour rust-fractal-core).

L'orbite de reference stocke ces indices dans `extended_iterations: Vec<u32>` pour que
la boucle pixel sache quand basculer sans verifier a chaque iteration.

### Puissances generiques (Multibrot)

rust-fractal-core supporte les puissances Mandelbrot arbitraires via des coefficients
du triangle de Pascal (`const FRACTAL_POWER`), avec const generics pour specialiser
le code a la compilation. fractall-rust supporte le Multibrot mais est principalement
optimise pour la puissance 2.

---

## 12. Pipeline complet

### Vue d'ensemble

```
1. Calcul de l'orbite de reference (GMP, haute precision)
   ↓
2. Construction de la table BLA (multi-niveaux, matrices 2×2)
   ↓
3. Construction de la table de series (optionnel, avec validation par sondes)
   ↓
4. Pour chaque pixel (en parallele avec rayon):
   a. Calculer dc = offset du pixel par rapport au centre
   b. Appliquer le skip de serie (global ou per-pixel en mode tile)
   c. Initialiser delta (z_0 = 0 pour Mandelbrot, z_0 = dc pour Julia)
   d. Boucle d'iteration (par lots de 256):
      i.   Essayer un skip par BLA (plus grand saut valide)
      ii.  Si pas de BLA valide, faire une iteration de perturbation
      iii. Verifier le rebasing: |Z_m + z_n|^2 < |z_n|^2 * 0.5
      iv.  Verifier le bailout: |Z_m + z_n|^2 > bailout^2
      v.   Verifier le glitch: |z_n|^2 > G^2 · max(|Z_n|^2, 1e-6)
   e. Retourner: iteration, z_final, glitched, distance
   ↓
5. Correction des glitchs (clustering + references secondaires, jusqu'a 3 rounds)
   ↓
6. Fallback GMP si >30% pixels glitches
   ↓
7. Coloration
```

rust-fractal-core utilise un pipeline simplifie (pas de BLA, pas de correction post-render active):
```
1. Orbite de reference (GMP)
   ↓
2. Approximation par series + validation par sondes
   ↓
3. Iteration perturbation par pixel (avec rebasing inline)
   ↓
4. [Correction glitchs - actuellement desactivee dans le code]
   ↓
5. Coloration
```

### Dispatch automatique (AlgorithmMode::Auto)

```
pixel_size > ~1e-5   → GPU f32 standard (rapide, pas de perturbation)
pixel_size > ~1e-13  → CPU f64 standard (rapide, pas de perturbation)
pixel_size ~1e-13    → perturbation f64/ComplexExp + BLA + series
pixel_size < ~4e-16  → reference GMP (zoom > 1e16)
```

### Precision GMP

La precision est calculee automatiquement:
```
bits = max(24, 24 + floor(log2(zoom * height)))
```
Clampee dans `[128, 8192]`. Les coordonnees haute precision sont stockees en String
dans `FractalParams` pour eviter la perte de precision lors du passage en f64.

---

## 13. Comparaison des implementations

### fractall-rust vs Fraktaler-3

| Aspect | Fraktaler-3 | fractall-rust |
|--------|-------------|---------------|
| BLA coefficients | Toujours `mat2<real>` | Complexes (conforme) ou `Matrix2x2` (non-conforme) |
| Rebasing | Toujours actif | Actif avec stride configurable + hysteresis |
| Series approx | 2 series bivariees reelles (abs variations) | Series complexes uniquement |
| Formules hybrides | Support complet | Support partiel (Hybrid BLA) |
| GPU | OpenGL compute shaders | wgpu (Vulkan/Metal/DX12, f32 uniquement) |
| Precision | floatexp + softfloat + mpfr | FloatExp + rug (GMP/MPFR) |

### fractall-rust vs rust-fractal-core

| Aspect | rust-fractal-core | fractall-rust |
|--------|-------------------|---------------|
| Skip iterations | Series uniquement (pas de BLA) | BLA + series |
| Representation delta | Scaled doubles (mantisse f64 + scale_factor) | ComplexExp (mantisse + exposant par composante) |
| Glitch detection | Rebasing inline (z_norm < delta_norm) | Pauldelbrot (|z|² > G² · |Z|²) + rebasing |
| Glitch correction | Secondary refs depuis snapshots HP (desactive) | Clustering flood-fill + refs secondaires (actif) |
| Series validation | Sondes + mode tile | Sondes + mode tile |
| Puissances | Generique via triangle de Pascal (const generic) | Principalement puissance 2 |
| Types fractales | Mandelbrot, Burning Ship | Mandelbrot, Julia, BurningShip, Tricorn |
| Batch iterations | 400/power iterations | 256 iterations |
| Interior detection | Non | Oui (ExtendedDualComplex) |
| Distance estimation | Jacobien inline (const generic) | DualComplex / ExtendedDualComplex |
| Profondeur verifiee | > E50000 | Non specifie |
| GPU | Non | wgpu (perturbation.wgsl) |

---

## 14. Glossaire

| Terme | Definition |
|-------|-----------|
| **Perturbation** | Technique de calcul des orbites par rapport a une reference |
| **BLA** | Bilinear Approximation - saut de plusieurs iterations par approximation lineaire |
| **Rebasing** | Reset de la reference quand le pixel approche un point critique |
| **Glitch** | Pixel mal calcule car le delta est devenu trop grand |
| **Pauldelbrot** | Critere de detection de glitch base sur |z_n|/|Z_n| |
| **diffabs** | Fonction |c+d|-|c| avec analyse de cas pour eviter l'annulation |
| **FloatExp / ComplexExp** | Representation mantisse+exposant pour les tres petits nombres |
| **ComplexExtended** | Equivalent rust-fractal-core de ComplexExp (mantisse Complex<f64> + exposant i32) |
| **Scaled doubles** | Technique ou un facteur d'echelle f64 est pre-calcule via ldexp |
| **Extended iteration** | Iteration ou les valeurs f64 sous-depassent (< 1e-300), necessitant FloatExp |
| **Conforme** | Transformation preservant les angles (representable par mult. complexe) |
| **Non-conforme** | Transformation ne preservant pas les angles (necessite matrice 2×2) |
| **Valeur singuliere** | sup\|M\| et inf\|M\| - normes operateur de la matrice |
| **Folding line** | Ligne de pliage (axe Re=0 ou Im=0) ou la valeur absolue cree une discontinuite |
| **Probe / Sonde** | Point-test utilise pour valider l'approximation par series |
| **Tile mode** | Mode ou chaque pixel peut sauter un nombre different d'iterations (per-region) |
