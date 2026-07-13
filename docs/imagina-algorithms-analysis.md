# Imagina/Algorithms — analyse de référence (2026-07-14)

> Source : https://github.com/ImaginaFractal/Algorithms (GPL-3, clone
> `~/src/Imagina-Algorithms`, HEAD `50c0b57`). Auteur : Zhuoran Yu (5E-324),
> l'inventeur du **rebasing** (que F3 a adopté, et nous via F3) et l'auteur
> d'Imagina, réputé le renderer Mandelbrot CPU le plus rapide. 1169 lignes,
> 5 évaluateurs Mandelbrot-only, f64 (pas de tier floatexp dans ces samples).
> Rôle pour nous (TODO §G8.2) : successeur de « porter l'approche F3 » pour
> les prochaines itérations perf. Ce doc = pendant de
> `fraktaler-3-analysis.md` pour ce dépôt.

## Inventaire

| Évaluateur | Idée | Verdict pour fractall |
|---|---|---|
| `Perturbation` | Boucle rebasing canonique, pas de LA | Déjà porté (via F3) |
| `PTWithCompression` | **Compression d'orbite par waypoints** (orbite fantôme f64 resynchronisée) | **À porter — mémoire ~O(waypoints)** |
| `MipLA` | BLA mip power-of-2 (≈ notre `BlaTableUnified`) | Déjà équivalent ; 2 micro-idées |
| `HarmonicLLA` | **LA multi-étages segmentée aux dips** (variante EN PRODUCTION dans Imagina) | **Le vrai après-BLA — prototype à faire** |
| `HarmonicMLA` | Variante segmentation par magnitude (plus simple) | Chemin d'implémentation plus court vers LLA |

## Le socle commun (identique ou déjà chez nous)

- **Boucle pixel** : `dz ← dz·(Z+z) + dc` (forme factorisée de `δ(2Z+δ)+dc`),
  rebase quand `j == refLen || |z| < |dz|` → `Z=0, dz=z, j=0`. C'est
  exactement notre rebasing F3-strict + rebase-at-end (`pixel_loop.rs`) —
  normal, F3 l'a pris de Zhuoran. **Aucune glitch-detection nulle part** :
  valide notre architecture post-Pauldelbrot.
- **Arrêt de référence** : `radius·|dzdc|·2 > |z|` (chebyshev) **ou** `|z|>4`.
  Le premier critère est l'atom-domain « la vue est couverte par le disque de
  linéarité » — appliqué **inconditionnellement** à toutes les références
  (avec marge ×2), là où nous le gâtons par plage de zoom (deep>1e280,
  mid<1e-13) après les sessions G2. Le second : une réf peut s'arrêter dès
  |z|>4 — le rebase-at-end absorbe la suite. Nous gardons l'orbite pleine
  jusqu'au bailout : marge d'optimisation mineure (le pré-check f64 rend la
  boucle GMP déjà optimale, l'arrêt à |z|>4 vs 25 n'économise que ~quelques
  itérations).
- **Norme de Chebyshev** (`max(|re|,|im|)`) partout où nous utilisons
  `norm_sqr` : 2 abs + cmp au lieu de 2 mul + add. Micro-gain candidat sur le
  hot-path validité BLA (à A/B mesurer, prob. neutre — les compares `r²`
  évitent déjà le sqrt).
- Bailout **asymétrique** : réf 4, pixels 64 (smooth). Sans intérêt pour nous
  (ER=25 aligné F3 pour la parité).

## PTWithCompression — compression d'orbite par waypoints

**Mécanique** (`ReferenceCompressor`, 63 lignes) : pendant le calcul HP de la
référence, on itère EN PARALLÈLE une **orbite fantôme f64** `z ← z²+c` (même
formule, précision pixel). Tant que le fantôme colle à la vraie orbite
(`cheb(z − Z) ≤ cheb(Z)·2⁻³²`), on ne stocke RIEN. Quand il décroche, on
émet un waypoint `{Z, itération}` et on **resynchronise** le fantôme
(`z = Z`). Décompression (`Next()`) : rejouer `z²+C` en f64, snap aux
waypoints. **Reset()** = retour à l'itération 0 (= ce que fait le rebase).

**Pourquoi ça marche** : l'erreur du fantôme croît ~exponentiellement au taux
de Lyapunov local ; à tolérance 2⁻³², un waypoint tous les ~32/log₂(λ) pas.
Sur les longues quasi-périodes (intérieur, minibrots), λ→1 → compression
massive (10²-10⁴× typique d'après les données de terrain d'Imagina/KF).

**Coût de décompression** : 1 carré complexe f64 + 1 compare par pas — du
même ordre qu'un load DRAM raté ; sur les orbites de 10⁷+ iters qui ne
tiennent pas en cache, Zhuoran rapporte que c'est souvent PLUS RAPIDE que la
lecture mémoire. C'est notre classe wfs_mb/opus2/dragon.

**Ce que ça donnerait chez nous** :
- `z_ref_f64` (16 o/iter) + `z_ref` ComplexExp (32 o/iter) → O(waypoints).
  dragon 5 M iters : ~240 Mo → quelques Mo. wfs_mb 10 M : ~2 GB → ~Mo.
- ⚠️ **Friction #1 — accès aléatoire** : notre boucle lit `z_ref[m]` après
  chaque saut BLA (check de rebase au point d'atterrissage). Le décompresseur
  est séquentiel. Solutions : (a) stocker `Z_atterrissage` dans le nœud BLA
  (+16 o/nœud, atterrissage déterministe `start+2^k`) ; (b) approche
  HarmonicLLA : les segments LA portent leur `Z` de départ (cf. ci-dessous)
  et l'orbite brute n'est lue QUE dans la phase directe séquentielle. (b) est
  la vraie architecture cible ; (a) est le pont incrémental.
- ⚠️ **Friction #2 — le fantôme doit être la MÊME formule** : trivial pour
  Mandelbrot ; pour les types bytecode, le fantôme = `interp.rs` f64 (déjà
  là). Les types à abs (BS/Celtic…) décrochent plus vite près des plis →
  compression moindre, correction inchangée (les waypoints compensent).
- ⚠️ **Friction #3 — tier exp** : à zoom >1e13 la boucle lit l'orbite en
  ComplexExp. Le fantôme f64 déborde/underflow ? Non : l'ORBITE de référence
  vit à l'échelle O(1) (c'est δ qui est minuscule) — le fantôme f64 suffit
  tant que la réf ne passe pas sous ~1e-308, ce qui n'arrive qu'aux
  quasi-nucleus (|Z_P|≈0) : waypoint forcé à ces points (le compresseur le
  fait naturellement : décrochage relatif). À vérifier sur e22522
  (réf frôlant 0 à prec≫1023 — cas piège connu).
- **Verrous existants** : goldens 21 + quality 14 + fuzz. Bit-exactitude
  attendue : les valeurs lues sont soit le waypoint exact (= l'ancienne
  valeur stockée), soit un f64 recalculé — PAS bit-identiques à l'orbite
  arrondie depuis GMP entre les waypoints. ⇒ verrou = pixel-exact sur les
  goldens quand même (l'erreur est sous 2⁻³² relatif, l'itération escape ne
  bouge pas), sinon tolérance quality PASS. Prototype env-gated d'abord.

## MipLA — notre BLA, à 2 nuances près

Isomorphe à `BlaTableUnified` : nœuds single-step `A=2z, B=1,
r=|z|·2⁻²⁴`, merge `r=min(r₁, r₂/|A₁|)` + `rc`, niveaux power-of-2, lookup
aligné (`index%2` → break = notre `trailing_zeros`). Différences :

1. **Chaînage des sauts** : après un saut réussi, MipLA re-lookup
   IMMÉDIATEMENT au nouvel index (boucle interne) et n'exécute le check de
   rebase qu'après la chaîne. Nous re-passons par le tour de boucle complet
   (rebase check + step direct) entre deux sauts. Restructuration locale de
   `pixel_loop.rs` — candidat micro-perf mesurable (les cas BLA-bound :
   glitch_test_2, e113).
2. Ils gardent le niveau 0 en mémoire (96 o/iter de table !) — nous skippons
   les niveaux 0-2 (`BLA_SKIP_LEVELS=3`, ~20 o/iter). Notre choix est
   meilleur (et aligné F3) ; rien à prendre.

## HarmonicLLA — le vrai différenciateur (variante en prod dans Imagina)

Rupture avec le mip power-of-2. Trois idées imbriquées :

### 1. Segments à longueur variable, coupés aux « dips »
Un segment LA = `dz ↦ A·dz' + B·dc` où `dz' = dz·(Z₀+z₀)` est le **premier
pas quadratique EXACT** (le seul terme non-linéaire de la perturbation est
δ² ; il est traité exactement là où δ est le plus gros relatif). Extension
d'un pas : `A ← 2z·A`, `B ← 2z·B+1`, rayon `min(r, |z|·2⁻²⁴/|A|)`. Un
**dip** = chute du rayon de validité > 2⁻¹⁰ en un pas = passage de l'orbite
près de 0 (frontière d'atome). Les segments s'arrêtent aux dips → ils
épousent la STRUCTURE de l'orbite au lieu des frontières 2^k arbitraires.

### 2. La période détectée plafonne les segments — hiérarchie harmonique
Le premier dip du niveau 0 ≈ la **période** de l'atome dominant. Les
segments sont plafonnés à `Length ≤ Period`. L'étage suivant compose les
segments de l'étage précédent, redétecte SA période (super-période =
minibrot parent), etc. Les étages matérialisent la hiérarchie
période/super-période — d'où « Harmonic ». Mémoire : ~n/période segments à
l'étage 0, décroissance géométrique ensuite (≪ notre table mip pour les réfs
à période courte).

### 3. Évaluation par descente d'étage (pas de lookup par itération)
Le pixel démarre à l'étage LE PLUS HAUT et y reste tant que les rayons
tiennent (1 check de validité par SEGMENT accepté). Sur échec →
`j = step.NextStageLAIndex` : descente à l'étage plus fin, au bon index.
Après l'étage 0 → itération directe sur l'orbite brute. **Le rebase
fonctionne à chaque étage** (`j==end || |z|<|dz|` → `j=begin`). Notre boucle
fait un lookup multi-niveaux PAR position — Harmonic fait ~O(sauts) checks
au lieu de ~O(itérations) lookups.

Bonus architecture : chaque segment porte son `Z` de départ → la phase LA ne
touche JAMAIS l'orbite brute (cache-friendly, et pré-requis naturel de la
compression d'orbite ci-dessus).

### HarmonicMLA (« Mag »)
Même évaluateur ; la segmentation remplace la détection de dip par un
critère magnitude : période = quand min courant de |Z| chute sous
`prevMin·2⁻⁴` ; seuil de coupe = `sqrt(prevMin·min)` (moyenne géométrique).
Plus simple à implémenter/déboguer que LLA — bon premier jalon d'un port.

### Généralisation non-conforme (notre exigence en plus)
Ces samples sont Mandelbrot-only (`A=2z` en dur). Chez nous : l'extension
d'un pas = notre single-step dual-number (`bla_dual.rs`), la composition =
notre merge Mat2 — les récurrences se transposent en Mat2 sans obstacle
théorique. Le premier-pas-quadratique-exact devient « exécuter 1 pas
bytecode exact puis appliquer le segment » (déjà la forme de notre boucle).
La détection de dip par effondrement du rayon est générique. Seul le
plafonnement par période suppose une quasi-périodicité que les types à abs
cassent près des plis — dégradation gracieuse attendue (segments courts),
à mesurer.

## Comparaison synthétique

| Dimension | fractall (bytecode) | Imagina/Algorithms |
|---|---|---|
| Rebasing | F3-strict + rebase-at-end | Identique (source originelle) |
| LA | BLA Mat2 mip 2^k, skip levels 0-2, lookup aligné/position | LLA segments variables aux dips, étages par période, descente |
| Premier pas de saut | linéarisé (dans le nœud) | **quadratique exact** → rayons plus larges |
| Coût contrôle boucle | lookup multi-niveaux par position | 1 check par segment accepté |
| Orbite en RAM | 48 o/iter (f64+ComplexExp) pleine longueur | 16 o/iter pleine (samples) ; **O(waypoints) avec compression** |
| Arrêt de réf | atom-domain gaté par plage zoom | atom-domain inconditionnel ×2 + bailout 4 |
| Types | 10+ familles, Mat2 non-conforme, tiers f64/exp/dd | Mandelbrot f64 only |
| Glitch handling | rebasing only | rebasing only (validation) |

## Recommandations de portage (ordre proposé)

1. **PTWithCompression** — cible : classe robustesse/mémoire (wfs_mb 2 GB,
   opus2 8 GB, dragon 240 Mo d'orbite) + probable gain vitesse cache sur
   orbites ≥10⁶ iters. Pont incrémental : stocker `Z_atterrissage` dans
   `BlaMultiStep` (+16 o/nœud) pour lever la friction d'accès aléatoire,
   compresser `z_ref{,_f64}`. Env-gated, A/B sur wfs_mb/dragon/opus2,
   verrous goldens+quality+fuzz. Effort : moyen. Risque : faible (borné par
   la tolérance 2⁻³², resync exact aux waypoints).
2. **Prototype HarmonicMLA→LLA Mandelbrot f64** (`FRACTALL_HARMONIC_LA=1`) —
   cible : l'axe vitesse au-delà de F3 (le levier qui fait d'Imagina la
   référence). Commencer par MLA (segmentation magnitude, plus simple),
   mesurer sur les cas BLA-bound du harness (glitch_test_2, e113, dragon,
   e50) vs BLA mat2. Si gain confirmé → LLA (dips), puis généralisation Mat2
   + tier exp. Effort : élevé (multi-itérations). Risque : moyen (nouvelle
   boucle pixel → verrous fuzz/quality/goldens indispensables).
3. **Micro-A/B rapides** (une itération) : chaînage des sauts BLA avant le
   check de rebase (MipLA style) ; chebyshev vs norm_sqr sur la validité.
   Mesure standard-tier sur cas BLA-bound, revert si neutre.
4. **Non-goals** : bailout asymétrique (parité F3), abandon du Mat2 (nos
   types non-conformes en dépendent), reprise du niveau 0 en table (leur
   choix mémoire est pire que le nôtre).

## Pointeurs source

- Rebasing canonique : `Perturbation.cpp:34-62`
- Compression : `PTWithCompression.h:20-84` (compressor/décompressor)
- Arrêt de réf atom-domain : `*.cpp` `ComputeOrbit` (`radius·|dzdc|·2 > |z|`)
- Merge BLA : `MipLA.h:32-41` ; lookup aligné : `MipLA.cpp:72-89`
- Segment LLA (premier pas exact + dip) : `HarmonicLLA.h:39-83`
- Détection de période par dip : `HarmonicLLA.cpp:48-102`
- Descente d'étages : `HarmonicLLA.cpp:171-208` (idem MLA)
- Segmentation magnitude : `HarmonicMLA.cpp:48-102`
