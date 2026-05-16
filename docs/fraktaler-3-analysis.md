# Fraktaler-3 — Analyse technique détaillée

> Source de vérité pour le port Rust. Toutes les références sont au repo `https://code.mathr.co.uk/fraktaler-3` (mirror : `https://github.com/icalvo/fraktaler-3`) et aux blog posts de Claude Heiland-Allen (`mathr.co.uk/blog/`).

> **Caveats préalables**. Deux formules folkloriques souvent attribuées à Fraktaler-3 (`prec = max(24, 24 + floor(log2(zoom * height)))` clampée à `[128, 8192]`, et le critère Pauldelbrot `|z+Z| < G·|Z|`) **ne sont pas utilisées telles quelles** dans la source. Voir §3 et §6.

---

## 1. Architecture haut niveau

**Langages.** C++17 (~68%), C (~27%, surtout kernels OpenCL et glue plateforme), Makefile. Licence AGPL-3.0-only, Claude Heiland-Allen 2021-2023+.

**Build.** `Makefile` racine + `Android.mk` dans `src/`. Cross-compile vers Linux natif, x86/ARM Windows (via `llvm-mingw`), Web (Emscripten, nécessite `SharedArrayBuffer`), Android (NDK API 21+).

**Layout top-level.**
```
fraktaler-3/
├── app/                # Android shell
├── src/                # tout le moteur + UI
├── Makefile
├── README.md, INDEX.txt, LICENSE.md
└── fraktaler-3.css / .ico
```

**Arbre `src/`** organisé par (a) types numériques, (b) formule/perturbation/BLA, (c) rendu/dispatch, (d) UI/plateforme :
- Types : `complex.h`, `dual.h`, `matrix.h`, `floatexp.h`, `softfloat.h`, `float128.h`, `types.h`
- Formule + perturbation : `hybrid.h`, `hybrid.cc`, `bla.h`, `bla.cc`
- Rendu : `engine.h`, `engine.cc`, `render.h`, `render.cc`, `parallel.h`, `stats.h`
- I/O : `image.h`, `image_raw.{cc,h}`, `image_rgb.{cc,h}`, `exr.h`, `histogram.{cc,h}`
- Params & TOML : `param.{cc,h}`, `source.{cc,h}`, `version.{cc,h}`, `wisdom.{cc,h}`
- OpenCL : `opencl.{cc,h}` + kernels assemblés runtime `cl-pre.cl` + corps généré + `cl-post.cl`
- Display/GUI : `display.h`, `display_gles.{cc,h}`, `gles2.{cc,h}`, `glutil.{cc,h}`, `gui.cc`
- Entry points : `main.{cc,h}`, `batch.cc`

**Dépendances clés.** SDL2 + OpenGL ES + Dear ImGui (GUI), MPFR/MPC via `libmpfrc++` (orbites HP), GLM, OpenEXR (export raw compatible KFR), OpenCL (optionnel, via CLEW pour chargement runtime), zlib/png (screenshots). Web : SDL2 → port Emscripten, OpenGL ES → WebGL 2.

**CLI vs GUI.** Binaire unique, mode choisi via `argv` : `--interactive` (SDL2 + ImGui), `--batch` (headless TOML → PNG/EXR/KFR), mode wisdom-benchmark qui mesure chaque `(type × device)` et écrit un JSON utilisé pour choisir les backends. Pipeline : `main.cc` → `engine.cc` → `render.cc` → `hybrid.cc`.

---

## 2. Formules supportées

Fraktaler-3 **n'a pas d'enum de types**. Il a un **compilateur de formules hybrides génériques**. Chaque formule est une liste de "phases" (`phybrid1` dans `param.h`) ; chaque phase a `(abs_x, abs_y, neg_x, neg_y, p, q)` + puissance. `param.cc::compile_formula()` traduit chaque phase en **bytecode**, exécuté par interpréteur (CPU) ou compilé en kernel OpenCL (GPU).

**Jeu d'opcodes** (8 instructions) :
```
op_add    op_store    op_sqr    op_mul
op_absx   op_absy     op_negx   op_negy
```

Puissances arbitraires (Multibrot) : chaînes de `op_sqr` + `op_mul` via décomposition binaire de l'exposant, avec `op_store` pour mémoriser `Z`. Burning Ship / Tricorn / Mandelbar / Celtic / Perpendicular / Buffalo : combinaisons d'`absx`/`absy`/`negx`/`negy`.

**Chaînage de phases = hybrides.** Plusieurs phases itérées cycliquement (`phase = (phase + 1) % opss.size()`). Permet par exemple "5× Mandelbrot puis 3× Burning Ship". **C'est LA feature centrale** : tous les Mandelbrot/Burning Ship/Tricorn/Celtic/Perpendicular/Buffalo/Multibrot^n et leurs hybrides sont *des paramètres*, pas des codepaths distincts.

**Hors scope (vs notre projet Rust).** Pas de Newton, Phoenix, Magnet, Nova, Mandelbulb, Lyapunov, Buddhabrot/Nebulabrot, Von Koch/Dragon dans le moteur core. Nova existe dans un blog séparé (cf. §13). Julia : on fixe `C` constant dans la formule, pas un type distinct.

---

## 3. Perturbation theory

**Stockage de l'orbite référence** (`hybrid_reference` dans `hybrid.cc`). Itérée en MPFR (`mpfr_t Z_x, Z_y`) et **simultanément castée en basse précision** : `Zp[i] = complex<t>(mpfr_get<t>(Z_x, RNDN), mpfr_get<t>(Z_y, RNDN))`. Type `t` ∈ `{float, double, long double, floatexp, softfloat, float128}`. `floatexp` = mantisse + exposant (§12), pour orbites trop profondes pour double. **Une array par phase** pour hybrides : `std::vector<std::vector<complex<t>>> Z`.

**Bailout de la référence.** Hardcodé `1e10` pour la norme carrée (commentaire `// FIXME escape radius`). Le bailout utilisateur (`escape_radius`, défaut 625 = 25²) ne sert que pour les pixels.

**Cap iterations référence.** `par.p.bailout.maximum_reference_iterations`, défaut = `bailout.iterations` (1024).

**Formule de précision — la vraie dans la source.** Dans `engine.cc` :
```cpp
mpfr_prec_t prec = std::max(24, 24 + (par.zoom * par.p.image.height).exp);
```
Dans `param.cc` pour params dérivés :
```cpp
mpfr_prec_t prec = 24 + floatexp(out.zoom).exp;
```

Donc la vraie formule : `prec = max(24, 24 + exp(zoom * height))` où `.exp` est le champ exposant binaire entier d'un `floatexp`. Moralement équivalent à `24 + ⌈log2(zoom·height)⌉` mais lit directement le champ exposant, jamais `log2()`. **Pas de clamp `[128, 8192]` dans la source** — uniquement le `max(24, …)`. Pour Newton : `prec = 24 + 3 * max(prec(center.x), prec(center.y))`.

**Itération pixel (perturbation).** Dans `hybrid_render`, l'état pixel est un `z` basse précision + un couple `(phase, m)` pointant dans l'array référence. Boucle interne simplifiée :

```cpp
while (n < Iterations && Zz2 < ER2 && IR < dZ &&
       iters_ptb < PerturbIterations && steps_bla < BLASteps)
{
  // 1. Essai BLA
  const blaR2<real> *b = bla[phase].lookup(m, z2);
  if (b) {
    z = b->A * z + b->B * c;
    z2 = normx(z);
    n += b->l;  m += b->l;  steps_bla++;
    continue;
  }
  // 2. Sinon, un pas perturbation via hybrid_perturb (interpréteur d'opcodes)
  m += 1; n += 1; iters_ptb++;

  // 3. Check de rebasing
  const real Zz2 = normx(Zp[phase][m] + z);
  if (Zz2 < z2 || m + 1 == count_t(Zp[phase].size())) {
    z   = Zp[phase][m] + z;        // promotion delta → absolu
    z2  = Zz2;
    phase = (phase + m) % Zp.size();
    m = 0;
  }
}
```

**Stratégie de rebasing.** Punch line du blog 2022-02-21 "Deep zoom theory and practice (again)". Plutôt que détecter les glitches *après* et re-render, Fraktaler-3 **rebase proactivement** : à chaque pas, si `|Z+z| < |z|` (le point perturbé est devenu plus proche du point critique que la référence), on réinstalle l'itération sur la référence qui minimise `|(Z − Z_o) + z|`, repartant à `m=0`. Pour hybrides, chaque rotation de phase a sa propre orbite et le rebasing choisit. **C'est pour ça que Fraktaler-3 n'a pas de paramètre `glitch_tolerance`** (cf. §6).

**Bailout pixel.** `ER2 = escape_radius²` (défaut 625² = 390 625, donc rayon 25 — large pour bien conditionner l'itération smooth). `IR = bailout.inscape_radius` (défaut 1/1024) = bailout *intérieur* : quand la norme de la dérivée `dZ` passe sous `IR`, le pixel est classé intérieur (cf. §8).

---

## 4. BLA (Bilinear / "Ball Linear" Approximation)

L'auteur écrit parfois "Ball Linear" dans les commentaires source, "Bivariate Linear" dans les blogs — même chose.

**Structure** (`bla.h`) :
```cpp
template <typename real> struct blaR2 {
  mat2<real> A;     // partie linéaire perturbation (matrice 2×2 réelle, pas complexe)
  mat2<real> B;     // partie linéaire offset pixel
  real      r2;     // rayon de validité au carré sur |z|
  count_t   l;      // nombre d'itérations sautées
};

template <typename real> struct blasR2 {
  count_t M;
  count_t L;                                         // profondeur de l'arbre
  std::vector<std::vector<blaR2<real>>> b;           // b[level][index]
  const blaR2<real> *lookup(count_t m, real z2) const noexcept;
};
```

**Une `blasR2` par phase** pour hybrides (`hybrid_blas` dans `hybrid.cc`).

**Niveaux.** Level 0 a `M-1` entrées single-step ; chaque niveau halves : `m = (m+1)>>1`, jusqu'à 1. `L = ⌈log2(M-1)⌉ + 1`.

**Construction single-step — `hybrid_bla()`** :
```cpp
real e = 1 / L;                                   // L est ici un scalaire de précision
dual<2, real> x(Z.x); x.dx[0] = 1;
dual<2, real> y(Z.y); y.dx[1] = 1;
complex<dual<2, real>> W(x, y);
complex<dual<2, real>> W_stored(W);
complex<dual<2, real>> C(0, 0);                   // FIXME
real r = e * abs(Z) * degree / (degree*(degree-1)/2);
mat2<real> A0(I);
for (const auto &op : ops) {
  complex<real> W0(W.x.x, W.y.x);
  complex<real> W0_stored(W_stored.x.x, W_stored.y.x);
  hybrid_plain(op, C, W, W_stored);
  const mat2<real> A(W.x.dx[0], W.x.dx[1], W.y.dx[0], W.y.dx[1]);
  switch (op) {
    case op_add:  return blaR2<real>{ A, I, r*r, 1 };
    case op_store: break;
    case op_sqr:  r = min(r, e * abs(W0) / sup(A0)); break;
    case op_mul:  r = min(r, e * min(abs(W0), abs(W0_stored)) / sup(A0)); break;
    case op_absx: r = min(r, abs(W0.x) / 2 / sup(A0)); break;
    case op_absy: r = min(r, abs(W0.y) / 2 / sup(A0)); break;
    case op_negx: break;
    case op_negy: break;
  }
  A0 = A;
}
```

**Points clés à internaliser :**
- `A` et `B` sont **toujours des matrices 2×2 réelles**. Les cas conformes (Mandelbrot pur) sont représentés comme `[[a,-b],[b,a]]` implicitement — pas de cas spécial pour la mul complexe. Choix : "non-conformal BLA always on".
- Rayon initial : `r = ε · |Z| · n / C(n,2)` où `n` = degré, `ε = 1/L` = **précision machine**, *pas* le `L` du merge. Nom `L` surchargé.
- Rayon **rétréci** par chaque op non-linéaire : `r ← min(r, ε·|W₀|/sup(A₀))` pour squaring/mul, `r ← min(r, |W₀.x|/(2·sup(A₀)))` pour un fold `absx` (Burning-Ship). Le fold n'a pas de facteur `ε` — il est exact, le rayon empêche juste de traverser la ligne de pli.
- `sup(A)` = **norme opérateur 2** (plus grande valeur singulière), calculée dans `matrix.h` : `sup(A) = (T + sqrt(max(0, T² − 4D))) / 2` avec `T = trace(AᵀA)`, `D = det(AᵀA)`.
- Les paramètres `h` et `k` dans `blas_init1`/`blasR2` sont **passés mais inutilisés** (`(void)h; (void)k;` en début de `hybrid_bla`). Vestigial / réservé pour tuning futur.

**Formule de merge — `merge()` dans `bla.h`** :
```cpp
const count_t l = x.l + y.l;
const mat2<real> A = y.A * x.A;
const mat2<real> B = y.A * x.B + y.B;
const real xA = sup(x.A);
const real xB = sup(x.B);
const real r  = min(sqrt(x.r2), max(real(0), (sqrt(y.r2) - xB * c) / xA));
const real r2 = r * r;
```
Avec `c = pixel_spacing = max|c|` sur l'image. Formule de composition canonique du blog 2022-02-21 :
```
A_{y∘x} = A_y · A_x
B_{y∘x} = A_y · B_x + B_y
r_{y∘x} = min(r_x, max(0, (r_y − |B_x|·max|c|) / |A_x|))
l_{y∘x} = l_x + l_y
```

**Lookup — `bla.cc`** :
```cpp
const blaR2<real> *ret = 0;
count_t ix = m - 1;
for (count_t level = 0; level < L; ++level) {
  count_t ixm = (ix << level) + 1;
  if (m == ixm && z2 < b[level][ix].r2) {
    ret = &b[level][ix];
  } else {
    break;
  }
  ix = ix >> 1;
}
return ret;
```

À l'itération `m` avec `z2`, on remonte de la feuille tant que `(m == aligned-index)` ET `z2 < r²`, en gardant le **skip le plus profond**. C'est l'alignement `ixm = (ix << level) + 1` (avec `m=1` mappé à leaf 0 à tous niveaux) qui correspond à notre bugfix #1.

**Seuil.** `engine.cc` : `const float precision = count_t(1) << 24; // FIXME`. Donc Fraktaler-3 utilise `ε = 1/2²⁴ ≈ 6e-8` comme seul seuil BLA, peu importe le type réel sous-jacent. Notre `bla_threshold = 1e-8` est dans le même ordre. **Pas de "BLA validity scale"** dans F3.

**Non-conformal BLA.** F3 utilise **toujours** des matrices 2×2 réelles. Pas de SVD explicite — formule fermée trace/det. `inf` (plus petite valeur singulière) sert au check intérieur `dZ` (§8).

---

## 5. Series approximation

**Fraktaler-3 ne fait pas de series.** Pas de paramètre `series_order`, pas de table de coefficients `A_n^k`. Le skip pixel se fait via BLA uniquement.

Justification (blog 2022-02-21) : BLA "est conceptuellement plus simple, plus facile à implémenter, plus facile à paralléliser, a des conditions d'arrêt mieux comprises, est plus général (s'applique à Burning Ship, hybrides…)". Pour les hybrides — *raison d'être* de Fraktaler-3 — series n'est même pas applicable.

Le blog 2021-05-14 parle de series + probe-points + biseries près des minibrots, mais ces techniques sont décrites comme appartenant à Kalles Fraktaler 2, pas à F3. **Pour matcher F3, pas besoin de series ; si on la garde, on est superset.**

---

## 6. Glitch detection & correction

**Fraktaler-3 n'a pas de détecteur Pauldelbrot ni de sélection de références secondaires.** C'est le plus gros écart vs les implémentations Kalles Fraktaler.

À la place, **le rebasing de §3 EST le mécanisme anti-glitch**. Blog 2022-02-21, explicite : "[Le rebasing] évite les glitches plutôt que les détecter après coup, améliorant à la fois l'efficacité et la correction." Quand `|Z+z| < |z|` (condition `Zz2 < z2`), on bascule sur une nouvelle référence (`phase` + `m=0`) **sans paramètre de tolérance**. Les travaux antérieurs de l'auteur (blog Burning Ship 2018-01-04) utilisaient bien Pauldelbrot `|(X+x) + i(Y+y)|² < 10⁻³ |X + iY|²` avec une file de références secondaires, **mais cette approche a été retirée** au profit du rebasing.

**Conséquences pour le port Rust.** Notre `glitch_tolerance` (1e-4), `max_secondary_refs` (3), `min_glitch_cluster_size` (100), tout `glitch.rs` (flood-fill clustering) **n'ont pas d'analogue dans F3**. Choix légitime mais différent, hérité de Kalles Fraktaler. La voie F3 : rebasing agressif proactif par pixel par itération.

**Critère Pauldelbrot pour mémoire** : pixel glitché à itération `k` quand `|x_k + w_k|² < τ · |x_k|²` pour tolérance `τ ∈ [10⁻⁸, 10⁻²]`. Pas de preuve formelle mais marche en pratique.

---

## 7. Distance estimation

DE via **dual numbers à deux parties** trackant la dérivée de `Z+z` par rapport à l'offset pixel `c`. Setup dual (`hybrid.h`) : `dx[0] = 1, dx[1] = 0` pour la direction X et `dx[2] = 1, dx[3] = 0` pour Y (4 partielles scalaires, packées comme Jacobien 2×2).

Dans `hybrid_render`, la valeur d'évasion dual-augmentée donne une matrice dérivée 2×2 `J = ((dRe/dcx, dRe/dcy), (dIm/dcx, dIm/dcy))`. DE extérieure :
```
de = |Z| · log|Z| / sup(J)        (les deux directions équivalentes sous sup-norm)
```

Avec variantes par direction `DEX` et `DEY` dispos au shader. Ligne verbatim (`hybrid.cc`) :
```cpp
complex<double> de = norm(Z1) * log(abs(Z1)) / dC;
```

DE intérieure : même machinerie dual mais dérivée wrt le point critique au lieu de wrt `c` (cf. §8).

---

## 8. Interior detection

Detection intérieur tourne **dans la même boucle d'itération** via la clause `IR < dZ` du while. `dZ` = norme opérateur du Jacobien de `Z+z` wrt l'**orbite référence du point critique** (tracking dual *séparé* de la dérivée pixel-c utilisée pour DE). Quand cette norme passe sous `IR` (défaut `1/1024`), pixel = intérieur.

Pour formules non-conformes, même machinerie matrice 2×2 `sup`/`inf` (`matrix.h`). Recette du blog 2022-02-21 : "utiliser dual numbers à 4 parties combiné avec normes opérateur matricielles". Notre `ExtendedDualComplex` (5 composantes — valeur + 4 partielles) est l'analogue correct ; F3 le packe comme `complex<dual<2, real>>` (5 scalaires effectifs aussi).

---

## 9. Coloring modes

Fraktaler-3 v3+ **n'a pas d'enum de modes**. Le coloring est un **shader GLSL fragment éditable** par l'utilisateur, implémentant `vec3 colour(void)` avec accès à des scalaires par pixel via uniforms/varyings :
- `N0`, `N1` — compteurs entiers
- `NF` — smooth iteration (continuous escape time)
- `T` — temps fractionnaire
- `DEX`, `DEY` — composantes distance estimate
- coords pixel, taille image, zoom (log₂), temps écoulé
- `getHistogram(x)` — fonction de rang-quantile retournant la position normalisée de `x` dans l'histogramme par image (coloring histogramme ajouté en 3.1)

Shader par défaut : DE monochrome avec sliders brightness/contrast/exposure/gamma.

**Pas d'orbit traps built-in** (Point/Line/Cross/Circle). L'utilisateur les écrit dans le shader si voulu. Notre `orbit_traps.rs` est superset.

Histogram-based coloring (3.1) : moteur construit histogrammes par canal sur `N0`, `NF`, etc., expose au shader pour normalisation adaptative. Post-processing brightness/contrast/gamma/exposure aussi dans le shader couleur.

---

## 10. Espaces couleur / palettes

**Pas de palette built-in.** Toute la couleur vient du shader GLSL utilisateur. Préambule fournit helpers `linear ↔ sRGB` et `HSV → RGB`, mais pas de LCH, pas de palettes curatées (Fire, Plasma…), pas de slider `color_repeat`. Nos 27 palettes + RGB/HSB/LCH + color_repeat sont des ajouts par rapport à F3.

---

## 11. GPU implementation

**API.** OpenCL 1.2 (chargé runtime via CLEW, le binaire tourne sans). **Pas de compute shaders, pas de Vulkan, pas de Metal natif.** Display via OpenGL ES 2/3 (ou WebGL 2) mais uniquement pour blit framebuffer + shader couleur fragment.

**Assemblage du kernel.** `cl-pre.cl` fournit un gros préambule avec abstractions de types numériques (`float`/`double`/`floatexp`/`softfloat` ont chacun `add/sub/mul/div/sqr/sqrt/abs/exp/log/sin/cos/floor/atan/hypot/nextafter/cmp` derrière macro `NUMBER_TYPE`), plus `struct complex`, `struct dual`, `struct complexdual`, `struct mat2`. `cl-post.cl` ferme. Entre les deux, `hybrid_perturb_opencl()` émet une string C qui est le *corps* de l'itération par pixel pour le bytecode courant. Les 3 morceaux concaténés et `clCreateProgramWithSource`/`clBuildProgram` au render time.

**Types numériques GPU.** Même menu f32/f64/floatexp/softfloat que CPU. f64 conditionnel à l'extension `cl_khr_fp64` du device. `float128` CPU-only.

**Oui, GPU fait perturbation + BLA.** Le kernel OpenCL utilise la même `bla[phase].lookup(m, z2)` que CPU. Orbite référence et tables BLA calculées CPU (MPFR pas sur GPU) et uploadées comme buffers ; boucle interne par pixel (perturb + BLA + rebase) sur GPU. Tiles 128×128 par défaut.

**Wisdom.** `wisdom.{cc,h}` benchmark une fois par paire `(device, type)` et écrit JSON avec timings + ranges + grouping hardware. Le renderer lit ça et choisit, pour le zoom courant, le *type le plus rapide qui est représentable*. Multi-device simultané possible — différentes tiles à différents devices.

---

## 12. Modes de précision — quand chaque type s'active

Sélection driven par **wisdom** (mesuré), pas par seuils zoom hardcodés. Progression naturelle :

| Type | C++ | Range | Usage |
|---|---|---|---|
| `float` (f32) | `float` | 2^±127 | zooms peu profonds sur GPU |
| `double` (f64) | `double` | 2^±1023 | jusqu'à ~10³⁰⁰ |
| `long double` | `long double` | 2^±16382 (x86) | rare |
| `floatexp<float>` | mantisse `float` + `int64` exp | ~2^±2⁵⁵ | zooms moyens-profonds |
| `floatexp<double>` ("doubleexp", 3.1) | mantisse `double` + `int64` exp | ~2^±2⁵⁵, mantisse double | zooms profonds avec plus de précision |
| `softfloat` | software 32-bit | configurable | devices OpenCL sans `fp64` |
| `float128` (3.1) | quad precision | 2^±16383, 113-bit mantisse | profondeurs extrêmes |
| MPFR | `mpreal` | arbitraire | référence orbit UNIQUEMENT |

**MPFR utilisée uniquement pour l'orbite référence et Newton**, jamais par pixel. Référence calculée en MPFR à `prec = max(24, 24 + (zoom·height).exp)` puis castée vers le type basse précision choisi.

`floatexp` (`floatexp.h`) : paire `(mantisse, int64 exp)`, valeur représentée = `ldexp(mantisse, exp)`. Mul : additionne exposants, multiplie mantisses. Add : aligne le plus petit exposant sur le plus grand, renormalise. `EXP_MIN/EXP_MAX = ±2^55` → range effectivement illimitée. Précision mantisse = celle du float/double sous-jacent.

---

## 13. Choix algorithmiques notables / astuces

- **Rebasing-as-glitch-avoidance.** Le plus gros choix de design. Blog 2022-02-21. Per-pixel/per-iteration `if (|Z+z| < |z|) reseat sur la référence la plus proche`. Pas de Pauldelbrot, pas de clustering, pas de queue de références secondaires.
- **Orbites référence phase-aware.** Pour formules hybrides à cycle `k`, l'algo stocke `k` orbites référence (une par rotation de phase) et le rebasing choisit. Seule façon de faire BLA correct pour hybrides.
- **Matrices 2×2 réelles partout.** BLA `A` et `B` toujours `mat2<real>`. Coût constant minime sur formules conformes mais code unifié, Burning Ship + Tricorn gratis.
- **Interpréteur d'opcodes pour formules.** Phases compilées en bytecode 8-opcodes. CPU inner loop et générateur kernel OpenCL consomment le même bytecode. Raison architecturale du first-class hybrid.
- **Sélection backend par wisdom.** Plutôt que heuristique "GPU si zoom < X sinon CPU", F3 mesure chaque combo device × type une fois et pick le plus rapide viable. Renders multi-device de routine.
- **Histogrammes pré-fragment-shader.** Histogrammes sur `N0/NF/DE` calculés avant coloring → le GLSL utilisateur peut appeler `getHistogram(value)` pour rang percentile. Palettes histogram-equalisées triviales.
- **Double store de l'orbite référence.** Pendant l'itération MPFR, le cast basse précision se fait *dans* la boucle d'itération → orbite matérialisée dans le type de travail sans second passage.
- **Rescaled iterations** (blog 2021, attribué à Pauldelbrot "Rescaled Iterations in Nanoscope"). Quand `|Z|` référence très petit, les deltas underflowent même en double ; F3 tourne une fenêtre d'itérations floatexp puis rescale. Utilisé en interne par les paths `floatexp`/`softfloat`.
- **Newton zooming.** `hybrid_period` / `hybrid_center` / `hybrid_size` : localisation automatique des minibrots — trouve le point périodique de période `p` le plus proche, refine via Newton (boost précision : `prec = 24 + 3 * prec_input`), utilise les valeurs propres du multiplieur local pour la transformation auto-zoom. Modes atom-domain et zoom absolu exposés.
- **`diffabs` pour perturbation Burning Ship.** Le `|·|` non-analytique du Burning Ship est géré par `diffabs(c, d)` (piecewise 4-cas) qui donne la différence exacte `|c+d| - |c|` sans cancellation catastrophique. Blog 2018-01-04, originellement par `laser blaster` sur fractalforums.
- **Conjugaison affine pour Nova-like** (blog 2021-09-27). Point critique à `z=1` déplacé à `z=0` par substitution affine → perturbation standard marche. Pas dans le moteur principal mais utile si on ajoute Nova.
- **Export EXR raw.** Données brutes d'itération sauvées en OpenEXR, compatible Kalles Fraktaler 2+ raw et `zoomasm` pour assemblage vidéo-zoom. Permet recolorer/animer sans re-rendre.
- **TOML pour paramètres** (batch mode + persistance GUI). Drag-and-drop TOML ou clipboard texte restaure paramètres.

---

## 14. Autres choses qui comptent pour matcher F3

- **`ER = 25` par défaut** (pas 2) : garde l'itération smooth bien conditionnée, `log(log|Z|)/log(degree)` est plus stable quand `|Z|` est bien dégagé du bailout.
- **`IR = 1/1024`** est un seuil de norme de dérivée, pas de seuil sur `|z|`.
- **Reference bailout = `1e10` hardcoded** dans `hybrid_reference`. À mirror pour matcher les profondeurs de référence.
- **`maximum_bla_steps` et `maximum_perturb_iterations` sont des caps séparés** (défaut 1024 les deux). Anti-runaway en cas dégénéré. À monter pour deep dwell.
- **Pixel spacing pour BLA = `4/zoom/height`** (verbatim `engine.cc`). C'est le `c` dans le shrink `|B_x|·max|c|` du merge — bien utiliser la *hauteur* d'image (le côté court après application de 4/zoom = largeur totale), pas juste `span`. Notre bugfix #11 sur images non-carrées correspond exactement à cette considération.
- **Seuil BLA `ε = 1/2²⁴` hardcoded**, non type-dépendant. Même en `float128`, BLA accepte ~6e-8 d'erreur relative par skip. Resserrer ce seuil pour types plus hauts changerait légèrement la qualité.
- **Pas de DE intérieure** dans le path principal. Pixels intérieurs ont juste un flag ; le shader couleur peut utiliser `DEX`/`DEY` (qui portent des valeurs sensées même depuis détection intérieur), mais F3 ne calcule pas de métrique DE intérieur séparée.
- **Pas d'anti-aliasing CPU autre que subframes.** AA via N "subframes" avec positions samples jitterées (séquences `burtle_hash`/`radical_inverse` basse-discrépance dans `hybrid.h`) puis moyenne. Le canal DE est par-coord-pixel et *pas* AA'd comme part de l'escape time ; utilisé par le shader.
- **Le sign issue Burning Ship dans dual numbers** (notre bugfix #3) : F3 gère ça dans `hybrid_plain` pour `op_absx`/`op_absy` (les folds changent signes de certaines dual partielles). Pas caché dans la construction BLA ; c'est dans l'interpréteur d'opcodes que BLA setup et perturb loop appellent tous deux. Notre `mul_signed` sur `ExtendedDualComplex` mirror une action per-opcode que F3 fait dans le bytecode.
- **Pas de round-trip metadata PNG.** F3 utilise TOML (et KFR pour raw + params) plutôt qu'embed dans `tEXt` PNG. Notre drag-and-drop PNG est UX plus jolie mais pas la voie F3.

---

## Incertitudes / non vérifié

- Contenu exact de `cl-post.cl` (épilogue kernel). Probablement boilerplate `__kernel` wrapper.
- Forme exacte du numérateur DE par-direction dans `DEX`/`DEY` visibles au shader (si inclut le `log|Z|` ou si le shader le fait). Source dans `hybrid_render` mais non quoté.
- Si `softfloat` est un float software 32-bit custom ou un port de `softfloat.h` (Berkeley). Le nom suggère custom.
- Le path Android utilise le même moteur mais la glue plateforme dans `app/` et `display_gles.cc` diffère.
- Format wisdom : JSON-ish, schema dans `wisdom.cc`, pas user-facing.

---

## Références

- Repo : `https://code.mathr.co.uk/fraktaler-3` (mirror : `https://github.com/icalvo/fraktaler-3`)
- Docs : `https://fraktaler.mathr.co.uk/`
- Deep zoom theory and practice (2021) : `https://mathr.co.uk/blog/2021-05-14_deep_zoom_theory_and_practice.html`
- Deep zoom theory and practice (again) (2022) : `https://mathr.co.uk/blog/2022-02-21_deep_zoom_theory_and_practice_again.html` — **canonique** pour rebasing + BLA + matrices non-conformes
- The Burning Ship (2018) : `https://mathr.co.uk/blog/2018-01-04_the_burning_ship.html` — `diffabs`, path Pauldelbrot historique
- Perturbing Nova (2021) : `https://mathr.co.uk/blog/2021-09-27_perturbing_nova.html` — trick conjugaison affine
- Perturbation techniques (paper) : `https://mathr.co.uk/mandelbrot/perturbation.pdf`
- Wikibooks : `https://en.wikibooks.org/wiki/Fractals/fraktaler-3`

**Fichiers source clés** (paths dans le mirror) :
- `src/hybrid.h`, `src/hybrid.cc` — interpréteur opcodes, référence, driver BLA build/lookup, render loop, émission OpenCL
- `src/bla.h`, `src/bla.cc` — `blaR2`/`blasR2`, `merge()`, `lookup()`, `blas_init1`, `blas_merge`
- `src/matrix.h` — `mat2`, `sup`, `inf` (normes opérateur)
- `src/dual.h` — `dual<D,T>`, `diffabs`
- `src/floatexp.h`, `src/softfloat.h`, `src/float128.h` — types extended-range/precision
- `src/engine.cc` — sélection précision (`prec = max(24, 24 + (zoom·height).exp)`), setup BLA
- `src/param.{cc,h}` — TOML I/O, compilation formule → bytecode, defaults
- `src/render.cc` — dispatch tile, CPU vs OpenCL
- `src/wisdom.{cc,h}` — benchmark + sélection device
- `src/cl-pre.cl` — préambule OpenCL types numériques
- `src/opencl.{cc,h}` — kernel build + plumbing buffers
- `src/gui.cc`, `src/main.cc`, `src/batch.cc` — UI / CLI / dispatch
