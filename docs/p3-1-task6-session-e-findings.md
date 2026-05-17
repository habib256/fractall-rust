# P3.1 #6 Session E — Bytecode activé par défaut

État au 2026-05-17 sur branche `Stable`. Suite de
`docs/p3-1-task6-session-d-findings.md` (non créé mais résumé dans le
commit Session D `6c74362`).

## Changements Session E

### 1. Julia variants via `delta_initial`

`iterate_pixel_unified` prend maintenant un paramètre `delta_initial:
Complex64`. Pour Julia (z₀ = pixel, c = seed) :
- `delta_initial = pixel - cref` (fourni par le caller comme delta0)
- `c_for_add = seed` (constant pour ref et pixel)
- `dc_for_add = 0`

Pour Mandelbrot-like : `delta_initial = 0`, `c_for_add = cref`,
`dc_for_add = dc`.

### 2. `use_bytecode_engine = true` par défaut

- `definitions.rs::default_params_for_type` : default à `true`
- CLI : `--bytecode` passe à `default_value_t = true` avec
  `ArgAction::Set` → `--no-bytecode` désactive
- Path legacy reste actif comme fallback (types non supportés par
  `compile_formula`, deep zoom GMP `pixel_size < 1e-13`, features
  avancées distance/interior/orbit_traps)

### 3. Bug fix CLI

Avant Session E, `params.use_bytecode_engine = cli.bytecode;` écrasait
le défaut `true` par `false` si l'utilisateur ne passait pas
`--bytecode`. Le défaut ne s'appliquait jamais. Corrigé avec
`default_value_t = true` côté clap.

## Validation

### Tests Julia bytecode

Via CLI legacy vs bytecode, sur 160×100 :

| Cas | % pixels diff | % lourd |
|-----|---------------|---------|
| Julia f64 (zoom 1) | **0.00 %** | 0.00 % |
| JuliaBS perturbation 1e3 | **0.00 %** | 0.00 % |

**Pixel-perfect** sur Julia variants avec le delta_initial fix.

### Goldens régénérés

Avec bytecode par défaut, 4 cas perturbation Mandelbrot/BS divergent
des anciens goldens (legacy comme référence). Régénérés en mode bytecode
comme **nouvelle référence cible** :

| Golden | Diff vs ancien (legacy) | Statut |
|--------|------------------------|--------|
| mandelbrot_perturb_1e6 | 93.66 % (mean 4 %/canal) | régénéré |
| burning_ship_perturb_1e3 | 5.22 % | régénéré |
| mandelbrot_minibrot_1e8 | 0.29 % | régénéré |
| burning_ship_zoom_1e5 | 6.02 % | régénéré |

Les 8 autres goldens (f64 standard + types non-bytecode) sont
**identiques au bit près** entre legacy et bytecode — leur chemin ne
passe pas par try_bytecode_unified_path (Newton, etc.) ou produit
exactement le même résultat (Mandelbrot zoom 1 f64 standard utilise
le path direct iterate_point, pas iterate_pixel).

## Stats

131 unit tests + 12 goldens (régénérés) verts.

## Reste pour fermer #6 + #8

| Étape | Difficulté |
|-------|------------|
| ComplexExp pour deep zoom > 1e13 dans le pixel loop unifié | Moyen |
| Multi-phase (hybrides) | Faible |
| Suppression `glitch.rs` | Moyen (params CLI à retirer aussi) |
| Suppression `nonconformal.rs` | Faible (utilisé seulement par bla.rs path conformal) |

Le path bytecode est maintenant **la voie par défaut** pour tous les
types escape-time supportés en perturbation f64. La machinerie legacy
(Pauldelbrot glitch detection + clustering + secondary refs) reste
disponible via `--no-bytecode` mais devient progressivement obsolète.
