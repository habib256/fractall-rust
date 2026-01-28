# CLAUDE.md

## Build & Run

```bash
cargo build --release
cargo run --release --bin fractall-cli -- --type 3 --width 1920 --height 1080 --output out.png
cargo run --release --bin fractall-gui
```

Prerequis: GMP/MPFR/MPC (pour `rug`).

## Architecture

```
src/
├── main.rs              # CLI (clap)
├── main_gui.rs          # GUI (egui/eframe)
├── fractal/
│   ├── mod.rs           # exports + default_params_for_type()
│   ├── types.rs         # FractalType, FractalParams, AlgorithmMode
│   ├── definitions.rs   # constantes par type
│   ├── iterations.rs    # escape-time f64
│   ├── gmp.rs           # precision arbitraire (rug/mpc)
│   ├── lyapunov.rs      # Lyapunov + LyapunovPreset
│   ├── buddhabrot.rs    # Buddhabrot/Nebulabrot
│   ├── vectorial.rs     # Von Koch, Dragon
│   └── perturbation/
│       ├── mod.rs       # render_perturbation_cancellable_with_reuse()
│       ├── types.rs     # ComplexExp (mantisse f64 + exposant)
│       ├── orbit.rs     # ReferenceOrbitCache
│       ├── bla.rs       # BlaTable (Bilinear Approximation)
│       ├── delta.rs     # iterate_pixel()
│       └── series.rs    # approximation par series
├── render/
│   ├── mod.rs
│   └── escape_time.rs   # dispatcher + should_use_perturbation()
├── gpu/
│   ├── mod.rs           # GpuRenderer (wgpu)
│   ├── mandelbrot_f32/f64/ds.wgsl
│   ├── julia_f32/f64.wgsl
│   ├── burning_ship_f32/f64.wgsl
│   └── perturbation.wgsl
├── gui/
│   ├── mod.rs
│   ├── app.rs           # FractallApp (egui)
│   ├── progressive.rs   # rendu multi-passes
│   └── texture.rs
├── color/
│   ├── mod.rs
│   └── palettes.rs
└── io/
    ├── mod.rs
    └── png.rs
```

## Systeme de coordonnees

Le code utilise **center + span** au lieu de xmin/xmax/ymin/ymax:
```rust
pub struct FractalParams {
    pub center_x: f64,  // centre X (pour compatibilité GPU/CPU standard)
    pub center_y: f64,  // centre Y (pour compatibilité GPU/CPU standard)
    pub span_x: f64,    // largeur totale (pour compatibilité GPU/CPU standard)
    pub span_y: f64,    // hauteur totale (pour compatibilité GPU/CPU standard)
    
    // Coordonnées haute précision (String) pour préserver la précision arbitraire
    // Utilisées pour les calculs GMP aux zooms profonds (>10^15)
    // Si None, les valeurs f64 sont utilisées (compatibilité GPU/CPU standard)
    pub center_x_hp: Option<String>,
    pub center_y_hp: Option<String>,
    pub span_x_hp: Option<String>,
    pub span_y_hp: Option<String>,
}
```
Avantage: evite la soustraction de grands nombres proches lors de zooms profonds (>1e15).
Les coordonnées haute précision en String préservent la précision arbitraire pour les calculs GMP,
évitant les pertes de précision lors de la conversion f64 → GMP aux zooms très profonds (>10^16).

## Dispatch rendu (escape_time.rs)

```
AlgorithmMode::Auto → should_use_perturbation()?
  - GPU f32: pixel_size < 1e-5  → perturbation
  - CPU f64: pixel_size < 1e-13 → perturbation
  - Zoom extreme (pixel_size < 1e-15) → should_use_full_gmp_perturbation() → force GMP complet (render_perturbation_gmp_path)
  - sinon: f64 standard

Modes forces: StandardF64 | StandardDS | Perturbation | ReferenceGmp

Précision GMP:
  - Calcul automatique via compute_perturbation_precision_bits() pour perturbation
  - Utilise les String haute précision (center_x_hp, etc.) si disponibles
  - Toutes les opérations GMP utilisent la même précision calculée (pas le preset)
  - Propagation de précision garantie dans iterate_pixel_gmp() via Complex::with_val(prec, ...)
  - Les valeurs GMP sont créées explicitement avec la précision calculée pour éviter les incohérences
```

## Perturbation (deep zoom)

**Supporte**: Mandelbrot, Julia, BurningShip, Tricorn

Pipeline:
1. Orbite de reference au centre (GMP, precision auto-calculee via `compute_perturbation_precision_bits()`)
2. Table BLA precalculee pour sauter des iterations (desactivee pour zoom >10^15)
3. Pixels = delta par rapport a reference:
   - Zooms moyens: ComplexExp (mantisse f64 + exposant)
   - Zooms profonds (>10^15): GMP complet (`iterate_pixel_gmp()`)
4. Detection glitchs + correction en GMP

**Calcul dc** (offset pixel vs centre) - `perturbation/mod.rs:compute_dc_gmp()`:
- Utilise les String haute précision (`span_x_hp`, `span_y_hp`) si disponibles
- Sinon fallback sur f64 pour compatibilité
- Calcul direct en GMP: `dc = (i/width - 0.5) * span_x` (idem pour y)

**Précision GMP automatique** (`compute_perturbation_precision_bits()`):
- Par défaut (aligné référence C++ Fraktaler-3): `bits = max(24, 24 + floor(log2(zoom * height)))`, puis clamp 128..8192 (équivalent de  
  `prec = max(24, 24 + (par.zoom * par.p.image.height).exp)` dans Fraktaler-3 `param.cc`).
- Option politique conservative (`use_reference_precision_formula = false`): `log2(zoom) + safety_margin`, plage 128–8192 bits.

**Référence et alignement C++ (Fraktaler-3):**
- **Référence:** Code C++ Fraktaler-3 (`fraktaler-3-3.1/src`) — comportement cible pour la perturbation.
- **Aligné:** Rebasing (condition, fin d’orbite, phase), formule de fusion BLA avec `cref_norm` = |c| (équivalent du scalaire `c` dans `merge(..., c)` C++), limites séparées `max_perturb_iterations` / `max_bla_steps` (PerturbIterations / BLASteps).
- **Écarts assumés:** Précision par défaut = formule C++ (alignée référence). Calcul de dc peut utiliser GMP aux zooms extrêmes (référence C++ reste en précision limitée).

**Cache** (`ReferenceOrbitCache`): orbite + BLA reutilises si meme centre (comparaison en String GMP)/type/precision.

## Parametres perturbation (FractalParams)

| Champ | Description | Defaut |
|-------|-------------|--------|
| `bla_threshold` | seuil delta pour activer BLA | 1e-8 |
| `bla_validity_scale` | multiplicateur rayon BLA (>1 = agressif) | 1.0 |
| `glitch_tolerance` | tolerance Pauldelbrot | 1e-4 |
| `series_order` | ordre serie (0=off, 1=lin, 2=quad) | 0 |
| `series_threshold` | seuil delta pour serie | 1e-6 |
| `series_error_tolerance` | erreur max serie | 1e-10 |
| `series_standalone` | approximation serie standalone (sans BLA) | false |
| `glitch_neighbor_pass` | detection voisinage | true |
| `max_secondary_refs` | nombre max references secondaires (0=off, 3=recom) | 3 |
| `min_glitch_cluster_size` | taille min cluster pour reference secondaire | 100 |
| `multibrot_power` | puissance z^d + c | 2.5 |
| `max_perturb_iterations` | cap itérations perturbation (0 = illimité; aligné C++ PerturbIterations) | 1024 |
| `max_bla_steps` | cap pas BLA (0 = illimité; aligné C++ BLASteps) | 1024 |
| `use_reference_precision_formula` | précision = formule C++ (24 + exp(zoom*height)) | true |

## Types de fractales (--type N)

| ID | Type | Algo |
|----|------|------|
| 1 | Von Koch | vectoriel |
| 2 | Dragon | vectoriel |
| 3 | Mandelbrot | escape-time + perturbation |
| 4 | Julia | escape-time + perturbation |
| 5 | Julia Sin | f64/GMP |
| 6 | Newton | f64/GMP |
| 7 | Phoenix | f64/GMP |
| 8 | Buffalo | f64/GMP |
| 9 | Barnsley Julia | f64/GMP |
| 10 | Barnsley Mandelbrot | f64/GMP |
| 11 | Magnet Julia | f64/GMP |
| 12 | Magnet Mandelbrot | f64/GMP |
| 13 | Burning Ship | escape-time + perturbation |
| 14 | Tricorn | f64/GMP |
| 15 | Mandelbulb (2D power 8) | f64/GMP |
| 16 | Buddhabrot | special |
| 17 | Lyapunov | special (presets) |
| 18 | Perpendicular Burning Ship | f64/GMP |
| 19 | Celtic | f64/GMP |
| 20 | Alpha Mandelbrot | f64/GMP |
| 21 | Pickover Stalks | f64/GMP |
| 22 | Nova | f64/GMP |
| 23 | Multibrot | f64/GMP |
| 24 | Nebulabrot | special |

## CLI

```
--type N              # type fractale (1-24)
--width/height        # dimensions image
--center_x/center_y   # centre plan complexe
--xmin/xmax/ymin/ymax # bornes (convertis en center+span)
--iterations          # max iterations
--palette 0-8         # palette couleurs
--color_repeat        # repetitions gradient (2-40)
--gmp                 # force precision arbitraire
--precision-bits      # bits GMP (defaut 256)
--algorithm           # auto|f64|standard|ds|double-single|perturbation|perturb|gmp|referencegmp|reference-gmp
--bla_threshold       # seuil BLA (ex: 1e-8)
--bla_validity_scale  # scale BLA (ex: 2.0)
--glitch_tolerance    # tolerance glitch (ex: 1e-4)
--multibrot_power     # puissance Multibrot (defaut 2.5, ex: 3.0)
--lyapunov_preset     # standard|zircon-city|jellyfish|asymmetric|spaceship|heavy-blocks
--output FILE         # fichier PNG sortie
```

## GPU (wgpu)

Shaders disponibles:
- `mandelbrot_f32.wgsl` / `julia_f32.wgsl` / `burning_ship_f32.wgsl`: precision simple
- `mandelbrot_f64.wgsl` / `julia_f64.wgsl` / `burning_ship_f64.wgsl`: double (si supporte)
- `mandelbrot_ds.wgsl`: Double-Single (emule f64 avec 2x f32)
- `perturbation.wgsl`: perturbation GPU

Selection automatique selon zoom et support materiel.

## GUI (FractallApp)

- Rendu progressif multi-passes (preview → full)
- Coordonnees haute precision:
  - Stockees en String (`center_x_hp`, `center_y_hp`, `span_x_hp`, `span_y_hp`)
  - Synchronisees vers `FractalParams` via `sync_hp_to_params()`
  - Les String sont stockees directement dans `FractalParams` pour preserver la precision
  - Conversion String → rug::Float pour calculs GMP aux zooms profonds (>10^15)
  - Conversion String → f64 pour compatibilite GPU/CPU standard
- Selection rectangulaire pour zoom
- Cache orbite/BLA entre re-rendus
- Switch CPU/GPU
- Affichage stats (centre, iterations, zoom) sous la barre de menu
- Selection palette avec apercu visuel (image gradient)
