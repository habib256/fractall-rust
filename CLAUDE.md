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
    pub center_x: f64,  // centre X
    pub center_y: f64,  // centre Y
    pub span_x: f64,    // largeur totale
    pub span_y: f64,    // hauteur totale
}
```
Avantage: evite la soustraction de grands nombres proches lors de zooms profonds (>1e15).

## Dispatch rendu (escape_time.rs)

```
AlgorithmMode::Auto → should_use_perturbation()?
  - GPU f32: pixel_size < 1e-5  → perturbation
  - CPU f64: pixel_size < 1e-13 → perturbation
  - Zoom extreme (pixel_size < 1e-15) → force GMP (pas perturbation)
  - sinon: f64 standard

Modes forces: StandardF64 | StandardDS | Perturbation | ReferenceGmp
```

## Perturbation (deep zoom)

**Supporte**: Mandelbrot, Julia, BurningShip

Pipeline:
1. Orbite de reference au centre (GMP, precision auto-calculee)
2. Table BLA precalculee pour sauter des iterations
3. Pixels = delta par rapport a reference (ComplexExp: mantisse f64 + exposant)
4. Detection glitchs + correction en GMP

**Calcul dc** (offset pixel vs centre) - `perturbation/mod.rs:205-226`:
```rust
let dc_re = (i as f64 * inv_width - 0.5) * x_range;
let dc_im = (j as f64 * inv_height - 0.5) * y_range;
```

**Cache** (`ReferenceOrbitCache`): orbite + BLA reutilises si meme centre/type/precision.

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
- Coordonnees haute precision (String → rug::Float pour zooms >1e15)
- Selection rectangulaire pour zoom
- Cache orbite/BLA entre re-rendus
- Switch CPU/GPU
