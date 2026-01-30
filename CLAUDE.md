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
│   ├── types.rs         # FractalType, FractalParams, AlgorithmMode, ColorSpace (Serialize/Deserialize)
│   ├── definitions.rs   # constantes par type + LyapunovPreset
│   ├── iterations.rs    # escape-time f64
│   ├── gmp.rs           # precision arbitraire (rug/mpc)
│   ├── lyapunov.rs      # Lyapunov exponent (Serialize/Deserialize)
│   ├── buddhabrot.rs    # Buddhabrot/Nebulabrot
│   ├── vectorial.rs     # Von Koch, Dragon
│   ├── orbit_traps.rs   # Orbit trap detection (Serialize/Deserialize)
│   └── perturbation/
│       ├── mod.rs       # render_perturbation_cancellable_with_reuse()
│       ├── types.rs     # ComplexExp, FloatExp (mantisse + exposant)
│       ├── orbit.rs     # ReferenceOrbitCache
│       ├── bla.rs       # BlaTable (Bilinear Approximation)
│       ├── delta.rs     # iterate_pixel()
│       ├── series.rs    # Taylor series approximation
│       ├── distance.rs  # Distance estimation (dual numbers)
│       ├── interior.rs  # Interior detection (ExtendedDualComplex)
│       ├── nonconformal.rs # Non-conformal BLA (Tricorn, Burning Ship)
│       └── glitch.rs    # Glitch cluster detection
├── render/
│   ├── mod.rs
│   └── escape_time.rs   # dispatcher + should_use_gmp_reference()
├── gpu/
│   ├── mod.rs           # GpuRenderer (wgpu)
│   ├── mandelbrot_f32.wgsl / mandelbrot_f64.wgsl
│   ├── julia_f32.wgsl / julia_f64.wgsl
│   ├── burning_ship_f32.wgsl / burning_ship_f64.wgsl
│   └── perturbation.wgsl
├── gui/
│   ├── mod.rs
│   ├── app.rs           # FractallApp (egui) + drag-and-drop
│   ├── progressive.rs   # rendu multi-passes
│   └── texture.rs
├── color/
│   ├── mod.rs
│   ├── palettes.rs      # 13 palettes predefinies
│   └── color_models.rs  # RGB, HSB, LCH conversions
└── io/
    ├── mod.rs
    └── png.rs           # save_png_with_metadata(), load_png_metadata()
```

## Systeme de coordonnees

Le code utilise **center + span** au lieu de xmin/xmax/ymin/ymax:
```rust
pub struct FractalParams {
    pub center_x: f64,              // centre X (GPU/CPU standard)
    pub center_y: f64,              // centre Y
    pub span_x: f64,                // largeur totale
    pub span_y: f64,                // hauteur totale

    // Haute precision (String) pour zooms >10^15
    pub center_x_hp: Option<String>,
    pub center_y_hp: Option<String>,
    pub span_x_hp: Option<String>,
    pub span_y_hp: Option<String>,
}
```
Avantage: evite la soustraction de grands nombres proches lors de zooms profonds.

## Metadonnees PNG

Les images PNG generees contiennent les parametres complets de la fractale dans un chunk tEXt:
- Cle: `fractall-params`
- Valeur: JSON serialise de `FractalParams` (incluant coordonnees HP)

**Fonctions** (`io/png.rs`):
- `save_png_with_metadata()`: Sauvegarde PNG + metadonnees JSON
- `load_png_metadata()`: Charge les FractalParams depuis un PNG

**Drag-and-drop**: Glisser un PNG sur la fenetre GUI restaure l'etat exact de la fractale.

## Dispatch rendu (escape_time.rs)

```
AlgorithmMode::Auto:
  - Zooms 1e1 - 1e16: CPU f64 standard (rapide)
  - Zooms > 1e16: GMP reference (precision necessaire)
  - Perturbation f64 desactivee en Auto (trop lente)

Modes forces: StandardF64 | Perturbation | ReferenceGmp

Precision GMP:
  - Calcul auto via compute_perturbation_precision_bits()
  - Utilise les String HP si disponibles
  - Propagation precision via Complex::with_val(prec, ...)
```

## Perturbation (deep zoom)

**Supporte**: Mandelbrot, Julia, BurningShip, Tricorn

**Pipeline**:
1. Orbite reference au centre (GMP, precision auto)
2. Table BLA pour sauter des iterations
3. Pixels = delta par rapport a reference (ComplexExp ou GMP)
4. Detection glitchs + correction

**Modules specialises**:
- `distance.rs`: Estimation distance via DualComplex (differentiation auto)
- `interior.rs`: Detection interieur via ExtendedDualComplex (5 composantes)
- `nonconformal.rs`: BLA matriciel pour Tricorn/Burning Ship (valeurs singulieres)
- `glitch.rs`: Clustering de pixels glitches (flood-fill) + references secondaires

**Precision GMP** (`compute_perturbation_precision_bits()`):
- Formule C++ Fraktaler-3: `bits = max(24, 24 + floor(log2(zoom * height)))`, clamp 128..8192
- Option conservative: `log2(zoom) + margin`

**Cache** (`ReferenceOrbitCache`): orbite + BLA reutilises si meme centre/type/precision.

## Parametres perturbation

| Champ | Description | Defaut |
|-------|-------------|--------|
| `bla_threshold` | seuil delta BLA | 1e-8 |
| `bla_validity_scale` | multiplicateur rayon BLA | 1.0 |
| `glitch_tolerance` | tolerance Pauldelbrot | 1e-4 |
| `series_order` | ordre serie (0=off) | 0 |
| `max_secondary_refs` | references secondaires (0=off) | 3 |
| `min_glitch_cluster_size` | taille min cluster | 100 |
| `max_perturb_iterations` | cap iterations | 1024 |
| `max_bla_steps` | cap pas BLA | 1024 |
| `use_reference_precision_formula` | formule C++ | true |

## Couleur

**Espaces couleur** (color_models.rs):
- RGB: standard
- HSB: Teinte-Saturation-Luminosite (interpolation circulaire)
- LCH: Luminance-Chroma-Hue via CIE Lab (perceptuellement uniforme)

**Orbit traps** (orbit_traps.rs):
- Types: Point, Line, Cross, Circle
- Tracking distance minimale sur l'orbite

## Types de fractales (--type N)

| ID | Type | Algo |
|----|------|------|
| 1-2 | Von Koch, Dragon | vectoriel |
| 3 | Mandelbrot | escape-time + perturbation |
| 4 | Julia | escape-time + perturbation |
| 5-12 | Julia Sin, Newton, Phoenix, Buffalo, Barnsley, Magnet | f64/GMP |
| 13 | Burning Ship | escape-time + perturbation |
| 14 | Tricorn | escape-time + perturbation |
| 15-23 | Mandelbulb, Celtic, Alpha, Pickover, Nova, Multibrot... | f64/GMP |
| 16, 24 | Buddhabrot, Nebulabrot | special |
| 17 | Lyapunov | special (6 presets) |

## CLI

```
--type N              # type fractale (1-24)
--width/height        # dimensions
--center-x/center-y   # centre
--iterations          # max iterations
--palette 0-12        # palette
--color_repeat        # repetitions gradient
--algorithm           # auto|f64|perturbation|gmp
--precision-bits      # bits GMP (defaut 256)
--bla_threshold       # seuil BLA
--glitch_tolerance    # tolerance glitch
--multibrot_power     # puissance Multibrot
--lyapunov_preset     # standard|zircon-city|jellyfish|asymmetric|spaceship|heavy-blocks
--output FILE         # PNG sortie (avec metadonnees)
```

## GPU (wgpu)

Shaders:
- `mandelbrot_f32/f64.wgsl`, `julia_f32/f64.wgsl`, `burning_ship_f32/f64.wgsl`
- `perturbation.wgsl`

Selection automatique selon zoom et support materiel.

## GUI (FractallApp)

- Rendu progressif multi-passes (preview -> full)
- Coordonnees HP en String, sync vers FractalParams
- Selection rectangulaire pour zoom
- Cache orbite/BLA entre re-rendus
- Switch CPU/GPU
- Stats: centre, iterations, zoom
- Apercu palettes
- **Drag-and-drop**: Glisser un PNG pour restaurer l'etat
- **Sauvegarde (S)**: PNG avec metadonnees integrees

## Raccourcis clavier GUI

| Touche | Action |
|--------|--------|
| F1-F12 | Changer type fractale |
| C | Cycler palette |
| R | Cycler color_repeat |
| S | Screenshot PNG (avec metadonnees) |
| +/= | Zoom avant |
| - | Zoom arriere |
| 0 | Reset vue |
