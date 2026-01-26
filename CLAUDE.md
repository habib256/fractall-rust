# CLAUDE.md

## Build & Run

```bash
cargo build --release                    # CLI + GUI
cargo run --release --bin fractall-cli -- --type 3 --width 1920 --height 1080 --output out.png
cargo run --release --bin fractall-gui
```

Requiert GMP/MPFR/MPC installes (pour `rug`).

## Architecture

```
src/
├── main.rs / main_gui.rs      # CLI (clap) / GUI (egui)
├── fractal/
│   ├── types.rs               # FractalType, FractalParams, AlgorithmMode
│   ├── iterations.rs          # escape-time f64
│   ├── gmp.rs                  # precision arbitraire (rug)
│   ├── perturbation/          # deep zoom (>1e14)
│   │   ├── mod.rs             # render_perturbation_with_cache()
│   │   ├── orbit.rs           # ReferenceOrbitCache
│   │   ├── bla.rs             # BlaTable (Bilinear Approximation)
│   │   ├── delta.rs           # iteration delta
│   │   └── series.rs          # approximation par series
│   └── [lyapunov|buddhabrot|vectorial].rs
├── render/escape_time.rs      # dispatcher selon AlgorithmMode
├── gpu/                       # wgpu + shaders WGSL
└── gui/                       # app.rs, progressive.rs
```

## Dispatch rendu (escape_time.rs)

```
AlgorithmMode::Auto → should_use_perturbation() ?
  - GPU f32: pixel_size < 1e-6 * scale  → perturbation
  - CPU f64: pixel_size < 1e-14 * scale → perturbation
  - sinon: f64 standard

Modes forces: StandardF64 | StandardDS (GPU) | Perturbation | ReferenceGmp
```

## Perturbation (deep zoom)

**Supporte**: Mandelbrot, Julia, BurningShip

1. Orbite de reference au centre (GMP)
2. Pixels = delta par rapport a reference (f64 + ComplexExp)
3. BLA pour sauter des iterations
4. Correction glitchs en GMP

**Calcul dc** (offset pixel vs centre) - `perturbation/mod.rs:200-221`:
```rust
let dc_re = (i as f64 * inv_width - 0.5) * x_range;
let dc_im = (j as f64 * inv_height - 0.5) * y_range;
```
Formule normalisee qui evite la soustraction de grands nombres proches.

**Cache** (`ReferenceOrbitCache`): orbite + BLA reutilises si meme centre/type/precision.

## Types de fractales (--type N)

| ID | Type | Algo |
|----|------|------|
| 1-2 | Von Koch, Dragon | vectoriel |
| 3 | Mandelbrot | escape-time + perturbation |
| 4 | Julia | escape-time + perturbation |
| 13 | Burning Ship | escape-time + perturbation |
| 5-12,14-15,18-23 | Autres escape-time | f64/GMP uniquement |
| 16,24 | Buddhabrot, Nebulabrot | special |
| 17 | Lyapunov | special |

## CLI essentiels

```
--type N          # 1-24
--width/height    # dimensions
--xmin/xmax/ymin/ymax  # bornes plan complexe
--iterations      # max iterations
--palette 0-8     # couleurs
--gmp             # precision arbitraire
--precision-bits  # bits GMP (defaut 256)
```
