# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build both CLI and GUI binaries (release mode)
cargo build --release

# Build CLI only
cargo build --release --bin fractall-cli

# Build GUI only
cargo build --release --bin fractall-gui

# Run CLI (example: Mandelbrot)
cargo run --release --bin fractall-cli -- --type 3 --width 1920 --height 1080 --output mandelbrot.png

# Run GUI
cargo run --release --bin fractall-gui
```

**Note:** The `rug` crate requires native GMP, MPFR, and MPC libraries installed on the system.

## Architecture

### Module Structure

```
src/
├── main.rs           # CLI entry (clap-based argument parsing)
├── main_gui.rs       # GUI entry (egui/eframe)
├── fractal/          # Core fractal computation
│   ├── types.rs      # FractalType enum, FractalParams struct
│   ├── definitions.rs# Default parameters per fractal type
│   ├── iterations.rs # f64 iteration functions (19 fractal types)
│   └── gmp.rs        # GMP/MPFR high-precision iterations
├── render/
│   └── escape_time.rs# Rendering dispatcher (f64 vs GMP path)
├── color/
│   └── palettes.rs   # 9 color palettes with gradient interpolation
├── gui/
│   ├── app.rs        # Main egui application state
│   └── texture.rs    # Texture conversion utilities
└── io/
    └── png.rs        # PNG export with parallel colorization
```

### Dual Precision Paths

The renderer has two paths selected by `params.use_gmp`:
- **f64 path** (`iterations.rs`): Standard double precision, parallelized with Rayon
- **GMP path** (`gmp.rs`): Arbitrary precision via `rug` crate, sequential execution

### Data Flow

1. `FractalParams` configured with type defaults from `definitions.rs`
2. `render_escape_time(&params)` returns `(Vec<u32>, Vec<Complex64>)` - iterations and final z values
3. `save_png()` colorizes in parallel and writes output

### Key Types

- `FractalType`: Enum for 19 fractal types (Mandelbrot=3, Julia=4, etc.)
- `FractalParams`: Complete render configuration (dimensions, bounds, iterations, colors, GMP settings)
- `FractalResult`: Iteration result with count and final z value

## CLI Arguments

| Argument | Description |
|----------|-------------|
| `--type N` | Fractal type (3-23) |
| `--width`, `--height` | Output dimensions |
| `--xmin/xmax/ymin/ymax` | Complex plane bounds |
| `--center-x/y` | Re-center view |
| `--iterations` | Max iteration override |
| `--palette` | Color palette (0-8, default 6=Plasma) |
| `--color-repeat` | Gradient repetitions (2-40) |
| `--gmp` | Enable high-precision mode |
| `--precision-bits` | GMP precision (default 256) |
| `--output` | Output PNG path |

## Fractal Types

Types 1-2 are vector fractals (not implemented in Rust yet). Types 3-23 are escape-time or special:
- 3: Mandelbrot, 4: Julia, 5: Julia Sin, 6: Newton, 7: Phoenix
- 8: Buffalo, 9-10: Barnsley J/M, 11-12: Magnet J/M
- 13: Burning Ship, 14: Tricorn, 15: Mandelbulb
- 17: Lyapunov Zircon City (special algorithm, not escape-time)
- 18: Perp. Burning Ship, 19: Celtic, 20: Alpha Mandelbrot
- 21: Pickover Stalks, 22: Nova, 23: Multibrot
