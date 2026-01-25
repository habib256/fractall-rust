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
│   ├── iterations.rs # f64 iteration functions (escape-time fractals)
│   ├── gmp.rs        # GMP/MPFR high-precision iterations
│   ├── lyapunov.rs   # Lyapunov exponent fractal (Zircon City)
│   ├── vectorial.rs  # Von Koch and Dragon vectorial fractals
│   └── buddhabrot.rs # Buddhabrot and Nebulabrot algorithms
├── render/
│   └── escape_time.rs# Rendering dispatcher (f64 vs GMP vs special paths)
├── color/
│   └── palettes.rs   # 9 color palettes with gradient interpolation
├── gui/
│   ├── app.rs        # Main egui application state
│   └── texture.rs    # Texture conversion utilities
└── io/
    └── png.rs        # PNG export with parallel colorization
```

### Rendering Paths

The renderer dispatches to different algorithms based on fractal type:

**Special algorithms** (custom rendering, not escape-time):
- `VonKoch`, `Dragon` → `vectorial.rs` (L-system based vector fractals)
- `Buddhabrot`, `Nebulabrot` → `buddhabrot.rs` (orbit density visualization)
- `Lyapunov` → `lyapunov.rs` (Lyapunov exponent calculation)

**Escape-time fractals** (all other types):
- **f64 path** (`iterations.rs`): Standard double precision, parallelized with Rayon
- **GMP path** (`gmp.rs`): Arbitrary precision via `rug` crate, selected by `params.use_gmp`

### Data Flow

1. `FractalParams` configured with type defaults from `definitions.rs`
2. `render_escape_time(&params)` returns `(Vec<u32>, Vec<Complex64>)` - iterations and final z values
3. `save_png()` colorizes in parallel and writes output

### Key Types

- `FractalType`: Enum for 24 fractal types (VonKoch=1 through Nebulabrot=24)
- `FractalParams`: Complete render configuration (dimensions, bounds, iterations, colors, GMP settings)
- `FractalResult`: Iteration result with count and final z value

## CLI Arguments

| Argument | Description |
|----------|-------------|
| `--type N` | Fractal type (1-24) |
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

**Vectorial fractals** (L-system based):
- 1: Von Koch, 2: Dragon

**Escape-time fractals**:
- 3: Mandelbrot, 4: Julia, 5: Julia Sin, 6: Newton, 7: Phoenix
- 8: Buffalo, 9-10: Barnsley J/M, 11-12: Magnet J/M
- 13: Burning Ship, 14: Tricorn, 15: Mandelbulb
- 18: Perp. Burning Ship, 19: Celtic, 20: Alpha Mandelbrot
- 21: Pickover Stalks, 22: Nova, 23: Multibrot

**Special algorithms**:
- 16: Buddhabrot (orbit density)
- 17: Lyapunov Zircon City (Lyapunov exponent)
- 24: Nebulabrot (color orbit density)
