# CLAUDE.md

Guide rapide pour aider un LLM a trouver les bons fichiers et flux dans ce depot.

## Commandes de build et run

```bash
# Build CLI + GUI (release)
cargo build --release

# Build CLI uniquement
cargo build --release --bin fractall-cli

# Build GUI uniquement
cargo build --release --bin fractall-gui

# Run CLI (ex: Mandelbrot)
cargo run --release --bin fractall-cli -- --type 3 --width 1920 --height 1080 --output mandelbrot.png

# Run GUI
cargo run --release --bin fractall-gui
```

**Note:** `rug` exige GMP/MPFR/MPC installes localement.

## Carte du code (chemins clefs)

```
src/
├── main.rs            # CLI (clap)
├── main_gui.rs        # GUI (egui/eframe)
├── fractal/           # coeur des algos
│   ├── types.rs       # FractalType, FractalParams, AlgorithmMode
│   ├── definitions.rs # parametres par defaut
│   ├── iterations.rs  # escape-time en f64
│   ├── gmp.rs         # precision arbitraire via rug
│   ├── lyapunov.rs    # fractale de Lyapunov
│   ├── vectorial.rs   # Von Koch, Dragon
│   ├── buddhabrot.rs  # Buddhabrot, Nebulabrot
│   └── perturbation/  # algorithme perturbation (deep zoom)
│       ├── mod.rs     # render_perturbation_with_cache()
│       ├── orbit.rs   # ReferenceOrbitCache, orbite de reference
│       ├── bla.rs     # BlaTable (Bilinear Approximation)
│       ├── delta.rs   # iteration delta (perturbation)
│       └── types.rs   # ComplexExp (mantisse + exposant)
├── render/
│   └── escape_time.rs # dispatcher f64/GMP/perturbation
├── color/
│   └── palettes.rs    # palettes et interpolation
├── gui/
│   ├── app.rs         # etat et UI egui, orbit_cache
│   ├── progressive.rs # rendu progressif multi-resolution
│   └── texture.rs     # conversion image -> texture
├── gpu/
│   ├── mod.rs         # pipeline GPU (wgpu), perturbation GPU
│   └── *.wgsl         # shaders (mandelbrot, perturbation)
└── io/
    └── png.rs         # export PNG parallelise
```

## Points d entree (ou aller selon la question)

- Ajout d un type de fractale: `fractal/types.rs`, `fractal/definitions.rs`
- Algo special (non escape-time): `fractal/vectorial.rs`, `fractal/buddhabrot.rs`, `fractal/lyapunov.rs`
- Escape-time standard: `fractal/iterations.rs` + `render/escape_time.rs`
- Deep zoom / perturbation: `fractal/perturbation/` (orbit.rs, bla.rs, delta.rs)
- Cache d'orbite: `fractal/perturbation/orbit.rs` (`ReferenceOrbitCache`)
- Seuil de bascule perturbation: `render/escape_time.rs` (`should_use_perturbation`)
- Precision arbitraire: `fractal/gmp.rs` et `params.use_gmp`
- Couleurs / palette: `color/palettes.rs`
- Export PNG: `io/png.rs`
- GUI: `gui/app.rs` + `gui/progressive.rs`
- GPU / shaders: `gpu/mod.rs` + `gpu/*.wgsl`

## Chemins de rendu

Le renderer choisit l algo selon `FractalType` et `AlgorithmMode`:

- **Special** (non escape-time): `vectorial.rs`, `buddhabrot.rs`, `lyapunov.rs`
- **Escape-time** (Mandelbrot, Julia, BurningShip):
  - **Standard f64**: `iterations.rs` (Rayon) - zoom jusqu'a ~1e14
  - **Perturbation**: `perturbation/` - deep zoom au-dela de 1e14
  - **GMP**: `gmp.rs` via `params.use_gmp` - precision arbitraire

## Algorithme Perturbation (Deep Zoom)

Pour les zooms profonds (>1e14), la precision f64 ne suffit plus. L'algorithme perturbation:

1. Calcule une **orbite de reference** au centre (GMP haute precision)
2. Calcule les pixels comme **delta** par rapport a cette reference (f64)
3. Utilise **BLA** (Bilinear Approximation) pour sauter des iterations
4. Corrige les **glitchs** en parallele avec Rayon

**Cache d'orbite** (`ReferenceOrbitCache`):
- L'orbite et la table BLA sont cachees entre les frames
- Validite: meme centre, meme type, precision >= , iteration_max >=
- Accelere les re-rendus au meme point (quasi-instantane)

**Seuils de bascule** (`should_use_perturbation`):
- GPU f32: pixel_size < 1e-6 * scale
- CPU f64: pixel_size < 1e-14 * scale

## Flux principal de donnees

1. `FractalParams` (defaults via `definitions.rs`)
2. `render_escape_time(&params)` -> `(Vec<u32>, Vec<Complex64>)`
3. `save_png()` colorise et ecrit l image

## Arguments CLI

| Argument | Description |
|----------|-------------|
| `--type N` | Type de fractale (1-24) |
| `--width`, `--height` | Dimensions |
| `--xmin/xmax/ymin/ymax` | Bornes du plan complexe |
| `--center-x/y` | Re-centrage |
| `--iterations` | Max iterations |
| `--palette` | Palette (0-8, defaut 6) |
| `--color-repeat` | Repetitions gradient (2-40) |
| `--gmp` | Active precision arbitraire |
| `--precision-bits` | Precision GMP (defaut 256) |
| `--output` | Chemin PNG |

## Types de fractales

**Vectorielles**:
- 1: Von Koch, 2: Dragon

**Escape-time**:
- 3: Mandelbrot, 4: Julia, 5: Julia Sin, 6: Newton, 7: Phoenix
- 8: Buffalo, 9-10: Barnsley J/M, 11-12: Magnet J/M
- 13: Burning Ship, 14: Tricorn, 15: Mandelbulb
- 18: Perp. Burning Ship, 19: Celtic, 20: Alpha Mandelbrot
- 21: Pickover Stalks, 22: Nova, 23: Multibrot

**Speciaux**:
- 16: Buddhabrot
- 17: Lyapunov Zircon City
- 24: Nebulabrot
