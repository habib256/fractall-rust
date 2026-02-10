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
│   ├── types.rs         # FractalType, FractalParams, AlgorithmMode, ColorSpace, PlaneTransform
│   ├── definitions.rs   # constantes par type + LyapunovPreset
│   ├── iterations.rs    # escape-time f64
│   ├── gmp.rs           # precision arbitraire (rug/mpc)
│   ├── lyapunov.rs      # Lyapunov exponent
│   ├── buddhabrot.rs    # Buddhabrot/Nebulabrot
│   ├── vectorial.rs     # Von Koch, Dragon
│   ├── orbit_traps.rs   # Orbit trap detection (Point, Line, Cross, Circle)
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
│   └── perturbation.wgsl  # BLA cache, series approx, adaptive glitch
├── gui/
│   ├── mod.rs
│   ├── app.rs           # FractallApp (egui) + drag-and-drop + HQ render
│   ├── progressive.rs   # rendu multi-passes
│   └── texture.rs
├── color/
│   ├── mod.rs
│   ├── palettes.rs      # 13 palettes predefinies (0-12)
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

**Palettes** (13 disponibles, index 0-12):
Fire, Ocean, Forest, Violet, Rainbow, Sunset, Plasma, Ice, Cosmic, Neon, Twilight, Emboss, Waves

**Espaces couleur** (color_models.rs):
- RGB: standard
- HSB: Teinte-Saturation-Luminosite (interpolation circulaire)
- LCH: Luminance-Chroma-Hue via CIE Lab (perceptuellement uniforme)

**Modes de colorisation** (OutColoringMode, 15 modes):
| Mode | Description |
|------|-------------|
| Iterations | Couleur basee sur nombre d'iterations |
| IterReal | Iterations + partie reelle de z |
| IterImag | Iterations + partie imaginaire de z |
| IterRealImag | Iterations + re/im |
| IterAll | Combinaison complete |
| Binary | Noir/blanc binaire |
| Biomorphs | Motifs biologiques |
| Potential | Potentiel electrique |
| ColorDecomp | Decomposition par angle |
| Smooth | Lissage logarithmique (defaut) |
| OrbitTraps | Distance aux pieges geometriques |
| Wings | Motifs ailes via sinh() |
| Distance | Gradient base sur distance estimee |
| DistanceAO | Distance + ambient occlusion |
| Distance3D | Effet 3D via gradient distance |

**Orbit traps** (orbit_traps.rs):
| Type | Description |
|------|-------------|
| Point | Distance a l'origine |
| Line | Distance a une ligne (angle configurable) |
| Cross | Distance a une croix H+V |
| Circle | Distance a un cercle (centre, rayon) |

## Transformations de plan (XaoS-style)

| ID | Nom | Formule |
|----|-----|---------|
| 0 | Mu | c (normal) |
| 1 | Inversion | 1/c |
| 2 | InversionShifted | 1/(c + 0.25) |
| 3 | Lambda | 4*c*(1-c) |
| 4 | InversionLambda | 1/(4*c*(1-c)) |
| 5 | InversionLambdaMinus1 | 1/(4*c*(1-c)) - 1 |
| 6 | InversionSpecial | 1/(c - 1.40115) |

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
| 25-31 | Burning Ship Julia, Tricorn Julia, Celtic Julia, Buffalo Julia, Multibrot Julia, Perp. Burning Ship Julia, Alpha Mandelbrot Julia | f64/GMP |

**Paires Mandelbrot / Julia** (preview Julia et touche J en GUI): Mandelbrot↔Julia, Barnsley↔Barnsley Julia, Magnet↔Magnet Julia, Burning Ship↔Burning Ship Julia, Tricorn↔Tricorn Julia, Celtic↔Celtic Julia, Buffalo↔Buffalo Julia, Multibrot↔Multibrot Julia, Perpendicular Burning Ship↔Perp. Burning Ship Julia, Alpha Mandelbrot↔Alpha Mandelbrot Julia.

## CLI

```
# Base
--type N              # type fractale (1-24 standard, 25-31 variantes Julia)
--width/height        # dimensions
--center-x/center-y   # centre
--iterations          # max iterations
--output FILE         # PNG sortie (avec metadonnees)

# Couleur
--palette 0-12        # palette (13 disponibles)
--color-repeat        # repetitions gradient (1-120)
--outcoloring MODE    # mode colorisation (smooth, distance, orbit-traps, wings...)
--color-space         # rgb|hsb|lch

# Algorithme
--algorithm           # auto|f64|perturbation|gmp
--precision-bits      # bits GMP (defaut 256)
--plane N             # transformation de plan (0-6)

# Perturbation
--bla-threshold       # seuil BLA
--glitch-tolerance    # tolerance glitch

# Features avancees
--enable-distance-estimation  # estimation distance (dual numbers)
--enable-interior-detection   # detection interieur
--interior-threshold          # seuil interieur (defaut 0.001)
--enable-orbit-traps          # activer orbit traps

# Specifiques
--multibrot-power     # puissance Multibrot
--lyapunov-preset     # standard|zircon-city|jellyfish|asymmetric|spaceship|heavy-blocks
```

## GPU (wgpu)

Shaders:
- `mandelbrot_f32/f64.wgsl`, `julia_f32/f64.wgsl`, `burning_ship_f32/f64.wgsl`
- `perturbation.wgsl` (BLA cache workgroup, series approx, adaptive glitch tolerance)

Selection automatique selon zoom et support materiel.

## GUI (FractallApp)

**Menu Type** (racine):
- Mandelbrots a la racine: Mandelbrot, Barnsley Mandelbrot, Magnet Mandelbrot, Burning Ship, Perp. Burning Ship, Tricorn, Celtic, Buffalo, Multibrot, Alpha Mandelbrot
- Dossier **Julia all** (apres Alpha Mandelbrot): toutes les variantes Julia
- Separateur puis Mandelbulb, puis Julia Sin, Newton, Phoenix, Pickover Stalks, Nova
- Dossiers Densite (Buddhabrot, Nebulabrot) et Lyapunov (presets)
- Vector (Von Koch, Dragon) retire du menu

**Rendu**:
- Rendu progressif multi-passes (preview -> full)
- Recolorisation asynchrone (ne bloque pas l'UI lors du changement de palette/color_repeat)
- Cache orbite/BLA entre re-rendus

**Fonctionnalites**:
- Coordonnees HP en String, sync vers FractalParams
- Selection rectangulaire pour zoom
- Switch CPU/GPU
- Stats: centre, iterations, zoom
- Apercu palettes
- Preview Julia au survol (types ayant une variante Julia) + touche J pour basculer en vue Julia

**Rendu haute resolution**:
- Presets: Window, 4K (3840x2160), 8K (7680x4320)
- Rendu asynchrone avec barre de progression

**Import/Export**:
- **Drag-and-drop**: Glisser un PNG pour restaurer l'etat
- **Sauvegarde (S)**: PNG avec metadonnees integrees

## Raccourcis clavier GUI

| Touche | Action |
|--------|--------|
| F1-F12 | Changer type fractale (F1=Mandelbrot, F2=Julia...) |
| C | Cycler palette (0-12) |
| R | Cycler color_repeat (+1, max 120) |
| S | Screenshot PNG (avec metadonnees) |
| +/= | Zoom avant (1.5x au centre) |
| - | Zoom arriere (1.5x) |
| 0 | Reset vue par defaut |
| Souris | Selection rectangle pour zoom |

## Threading

**Rendu progressif**: Thread dedie pour chaque passe, communication via mpsc channels.

**Recolorisation**: Thread separe pour eviter de bloquer l'UI lors des changements de palette/color_repeat. Systeme de versioning pour ignorer les resultats obsoletes si l'utilisateur change rapidement le slider.

**Rendu HQ**: Thread dedie avec messages de progression (Progress/Done/Error).

## Bugs connus restants (non corriges)

Les bugs suivants ont ete identifies mais necessitent une analyse plus approfondie:

1. **BLA table off-by-one** (`perturbation/bla.rs`): `start_idx=1` au level 1 cause un decalage d'index entre la table BLA et les requetes `level_nodes[n]`. Impact visible aux zooms profonds.
2. **GMP perturbation z_ref stale** (`perturbation/delta.rs:885-1062`): `iterate_pixel_gmp` utilise `z_ref[n]` apres avoir calcule `delta_{n+1}`. Le compteur `n` est incremente trop tard.
3. **Series skip non-fonctionnel pour Mandelbrot** (`perturbation/series.rs`): La serie utilise `delta_0 = 0` pour Mandelbrot, donc `compute_series_skip` retourne toujours `None`. La table est construite inutilement.
4. **Burning Ship BLA sign manquant en dual-number** (`perturbation/delta.rs:1233-1238`): La transformation de signe pour Burning Ship n'est pas appliquee dans le path dual-number (distance estimation).
5. **Fausse detection interieur avec perturbation** (`gui/app.rs`): `interior_flag_encoded` infere la detection interieur depuis `!distances.is_empty()` mais la perturbation alloue toujours un vecteur distances.
6. **Reuse progressif sans orbit/distance** (`render/escape_time.rs`): Les pixels reutilises n'ont pas de donnees orbit/distance, causant un motif damier avec Distance/OrbitTraps.
7. **Coordonnees pixel asymetriques** (`render/escape_time.rs`): Le mapping `i/width` au lieu de `(i+0.5)/width` cree un decalage d'un demi-pixel.
8. **Preview palette ignore color space** (`color/palettes.rs`): `generate_palette_preview` utilise toujours RGB, pas HSB/LCH.
