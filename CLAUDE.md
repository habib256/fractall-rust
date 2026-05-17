# CLAUDE.md

> Carte rapide du repo pour reprendre une session sans relire tout le code.

## Build & Run

```bash
cargo build --release
cargo run --release --bin fractall-cli -- --type 3 --width 1920 --height 1080 --output out.png
cargo run --release --bin fractall-gui
```

Prérequis natifs : GMP / MPFR / MPC (pour `rug`).

## Tests

- **Unit tests** : `cargo test --release --bin fractall-cli`. Couvre `perturbation/{bla,delta,series,nonconformal,distance,interior,glitch,orbit,types,mod}`, `lyapunov`, `progressive`, et tout le `bytecode/` (`compile`, `interp{,_gmp}`, `bla_dual`, `delta_form`, `pixel_loop{,_exp}`).
- **Golden image tests** (`tests/golden_images.rs`) : 10 cas couvrant les paths f64 standard, perturbation, deep zoom GMP et types non-bytecode. Le binaire `fractall-cli` est invoqué pour chaque cas, le PNG décodé, et les pixels comparés à `tests/golden/<name>.png` — une régression d'un seul pixel fait échouer le test.
  - Lancer : `cargo test --release --test golden_images`
  - Régénérer (après modification intentionnelle) : `FRACTALL_UPDATE_GOLDENS=1 cargo test --release --test golden_images`. **Toujours** ouvrir visuellement les nouveaux PNG avant de les commit.
  - À utiliser comme garde-fou pour tout refactor du moteur (perturbation, BLA, color pipeline, bytecode P3.1, etc.).

Pas de pipeline CI/CD (cf. TODO P2.1).

## Architecture

```
src/
├── main.rs              # CLI (clap)
├── main_gui.rs          # GUI (egui/eframe)
├── fractal/
│   ├── mod.rs           # exports + default_params_for_type()
│   ├── types.rs         # FractalType, FractalParams, AlgorithmMode, ColorSpace, PlaneTransform
│   ├── definitions.rs   # constantes par type + LyapunovPreset
│   ├── iterations.rs    # escape-time f64 + dispatch bytecode unifié (iterate_via_bytecode)
│   ├── gmp.rs           # précision arbitraire (rug / mpc)
│   ├── lyapunov.rs      # Lyapunov exponent
│   ├── buddhabrot.rs    # Buddhabrot / Nebulabrot / Anti-Buddhabrot
│   ├── vectorial.rs     # Von Koch, Dragon
│   ├── orbit_traps.rs   # Orbit trap detection (Point, Line, Cross, Circle)
│   ├── bytecode/        # Moteur unifié Fraktaler-3 (P3.1, défaut)
│   │   ├── mod.rs          # Op, Phase, Formula (mono / hybride multi-phase)
│   │   ├── compile.rs      # compile_formula(type, multibrot_power) -> Formula
│   │   ├── interp.rs       # iterate_bytecode_f64 (path standard)
│   │   ├── interp_gmp.rs   # interpréteur GMP pour orbite référence
│   │   ├── delta_form.rs   # DeltaState (f64) / DeltaStateExp (ComplexExp) — règle de chaîne par opcode
│   │   ├── bla_dual.rs     # BLA mat2 unifié construit via dual-numbers (BlaTableUnified, merge F3)
│   │   ├── pixel_loop.rs   # BLA mat2 + delta-form + rebasing F3 (f64)
│   │   ├── pixel_loop_exp.rs # idem avec ComplexExp (deep zoom > 1e13)
│   │   └── tests.rs        # tests d'invariance + intégration
│   └── perturbation/    # Path GMP deep zoom (legacy + ponts vers bytecode)
│       ├── mod.rs          # render_perturbation_cancellable_with_reuse() — dispatcher
│       ├── types.rs        # ComplexExp, FloatExp (mantisse + exposant)
│       ├── orbit.rs        # ReferenceOrbit, ReferenceOrbitCache, HybridBlaReferences
│       ├── bla.rs          # BlaTable conformal (utilisée par path GMP deep zoom)
│       ├── nonconformal.rs # BLA matriciel pour Tricorn/Burning Ship (path GMP)
│       ├── delta.rs        # iterate_pixel{,_gmp} — passe par bytecode si use_bytecode_engine
│       ├── series.rs       # Taylor series approximation
│       ├── glitch.rs       # Clustering Pauldelbrot (path legacy uniquement)
│       └── debug_pure_f3.rs # Helpers de debug pour le path F3 pur
├── render/
│   ├── mod.rs
│   └── escape_time.rs   # dispatcher + should_use_perturbation() / should_use_gmp_reference()
├── gpu/
│   ├── mod.rs              # GpuRenderer (wgpu) + pipeline bytecode (P3.1 #7)
│   ├── mandelbrot_f32.wgsl # path standard f32
│   ├── julia_f32.wgsl
│   ├── burning_ship_f32.wgsl
│   ├── perturbation.wgsl   # BLA cache, series approx, adaptive glitch
│   ├── bytecode_kernel.wgsl # runtime bytecode (étend la couverture GPU à BS/Tricorn/Celtic/Buffalo/PerpBS/Multibrot sans shader dédié)
│   └── bytecode_kernel_test.rs # tests d'équivalence GPU vs CPU bytecode
├── gui/
│   ├── mod.rs
│   ├── app.rs           # FractallApp (egui) — menu, drag-and-drop, HQ render, raccourcis
│   ├── progressive.rs   # rendu multi-passes
│   └── texture.rs
├── color/
│   ├── mod.rs
│   ├── palettes.rs      # 27 palettes prédéfinies (0-26)
│   └── color_models.rs  # RGB, HSB, LCH conversions
└── io/
    ├── mod.rs
    └── png.rs           # save_png_with_metadata(), load_png_metadata()
```

## Dépendances principales (Cargo.toml)

| Crate | Version | Rôle |
|-------|---------|------|
| clap | 4.5 | Parsing CLI (derive) |
| image | 0.25 | Support PNG |
| num-complex | 0.4 | Arithmétique complexe |
| rayon | 1.11 | Parallélisme multi-thread |
| eframe | 0.29 | Framework GUI (wrapper egui, backend wgpu) |
| egui | 0.29 | UI immediate mode |
| wgpu | 22.1 | GPU (Vulkan/Metal/DX12) |
| pollster | 0.3 | Runtime async pour GPU |
| bytemuck | 1.15 | Sérialisation buffers GPU |
| rug | 1.24 | Précision arbitraire (GMP/MPFR/MPC) |
| serde / serde_json | 1.0 | Sérialisation JSON (métadonnées PNG) |
| png | 0.17 | Accès chunks PNG bas niveau |
| toml | 0.8 | Loader de fichiers de paramètres (cf. `toml/`) |
| naga (dev) | 22.1 | Validation des shaders WGSL en tests |

Deux binaires : `fractall-cli` (`src/main.rs`) et `fractall-gui` (`src/main_gui.rs`).

## Système de coordonnées

Le code utilise **center + span** plutôt que xmin/xmax/ymin/ymax :

```rust
pub struct FractalParams {
    pub center_x: f64,              // centre X (GPU/CPU standard)
    pub center_y: f64,
    pub span_x: f64,                // largeur totale
    pub span_y: f64,

    // Haute précision (String GMP) pour zooms > 10^15
    pub center_x_hp: Option<String>,
    pub center_y_hp: Option<String>,
    pub span_x_hp: Option<String>,
    pub span_y_hp: Option<String>,
}
```

Avantage : évite la soustraction de grands nombres proches lors des zooms profonds.

## Métadonnées PNG

Les PNG générés contiennent l'intégralité des paramètres dans un chunk `tEXt` :
- Clé : `fractall-params`
- Valeur : JSON sérialisé de `FractalParams` (incluant coordonnées HP)

API (`io/png.rs`) :
- `save_png_with_metadata()` : PNG + métadonnées JSON
- `load_png_metadata()` : restaure `FractalParams` depuis un PNG

**Drag-and-drop** : glisser un PNG sur la fenêtre GUI restaure l'état exact.

## Dispatch rendu (`render/escape_time.rs`)

```
AlgorithmMode::Auto :
  - Types spéciaux :
    - VonKoch, Dragon                                  -> rendu vectoriel
    - Buddhabrot / Nebulabrot / Anti-Buddhabrot        -> rendu densité (f64 ou GMP)
    - Lyapunov                                         -> calcul exposant (f64 ou GMP)
  - Types escape-time (Mandelbrot, Julia, BurningShip, Tricorn) :
    - should_use_perturbation() (zoom > ~1e13)         -> perturbation + BLA
    - should_use_gmp_reference() (zoom > 1e16)         -> GMP reference
    - sinon                                            -> CPU f64 standard
  - Perturbation incompatible avec plane_transform != Mu -> fallback f64/GMP
  - Autres types                                       -> f64 ou GMP selon use_gmp

Modes forcés : StandardF64 | Perturbation | ReferenceGmp
```

### Moteur bytecode (défaut depuis P3.1 Session E)

Si `params.use_bytecode_engine` (true par défaut) ET `compile_formula(type, power)`
réussit, le pixel est servi par le path bytecode unifié :

- **CPU f64 standard** : `iterations.rs::iterate_via_bytecode` (Mandelbrot, Julia,
  BS, Tricorn, Celtic, Buffalo, PerpBS, Multibrot puiss. entière, + variantes Julia).
  Supporte distance estimation, interior detection, orbit traps via tracking de `dz`.
- **CPU perturbation** : `perturbation/delta.rs::try_bytecode_unified_path` →
  `bytecode/pixel_loop.rs::iterate_pixel_unified_full` (f64) ou
  `pixel_loop_exp.rs::iterate_pixel_unified_exp` (ComplexExp, deep zoom > 1e13).
  Rebasing F3 strict (`|Z[m+1] + δ|² < |δ|²`) remplace la glitch detection
  Pauldelbrot legacy.
- **GPU** : `gpu/mod.rs::try_render_bytecode` → `bytecode_kernel.wgsl`. Encode
  `Formula` en `Vec<u32>` storage buffer, interprété en parallèle. Couvre toute
  la famille escape-time supportée par le bytecode sur n'importe quel device wgpu.

Fallback automatique sur le path legacy si : type non compilable, `--no-bytecode`,
ou path GMP deep zoom + features avancées qui ne sont pas encore portées sur
pixel_loop (cas marginal).

### Précision GMP perturbation

Formule C++ Fraktaler-3 : `bits = max(24, 24 + floor(log2(zoom * height)))`,
clamp `[128, 8192]`. Option conservative : `log2(zoom) + margin`. Voir
`compute_perturbation_precision_bits()`.

**Cache** (`ReferenceOrbitCache`) : orbite + BLA réutilisées si même
centre / type / précision.

**Documentation détaillée** : `perturbation.md` à la racine et `docs/fraktaler-3-analysis.md`.

## Paramètres perturbation

| Champ | Description | Défaut |
|-------|-------------|--------|
| `bla_threshold` | seuil delta BLA | 1e-8 |
| `bla_validity_scale` | multiplicateur rayon BLA | 1.0 |
| `glitch_tolerance` | tolerance Pauldelbrot (path legacy) | 1e-4 |
| `series_order` | ordre série (0 = off) | 2 |
| `max_secondary_refs` | références secondaires (0 = off, path legacy) | 3 |
| `min_glitch_cluster_size` | taille min cluster (path legacy) | 100 |
| `max_perturb_iterations` | cap itérations par pixel | 1024 |
| `max_bla_steps` | cap pas BLA par pixel | 1024 |
| `use_reference_precision_formula` | formule C++ F3 | true |
| `use_bytecode_engine` | path unifié BLA mat2 + rebasing F3 | true |
| `jitter_scale` | sub-pixel AA (0 = off, 1 = full pixel) | 0.0 |

## Couleur

**Palettes** (27, index 0-26) : Fire, Ocean, Forest, Violet, Rainbow, Sunset,
**Plasma** (défaut), Ice, Cosmic, Neon, Twilight, Emboss, Waves, SynthRed,
LightYears, Blues, Coffee, Classic, Dimensions, Earth, FireIce, Habs, Jays, Slice,
Stardust, Strobe, SynthBlue.

**Espaces couleur** (`color_models.rs`) :
- RGB : standard
- HSB : Teinte / Saturation / Brillance (interpolation circulaire)
- LCH : Luminance / Chroma / Hue via CIE Lab (perceptuellement uniforme)

**Modes de colorisation** (`OutColoringMode`, 15 modes) :

| Mode | Description |
|------|-------------|
| Iter | Couleur basée sur nombre d'itérations |
| IterPlusReal / IterPlusImag / IterPlusRealImag / IterPlusAll | Iter + composantes de z |
| BinaryDecomposition | Noir/blanc selon signe z.im |
| Biomorphs | Motifs biologiques |
| Potential | Potentiel électrique |
| ColorDecomposition | Décomposition par angle |
| Smooth | Lissage logarithmique (**défaut**) |
| OrbitTraps | Distance aux pièges géométriques |
| Wings | Motifs ailes via sinh() |
| Distance | Gradient distance |
| DistanceAO | Distance + ambient occlusion |
| Distance3D | Effet 3D via gradient distance |

**Orbit traps** (`orbit_traps.rs`) : Point, Line, Cross, Circle.

## Transformations de plan (XaoS-style)

| ID | Nom | Formule |
|----|-----|---------|
| 0 | Mu | c (normal) |
| 1 | Inversion | 1/c |
| 2 | InversionShifted | 1/(c + 0.25) |
| 3 | Lambda | 4·c·(1-c) |
| 4 | InversionLambda | 1/(4·c·(1-c)) |
| 5 | InversionLambdaMinus1 | 1/(4·c·(1-c)) - 1 |
| 6 | InversionSpecial | 1/(c - 1.40115) |

## Types de fractales (`--type N`)

| ID | Type | Algo |
|----|------|------|
| 1  | Von Koch | vectoriel |
| 2  | Dragon | vectoriel |
| 3  | Mandelbrot | bytecode + perturbation |
| 4  | Julia | bytecode + perturbation |
| 5  | Julia Sin | f64 / GMP |
| 6  | Newton | f64 / GMP |
| 7  | Phoenix | f64 / GMP |
| 8  | Buffalo | bytecode |
| 9  | Barnsley Julia | f64 / GMP |
| 10 | Barnsley Mandelbrot | f64 / GMP |
| 11 | Magnet Julia | f64 / GMP |
| 12 | Magnet Mandelbrot | f64 / GMP |
| 13 | Burning Ship | bytecode + perturbation |
| 14 | Tricorn | bytecode + perturbation |
| 15 | Mandelbulb | f64 / GMP |
| 16 | Buddhabrot | densité |
| 17 | Lyapunov | spécial (6 presets) |
| 18 | Perpendicular Burning Ship | bytecode |
| 19 | Celtic | bytecode |
| 20 | Alpha Mandelbrot | f64 / GMP |
| 21 | Pickover Stalks | f64 / GMP |
| 22 | Nova | f64 / GMP |
| 23 | Multibrot | bytecode (puissances entières) sinon f64/GMP |
| 24 | Nebulabrot | densité |
| 25 | Burning Ship Julia | bytecode |
| 26 | Tricorn Julia | bytecode |
| 27 | Celtic Julia | bytecode |
| 28 | Buffalo Julia | bytecode |
| 29 | Multibrot Julia | bytecode (puissances entières) |
| 30 | Perp. Burning Ship Julia | bytecode |
| 31 | Alpha Mandelbrot Julia | f64 / GMP |
| 32 | Mandelbrot Sin | f64 / GMP |
| 33 | Anti-Buddhabrot | densité |

**Paires Mandelbrot ↔ Julia** (preview Julia + touche J en GUI) :
Mandelbrot↔Julia, Barnsley↔Barnsley Julia, Magnet↔Magnet Julia,
Burning Ship↔Burning Ship Julia, Tricorn↔Tricorn Julia, Celtic↔Celtic Julia,
Buffalo↔Buffalo Julia, Multibrot↔Multibrot Julia,
Perpendicular Burning Ship↔Perp. Burning Ship Julia,
Alpha Mandelbrot↔Alpha Mandelbrot Julia, Mandelbrot Sin↔Julia Sin.

## CLI

```
# Base
--type N                     # type fractale (1-33)
--width / --height           # dimensions (défaut 1920×1080)
--center-x / --center-y      # centre (f64)
--center-x-hp / --center-y-hp # centre haute précision (string, deep zooms > 10^15)
--zoom ZOOM                  # magnification (span = 4/zoom), notation scientifique OK (ex: 1.41e219)
--xmin / --xmax / --ymin / --ymax # alternative au centre+span
--iterations N               # max itérations
--output FILE                # PNG sortie (avec métadonnées)

# Couleur
--palette 0-26               # défaut 6 = Plasma
--color-repeat N             # répétitions gradient (1-120)
--outcoloring MODE           # smooth (défaut), iter, distance, orbit-traps, wings...
                             # cf. OutColoringMode::from_cli_name pour la liste exhaustive

# Algorithme
--algorithm                  # auto | f64 | perturbation | gmp
--precision-bits N           # bits GMP (défaut 256)
--plane N                    # transformation de plan (0-6 ou alias)
--no-bytecode                # désactive le moteur bytecode (path legacy glitch detection)

# Perturbation
--bla-threshold              # seuil BLA
--bla-validity-scale         # multiplicateur rayon BLA
--glitch-tolerance           # tolerance glitch (path legacy)

# Features avancées
--enable-distance-estimation # estimation distance (dual numbers, défaut off)
--enable-interior-detection  # détection intérieur
--interior-threshold         # seuil intérieur (défaut 0.001)
--gpu                        # rendu GPU (wgpu Metal/Vulkan/DX12)

# Spécifiques
--multibrot-power            # puissance Multibrot (défaut 2.5)
--lyapunov-preset            # standard | zircon-city | jellyfish | asymmetric | spaceship | heavy-blocks
```

## GPU (wgpu)

**Shaders** (workgroup 16×16) :
- `mandelbrot_f32.wgsl`, `julia_f32.wgsl`, `burning_ship_f32.wgsl` — paths legacy f32 par type
- `perturbation.wgsl` — BLA cache workgroup, série, glitch tolerance adaptative
- `bytecode_kernel.wgsl` — runtime bytecode unifié (P3.1 Task 7)

Tout est en **f32 GPU** (les shaders f64 ont été retirés). Pour deep zoom > ~10⁷
le CPU prend le relais via la perturbation + GMP.

**Sélection backend** :
- macOS : Metal
- Linux : Vulkan (prioritaire), puis OpenGL
- Windows : DirectX12 et Vulkan

## GUI (`FractallApp`)

### Menu Type
- Mandelbrots à la racine : Mandelbrot, Barnsley Mandelbrot, Magnet Mandelbrot,
  Burning Ship, Perp. Burning Ship, Tricorn, Celtic, Buffalo, Multibrot,
  Alpha Mandelbrot, Mandelbrot Sin
- Dossier **Julia all** : toutes les variantes Julia
- Mandelbulb, Julia Sin, Newton, Phoenix, Pickover Stalks, Nova en racine après séparateur
- Dossier **Densité** : Buddhabrot, Nebulabrot, Anti-Buddhabrot
- Dossier **Lyapunov** : 6 presets
- Von Koch / Dragon retirés du menu (toujours accessibles via CLI `--type 1/2`)

### Rendu
- Rendu progressif multi-passes (preview → full)
- Recolorisation asynchrone (changement palette/color_repeat sans bloquer l'UI)
- Cache orbite + BLA entre re-rendus

### Fonctionnalités
- Coordonnées HP (String) synchronisées vers `FractalParams`
- Sélection rectangulaire pour zoom
- Switch CPU / GPU
- Stats : centre, itérations, zoom
- Aperçu palettes
- Preview Julia au survol (types ayant une variante Julia) + touche `J` pour basculer

### Rendu haute résolution
Presets Window, 4K (3840×2160), 8K (7680×4320). Rendu asynchrone avec barre de
progression.

### Import / Export
- **Drag-and-drop** : glisser un PNG pour restaurer l'état
- **Sauvegarde (S)** : PNG avec métadonnées intégrées

## Raccourcis clavier GUI

| Touche | Action |
|--------|--------|
| F1-F12 | Type fractale (F1=Mandelbrot, F2=Julia, F3=JuliaSin, …, F12=Tricorn) |
| C | Cycler palette (0-26) |
| R | Cycler color_repeat (+1, max 120 ou max 8 pour types densité) |
| J | Basculer en Julia (utilise le seed du preview) ou activer le mode preview |
| S | Screenshot PNG (avec métadonnées) |
| +/= | Zoom avant (×1.5 au centre) |
| - | Zoom arrière (×1.5) |
| 0 | Reset vue par défaut (ignoré si focus sur champ itérations) |
| Enter | Valider le champ itérations |
| Molette | Zoom avant/arrière |
| Clic gauche + drag | Sélection rectangle pour zoom |
| Clic droit | Zoom arrière |
| Clic milieu + drag | Déplacement (pan) |

Les raccourcis +/=/-/0 sont désactivés en mode preview Julia.

## Threading

- **Rendu progressif** : un thread par passe, communication via `mpsc`.
- **Recolorisation** : thread dédié + système de versioning pour ignorer les
  résultats obsolètes si l'utilisateur change rapidement le slider.
- **Rendu HQ** : thread dédié avec messages `Progress / Done / Error`.
- **Parallélisme CPU** : `rayon::par_chunks_mut`. `AtomicBool` pour l'annulation.
