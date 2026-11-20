use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use clap::Parser;
// Les modules vivent dans la bibliothèque `fractall_cli` (src/lib.rs) : les
// importer sous leur nom court conserve les chemins existants (`fractal::…`).
use fractall_cli::{fractal, gpu, io, render};

use fractal::{
    apply_lyapunov_preset, default_params_for_type, AlgorithmMode, FractalParams, FractalType,
    LyapunovPreset, OutColoringMode, PlaneTransform, ViewHp,
};
use gpu::GpuRenderer;
use io::exr::save_iterations_exr;
use io::f3_toml::save_f3_toml;
use io::png::save_png_rgb_with_metadata;

/// Command-line fractal renderer built on the fractall engine.
#[derive(Parser, Debug)]
#[command(
    name = "fractall-cli",
    about = "Command-line fractal renderer (Mandelbrot, Julia, Burning Ship, deep zooms, …)",
    after_help = "Examples:\n  \
        fractall-cli --type 3 --width 1920 --height 1080 --output mandelbrot.png\n  \
        fractall-cli --from-png view.png --width 3840 --height 2160 --output view_4k.png\n  \
        fractall-cli --type 3 --zoom 1e20 --iterations 20000 --output deep.png \\\n      \
        --center-x-hp -0.74364388703715870475219150611477 \\\n      \
        --center-y-hp 0.13182590420531197049313205638514",
    version,
    author = "Arnaud Verhille and contributors"
)]
struct Cli {
    /// Fractal type (3=Mandelbrot, 4=Julia, 5=JuliaSin, ..., 23=Multibrot).
    /// Optional with --toml (default: 3 = Mandelbrot, rust-fractal-core /
    /// toml/ corpus format) and with --from-png (the PNG's type is kept).
    #[arg(long = "type")]
    fractal_type: Option<u8>,

    /// Multi-phase hybrid: comma-separated escape-time types, iterated
    /// cyclically. E.g. `--phases mandelbrot,burning_ship` (alternating
    /// Mandel-Ship), `--phases mandelbrot,mandelbrot,burning_ship` (2×M then
    /// BS). CPU only. Overrides --type for the formula.
    #[arg(long)]
    phases: Option<String>,

    /// Opcode formula in Fraktaler-3 format: words
    /// `add sqr mul store absx absy negx negy rot{DEG}`, each `add` ends a
    /// phase. E.g. `--opcodes "sqr rot{15} add"` (Mandelbrot rotated inside
    /// the formula), `--opcodes "sqr add absx absy sqr add"` (Mandel-Ship).
    /// Takes precedence over --phases/--type for the formula. CPU only.
    #[arg(long)]
    opcodes: Option<String>,

    /// Output image width in pixels (default 1920; with --from-png: the PNG's
    /// width, or derived from --height and the PNG's aspect ratio)
    #[arg(long)]
    width: Option<u32>,

    /// Output image height in pixels (default 1080; with --from-png: the PNG's
    /// height, or derived from --width and the PNG's aspect ratio)
    #[arg(long)]
    height: Option<u32>,

    /// Centre X of the complex plane (optional, otherwise the type's default)
    #[arg(long, allow_hyphen_values = true)]
    center_x: Option<f64>,

    /// Centre Y of the complex plane (optional, otherwise the type's default)
    #[arg(long, allow_hyphen_values = true)]
    center_y: Option<f64>,

    /// High-precision centre X (decimal string, for deep zooms beyond 1e15).
    /// Negative values can be passed directly: --center-x-hp -1.76…
    #[arg(long, allow_hyphen_values = true)]
    center_x_hp: Option<String>,

    /// High-precision centre Y (decimal string, for deep zooms beyond 1e15)
    #[arg(long, allow_hyphen_values = true)]
    center_y_hp: Option<String>,

    /// Zoom (magnification, horizontal span = 4/zoom). Scientific notation
    /// accepted (e.g. 1.41e219)
    #[arg(long)]
    zoom: Option<String>,

    /// Minimum X of the complex plane (takes precedence over center/zoom)
    #[arg(long, allow_hyphen_values = true)]
    xmin: Option<f64>,

    /// Maximum X of the complex plane
    #[arg(long, allow_hyphen_values = true)]
    xmax: Option<f64>,

    /// Minimum Y of the complex plane
    #[arg(long, allow_hyphen_values = true)]
    ymin: Option<f64>,

    /// Maximum Y of the complex plane
    #[arg(long, allow_hyphen_values = true)]
    ymax: Option<f64>,

    /// Maximum iteration count: a number, or `auto`. When omitted, the count
    /// comes from --toml / --from-png if they provide one; otherwise it is
    /// chosen automatically by measuring the view on a small probe render
    /// (escape-time types; other types keep their default). `auto` forces the
    /// measurement even over a --toml / --from-png value.
    #[arg(long, value_name = "N|auto", value_parser = parse_iterations_arg)]
    iterations: Option<IterationsArg>,

    /// Colour palette (0 to 26, default 6 = Plasma)
    #[arg(long)]
    palette: Option<u8>,

    /// Colour gradient repetitions (2-40, even values recommended, default 40)
    #[arg(long)]
    color_repeat: Option<u32>,

    /// Enable arbitrary-precision computation with GMP (via rug)
    #[arg(long)]
    gmp: bool,

    /// GMP precision in bits, a floor for the automatic precision (e.g. 128,
    /// 256, 512; default 256)
    #[arg(long)]
    precision_bits: Option<u32>,

    /// Algorithm mode (auto, f64, perturbation, gmp)
    #[arg(long)]
    algorithm: Option<String>,

    /// Delta threshold for enabling BLA (e.g. 1e-8)
    #[arg(long)]
    bla_threshold: Option<f64>,

    /// BLA validity radius multiplier (1.0 = conservative, >1 = aggressive,
    /// e.g. 2.0)
    #[arg(long)]
    bla_validity_scale: Option<f64>,

    /// Glitch tolerance (e.g. 1e-4)
    #[arg(long)]
    glitch_tolerance: Option<f64>,

    /// Power for Multibrot (z^d + c), default 2.5
    #[arg(long)]
    multibrot_power: Option<f64>,

    /// Julia seed: real part of c (e.g. --julia-re=-0.8).
    /// No effect on Mandelbrot-like types.
    #[arg(long, allow_hyphen_values = true)]
    julia_re: Option<f64>,

    /// Julia seed: imaginary part of c (e.g. --julia-im=0.156).
    #[arg(long, allow_hyphen_values = true)]
    julia_im: Option<f64>,

    /// Lyapunov preset (standard, zircon-city, jellyfish, asymmetric,
    /// spaceship, heavy-blocks)
    #[arg(long)]
    lyapunov_preset: Option<String>,

    /// Colouring mode (iter, iter+real, iter+imag, iter+real/imag, iter+all,
    /// binary, biomorphs, potential, color-decomp, smooth, orbit-traps,
    /// wings, distance, distance-ao, distance-3d; default smooth)
    #[arg(long)]
    outcoloring: Option<String>,

    /// Complex-plane transformation (0=mu, 1=1/mu, 2=1/(mu+0.25), 3=lambda,
    /// 4=1/lambda, 5=1/lambda-1, 6=1/(mu-1.40115); default 0).
    /// Accepted values: 0-6 or names (mu, 1/mu, 1/(mu+0.25), lambda,
    /// 1/lambda, 1/lambda-1, 1/(mu-1.40115))
    #[arg(long)]
    plane: Option<String>,

    /// Enable distance estimation (dual numbers, extra cost)
    #[arg(long)]
    enable_distance_estimation: bool,

    /// Enable interior detection (extended dual numbers)
    #[arg(long)]
    enable_interior_detection: bool,

    /// Interior detection threshold (default 0.001)
    #[arg(long)]
    interior_threshold: Option<f64>,

    /// Force the GPU (overrides the automatic device choice). Without
    /// `--gpu`/`--no-gpu`, the device is chosen from the machine benchmark
    /// and correctness constraints (`wisdom::select_device`).
    #[arg(long)]
    gpu: bool,

    /// Force the CPU (overrides the automatic device choice).
    #[arg(long)]
    no_gpu: bool,

    /// [DEPRECATED] The bytecode engine is on by default; this flag is a
    /// no-op kept for backward compatibility. Use --no-bytecode to disable.
    #[arg(long, hide = true)]
    bytecode: bool,

    /// Disable the unified bytecode engine and use the legacy path (no
    /// Fraktaler-3 rebasing: pixels judged unreliable are recomputed in GMP
    /// one by one). For debugging or comparison only.
    #[arg(long)]
    no_bytecode: bool,

    /// Enable the nucleus finder (Mandelbrot only): refines the centre to an
    /// exact minibrot with Newton's method before building the reference
    /// orbit. Useful for escape-time deep zooms where the reference orbit
    /// escapes too early. Extra cost: a period scan plus 1-64 Newton steps,
    /// a few seconds at moderate depth, more at very deep zoom. Inspired by
    /// Fraktaler-3.1 `hybrid_center`.
    #[arg(long)]
    find_nucleus: bool,

    /// Enable the **double-double** (~106-bit) tier for deep-zoom Mandelbrot
    /// perturbation — a pure-Rust equivalent of Fraktaler-3's `float128`, for
    /// ultra-sensitive spirals where the f64 mantissa (53 bits) saturates.
    /// Slower (~10×); enable only if needed. Mandelbrot escape-time only.
    #[arg(long)]
    dd_tier: bool,

    /// Plane rotation in degrees (counter-clockwise), applied to the
    /// pixel→c mapping (Fraktaler-3 `transform.rotate`). Overrides the TOML
    /// value if given.
    #[arg(long, allow_hyphen_values = true)]
    rotation: Option<f64>,

    /// Anti-aliasing: number of jittered sub-pixel samples averaged
    /// (1 = off, default). Each sample shifts the grid by a low-discrepancy
    /// offset (Halton, port of F3 `radical_inverse` + `triangle` tent), then
    /// the coloured renders are averaged. Improves thin edges (especially the
    /// Distance modes). CPU and GPU.
    #[arg(long)]
    aa_samples: Option<u32>,

    /// Sub-pixel jitter amplitude for AA, in pixels. 1.0 = full-width tent
    /// (default when --aa-samples > 1), 0.5 = half a pixel. No effect when
    /// --aa-samples = 1.
    #[arg(long)]
    jitter_scale: Option<f64>,

    /// Escape radius (bailout) of the |z| ≥ bailout test. Defaults to the
    /// type's value (25 for the escape-time family). Used notably to match
    /// Fraktaler-3's escape radius in the parity harness.
    #[arg(long)]
    bailout: Option<f64>,

    /// Output PNG file (required, except with --wisdom-bench)
    #[arg(long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// ALSO write the raw render buffers (iterations, final z, distances if
    /// any) to a `.fmap` map: recolourable without recomputation via
    /// --from-map; exchange format of fractall-video keyframes.
    /// Incompatible with --aa-samples > 1 (AA output = averaged RGB).
    #[arg(long, value_name = "FILE.fmap")]
    output_map: Option<PathBuf>,

    /// Recolour an existing `.fmap` map WITHOUT recomputation: loads the
    /// map's buffers and parameters, applies the explicit colour overrides
    /// (--palette, --color-repeat, --outcoloring) and writes the --output
    /// PNG. Other rendering options are ignored.
    #[arg(long, value_name = "FILE.fmap", conflicts_with = "from_png")]
    from_map: Option<PathBuf>,

    /// Re-render the exact view saved in a fractall PNG (key S in
    /// fractall-gui, or any fractall-cli output): all parameters come from
    /// the PNG's embedded metadata, and every option given explicitly
    /// overrides them (--width/--height, --iterations, --palette,
    /// --algorithm, --aa-samples, …). Resizing keeps the centre and the
    /// vertical extent of the view; giving only --width or --height keeps
    /// the PNG's aspect ratio. E.g.
    /// `--from-png view.png --width 3840 --height 2160 --output big.png`.
    #[arg(long, value_name = "FILE.png", conflicts_with = "toml")]
    from_png: Option<PathBuf>,

    /// Benchmark the rendering techniques (CPU std/perturbation f64-exp-dd,
    /// plus GPU f32 if available) and store the measured throughputs
    /// (iters/s) in the per-machine wisdom file
    /// (~/.config/fractall/wisdom.toml, override FRACTALL_WISDOM_FILE), used
    /// by the automatic device and algorithm choice.
    #[arg(long, default_value_t = false)]
    wisdom_bench: bool,

    /// Load a lightweight TOML (real/imag/zoom) or a native Fraktaler-3 TOML
    /// (location.*, bailout.*, transform.*, [[formula]]). Explicit command-line
    /// options still take precedence.
    #[arg(long, value_name = "FILE")]
    toml: Option<PathBuf>,

    /// Export the raw iterations as EXR in Fraktaler-3 format (channels
    /// N0=uint32 iterations, NF=float smooth fraction), for apples-to-apples
    /// comparison via scripts/compare_f3.py. The --output PNG is still
    /// written.
    #[arg(long, value_name = "FILE.exr")]
    export_iterations: Option<PathBuf>,

    /// Export the resolved parameters as a native Fraktaler-3 3.1 TOML.
    #[arg(long, value_name = "FILE.toml")]
    export_f3_toml: Option<PathBuf>,
}

/// Valeur de `--iterations` : un nombre ou `auto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IterationsArg {
    Count(u32),
    Auto,
}

fn parse_iterations_arg(s: &str) -> Result<IterationsArg, String> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("auto") {
        return Ok(IterationsArg::Auto);
    }
    match s.parse::<u32>() {
        Ok(0) => Err("must be at least 1 (or `auto`)".into()),
        Ok(n) => Ok(IterationsArg::Count(n)),
        Err(_) => Err(format!("expected a positive integer or `auto`, got '{s}'")),
    }
}

/// D'où vient le plafond d'itérations du rendu (ordre de résolution de
/// `--iterations`, cf. `resolve_params`). `Auto` = la sonde
/// `wisdom::auto_iterations_with` le fixera dans `main`, AVANT le rendu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IterChoice {
    /// `--iterations N` explicite.
    User,
    /// Valeur de `--toml` / `--from-png`, `--iterations` absent.
    Source,
    /// Défaut du type (type sans plafond « escape-time »).
    Default,
    /// Sonde automatique ; `explicit` = `--iterations auto` écrit.
    Auto { explicit: bool },
}

impl IterChoice {
    fn wisdom_source(self) -> fractal::wisdom::IterationSource {
        use fractal::wisdom::IterationSource as S;
        match self {
            IterChoice::User => S::User,
            IterChoice::Source => S::Source,
            IterChoice::Default => S::Default,
            IterChoice::Auto { .. } => S::Auto,
        }
    }
}

/// Erreur de résolution des paramètres : message utilisateur + code de sortie
/// (2 = usage, 1 = valeur invalide / fichier illisible).
#[derive(Debug)]
struct CliError {
    code: i32,
    msg: String,
}

impl CliError {
    fn usage(msg: impl Into<String>) -> Self {
        Self { code: 2, msg: msg.into() }
    }
    fn invalid(msg: impl Into<String>) -> Self {
        Self { code: 1, msg: msg.into() }
    }
}

/// Paramètres finals d'un rendu CLI + réglages AA (hors `FractalParams`).
struct Resolved {
    params: FractalParams,
    aa_samples: u32,
    aa_jitter_scale: f64,
    /// Provenance du plafond d'itérations ; `Auto` ⇒ sonde avant le rendu.
    iterations: IterChoice,
    /// Le TOML a aligné `max_perturb_iterations` / `max_bla_steps` sur ses
    /// itérations : une valeur automatique doit les suivre (pas de cap TOML
    /// périmé à côté d'un plafond différent).
    caps_follow_iterations: bool,
}

/// Champs extraits d'un TOML de paramètres (format léger rust-fractal-core,
/// compatible avec le corpus `toml/`).
struct TomlParams {
    real: String,
    imag: String,
    zoom: String,
    iterations: Option<u32>,
    escape_radius: Option<f64>,
    rotate: Option<f64>,
    /// G4 jalon 5e : séquence de phases hybride, format `--phases` CLI
    /// (`"mandelbrot,burning_ship"`). Consommé par le harness (cas de parité
    /// hybride vs F3 natif — compare_f3.py émet les blocs `[[formula]]`).
    phases: Option<String>,
    /// G4 Op::Rot : formule opcodes agrégée depuis les blocs `[[formula]]`
    /// F3 natifs du TOML (structurés abs/neg/power compilés en mots, blocs
    /// `opcodes = "…"` repris tels quels), OU clé légère top-level
    /// `opcodes = "…"` (format `--opcodes`, consommée par compare_f3.py).
    /// `None` si aucun des deux.
    formula_opcodes: Option<String>,
}

fn load_toml_params(path: &std::path::Path) -> TomlParams {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error reading TOML {}: {}", path.display(), e);
        std::process::exit(1);
    });
    let table: toml::Table = content.parse().unwrap_or_else(|e| {
        eprintln!("Error parsing TOML {}: {}", path.display(), e);
        std::process::exit(1);
    });

    let scalar_string = |v: &toml::Value| -> Option<String> {
        if let Some(s) = v.as_str() {
            Some(s.to_string())
        } else if let Some(f) = v.as_float() {
            Some(f.to_string())
        } else if let Some(i) = v.as_integer() {
            Some(i.to_string())
        } else {
            None
        }
    };
    let take_str = |key: &str| -> Option<String> { table.get(key).and_then(&scalar_string) };
    let take_nested_str = |section: &str, key: &str| -> Option<String> {
        table
            .get(section)
            .and_then(|v| v.as_table())
            .and_then(|section| section.get(key))
            .and_then(&scalar_string)
    };

    let real = take_str("real").or_else(|| take_nested_str("location", "real")).unwrap_or_else(|| {
        eprintln!("TOML {}: missing 'real' field", path.display());
        std::process::exit(1);
    });
    let imag = take_str("imag").or_else(|| take_nested_str("location", "imag")).unwrap_or_else(|| {
        eprintln!("TOML {}: missing 'imag' field", path.display());
        std::process::exit(1);
    });
    let zoom = take_str("zoom").or_else(|| take_nested_str("location", "zoom")).unwrap_or_else(|| {
        eprintln!("TOML {}: missing 'zoom' field", path.display());
        std::process::exit(1);
    });
    let iterations = table.get("iterations").or_else(|| {
            table.get("bailout").and_then(|v| v.as_table()).and_then(|b| b.get("iterations"))
        })
        .and_then(|v| v.as_integer())
        .map(|i| {
            if i > u32::MAX as i64 {
                eprintln!(
                "TOML {}: iterations={} > u32::MAX, clamped to {}.",
                path.display(),
                i,
                u32::MAX
            );
                u32::MAX
            } else if i < 0 {
                eprintln!(
                    "TOML {}: negative iterations ({}), forced to 1024",
                    path.display(),
                    i
                );
                1024
            } else {
                i as u32
            }
        });
    let rotate = table
        .get("rotate")
        .or_else(|| table.get("transform").and_then(|v| v.as_table()).and_then(|t| t.get("rotate")))
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)));
    let escape_radius = table
        .get("bailout")
        .and_then(|v| v.as_table())
        .and_then(|b| b.get("escape_radius"))
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)));
    let phases = take_str("phases");

    // Blocs `[[formula]]` F3 natifs (G4 Op::Rot) : chaque bloc = une phase.
    // Champ `opcodes` prioritaire (seule syntaxe portant rot{θ}) ; sinon
    // compilation des champs structurés abs_x/abs_y/neg_x/neg_y/power dans
    // l'ordre F3 (`param.cc::compile_formula`) — le `store` du besoin mul est
    // ajouté au parse (`ensure_store_prefix`), pas ici.
    let formula_opcodes = table
        .get("formula")
        .and_then(|v| v.as_array())
        .map(|blocks| {
            let mut phases_words: Vec<String> = Vec::new();
            for block in blocks {
                let Some(b) = block.as_table() else { continue };
                if let Some(ops) = b.get("opcodes").and_then(|v| v.as_str()) {
                    let mut w = ops.trim().to_string();
                    if !w.ends_with("add") {
                        w.push_str(" add");
                    }
                    phases_words.push(w);
                    continue;
                }
                let flag = |k: &str| b.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
                let mut words: Vec<String> = Vec::new();
                if flag("abs_x") {
                    words.push("absx".into());
                }
                if flag("abs_y") {
                    words.push("absy".into());
                }
                if flag("neg_x") {
                    words.push("negx".into());
                }
                if flag("neg_y") {
                    words.push("negy".into());
                }
                let power = b
                    .get("power")
                    .and_then(|v| v.as_integer())
                    .unwrap_or(2)
                    .max(2) as u32;
                let highest_bit = 31 - power.leading_zeros();
                if power != (1u32 << highest_bit) {
                    words.push("store".into());
                }
                for bit in (0..highest_bit).rev() {
                    words.push("sqr".into());
                    if (power >> bit) & 1 == 1 {
                        words.push("mul".into());
                    }
                }
                words.push("add".into());
                phases_words.push(words.join(" "));
            }
            phases_words.join(" ")
        })
        .filter(|s| !s.trim().is_empty())
        // Clé légère top-level `opcodes` (mirror --opcodes) si pas de [[formula]].
        .or_else(|| take_str("opcodes").filter(|s| !s.trim().is_empty()));

    TomlParams {
        real,
        imag,
        zoom,
        iterations,
        escape_radius,
        rotate,
        phases,
        formula_opcodes,
    }
}

/// `--wisdom-bench` (G9.2) : mesure les débits effectifs par technique (CPU
/// std/perturbation f64-exp-dd via le dispatcher unique ; GPU f32 via
/// `render_dispatch` si un GPU est disponible) et persiste le fichier wisdom
/// machine. Cf. `fractal/wisdom_bench.rs` (modèle F3 `wisdom.cc`).
fn run_wisdom_bench() {
    const TARGET_SECONDS: f64 = 1.5;
    let cancel = AtomicBool::new(false);
    let gpu = GpuRenderer::new();
    // Une seule closure GPU générale (`render_dispatch` auto-dispatche std vs
    // perturbation selon le zoom de la frame) → `run_bench` s'en sert pour
    // gpu_std_f32 (vue shallow) ET gpu_perturb_f64 (vue e30), G9.5.
    let gpu_render = gpu.as_ref().map(|g| {
        move |params: &fractal::FractalParams| {
            g.render_dispatch(
                render::GpuRenderPlan::for_params(params),
                params,
                &cancel,
                None,
                None,
            )
            .map(|r| r.iterations)
        }
    });
    let file = match &gpu_render {
        Some(f) => fractal::wisdom_bench::run_bench(
            Some(f as &dyn Fn(&fractal::FractalParams) -> Option<Vec<u32>>),
            TARGET_SECONDS,
        ),
        None => {
            eprintln!("[WISDOM-BENCH] GPU unavailable — benchmarking the CPU only");
            fractal::wisdom_bench::run_bench(None, TARGET_SECONDS)
        }
    };
    match fractal::wisdom_bench::save(&file) {
        Ok(path) => println!("Machine wisdom written: {}", path.display()),
        Err(e) => {
            eprintln!("Failed to write the wisdom file: {e}");
            std::process::exit(1);
        }
    }
}

/// Chaînes HP de la vue à persister (repli sur les miroirs f64).
fn hp_strings(params: &FractalParams) -> [String; 4] {
    [
        params.center_x_hp.clone().unwrap_or_else(|| params.center_x.to_string()),
        params.center_y_hp.clone().unwrap_or_else(|| params.center_y.to_string()),
        params.span_x_hp.clone().unwrap_or_else(|| params.span_x.to_string()),
        params.span_y_hp.clone().unwrap_or_else(|| params.span_y.to_string()),
    ]
}

/// `--from-map` (G12 jalon 1) : recharge une map `.fmap`, applique les
/// overrides couleur explicites du CLI, recolorise et écrit le PNG. Zéro
/// rendu : c'est le chemin « recoloriser sans recalculer ».
fn run_from_map(cli: &Cli, map_path: &std::path::Path, output_path: &std::path::Path) {
    let map = match io::fmap::load_fmap(map_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading map {}: {}", map_path.display(), e);
            std::process::exit(1);
        }
    };
    let mut params = map.params.clone();
    if let Some(p) = cli.palette {
        params.color.color_mode = p;
    }
    if let Some(cr) = cli.color_repeat {
        params.color.color_repeat = cr.max(1);
    }
    if let Some(ref oc) = cli.outcoloring {
        match OutColoringMode::from_cli_name(oc) {
            Some(mode) => params.color.out_coloring_mode = mode,
            None => {
                eprintln!("Invalid colouring mode: '{oc}'");
                std::process::exit(1);
            }
        }
    }

    let [center_x_hp, center_y_hp, span_x_hp, span_y_hp] = hp_strings(&params);
    // Le canal `distances` de la map alimente les modes Distance*/DistanceAO/
    // Distance3D. Vérification G5 : un mode qui requiert un canal absent de
    // la map est une ERREUR (le .fmap ne persiste jamais `orbits`, et
    // `distances` seulement si le rendu l'a produit) — pas une retombée
    // silencieuse sur Smooth.
    let distances = map.distances.as_deref().unwrap_or(&[]);
    if let Err(e) =
        render::required_channels(&params).validate(map.iterations.len(), distances, &[])
    {
        eprintln!("Error: cannot recolour from this map — {e}");
        std::process::exit(1);
    }
    let buffer =
        io::png::colorize_to_rgb_with_extras(&params, &map.iterations, &map.zs, distances, &[]);
    if let Err(e) = save_png_rgb_with_metadata(
        &params,
        &buffer,
        output_path,
        &center_x_hp,
        &center_y_hp,
        &span_x_hp,
        &span_y_hp,
    ) {
        eprintln!("Error writing the PNG: {e}");
        std::process::exit(1);
    }
    println!(
        "PNG recoloured from {}: {}",
        map_path.display(),
        output_path.display()
    );
}

/// Résolution COMPLÈTE des paramètres de rendu depuis la ligne de commande :
/// point de départ (défauts du type, TOML ou PNG `--from-png`) puis overrides
/// explicites. Pure hors `--toml` (qui sort en erreur lui-même) : testée.
///
/// Avec `--from-png`, une option non fournie garde la valeur du PNG — c'est
/// pourquoi les options à défaut (`--width`, `--palette`, `--plane`,
/// `--precision-bits`, `--aa-samples`…) sont des `Option` et les drapeaux
/// booléens n'agissent que posés (ils ne peuvent pas désactiver un réglage
/// du PNG). Hors `--from-png`, le comportement historique est inchangé.
fn resolve_params(cli: &Cli) -> Result<Resolved, CliError> {
    let from_png = cli.from_png.is_some();
    // Une source (--from-png, ou --toml avec `iterations`) fournit-elle le
    // plafond d'itérations ? Si oui, `--iterations` absent la garde.
    let mut source_has_iterations = from_png;
    let mut caps_follow_iterations = false;

    let mut params = if let Some(ref png_path) = cli.from_png {
        let params = io::png::params_for_rerender(png_path, cli.width, cli.height)
            .map_err(CliError::invalid)?;
        if let Some(id) = cli.fractal_type {
            if FractalType::from_id(id) != Some(params.fractal_type) {
                eprintln!(
                    "--type {id} ignored: --from-png keeps the PNG's fractal type ({:?})",
                    params.fractal_type
                );
            }
        }
        params
    } else {
        // Si --toml est fourni sans --type, défaut Mandelbrot (le corpus toml/
        // ne contient que des Mandelbrot deep zoom au format rust-fractal-core).
        let fractal_type_id = match (cli.fractal_type, &cli.toml) {
            (Some(id), _) => id,
            (None, Some(_)) => 3,
            // --phases (hybride G4) implique la convention Mandelbrot (δ₀=0,
            // dc=pixel) → type 3 par défaut si --type/--toml absents.
            (None, None) if cli.phases.is_some() => 3,
            (None, None) => {
                return Err(CliError::usage(
                    "--type is required (or use --from-png <FILE.png> to re-render a saved view, \
                     or --toml <FILE> for the rust-fractal-core format)",
                ));
            }
        };
        let fractal_type = FractalType::from_id(fractal_type_id).ok_or_else(|| {
            CliError::invalid(format!(
                "Invalid fractal type: {fractal_type_id} (expected 1 to 33, see --help)"
            ))
        })?;
        let width = cli.width.unwrap_or(1920);
        let height = cli.height.unwrap_or(1080);
        if width == 0 || height == 0 {
            return Err(CliError::invalid(format!(
                "Error: --width/--height must be >= 1 (got {width}x{height})"
            )));
        }
        default_params_for_type(fractal_type, width, height)
    };

    // Hybride multi-phase (G4) : --phases type1,type2,… → hybrid_phases.
    if let Some(ref spec) = cli.phases {
        let mut phases = Vec::new();
        for name in spec.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
            match FractalType::from_hybrid_name(name) {
                Some(ft) => phases.push(ft),
                None => return Err(CliError::usage(format!("--phases: unknown type '{name}'"))),
            }
        }
        if !phases.is_empty() {
            params.formula.hybrid_phases = Some(phases);
        }
    }

    // Formule opcodes F3 (G4 Op::Rot) : --opcodes prioritaire sur --phases
    // pour la formule (précédence dans formula_for_params). Validation tôt.
    if let Some(ref ops) = cli.opcodes {
        if fractall_cli::fractal::bytecode::parse_opcodes_formula(ops).is_none() {
            return Err(CliError::usage(format!(
                "--opcodes: invalid formula '{ops}' (words: add sqr mul store absx absy negx \
                 negy rot{{DEG}}; each phase ends with add)"
            )));
        }
        params.formula.hybrid_opcodes = Some(ops.clone());
    }

    // Applique d'abord les paramètres TOML (les overrides CLI explicites
    // restent prioritaires car traités après).
    if let Some(ref toml_path) = cli.toml {
        let t = load_toml_params(toml_path);
        let view = ViewHp::from_center_and_zoom(
            &t.real,
            &t.imag,
            &t.zoom,
            params.width,
            params.height,
            1024,
        )
        .ok_or_else(|| {
            CliError::invalid(format!(
                "TOML {}: unreadable zoom: '{}'",
                toml_path.display(),
                t.zoom
            ))
        })?;
        view.write_to_params(&mut params);
        // Le centre décimal fourni est plus exact que sa matérialisation MPFR
        // finie. Le conserver textuellement pour la référence perturbée.
        params.center_x_hp = Some(t.real.clone());
        params.center_y_hp = Some(t.imag.clone());

        // Phases hybrides du TOML (G4 jalon 5e) — un --phases CLI explicite
        // reste prioritaire (déjà appliqué plus haut).
        if cli.phases.is_none() {
            if let Some(ref spec) = t.phases {
                let mut phases = Vec::new();
                for name in spec.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
                    match FractalType::from_hybrid_name(name) {
                        Some(ft) => phases.push(ft),
                        None => {
                            return Err(CliError::usage(format!(
                                "TOML {}: phases — unknown type '{name}'",
                                toml_path.display()
                            )));
                        }
                    }
                }
                if !phases.is_empty() {
                    params.formula.hybrid_phases = Some(phases);
                }
            }
        }

        // Blocs [[formula]] F3 natifs (G4 Op::Rot) — un --opcodes CLI
        // explicite reste prioritaire.
        if cli.opcodes.is_none() {
            if let Some(ref ops) = t.formula_opcodes {
                if fractall_cli::fractal::bytecode::parse_opcodes_formula(ops).is_none() {
                    return Err(CliError::usage(format!(
                        "TOML {}: invalid [[formula]] opcodes '{}'",
                        toml_path.display(),
                        ops
                    )));
                }
                params.formula.hybrid_opcodes = Some(ops.clone());
            }
        }

        if let Some(iters) = t.iterations {
            params.iteration_max = iters;
            // F3 batch sets maximum_perturb_iterations / maximum_bla_steps to the user
            // `iterations` field (see fraktaler-3-3.1/src/param.h:38 defaults + the F3
            // wrapper we write in scripts/compare_f3.py). Fractall's defaults (1024) cap
            // pixel iteration way below iter_max for deep zooms, collapsing every pixel
            // onto a single iter count (e.g. e1000 stopped at 1028 while F3 reached
            // 16616+). Mirror F3's semantics here so a TOML in toml/ produces the same
            // effective caps on both engines.
            params.perturbation.max_perturb_iterations = iters;
            params.perturbation.max_bla_steps = iters;
            source_has_iterations = true;
            caps_follow_iterations = true;
        }

        if cli.bailout.is_none() {
            if let Some(radius) = t.escape_radius {
                params.bailout = radius;
            }
        }

        if let Some(rot) = t.rotate {
            params.rotation = rot;
        }
    }

    // Override des coordonnées si demandé (bornes xmin/xmax/ymin/ymax).
    // Les bornes CLI sont converties en centre + span.
    // Utiliser center+span comme source de vérité au lieu de xmin/xmax/ymin/ymax
    let has_bounds =
        cli.xmin.is_some() || cli.xmax.is_some() || cli.ymin.is_some() || cli.ymax.is_some();
    if has_bounds {
        // Calculer les bornes depuis center+span pour les valeurs par défaut
        let default_xmin = params.center_x - params.span_x * 0.5;
        let default_xmax = params.center_x + params.span_x * 0.5;
        let default_ymin = params.center_y - params.span_y * 0.5;
        let default_ymax = params.center_y + params.span_y * 0.5;

        let xmin = cli.xmin.unwrap_or(default_xmin);
        let xmax = cli.xmax.unwrap_or(default_xmax);
        let ymin = cli.ymin.unwrap_or(default_ymin);
        let ymax = cli.ymax.unwrap_or(default_ymax);
        params.set_bounds(xmin, xmax, ymin, ymax);
        // Les bornes f64 remplacent TOUTE coordonnée HP antérieure (--toml,
        // --from-png) : les paths HP-aware liraient sinon les anciennes chaînes.
        params.center_x_hp = None;
        params.center_y_hp = None;
        params.span_x_hp = None;
        params.span_y_hp = None;
    }

    // Recentrage éventuel (prioritaire sur les bornes).
    if let Some(cx) = cli.center_x {
        params.center_x = cx;
        params.center_x_hp = None; // override f64 → la chaîne HP (--toml) serait prioritaire
    }
    if let Some(cy) = cli.center_y {
        params.center_y = cy;
        params.center_y_hp = None;
    }

    // Coordonnées haute précision (HP) pour deep zooms.
    if let Some(ref cx_hp) = cli.center_x_hp {
        params.center_x_hp = Some(cx_hp.clone());
        // Mettre à jour le f64 aussi (approximation pour les paths non-HP)
        if let Ok(v) = cx_hp.parse::<f64>() {
            params.center_x = v;
        }
    }
    if let Some(ref cy_hp) = cli.center_y_hp {
        params.center_y_hp = Some(cy_hp.clone());
        if let Ok(v) = cy_hp.parse::<f64>() {
            params.center_y = v;
        }
    }

    // Zoom -> span HP conversion.
    if let Some(ref zoom_str) = cli.zoom {
        let [cx, cy, _, _] = hp_strings(&params);
        let view =
            ViewHp::from_center_and_zoom(&cx, &cy, zoom_str, params.width, params.height, 1024)
                .ok_or_else(|| CliError::invalid(format!("Cannot parse the zoom: '{zoom_str}'")))?;
        view.write_to_params(&mut params);
        // Ne pas remplacer une saisie décimale exacte par l'expansion de son
        // approximation binaire MPFR : à deep zoom cela déplace la référence.
        params.center_x_hp = Some(cx);
        params.center_y_hp = Some(cy);
    }

    // Override des itérations si fourni (nombre explicite). `auto` et
    // l'absence sont résolus en fin de fonction, sur les paramètres FINALS
    // (type, formule hybride/opcodes) — cf. `iterations_choice`.
    if let Some(IterationsArg::Count(iters)) = cli.iterations {
        params.iteration_max = iters;
    }

    // Palette et répétitions de couleurs (défaut historique 6 si absente ;
    // avec --from-png, celle du PNG).
    if let Some(p) = cli.palette {
        params.color.color_mode = p;
    } else if !from_png {
        params.color.color_mode = 6;
    }
    // `color_repeat` : override explicite seulement — le défaut vient du type
    // (`default_params_for_type` : 1 pour les densités, 2 Lyapunov, 40 sinon),
    // comme dans la GUI. Forcer 40 rendait Buddhabrot/Nebulabrot en 40 cycles.
    if let Some(cr) = cli.color_repeat {
        params.color.color_repeat = cr.max(1);
    }

    // GMP haute précision.
    if !from_png || cli.gmp {
        params.engine.use_gmp = cli.gmp;
    }
    match cli.precision_bits {
        Some(bits) => params.engine.precision_bits = bits.max(64),
        None if !from_png => params.engine.precision_bits = 256,
        None => {}
    }

    // Mode d'algorithme.
    if let Some(mode) = &cli.algorithm {
        let parsed = AlgorithmMode::from_cli_name(mode).ok_or_else(|| {
            CliError::invalid(format!(
                "Invalid algorithm mode: '{mode}'. Options: auto, f64, perturbation, gmp"
            ))
        })?;
        params.engine.algorithm_mode = parsed;
    }

    if let Some(bla_threshold) = cli.bla_threshold {
        if bla_threshold > 0.0 {
            params.perturbation.bla_threshold = bla_threshold;
        }
    }
    if let Some(bla_validity_scale) = cli.bla_validity_scale {
        if bla_validity_scale > 0.0 {
            params.perturbation.bla_validity_scale = bla_validity_scale;
        }
    }
    if let Some(glitch_tolerance) = cli.glitch_tolerance {
        if glitch_tolerance > 0.0 {
            params.perturbation.glitch_tolerance = glitch_tolerance;
        }
    }
    if let Some(multibrot_power) = cli.multibrot_power {
        if multibrot_power > 0.0 {
            params.formula.multibrot_power = multibrot_power;
        }
    }
    // Graine Julia (c constant, le pixel est z₀) : seulement pour les types
    // dont la convention est Julia — ailleurs `seed` est le z₀ de l'orbite et
    // le déplacer changerait de fractale.
    if cli.julia_re.is_some() || cli.julia_im.is_some() {
        if fractall_cli::fractal::bytecode::Formula::is_julia_for(params.fractal_type) {
            if let Some(re) = cli.julia_re.filter(|v| v.is_finite()) {
                params.seed.re = re;
            }
            if let Some(im) = cli.julia_im.filter(|v| v.is_finite()) {
                params.seed.im = im;
            }
        } else {
            eprintln!(
                "--julia-re / --julia-im ignored: {:?} is not a Julia type",
                params.fractal_type
            );
        }
    }

    // Distance estimation et interior detection (avec --from-png, un
    // drapeau absent garde la valeur du PNG).
    if !from_png || cli.enable_distance_estimation {
        params.channels.enable_distance_estimation = cli.enable_distance_estimation;
    }
    if !from_png || cli.enable_interior_detection {
        params.channels.enable_interior_detection = cli.enable_interior_detection;
    }
    match cli.interior_threshold {
        Some(t) if t > 0.0 => params.channels.interior_threshold = t,
        None if !from_png => params.channels.interior_threshold = 0.001,
        _ => {}
    }

    // Tier double-double (~106 b) opt-in pour les spirales deep-zoom sensibles.
    if cli.dd_tier {
        params.engine.use_dd_tier = true;
    }

    // Moteur d'itération bytecode (Fraktaler-3 style)
    // Activé par défaut depuis Session E. --no-bytecode pour désactiver.
    if cli.no_bytecode {
        params.engine.use_bytecode_engine = false;
    }
    let _ = cli.bytecode; // legacy flag, no-op (default already true)

    if cli.find_nucleus {
        params.engine.find_nucleus = true;
    }

    // Rotation CLI : prioritaire sur la valeur TOML (cf. doc --rotation).
    if let Some(rot) = cli.rotation {
        params.rotation = rot;
    }

    // Escape radius CLI : override le défaut par type (alignement ER avec F3).
    if let Some(b) = cli.bailout {
        params.bailout = b;
    }

    // Anti-aliasing multi-sample (per-frame jitter). jitter_scale est
    // enregistré dans les métadonnées ; l'offset sous-pixel de chaque sample
    // est appliqué dans la boucle d'accumulation. Le nombre de samples n'est
    // PAS persisté dans le PNG : --from-png rend sans AA sauf --aa-samples.
    let aa_samples = cli.aa_samples.unwrap_or(1).max(1);
    let png_jitter = (from_png && params.sampling.jitter_scale > 0.0)
        .then_some(params.sampling.jitter_scale);
    let aa_jitter_scale = cli.jitter_scale.or(png_jitter).unwrap_or(1.0);
    if aa_samples > 1 {
        params.sampling.jitter_scale = aa_jitter_scale;
    }

    match params.engine.algorithm_mode {
        AlgorithmMode::ReferenceGmp => params.engine.use_gmp = true,
        AlgorithmMode::StandardF64 => params.engine.use_gmp = false,
        _ => {}
    }

    // Preset Lyapunov (si applicable).
    if params.fractal_type == FractalType::Lyapunov {
        if let Some(preset_name) = &cli.lyapunov_preset {
            let preset = LyapunovPreset::from_cli_name(preset_name).ok_or_else(|| {
                CliError::invalid(format!(
                    "Invalid Lyapunov preset: '{preset_name}'. Options: standard, zircon-city, \
                     jellyfish, asymmetric, spaceship, heavy-blocks"
                ))
            })?;
            apply_lyapunov_preset(&mut params, preset);
        }
    }

    // Mode de colorisation (outcoloring, défaut smooth ; avec --from-png,
    // celui du PNG).
    let outcoloring = cli
        .outcoloring
        .as_deref()
        .or((!from_png).then_some("smooth"));
    if let Some(outcoloring) = outcoloring {
        params.color.out_coloring_mode =
            OutColoringMode::from_cli_name(outcoloring).ok_or_else(|| {
                CliError::invalid(format!(
                    "Invalid colouring mode: '{outcoloring}'. Options: iter, iter+real, \
                     iter+imag, iter+real/imag, iter+all, binary, biomorphs, potential, \
                     color-decomp, smooth, orbit-traps, wings, distance, distance-ao, distance-3d"
                ))
            })?;
    }

    // Invariant inter-groupes `channels ⊇ required_channels` : règle UNIQUE
    // partagée avec la GUI (`render::ensure_required_channels`) — la
    // colorisation vérifiée (G5 `RenderOutput`) refuse un canal absent au lieu
    // de retomber silencieusement sur Smooth.
    render::ensure_required_channels(&mut params);

    // Transformation du plan (XaoS-style).
    let plane = cli.plane.as_deref().or((!from_png).then_some("0"));
    if let Some(plane) = plane {
        params.plane_transform = PlaneTransform::from_cli_name(plane).ok_or_else(|| {
            CliError::invalid(format!(
                "Invalid plane: '{plane}'. Options: 0-6, mu, 1/mu, 1/(mu+0.25), lambda, \
                 1/lambda, 1/lambda-1, 1/(mu-1.40115)"
            ))
        })?;
    }

    // Garde-fou : un f64 non fini (`--rotation nan`, `--center-x inf`…) se
    // sérialise en `null` dans les métadonnées PNG, que serde refuse au
    // rechargement (drag-and-drop GUI, --from-map) — bug 2026-08-23.
    if let Some(name) = non_finite_param(&params) {
        return Err(CliError::invalid(format!(
            "Error: parameter '{name}' is not finite (NaN/inf) — value rejected"
        )));
    }

    let iterations = iterations_choice(cli.iterations, source_has_iterations, &params);

    Ok(Resolved {
        params,
        aa_samples,
        aa_jitter_scale,
        iterations,
        caps_follow_iterations,
    })
}

/// Ordre de résolution du plafond d'itérations (pur, testé) :
/// `--iterations N` → N ; `--iterations auto` → sonde, même si une source
/// fournit une valeur ; absent + valeur de `--toml`/`--from-png` → cette
/// valeur ; absent sinon → sonde si le type a un plafond « escape-time »
/// (`wisdom::auto_iterations_applicable`), défaut du type autrement.
fn iterations_choice(
    arg: Option<IterationsArg>,
    source_has_iterations: bool,
    params: &FractalParams,
) -> IterChoice {
    match arg {
        Some(IterationsArg::Count(_)) => IterChoice::User,
        Some(IterationsArg::Auto) => IterChoice::Auto { explicit: true },
        None if source_has_iterations => IterChoice::Source,
        None if fractal::wisdom::auto_iterations_applicable(params) => {
            IterChoice::Auto { explicit: false }
        }
        None => IterChoice::Default,
    }
}

/// Lignes stderr de la sonde automatique (anglais, une ligne d'information,
/// plus un avertissement si la sonde n'a pas pu conclure franchement).
fn auto_iterations_messages(r: &fractal::wisdom::AutoIterResult) -> Vec<String> {
    let summary = r.summary_line();
    let detail = summary.strip_prefix("Auto iterations: ").unwrap_or(&summary);
    let mut line = format!("Iterations: auto \u{2192} {detail}");
    if line.chars().count() <= 90 {
        line.push_str(" \u{2014} --iterations N overrides");
    }
    let mut lines = vec![line];
    if r.no_escape {
        lines.push(format!(
            "Warning: no pixel escaped within {} iterations on the probe, so the view looks \
             entirely interior (black). If it should show detail, pass a higher --iterations N.",
            r.last_probe_cap
        ));
    }
    if r.ceiling_hit {
        lines.push(format!(
            "Warning: the automatic iteration count hit its ceiling ({}) and the view may need \
             more; pass --iterations N to set it explicitly.",
            r.iteration_max
        ));
    }
    lines
}

/// Exécute la sonde automatique si `choice` la demande : fixe
/// `params.iteration_max` (et les caps perturbation qui suivaient le TOML),
/// écrit la ligne d'information et laisse dans `orbit_cache` l'orbite de
/// référence du dernier tour, réutilisable par le rendu final (même vue,
/// précision alignée par la sonde). `NotApplicable` garde la valeur courante
/// (défaut du type ; note seulement si `--iterations auto` était écrit).
/// Renvoie la provenance EFFECTIVE (pour `[WISDOM] iter_src=`).
fn apply_auto_iterations(
    params: &mut FractalParams,
    choice: IterChoice,
    caps_follow_iterations: bool,
    cancel: &Arc<AtomicBool>,
    orbit_cache: &mut Option<Arc<fractal::perturbation::ReferenceOrbitCache>>,
) -> IterChoice {
    let IterChoice::Auto { explicit } = choice else {
        return choice;
    };
    use fractal::wisdom::{auto_iterations_with, AutoIterError, AutoIterOptions};
    match auto_iterations_with(params, cancel, AutoIterOptions::default(), orbit_cache) {
        Ok(r) => {
            params.iteration_max = r.iteration_max;
            if caps_follow_iterations {
                params.perturbation.max_perturb_iterations = r.iteration_max;
                params.perturbation.max_bla_steps = r.iteration_max;
            }
            for line in auto_iterations_messages(&r) {
                eprintln!("{line}");
            }
            choice
        }
        Err(AutoIterError::NotApplicable) => {
            if explicit {
                eprintln!(
                    "Note: --iterations auto does not apply to {:?} (not an escape-time type); \
                     using {} iterations.",
                    params.fractal_type, params.iteration_max
                );
            }
            IterChoice::Default
        }
        Err(AutoIterError::Cancelled) => {
            eprintln!("Error: automatic iteration probe interrupted");
            std::process::exit(1);
        }
    }
}

fn main() {
    let cli = Cli::parse();

    // Mode bench wisdom (G9.2) : mesure les débits par technique et persiste
    // le fichier wisdom machine, puis sort. Pas de rendu --output.
    if cli.wisdom_bench {
        run_wisdom_bench();
        return;
    }
    let Some(output_path) = cli.output.clone() else {
        eprintln!("--output is required (except with --wisdom-bench)");
        std::process::exit(2);
    };

    // Recolorisation depuis une map .fmap (G12 jalon 1) : aucun rendu, aucun
    // --type requis — tout vient de la map, sauf les overrides couleur.
    if let Some(ref map_path) = cli.from_map {
        run_from_map(&cli, map_path, &output_path);
        return;
    }

    let Resolved {
        mut params,
        aa_samples,
        aa_jitter_scale,
        iterations,
        caps_follow_iterations,
    } = match resolve_params(&cli) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{}", e.msg);
            std::process::exit(e.code);
        }
    };

    // Le temps de rendu rapporté inclut la sonde d'itérations automatique.
    let start_time = std::time::Instant::now();
    let cancel = Arc::new(AtomicBool::new(false));

    // Itérations automatiques (wisdom_iter) : sonde CPU par le dispatcher
    // unique, AVANT le choix du device et l'export F3 (qui voient ainsi la
    // valeur finale). L'orbite de référence du dernier tour reste dans
    // `orbit_cache` et sert au rendu final ; les samples AA re-rendent avec
    // ce même `params`.
    let mut orbit_cache = None;
    fractal::wisdom::set_iteration_source(iterations.wisdom_source());
    let iterations = apply_auto_iterations(
        &mut params,
        iterations,
        caps_follow_iterations,
        &cancel,
        &mut orbit_cache,
    );
    fractal::wisdom::set_iteration_source(iterations.wisdom_source());
    let params = params;

    if let Some(path) = cli.export_f3_toml.as_deref() {
        if let Err(error) = save_f3_toml(path, &params) {
            eprintln!("Error exporting F3 TOML {}: {error}", path.display());
            std::process::exit(1);
        }
        println!("F3 TOML written: {}", path.display());
    }

    // Calcul escape-time (CPU ou GPU).
    // Sortie TYPÉE du dispatcher (G5 `RenderOutput`) : les canaux annexes
    // (distances/orbites) voyagent avec iterations/zs jusqu'à la colorisation
    // vérifiée et --output-map — plus de tuple partiel qui les jette.
    let final_plan = if cli.no_gpu {
        render::RenderPlan::for_device(&params, fractal::wisdom::Device::Cpu)
    } else if cli.gpu {
        render::RenderPlan::for_device(&params, fractal::wisdom::Device::Gpu)
    } else {
        // Replanification sur les paramètres FINALS (canaux, transform et
        // formule ont pu changer depuis l'analyse initiale des options).
        render::RenderPlan::auto(&params, true)
    };
    let gpu_renderer = if final_plan.wisdom().device == fractal::wisdom::Device::Gpu {
        GpuRenderer::new()
    } else {
        None
    };
    if let Some(gpu) = gpu_renderer.as_ref() {
        println!("GPU initialised ({})", gpu.precision_label());
    }
    let planned = render::render_planned(
        render::PlannedRenderRequest::new(final_plan, render::RenderRequest::new(&params, &cancel)),
        gpu_renderer.as_ref(),
        &mut orbit_cache,
    );
    let (out, rendered_on_gpu): (render::RenderOutput, bool) = match planned {
        Some(result) => {
            if result.fell_back_to_cpu {
                eprintln!("GPU unavailable or incompatible, falling back to CPU");
            } else if result.device == fractal::wisdom::Device::Gpu {
                println!(
                    "Mode: GPU {}",
                    if result.used_perturbation {
                        "perturbation"
                    } else {
                        "standard"
                    }
                );
            }
            (result.output, result.device == fractal::wisdom::Device::Gpu)
        }
        None => {
            eprintln!("Error: render interrupted or type not supported by the dispatcher");
            std::process::exit(1);
        }
    };

    let render_time = start_time.elapsed();

    // Export EXR raw au format F3 si demandé (--export-iterations).
    if let Some(ref exr_path) = cli.export_iterations {
        // Doit matcher l'escape radius effectif du renderer pour que NF soit cohérent
        // — sinon les pixels qui viennent juste d'échapper (z² ≈ bailout²) tombent
        // sous le seuil F3 et NF s'effondre à 0.
        let bailout_sq = params.bailout * params.bailout;
        // Degré de la formule dérivé du bytecode (aligné F3 `opcodes_degree`,
        // `param.cc:970`). Pour Mandelbrot/BS/Tricorn/Celtic → 2, pour
        // Multibrot puissance entière → la puissance, etc. Fallback sur
        // `multibrot_power` quand la formule n'est pas compilable (types
        // non-escape-time : Newton, Phoenix, Magnet, Lyapunov, …). Cf. P0
        // parité F3 : sans ça, NF utilise degree=2.5 par défaut pour tous
        // les types, ce qui décale le smooth iter de ~0.1-0.2 pour les
        // pixels juste échappés.
        let degree = match fractall_cli::fractal::bytecode::compile_formula(
            params.fractal_type,
            params.formula.multibrot_power,
        ) {
            Some(formula) => fractall_cli::fractal::bytecode::formula_last_degree(&formula) as f64,
            None => params.formula.multibrot_power.max(2.0),
        };
        match save_iterations_exr(
            exr_path,
            params.width as usize,
            params.height as usize,
            &out.iterations,
            &out.zs,
            params.iteration_max,
            bailout_sq,
            degree,
        ) {
            Ok(()) => println!("Raw EXR written: {}", exr_path.display()),
            Err(e) => {
                eprintln!("Error writing EXR {}: {}", exr_path.display(), e);
                std::process::exit(1);
            }
        }
    }

    // Export PNG avec métadonnées.
    let save_start = std::time::Instant::now();
    // Utiliser les strings HP si disponibles, sinon convertir f64
    let [center_x_hp, center_y_hp, span_x_hp, span_y_hp] = hp_strings(&params);
    let png_result = if aa_samples > 1 {
        // Anti-aliasing multi-sample (per-frame jitter) : sample 0 = le rendu de
        // base déjà calculé (offset (0,0)), samples 1..N re-rendus avec un offset
        // sous-pixel low-discrepancy, puis moyenne en espace RGB.
        let n_bytes = params.width as usize * params.height as usize * 3;
        let mut accum = vec![0f64; n_bytes];
        let accumulate = |accum: &mut [f64], rgb: &[u8]| {
            for (a, &c) in accum.iter_mut().zip(rgb.iter()) {
                *a += c as f64;
            }
        };
        // Décorrélation Cranley-Patterson F3 par pixel (`jitter::pixel_offset`) :
        // chaque sample (y compris k=0) est jitteré par pixel → pas de motif AA
        // corrélé à bas N. Tous les samples sont re-rendus (k=0 n'est plus le
        // centre exact ; le rendu de base ci-dessus n'est réutilisé qu'à N=1).
        for k in 0..aa_samples as u64 {
            let mut p = params.clone();
            p.sampling.aa_jitter = Some((k, aa_jitter_scale));
            // Dispatcher COMPLET : les canaux distances/orbites sont requis par
            // les modes Distance*/OrbitTraps/Wings (sinon retombée silencieuse
            // sur Smooth — classe « colorisation unique », cf. CLAUDE.md).
            let mut aa_cache = None;
            let sample = if rendered_on_gpu {
                render::render_planned(
                    render::PlannedRenderRequest::new(
                        final_plan,
                        render::RenderRequest::new(&p, &cancel),
                    ),
                    gpu_renderer.as_ref(),
                    &mut aa_cache,
                )
                .map(|r| r.output)
            } else {
                render::render_request(render::RenderRequest::new(&p, &cancel), &mut aa_cache)
            };
            let Some(sample) = sample else {
                eprintln!("Error: AA render interrupted (sample {})", k + 1);
                std::process::exit(1);
            };
            let rgb = match io::png::colorize_output(&p, &sample) {
                Ok(rgb) => rgb,
                Err(e) => {
                    eprintln!("Colouring error (sample {}): {e}", k + 1);
                    std::process::exit(1);
                }
            };
            accumulate(&mut accum, &rgb);
            println!("[AA] sample {}/{}", k + 1, aa_samples);
        }
        let inv_n = 1.0 / aa_samples as f64;
        let avg: Vec<u8> = accum
            .iter()
            .map(|&s| (s * inv_n).round().clamp(0.0, 255.0) as u8)
            .collect();
        io::png::save_png_rgb_with_metadata_aa(
            &params,
            &avg,
            &output_path,
            &center_x_hp,
            &center_y_hp,
            &span_x_hp,
            &span_y_hp,
            aa_samples,
        )
    } else {
        // Colorisation VÉRIFIÉE (G5) : les canaux annexes voyagent avec le
        // rendu, et un canal requis absent est une ERREUR — pas une retombée
        // silencieuse sur Smooth (le PNG du CLI divergeait de l'affichage GUI).
        let buffer = match io::png::colorize_output(&params, &out) {
            Ok(rgb) => rgb,
            Err(e) => {
                eprintln!("Colouring error: {e}");
                std::process::exit(1);
            }
        };
        save_png_rgb_with_metadata(
            &params,
            &buffer,
            &output_path,
            &center_x_hp,
            &center_y_hp,
            &span_x_hp,
            &span_y_hp,
        )
    };
    if let Err(e) = png_result {
        eprintln!("Error writing the PNG: {e}");
        std::process::exit(1);
    }

    // Map .fmap (G12 jalon 1) : buffers bruts recolorisables sans recalcul.
    if let Some(ref map_path) = cli.output_map {
        if aa_samples > 1 {
            eprintln!(
                "[MAP] --output-map ignored with --aa-samples > 1 (AA output is an RGB average, not iteration channels)"
            );
        } else {
            let mut params_to_save = params.clone();
            params_to_save.center_x_hp = Some(center_x_hp.clone());
            params_to_save.center_y_hp = Some(center_y_hp.clone());
            params_to_save.span_x_hp = Some(span_x_hp.clone());
            params_to_save.span_y_hp = Some(span_y_hp.clone());
            let map = io::fmap::FractalMap {
                params: params_to_save,
                iterations: out.iterations.clone(),
                zs: out.zs.clone(),
                distances: (!out.distances.is_empty()).then(|| out.distances.clone()),
            };
            match io::fmap::save_fmap(&map, map_path) {
                Ok(()) => println!("Map written: {}", map_path.display()),
                Err(e) => {
                    eprintln!("Error writing the map: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
    let save_time = save_start.elapsed();
    let total_time = start_time.elapsed();

    // Affichage du temps de génération. ⚠️ Ligne lue par le harness
    // (`scripts/harnesslib/axes_perf.py::_RE_GEN`, verrou
    // `scripts/test_harness_guard.py`) : ne changer le format qu'avec eux.
    println!(
        "Done in {:.2}s (render: {:.2}s, save: {:.2}s)",
        total_time.as_secs_f64(),
        render_time.as_secs_f64(),
        save_time.as_secs_f64()
    );
}

/// Nom du premier paramètre f64 non fini, s'il y en a un.
fn non_finite_param(p: &fractal::FractalParams) -> Option<&'static str> {
    let checks: [(&str, f64); 11] = [
        ("center_x", p.center_x),
        ("center_y", p.center_y),
        ("span_x", p.span_x),
        ("span_y", p.span_y),
        ("rotation", p.rotation),
        ("bailout", p.bailout),
        ("color_offset", p.color.color_offset),
        ("multibrot_power", p.formula.multibrot_power),
        ("jitter_scale", p.sampling.jitter_scale),
        ("seed.re", p.seed.re),
        ("seed.im", p.seed.im),
    ];
    checks.iter().find(|(_, v)| !v.is_finite()).map(|(n, _)| *n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("fractall_cli_{}_{}", std::process::id(), name))
    }

    fn resolve(args: &[&str]) -> FractalParams {
        let mut argv = vec!["fractall-cli"];
        argv.extend_from_slice(args);
        resolve_params(&Cli::parse_from(argv)).unwrap_or_else(|e| panic!("{}", e.msg)).params
    }

    /// Rendu CPU + colorisation + PNG exactement comme `main` (hors AA).
    fn render_to_png(params: &FractalParams, path: &std::path::Path) -> Vec<u8> {
        let cancel = Arc::new(AtomicBool::new(false));
        let out = render::render_request(render::RenderRequest::new(params, &cancel), &mut None)
            .expect("render");
        let rgb = io::png::colorize_output(params, &out).expect("colorize");
        let [cx, cy, sx, sy] = hp_strings(params);
        save_png_rgb_with_metadata(params, &rgb, path, &cx, &cy, &sx, &sy).expect("save");
        rgb
    }

    /// `--from-png` à taille identique sur un PNG produit par le CLI :
    /// pixel-identique au rendu d'origine (vue, couleurs, moteur restaurés).
    #[test]
    fn from_png_same_size_rerender_is_pixel_identical() {
        let png = tmp("orig.png");
        let png_s = png.to_str().unwrap();
        let original = resolve(&[
            "--type", "3", "--width", "64", "--height", "48", "--no-gpu",
            "--center-x-hp", "-0.743643887037158704752191506114774",
            "--center-y-hp", "0.131825904205311970493132056385139",
            "--zoom", "1e7", "--iterations", "3000", "--palette", "11",
            "--color-repeat", "8", "--output", png_s,
        ]);
        let rgb_original = render_to_png(&original, &png);

        let again = resolve(&["--from-png", png_s, "--no-gpu", "--output", "unused.png"]);
        assert_eq!(again.iteration_max, 3000);
        assert_eq!(again.color.color_mode, 11);
        assert_eq!(again.color.color_repeat, 8);
        assert_eq!((again.width, again.height), (64, 48));
        let rgb_again = render_to_png(&again, &tmp("again.png"));
        assert!(rgb_original == rgb_again, "--from-png re-render differs from the original");

        // Overrides explicites : appliqués, le reste vient du PNG.
        let over = resolve(&[
            "--from-png", png_s, "--palette", "2", "--iterations", "500", "--width", "32",
            "--output", "unused.png",
        ]);
        assert_eq!(over.color.color_mode, 2);
        assert_eq!(over.iteration_max, 500);
        assert_eq!(over.color.color_repeat, 8);
        assert_eq!((over.width, over.height), (32, 24));
        assert_eq!(over.center_x_hp, again.center_x_hp);
        assert_eq!(over.span_y_hp, again.span_y_hp);

        for p in [png, tmp("again.png")] {
            let _ = std::fs::remove_file(p);
        }
    }

    /// Changement de ratio : centre et étendue verticale conservés.
    #[test]
    fn from_png_aspect_change_keeps_center_and_span_y() {
        let png = tmp("aspect.png");
        let png_s = png.to_str().unwrap();
        let original = resolve(&[
            "--type", "3", "--width", "40", "--height", "30", "--iterations", "200",
            "--output", png_s,
        ]);
        render_to_png(&original, &png);
        let wide = resolve(&[
            "--from-png", png_s, "--width", "80", "--height", "30", "--output", "unused.png",
        ]);
        assert_eq!(wide.center_x_hp, Some(original.center_x.to_string()));
        assert_eq!(wide.center_y_hp, Some(original.center_y.to_string()));
        assert_eq!(wide.span_y_hp, Some(original.span_y.to_string()));
        assert_eq!(wide.span_y, original.span_y);
        assert!((wide.span_x - original.span_y * 80.0 / 30.0).abs() < 1e-12);
        let _ = std::fs::remove_file(png);
    }

    #[test]
    fn from_png_without_metadata_is_an_explicit_error() {
        let plain = tmp("plain.png");
        image::RgbImage::new(4, 4).save(&plain).unwrap();
        let cli = Cli::parse_from([
            "fractall-cli", "--from-png", plain.to_str().unwrap(), "--output", "x.png",
        ]);
        let err = resolve_params(&cli).err().expect("must fail");
        assert!(err.msg.contains("not a fractall PNG"), "{}", err.msg);
        let _ = std::fs::remove_file(plain);

        let cli = Cli::parse_from([
            "fractall-cli", "--from-png", "/nonexistent/view.png", "--output", "x.png",
        ]);
        let err = resolve_params(&cli).err().expect("must fail");
        assert!(err.msg.contains("file not found"), "{}", err.msg);
    }

    fn resolve_full(args: &[&str]) -> Resolved {
        let mut argv = vec!["fractall-cli"];
        argv.extend_from_slice(args);
        resolve_params(&Cli::parse_from(argv)).unwrap_or_else(|e| panic!("{}", e.msg))
    }

    #[test]
    fn iterations_argument_parses_numbers_and_auto() {
        assert_eq!(parse_iterations_arg("5000"), Ok(IterationsArg::Count(5000)));
        assert_eq!(parse_iterations_arg("auto"), Ok(IterationsArg::Auto));
        assert_eq!(parse_iterations_arg("AUTO"), Ok(IterationsArg::Auto));
        assert!(parse_iterations_arg("0").is_err());
        assert!(parse_iterations_arg("lots").is_err());
        assert!(Cli::try_parse_from(["fractall-cli", "--iterations", "x"]).is_err());
    }

    /// Ordre de résolution : nombre explicite > `auto` explicite > valeur
    /// d'une source > sonde (types escape-time) > défaut du type.
    #[test]
    fn iterations_resolution_order() {
        // Nombre explicite : tel quel, pas de sonde.
        let r = resolve_full(&["--type", "3", "--iterations", "777", "--output", "x.png"]);
        assert_eq!(r.iterations, IterChoice::User);
        assert_eq!(r.params.iteration_max, 777);

        // Absent, type escape-time, aucune source : sonde.
        let r = resolve_full(&["--type", "3", "--output", "x.png"]);
        assert_eq!(r.iterations, IterChoice::Auto { explicit: false });

        // Absent, type sans plafond escape-time : défaut du type, inchangé.
        let r = resolve_full(&["--type", "6", "--output", "x.png"]);
        assert_eq!(r.iterations, IterChoice::Default);
        assert_eq!(
            r.params.iteration_max,
            default_params_for_type(FractalType::Newton, 1920, 1080).iteration_max
        );

        // `auto` explicite sur un type non applicable : la sonde dira
        // NotApplicable (note à l'écran), le défaut reste en place.
        let r = resolve_full(&["--type", "6", "--iterations", "auto", "--output", "x.png"]);
        assert_eq!(r.iterations, IterChoice::Auto { explicit: true });

        // TOML avec `iterations` : la valeur de la source, caps alignés.
        let r = resolve_full(&["--toml", "toml/e50.toml", "--output", "x.png"]);
        assert_eq!(r.iterations, IterChoice::Source);
        assert!(r.caps_follow_iterations);
        assert_eq!(r.params.perturbation.max_perturb_iterations, r.params.iteration_max);

        // `auto` explicite l'emporte sur la valeur du TOML.
        let r = resolve_full(&["--toml", "toml/e50.toml", "--iterations", "Auto", "--output", "x.png"]);
        assert_eq!(r.iterations, IterChoice::Auto { explicit: true });
        assert!(r.caps_follow_iterations);

        // TOML + nombre explicite : le nombre.
        let r = resolve_full(&["--toml", "toml/e50.toml", "--iterations", "1234", "--output", "x.png"]);
        assert_eq!(r.iterations, IterChoice::User);
        assert_eq!(r.params.iteration_max, 1234);
    }

    #[test]
    fn iterations_from_png_source_and_explicit_auto() {
        let png = tmp("iter_src.png");
        let png_s = png.to_str().unwrap();
        let original = resolve(&[
            "--type", "3", "--width", "24", "--height", "16", "--iterations", "321",
            "--output", png_s,
        ]);
        render_to_png(&original, &png);
        let r = resolve_full(&["--from-png", png_s, "--output", "x.png"]);
        assert_eq!(r.iterations, IterChoice::Source);
        assert_eq!(r.params.iteration_max, 321);
        let r = resolve_full(&["--from-png", png_s, "--iterations", "auto", "--output", "x.png"]);
        assert_eq!(r.iterations, IterChoice::Auto { explicit: true });
        let _ = std::fs::remove_file(png);
    }

    /// Bout en bout : `--iterations` absent sur une vue Mandelbrot moyennement
    /// profonde (path perturbation) → la sonde fixe le plafond, le rendu
    /// réutilise son orbite ; le PNG porte la valeur NUMÉRIQUE finale et
    /// `--from-png` le reproduit au pixel près SANS nouvelle sonde.
    #[test]
    fn auto_iterations_end_to_end_png_metadata_and_rerender() {
        let png = tmp("auto_e2e.png");
        let png_s = png.to_str().unwrap();
        let Resolved {
            mut params,
            iterations,
            caps_follow_iterations,
            ..
        } = resolve_full(&[
            "--type", "3", "--width", "96", "--height", "64", "--no-gpu",
            "--center-x-hp", "-0.743643887037158704752191506114774",
            "--center-y-hp", "0.131825904205311970493132056385139",
            "--zoom", "1e14", "--output", png_s,
        ]);
        assert_eq!(iterations, IterChoice::Auto { explicit: false });
        let default_iters = params.iteration_max;
        let cancel = Arc::new(AtomicBool::new(false));
        let mut cache = None;
        let effective =
            apply_auto_iterations(&mut params, iterations, caps_follow_iterations, &cancel, &mut cache);
        assert_eq!(effective, iterations);
        assert!(cache.is_some(), "the probe leaves its reference orbit for the final render");
        let auto_iters = params.iteration_max;
        assert_ne!(auto_iters, default_iters, "the probe must have chosen a value");

        // Rendu final comme `main` : dispatcher unique AVEC le cache de la sonde.
        let out = render::render_request(render::RenderRequest::new(&params, &cancel), &mut cache)
            .expect("render");
        let rgb = io::png::colorize_output(&params, &out).expect("colorize");
        let [cx, cy, sx, sy] = hp_strings(&params);
        save_png_rgb_with_metadata(&params, &rgb, &png, &cx, &cy, &sx, &sy).expect("save");

        let saved = io::png::load_png_metadata(&png).expect("fractall PNG metadata");
        assert_eq!(saved.iteration_max, auto_iters);

        let again = resolve_full(&["--from-png", png_s, "--no-gpu", "--output", "x.png"]);
        assert_eq!(again.iterations, IterChoice::Source);
        assert_eq!(again.params.iteration_max, auto_iters);
        let rgb_again = render_to_png(&again.params, &tmp("auto_e2e_again.png"));
        assert!(rgb == rgb_again, "--from-png re-render of an auto-iteration PNG differs");

        for p in [png, tmp("auto_e2e_again.png")] {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn auto_iterations_messages_are_one_line_plus_warnings() {
        let mut r = fractal::wisdom::AutoIterResult {
            iteration_max: 42_000,
            rounds: 2,
            last_probe_cap: 160_000,
            elapsed: std::time::Duration::from_millis(410),
            probe_size: (85, 48),
            max_escaped: 34_565,
            q99_escaped: Some(30_000),
            escaped_fraction: 0.6,
            predicted_change: 0.001,
            ceiling_hit: false,
            no_escape: false,
        };
        let lines = auto_iterations_messages(&r);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].starts_with("Iterations: auto \u{2192} 42000 ("), "{}", lines[0]);
        assert!(!lines[0].starts_with('['), "must not look like a harness tag");
        r.no_escape = true;
        let lines = auto_iterations_messages(&r);
        assert_eq!(lines.len(), 2);
        assert!(lines[1].starts_with("Warning:") && lines[1].contains("--iterations N"));
    }

    #[test]
    fn type_is_still_required_without_from_png() {
        let cli = Cli::parse_from(["fractall-cli", "--output", "x.png"]);
        assert_eq!(resolve_params(&cli).err().unwrap().code, 2);
    }
}
