use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use clap::Parser;
use rug::Float;

mod fractal;
mod color;
mod render;
mod io;
mod gpu;

use fractal::{AlgorithmMode, apply_lyapunov_preset, default_params_for_type, FractalType, LyapunovPreset, OutColoringMode, PlaneTransform};
use render::render_escape_time;
use io::png::{colorize_to_rgb, save_png_rgb_with_metadata, save_png_with_metadata};
use fractal::jitter::sample_offset;
use io::exr::save_iterations_exr;
use gpu::GpuRenderer;

/// Utilitaire CLI pour générer des fractales basées sur fractall.
///
/// Exemple d'utilisation :
///   fractall-cli --type 3 --width 1920 --height 1080 --output mandelbrot.png
#[derive(Parser, Debug)]
#[command(
    name = "fractall-cli",
    about = "Générateur de fractales (Mandelbrot, Julia, etc.) en ligne de commande",
    version,
    author = "Arnaud Verhille et contributeurs"
)]
struct Cli {
    /// Type de fractale (3=Mandelbrot, 4=Julia, 5=JuliaSin, ..., 23=Multibrot).
    /// Optionnel quand --toml est fourni (défaut : 3 = Mandelbrot, format
    /// rust-fractal-core / corpus toml/).
    #[arg(long = "type")]
    fractal_type: Option<u8>,

    /// Largeur de l'image de sortie en pixels
    #[arg(long, default_value_t = 1920)]
    width: u32,

    /// Hauteur de l'image de sortie en pixels
    #[arg(long, default_value_t = 1080)]
    height: u32,

    /// Centre X du plan complexe (optionnel, sinon valeurs par défaut du type)
    #[arg(long)]
    center_x: Option<f64>,

    /// Centre Y du plan complexe (optionnel, sinon valeurs par défaut du type)
    #[arg(long)]
    center_y: Option<f64>,

    /// Centre X haute précision (string, pour deep zooms > 10^15)
    #[arg(long)]
    center_x_hp: Option<String>,

    /// Centre Y haute précision (string, pour deep zooms > 10^15)
    #[arg(long)]
    center_y_hp: Option<String>,

    /// Zoom (magnification, span = 4/zoom). Supporte notation scientifique (ex: 1.41e219)
    #[arg(long)]
    zoom: Option<String>,

    /// Coordonnée minimale X du plan complexe (prioritaire sur center_x/zoom)
    #[arg(long)]
    xmin: Option<f64>,

    /// Coordonnée maximale X du plan complexe
    #[arg(long)]
    xmax: Option<f64>,

    /// Coordonnée minimale Y du plan complexe
    #[arg(long)]
    ymin: Option<f64>,

    /// Coordonnée maximale Y du plan complexe
    #[arg(long)]
    ymax: Option<f64>,

    /// Nombre maximal d'itérations (sinon valeur par défaut du type)
    #[arg(long)]
    iterations: Option<u32>,

    /// Palette de couleurs (0 à 8, voir documentation)
    #[arg(long, default_value_t = 6)]
    palette: u8,

    /// Répétitions du gradient de couleur (2-40, pairs recommandés)
    #[arg(long, default_value_t = 40)]
    color_repeat: u32,

    /// Active le calcul haute précision avec GMP (via rug)
    #[arg(long)]
    gmp: bool,

    /// Précision GMP en bits (ex. 128, 256, 512)
    #[arg(long, default_value_t = 256)]
    precision_bits: u32,

    /// Mode d'algorithme (auto, f64, perturbation, gmp)
    #[arg(long)]
    algorithm: Option<String>,

    /// Seuil delta pour activer BLA (ex: 1e-8)
    #[arg(long)]
    bla_threshold: Option<f64>,

    /// Multiplicateur du rayon de validité BLA (1.0 = conservateur, >1 = agressif, ex: 2.0)
    #[arg(long)]
    bla_validity_scale: Option<f64>,

    /// Tolérance de glitch (ex: 1e-4)
    #[arg(long)]
    glitch_tolerance: Option<f64>,

    /// Puissance pour Multibrot (z^d + c), défaut 2.5
    #[arg(long)]
    multibrot_power: Option<f64>,

    /// Preset Lyapunov (standard, zircon-city, jellyfish, asymmetric, spaceship, heavy-blocks)
    #[arg(long)]
    lyapunov_preset: Option<String>,

    /// Mode de colorisation (iter, iter+real, iter+imag, iter+real/imag, iter+all, binary, biomorphs, potential, color-decomp, smooth)
    #[arg(long, default_value = "smooth")]
    outcoloring: String,

    /// Transformation du plan complexe (0=mu, 1=1/mu, 2=1/(mu+0.25), 3=lambda, 4=1/lambda, 5=1/lambda-1, 6=1/(mu-1.40115))
    /// Valeurs acceptées: 0-6 ou noms (mu, 1/mu, 1/(mu+0.25), lambda, 1/lambda, 1/lambda-1, 1/(mu-1.40115))
    #[arg(long, default_value = "0")]
    plane: String,

    /// Active l'estimation de distance (dual numbers, overhead supplementaire)
    #[arg(long)]
    enable_distance_estimation: bool,

    /// Active la detection de l'interieur (extended dual numbers)
    #[arg(long)]
    enable_interior_detection: bool,

    /// Seuil de detection interieur (defaut 0.001)
    #[arg(long, default_value_t = 0.001)]
    interior_threshold: f64,

    /// Utiliser le GPU pour le rendu (wgpu: Metal/Vulkan/DX12)
    #[arg(long)]
    gpu: bool,

    /// [DEPRECATED] Le moteur bytecode est activé par défaut depuis P3.1
    /// Session E. Ce flag ne sert qu'à rétro-compatibilité et passer
    /// --bytecode est désormais un no-op (== default). Utiliser --no-bytecode
    /// pour désactiver explicitement.
    #[arg(long, hide = true)]
    bytecode: bool,

    /// Désactive le moteur bytecode unifié et tombe sur le path legacy
    /// (glitch detection Pauldelbrot + clustering + secondary refs).
    /// À n'utiliser que pour debug ou comparaison.
    #[arg(long)]
    no_bytecode: bool,

    /// Active le nucleus finder (Mandelbrot only) : raffine le centre vers
    /// un minibrot exact via Newton avant de build l'orbite référence. Utile
    /// pour les deep zoom escape-time où la ref orbit s'évade prématurément
    /// (cf. e1000, e1121, e8000...). Coût additionnel : un scan de période
    /// + 1-64 itérations Newton à `precision_bits`, soit quelques secondes
    /// à zoom modéré, plus à très deep zoom. Inspiré de Fraktaler-3.1
    /// `hybrid_center`.
    #[arg(long)]
    find_nucleus: bool,

    /// Active le tier haute précision **double-double** (~106 bits) pour le path
    /// perturbation deep-zoom Mandelbrot. Équivalent pur-Rust du `float128` de
    /// Fraktaler-3 : itère référence + delta en ~106 bits (sans BLA) pour les
    /// spirales ultra-sensibles où la mantisse f64 (53 b) sature (écart
    /// d'itération vs GMP, cf. e30/e50). Plus lent (~10×) — n'activer que si
    /// nécessaire. Mandelbrot escape-time uniquement.
    #[arg(long)]
    dd_tier: bool,

    /// Rotation du plan en degrés (CCW). Appliquée au mapping pixel→c
    /// (équivalent F3 `transform.rotate`). Override la valeur du TOML si fournie.
    #[arg(long)]
    rotation: Option<f64>,

    /// Anti-aliasing : nombre d'échantillons sous-pixel jitterés moyennés
    /// (1 = désactivé, défaut). Chaque sample décale la grille d'un offset
    /// low-discrepancy (Halton, port F3 `radical_inverse` + tente `triangle`)
    /// puis les rendus colorés sont moyennés. Améliore les bords fins
    /// (surtout modes Distance/DE). CPU uniquement (ignoré avec --gpu).
    #[arg(long, default_value_t = 1)]
    aa_samples: u32,

    /// Amplitude du jitter sous-pixel pour l'AA, en pixels. 1.0 = tente pleine
    /// largeur (défaut quand --aa-samples > 1), 0.5 = demi-pixel. Sans effet si
    /// --aa-samples = 1. Pilote `FractalParams::jitter_scale`.
    #[arg(long)]
    jitter_scale: Option<f64>,

    /// Escape radius (bailout) du test d'évasion |z| ≥ bailout. Par défaut, la
    /// valeur par type (25 pour la famille escape-time, cf. ESCAPE_TIME_BAILOUT).
    /// Sert notamment à aligner l'ER avec Fraktaler-3 dans le harness de parité.
    #[arg(long)]
    bailout: Option<f64>,

    /// Fichier de sortie PNG (requis, sauf avec --wisdom-bench)
    #[arg(long, value_name = "FICHIER")]
    output: Option<PathBuf>,

    /// Benche les techniques de rendu (CPU std/perturbation f64-exp-dd + GPU
    /// f32 si disponible) et persiste les débits mesurés (iters/s) dans le
    /// fichier wisdom par machine (~/.config/fractall/wisdom.toml, override
    /// FRACTALL_WISDOM_FILE). Consommé par le plan wisdom (ligne [WISDOM]
    /// bench=…, arbitrage device à venir). Modèle Fraktaler-3 wisdom.cc.
    #[arg(long, default_value_t = false)]
    wisdom_bench: bool,

    /// Charge un fichier TOML de paramètres (format rust-fractal-core léger:
    /// real/imag/zoom/iterations[/rotate]). Les overrides CLI restent prioritaires.
    /// Utilisé pour le harness de parité Fraktaler-3 (corpus toml/).
    #[arg(long, value_name = "FICHIER")]
    toml: Option<PathBuf>,

    /// Exporte les itérations brutes en EXR au format Fraktaler-3
    /// (channels N0=uint32 iter, NF=float smooth fraction). Permet la
    /// comparaison apples-to-apples via scripts/compare_f3.py. Le PNG --output
    /// reste produit en plus.
    #[arg(long, value_name = "FICHIER.exr")]
    export_iterations: Option<PathBuf>,
}

/// Champs extraits d'un TOML de paramètres (format léger rust-fractal-core,
/// compatible avec le corpus `toml/`).
struct TomlParams {
    real: String,
    imag: String,
    zoom: String,
    iterations: Option<u32>,
    rotate: Option<f64>,
}

fn load_toml_params(path: &std::path::Path) -> TomlParams {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Erreur lecture TOML {}: {}", path.display(), e);
        std::process::exit(1);
    });
    let table: toml::Table = content.parse().unwrap_or_else(|e| {
        eprintln!("Erreur parsing TOML {}: {}", path.display(), e);
        std::process::exit(1);
    });

    let take_str = |key: &str| -> Option<String> {
        let v = table.get(key)?;
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

    let real = take_str("real").unwrap_or_else(|| {
        eprintln!("TOML {}: champ 'real' manquant", path.display());
        std::process::exit(1);
    });
    let imag = take_str("imag").unwrap_or_else(|| {
        eprintln!("TOML {}: champ 'imag' manquant", path.display());
        std::process::exit(1);
    });
    let zoom = take_str("zoom").unwrap_or_else(|| {
        eprintln!("TOML {}: champ 'zoom' manquant", path.display());
        std::process::exit(1);
    });
    let iterations = table.get("iterations").and_then(|v| v.as_integer()).map(|i| {
        if i > u32::MAX as i64 {
            eprintln!(
                "TOML {}: iterations={} > u32::MAX, clamp à {}. TODO: passer iteration_max en u64.",
                path.display(),
                i,
                u32::MAX
            );
            u32::MAX
        } else if i < 0 {
            eprintln!("TOML {}: iterations négatif ({}), forcé à 1024", path.display(), i);
            1024
        } else {
            i as u32
        }
    });
    let rotate = table
        .get("rotate")
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)));

    TomlParams { real, imag, zoom, iterations, rotate }
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
            g.render_dispatch(params, &cancel, None, None).map(|r| r.iterations)
        }
    });
    let file = match &gpu_render {
        Some(f) => fractal::wisdom_bench::run_bench(
            Some(f as &dyn Fn(&fractal::FractalParams) -> Option<Vec<u32>>),
            TARGET_SECONDS,
        ),
        None => {
            eprintln!("[WISDOM-BENCH] GPU indisponible — bench CPU uniquement");
            fractal::wisdom_bench::run_bench(None, TARGET_SECONDS)
        }
    };
    match fractal::wisdom_bench::save(&file) {
        Ok(path) => println!("Wisdom machine écrit : {}", path.display()),
        Err(e) => {
            eprintln!("Échec d'écriture du wisdom : {e}");
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
        eprintln!("--output est requis (sauf avec --wisdom-bench)");
        std::process::exit(2);
    };

    // Si --toml est fourni sans --type, défaut Mandelbrot (le corpus toml/ ne
    // contient que des Mandelbrot deep zoom au format rust-fractal-core).
    let fractal_type_id = match (cli.fractal_type, &cli.toml) {
        (Some(id), _) => id,
        (None, Some(_)) => 3,
        (None, None) => {
            eprintln!("--type est requis (ou utilisez --toml <FICHIER> pour le format rust-fractal-core)");
            std::process::exit(2);
        }
    };
    let fractal_type = match FractalType::from_id(fractal_type_id) {
        Some(t) => t,
        None => {
            eprintln!(
                "Type de fractale invalide: {} (attendu entre 3 et 23, sauf types spéciaux)",
                fractal_type_id
            );
            std::process::exit(1);
        }
    };

    // Paramètres par défaut pour ce type.
    let mut params = default_params_for_type(fractal_type, cli.width, cli.height);

    // Applique d'abord les paramètres TOML (les overrides CLI explicites
    // restent prioritaires car traités après).
    if let Some(ref toml_path) = cli.toml {
        let t = load_toml_params(toml_path);
        let prec = 1024u32;

        // Centre HP (string GMP).
        params.center_x_hp = Some(t.real.clone());
        params.center_y_hp = Some(t.imag.clone());
        if let Ok(p) = Float::parse(&t.real) {
            params.center_x = Float::with_val(prec, p).to_f64();
        }
        if let Ok(p) = Float::parse(&t.imag) {
            params.center_y = Float::with_val(prec, p).to_f64();
        }

        // Zoom -> span (utilise la résolution effective courante, qui peut
        // encore changer si --width/--height sont fournis ; on recalcule plus
        // bas si besoin).
        let zoom_gmp = Float::parse(&t.zoom)
            .map(|p| Float::with_val(prec, p))
            .unwrap_or_else(|_| {
                eprintln!("TOML {}: zoom illisible: '{}'", toml_path.display(), t.zoom);
                std::process::exit(1);
            });
        let four = Float::with_val(prec, 4.0);
        let span_x_gmp = four / &zoom_gmp;
        let aspect = params.height as f64 / params.width as f64;
        let span_y_gmp = Float::with_val(prec, &span_x_gmp * aspect);
        params.span_x_hp = Some(span_x_gmp.to_string());
        params.span_y_hp = Some(span_y_gmp.to_string());
        params.span_x = span_x_gmp.to_f64();
        params.span_y = span_y_gmp.to_f64();

        if let Some(iters) = t.iterations {
            params.iteration_max = iters;
            // F3 batch sets maximum_perturb_iterations / maximum_bla_steps to the user
            // `iterations` field (see fraktaler-3-3.1/src/param.h:38 defaults + the F3
            // wrapper we write in scripts/compare_f3.py). Fractall's defaults (1024) cap
            // pixel iteration way below iter_max for deep zooms, collapsing every pixel
            // onto a single iter count (e.g. e1000 stopped at 1028 while F3 reached
            // 16616+). Mirror F3's semantics here so a TOML in toml/ produces the same
            // effective caps on both engines.
            params.max_perturb_iterations = iters;
            params.max_bla_steps = iters;
        }

        if let Some(rot) = t.rotate {
            params.rotation = rot;
        }
    }

    // Override des coordonnées si demandé (bornes xmin/xmax/ymin/ymax).
    // Les bornes CLI sont converties en centre + span.
    // Utiliser center+span comme source de vérité au lieu de xmin/xmax/ymin/ymax
    let has_bounds = cli.xmin.is_some() || cli.xmax.is_some() || cli.ymin.is_some() || cli.ymax.is_some();
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
    }

    // Recentrage éventuel (prioritaire sur les bornes).
    if cli.center_x.is_some() || cli.center_y.is_some() {
        if let Some(cx) = cli.center_x {
            params.center_x = cx;
        }
        if let Some(cy) = cli.center_y {
            params.center_y = cy;
        }
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
        // Parse zoom with GMP arbitrary precision
        let prec = 1024u32;
        let zoom_gmp = Float::parse(zoom_str)
            .map(|parsed| Float::with_val(prec, parsed))
            .unwrap_or_else(|_| {
                eprintln!("Impossible de parser le zoom: '{}'", zoom_str);
                std::process::exit(1);
            });
        let four = Float::with_val(prec, 4.0);
        let span_x_gmp = four / &zoom_gmp;
        let aspect = params.height as f64 / params.width as f64;
        let span_y_gmp = Float::with_val(prec, &span_x_gmp * aspect);

        params.span_x_hp = Some(span_x_gmp.to_string());
        params.span_y_hp = Some(span_y_gmp.to_string());
        // Approximation f64 (sera 0.0 pour deep zooms, mais HP est utilisé)
        params.span_x = span_x_gmp.to_f64();
        params.span_y = span_y_gmp.to_f64();
    }

    // Override des itérations si fourni.
    if let Some(iters) = cli.iterations {
        if iters > 0 {
            params.iteration_max = iters;
        }
    }

    // Palette et répétitions de couleurs.
    params.color_mode = cli.palette;
    params.color_repeat = cli.color_repeat.max(1);

    // GMP haute précision.
    params.use_gmp = cli.gmp;
    params.precision_bits = cli.precision_bits.max(64);

    // Mode d'algorithme.
    if let Some(mode) = &cli.algorithm {
        match AlgorithmMode::from_cli_name(mode) {
            Some(parsed) => {
                params.algorithm_mode = parsed;
            }
            None => {
                eprintln!(
                    "Mode d'algorithme invalide: '{}'. Options: auto, f64, perturbation, gmp",
                    mode
                );
                std::process::exit(1);
            }
        }
    }

    if let Some(bla_threshold) = cli.bla_threshold {
        if bla_threshold > 0.0 {
            params.bla_threshold = bla_threshold;
        }
    }
    if let Some(bla_validity_scale) = cli.bla_validity_scale {
        if bla_validity_scale > 0.0 {
            params.bla_validity_scale = bla_validity_scale;
        }
    }
    if let Some(glitch_tolerance) = cli.glitch_tolerance {
        if glitch_tolerance > 0.0 {
            params.glitch_tolerance = glitch_tolerance;
        }
    }
    if let Some(multibrot_power) = cli.multibrot_power {
        if multibrot_power > 0.0 {
            params.multibrot_power = multibrot_power;
        }
    }

    // Distance estimation et interior detection
    params.enable_distance_estimation = cli.enable_distance_estimation;
    params.enable_interior_detection = cli.enable_interior_detection;
    if cli.interior_threshold > 0.0 {
        params.interior_threshold = cli.interior_threshold;
    }

    // Tier double-double (~106 b) opt-in pour les spirales deep-zoom sensibles.
    if cli.dd_tier {
        params.use_dd_tier = true;
    }

    // Moteur d'itération bytecode (Fraktaler-3 style)
    // Activé par défaut depuis Session E. --no-bytecode pour désactiver.
    if cli.no_bytecode {
        params.use_bytecode_engine = false;
    }
    let _ = cli.bytecode; // legacy flag, no-op (default already true)

    if cli.find_nucleus {
        params.find_nucleus = true;
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
    // est appliqué dans la boucle d'accumulation plus bas. AA CPU uniquement.
    let aa_samples = cli.aa_samples.max(1);
    let aa_jitter_scale = cli.jitter_scale.unwrap_or(1.0);
    if aa_samples > 1 {
        params.jitter_scale = aa_jitter_scale;
        if cli.gpu {
            eprintln!(
                "[AA] --aa-samples {aa_samples} ignoré en mode --gpu (anti-aliasing CPU uniquement)"
            );
        }
    }

    match params.algorithm_mode {
        AlgorithmMode::ReferenceGmp => params.use_gmp = true,
        AlgorithmMode::StandardF64 => params.use_gmp = false,
        _ => {}
    }

    // Preset Lyapunov (si applicable).
    if fractal_type == FractalType::Lyapunov {
        if let Some(preset_name) = &cli.lyapunov_preset {
            match LyapunovPreset::from_cli_name(preset_name) {
                Some(preset) => {
                    apply_lyapunov_preset(&mut params, preset);
                }
                None => {
                    eprintln!(
                        "Preset Lyapunov invalide: '{}'. Options: standard, zircon-city, jellyfish, asymmetric, spaceship, heavy-blocks",
                        preset_name
                    );
                    std::process::exit(1);
                }
            }
        }
    }

    // Mode de colorisation (outcoloring).
    match OutColoringMode::from_cli_name(&cli.outcoloring) {
        Some(mode) => {
            params.out_coloring_mode = mode;
        }
        None => {
            eprintln!(
                "Mode de colorisation invalide: '{}'. Options: iter, iter+real, iter+imag, iter+real/imag, iter+all, binary, biomorphs, potential, color-decomp, smooth",
                cli.outcoloring
            );
            std::process::exit(1);
        }
    }

    // Transformation du plan (XaoS-style).
    match PlaneTransform::from_cli_name(&cli.plane) {
        Some(plane) => {
            params.plane_transform = plane;
        }
        None => {
            eprintln!(
                "Plane invalide: '{}'. Options: 0-6, mu, 1/mu, 1/(mu+0.25), lambda, 1/lambda, 1/lambda-1, 1/(mu-1.40115)",
                cli.plane
            );
            std::process::exit(1);
        }
    }

    // Calcul escape-time (CPU ou GPU).
    let start_time = std::time::Instant::now();
    let cancel = Arc::new(AtomicBool::new(false));

    let (iterations, zs) = if cli.gpu {
        match GpuRenderer::new() {
            Some(gpu) => {
                println!("GPU initialisé ({})", gpu.precision_label());
                // Dispatch GPU partagé avec la GUI (cf. `GpuRenderer::render_dispatch`).
                match gpu.render_dispatch(&params, &cancel, None, None) {
                    Some(r) => {
                        println!(
                            "Mode: GPU {}",
                            if r.used_perturbation { "perturbation" } else { "standard" }
                        );
                        (r.iterations, r.zs)
                    }
                    None => {
                        println!("Type {:?} non rendu par le GPU → fallback CPU...", params.fractal_type);
                        render_escape_time(&params)
                    }
                }
            }
            None => {
                eprintln!("GPU non disponible, fallback CPU");
                render_escape_time(&params)
            }
        }
    } else {
        render_escape_time(&params)
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
        let degree = match crate::fractal::bytecode::compile_formula(
            params.fractal_type,
            params.multibrot_power,
        ) {
            Some(formula) => crate::fractal::bytecode::formula_last_degree(&formula) as f64,
            None => params.multibrot_power.max(2.0),
        };
        match save_iterations_exr(
            exr_path,
            params.width as usize,
            params.height as usize,
            &iterations,
            &zs,
            params.iteration_max,
            bailout_sq,
            degree,
        ) {
            Ok(()) => println!("EXR raw écrit: {}", exr_path.display()),
            Err(e) => {
                eprintln!("Erreur écriture EXR {}: {}", exr_path.display(), e);
                std::process::exit(1);
            }
        }
    }

    // Export PNG avec métadonnées.
    let save_start = std::time::Instant::now();
    // Utiliser les strings HP si disponibles, sinon convertir f64
    let center_x_hp = params.center_x_hp.clone().unwrap_or_else(|| params.center_x.to_string());
    let center_y_hp = params.center_y_hp.clone().unwrap_or_else(|| params.center_y.to_string());
    let span_x_hp = params.span_x_hp.clone().unwrap_or_else(|| params.span_x.to_string());
    let span_y_hp = params.span_y_hp.clone().unwrap_or_else(|| params.span_y.to_string());
    let png_result = if aa_samples > 1 && !cli.gpu {
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
        accumulate(&mut accum, &colorize_to_rgb(&params, &iterations, &zs));
        for k in 1..aa_samples as u64 {
            let (ox, oy) = sample_offset(k);
            let mut p = params.clone();
            p.aa_subpixel_offset = [ox * aa_jitter_scale, oy * aa_jitter_scale];
            let (it, zz) = render_escape_time(&p);
            accumulate(&mut accum, &colorize_to_rgb(&p, &it, &zz));
            println!("[AA] sample {}/{}", k + 1, aa_samples);
        }
        let inv_n = 1.0 / aa_samples as f64;
        let avg: Vec<u8> = accum
            .iter()
            .map(|&s| (s * inv_n).round().clamp(0.0, 255.0) as u8)
            .collect();
        save_png_rgb_with_metadata(
            &params,
            &avg,
            &output_path,
            &center_x_hp,
            &center_y_hp,
            &span_x_hp,
            &span_y_hp,
        )
    } else {
        save_png_with_metadata(
            &params,
            &iterations,
            &zs,
            &output_path,
            &center_x_hp,
            &center_y_hp,
            &span_x_hp,
            &span_y_hp,
        )
    };
    if let Err(e) = png_result {
        eprintln!("Erreur lors de l'écriture du PNG: {e}");
        std::process::exit(1);
    }
    let save_time = save_start.elapsed();
    let total_time = start_time.elapsed();

    // Affichage du temps de génération.
    println!(
        "Génération terminée en {:.2}s (rendu: {:.2}s, sauvegarde: {:.2}s)",
        total_time.as_secs_f64(),
        render_time.as_secs_f64(),
        save_time.as_secs_f64()
    );
}

