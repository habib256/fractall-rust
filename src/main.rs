use std::path::PathBuf;

use clap::Parser;

mod fractal;
mod color;
mod render;
mod io;

use fractal::{AlgorithmMode, apply_lyapunov_preset, default_params_for_type, FractalType, LyapunovPreset, OutColoringMode, PlaneTransform};
use render::render_escape_time;
use io::png::save_png;

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
    /// Type de fractale (3=Mandelbrot, 4=Julia, 5=JuliaSin, ..., 23=Multibrot)
    #[arg(long = "type")]
    fractal_type: u8,

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

    /// Fichier de sortie PNG
    #[arg(long, value_name = "FICHIER")]
    output: PathBuf,
}

fn main() {
    let cli = Cli::parse();

    // Conversion du type numérique vers l'enum interne.
    let fractal_type = match FractalType::from_id(cli.fractal_type) {
        Some(t) => t,
        None => {
            eprintln!(
                "Type de fractale invalide: {} (attendu entre 3 et 23, sauf types spéciaux)",
                cli.fractal_type
            );
            std::process::exit(1);
        }
    };

    // Paramètres par défaut pour ce type.
    let mut params = default_params_for_type(fractal_type, cli.width, cli.height);

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

    // Calcul escape-time.
    let start_time = std::time::Instant::now();
    let (iterations, zs) = render_escape_time(&params);
    let render_time = start_time.elapsed();

    // Export PNG.
    let save_start = std::time::Instant::now();
    if let Err(e) = save_png(&params, &iterations, &zs, &cli.output) {
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

