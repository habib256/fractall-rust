use std::path::PathBuf;

use clap::Parser;

mod fractal;
mod color;
mod render;
mod io;

use fractal::{AlgorithmMode, apply_lyapunov_preset, default_params_for_type, FractalType, LyapunovPreset};
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

    /// Tolérance de glitch (ex: 1e-4)
    #[arg(long)]
    glitch_tolerance: Option<f64>,

    /// Preset Lyapunov (standard, zircon-city, jellyfish, asymmetric, spaceship, heavy-blocks)
    #[arg(long)]
    lyapunov_preset: Option<String>,

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

    // Override des coordonnées si demandé.
    if let Some(xmin) = cli.xmin {
        params.xmin = xmin;
    }
    if let Some(xmax) = cli.xmax {
        params.xmax = xmax;
    }
    if let Some(ymin) = cli.ymin {
        params.ymin = ymin;
    }
    if let Some(ymax) = cli.ymax {
        params.ymax = ymax;
    }

    // Recentrage éventuel.
    if cli.center_x.is_some() || cli.center_y.is_some() {
        let span_x = params.xmax - params.xmin;
        let span_y = params.ymax - params.ymin;
        let cx = cli
            .center_x
            .unwrap_or((params.xmin + params.xmax) / 2.0);
        let cy = cli
            .center_y
            .unwrap_or((params.ymin + params.ymax) / 2.0);

        params.xmin = cx - span_x / 2.0;
        params.xmax = cx + span_x / 2.0;
        params.ymin = cy - span_y / 2.0;
        params.ymax = cy + span_y / 2.0;
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
    if let Some(glitch_tolerance) = cli.glitch_tolerance {
        if glitch_tolerance > 0.0 {
            params.glitch_tolerance = glitch_tolerance;
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

    // Calcul escape-time.
    let (iterations, zs) = render_escape_time(&params);

    // Export PNG.
    if let Err(e) = save_png(&params, &iterations, &zs, &cli.output) {
        eprintln!("Erreur lors de l'écriture du PNG: {e}");
        std::process::exit(1);
    }
}

