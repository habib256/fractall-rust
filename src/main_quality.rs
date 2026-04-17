use std::path::PathBuf;

use clap::{Parser, Subcommand};

mod fractal;
mod color;
mod render;
mod io;
mod quality;

use fractal::FractalType;
use quality::{apply_zoom, compare, params_from_preset, ComparisonOptions};
use quality::metrics::Thresholds;
use quality::presets;
use quality::report::{write_report, write_suite_summary, print_summary_line, ReportInputs};

/// Compare the perturbation pipeline against a pure-GMP reference render,
/// producing per-pixel metrics and PNG diff heatmaps for regression diagnostics.
#[derive(Parser, Debug)]
#[command(
    name = "fractall-quality",
    about = "Qualité/régression perturbation vs GMP pur",
    version,
)]
struct Cli {
    #[arg(long, default_value_t = 256, global = true)]
    width: u32,

    #[arg(long, default_value_t = 256, global = true)]
    height: u32,

    #[arg(long, default_value = "quality-reports", global = true)]
    output_dir: PathBuf,

    /// Override the iteration cap (else uses the preset value).
    #[arg(long, global = true)]
    iterations: Option<u32>,

    /// Override the GMP precision in bits (else auto via perturbation formula).
    #[arg(long, global = true)]
    precision_bits: Option<u32>,

    /// Max iteration-diff tolerated for a PASS verdict.
    #[arg(long, default_value_t = 1.0, global = true)]
    pass_max_iter_diff: f64,

    /// Max fraction of divergent pixels tolerated for PASS.
    #[arg(long, default_value_t = 0.001, global = true)]
    pass_divergence_ratio: f64,

    /// Max iteration-diff tolerated for WARN (above -> FAIL).
    #[arg(long, default_value_t = 3.0, global = true)]
    warn_max_iter_diff: f64,

    #[arg(long, default_value_t = 0.01, global = true)]
    warn_divergence_ratio: f64,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Lister les presets disponibles.
    List,
    /// Exécuter un preset par nom.
    Preset {
        name: String,
    },
    /// Exécuter toute la suite.
    Suite,
    /// Exécuter une scène personnalisée.
    Compare {
        /// Type de fractale (id entier, voir fractall-cli --help)
        #[arg(long = "type")]
        fractal_type: u8,
        #[arg(long)]
        center_x_hp: String,
        #[arg(long)]
        center_y_hp: String,
        #[arg(long)]
        zoom: String,
        /// Julia seed (requis pour --type 4)
        #[arg(long)]
        julia_re: Option<f64>,
        #[arg(long)]
        julia_im: Option<f64>,
        /// Nom de sortie (dossier sous output-dir). Défaut "custom".
        #[arg(long, default_value = "custom")]
        name: String,
    },
}

fn main() {
    let cli = Cli::parse();

    let opt = ComparisonOptions {
        width: cli.width,
        height: cli.height,
        max_iterations: cli.iterations,
        precision_bits: cli.precision_bits,
        thresholds: Thresholds {
            max_iter_diff_pass: cli.pass_max_iter_diff,
            divergence_ratio_pass: cli.pass_divergence_ratio,
            max_iter_diff_warn: cli.warn_max_iter_diff,
            divergence_ratio_warn: cli.warn_divergence_ratio,
        },
    };

    match cli.command {
        Command::List => {
            println!("Presets disponibles:");
            for p in presets::PRESETS {
                println!("  {:<30} {:?} zoom={} iter={}",
                    p.name, p.fractal_type, p.zoom, p.iterations);
                println!("    {}", p.description);
            }
        }
        Command::Preset { name } => {
            let preset = presets::find(&name).unwrap_or_else(|| {
                eprintln!("Preset inconnu: '{}'. Liste: {:?}", name, presets::names());
                std::process::exit(1);
            });
            let params = params_from_preset(preset, &opt);
            match compare(&params, &opt) {
                Ok(out) => {
                    print_summary_line(preset.name, &out.metrics);
                    let inputs = ReportInputs {
                        preset_name: preset.name,
                        params: &out.params,
                        pert_iters: &out.pert_iters,
                        pert_zs: &out.pert_zs,
                        gmp_iters: &out.gmp_iters,
                        gmp_zs: &out.gmp_zs,
                        metrics: &out.metrics,
                    };
                    if let Err(e) = write_report(&cli.output_dir, &inputs) {
                        eprintln!("Erreur écriture rapport: {e}");
                        std::process::exit(1);
                    }
                    println!("Rapport: {}/{}/report.md", cli.output_dir.display(), preset.name);
                }
                Err(e) => {
                    eprintln!("Erreur compare: {e}");
                    std::process::exit(1);
                }
            }
        }
        Command::Suite => {
            let mut rows: Vec<(String, quality::metrics::QualityMetrics)> = Vec::new();
            for preset in presets::PRESETS {
                println!("\n=== {} ===", preset.name);
                println!("{}", preset.description);
                let params = params_from_preset(preset, &opt);
                match compare(&params, &opt) {
                    Ok(out) => {
                        print_summary_line(preset.name, &out.metrics);
                        let inputs = ReportInputs {
                            preset_name: preset.name,
                            params: &out.params,
                            pert_iters: &out.pert_iters,
                            pert_zs: &out.pert_zs,
                            gmp_iters: &out.gmp_iters,
                            gmp_zs: &out.gmp_zs,
                            metrics: &out.metrics,
                        };
                        if let Err(e) = write_report(&cli.output_dir, &inputs) {
                            eprintln!("Erreur écriture rapport {}: {e}", preset.name);
                            continue;
                        }
                        rows.push((preset.name.to_string(), out.metrics));
                    }
                    Err(e) => {
                        eprintln!("Erreur {}: {e}", preset.name);
                    }
                }
            }
            if let Err(e) = write_suite_summary(&cli.output_dir, &rows) {
                eprintln!("Erreur écriture summary: {e}");
                std::process::exit(1);
            }
            println!("\nSuite summary: {}/suite-summary.md", cli.output_dir.display());
        }
        Command::Compare { fractal_type, center_x_hp, center_y_hp, zoom, julia_re, julia_im, name } => {
            let ft = FractalType::from_id(fractal_type).unwrap_or_else(|| {
                eprintln!("Type invalide: {}", fractal_type);
                std::process::exit(1);
            });
            let mut params = fractal::default_params_for_type(ft, opt.width, opt.height);
            params.center_x_hp = Some(center_x_hp.clone());
            params.center_y_hp = Some(center_y_hp.clone());
            if let Ok(v) = center_x_hp.parse::<f64>() {
                params.center_x = v;
            }
            if let Ok(v) = center_y_hp.parse::<f64>() {
                params.center_y = v;
            }
            apply_zoom(&mut params, &zoom);
            if let Some(iters) = opt.max_iterations {
                params.iteration_max = iters;
            }
            if let Some(bits) = opt.precision_bits {
                params.precision_bits = bits;
            }
            if ft == FractalType::Julia {
                let re = julia_re.unwrap_or_else(|| {
                    eprintln!("--julia-re requis pour Julia");
                    std::process::exit(1);
                });
                let im = julia_im.unwrap_or(0.0);
                params.seed = num_complex::Complex64::new(re, im);
            }
            match compare(&params, &opt) {
                Ok(out) => {
                    print_summary_line(&name, &out.metrics);
                    let inputs = ReportInputs {
                        preset_name: &name,
                        params: &out.params,
                        pert_iters: &out.pert_iters,
                        pert_zs: &out.pert_zs,
                        gmp_iters: &out.gmp_iters,
                        gmp_zs: &out.gmp_zs,
                        metrics: &out.metrics,
                    };
                    if let Err(e) = write_report(&cli.output_dir, &inputs) {
                        eprintln!("Erreur écriture rapport: {e}");
                        std::process::exit(1);
                    }
                    println!("Rapport: {}/{}/report.md", cli.output_dir.display(), name);
                }
                Err(e) => {
                    eprintln!("Erreur compare: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
}
