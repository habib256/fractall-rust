use std::path::PathBuf;

use clap::{Parser, Subcommand};

use fractall_cli::{fractal, gpu, quality, render};

use fractal::FractalType;
use quality::metrics::Thresholds;
use quality::presets;
use quality::report::{
    print_summary_line, write_report, write_suite_summary, write_suite_summary_json, ReportInputs,
};
use quality::{apply_zoom, compare, compare_gpu, params_from_preset, ComparisonOptions};

/// Compare the perturbation pipeline against a pure-GMP reference render,
/// producing per-pixel metrics and PNG diff heatmaps for regression diagnostics.
#[derive(Parser, Debug)]
#[command(
    name = "fractall-quality",
    about = "Qualité/régression perturbation vs GMP pur",
    version
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

    /// p99 iteration-diff above which -> FAIL (robust to sparse boundary
    /// outliers; a high `max` on a few dispersed pixels stays WARN, cf. G6).
    #[arg(long, default_value_t = 1.0, global = true)]
    fail_p99_iter_diff: f64,

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
    Preset { name: String },
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
    /// Parité GPU↔CPU sur l'échelle de presets GPU (verrou G9.4/9.5).
    /// Rapports sous <output-dir>/gpu/ ; "pert" = GPU, "gmp" = juge GMP pur.
    GpuSuite,
    /// Chronométrage GPU vs CPU (G14.5) : temps total GPU (+ correction CPU
    /// des pixels non fiables, + repli plein cadre éventuel) contre le CPU
    /// seul (Auto, et perturbation forcée en option), runs entrelacés,
    /// ventilation orbite / upload / GPU / readback / correction. Accord des
    /// images GPU↔CPU rapporté (le juge GMP reste `gpu-suite`).
    GpuBench {
        /// Cas `preset[@zoom]` (presets CPU ou GPU) ; défaut : échelle 1e4→1e25.
        #[arg(long = "case")]
        cases: Vec<String>,
        /// Tailles carrées (px de côté).
        #[arg(long, value_delimiter = ',', default_value = "512,1024,2048")]
        sizes: Vec<u32>,
        /// Runs par bras (entrelacés).
        #[arg(long, default_value_t = 5)]
        runs: usize,
        /// Bras CPU perturbation forcée en plus de CPU Auto.
        #[arg(long)]
        cpu_perturbation: bool,
    },
    /// Parité GPU↔CPU sur une scène personnalisée (mêmes options que compare).
    GpuCompare {
        /// Type de fractale (id entier ; GPU : Mandelbrot/Julia/BurningShip)
        #[arg(long = "type")]
        fractal_type: u8,
        #[arg(long)]
        center_x_hp: String,
        #[arg(long)]
        center_y_hp: String,
        #[arg(long)]
        zoom: String,
        #[arg(long)]
        julia_re: Option<f64>,
        #[arg(long)]
        julia_im: Option<f64>,
        /// Nom de sortie (dossier sous output-dir/gpu). Défaut "gpu-custom".
        #[arg(long, default_value = "gpu-custom")]
        name: String,
    },
}

/// Closure de rendu GPU pour `compare_gpu` : dispatch unifié (std f32 ou
/// perturbation f32 selon zoom/type), `None` si config non supportée.
fn gpu_render_closure(
    renderer: &gpu::GpuRenderer,
) -> impl Fn(&fractal::FractalParams) -> Option<(Vec<u32>, Vec<num_complex::Complex64>)> + '_ {
    move |params| {
        let cancel = std::sync::atomic::AtomicBool::new(false);
        renderer
            .render_dispatch(
                render::GpuRenderPlan::for_params(params),
                params,
                &cancel,
                None,
                None,
            )
            .map(|r| (r.iterations, r.zs))
    }
}

/// Initialise le GPU ou sort avec un message clair (code 2 = indisponible).
fn gpu_renderer_or_exit() -> gpu::GpuRenderer {
    match gpu::GpuRenderer::new() {
        Some(r) => r,
        None => {
            eprintln!("GPU indisponible (wgpu n'a trouvé aucun adaptateur) — gpu-compare/gpu-suite impossibles");
            std::process::exit(2);
        }
    }
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
            p99_iter_diff_fail: cli.fail_p99_iter_diff,
            divergence_ratio_warn: cli.warn_divergence_ratio,
        },
    };

    match cli.command {
        Command::List => {
            println!("Presets disponibles:");
            for p in presets::PRESETS {
                println!(
                    "  {:<30} {:?} zoom={} iter={}",
                    p.name, p.fractal_type, p.zoom, p.iterations
                );
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
                    println!(
                        "Rapport: {}/{}/report.md",
                        cli.output_dir.display(),
                        preset.name
                    );
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
            if let Err(e) = write_suite_summary_json(&cli.output_dir, &rows, &opt.thresholds) {
                eprintln!("Erreur écriture summary JSON: {e}");
                std::process::exit(1);
            }
            println!(
                "\nSuite summary: {}/suite-summary.md",
                cli.output_dir.display()
            );
        }
        Command::Compare {
            fractal_type,
            center_x_hp,
            center_y_hp,
            zoom,
            julia_re,
            julia_im,
            name,
        } => {
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
                params.engine.precision_bits = bits;
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
        Command::GpuSuite => {
            let renderer = gpu_renderer_or_exit();
            let render_gpu = gpu_render_closure(&renderer);
            let gpu_dir = cli.output_dir.join("gpu");
            let mut rows: Vec<(String, quality::metrics::QualityMetrics)> = Vec::new();
            for preset in presets::GPU_PRESETS {
                println!("\n=== {} ===", preset.name);
                println!("{}", preset.description);
                let params = params_from_preset(preset, &opt);
                match compare_gpu(&params, &opt, &render_gpu) {
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
                        if let Err(e) = write_report(&gpu_dir, &inputs) {
                            eprintln!("Erreur écriture rapport: {e}");
                        }
                        rows.push((preset.name.to_string(), out.metrics));
                    }
                    Err(e) => {
                        eprintln!("Erreur gpu-compare {}: {e}", preset.name);
                    }
                }
            }
            if let Err(e) = write_suite_summary(&gpu_dir, &rows) {
                eprintln!("Erreur écriture summary: {e}");
            }
            if let Err(e) = write_suite_summary_json(&gpu_dir, &rows, &opt.thresholds) {
                eprintln!("Erreur écriture summary JSON: {e}");
            }
            println!(
                "\nGPU suite summary: {}/suite-summary.md",
                gpu_dir.display()
            );
        }
        Command::GpuBench {
            cases,
            sizes,
            runs,
            cpu_perturbation,
        } => {
            let renderer = gpu_renderer_or_exit();
            run_gpu_bench(&renderer, &cases, &sizes, runs, cpu_perturbation, &opt);
        }
        Command::GpuCompare {
            fractal_type,
            center_x_hp,
            center_y_hp,
            zoom,
            julia_re,
            julia_im,
            name,
        } => {
            let renderer = gpu_renderer_or_exit();
            let render_gpu = gpu_render_closure(&renderer);
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
            if ft == FractalType::Julia {
                let re = julia_re.unwrap_or_else(|| {
                    eprintln!("--julia-re requis pour Julia");
                    std::process::exit(1);
                });
                let im = julia_im.unwrap_or(0.0);
                params.seed = num_complex::Complex64::new(re, im);
            }
            let gpu_dir = cli.output_dir.join("gpu");
            match compare_gpu(&params, &opt, &render_gpu) {
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
                    if let Err(e) = write_report(&gpu_dir, &inputs) {
                        eprintln!("Erreur écriture rapport: {e}");
                        std::process::exit(1);
                    }
                    println!("Rapport: {}/{}/report.md", gpu_dir.display(), name);
                }
                Err(e) => {
                    eprintln!("Erreur gpu-compare: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
}

/// Cas par défaut de `gpu-bench` : plage du kernel perturbation f32
/// (span ≥ 1e-28) — seahorse 1e4→1e8, e13, e17, centre e30 dézoomé, Julia.
const GPU_BENCH_DEFAULT_CASES: &[&str] = &[
    "gpu-seahorse-1e4",
    "gpu-seahorse-1e6",
    "gpu-seahorse-1e8",
    "gpu-mandelbrot-e13",
    "mandelbrot-e17",
    "mandelbrot-e30@1e20@20000",
    "mandelbrot-e30@1e25@20000",
    "julia-siegel-disk",
];

fn find_any_preset(name: &str) -> Option<&'static presets::Preset> {
    presets::find(name).or_else(|| presets::GPU_PRESETS.iter().find(|p| p.name == name))
}

fn median_of(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
}

fn min_of(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::INFINITY, f64::min)
}

fn run_gpu_bench(
    renderer: &gpu::GpuRenderer,
    cases: &[String],
    sizes: &[u32],
    runs: usize,
    cpu_perturbation: bool,
    opt: &ComparisonOptions,
) {
    use std::sync::atomic::AtomicBool;
    use std::time::Instant;
    let cases: Vec<String> = if cases.is_empty() {
        GPU_BENCH_DEFAULT_CASES.iter().map(|s| s.to_string()).collect()
    } else {
        cases.to_vec()
    };
    println!(
        "# gpu-bench — {} — kernel perturbation {} — {} runs/bras (entrelacés)",
        renderer.adapter_label(),
        if renderer.perturbation_kernel_is_f32() { "f32+fiabilité" } else { "f64" },
        runs
    );
    println!(
        "| cas | taille | GPU total min/med (s) | orbite | upload | GPU | readback | fix CPU | px→CPU | repli | CPU Auto min/med (s) | CPU pert min/med (s) | gain (min) | px ≠ CPU |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for case in &cases {
        // `preset[@zoom[@itérations]]`
        let mut parts = case.split('@');
        let name = parts.next().unwrap_or_default();
        let zoom = parts.next().filter(|z| !z.is_empty());
        let iters: Option<u32> = parts.next().and_then(|i| i.parse().ok());
        let Some(preset) = find_any_preset(name) else {
            eprintln!("cas inconnu : {name}");
            continue;
        };
        for &size in sizes {
            let o = ComparisonOptions {
                width: size,
                height: size,
                max_iterations: opt.max_iterations,
                precision_bits: opt.precision_bits,
                ..Default::default()
            };
            let mut params = params_from_preset(preset, &o);
            if let Some(z) = zoom {
                apply_zoom(&mut params, z);
            }
            if let Some(n) = iters {
                params.iteration_max = n;
            }
            let mut pert_params = params.clone();
            pert_params.engine.algorithm_mode = fractal::AlgorithmMode::Perturbation;
            let cancel = AtomicBool::new(false);
            let (mut t_gpu, mut t_cpu, mut t_pert) = (Vec::new(), Vec::new(), Vec::new());
            let mut tims = Vec::new();
            let mut gpu_iters: Option<Vec<u32>> = None;
            let mut cpu_iters: Option<Vec<u32>> = None;
            let mut fell_back_runs = 0usize;
            let mut gpu_unavailable = false;
            for _ in 0..runs {
                // Bras GPU = ce que fait `fractall-cli --gpu` : dispatch GPU,
                // puis repli CPU plein cadre si `None`.
                let t0 = Instant::now();
                let r = renderer.render_dispatch(
                    render::GpuRenderPlan::for_params(&params),
                    &params,
                    &cancel,
                    None,
                    None,
                );
                let tim = renderer.take_last_perturb_timings();
                let iters = match r {
                    Some(r) => r.iterations,
                    None => {
                        if tim.map_or(true, |t| !t.fell_back) {
                            gpu_unavailable = true;
                        }
                        fell_back_runs += 1;
                        render::render_escape_time(&params).iterations
                    }
                };
                t_gpu.push(t0.elapsed().as_secs_f64());
                if let Some(t) = tim {
                    tims.push(t);
                }
                gpu_iters.get_or_insert(iters);

                let t0 = Instant::now();
                let out = render::render_escape_time(&params);
                t_cpu.push(t0.elapsed().as_secs_f64());
                cpu_iters.get_or_insert(out.iterations);

                if cpu_perturbation {
                    let t0 = Instant::now();
                    let _ = render::render_escape_time(&pert_params);
                    t_pert.push(t0.elapsed().as_secs_f64());
                }
            }
            let avg = |f: &dyn Fn(&gpu::GpuPerturbTimings) -> f64| {
                if tims.is_empty() {
                    f64::NAN
                } else {
                    tims.iter().map(f).sum::<f64>() / tims.len() as f64
                }
            };
            let frac = avg(&|t| t.unreliable as f64 / t.pixels.max(1) as f64);
            let diff = match (&gpu_iters, &cpu_iters) {
                (Some(a), Some(b)) => {
                    a.iter().zip(b).filter(|(x, y)| x.abs_diff(**y) > 1).count() as f64
                        / a.len().max(1) as f64
                }
                _ => f64::NAN,
            };
            let pert_cell = if cpu_perturbation {
                format!("{:.3} / {:.3}", min_of(&t_pert), median_of(&t_pert))
            } else {
                "-".into()
            };
            println!(
                "| {} | {}² | {:.3} / {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.2}% | {} | {:.3} / {:.3} | {} | {:.2}× | {:.3}% |",
                case,
                size,
                min_of(&t_gpu),
                median_of(&t_gpu),
                avg(&|t| t.ref_s),
                avg(&|t| t.upload_s),
                avg(&|t| t.gpu_s),
                avg(&|t| t.readback_s),
                avg(&|t| t.cpu_fix_s),
                100.0 * frac,
                if gpu_unavailable {
                    "GPU n/a".to_string()
                } else {
                    format!("{}/{}", fell_back_runs, runs)
                },
                min_of(&t_cpu),
                median_of(&t_cpu),
                pert_cell,
                min_of(&t_cpu) / min_of(&t_gpu),
                100.0 * diff,
            );
        }
    }
}
