pub mod metrics;
pub mod presets;
pub mod report;

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

use num_complex::Complex64;
use rug::Float;

use crate::fractal::{AlgorithmMode, FractalParams, FractalType};
use crate::render::render_escape_time_cancellable_with_reuse;

use metrics::{QualityMetrics, Thresholds};
use presets::Preset;

pub struct ComparisonOutput {
    pub params: FractalParams,
    pub pert_iters: Vec<u32>,
    pub pert_zs: Vec<Complex64>,
    pub gmp_iters: Vec<u32>,
    pub gmp_zs: Vec<Complex64>,
    pub metrics: QualityMetrics,
}

pub struct ComparisonOptions {
    pub width: u32,
    pub height: u32,
    pub max_iterations: Option<u32>,
    pub precision_bits: Option<u32>,
    pub thresholds: Thresholds,
}

impl Default for ComparisonOptions {
    fn default() -> Self {
        ComparisonOptions {
            width: 256,
            height: 256,
            max_iterations: None,
            precision_bits: None,
            thresholds: Thresholds::default(),
        }
    }
}

pub fn compare(params: &FractalParams, opt: &ComparisonOptions) -> Result<ComparisonOutput, String> {
    // NB : la perturbation FONCTIONNE pour d'autres types bytecode (Multibrot,
    // variantes Julia… — vérifié == f64 direct, cf.
    // `perturbation::tests::perturbation_matches_f64_{celtic,buffalo,perpendicular…}`).
    // La comparaison vs GMP exige un renderer GMP du type ET un lieu où la ground
    // truth converge à la précision demandée. Celtic/Buffalo/PerpBS sont admis :
    // leurs presets (`presets.rs`) visent des frontières lisses HORS axes de
    // pliage — l'antenne -1.75 de Buffalo/PerpBS, elle, exige 256+ b (GMP-128
    // non convergé, cf. TODO G3) et un écart pert↔GMP y reflèterait le plancher
    // f64, PAS un bug. Pour un `compare` ad-hoc sur ces familles hirsutes,
    // vérifier la stabilité GMP par précision (P vs 2P) avant de conclure.
    if !matches!(
        params.fractal_type,
        FractalType::Mandelbrot
            | FractalType::Julia
            | FractalType::BurningShip
            | FractalType::Tricorn
            | FractalType::Celtic
            | FractalType::Buffalo
            | FractalType::PerpendicularBurningShip
    ) {
        return Err(format!(
            "QA suite vs GMP limitée aux types Mandelbrot-family à renderer GMP vérifié \
             (M/J/BS/Tricorn/Celtic/Buffalo/PerpBS); pas de comparaison fiable pour {:?}",
            params.fractal_type
        ));
    }

    let cancel = Arc::new(AtomicBool::new(false));

    let mut pert_params = params.clone();
    pert_params.algorithm_mode = AlgorithmMode::Perturbation;
    pert_params.use_gmp = false;

    let mut gmp_params = params.clone();
    gmp_params.algorithm_mode = AlgorithmMode::ReferenceGmp;
    gmp_params.use_gmp = true;

    println!(
        "[quality] rendering perturbation: {}x{} type={:?} iter_max={} bits={}",
        pert_params.width, pert_params.height, pert_params.fractal_type,
        pert_params.iteration_max, pert_params.precision_bits,
    );
    let t0 = Instant::now();
    let (pert_iters, pert_zs, _, _) = render_escape_time_cancellable_with_reuse(&pert_params, &cancel, None, &mut None, None)
        .ok_or_else(|| "Perturbation render cancelled or failed".to_string())?;
    let perturb_time_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("[quality] perturbation done in {:.0} ms", perturb_time_ms);

    // Afficher la précision EFFECTIVE (formule zoom-aware), pas le champ
    // utilisateur (plancher, souvent 256) : le rendu GMP interne recalcule via
    // `compute_perturbation_precision_bits` — imprimer le plancher a fait
    // croire à un probe invalide (seahorse 1e1392 : « bits=256 » affiché,
    // 4654 b réellement utilisés).
    println!(
        "[quality] rendering pure GMP reference (slow): {}x{} bits={}",
        gmp_params.width,
        gmp_params.height,
        crate::fractal::perturbation::compute_perturbation_precision_bits(&gmp_params),
    );
    let t1 = Instant::now();
    let (gmp_iters, gmp_zs, _, _) = render_escape_time_cancellable_with_reuse(&gmp_params, &cancel, None, &mut None, None)
        .ok_or_else(|| "GMP reference render cancelled or failed".to_string())?;
    let gmp_time_ms = t1.elapsed().as_secs_f64() * 1000.0;
    println!("[quality] GMP done in {:.0} ms", gmp_time_ms);

    let m = metrics::compute(
        params.width,
        params.height,
        params.iteration_max,
        &pert_iters,
        &pert_zs,
        &gmp_iters,
        &gmp_zs,
        perturb_time_ms,
        gmp_time_ms,
        opt.thresholds,
    );

    Ok(ComparisonOutput {
        params: params.clone(),
        pert_iters,
        pert_zs,
        gmp_iters,
        gmp_zs,
        metrics: m,
    })
}

/// Compare le rendu **GPU** (dispatch unifié `GpuRenderer::render_dispatch`,
/// injecté par le binaire via closure — ce module ne dépend pas de `gpu/`) au
/// **juge GMP pur** (ground truth, même standard que `compare`). Verrou de
/// parité device du jalon G9.4/9.5 : l'auto-GPU exige que le kernel passe ce
/// gate aux seuils standard.
///
/// ⚠️ Le juge N'EST PAS le CPU Auto : à iterations élevées le f64 direct
/// diverge lui-même de la vérité sur le bord chaotique (mesuré seahorse
/// 1e6/5000 iters 256² : CPU-std vs GMP div ~6 %, quand CPU-perturbation vs
/// GMP div 0.001 — la perturbation est ancrée sur une référence GMP exacte).
/// Un juge Auto pénaliserait un kernel GPU perturbation PLUS JUSTE que lui
/// (observé : kernel df64 « FAIL p99 174 » vs juge Auto, artefact du juge).
///
/// Réutilise `ComparisonOutput` : champs `pert_*` = GPU, `gmp_*` = juge GMP
/// (mêmes rapports/heatmaps ; `pert.png` = image GPU, `gmp.png` = image juge).
pub fn compare_gpu(
    params: &FractalParams,
    opt: &ComparisonOptions,
    render_gpu: &dyn Fn(&FractalParams) -> Option<(Vec<u32>, Vec<Complex64>)>,
) -> Result<ComparisonOutput, String> {
    let cancel = Arc::new(AtomicBool::new(false));

    println!(
        "[quality] rendering GPU: {}x{} type={:?} iter_max={}",
        params.width, params.height, params.fractal_type, params.iteration_max,
    );
    let t0 = Instant::now();
    let (gpu_iters, gpu_zs) = render_gpu(params)
        .ok_or_else(|| "rendu GPU indisponible (pas de GPU, ou type/config non supporté)".to_string())?;
    let gpu_time_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("[quality] GPU done in {:.0} ms", gpu_time_ms);

    let mut cpu_params = params.clone();
    cpu_params.algorithm_mode = AlgorithmMode::ReferenceGmp;
    cpu_params.use_gmp = true;
    println!(
        "[quality] rendering GMP judge (ground truth): {}x{}",
        cpu_params.width, cpu_params.height,
    );
    let t1 = Instant::now();
    let (cpu_iters, cpu_zs, _, _) =
        render_escape_time_cancellable_with_reuse(&cpu_params, &cancel, None, &mut None, None)
            .ok_or_else(|| "GMP judge render cancelled or failed".to_string())?;
    let cpu_time_ms = t1.elapsed().as_secs_f64() * 1000.0;
    println!("[quality] GMP judge done in {:.0} ms", cpu_time_ms);

    let m = metrics::compute(
        params.width,
        params.height,
        params.iteration_max,
        &gpu_iters,
        &gpu_zs,
        &cpu_iters,
        &cpu_zs,
        gpu_time_ms,
        cpu_time_ms,
        opt.thresholds,
    );

    Ok(ComparisonOutput {
        params: params.clone(),
        pert_iters: gpu_iters,
        pert_zs: gpu_zs,
        gmp_iters: cpu_iters,
        gmp_zs: cpu_zs,
        metrics: m,
    })
}

/// Build a FractalParams for a preset by merging preset fields onto default_params_for_type.
pub fn params_from_preset(preset: &Preset, opt: &ComparisonOptions) -> FractalParams {
    let mut params = crate::fractal::default_params_for_type(
        preset.fractal_type,
        opt.width,
        opt.height,
    );

    params.center_x_hp = Some(preset.center_x_hp.to_string());
    params.center_y_hp = Some(preset.center_y_hp.to_string());
    if let Ok(v) = preset.center_x_hp.parse::<f64>() {
        params.center_x = v;
    }
    if let Ok(v) = preset.center_y_hp.parse::<f64>() {
        params.center_y = v;
    }

    apply_zoom(&mut params, preset.zoom);

    params.iteration_max = opt.max_iterations.unwrap_or(preset.iterations);
    // Precedence: CLI --precision-bits > preset override > default (256 via default_params_for_type).
    if let Some(bits) = opt.precision_bits {
        params.precision_bits = bits;
    } else if let Some(bits) = preset.precision_bits {
        params.precision_bits = bits;
    }

    if let Some((sre, sim)) = preset.julia_seed {
        params.seed = Complex64::new(sre, sim);
    }
    if let Some(power) = preset.multibrot_power {
        params.multibrot_power = power;
    }

    // Tier double-double (~106 b) pour les points **ultra-sensibles** où la
    // mantisse f64 (53 b) sature — l'amplification de Lyapunov transforme le
    // 2⁻⁵² d'arrondi en écart d'itération vs GMP (cf. TODO G2). Équivalent du
    // float128 de Fraktaler-3 (sélectionné par son wisdom). Concerne les
    // spirales profondes (e30/e50/e100) et le seahorse 1e8 (edge à sensibilité
    // extrême : f64 WARN div 0.0018 même après le fix epsilon BLA 2⁻⁵³, dd le
    // rend pixel-exact). Les autres presets (e13/e17/julia/BS/tricorn) matchent
    // la GMP en f64/ComplexExp.
    //
    // ⚠️ **misiurewicz-m32 (1e12) et mandelbrot-e18-minibrot (1e18) RETIRÉS de
    // la liste dd (2026-07-15)** : depuis le fix de l'epsilon de validité BLA
    // (2⁻²⁴ f32 → 2⁻⁵³ f64, cf. `delta::BLA_MANTISSA_EPSILON`), ils sont
    // **pixel-exacts en f64 pur** vs GMP (max_diff=0 à 96²/128²/160²) — leur
    // « besoin dd » venait de l'over-skip BLA, PAS d'un vrai plancher de
    // mantisse. Les laisser en f64 fait que la QA vérifie le path que la
    // **production utilise réellement** (dd = opt-in) → verrou anti-régression
    // f64 sur ces scènes (complète les goldens e15/e20/e50).
    params.use_dd_tier = matches!(
        preset.name,
        "mandelbrot-e30" | "mandelbrot-e50" | "mandelbrot-e100" | "seahorse-valley"
    );

    params
}

/// Apply --zoom (notation scientifique) to params.span_x_hp / span_y_hp, matching src/main.rs:215.
pub fn apply_zoom(params: &mut FractalParams, zoom_str: &str) {
    let prec = 1024u32;
    let zoom_gmp = match Float::parse(zoom_str) {
        Ok(parsed) => Float::with_val(prec, parsed),
        Err(_) => {
            eprintln!("[quality] cannot parse zoom '{}', using 1.0", zoom_str);
            Float::with_val(prec, 1.0)
        }
    };
    let four = Float::with_val(prec, 4.0);
    let span_x_gmp = four / &zoom_gmp;
    let aspect = params.height as f64 / params.width as f64;
    let span_y_gmp = Float::with_val(prec, &span_x_gmp * aspect);
    params.span_x = span_x_gmp.to_f64();
    params.span_y = span_y_gmp.to_f64();
    params.span_x_hp = Some(span_x_gmp.to_string());
    params.span_y_hp = Some(span_y_gmp.to_string());
}
