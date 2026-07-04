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
    if !matches!(
        params.fractal_type,
        FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip | FractalType::Tricorn
    ) {
        return Err(format!(
            "Perturbation is only supported for Mandelbrot/Julia/BurningShip/Tricorn; got {:?}",
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
    let (pert_iters, pert_zs, _, _) = render_escape_time_cancellable_with_reuse(&pert_params, &cancel, None, &mut None)
        .ok_or_else(|| "Perturbation render cancelled or failed".to_string())?;
    let perturb_time_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("[quality] perturbation done in {:.0} ms", perturb_time_ms);

    println!(
        "[quality] rendering pure GMP reference (slow): {}x{} bits={}",
        gmp_params.width, gmp_params.height, gmp_params.precision_bits,
    );
    let t1 = Instant::now();
    let (gmp_iters, gmp_zs, _, _) = render_escape_time_cancellable_with_reuse(&gmp_params, &cancel, None, &mut None)
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

    // Tier double-double (~106 b) pour les spirales ultra-sensibles profondes
    // où la mantisse f64 (53 b) du path ComplexExp sature (amplification de
    // Lyapunov du 2⁻⁵² d'arrondi → écart d'itération vs GMP, cf. TODO G2).
    // Équivalent du float128 de Fraktaler-3. e30/e50 partagent le centre
    // spirale `-0.04947…−0.67478…` ; e100 une zone comparable.
    params.use_dd_tier = matches!(
        preset.name,
        "mandelbrot-e30" | "mandelbrot-e50" | "mandelbrot-e100"
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
