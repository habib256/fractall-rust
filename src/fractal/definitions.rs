use num_complex::Complex64;

use crate::fractal::{AlgorithmMode, FractalParams, FractalType, OutColoringMode};
use crate::fractal::lyapunov::{LyapunovConfig, LyapunovPreset};

/// Construit des paramètres avec les valeurs par défaut du type,
/// en reprenant la logique de `fractal_definitions.c`.
pub fn default_params_for_type(fractal_type: FractalType, width: u32, height: u32) -> FractalParams {
    // Valeurs communes
    let mut params = FractalParams {
        width,
        height,
        center_x: 0.0,
        center_y: 0.0,
        span_x: 0.0,
        span_y: 0.0,
        seed: Complex64::new(0.0, 0.0),
        iteration_max: 2500,
        bailout: 4.0,
        fractal_type,
        color_mode: 6,   // SmoothPlasma (défaut dans le projet C)
        color_repeat: 40,
        use_gmp: false,
        precision_bits: 256,
        algorithm_mode: AlgorithmMode::Auto,
        bla_threshold: 1e-8,
        bla_validity_scale: 1.0,
        glitch_tolerance: 1e-4,
        series_order: 2,
        series_threshold: 1e-6,
        series_error_tolerance: 1e-9,
        glitch_neighbor_pass: true,
        series_standalone: false,
        max_secondary_refs: 3,
        min_glitch_cluster_size: 100,
        multibrot_power: 2.5,
        lyapunov_preset: LyapunovPreset::default(),
        lyapunov_sequence: Vec::new(),
        // Enable distance estimation and interior detection by default for better rendering
        // These features use dual numbers (Section 5 and 6 of deep zoom theory)
        enable_distance_estimation: false, // Can be enabled for distance field coloring
        enable_interior_detection: true,   // Enable by default to properly color interior points
        interior_threshold: 0.001,
        out_coloring_mode: OutColoringMode::Smooth,
    };

    match fractal_type {
        FractalType::VonKoch => {
            // Von Koch - flocon de neige (vectoriel)
            params.center_x = 0.5;
            params.center_y = 0.5;
            params.span_x = 1.0;
            params.span_y = 1.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 8; // Profondeur de récursion max
        }
        FractalType::Dragon => {
            // Courbe du dragon (vectoriel)
            params.center_x = 0.5;
            params.center_y = 0.5;
            params.span_x = 1.0;
            params.span_y = 1.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 20; // Profondeur de récursion max
        }
        FractalType::Mandelbrot => {
            // Mendelbrot_def: xmin=-2.5, xmax=1.5, ymin=-1.5, ymax=1.5
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::Julia => {
            // Julia_def: xmin=-2.0, xmax=2.0, ymin=-1.5, ymax=1.5
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.36228, -0.0777);
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::JuliaSin => {
            // JuliaSin_def: xmin=-PI, xmax=PI, ymin=-2.0, ymax=2.0
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 2.0 * std::f64::consts::PI;
            params.span_y = 4.0;
            params.seed = Complex64::new(1.0, 0.1);
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::Newton => {
            // Newton_def: xmin=-3.0, xmax=3.0, ymin=-2.0, ymax=2.0
            params.seed = Complex64::new(8.0, 0.0);
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 6.0;
            params.span_y = 4.0;
            params.bailout = 4.0;
            params.iteration_max = 1000;
        }
        FractalType::Phoenix => {
            // Phoenix_def: xmin=-2.0, xmax=2.0, ymin=-1.5, ymax=1.5
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::BarnsleyJulia => {
            // Barnsley1j_def: xmin=-4.0, xmax=4.0, ymin=-3.0, ymax=3.0
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 8.0;
            params.span_y = 6.0;
            params.seed = Complex64::new(1.1, 0.6);
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::BarnsleyMandelbrot => {
            // Barnsley1m_def: xmin=-3.0, xmax=3.0, ymin=-2.0, ymax=2.0
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 6.0;
            params.span_y = 4.0;
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::MagnetJulia => {
            // Magnet1j_def: xmin=-2.0, xmax=2.0, ymin=-2.0, ymax=2.0
            params.seed = Complex64::new(1.625458, -0.306159);
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 4.0;
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::MagnetMandelbrot => {
            // Magnet1m_def: xmin=-3.0, xmax=2.0, ymin=-2.0, ymax=2.0
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 5.0;
            params.span_y = 4.0;
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::BurningShip => {
            // BurningShip_def: xmin=-2.5, xmax=1.5, ymin=-2.0, ymax=2.0
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 4.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::Buffalo => {
            // Buffalo_def: xmin=-2.5, xmax=1.5, ymin=-2.0, ymax=2.0
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 4.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::Tricorn => {
            // Tricorn_def: xmin=-2.5, xmax=1.5, ymin=-1.5, ymax=1.5
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::Mandelbulb => {
            // Mandelbulb_def: xmin=-1.5, xmax=1.5, ymin=-1.5, ymax=1.5
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 3.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::Buddhabrot => {
            // Buddhabrot_def: xmin=-2.5, xmax=1.5, ymin=-1.5, ymax=1.5
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 220;
        }
        FractalType::Lyapunov => {
            // Lyapunov_def - Zircon City par défaut
            apply_lyapunov_preset(&mut params, LyapunovPreset::ZirconCity);
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 2000;
        }
        FractalType::PerpendicularBurningShip => {
            // PerpendicularBurningShip_def: xmin=-2.5, xmax=1.5, ymin=-1.5, ymax=1.5
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::Celtic => {
            // Celtic_def: xmin=-2.0, xmax=1.0, ymin=-1.5, ymax=1.5
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 3.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::AlphaMandelbrot => {
            // AlphaMandelbrot_def: xmin=-2.5, xmax=1.5, ymin=-1.5, ymax=1.5
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 2000;
        }
        FractalType::PickoverStalks => {
            // PickoverStalks_def: xmin=-2.0, xmax=1.0, ymin=-1.5, ymax=1.5
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 3.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 100.0;
            params.iteration_max = 1000;
            params.color_repeat = 2;
        }
        FractalType::Nova => {
            // Nova_def: xmin=-3.0, xmax=3.0, ymin=-2.0, ymax=2.0
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 6.0;
            params.span_y = 4.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 20.0;
            params.iteration_max = 500;
        }
        FractalType::Multibrot => {
            // Multibrot_def: xmin=-2.5, xmax=1.5, ymin=-1.5, ymax=1.5
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::Nebulabrot => {
            // Nebulabrot_def: xmin=-2.5, xmax=1.5, ymin=-1.5, ymax=1.5
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
    }

    params
}

/// Applique un preset Lyapunov aux paramètres.
/// Met à jour les bornes du domaine et la séquence.
pub fn apply_lyapunov_preset(params: &mut FractalParams, preset: LyapunovPreset) {
    let config = LyapunovConfig::from_preset(preset);
    params.lyapunov_preset = preset;
    params.lyapunov_sequence = config.sequence;
    params.center_x = (config.xmin + config.xmax) * 0.5;
    params.center_y = (config.ymin + config.ymax) * 0.5;
    params.span_x = config.xmax - config.xmin;
    params.span_y = config.ymax - config.ymin;
}

