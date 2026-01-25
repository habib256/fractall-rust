use num_complex::Complex64;

use crate::fractal::{AlgorithmMode, FractalParams, FractalType};
use crate::fractal::lyapunov::{LyapunovConfig, LyapunovPreset};

/// Construit des paramètres avec les valeurs par défaut du type,
/// en reprenant la logique de `fractal_definitions.c`.
pub fn default_params_for_type(fractal_type: FractalType, width: u32, height: u32) -> FractalParams {
    // Valeurs communes
    let mut params = FractalParams {
        width,
        height,
        xmin: 0.0,
        xmax: 0.0,
        ymin: 0.0,
        ymax: 0.0,
        seed: Complex64::new(0.0, 0.0),
        iteration_max: 1000,
        bailout: 4.0,
        fractal_type,
        color_mode: 6,   // SmoothPlasma (défaut dans le projet C)
        color_repeat: 40,
        use_gmp: false,
        precision_bits: 256,
        algorithm_mode: AlgorithmMode::Auto,
        bla_threshold: 1e-8,
        glitch_tolerance: 1e-4,
        lyapunov_preset: LyapunovPreset::default(),
        lyapunov_sequence: Vec::new(),
    };

    match fractal_type {
        FractalType::VonKoch => {
            // Von Koch - flocon de neige (vectoriel)
            params.xmin = 0.0;
            params.xmax = 1.0;
            params.ymin = 0.0;
            params.ymax = 1.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 8; // Profondeur de récursion max
        }
        FractalType::Dragon => {
            // Courbe du dragon (vectoriel)
            params.xmin = 0.0;
            params.xmax = 1.0;
            params.ymin = 0.0;
            params.ymax = 1.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 20; // Profondeur de récursion max
        }
        FractalType::Mandelbrot => {
            // Mendelbrot_def
            params.xmin = -2.5;
            params.xmax = 1.5;
            params.ymin = -1.5;
            params.ymax = 1.5;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 9370;
        }
        FractalType::Julia => {
            // Julia_def
            params.xmin = -2.0;
            params.xmax = 2.0;
            params.ymin = -1.5;
            params.ymax = 1.5;
            params.seed = Complex64::new(0.36228, -0.0777);
            params.bailout = 4.0;
            params.iteration_max = 6250;
        }
        FractalType::JuliaSin => {
            // JuliaSin_def
            params.xmin = -std::f64::consts::PI;
            params.xmax = std::f64::consts::PI;
            params.ymin = -2.0;
            params.ymax = 2.0;
            params.seed = Complex64::new(1.0, 0.1);
            params.bailout = 4.0;
            params.iteration_max = 6250;
        }
        FractalType::Newton => {
            // Newton_def
            params.seed = Complex64::new(8.0, 0.0);
            params.xmin = -3.0;
            params.xmax = 3.0;
            params.ymin = -2.0;
            params.ymax = 2.0;
            params.bailout = 4.0;
            params.iteration_max = 1000;
        }
        FractalType::Phoenix => {
            // Phoenix_def
            params.xmin = -2.0;
            params.xmax = 2.0;
            params.ymin = -1.5;
            params.ymax = 1.5;
            params.bailout = 4.0;
            params.iteration_max = 9370;
        }
        FractalType::BarnsleyJulia => {
            // Barnsley1j_def
            params.xmin = -4.0;
            params.xmax = 4.0;
            params.ymin = -3.0;
            params.ymax = 3.0;
            params.seed = Complex64::new(1.1, 0.6);
            params.bailout = 4.0;
            params.iteration_max = 3120;
        }
        FractalType::BarnsleyMandelbrot => {
            // Barnsley1m_def
            params.xmin = -3.0;
            params.xmax = 3.0;
            params.ymin = -2.0;
            params.ymax = 2.0;
            params.bailout = 4.0;
            params.iteration_max = 9370;
        }
        FractalType::MagnetJulia => {
            // Magnet1j_def
            params.seed = Complex64::new(1.625458, -0.306159);
            params.xmin = -2.0;
            params.xmax = 2.0;
            params.ymin = -2.0;
            params.ymax = 2.0;
            params.bailout = 4.0;
            params.iteration_max = 9370;
        }
        FractalType::MagnetMandelbrot => {
            // Magnet1m_def
            params.xmin = -3.0;
            params.xmax = 2.0;
            params.ymin = -2.0;
            params.ymax = 2.0;
            params.bailout = 4.0;
            params.iteration_max = 9370;
        }
        FractalType::BurningShip => {
            // BurningShip_def
            params.xmin = -2.5;
            params.xmax = 1.5;
            params.ymin = -2.0;
            params.ymax = 2.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 9370;
        }
        FractalType::Buffalo => {
            // Buffalo_def
            params.xmin = -2.5;
            params.xmax = 1.5;
            params.ymin = -2.0;
            params.ymax = 2.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 9370;
        }
        FractalType::Tricorn => {
            // Tricorn_def
            params.xmin = -2.5;
            params.xmax = 1.5;
            params.ymin = -1.5;
            params.ymax = 1.5;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 9370;
        }
        FractalType::Mandelbulb => {
            // Mandelbulb_def
            params.xmin = -1.5;
            params.xmax = 1.5;
            params.ymin = -1.5;
            params.ymax = 1.5;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 9370;
        }
        FractalType::Buddhabrot => {
            // Buddhabrot_def - densité de trajectoires
            params.xmin = -2.5;
            params.xmax = 1.5;
            params.ymin = -1.5;
            params.ymax = 1.5;
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
            // PerpendicularBurningShip_def
            params.xmin = -2.5;
            params.xmax = 1.5;
            params.ymin = -1.5;
            params.ymax = 1.5;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 5000;
        }
        FractalType::Celtic => {
            // Celtic_def
            params.xmin = -2.0;
            params.xmax = 1.0;
            params.ymin = -1.5;
            params.ymax = 1.5;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 5000;
        }
        FractalType::AlphaMandelbrot => {
            // AlphaMandelbrot_def
            params.xmin = -2.5;
            params.xmax = 1.5;
            params.ymin = -1.5;
            params.ymax = 1.5;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 2000;
        }
        FractalType::PickoverStalks => {
            // PickoverStalks_def
            params.xmin = -2.0;
            params.xmax = 1.0;
            params.ymin = -1.5;
            params.ymax = 1.5;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 100.0;
            params.iteration_max = 1000;
            params.color_repeat = 2;
        }
        FractalType::Nova => {
            // Nova_def
            params.xmin = -3.0;
            params.xmax = 3.0;
            params.ymin = -2.0;
            params.ymax = 2.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 20.0;
            params.iteration_max = 500;
        }
        FractalType::Multibrot => {
            // Multibrot_def
            params.xmin = -2.5;
            params.xmax = 1.5;
            params.ymin = -1.5;
            params.ymax = 1.5;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 5000;
        }
        FractalType::Nebulabrot => {
            // Nebulabrot_def - RGB densité (R=50, G=500, B=5000 iter)
            params.xmin = -2.5;
            params.xmax = 1.5;
            params.ymin = -1.5;
            params.ymax = 1.5;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 5000;
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
    params.xmin = config.xmin;
    params.xmax = config.xmax;
    params.ymin = config.ymin;
    params.ymax = config.ymax;
}

