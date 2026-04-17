use crate::fractal::FractalType;

#[derive(Debug, Clone)]
pub struct Preset {
    pub name: &'static str,
    pub description: &'static str,
    pub fractal_type: FractalType,
    pub center_x_hp: &'static str,
    pub center_y_hp: &'static str,
    pub zoom: &'static str,
    pub iterations: u32,
    pub multibrot_power: Option<f64>,
    pub julia_seed: Option<(f64, f64)>,
}

/// Fixed deep-zoom scenes covering the risk areas of the perturbation pipeline.
/// Zooms range from 1e8 (f64 threshold) to 1e18 (GMP reference mandatory).
pub const PRESETS: &[Preset] = &[
    Preset {
        name: "seahorse-valley",
        description: "Mandelbrot Seahorse Valley at zoom 1e8 — classic shallow deep-zoom.",
        fractal_type: FractalType::Mandelbrot,
        center_x_hp: "-0.743643887037158704752191506114774",
        center_y_hp: "0.131825904205311970493132056385139",
        zoom: "1e8",
        iterations: 4096,
        multibrot_power: None,
        julia_seed: None,
    },
    Preset {
        name: "mandelbrot-e13",
        description: "Mandelbrot at zoom 1e13 — just above perturbation activation threshold.",
        fractal_type: FractalType::Mandelbrot,
        center_x_hp: "-1.7499537683537087",
        center_y_hp: "0.0",
        zoom: "1e13",
        iterations: 16384,
        multibrot_power: None,
        julia_seed: None,
    },
    Preset {
        name: "mandelbrot-e17",
        description: "Mandelbrot at zoom 1e17 — forces GMP perturbation path.",
        fractal_type: FractalType::Mandelbrot,
        center_x_hp: "-1.7499537683537087215208540815925",
        center_y_hp: "0.0",
        zoom: "1e17",
        iterations: 32768,
        multibrot_power: None,
        julia_seed: None,
    },
    Preset {
        name: "misiurewicz-m32",
        description: "Mandelbrot Misiurewicz point M32,2 — stress test for BLA escape precision.",
        fractal_type: FractalType::Mandelbrot,
        center_x_hp: "-0.77568377",
        center_y_hp: "0.13646737",
        zoom: "1e12",
        iterations: 8192,
        multibrot_power: None,
        julia_seed: None,
    },
    Preset {
        name: "julia-siegel-disk",
        description: "Julia c=-0.8+0.156i at zoom 1e10 — non-trivial Julia perturbation.",
        fractal_type: FractalType::Julia,
        center_x_hp: "0.0",
        center_y_hp: "0.0",
        zoom: "1e10",
        iterations: 4096,
        multibrot_power: None,
        julia_seed: Some((-0.8, 0.156)),
    },
    Preset {
        name: "burning-ship-antenna",
        description: "Burning Ship antenna at zoom 1e9 — non-conformal BLA (matrix 2x2).",
        fractal_type: FractalType::BurningShip,
        center_x_hp: "-1.7492060",
        center_y_hp: "-0.0286762",
        zoom: "1e9",
        iterations: 4096,
        multibrot_power: None,
        julia_seed: None,
    },
    Preset {
        name: "tricorn-spiral",
        description: "Tricorn spiral at zoom 1e8 — non-conformal (conjugation in iteration).",
        fractal_type: FractalType::Tricorn,
        center_x_hp: "-0.7",
        center_y_hp: "0.15",
        zoom: "1e8",
        iterations: 2048,
        multibrot_power: None,
        julia_seed: None,
    },
    Preset {
        name: "mandelbrot-e18-minibrot",
        description: "Mandelbrot deep minibrot at zoom 1e18 — extreme regime, GMP only.",
        fractal_type: FractalType::Mandelbrot,
        center_x_hp: "-1.74995376835370872152085408159254600000000",
        center_y_hp: "0.0",
        zoom: "1e18",
        iterations: 65536,
        multibrot_power: None,
        julia_seed: None,
    },
];

pub fn find(name: &str) -> Option<&'static Preset> {
    PRESETS.iter().find(|p| p.name == name)
}

pub fn names() -> Vec<&'static str> {
    PRESETS.iter().map(|p| p.name).collect()
}
