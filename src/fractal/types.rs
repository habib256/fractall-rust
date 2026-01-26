use num_complex::Complex64;

/// Types de fractales pris en charge par la version CLI.
///
/// Les identifiants correspondent aux `type` C :
/// 1=Von Koch, 2=Dragon (vectoriels),
/// 3=Mandelbrot, 4=Julia, 5=JuliaSin, 6=Newton, 7=Phoenix,
/// 8=Buffalo, 9=Barnsley Julia, 10=Barnsley Mandelbrot,
/// 11=Magnet Julia, 12=Magnet Mandelbrot, 13=Burning Ship,
/// 14=Tricorn, 15=Mandelbulb, 16=Buddhabrot, 17=Lyapunov,
/// 18=Perpendicular Burning Ship, 19=Celtic, 20=Alpha Mandelbrot,
/// 21=Pickover Stalks, 22=Nova, 23=Multibrot, 24=Nebulabrot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FractalType {
    VonKoch,
    Dragon,
    Mandelbrot,
    Julia,
    JuliaSin,
    Newton,
    Phoenix,
    Buffalo,
    BarnsleyJulia,
    BarnsleyMandelbrot,
    MagnetJulia,
    MagnetMandelbrot,
    BurningShip,
    Tricorn,
    Mandelbulb,
    Buddhabrot,
    Lyapunov,
    PerpendicularBurningShip,
    Celtic,
    AlphaMandelbrot,
    PickoverStalks,
    Nova,
    Multibrot,
    Nebulabrot,
}

impl FractalType {
    /// Convertit un identifiant numérique (comme dans la version C) en enum.
    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            1 => Some(FractalType::VonKoch),
            2 => Some(FractalType::Dragon),
            3 => Some(FractalType::Mandelbrot),
            4 => Some(FractalType::Julia),
            5 => Some(FractalType::JuliaSin),
            6 => Some(FractalType::Newton),
            7 => Some(FractalType::Phoenix),
            8 => Some(FractalType::Buffalo),
            9 => Some(FractalType::BarnsleyJulia),
            10 => Some(FractalType::BarnsleyMandelbrot),
            11 => Some(FractalType::MagnetJulia),
            12 => Some(FractalType::MagnetMandelbrot),
            13 => Some(FractalType::BurningShip),
            14 => Some(FractalType::Tricorn),
            15 => Some(FractalType::Mandelbulb),
            16 => Some(FractalType::Buddhabrot),
            17 => Some(FractalType::Lyapunov),
            18 => Some(FractalType::PerpendicularBurningShip),
            19 => Some(FractalType::Celtic),
            20 => Some(FractalType::AlphaMandelbrot),
            21 => Some(FractalType::PickoverStalks),
            22 => Some(FractalType::Nova),
            23 => Some(FractalType::Multibrot),
            24 => Some(FractalType::Nebulabrot),
            _ => None,
        }
    }

    /// Identifiant numérique C correspondant.
    #[allow(dead_code)]
    pub fn id(self) -> u8 {
        match self {
            FractalType::VonKoch => 1,
            FractalType::Dragon => 2,
            FractalType::Mandelbrot => 3,
            FractalType::Julia => 4,
            FractalType::JuliaSin => 5,
            FractalType::Newton => 6,
            FractalType::Phoenix => 7,
            FractalType::Buffalo => 8,
            FractalType::BarnsleyJulia => 9,
            FractalType::BarnsleyMandelbrot => 10,
            FractalType::MagnetJulia => 11,
            FractalType::MagnetMandelbrot => 12,
            FractalType::BurningShip => 13,
            FractalType::Tricorn => 14,
            FractalType::Mandelbulb => 15,
            FractalType::Buddhabrot => 16,
            FractalType::Lyapunov => 17,
            FractalType::PerpendicularBurningShip => 18,
            FractalType::Celtic => 19,
            FractalType::AlphaMandelbrot => 20,
            FractalType::PickoverStalks => 21,
            FractalType::Nova => 22,
            FractalType::Multibrot => 23,
            FractalType::Nebulabrot => 24,
        }
    }

    #[allow(dead_code)]
    pub fn name(self) -> &'static str {
        match self {
            FractalType::VonKoch => "Von Koch",
            FractalType::Dragon => "Dragon",
            FractalType::Mandelbrot => "Mandelbrot",
            FractalType::Julia => "Julia",
            FractalType::JuliaSin => "Julia Sin",
            FractalType::Newton => "Newton",
            FractalType::Phoenix => "Phoenix",
            FractalType::Buffalo => "Buffalo",
            FractalType::BarnsleyJulia => "Barnsley Julia",
            FractalType::BarnsleyMandelbrot => "Barnsley Mandelbrot",
            FractalType::MagnetJulia => "Magnet Julia",
            FractalType::MagnetMandelbrot => "Magnet Mandelbrot",
            FractalType::BurningShip => "Burning Ship",
            FractalType::Tricorn => "Tricorn",
            FractalType::Mandelbulb => "Mandelbulb (2D power 8)",
            FractalType::Buddhabrot => "Buddhabrot",
            FractalType::Lyapunov => "Lyapunov",
            FractalType::PerpendicularBurningShip => "Perpendicular Burning Ship",
            FractalType::Celtic => "Celtic",
            FractalType::AlphaMandelbrot => "Alpha Mandelbrot",
            FractalType::PickoverStalks => "Pickover Stalks",
            FractalType::Nova => "Nova",
            FractalType::Multibrot => "Multibrot",
            FractalType::Nebulabrot => "Nebulabrot",
        }
    }
}

/// Mode d'algorithme pour le rendu escape-time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlgorithmMode {
    Auto,
    StandardF64,
    Perturbation,
    ReferenceGmp,
}

impl AlgorithmMode {
    #[allow(dead_code)]
    pub fn name(self) -> &'static str {
        match self {
            AlgorithmMode::Auto => "Auto",
            AlgorithmMode::StandardF64 => "Standard f64",
            AlgorithmMode::Perturbation => "Perturbation",
            AlgorithmMode::ReferenceGmp => "Reference GMP",
        }
    }

    #[allow(dead_code)]
    pub fn from_cli_name(value: &str) -> Option<Self> {
        match value.trim().to_lowercase().as_str() {
            "auto" => Some(AlgorithmMode::Auto),
            "f64" | "standard" | "standardf64" => Some(AlgorithmMode::StandardF64),
            "perturbation" | "perturb" => Some(AlgorithmMode::Perturbation),
            "gmp" | "referencegmp" | "reference-gmp" => Some(AlgorithmMode::ReferenceGmp),
            _ => None,
        }
    }
}

use crate::fractal::lyapunov::LyapunovPreset;

/// Paramètres d'une fractale pour le rendu escape-time.
///
/// Cette structure est une version simplifiée de `struct fractal` en C,
/// adaptée au mode non interactif/CLI.
#[derive(Clone, Debug)]
pub struct FractalParams {
    pub width: u32,
    pub height: u32,

    pub xmin: f64,
    pub xmax: f64,
    pub ymin: f64,
    pub ymax: f64,

    pub seed: Complex64,
    pub iteration_max: u32,
    pub bailout: f64,

    pub fractal_type: FractalType,

    /// Palette (0-8) comme dans la version C.
    pub color_mode: u8,
    /// Nombre de répétitions du gradient (2-40).
    pub color_repeat: u32,

    /// Active le chemin GMP pour la haute précision.
    pub use_gmp: bool,
    /// Précision GMP en bits (ex. 128, 256, 512).
    pub precision_bits: u32,

    /// Mode d'algorithme pour Mandelbrot (auto/f64/perturbation/GMP).
    pub algorithm_mode: AlgorithmMode,
    /// Seuil delta pour activer BLA.
    pub bla_threshold: f64,
    /// Multiplicateur du rayon de validité BLA (1.0 = conservateur, >1 = agressif).
    pub bla_validity_scale: f64,
    /// Tolérance de glitch (Pauldelbrot).
    pub glitch_tolerance: f64,
    /// Ordre de la série (0=off, 1=linéaire, 2=quadratique).
    pub series_order: u8,
    /// Seuil de delta pour activer la série.
    pub series_threshold: f64,
    /// Tolérance d'erreur estimée pour la série.
    pub series_error_tolerance: f64,
    /// Active la passe voisinage pour détecter les glitches.
    pub glitch_neighbor_pass: bool,

    /// Puissance pour Multibrot (z^d + c), défaut 2.5. Utilisé aussi pour le calcul BLA.
    pub multibrot_power: f64,

    /// Preset Lyapunov sélectionné.
    pub lyapunov_preset: LyapunovPreset,
    /// Séquence Lyapunov (true=A, false=B). Si vide, utilise la séquence par défaut.
    pub lyapunov_sequence: Vec<bool>,
}

/// Résultat du calcul d'un point de fractale.
#[derive(Clone, Copy, Debug)]
pub struct FractalResult {
    pub iteration: u32,
    pub z: Complex64,
}

