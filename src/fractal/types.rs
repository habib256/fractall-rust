use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use super::orbit_traps::OrbitTrapType;
use rug;

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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

/// Espace colorimétrique pour les gradients
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum ColorSpace {
    /// Espace RGB standard (défaut)
    #[default]
    Rgb,
    /// Espace HSB/HSV (Teinte, Saturation, Brillance) - transitions plus naturelles
    Hsb,
    /// Espace LCH (Luminosité, Chroma, Teinte) - perceptuellement uniforme
    Lch,
}

impl ColorSpace {
    #[allow(dead_code)]
    pub fn all() -> &'static [ColorSpace] {
        &[ColorSpace::Rgb, ColorSpace::Hsb, ColorSpace::Lch]
    }
    
    #[allow(dead_code)]
    pub fn name(self) -> &'static str {
        match self {
            ColorSpace::Rgb => "RGB",
            ColorSpace::Hsb => "HSB",
            ColorSpace::Lch => "LCH",
        }
    }
}

/// Mode de colorisation pour les pixels extérieurs (XaoS-style outcoloring).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum OutColoringMode {
    /// Discrete iteration bands (no smoothing)
    Iter,
    /// iteration + z.re contribution
    IterPlusReal,
    /// iteration + z.im contribution
    IterPlusImag,
    /// iteration + atan(z.re/z.im) ratio
    IterPlusRealImag,
    /// combination of real, imag, and ratio
    IterPlusAll,
    /// different color based on sign of z.im
    BinaryDecomposition,
    /// boundary detection (|z.re| < 10 or |z.im| < 10)
    Biomorphs,
    /// electrostatic potential: log(|z|) / 2^n
    Potential,
    /// based on angle/phase of z: atan2(z.im, z.re)
    ColorDecomposition,
    /// current default smooth coloring
    #[default]
    Smooth,
    /// Coloration basée sur orbit traps (distance minimale à une forme géométrique)
    OrbitTraps,
    /// Technique Wings utilisant sinh() sur l'orbite pour motifs en ailes
    Wings,
    /// Distance field gradient (smooth distance-based coloring)
    Distance,
    /// Distance with ambient occlusion effect (darker near edges)
    DistanceAO,
    /// 3D shading effect using distance gradient
    Distance3D,
}

impl OutColoringMode {
    /// Returns all available outcoloring modes.
    #[allow(dead_code)]
    pub fn all() -> &'static [OutColoringMode] {
        &[
            OutColoringMode::Iter,
            OutColoringMode::IterPlusReal,
            OutColoringMode::IterPlusImag,
            OutColoringMode::IterPlusRealImag,
            OutColoringMode::IterPlusAll,
            OutColoringMode::BinaryDecomposition,
            OutColoringMode::Biomorphs,
            OutColoringMode::Potential,
            OutColoringMode::ColorDecomposition,
            OutColoringMode::Smooth,
            OutColoringMode::OrbitTraps,
            OutColoringMode::Wings,
            OutColoringMode::Distance,
            OutColoringMode::DistanceAO,
            OutColoringMode::Distance3D,
        ]
    }

    /// Display name for UI.
    #[allow(dead_code)]
    pub fn name(self) -> &'static str {
        match self {
            OutColoringMode::Iter => "Iter",
            OutColoringMode::IterPlusReal => "Iter+Real",
            OutColoringMode::IterPlusImag => "Iter+Imag",
            OutColoringMode::IterPlusRealImag => "Iter+Real/Imag",
            OutColoringMode::IterPlusAll => "Iter+All",
            OutColoringMode::BinaryDecomposition => "Binary Decomp",
            OutColoringMode::Biomorphs => "Biomorphs",
            OutColoringMode::Potential => "Potential",
            OutColoringMode::ColorDecomposition => "Color Decomp",
            OutColoringMode::Smooth => "Smooth",
            OutColoringMode::OrbitTraps => "Orbit Traps",
            OutColoringMode::Wings => "Wings",
            OutColoringMode::Distance => "Distance",
            OutColoringMode::DistanceAO => "Distance AO",
            OutColoringMode::Distance3D => "Distance 3D",
        }
    }

    /// Numeric ID for serialization.
    #[allow(dead_code)]
    pub fn id(self) -> u8 {
        match self {
            OutColoringMode::Iter => 0,
            OutColoringMode::IterPlusReal => 1,
            OutColoringMode::IterPlusImag => 2,
            OutColoringMode::IterPlusRealImag => 3,
            OutColoringMode::IterPlusAll => 4,
            OutColoringMode::BinaryDecomposition => 5,
            OutColoringMode::Biomorphs => 6,
            OutColoringMode::Potential => 7,
            OutColoringMode::ColorDecomposition => 8,
            OutColoringMode::Smooth => 9,
            OutColoringMode::OrbitTraps => 10,
            OutColoringMode::Wings => 11,
            OutColoringMode::Distance => 12,
            OutColoringMode::DistanceAO => 13,
            OutColoringMode::Distance3D => 14,
        }
    }

    /// Create from numeric ID.
    #[allow(dead_code)]
    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(OutColoringMode::Iter),
            1 => Some(OutColoringMode::IterPlusReal),
            2 => Some(OutColoringMode::IterPlusImag),
            3 => Some(OutColoringMode::IterPlusRealImag),
            4 => Some(OutColoringMode::IterPlusAll),
            5 => Some(OutColoringMode::BinaryDecomposition),
            6 => Some(OutColoringMode::Biomorphs),
            7 => Some(OutColoringMode::Potential),
            8 => Some(OutColoringMode::ColorDecomposition),
            9 => Some(OutColoringMode::Smooth),
            10 => Some(OutColoringMode::OrbitTraps),
            11 => Some(OutColoringMode::Wings),
            12 => Some(OutColoringMode::Distance),
            13 => Some(OutColoringMode::DistanceAO),
            14 => Some(OutColoringMode::Distance3D),
            _ => None,
        }
    }

    /// CLI name for command-line argument.
    #[allow(dead_code)]
    pub fn cli_name(self) -> &'static str {
        match self {
            OutColoringMode::Iter => "iter",
            OutColoringMode::IterPlusReal => "iter+real",
            OutColoringMode::IterPlusImag => "iter+imag",
            OutColoringMode::IterPlusRealImag => "iter+real/imag",
            OutColoringMode::IterPlusAll => "iter+all",
            OutColoringMode::BinaryDecomposition => "binary",
            OutColoringMode::Biomorphs => "biomorphs",
            OutColoringMode::Potential => "potential",
            OutColoringMode::ColorDecomposition => "color-decomp",
            OutColoringMode::Smooth => "smooth",
            OutColoringMode::OrbitTraps => "orbit-traps",
            OutColoringMode::Wings => "wings",
            OutColoringMode::Distance => "distance",
            OutColoringMode::DistanceAO => "distance-ao",
            OutColoringMode::Distance3D => "distance-3d",
        }
    }

    /// Parse from CLI name.
    #[allow(dead_code)]
    pub fn from_cli_name(value: &str) -> Option<Self> {
        match value.trim().to_lowercase().as_str() {
            "iter" | "0" => Some(OutColoringMode::Iter),
            "iter+real" | "iterreal" | "1" => Some(OutColoringMode::IterPlusReal),
            "iter+imag" | "iterimag" | "2" => Some(OutColoringMode::IterPlusImag),
            "iter+real/imag" | "iterrealimag" | "3" => Some(OutColoringMode::IterPlusRealImag),
            "iter+all" | "iterall" | "4" => Some(OutColoringMode::IterPlusAll),
            "binary" | "binary-decomp" | "binarydecomp" | "5" => Some(OutColoringMode::BinaryDecomposition),
            "biomorphs" | "biomorph" | "6" => Some(OutColoringMode::Biomorphs),
            "potential" | "7" => Some(OutColoringMode::Potential),
            "color-decomp" | "colordecomp" | "decomp" | "8" => Some(OutColoringMode::ColorDecomposition),
            "smooth" | "9" => Some(OutColoringMode::Smooth),
            "orbit-traps" | "orbittraps" | "10" => Some(OutColoringMode::OrbitTraps),
            "wings" | "11" => Some(OutColoringMode::Wings),
            "distance" | "12" => Some(OutColoringMode::Distance),
            "distance-ao" | "distanceao" | "13" => Some(OutColoringMode::DistanceAO),
            "distance-3d" | "distance3d" | "14" => Some(OutColoringMode::Distance3D),
            _ => None,
        }
    }
}

/// Complex plane transformation mode (XaoS-style).
/// Transforms coordinate c before fractal iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PlaneTransform {
    /// Normal mode: c = c (no transformation)
    #[default]
    Mu,
    /// Inversion: c = 1/c (infinity <-> 0)
    Inversion,
    /// Inversion with shifted center: c = 1/(c + 0.25)
    InversionShifted,
    /// Lambda plane: c = 4*mu*(1-mu)
    Lambda,
    /// Inverse lambda: c = 1/(4*mu*(1-mu))
    InversionLambda,
    /// Inverse lambda minus 1: c = 1/(4*mu*(1-mu)) - 1
    InversionLambdaMinus1,
    /// Special inversion for Mandelbrot details: c = 1/(c - 1.40115)
    InversionSpecial,
}

impl PlaneTransform {
    /// Returns all available plane transforms.
    #[allow(dead_code)]
    pub fn all() -> &'static [PlaneTransform] {
        &[
            PlaneTransform::Mu,
            PlaneTransform::Inversion,
            PlaneTransform::InversionShifted,
            PlaneTransform::Lambda,
            PlaneTransform::InversionLambda,
            PlaneTransform::InversionLambdaMinus1,
            PlaneTransform::InversionSpecial,
        ]
    }

    /// Display name for UI.
    #[allow(dead_code)]
    pub fn name(self) -> &'static str {
        match self {
            PlaneTransform::Mu => "μ (normal)",
            PlaneTransform::Inversion => "1/μ",
            PlaneTransform::InversionShifted => "1/(μ+0.25)",
            PlaneTransform::Lambda => "λ",
            PlaneTransform::InversionLambda => "1/λ",
            PlaneTransform::InversionLambdaMinus1 => "1/λ-1",
            PlaneTransform::InversionSpecial => "1/(μ-1.40115)",
        }
    }

    /// Numeric ID for serialization.
    #[allow(dead_code)]
    pub fn id(self) -> u8 {
        match self {
            PlaneTransform::Mu => 0,
            PlaneTransform::Inversion => 1,
            PlaneTransform::InversionShifted => 2,
            PlaneTransform::Lambda => 3,
            PlaneTransform::InversionLambda => 4,
            PlaneTransform::InversionLambdaMinus1 => 5,
            PlaneTransform::InversionSpecial => 6,
        }
    }

    /// Create from numeric ID.
    #[allow(dead_code)]
    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(PlaneTransform::Mu),
            1 => Some(PlaneTransform::Inversion),
            2 => Some(PlaneTransform::InversionShifted),
            3 => Some(PlaneTransform::Lambda),
            4 => Some(PlaneTransform::InversionLambda),
            5 => Some(PlaneTransform::InversionLambdaMinus1),
            6 => Some(PlaneTransform::InversionSpecial),
            _ => None,
        }
    }

    /// CLI name for command-line argument.
    #[allow(dead_code)]
    pub fn cli_name(self) -> &'static str {
        match self {
            PlaneTransform::Mu => "mu",
            PlaneTransform::Inversion => "1/mu",
            PlaneTransform::InversionShifted => "1/(mu+0.25)",
            PlaneTransform::Lambda => "lambda",
            PlaneTransform::InversionLambda => "1/lambda",
            PlaneTransform::InversionLambdaMinus1 => "1/lambda-1",
            PlaneTransform::InversionSpecial => "1/(mu-1.40115)",
        }
    }

    /// Parse from CLI name.
    #[allow(dead_code)]
    pub fn from_cli_name(value: &str) -> Option<Self> {
        match value.trim().to_lowercase().as_str() {
            "mu" | "0" => Some(PlaneTransform::Mu),
            "1/mu" | "inv" | "inversion" | "1" => Some(PlaneTransform::Inversion),
            "1/(mu+0.25)" | "inv-shifted" | "2" => Some(PlaneTransform::InversionShifted),
            "lambda" | "3" => Some(PlaneTransform::Lambda),
            "1/lambda" | "inv-lambda" | "4" => Some(PlaneTransform::InversionLambda),
            "1/lambda-1" | "inv-lambda-1" | "5" => Some(PlaneTransform::InversionLambdaMinus1),
            "1/(mu-1.40115)" | "inv-special" | "6" => Some(PlaneTransform::InversionSpecial),
            _ => None,
        }
    }

    /// Apply the plane transformation to a complex coordinate in GMP precision.
    /// This version avoids the GMP → f64 → GMP conversion that loses precision at deep zooms.
    pub fn transform_gmp(self, mu: &rug::Complex, prec: u32) -> rug::Complex {
        use rug::Float;
        match self {
            PlaneTransform::Mu => mu.clone(),
            PlaneTransform::Inversion => {
                // c = 1/mu = conj(mu) / |mu|^2
                let mut denom = mu.real().clone();
                denom *= mu.real();
                let mut im_sq = mu.imag().clone();
                im_sq *= mu.imag();
                denom += &im_sq;
                let threshold = Float::with_val(prec, 1e-20);
                if denom < threshold {
                    return rug::Complex::with_val(prec, (Float::with_val(prec, 1e10), Float::with_val(prec, 0.0)));
                }
                let mut result = mu.clone().conj();
                result /= &denom;
                result
            }
            PlaneTransform::InversionShifted => {
                // c = 1/(mu + 0.25)
                let mut shifted = mu.clone();
                shifted += rug::Complex::with_val(prec, (Float::with_val(prec, 0.25), Float::with_val(prec, 0.0)));
                let mut denom = shifted.real().clone();
                denom *= shifted.real();
                let mut im_sq = shifted.imag().clone();
                im_sq *= shifted.imag();
                denom += &im_sq;
                let threshold = Float::with_val(prec, 1e-20);
                if denom < threshold {
                    return rug::Complex::with_val(prec, (Float::with_val(prec, 1e10), Float::with_val(prec, 0.0)));
                }
                let mut result = shifted.conj();
                result /= &denom;
                result
            }
            PlaneTransform::Lambda => {
                // c = 4*mu*(1-mu)
                let one = rug::Complex::with_val(prec, (Float::with_val(prec, 1.0), Float::with_val(prec, 0.0)));
                let mut one_minus_mu = one.clone();
                one_minus_mu -= mu;
                let mut result = mu.clone();
                result *= &one_minus_mu;
                result *= Float::with_val(prec, 4.0);
                result
            }
            PlaneTransform::InversionLambda => {
                // c = 1/(4*mu*(1-mu))
                let one = rug::Complex::with_val(prec, (Float::with_val(prec, 1.0), Float::with_val(prec, 0.0)));
                let mut one_minus_mu = one.clone();
                one_minus_mu -= mu;
                let mut lambda = mu.clone();
                lambda *= &one_minus_mu;
                lambda *= Float::with_val(prec, 4.0);
                let mut denom = lambda.real().clone();
                denom *= lambda.real();
                let mut im_sq = lambda.imag().clone();
                im_sq *= lambda.imag();
                denom += &im_sq;
                let threshold = Float::with_val(prec, 1e-20);
                if denom < threshold {
                    return rug::Complex::with_val(prec, (Float::with_val(prec, 1e10), Float::with_val(prec, 0.0)));
                }
                let mut result = lambda.conj();
                result /= &denom;
                result
            }
            PlaneTransform::InversionLambdaMinus1 => {
                // c = 1/(4*mu*(1-mu)) - 1
                let one = rug::Complex::with_val(prec, (Float::with_val(prec, 1.0), Float::with_val(prec, 0.0)));
                let mut one_minus_mu = one.clone();
                one_minus_mu -= mu;
                let mut lambda = mu.clone();
                lambda *= &one_minus_mu;
                lambda *= Float::with_val(prec, 4.0);
                let mut denom = lambda.real().clone();
                denom *= lambda.real();
                let mut im_sq = lambda.imag().clone();
                im_sq *= lambda.imag();
                denom += &im_sq;
                let threshold = Float::with_val(prec, 1e-20);
                if denom < threshold {
                    return rug::Complex::with_val(prec, (Float::with_val(prec, 1e10), Float::with_val(prec, 0.0)));
                }
                let mut result = lambda.conj();
                result /= &denom;
                result -= &one;
                result
            }
            PlaneTransform::InversionSpecial => {
                // c = 1/(mu - 1.40115)
                let shift_val = Float::with_val(prec, 1.40115);
                let mut shifted = mu.clone();
                shifted -= rug::Complex::with_val(prec, (shift_val.clone(), Float::with_val(prec, 0.0)));
                let mut denom = shifted.real().clone();
                denom *= shifted.real();
                let mut im_sq = shifted.imag().clone();
                im_sq *= shifted.imag();
                denom += &im_sq;
                let threshold = Float::with_val(prec, 1e-20);
                if denom < threshold {
                    return rug::Complex::with_val(prec, (Float::with_val(prec, 1e10), Float::with_val(prec, 0.0)));
                }
                let mut result = shifted.conj();
                result /= &denom;
                result
            }
        }
    }

    /// Apply the plane transformation to a complex coordinate.
    #[inline]
    pub fn transform(self, mu: Complex64) -> Complex64 {
        match self {
            PlaneTransform::Mu => mu,
            PlaneTransform::Inversion => {
                // c = 1/mu = conj(mu) / |mu|^2
                let denom = mu.norm_sqr();
                if denom < 1e-20 {
                    return Complex64::new(1e10, 0.0);
                }
                mu.conj() / denom
            }
            PlaneTransform::InversionShifted => {
                // c = 1/(mu + 0.25)
                let shifted = mu + Complex64::new(0.25, 0.0);
                let denom = shifted.norm_sqr();
                if denom < 1e-20 {
                    return Complex64::new(1e10, 0.0);
                }
                shifted.conj() / denom
            }
            PlaneTransform::Lambda => {
                // c = 4*mu*(1-mu)
                let one = Complex64::new(1.0, 0.0);
                mu * (one - mu) * 4.0
            }
            PlaneTransform::InversionLambda => {
                // c = 1/(4*mu*(1-mu))
                let one = Complex64::new(1.0, 0.0);
                let lambda = mu * (one - mu) * 4.0;
                let denom = lambda.norm_sqr();
                if denom < 1e-20 {
                    return Complex64::new(1e10, 0.0);
                }
                lambda.conj() / denom
            }
            PlaneTransform::InversionLambdaMinus1 => {
                // c = 1/(4*mu*(1-mu)) - 1
                let one = Complex64::new(1.0, 0.0);
                let lambda = mu * (one - mu) * 4.0;
                let denom = lambda.norm_sqr();
                if denom < 1e-20 {
                    return Complex64::new(1e10, 0.0);
                }
                lambda.conj() / denom - one
            }
            PlaneTransform::InversionSpecial => {
                // c = 1/(mu - 1.40115)
                let shifted = mu - Complex64::new(1.40115, 0.0);
                let denom = shifted.norm_sqr();
                if denom < 1e-20 {
                    return Complex64::new(1e10, 0.0);
                }
                shifted.conj() / denom
            }
        }
    }
}

use crate::fractal::lyapunov::LyapunovPreset;

/// Paramètres d'une fractale pour le rendu escape-time.
///
/// Cette structure est une version simplifiée de `struct fractal` en C,
/// adaptée au mode non interactif/CLI.
///
/// Les coordonnées du plan complexe sont représentées par centre + étendue
/// (center_x/center_y + span_x/span_y) plutôt que par bornes (xmin/xmax/ymin/ymax).
/// Cela permet des zooms profonds (> 1e15) sans perte de précision f64.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FractalParams {
    pub width: u32,
    pub height: u32,

    /// Centre X du plan complexe.
    pub center_x: f64,
    /// Centre Y du plan complexe.
    pub center_y: f64,
    /// Étendue (largeur) du plan complexe.
    pub span_x: f64,
    /// Étendue (hauteur) du plan complexe.
    pub span_y: f64,
    
    /// Coordonnées haute précision (String) pour préserver la précision arbitraire.
    /// Utilisées pour les calculs GMP aux zooms profonds (>10^15).
    /// Si None, les valeurs f64 sont utilisées (compatibilité GPU/CPU standard).
    pub center_x_hp: Option<String>,
    pub center_y_hp: Option<String>,
    pub span_x_hp: Option<String>,
    pub span_y_hp: Option<String>,

    pub seed: Complex64,
    pub iteration_max: u32,
    pub bailout: f64,

    pub fractal_type: FractalType,

    /// Palette (0-8) comme dans la version C.
    pub color_mode: u8,
    /// Nombre de répétitions du gradient (2-40).
    pub color_repeat: u32,
    /// Espace colorimétrique pour les gradients (RGB, HSB, LCH)
    pub color_space: ColorSpace,

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

    /// Active l'approximation par série standalone (sans BLA).
    /// Permet de sauter des itérations initiales en utilisant une série de Taylor.
    pub series_standalone: bool,

    /// Nombre maximum de références secondaires pour la correction de glitchs.
    /// 0 = désactivé, 3 = valeur recommandée pour un bon compromis performance/qualité.
    pub max_secondary_refs: u8,

    /// Taille minimale d'un cluster de glitchs pour justifier une référence secondaire.
    /// Les petits clusters sont recalculés en GMP directement.
    pub min_glitch_cluster_size: u32,

    /// Puissance pour Multibrot (z^d + c), défaut 2.5. Utilisé aussi pour le calcul BLA.
    pub multibrot_power: f64,

    /// Nombre maximum d'itérations de perturbation par pixel (aligné C++ Fraktaler-3: PerturbIterations).
    /// 0 = illimité (comportement historique). Défaut 1024.
    pub max_perturb_iterations: u32,
    /// Nombre maximum de pas BLA par pixel (aligné C++ Fraktaler-3: BLASteps).
    /// 0 = illimité. Défaut 1024.
    pub max_bla_steps: u32,
    /// Utiliser la formule de précision de la référence C++ (prec = max(24, 24 + exp(zoom*height))).
    /// Si true (défaut), utilise la formule C++ Fraktaler-3. Si false, utilise une politique plus conservative (log2(zoom) + marge par palier).
    pub use_reference_precision_formula: bool,

    /// Preset Lyapunov sélectionné.
    pub lyapunov_preset: LyapunovPreset,
    /// Séquence Lyapunov (true=A, false=B). Si vide, utilise la séquence par défaut.
    pub lyapunov_sequence: Vec<bool>,

    /// Active le calcul de distance estimation (nécessite DualComplex, ajoute overhead)
    pub enable_distance_estimation: bool,
    /// Active la détection d'intérieur (nécessite ExtendedDualComplex, ajoute overhead)
    pub enable_interior_detection: bool,
    /// Seuil pour détection d'intérieur (défaut 0.001)
    pub interior_threshold: f64,

    /// Mode de colorisation pour les pixels extérieurs (XaoS-style).
    pub out_coloring_mode: OutColoringMode,

    /// Complex plane transformation (XaoS-style).
    pub plane_transform: PlaneTransform,
    
    /// Active le calcul d'orbit traps (nécessite stockage de l'orbite complète)
    pub enable_orbit_traps: bool,
    /// Type d'orbit trap à utiliser
    pub orbit_trap_type: OrbitTrapType,
}

impl FractalParams {
    /// Borne minimale X (calculée à la demande).
    /// Conservée pour compatibilité, mais le code utilise maintenant center+span directement.
    #[inline]
    #[allow(dead_code)]
    pub fn xmin(&self) -> f64 {
        self.center_x - self.span_x * 0.5
    }

    /// Borne maximale X (calculée à la demande).
    /// Conservée pour compatibilité, mais le code utilise maintenant center+span directement.
    #[inline]
    #[allow(dead_code)]
    pub fn xmax(&self) -> f64 {
        self.center_x + self.span_x * 0.5
    }

    /// Borne minimale Y (calculée à la demande).
    /// Conservée pour compatibilité, mais le code utilise maintenant center+span directement.
    #[inline]
    #[allow(dead_code)]
    pub fn ymin(&self) -> f64 {
        self.center_y - self.span_y * 0.5
    }

    /// Borne maximale Y (calculée à la demande).
    /// Conservée pour compatibilité, mais le code utilise maintenant center+span directement.
    #[inline]
    #[allow(dead_code)]
    pub fn ymax(&self) -> f64 {
        self.center_y + self.span_y * 0.5
    }

    /// Étendue X (identique à span_x, pour compatibilité).
    #[inline]
    #[allow(dead_code)]
    pub fn x_range(&self) -> f64 {
        self.span_x
    }

    /// Étendue Y (identique à span_y, pour compatibilité).
    #[inline]
    #[allow(dead_code)]
    pub fn y_range(&self) -> f64 {
        self.span_y
    }

    /// Définit les bornes à partir de xmin/xmax/ymin/ymax (pour compatibilité CLI).
    #[allow(dead_code)]
    pub fn set_bounds(&mut self, xmin: f64, xmax: f64, ymin: f64, ymax: f64) {
        self.center_x = (xmin + xmax) * 0.5;
        self.center_y = (ymin + ymax) * 0.5;
        self.span_x = xmax - xmin;
        self.span_y = ymax - ymin;
    }
}

/// Résultat du calcul d'un point de fractale.
#[derive(Clone, Debug)]
pub struct FractalResult {
    pub iteration: u32,
    pub z: Complex64,
    /// Données d'orbite pour orbit traps (None si orbit traps désactivés)
    #[allow(dead_code)]
    pub orbit: Option<crate::fractal::orbit_traps::OrbitData>,
}

