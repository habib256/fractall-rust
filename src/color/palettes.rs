use num_complex::Complex64;
use crate::fractal::OutColoringMode;

/// Identifiants de palettes, alignés sur `colorization.h`.
#[derive(Clone, Copy, Debug)]
pub enum PaletteId {
    Fire = 0,
    Ocean = 1,
    Forest = 2,
    Violet = 3,
    Rainbow = 4,
    Sunset = 5,
    Plasma = 6,
    Ice = 7,
    Cosmic = 8,
}

impl PaletteId {
    pub fn from_u8(id: u8) -> Self {
        match id {
            0 => PaletteId::Fire,
            1 => PaletteId::Ocean,
            2 => PaletteId::Forest,
            3 => PaletteId::Violet,
            4 => PaletteId::Rainbow,
            5 => PaletteId::Sunset,
            6 => PaletteId::Plasma,
            7 => PaletteId::Ice,
            8 => PaletteId::Cosmic,
            _ => PaletteId::Plasma,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct GradientStop {
    position: f64, // [0.0, 1.0]
    r: u8,
    g: u8,
    b: u8,
}

#[derive(Clone, Copy, Debug)]
struct Gradient {
    _name: &'static str,
    stops: &'static [GradientStop],
}

// Palettes traduites depuis colorization.c

const FIRE_STOPS: [GradientStop; 4] = [
    GradientStop { position: 0.00, r: 0, g: 0, b: 0 },       // Black
    GradientStop { position: 0.33, r: 255, g: 0, b: 0 },     // Red
    GradientStop { position: 0.66, r: 255, g: 255, b: 0 },   // Yellow
    GradientStop { position: 1.00, r: 255, g: 255, b: 255 }, // White
];

const OCEAN_STOPS: [GradientStop; 4] = [
    GradientStop { position: 0.00, r: 0, g: 0, b: 0 },       // Black
    GradientStop { position: 0.33, r: 0, g: 0, b: 255 },     // Blue
    GradientStop { position: 0.66, r: 0, g: 255, b: 255 },   // Cyan
    GradientStop { position: 1.00, r: 255, g: 255, b: 255 }, // White
];

const FOREST_STOPS: [GradientStop; 4] = [
    GradientStop { position: 0.00, r: 0, g: 0, b: 0 },       // Black
    GradientStop { position: 0.33, r: 0, g: 180, b: 0 },     // Dark Green
    GradientStop { position: 0.66, r: 200, g: 255, b: 0 },   // Yellow/Light Green
    GradientStop { position: 1.00, r: 255, g: 255, b: 255 }, // White
];

const VIOLET_STOPS: [GradientStop; 4] = [
    GradientStop { position: 0.00, r: 0, g: 0, b: 0 },       // Black
    GradientStop { position: 0.33, r: 128, g: 0, b: 200 },   // Dark Violet
    GradientStop { position: 0.66, r: 255, g: 100, b: 255 }, // Pink/Magenta
    GradientStop { position: 1.00, r: 255, g: 255, b: 255 }, // White
];

const RAINBOW_STOPS: [GradientStop; 7] = [
    GradientStop { position: 0.000, r: 255, g: 0, b: 0 },     // Red
    GradientStop { position: 0.166, r: 255, g: 165, b: 0 },   // Orange
    GradientStop { position: 0.333, r: 255, g: 255, b: 0 },   // Yellow
    GradientStop { position: 0.500, r: 0, g: 255, b: 0 },     // Green
    GradientStop { position: 0.666, r: 0, g: 255, b: 255 },   // Cyan
    GradientStop { position: 0.833, r: 0, g: 0, b: 255 },     // Blue
    GradientStop { position: 1.000, r: 180, g: 0, b: 255 },   // Violet
];

const SUNSET_STOPS: [GradientStop; 5] = [
    GradientStop { position: 0.00, r: 0, g: 0, b: 0 },        // Black
    GradientStop { position: 0.25, r: 255, g: 140, b: 0 },    // Orange
    GradientStop { position: 0.50, r: 255, g: 0, b: 0 },      // Red
    GradientStop { position: 0.75, r: 255, g: 0, b: 200 },    // Violet
    GradientStop { position: 1.00, r: 55, g: 0, b: 255 },     // Dark Blue
];

const PLASMA_STOPS: [GradientStop; 4] = [
    GradientStop { position: 0.00, r: 13, g: 8, b: 135 },     // Deep Blue
    GradientStop { position: 0.33, r: 126, g: 3, b: 168 },    // Violet
    GradientStop { position: 0.66, r: 240, g: 87, b: 100 },   // Pink/Coral
    GradientStop { position: 1.00, r: 240, g: 230, b: 50 },   // Yellow/Orange
];

const ICE_STOPS: [GradientStop; 4] = [
    GradientStop { position: 0.00, r: 255, g: 255, b: 255 },  // White
    GradientStop { position: 0.33, r: 150, g: 230, b: 255 },  // Light Cyan
    GradientStop { position: 0.66, r: 30, g: 90, b: 200 },    // Deep Blue
    GradientStop { position: 1.00, r: 5, g: 10, b: 30 },      // Near Black
];

const COSMIC_STOPS: [GradientStop; 9] = [
    GradientStop { position: 0.000, r: 0, g: 0, b: 0 },       // Deep Black
    GradientStop { position: 0.143, r: 0, g: 0, b: 51 },      // Blue Night
    GradientStop { position: 0.286, r: 0, g: 77, b: 64 },     // Dark Teal
    GradientStop { position: 0.429, r: 64, g: 224, b: 208 },  // Turquoise
    GradientStop { position: 0.571, r: 200, g: 220, b: 240 }, // Light Gray-Blue
    GradientStop { position: 0.714, r: 255, g: 255, b: 224 }, // Very Pale Yellow
    GradientStop { position: 0.857, r: 255, g: 215, b: 0 },   // Golden Yellow
    GradientStop { position: 0.929, r: 255, g: 165, b: 0 },   // Orange
    GradientStop { position: 1.000, r: 139, g: 0, b: 0 },     // Dark Red
];

const FIRE: Gradient = Gradient { _name: "SmoothFire", stops: &FIRE_STOPS };
const OCEAN: Gradient = Gradient { _name: "SmoothOcean", stops: &OCEAN_STOPS };
const FOREST: Gradient = Gradient { _name: "SmoothForest", stops: &FOREST_STOPS };
const VIOLET: Gradient = Gradient { _name: "SmoothViolet", stops: &VIOLET_STOPS };
const RAINBOW: Gradient = Gradient { _name: "SmoothRainbow", stops: &RAINBOW_STOPS };
const SUNSET: Gradient = Gradient { _name: "SmoothSunset", stops: &SUNSET_STOPS };
const PLASMA: Gradient = Gradient { _name: "SmoothPlasma", stops: &PLASMA_STOPS };
const ICE: Gradient = Gradient { _name: "SmoothIce", stops: &ICE_STOPS };
const COSMIC: Gradient = Gradient { _name: "SmoothCosmic", stops: &COSMIC_STOPS };

fn palette_for(id: PaletteId) -> Gradient {
    match id {
        PaletteId::Fire => FIRE,
        PaletteId::Ocean => OCEAN,
        PaletteId::Forest => FOREST,
        PaletteId::Violet => VIOLET,
        PaletteId::Rainbow => RAINBOW,
        PaletteId::Sunset => SUNSET,
        PaletteId::Plasma => PLASMA,
        PaletteId::Ice => ICE,
        PaletteId::Cosmic => COSMIC,
    }
}

fn gradient_interpolate(g: Gradient, mut t: f64) -> (u8, u8, u8) {
    let stops = g.stops;

    // Clamp t
    if t < 0.0 {
        t = 0.0;
    }
    if t > 1.0 {
        t = 1.0;
    }

    let eps = 1e-9;

    if t <= stops[0].position as f64 + eps {
        let s = stops[0];
        return (s.r, s.g, s.b);
    }
    let last = stops[stops.len() - 1];
    if t >= last.position as f64 - eps {
        return (last.r, last.g, last.b);
    }

    // Trouver le segment contenant t
    for w in stops.windows(2) {
        let a = w[0];
        let b = w[1];
        if t >= a.position as f64 - eps && t < b.position as f64 + eps {
            let denom = (b.position - a.position) as f64;
            let factor = if denom.abs() < std::f64::EPSILON {
                0.0
            } else {
                (t - a.position as f64) / denom
            };
            let lerp = |u: u8, v: u8| -> u8 {
                let u = u as f64;
                let v = v as f64;
                let val = u + factor * (v - u);
                val.clamp(0.0, 255.0) as u8
            };
            return (lerp(a.r, b.r), lerp(a.g, b.g), lerp(a.b, b.b));
        }
    }

    // Fallback
    (last.r, last.g, last.b)
}

fn smooth_iteration(iteration: u32, z: Complex64, iter_max: u32, bailout: f64) -> f64 {
    let iter = iteration as f64;
    let max = iter_max as f64;

    // Vérifier que z est fini avant de calculer sa norme
    // Si z est NaN ou infini, utiliser une valeur basée sur l'itération avec une petite variation
    // pour éviter que tous les pixels avec z invalide aient la même couleur (créant des blocs)
    if !z.re.is_finite() || !z.im.is_finite() {
        // Utiliser une valeur qui varie légèrement pour éviter les blocs identiques
        // Variation basée sur la partie fractionnaire de l'itération pour créer une transition
        let base = iter / max;
        let variation = ((iter as u32 % 4) as f64) * 0.01 / max; // Petite variation cyclique
        return (base + variation).min(0.999);
    }

    let mag = z.norm();

    // Vérifier que la norme est finie et valide
    if !mag.is_finite() || mag <= 0.0 {
        // Utiliser une valeur qui varie légèrement pour éviter les blocs identiques
        let base = iter / max;
        let variation = ((iter as u32 % 4) as f64) * 0.01 / max;
        return (base + variation).min(0.999);
    }

    // Points dans l'ensemble : utiliser iteration/max
    if iteration >= iter_max {
        return iter / max;
    }

    // Pour un rendu smooth, calculer nu basé sur la magnitude de z
    // Formule standard: nu = n + 1 - log₂(log₂(|z|))
    // Équivalent: nu = n + 1 - log(log(|z|) / log(2)) / log(2)
    // Cela donne une estimation continue du nombre d'itérations avant l'échappement
    
    // La magnitude doit être >= bailout pour que le calcul soit valide
    // Si mag < bailout, utiliser une interpolation pour éviter les discontinuités
    let log2 = 2.0_f64.ln();
    let bailout_val = bailout.max(2.0); // Utiliser au minimum 2.0 pour la formule
    
    if mag < bailout_val {
        // Pour mag < bailout, utiliser une interpolation linéaire vers la valeur à mag=bailout
        // pour éviter les discontinuités brutales avec les pixels voisins
        let t = (mag / bailout_val).min(1.0).max(0.0);
        let base_value = iter / max;
        // Calculer la valeur smooth à mag=bailout
        let log_mag_at_bailout = bailout_val.ln();
        let ratio_at_bailout = log_mag_at_bailout / log2;
        if ratio_at_bailout > 1.0 {
            let nu_at_bailout = ratio_at_bailout.ln() / log2;
            if nu_at_bailout.is_finite() && nu_at_bailout >= 0.0 && nu_at_bailout <= iter + 1.0 {
                let smooth_at_bailout = (iter + 1.0 - nu_at_bailout) / max;
                return base_value + t * (smooth_at_bailout - base_value);
            }
        }
        return base_value;
    }
    
    // Formule correcte: nu = log(log(|z|) / log(2)) / log(2)
    // où log(|z|) est le logarithme naturel de la magnitude
    let log_mag = mag.ln();
    
    if !log_mag.is_finite() || log_mag <= 0.0 {
        // Utiliser une valeur qui varie légèrement pour éviter les blocs identiques
        let base = iter / max;
        let variation = ((iter as u32 % 4) as f64) * 0.01 / max;
        return (base + variation).min(0.999);
    }

    // Calculer ratio = log(|z|) / log(2)
    let ratio = log_mag / log2;
    
    if ratio <= 1.0 || !ratio.is_finite() {
        // Si ratio <= 1, le calcul de nu ne serait pas valide
        // Utiliser une interpolation pour éviter les discontinuités
        if ratio > 0.0 && ratio.is_finite() {
            let t = ratio.min(1.0).max(0.0);
            let base_value = iter / max;
            // Petite variation pour éviter les blocs identiques
            return base_value + t * 0.05;
        }
        let base = iter / max;
        let variation = ((iter as u32 % 4) as f64) * 0.01 / max;
        return (base + variation).min(0.999);
    }
    
    // Calculer nu = log(ratio) / log(2) = log(log(|z|) / log(2)) / log(2)
    let nu = ratio.ln() / log2;

    if !nu.is_finite() {
        // Utiliser une valeur qui varie légèrement pour éviter les blocs identiques
        let base = iter / max;
        let variation = ((iter as u32 % 4) as f64) * 0.01 / max;
        return (base + variation).min(0.999);
    }
    
    // Clamp nu dans une plage raisonnable pour éviter les valeurs aberrantes
    let nu_clamped = nu.max(0.0).min(iter + 1.0);

    // Calcul smooth: iter + 1 - nu
    // Cela donne une valeur continue basée sur la magnitude de z
    let mut smooth = iter + 1.0 - nu_clamped;
    
    // Clamp pour éviter les valeurs hors limites
    if smooth < 0.0 {
        smooth = 0.0;
    }
    if smooth > max {
        smooth = max;
    }
    smooth / max
}

/// Compute discrete iteration value (no smoothing).
fn compute_iter_discrete(iteration: u32, _z: Complex64, iter_max: u32) -> f64 {
    iteration as f64 / iter_max as f64
}

/// Compute iteration + z.re contribution.
fn compute_iter_plus_real(iteration: u32, z: Complex64, iter_max: u32) -> f64 {
    let base = iteration as f64 / iter_max as f64;
    // Normalize z.re contribution (typically |z.re| < bailout^0.5)
    let real_contrib = (z.re.abs() / 10.0).min(1.0) * 0.2;
    (base + real_contrib).min(1.0)
}

/// Compute iteration + z.im contribution.
fn compute_iter_plus_imag(iteration: u32, z: Complex64, iter_max: u32) -> f64 {
    let base = iteration as f64 / iter_max as f64;
    // Normalize z.im contribution
    let imag_contrib = (z.im.abs() / 10.0).min(1.0) * 0.2;
    (base + imag_contrib).min(1.0)
}

/// Compute iteration + atan(z.re/z.im) ratio.
fn compute_iter_plus_real_imag(iteration: u32, z: Complex64, iter_max: u32) -> f64 {
    let base = iteration as f64 / iter_max as f64;
    // Compute ratio using atan2 to avoid division by zero
    let angle = z.re.atan2(z.im);
    let ratio_contrib = ((angle + std::f64::consts::PI) / (2.0 * std::f64::consts::PI)) * 0.2;
    if ratio_contrib.is_finite() {
        (base + ratio_contrib).min(1.0)
    } else {
        base
    }
}

/// Compute combination of real, imag, and ratio.
fn compute_iter_plus_all(iteration: u32, z: Complex64, iter_max: u32) -> f64 {
    let base = iteration as f64 / iter_max as f64;
    let real_contrib = (z.re.abs() / 10.0).min(1.0) * 0.1;
    let imag_contrib = (z.im.abs() / 10.0).min(1.0) * 0.1;
    let angle = z.re.atan2(z.im);
    let ratio_contrib = ((angle + std::f64::consts::PI) / (2.0 * std::f64::consts::PI)) * 0.1;
    let total = if ratio_contrib.is_finite() {
        base + real_contrib + imag_contrib + ratio_contrib
    } else {
        base + real_contrib + imag_contrib
    };
    total.min(1.0)
}

/// Compute biomorphs boundary detection value.
fn compute_biomorphs(iteration: u32, z: Complex64, iter_max: u32) -> f64 {
    // Biomorphs: different behavior at boundary
    if z.re.abs() < 10.0 || z.im.abs() < 10.0 {
        // At boundary: use discrete iteration
        iteration as f64 / iter_max as f64
    } else {
        // Outside boundary: use smooth
        smooth_iteration(iteration, z, iter_max, 2.0) // Default bailout
    }
}

/// Compute electrostatic potential.
fn compute_potential(iteration: u32, z: Complex64, iter_max: u32) -> f64 {
    let mag = z.norm();
    if mag <= 1.0 || !mag.is_finite() {
        return iteration as f64 / iter_max as f64;
    }

    // Potential: log(|z|) / 2^iteration
    let log_mag = mag.ln();
    let power = 2.0_f64.powi(iteration as i32);

    if power.is_infinite() || power == 0.0 {
        return iteration as f64 / iter_max as f64;
    }

    let potential = log_mag / power;

    if !potential.is_finite() {
        return iteration as f64 / iter_max as f64;
    }

    // Normalize potential to [0, 1]
    // Typical range is very small, so we scale it up
    let normalized = (potential.abs() * 1000.0).min(1.0);
    normalized
}

/// Compute color decomposition based on angle.
fn compute_color_decomposition(iteration: u32, z: Complex64, iter_max: u32) -> f64 {
    let base = smooth_iteration(iteration, z, iter_max, 2.0); // Default bailout
    // Use angle of z for color variation
    let angle = z.im.atan2(z.re);
    let normalized_angle = (angle + std::f64::consts::PI) / (2.0 * std::f64::consts::PI);

    if !normalized_angle.is_finite() {
        return base;
    }

    // Mix base with angle: 70% smooth, 30% angle
    (base * 0.7 + normalized_angle * 0.3).min(1.0)
}

/// Calcule la couleur RGB pour un pixel à partir de:
/// - son nombre d'itérations,
/// - la valeur finale de z,
/// - les paramètres de colorisation (palette, répétition, mode outcoloring).
///
/// Supporte 10 modes de colorisation inspirés de XaoS.
///
/// Supporte interior detection encodé dans z:
/// - Si z.im < 0: point intérieur (interior detection flag encoded in sign)
///   Les points intérieurs sont coloriés en noir (sauf pour BinaryDecomposition).
pub fn color_for_pixel(
    iteration: u32,
    z: Complex64,
    iter_max: u32,
    palette_index: u8,
    color_repeat: u32,
    out_coloring_mode: OutColoringMode,
) -> (u8, u8, u8) {
    // Points dans l'ensemble : noir
    if iteration >= iter_max {
        return (0, 0, 0);
    }

    // For BinaryDecomposition, we need the original z.im sign
    // Note: Interior detection via z.im < 0 is only valid when enable_interior_detection is true
    // and the flag is explicitly encoded. For normal escape-time fractals, z.im can be negative
    // naturally, so we should NOT use this as a signal unless interior detection is enabled.
    // Since color_for_pixel doesn't have access to params.enable_interior_detection,
    // we disable this check to avoid false positives that create black checkerboard patterns.
    let original_z_im_negative = z.im < 0.0;
    
    // DISABLED: Interior detection check removed to prevent false positives
    // The interior detection flag should only be used when explicitly enabled via
    // enable_interior_detection in FractalParams, and should be passed separately
    // rather than encoded in z.im sign to avoid conflicts with normal negative z.im values.
    // if is_interior && out_coloring_mode != OutColoringMode::BinaryDecomposition {
    //     return (0, 0, 0);
    // }

    // Restore z with positive im for calculations (except for sign check)
    let z_positive = Complex64::new(z.re, z.im.abs());

    // Compute base value based on outcoloring mode
    let mut t = match out_coloring_mode {
        OutColoringMode::Iter => compute_iter_discrete(iteration, z_positive, iter_max),
        OutColoringMode::IterPlusReal => compute_iter_plus_real(iteration, z_positive, iter_max),
        OutColoringMode::IterPlusImag => compute_iter_plus_imag(iteration, z_positive, iter_max),
        OutColoringMode::IterPlusRealImag => compute_iter_plus_real_imag(iteration, z_positive, iter_max),
        OutColoringMode::IterPlusAll => compute_iter_plus_all(iteration, z_positive, iter_max),
        OutColoringMode::BinaryDecomposition => smooth_iteration(iteration, z_positive, iter_max, 2.0), // Default bailout
        OutColoringMode::Biomorphs => compute_biomorphs(iteration, z_positive, iter_max),
        OutColoringMode::Potential => compute_potential(iteration, z_positive, iter_max),
        OutColoringMode::ColorDecomposition => compute_color_decomposition(iteration, z_positive, iter_max),
        OutColoringMode::Smooth => smooth_iteration(iteration, z_positive, iter_max, 2.0), // Default bailout
    };

    // Clamp t dans [0, 1) pour éviter les problèmes aux limites
    if t < 0.0 {
        t = 0.0;
    }
    if t >= 1.0 {
        t = 0.999_999;
    }

    let repeat_count = color_repeat.max(1) as f64;
    let cycle = (t * repeat_count).floor();
    let mut t_repeat = (t * repeat_count) % 1.0;

    // Éviter les valeurs exactement à 0 ou 1 qui créent des discontinuités
    if t_repeat < 0.000_001 && t > 0.0 {
        t_repeat = 0.000_001;
    }
    if t_repeat >= 0.999_999 {
        t_repeat = 0.999_999;
    }

    // Alternance endroit/envers pour créer des cycles de couleur
    if (cycle as i64) % 2 == 1 {
        t_repeat = 1.0 - t_repeat;
    }

    let palette = palette_for(PaletteId::from_u8(palette_index));
    let (r, g, b) = gradient_interpolate(palette, t_repeat);

    // Binary decomposition: invert color when original z.im < 0
    if out_coloring_mode == OutColoringMode::BinaryDecomposition && original_z_im_negative {
        // Invert the color
        return (255 - r, 255 - g, 255 - b);
    }

    (r, g, b)
}

/// Décode la couleur RGB pour Nebulabrot (encodée dans iterations et zs).
///
/// Le format d'encodage est:
/// - iterations: (R << 16) | (G << 8)
/// - z.re: B normalisé [0, 1]
pub fn color_for_nebulabrot_pixel(iteration: u32, z: Complex64) -> (u8, u8, u8) {
    let r = ((iteration >> 16) & 0xFF) as u8;
    let g = ((iteration >> 8) & 0xFF) as u8;
    let b = (z.re.clamp(0.0, 1.0) * 255.0) as u8;
    (r, g, b)
}

/// Colorisation pour Buddhabrot (densité).
///
/// Le format d'encodage est:
/// - z.re: valeur normalisée [0, 2] de la densité
pub fn color_for_buddhabrot_pixel(
    z: Complex64,
    palette_index: u8,
    _color_repeat: u32,
) -> (u8, u8, u8) {
    // z.re contient la densité normalisée * 2.0
    let normalized = (z.re / 2.0).clamp(0.0, 1.0);

    // Densité nulle = noir
    if normalized < 0.001 {
        return (0, 0, 0);
    }

    // Appliquer le gradient directement (pas de répétition pour Buddhabrot)
    let palette = palette_for(PaletteId::from_u8(palette_index));
    gradient_interpolate(palette, normalized)
}

