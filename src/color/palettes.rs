use num_complex::Complex64;

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

fn smooth_iteration(iteration: u32, z: Complex64, iter_max: u32) -> f64 {
    let iter = iteration as f64;
    let max = iter_max as f64;

    // Vérifier que z est fini avant de calculer sa norme
    // Si z est NaN ou infini, retourner une valeur par défaut
    if !z.re.is_finite() || !z.im.is_finite() {
        return iter / max;
    }

    let mag = z.norm();

    // Vérifier que la norme est finie et valide
    if !mag.is_finite() || mag <= 0.0 {
        return iter / max;
    }

    // Même logique que Colorization_SmoothIteration (sans cas Lyapunov)
    if iteration >= iter_max || mag < 7.4 {
        return iter / max;
    }

    let log_zn = mag.ln() / 2.0;
    if !log_zn.is_finite() {
        return iter / max;
    }

    let nu = (log_zn / 2.0_f64.ln()).ln() / 2.0_f64.ln();

    if !nu.is_finite() {
        return iter / max;
    }

    let mut smooth = iter + 1.0 - nu;
    if smooth < 0.0 {
        smooth = 0.0;
    }
    if smooth > max {
        smooth = max;
    }
    smooth / max
}

/// Calcule la couleur RGB pour un pixel à partir de:
/// - son nombre d'itérations,
/// - la valeur finale de z,
/// - les paramètres de colorisation (palette, répétition).
pub fn color_for_pixel(
    iteration: u32,
    z: Complex64,
    iter_max: u32,
    palette_index: u8,
    color_repeat: u32,
) -> (u8, u8, u8) {
    // Points dans l'ensemble : noir
    if iteration >= iter_max {
        return (0, 0, 0);
    }

    let mut t = smooth_iteration(iteration, z, iter_max);
    if t >= 1.0 {
        t = 0.999_999;
    }

    let repeat_count = color_repeat.max(1) as f64;
    let mut cycle = (t * repeat_count).floor();
    let mut t_repeat = (t * repeat_count) % 1.0;

    if t_repeat < 0.000_001 && t > 0.0 {
        t_repeat = 0.999_999;
        cycle -= 1.0;
        if cycle < 0.0 {
            cycle = 0.0;
        }
    }

    // Alternance endroit/envers
    if (cycle as i64) % 2 == 1 {
        t_repeat = 1.0 - t_repeat;
    }

    let palette = palette_for(PaletteId::from_u8(palette_index));
    gradient_interpolate(palette, t_repeat)
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

