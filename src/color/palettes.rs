use num_complex::Complex64;
use std::sync::OnceLock;
use crate::fractal::{OutColoringMode, ColorSpace};
use crate::fractal::orbit_traps::OrbitData;
use crate::color::color_models::{rgb_to_hsb, hsb_to_rgb, rgb_to_lch, lch_to_rgb, interpolate_hsb, interpolate_lch};

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
    Neon = 9,
    Twilight = 10,
    Emboss = 11,
    Waves = 12,
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
            9 => PaletteId::Neon,
            10 => PaletteId::Twilight,
            11 => PaletteId::Emboss,
            12 => PaletteId::Waves,
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
pub(crate) struct Gradient {
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

// Palette "Néon" Profond : Noir → Bleu Marine → Bleu → Marron → Rouge → Noir
const NEON_STOPS: [GradientStop; 6] = [
    GradientStop { position: 0.000, r: 0, g: 0, b: 0 },       // Noir
    GradientStop { position: 0.200, r: 0, g: 0, b: 128 },      // Bleu Marine
    GradientStop { position: 0.400, r: 0, g: 0, b: 255 },      // Bleu
    GradientStop { position: 0.600, r: 128, g: 0, b: 0 },      // Marron
    GradientStop { position: 0.800, r: 255, g: 0, b: 0 },      // Rouge
    GradientStop { position: 1.000, r: 0, g: 0, b: 0 },        // Retour au Noir
];

// Palette "Embossage" : 50% blanc extérieur, 45% dégradé transition, 5% blanc intérieur
const EMBOSS_STOPS: [GradientStop; 4] = [
    GradientStop { position: 0.000, r: 255, g: 255, b: 255 }, // Blanc pur (extérieur)
    GradientStop { position: 0.500, r: 255, g: 255, b: 255 }, // Blanc pur (fin extérieur)
    GradientStop { position: 0.950, r: 0, g: 0, b: 0 },       // Noir (transition)
    GradientStop { position: 1.000, r: 255, g: 255, b: 255 }, // Blanc pur (intérieur)
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
const NEON: Gradient = Gradient { _name: "Neon", stops: &NEON_STOPS };
const EMBOSS: Gradient = Gradient { _name: "Emboss", stops: &EMBOSS_STOPS };

/// Génère la palette Twilight avec 510 couleurs (gris-lavande → spectre cyclique)
fn generate_twilight_stops() -> Vec<GradientStop> {
    let mut stops = Vec::with_capacity(510);
    let start_color = (225, 216, 226); // Gris-lavande
    
    for i in 0..510 {
        let t = i as f64 / 509.0;
        // Transition cyclique douce à travers le spectre
        let hue = t * 360.0; // 0-360 degrés
        let (r, g, b) = hsv_to_rgb(hue, 0.7, 0.9); // Saturation 0.7, Valeur 0.9
        
        // Mélanger avec la couleur de départ pour transition douce
        let blend_factor = if t < 0.1 {
            t / 0.1 // Transition douce depuis gris-lavande
        } else {
            1.0
        };
        
        let r_final = ((start_color.0 as f64 * (1.0 - blend_factor) + r as f64 * blend_factor) as u8).min(255);
        let g_final = ((start_color.1 as f64 * (1.0 - blend_factor) + g as f64 * blend_factor) as u8).min(255);
        let b_final = ((start_color.2 as f64 * (1.0 - blend_factor) + b as f64 * blend_factor) as u8).min(255);
        
        stops.push(GradientStop {
            position: t,
            r: r_final,
            g: g_final,
            b: b_final,
        });
    }
    
    stops
}

/// Génère la palette Waves avec ondes sinusoïdales (nombres premiers pour fréquences)
fn generate_waves_stops() -> Vec<GradientStop> {
    let mut stops = Vec::with_capacity(256);
    let freq_r = 2.0; // Premier nombre premier
    let freq_g = 3.0; // Deuxième nombre premier
    let freq_b = 5.0; // Troisième nombre premier
    
    for i in 0..256 {
        let t = i as f64 / 255.0;
        let r = ((t * std::f64::consts::PI * 2.0 * freq_r).sin() * 0.5 + 0.5) * 255.0;
        let g = ((t * std::f64::consts::PI * 2.0 * freq_g).sin() * 0.5 + 0.5) * 255.0;
        let b = ((t * std::f64::consts::PI * 2.0 * freq_b).sin() * 0.5 + 0.5) * 255.0;
        
        stops.push(GradientStop {
            position: t,
            r: r.clamp(0.0, 255.0) as u8,
            g: g.clamp(0.0, 255.0) as u8,
            b: b.clamp(0.0, 255.0) as u8,
        });
    }
    
    stops
}

/// Conversion HSV vers RGB (helper pour Twilight)
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    
    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    
    (
        ((r + m) * 255.0).clamp(0.0, 255.0) as u8,
        ((g + m) * 255.0).clamp(0.0, 255.0) as u8,
        ((b + m) * 255.0).clamp(0.0, 255.0) as u8,
    )
}

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
        PaletteId::Neon => NEON,
        PaletteId::Twilight => Gradient { _name: "Twilight", stops: &[] }, // Généré dynamiquement
        PaletteId::Emboss => EMBOSS,
        PaletteId::Waves => Gradient { _name: "Waves", stops: &[] }, // Généré dynamiquement
    }
}

/// Génère une petite image de prévisualisation de la palette.
/// Retourne une ColorImage egui de taille width x height représentant le gradient de la palette.
pub fn generate_palette_preview(palette_index: u8, width: u32, height: u32) -> egui::ColorImage {
    let palette_id = PaletteId::from_u8(palette_index);
    let gradient = palette_for(palette_id);
    
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    
    for _y in 0..height {
        for x in 0..width {
            // Calculer la position dans le gradient (0.0 à 1.0)
            let t = x as f64 / (width - 1) as f64;
            let (r, g, b) = gradient_interpolate(gradient, t);
            rgba.push(r);
            rgba.push(g);
            rgba.push(b);
            rgba.push(255); // Alpha
        }
    }
    
    egui::ColorImage::from_rgba_unmultiplied([width as usize, height as usize], &rgba)
}

fn gradient_interpolate(g: Gradient, mut t: f64) -> (u8, u8, u8) {
    // Gérer les palettes dynamiques avec cache thread-safe
    let stops: &[GradientStop] = if g.stops.is_empty() {
        // Palette dynamique - déterminer laquelle basé sur le nom
        match g._name {
            "Twilight" => {
                static CACHE: OnceLock<Vec<GradientStop>> = OnceLock::new();
                CACHE.get_or_init(|| generate_twilight_stops())
            }
            "Waves" => {
                static CACHE: OnceLock<Vec<GradientStop>> = OnceLock::new();
                CACHE.get_or_init(|| generate_waves_stops())
            }
            _ => g.stops, // Fallback (ne devrait pas arriver)
        }
    } else {
        g.stops
    };

    // Clamp t
    if t < 0.0 {
        t = 0.0;
    }
    if t > 1.0 {
        t = 1.0;
    }

    let eps = 1e-9;

    if stops.is_empty() {
        return (0, 0, 0);
    }

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

/// Interpole un gradient en espace HSB pour transitions plus naturelles
fn gradient_interpolate_hsb(g: Gradient, mut t: f64) -> (u8, u8, u8) {
    // Gérer les palettes dynamiques avec cache thread-safe
    let stops: &[GradientStop] = if g.stops.is_empty() {
        match g._name {
            "Twilight" => {
                static CACHE: OnceLock<Vec<GradientStop>> = OnceLock::new();
                CACHE.get_or_init(|| generate_twilight_stops())
            }
            "Waves" => {
                static CACHE: OnceLock<Vec<GradientStop>> = OnceLock::new();
                CACHE.get_or_init(|| generate_waves_stops())
            }
            _ => g.stops,
        }
    } else {
        g.stops
    };

    if t < 0.0 {
        t = 0.0;
    }
    if t > 1.0 {
        t = 1.0;
    }

    let eps = 1e-9;

    if stops.is_empty() {
        return (0, 0, 0);
    }

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
            
            // Interpolation en espace HSB
            let hsb1 = rgb_to_hsb(a.r, a.g, a.b);
            let hsb2 = rgb_to_hsb(b.r, b.g, b.b);
            let hsb_interp = interpolate_hsb(hsb1, hsb2, factor);
            return hsb_to_rgb(hsb_interp);
        }
    }

    (last.r, last.g, last.b)
}

/// Interpole un gradient en espace LCH pour transitions perceptuellement uniformes
/// Utilise la formule proposée : L=75-75v, C=28+75-75v avec fonction cosinus
fn gradient_interpolate_lch(g: Gradient, mut t: f64) -> (u8, u8, u8) {
    // Gérer les palettes dynamiques avec cache thread-safe
    let stops: &[GradientStop] = if g.stops.is_empty() {
        match g._name {
            "Twilight" => {
                static CACHE: OnceLock<Vec<GradientStop>> = OnceLock::new();
                CACHE.get_or_init(|| generate_twilight_stops())
            }
            "Waves" => {
                static CACHE: OnceLock<Vec<GradientStop>> = OnceLock::new();
                CACHE.get_or_init(|| generate_waves_stops())
            }
            _ => g.stops,
        }
    } else {
        g.stops
    };

    if t < 0.0 {
        t = 0.0;
    }
    if t > 1.0 {
        t = 1.0;
    }

    let eps = 1e-9;

    if stops.is_empty() {
        return (0, 0, 0);
    }

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
            
            // Interpolation en espace LCH
            let lch1 = rgb_to_lch(a.r, a.g, a.b);
            let lch2 = rgb_to_lch(b.r, b.g, b.b);
            let lch_interp = interpolate_lch(lch1, lch2, factor);
            return lch_to_rgb(lch_interp);
        }
    }

    (last.r, last.g, last.b)
}

/// Interpole un gradient selon l'espace colorimétrique sélectionné
pub fn gradient_interpolate_with_space(g: Gradient, t: f64, color_space: ColorSpace) -> (u8, u8, u8) {
    match color_space {
        ColorSpace::Rgb => gradient_interpolate(g, t),
        ColorSpace::Hsb => gradient_interpolate_hsb(g, t),
        ColorSpace::Lch => gradient_interpolate_lch(g, t),
    }
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

/// Compute orbit traps coloring based on minimum distance to trap
fn compute_orbit_traps(orbit: Option<&OrbitData>, iteration: u32, iter_max: u32) -> f64 {
    match orbit {
        Some(orbit_data) if !orbit_data.points.is_empty() => {
            // Utiliser la distance minimale normalisée pour la coloration
            // Plus la distance est petite, plus la valeur est élevée
            let normalized_distance = if orbit_data.min_distance > 0.0 {
                // Normaliser la distance (inverser pour que proche = valeur élevée)
                1.0 / (1.0 + orbit_data.min_distance * 10.0)
            } else {
                1.0
            };
            
            // Mélanger avec l'itération pour variation
            let iter_factor = iteration as f64 / iter_max as f64;
            (normalized_distance * 0.7 + iter_factor * 0.3).min(1.0)
        }
        _ => {
            // Pas d'orbite disponible, utiliser itération simple
            iteration as f64 / iter_max as f64
        }
    }
}

/// Compute Wings coloring using sinh() on orbit
fn compute_wings(orbit: Option<&OrbitData>, iteration: u32, iter_max: u32) -> f64 {
    match orbit {
        Some(orbit_data) if !orbit_data.points.is_empty() => {
            // Calculer sinh sur les points de l'orbite pour créer des motifs en ailes
            let mut sum_sinh = 0.0;
            let mut count = 0;
            
            for &point in &orbit_data.points {
                let sinh_re = point.re.sinh();
                let sinh_im = point.im.sinh();
                let magnitude = (sinh_re * sinh_re + sinh_im * sinh_im).sqrt();
                sum_sinh += magnitude;
                count += 1;
            }
            
            if count > 0 {
                let avg_sinh = sum_sinh / count as f64;
                // Normaliser et mélanger avec itération
                let normalized = (avg_sinh / (1.0 + avg_sinh)).min(1.0);
                let iter_factor = iteration as f64 / iter_max as f64;
                (normalized * 0.6 + iter_factor * 0.4).min(1.0)
            } else {
                iteration as f64 / iter_max as f64
            }
        }
        _ => {
            iteration as f64 / iter_max as f64
        }
    }
}

/// Distance field gradient coloring.
/// Closer to boundary = higher value (brighter).
fn compute_distance_gradient(distance: f64, iteration: u32, iter_max: u32) -> f64 {
    if !distance.is_finite() || distance <= 0.0 || distance == f64::INFINITY {
        // Fall back to iteration-based coloring
        return iteration as f64 / iter_max as f64;
    }

    // Normalize distance (log scale works well for fractals)
    // Smaller distance = closer to boundary = higher value
    let log_dist = distance.ln();
    let normalized = (-log_dist / 10.0).clamp(0.0, 1.0);

    // Mix with iteration for variation
    let iter_factor = iteration as f64 / iter_max as f64;
    (normalized * 0.7 + iter_factor * 0.3).min(1.0)
}

/// Ambient occlusion style coloring.
/// Darker near edges (small distance), brighter further away.
fn compute_distance_ao(distance: f64, iteration: u32, iter_max: u32) -> f64 {
    if !distance.is_finite() || distance <= 0.0 || distance == f64::INFINITY {
        return iteration as f64 / iter_max as f64;
    }

    // AO factor: e^(-k/distance) where k is a scale factor
    let ao_scale = 0.5;
    let ao_factor = (1.0 - (-ao_scale / distance).exp()).clamp(0.0, 1.0);

    let iter_factor = iteration as f64 / iter_max as f64;
    (ao_factor * 0.6 + iter_factor * 0.4).min(1.0)
}

/// 3D shading effect using distance and z angle.
/// Simulates a light source from top-left.
fn compute_distance_3d(distance: f64, z: Complex64, iteration: u32, iter_max: u32) -> f64 {
    if !distance.is_finite() || distance <= 0.0 || distance == f64::INFINITY {
        return iteration as f64 / iter_max as f64;
    }

    // Simulate a light source from top-left (like traditional 3D shading)
    let angle = z.im.atan2(z.re);
    let light_angle = std::f64::consts::PI * 0.75; // Top-left light
    let dot = ((angle - light_angle).cos() + 1.0) / 2.0;

    // Combine with distance for depth
    let depth_factor = 1.0 / (1.0 + distance * 0.1);
    let shading = (dot * depth_factor * 0.5 + 0.5).clamp(0.0, 1.0);

    let iter_factor = iteration as f64 / iter_max as f64;
    (shading * 0.6 + iter_factor * 0.4).min(1.0)
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
    color_space: ColorSpace,
    orbit: Option<&OrbitData>,
    distance: Option<f64>,
    interior_flag_encoded: bool,
) -> (u8, u8, u8) {
    // Points dans l'ensemble : noir
    if iteration >= iter_max {
        return (0, 0, 0);
    }

    let original_z_im_negative = z.im < 0.0;
    let is_interior = interior_flag_encoded && z.im < 0.0;
    if is_interior && out_coloring_mode != OutColoringMode::BinaryDecomposition {
        return (0, 0, 0);
    }

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
        OutColoringMode::OrbitTraps => compute_orbit_traps(orbit, iteration, iter_max),
        OutColoringMode::Wings => compute_wings(orbit, iteration, iter_max),
        // Distance modes - require distance estimation to be enabled
        OutColoringMode::Distance => match distance {
            Some(d) => compute_distance_gradient(d, iteration, iter_max),
            None => smooth_iteration(iteration, z_positive, iter_max, 2.0),
        },
        OutColoringMode::DistanceAO => match distance {
            Some(d) => compute_distance_ao(d, iteration, iter_max),
            None => smooth_iteration(iteration, z_positive, iter_max, 2.0),
        },
        OutColoringMode::Distance3D => match distance {
            Some(d) => compute_distance_3d(d, z_positive, iteration, iter_max),
            None => smooth_iteration(iteration, z_positive, iter_max, 2.0),
        },
    };

    // Clamp t dans [0, 1) pour éviter les problèmes aux limites
    if t < 0.0 {
        t = 0.0;
    }
    if t >= 1.0 {
        t = 0.999_999;
    }

    // Utiliser color_repeat pour la longueur du cycle
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
    let (r, g, b) = gradient_interpolate_with_space(palette, t_repeat, color_space);

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

