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
    SynthRed = 13,
    LightYears = 14,
    Blues = 15,
    Coffee = 16,
    Classic = 17,
    Dimensions = 18,
    Earth = 19,
    FireIce = 20,
    Habs = 21,
    Jays = 22,
    Slice = 23,
    Stardust = 24,
    Strobe = 25,
    SynthBlue = 26,
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
            13 => PaletteId::SynthRed,
            14 => PaletteId::LightYears,
            15 => PaletteId::Blues,
            16 => PaletteId::Coffee,
            17 => PaletteId::Classic,
            18 => PaletteId::Dimensions,
            19 => PaletteId::Earth,
            20 => PaletteId::FireIce,
            21 => PaletteId::Habs,
            22 => PaletteId::Jays,
            23 => PaletteId::Slice,
            24 => PaletteId::Stardust,
            25 => PaletteId::Strobe,
            26 => PaletteId::SynthBlue,
            _ => PaletteId::Plasma,
        }
    }
}

/// Pre-computed palette lookup table for fast per-pixel coloring.
/// Eliminates expensive HSB/LCH trigonometric conversions by pre-computing
/// 4096 RGB entries covering the full [0, 1) gradient range.
pub struct PaletteLut {
    entries: Vec<(u8, u8, u8)>,
}

const PALETTE_LUT_SIZE: usize = 4096;

impl PaletteLut {
    /// Build a LUT for the given palette and color space.
    pub fn new(palette_index: u8, color_space: ColorSpace) -> Self {
        let palette = palette_for(PaletteId::from_u8(palette_index));
        let mut entries = Vec::with_capacity(PALETTE_LUT_SIZE);
        for i in 0..PALETTE_LUT_SIZE {
            let t = i as f64 / PALETTE_LUT_SIZE as f64;
            entries.push(gradient_interpolate_with_space(palette, t, color_space));
        }
        Self { entries }
    }

    /// Look up RGB color for t in [0, 1) with linear interpolation between adjacent entries.
    #[inline]
    pub fn lookup(&self, t: f64) -> (u8, u8, u8) {
        let t_clamped = t.clamp(0.0, 0.999_999);
        let pos = t_clamped * PALETTE_LUT_SIZE as f64;
        let idx = pos as usize;
        let frac = pos - idx as f64;

        let (r0, g0, b0) = self.entries[idx];
        if frac < 0.001 || idx + 1 >= PALETTE_LUT_SIZE {
            return (r0, g0, b0);
        }
        let (r1, g1, b1) = self.entries[idx + 1];
        let r = (r0 as f64 + (r1 as f64 - r0 as f64) * frac) as u8;
        let g = (g0 as f64 + (g1 as f64 - g0 as f64) * frac) as u8;
        let b = (b0 as f64 + (b1 as f64 - b0 as f64) * frac) as u8;
        (r, g, b)
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

const SYNTHRED_STOPS: [GradientStop; 6] = [
    GradientStop { position: 0.0, r: 16, g: 0, b: 55 },
    GradientStop { position: 0.2, r: 121, g: 0, b: 159 },
    GradientStop { position: 0.4, r: 252, g: 196, b: 255 },
    GradientStop { position: 0.6, r: 58, g: 26, b: 168 },
    GradientStop { position: 0.8, r: 8, g: 0, b: 36 },
    GradientStop { position: 1.0, r: 16, g: 0, b: 55 },
];
const SYNTHRED: Gradient = Gradient { _name: "SynthRed", stops: &SYNTHRED_STOPS };

const BLUES_STOPS: [GradientStop; 5] = [
    GradientStop { position: 0.00, r: 0, g: 0, b: 0 },
    GradientStop { position: 0.25, r: 0, g: 128, b: 255 },
    GradientStop { position: 0.50, r: 252, g: 181, b: 20 },
    GradientStop { position: 0.75, r: 0, g: 38, b: 84 },
    GradientStop { position: 1.00, r: 255, g: 255, b: 255 },
];
const BLUES: Gradient = Gradient { _name: "Blues", stops: &BLUES_STOPS };

const COFFEE_STOPS: [GradientStop; 4] = [
    GradientStop { position: 0.000, r: 255, g: 254, b: 214 },
    GradientStop { position: 0.333, r: 111, g: 69, b: 36 },
    GradientStop { position: 0.667, r: 37, g: 18, b: 0 },
    GradientStop { position: 1.000, r: 129, g: 93, b: 61 },
];
const COFFEE: Gradient = Gradient { _name: "Coffee", stops: &COFFEE_STOPS };

const CLASSIC_STOPS: [GradientStop; 16] = [
    GradientStop { position: 0.000, r: 0, g: 2, b: 0 },
    GradientStop { position: 0.067, r: 0, g: 7, b: 41 },
    GradientStop { position: 0.133, r: 0, g: 5, b: 94 },
    GradientStop { position: 0.200, r: 0, g: 43, b: 134 },
    GradientStop { position: 0.267, r: 8, g: 84, b: 178 },
    GradientStop { position: 0.333, r: 53, g: 124, b: 219 },
    GradientStop { position: 0.400, r: 112, g: 173, b: 251 },
    GradientStop { position: 0.467, r: 176, g: 221, b: 255 },
    GradientStop { position: 0.533, r: 226, g: 252, b: 255 },
    GradientStop { position: 0.600, r: 250, g: 250, b: 228 },
    GradientStop { position: 0.667, r: 255, g: 226, b: 160 },
    GradientStop { position: 0.733, r: 255, g: 193, b: 78 },
    GradientStop { position: 0.800, r: 242, g: 159, b: 0 },
    GradientStop { position: 0.867, r: 160, g: 103, b: 0 },
    GradientStop { position: 0.933, r: 66, g: 46, b: 0 },
    GradientStop { position: 1.000, r: 0, g: 2, b: 0 },
];
const CLASSIC: Gradient = Gradient { _name: "Classic", stops: &CLASSIC_STOPS };

const EARTH_STOPS: [GradientStop; 5] = [
    GradientStop { position: 0.00, r: 29, g: 11, b: 0 },
    GradientStop { position: 0.25, r: 27, g: 93, b: 0 },
    GradientStop { position: 0.50, r: 39, g: 60, b: 0 },
    GradientStop { position: 0.75, r: 248, g: 255, b: 181 },
    GradientStop { position: 1.00, r: 111, g: 73, b: 10 },
];
const EARTH: Gradient = Gradient { _name: "Earth", stops: &EARTH_STOPS };

const FIRE_ICE_STOPS: [GradientStop; 9] = [
    GradientStop { position: 0.000, r: 0, g: 0, b: 0 },
    GradientStop { position: 0.125, r: 255, g: 0, b: 0 },
    GradientStop { position: 0.250, r: 255, g: 255, b: 0 },
    GradientStop { position: 0.375, r: 255, g: 255, b: 255 },
    GradientStop { position: 0.500, r: 128, g: 255, b: 255 },
    GradientStop { position: 0.625, r: 0, g: 255, b: 255 },
    GradientStop { position: 0.750, r: 0, g: 128, b: 255 },
    GradientStop { position: 0.875, r: 0, g: 0, b: 255 },
    GradientStop { position: 1.000, r: 0, g: 0, b: 128 },
];
const FIRE_ICE: Gradient = Gradient { _name: "FireIce", stops: &FIRE_ICE_STOPS };

const HABS_STOPS: [GradientStop; 4] = [
    GradientStop { position: 0.000, r: 0, g: 0, b: 0 },
    GradientStop { position: 0.333, r: 175, g: 30, b: 45 },
    GradientStop { position: 0.667, r: 255, g: 255, b: 255 },
    GradientStop { position: 1.000, r: 25, g: 33, b: 104 },
];
const HABS: Gradient = Gradient { _name: "Habs", stops: &HABS_STOPS };

const JAYS_STOPS: [GradientStop; 5] = [
    GradientStop { position: 0.00, r: 0, g: 0, b: 0 },
    GradientStop { position: 0.25, r: 0, g: 128, b: 255 },
    GradientStop { position: 0.50, r: 0, g: 0, b: 160 },
    GradientStop { position: 0.75, r: 255, g: 255, b: 255 },
    GradientStop { position: 1.00, r: 255, g: 0, b: 0 },
];
const JAYS: Gradient = Gradient { _name: "Jays", stops: &JAYS_STOPS };

const SLICE_STOPS: [GradientStop; 16] = [
    GradientStop { position: 0.000, r: 0, g: 0, b: 0 },
    GradientStop { position: 0.067, r: 134, g: 32, b: 45 },
    GradientStop { position: 0.133, r: 194, g: 44, b: 131 },
    GradientStop { position: 0.200, r: 0, g: 0, b: 0 },
    GradientStop { position: 0.267, r: 29, g: 181, b: 140 },
    GradientStop { position: 0.333, r: 153, g: 94, b: 108 },
    GradientStop { position: 0.400, r: 159, g: 117, b: 165 },
    GradientStop { position: 0.467, r: 31, g: 33, b: 2 },
    GradientStop { position: 0.533, r: 255, g: 229, b: 133 },
    GradientStop { position: 0.600, r: 255, g: 255, b: 255 },
    GradientStop { position: 0.667, r: 0, g: 0, b: 0 },
    GradientStop { position: 0.733, r: 30, g: 111, b: 0 },
    GradientStop { position: 0.800, r: 156, g: 143, b: 137 },
    GradientStop { position: 0.867, r: 110, g: 44, b: 75 },
    GradientStop { position: 0.933, r: 40, g: 226, b: 233 },
    GradientStop { position: 1.000, r: 165, g: 28, b: 118 },
];
const SLICE: Gradient = Gradient { _name: "Slice", stops: &SLICE_STOPS };

const STARDUST_STOPS: [GradientStop; 5] = [
    GradientStop { position: 0.00, r: 2, g: 0, b: 43 },
    GradientStop { position: 0.25, r: 9, g: 21, b: 99 },
    GradientStop { position: 0.50, r: 90, g: 230, b: 216 },
    GradientStop { position: 0.75, r: 179, g: 255, b: 241 },
    GradientStop { position: 1.00, r: 255, g: 203, b: 190 },
];
const STARDUST: Gradient = Gradient { _name: "Stardust", stops: &STARDUST_STOPS };

const STROBE_STOPS: [GradientStop; 8] = [
    GradientStop { position: 0.000, r: 0, g: 0, b: 0 },
    GradientStop { position: 0.143, r: 255, g: 0, b: 0 },
    GradientStop { position: 0.286, r: 0, g: 0, b: 0 },
    GradientStop { position: 0.429, r: 24, g: 255, b: 0 },
    GradientStop { position: 0.571, r: 0, g: 0, b: 0 },
    GradientStop { position: 0.714, r: 0, g: 139, b: 255 },
    GradientStop { position: 0.857, r: 0, g: 0, b: 0 },
    GradientStop { position: 1.000, r: 253, g: 0, b: 255 },
];
const STROBE: Gradient = Gradient { _name: "Strobe", stops: &STROBE_STOPS };

const SYNTHBLUE_STOPS: [GradientStop; 6] = [
    GradientStop { position: 0.0, r: 16, g: 0, b: 55 },
    GradientStop { position: 0.2, r: 121, g: 0, b: 159 },
    GradientStop { position: 0.4, r: 252, g: 196, b: 255 },
    GradientStop { position: 0.6, r: 58, g: 26, b: 168 },
    GradientStop { position: 0.8, r: 8, g: 0, b: 36 },
    GradientStop { position: 1.0, r: 16, g: 0, b: 55 },
];
const SYNTHBLUE: Gradient = Gradient { _name: "SynthBlue", stops: &SYNTHBLUE_STOPS };

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

/// Génère la palette LightYears avec 128 couleurs (importée de rust-fractal-core)
fn generate_light_years_stops() -> Vec<GradientStop> {
    const RGB: &[u8] = &[
        0,0,0,190,35,41,108,225,132,82,174,214,241,73,144,233,187,241,166,179,235,135,60,219,
        153,62,12,13,94,36,183,6,28,179,222,71,200,77,18,139,187,67,3,31,166,9,125,90,31,37,56,
        203,212,93,245,150,252,19,59,69,10,137,13,174,219,28,154,32,50,64,238,80,253,54,120,50,
        73,18,125,158,246,173,220,73,242,20,79,102,64,68,196,107,208,50,183,48,34,161,59,145,34,
        246,139,225,157,176,218,31,2,153,202,157,114,185,128,44,73,153,197,126,128,233,213,201,
        234,178,191,83,204,191,214,103,126,214,20,142,220,45,239,131,102,97,73,87,143,105,255,
        209,205,97,156,157,30,114,114,22,240,29,230,74,79,132,215,2,119,44,57,232,201,203,83,51,
        30,18,12,158,116,212,213,244,164,212,159,53,126,89,34,50,207,207,204,244,45,144,211,143,
        211,72,217,230,117,229,42,29,43,247,192,135,129,120,95,14,68,212,0,80,190,141,97,21,5,
        123,51,59,7,24,31,130,218,146,112,206,84,100,62,133,177,248,21,105,4,106,70,14,115,150,
        47,22,217,212,104,103,74,74,247,104,87,208,22,250,118,173,17,187,136,36,174,82,254,121,
        67,37,219,244,60,229,216,211,69,11,206,40,96,197,245,151,61,89,89,138,39,208,45,118,205,
        201,194,73,212,104,37,121,106,64,97,8,59,177,20,17,165,106,140,193,40,11,169,214,140,151,
        135,21,241,47,149,154,29,225,155,193,233,126,192,167,154,168,181,194,134,154,191,84,35,
        217,231,144,85,209,209,40,56,161,108,217,78,94,102,156,48,225,113,217,254,165,226,159,
        155,12,226,101,71,180,70,42,56,130,169,137,118,122,121,99,194,120,
    ];
    let count = RGB.len() / 3;
    let mut stops = Vec::with_capacity(count);
    for i in 0..count {
        stops.push(GradientStop {
            position: i as f64 / (count - 1) as f64,
            r: RGB[i * 3],
            g: RGB[i * 3 + 1],
            b: RGB[i * 3 + 2],
        });
    }
    stops
}

/// Génère la palette Dimensions avec 1024 couleurs (importée de rust-fractal-core)
fn generate_dimensions_stops() -> Vec<GradientStop> {
    const RGB: &[u8] = &[
        71,109,87,204,219,205,31,115,25,17,30,15,110,127,96,216,235,233,90,223,242,33,44,33,
        174,136,22,219,218,209,55,85,121,9,42,34,18,254,156,204,235,222,88,96,92,38,42,21,219,
        242,82,231,241,215,99,163,114,35,25,34,186,42,158,215,218,230,5,172,157,16,34,31,129,
        105,91,238,219,206,248,121,33,50,28,14,159,107,83,213,223,225,17,151,191,27,35,41,199,
        134,142,239,213,216,187,40,59,40,35,19,140,246,99,222,228,205,112,54,11,41,13,17,218,52,
        131,221,201,221,20,27,107,20,32,29,145,236,126,215,249,207,51,231,6,29,42,15,181,111,
        116,232,206,235,148,8,240,47,23,59,233,180,232,241,234,242,166,162,175,26,28,51,46,65,
        240,218,210,236,169,90,119,26,30,23,39,152,66,217,226,230,170,129,246,29,47,54,68,250,
        193,219,250,222,159,222,54,40,47,26,167,158,154,222,228,226,83,142,127,31,39,19,168,174,
        26,239,234,203,218,173,73,33,42,31,52,168,182,222,237,232,197,203,151,41,55,19,137,244,
        4,218,232,201,82,89,80,40,37,29,245,207,156,231,234,225,74,135,118,29,44,16,163,220,10,
        221,230,201,78,90,72,29,42,21,157,251,100,233,250,226,184,223,182,40,34,27,137,49,35,
        220,220,216,100,186,170,38,48,29,210,198,66,217,234,201,1,146,19,6,48,30,50,239,223,219,
        253,243,176,255,193,49,50,49,221,146,206,222,210,229,27,9,100,27,2,39,194,7,213,225,203,
        227,83,89,80,28,40,28,147,233,149,211,241,237,18,165,217,34,20,27,255,0,0,229,192,203,
        47,12,100,22,4,41,130,20,229,220,203,234,106,75,119,45,41,14,255,255,0,251,227,205,224,
        31,117,39,35,41,95,254,212,226,228,243,186,45,208,49,11,28,211,46,21,249,205,218,253,68,
        195,53,17,42,175,73,146,229,227,210,133,218,6,18,41,31,11,111,249,210,206,252,143,10,
        239,40,23,59,181,177,239,225,230,221,91,136,1,13,47,19,13,247,151,219,230,237,209,66,
        216,45,28,32,155,161,47,216,231,227,48,160,239,32,44,45,214,195,128,223,237,230,42,174,
        184,33,24,24,226,19,13,251,211,224,252,139,252,36,45,39,39,225,63,209,249,214,103,240,
        126,14,51,17,14,172,12,211,236,203,147,187,82,28,29,34,79,52,196,208,219,229,61,177,109,
        25,25,14,141,24,5,233,225,202,198,251,83,31,44,12,56,101,16,229,205,223,253,15,245,60,
        22,60,233,162,237,243,242,223,185,250,21,45,36,6,180,44,33,232,206,221,153,76,209,24,12,
        54,46,22,230,213,196,223,134,22,30,17,26,7,9,192,32,219,237,207,214,180,97,46,49,29,156,
        218,137,213,231,220,19,101,97,25,32,37,186,155,200,240,236,225,211,207,72,44,35,23,144,
        75,113,215,201,220,48,9,119,24,15,18,150,115,25,233,230,203,185,200,73,45,35,10,175,85,
        10,243,224,196,244,180,33,31,37,11,6,116,61,216,209,210,195,27,94,35,14,31,85,85,155,
        212,217,242,88,121,254,29,39,55,147,194,186,223,232,217,113,133,26,21,22,21,61,50,146,
        201,224,226,20,212,136,25,32,44,181,45,217,214,201,226,8,40,64,15,28,9,112,185,14,228,
        245,203,185,252,84,51,38,21,227,54,86,229,229,222,77,253,163,23,38,42,114,57,178,207,
        211,230,15,104,138,28,21,26,215,66,72,234,215,208,129,127,63,37,27,28,171,95,167,234,
        223,232,176,159,162,26,23,30,32,28,81,215,213,220,159,149,152,22,29,49,18,83,242,222,
        210,233,229,68,96,58,30,13,237,177,10,247,237,202,212,191,83,30,26,41,32,18,248,219,205,
        224,192,96,17,27,12,23,24,6,168,202,208,214,63,132,18,38,18,12,244,14,78,237,223,201,
        122,241,6,29,45,31,113,124,243,234,210,240,233,26,148,53,3,43,194,3,198,217,214,226,14,
        179,81,11,31,23,76,72,105,201,211,205,6,87,12,16,14,24,128,27,182,209,200,237,21,46,190,
        12,11,45,76,44,173,214,223,236,110,215,192,40,35,41,211,69,137,246,199,210,228,0,20,42,
        15,9,113,122,53,216,220,226,86,110,229,17,40,46,55,210,139,226,230,226,227,107,141,40,
        22,24,96,75,51,213,226,220,82,207,186,24,35,53,110,80,241,211,205,239,48,33,146,36,35,
        42,241,250,195,239,238,244,143,125,227,23,37,33,42,171,39,211,222,219,116,78,187,46,38,
        34,254,231,85,232,250,202,76,242,7,41,46,12,253,129,93,223,236,203,3,230,5,13,41,12,105,
        105,93,223,206,224,153,14,172,35,26,31,127,196,83,214,221,212,58,48,85,21,8,17,110,20,
        57,215,218,208,83,194,81,34,43,22,193,150,100,218,228,209,21,147,45,8,23,17,43,44,98,
        208,228,229,95,252,204,42,43,51,242,96,209,223,223,238,15,159,172,15,32,48,108,104,215,
        209,230,246,37,213,230,14,33,38,79,52,78,208,209,226,62,94,203,31,41,50,187,235,197,224,
        251,222,79,248,52,19,43,32,79,100,210,222,231,244,174,220,214,42,43,42,167,127,122,236,
        217,213,195,85,56,26,23,26,15,99,154,224,234,215,253,244,40,42,32,22,84,19,139,229,194,
        231,221,6,184,59,28,38,251,218,126,229,230,212,55,92,42,15,26,13,67,118,67,229,222,211,
        241,129,92,53,17,26,189,14,120,245,223,223,243,243,141,42,41,27,98,85,79,203,233,218,0,
        255,142,13,46,31,106,113,113,226,214,230,173,73,198,32,23,45,86,114,168,225,210,230,185,
        43,142,52,22,27,234,133,78,233,221,226,102,111,202,13,18,49,8,38,191,214,197,241,178,14,
        211,43,32,33,168,248,58,227,222,199,122,0,10,28,3,31,109,25,239,233,201,242,231,58,169,
        44,38,27,123,253,53,230,236,220,192,111,177,43,21,26,153,63,31,224,205,196,114,53,8,40,
        35,28,211,232,221,246,238,222,231,146,29,29,20,26,6,21,181,209,198,236,140,40,177,22,6,
        37,38,13,123,197,206,225,15,111,154,6,17,43,33,29,195,219,219,233,195,200,143,44,52,27,
        157,218,80,231,227,213,166,70,101,23,27,26,21,151,108,209,226,214,125,131,76,46,22,13,
        250,49,33,239,225,223,133,222,224,44,43,46,219,128,150,240,237,216,172,238,49,47,31,6,
        204,10,0,226,195,193,75,20,17,25,28,28,129,208,214,233,231,226,211,110,66,53,22,19,214,
        73,92,249,220,214,248,160,96,54,49,31,186,238,152,221,225,229,56,37,153,30,32,28,188,
        220,75,240,223,218,205,41,145,46,20,35,167,119,141,225,210,209,106,31,2,43,7,3,239,25,
        22,233,211,225,98,139,255,40,23,43,227,48,90,250,220,229,246,188,216,58,34,27,218,86,5,
        219,214,195,7,103,26,27,21,18,212,71,125,230,212,210,101,96,31,22,37,17,76,200,108,212,
        245,209,95,236,34,17,55,20,47,209,130,212,228,235,120,89,225,20,31,58,40,160,245,203,
        239,226,59,228,36,32,44,17,204,128,102,237,209,229,166,16,204,30,14,31,75,99,47,223,203,
        203,180,1,53,44,20,16,172,164,75,239,226,202,214,115,12,50,31,12,190,138,87,232,221,231,
        139,104,238,26,25,41,71,100,93,211,224,215,89,162,102,14,31,25,28,90,101,219,218,208,
        196,126,39,27,23,5,27,61,4,199,225,210,42,215,149,13,27,29,69,4,90,215,221,233,128,241,
        247,30,53,40,113,186,74,218,233,229,104,155,233,21,51,54,67,255,200,203,224,245,27,10,
        235,17,15,47,113,116,142,207,210,212,19,38,29,17,25,23,124,165,162,223,218,228,134,49,
        133,29,15,29,104,77,105,205,203,219,6,22,119,3,4,23,25,12,68,223,216,230,229,193,247,39,
        53,58,87,232,224,208,230,224,52,83,38,29,15,29,181,38,197,227,197,219,108,13,32,37,15,29,
        192,107,203,220,212,243,38,65,215,13,11,42,73,28,123,215,217,231,118,178,198,36,37,41,
        172,120,133,235,211,226,180,39,152,33,33,44,87,231,207,230,223,219,230,30,21,28,16,29,0,
        104,213,191,232,248,3,223,241,11,33,38,85,47,67,226,209,210,194,96,88,24,37,14,1,201,27,
        215,240,202,190,191,64,35,36,32,91,103,192,204,210,245,12,50,245,6,9,54,36,26,189,214,
        219,225,152,198,85,21,29,15,16,40,36,221,204,217,227,68,174,31,33,48,26,197,211,194,217,
        227,1,10,78,26,27,11,209,206,11,244,242,195,219,202,22,30,36,5,27,93,18,199,214,215,39,
        96,179,9,22,50,35,80,221,214,210,229,152,71,81,33,11,40,114,24,246,211,203,248,47,77,
        213,17,22,29,91,100,23,224,223,196,177,155,15,31,37,8,72,143,51,222,214,210,180,45,99,
        39,32,40,139,211,222,233,226,244,195,74,202,47,31,36,181,179,92,229,233,212,128,161,77,
        37,23,33,175,25,192,239,196,223,211,14,62,26,4,34,4,18,212,208,201,245,137,61,220,20,15,
        53,23,64,204,223,219,221,233,160,41,37,29,35,69,76,245,205,215,224,46,114,18,8,46,28,23,
        254,213,223,247,226,233,194,68,32,50,24,24,207,127,216,240,215,178,184,63,51,24,8,237,
        12,2,243,210,207,179,144,128,45,45,24,183,218,66,245,218,228,248,0,228,61,25,47,242,200,
        154,237,217,227,126,7,133,24,22,18,69,176,15,215,224,218,128,88,203,16,12,27,4,13,13,
        205,219,205,113,214,103,24,30,40,86,27,219,222,222,231,161,224,102,26,38,21,50,82,69,
        213,214,221,131,100,170,40,16,27,190,35,52,216,225,210,8,236,104,22,29,18,169,1,45,230,
        195,212,148,33,121,38,6,44,157,15,233,224,208,240,112,121,162,41,18,36,222,25,130,244,
        196,211,203,16,33,46,21,34,170,156,244,232,237,225,163,213,31,51,27,31,252,6,217,228,
        210,244,49,148,211,30,34,47,191,124,171,237,223,228,176,137,130,40,41,46,144,192,241,
        227,230,240,142,122,154,38,17,24,165,17,38,225,210,215,109,138,158,27,21,27,114,34,58,
        221,204,217,131,71,154,20,22,49,32,107,243,201,224,232,48,160,86,36,37,31,246,140,165,
        251,233,229,232,195,138,36,40,48,62,126,248,224,209,245,201,22,183,35,26,44,83,187,173,
        214,239,227,104,198,115,40,45,35,219,167,167,242,232,222,194,162,83,34,49,31,83,231,165,
        212,228,233,89,65,170,26,20,34,120,97,104,233,222,232,218,150,223,34,22,36,60,29,67,226,
        212,201,224,142,11,35,26,9,61,67,67,222,208,201,190,72,15,55,36,15,253,223,109,231,250,
        219,69,252,119,18,39,40,75,61,204,201,222,238,9,190,177,20,54,30,156,249,66,229,241,217,
        147,149,143,27,21,28,70,19,84,215,213,202,126,157,8,43,40,14,222,165,108,233,218,226,
        116,50,177,18,9,31,34,25,76,211,204,214,131,83,107,23,38,33,56,226,164,207,243,218,75,
        193,50,15,30,37,45,47,252,225,213,226,225,131,29,58,18,20,246,17,133,240,220,235,145,
        213,221,29,58,40,93,255,101,226,245,217,186,177,109,44,48,32,169,209,149,226,231,231,
        116,110,172,41,28,35,219,118,115,243,211,207,202,46,17,51,9,20,213,27,144,239,195,234,
        173,9,198,47,17,37,207,128,102,245,215,228,226,68,192,55,28,54,218,160,247,238,240,245,
        161,235,184,33,30,39,109,7,130,224,212,209,160,165,13,24,32,28,35,98,216,218,224,249,
        186,169,246,31,26,61,63,45,245,213,199,239,115,17,139,32,20,45,141,144,223,209,240,237,
        7,251,147,23,56,19,180,199,8,239,224,195,206,70,29,29,29,23,26,167,157,196,236,218,18,
        194,63,31,27,25,233,29,137,241,200,228,171,44,157,48,30,22,214,197,23,218,226,210,1,86,
        130,22,38,31,182,224,119,222,227,228,67,64,176,9,18,40,8,80,146,222,203,216,240,20,55,
        61,16,36,251,108,239,224,212,250,11,65,232,21,33,48,157,199,156,228,221,220,141,39,74,
        23,12,20,44,63,86,226,202,205,241,25,27,56,32,15,208,231,94,247,238,225,244,150,181,33,
        41,47,27,184,199,221,215,216,216,13,2,30,1,4,30,2,32,220,209,218,204,142,184,26,20,46,4,
        19,190,201,198,217,80,40,20,23,24,6,104,154,28,224,240,205,162,241,87,48,50,11,222,164,
        6,238,240,212,153,228,164,32,37,37,105,68,136,220,203,211,132,30,25,44,14,31,223,82,223,
        246,204,226,218,26,57,38,26,34,91,188,221,230,214,250,225,1,254,38,11,46,80,87,116,221,
        207,211,158,44,45,48,13,19,228,67,108,239,211,228,160,94,192,38,18,39,146,53,123,227,
        211,223,142,112,136,26,22,21,70,64,35,215,210,205,126,90,78,29,42,15,113,251,45,220,250,
        200,117,222,30,39,30,23,199,22,156,237,203,227,172,74,134,42,32,29,167,185,104,226,246,
        233,112,253,231,25,40,41,93,67,102,223,223,210,168,191,49,43,32,10,179,65,34,228,200,
        215,116,10,157,34,24,38,157,186,151,222,240,218,91,208,67,38,29,10,220,31,16,219,208,
        212,5,106,156,29,29,37,229,129,146,234,228,219,116,170,83,30,48,29,127,218,151,213,228,
        240,52,81,245,31,27,33,200,141,20,216,219,204,3,85,87,20,24,41,160,107,242,238,227,247,
        217,179,211,50,49,42,185,220,132,225,243,228,89,201,165,33,56,41,176,253,165,215,236,
        214,18,107,18,33,39,17,251,212,123,226,223,211,30,48,38,23,34,8,155,230,33,242,241,199,
        251,169,29,56,48,21,200,218,146,238,234,237,179,125,220,31,29,38,
    ];
    let count = RGB.len() / 3;
    let mut stops = Vec::with_capacity(count);
    for i in 0..count {
        stops.push(GradientStop {
            position: i as f64 / (count - 1) as f64,
            r: RGB[i * 3],
            g: RGB[i * 3 + 1],
            b: RGB[i * 3 + 2],
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
        PaletteId::SynthRed => SYNTHRED,
        PaletteId::LightYears => Gradient { _name: "LightYears", stops: &[] }, // Généré dynamiquement
        PaletteId::Blues => BLUES,
        PaletteId::Coffee => COFFEE,
        PaletteId::Classic => CLASSIC,
        PaletteId::Dimensions => Gradient { _name: "Dimensions", stops: &[] }, // Généré dynamiquement
        PaletteId::Earth => EARTH,
        PaletteId::FireIce => FIRE_ICE,
        PaletteId::Habs => HABS,
        PaletteId::Jays => JAYS,
        PaletteId::Slice => SLICE,
        PaletteId::Stardust => STARDUST,
        PaletteId::Strobe => STROBE,
        PaletteId::SynthBlue => SYNTHBLUE,
    }
}

/// Génère une petite image de prévisualisation de la palette.
/// Retourne une ColorImage egui de taille width x height représentant le gradient de la palette.
#[allow(dead_code)]
pub fn generate_palette_preview(palette_index: u8, width: u32, height: u32, color_space: ColorSpace) -> egui::ColorImage {
    let palette_id = PaletteId::from_u8(palette_index);
    let gradient = palette_for(palette_id);

    let mut rgba = Vec::with_capacity((width * height * 4) as usize);

    for _y in 0..height {
        for x in 0..width {
            // Calculer la position dans le gradient (0.0 à 1.0)
            let t = x as f64 / (width - 1) as f64;
            let (r, g, b) = gradient_interpolate_with_space(gradient, t, color_space);
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
            "LightYears" => {
                static CACHE: OnceLock<Vec<GradientStop>> = OnceLock::new();
                CACHE.get_or_init(|| generate_light_years_stops())
            }
            "Dimensions" => {
                static CACHE: OnceLock<Vec<GradientStop>> = OnceLock::new();
                CACHE.get_or_init(|| generate_dimensions_stops())
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

/// Seuil pour la frontière des biomorphs (Pickover).
/// Doit être du même ordre que le bailout d'échappement (2–4) pour que les deux
/// branches (intérieur / extérieur frontière) soient visibles.
const BIOMORPH_THRESHOLD: f64 = 3.0;

/// Compute biomorphs boundary detection value.
/// Pickover biomorphs : frontière quand |Re(z)| ou |Im(z)| dépasse le seuil.
/// Avec un bailout typique de 2–4, un seuil à 10 rendait la branche "smooth"
/// quasi inaccessible (z à l'échappement a presque toujours Re,Im < 10).
fn compute_biomorphs(iteration: u32, z: Complex64, iter_max: u32) -> f64 {
    let t = BIOMORPH_THRESHOLD;
    if z.re.abs() < t && z.im.abs() < t {
        // À l'intérieur de la « frontière » biomorphe : itération discrète (contours nets)
        iteration as f64 / iter_max as f64
    } else {
        // Au-delà de la frontière : smooth (dégradé)
        smooth_iteration(iteration, z, iter_max, 2.0)
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
#[allow(dead_code)]
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
    color_for_pixel_with_lut(iteration, z, iter_max, palette_index, color_repeat,
        out_coloring_mode, color_space, orbit, distance, interior_flag_encoded, None)
}

/// Optimized version of color_for_pixel that uses a pre-computed palette LUT
/// to avoid expensive HSB/LCH conversions per pixel.
pub fn color_for_pixel_with_lut(
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
    lut: Option<&PaletteLut>,
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

    let (r, g, b) = if let Some(lut) = lut {
        lut.lookup(t_repeat)
    } else {
        let palette = palette_for(PaletteId::from_u8(palette_index));
        gradient_interpolate_with_space(palette, t_repeat, color_space)
    };

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
    color_repeat: u32,
) -> (u8, u8, u8) {
    // z.re contient la densité normalisée * 2.0
    let t = (z.re / 2.0).clamp(0.0, 1.0);

    // Densité nulle = noir
    if t < 0.001 {
        return (0, 0, 0);
    }

    // Appliquer color_repeat (cycles du gradient) comme pour les autres fractales
    let repeat_count = color_repeat.max(1) as f64;
    let cycle = (t * repeat_count).floor();
    let mut t_repeat = (t * repeat_count) % 1.0;
    if t_repeat < 0.000_001 && t > 0.0 {
        t_repeat = 0.000_001;
    }
    if t_repeat >= 0.999_999 {
        t_repeat = 0.999_999;
    }
    if (cycle as i64) % 2 == 1 {
        t_repeat = 1.0 - t_repeat;
    }

    let palette = palette_for(PaletteId::from_u8(palette_index));
    gradient_interpolate(palette, t_repeat)
}

