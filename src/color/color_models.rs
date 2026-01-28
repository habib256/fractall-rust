/// Conversions entre espaces colorimétriques pour améliorer les gradients perceptuels.

/// Espace colorimétrique HSB/HSV (Teinte, Saturation, Brillance/Valeur)
#[derive(Clone, Copy, Debug)]
pub struct Hsb {
    pub h: f64, // Teinte [0, 360]
    pub s: f64, // Saturation [0, 1]
    pub b: f64, // Brillance/Valeur [0, 1]
}

/// Espace colorimétrique LCH (Luminosité, Chroma, Teinte)
#[derive(Clone, Copy, Debug)]
pub struct Lch {
    pub l: f64, // Luminosité [0, 100]
    pub c: f64, // Chroma [0, ~150]
    pub h: f64, // Teinte [0, 360]
}

/// Convertit RGB vers HSB/HSV
pub fn rgb_to_hsb(r: u8, g: u8, b: u8) -> Hsb {
    let r_f = r as f64 / 255.0;
    let g_f = g as f64 / 255.0;
    let b_f = b as f64 / 255.0;
    
    let max = r_f.max(g_f.max(b_f));
    let min = r_f.min(g_f.min(b_f));
    let delta = max - min;
    
    let h = if delta == 0.0 {
        0.0
    } else if max == r_f {
        60.0 * (((g_f - b_f) / delta) % 6.0)
    } else if max == g_f {
        60.0 * (((b_f - r_f) / delta) + 2.0)
    } else {
        60.0 * (((r_f - g_f) / delta) + 4.0)
    };
    
    let h = if h < 0.0 { h + 360.0 } else { h };
    let s = if max == 0.0 { 0.0 } else { delta / max };
    let b = max;
    
    Hsb { h, s, b }
}

/// Convertit HSB/HSV vers RGB
pub fn hsb_to_rgb(hsb: Hsb) -> (u8, u8, u8) {
    let h = hsb.h;
    let s = hsb.s;
    let v = hsb.b;
    
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

/// Convertit RGB vers XYZ (espace CIE 1931)
fn rgb_to_xyz(r: u8, g: u8, b: u8) -> (f64, f64, f64) {
    // Conversion linéaire depuis sRGB
    let r_lin = srgb_to_linear(r as f64 / 255.0);
    let g_lin = srgb_to_linear(g as f64 / 255.0);
    let b_lin = srgb_to_linear(b as f64 / 255.0);
    
    // Matrice de transformation sRGB vers XYZ (D65)
    let x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375;
    let y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750;
    let z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041;
    
    (x, y, z)
}

/// Convertit XYZ vers Lab
fn xyz_to_lab(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    // Illuminant D65
    let xn = 0.95047;
    let yn = 1.00000;
    let zn = 1.08883;
    
    let fx = lab_f(x / xn);
    let fy = lab_f(y / yn);
    let fz = lab_f(z / zn);
    
    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);
    
    (l, a, b)
}

/// Convertit Lab vers LCH
fn lab_to_lch(l: f64, a: f64, b: f64) -> Lch {
    let c = (a * a + b * b).sqrt();
    let h = b.atan2(a).to_degrees();
    let h = if h < 0.0 { h + 360.0 } else { h };
    
    Lch { l, c, h }
}

/// Convertit RGB vers LCH (via XYZ et Lab)
pub fn rgb_to_lch(r: u8, g: u8, b: u8) -> Lch {
    let (x, y, z) = rgb_to_xyz(r, g, b);
    let (l, a, b_lab) = xyz_to_lab(x, y, z);
    lab_to_lch(l, a, b_lab)
}

/// Convertit LCH vers Lab
fn lch_to_lab(lch: Lch) -> (f64, f64, f64) {
    let a = lch.c * (lch.h.to_radians().cos());
    let b = lch.c * (lch.h.to_radians().sin());
    (lch.l, a, b)
}

/// Convertit Lab vers XYZ
fn lab_to_xyz(l: f64, a: f64, b: f64) -> (f64, f64, f64) {
    // Illuminant D65
    let xn = 0.95047;
    let yn = 1.00000;
    let zn = 1.08883;
    
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;
    
    let x = lab_f_inv(fx) * xn;
    let y = lab_f_inv(fy) * yn;
    let z = lab_f_inv(fz) * zn;
    
    (x, y, z)
}

/// Convertit XYZ vers RGB
fn xyz_to_rgb(x: f64, y: f64, z: f64) -> (u8, u8, u8) {
    // Matrice de transformation XYZ vers sRGB (D65)
    let r_lin = x * 3.2404542 + y * -1.5371385 + z * -0.4985314;
    let g_lin = x * -0.9692660 + y * 1.8760108 + z * 0.0415560;
    let b_lin = x * 0.0556434 + y * -0.2040259 + z * 1.0572252;
    
    let r = linear_to_srgb(r_lin).clamp(0.0, 1.0) * 255.0;
    let g = linear_to_srgb(g_lin).clamp(0.0, 1.0) * 255.0;
    let b = linear_to_srgb(b_lin).clamp(0.0, 1.0) * 255.0;
    
    (r as u8, g as u8, b as u8)
}

/// Convertit LCH vers RGB (via Lab et XYZ)
pub fn lch_to_rgb(lch: Lch) -> (u8, u8, u8) {
    let (l, a, b) = lch_to_lab(lch);
    let (x, y, z) = lab_to_xyz(l, a, b);
    xyz_to_rgb(x, y, z)
}

// Fonctions helper pour conversions sRGB

fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f64) -> f64 {
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn lab_f(t: f64) -> f64 {
    if t > 0.008856 {
        t.cbrt()
    } else {
        (7.787 * t) + (16.0 / 116.0)
    }
}

fn lab_f_inv(t: f64) -> f64 {
    if t > 0.008856 {
        t.powi(3)
    } else {
        (t - 16.0 / 116.0) / 7.787
    }
}

/// Interpole entre deux couleurs HSB
pub fn interpolate_hsb(hsb1: Hsb, hsb2: Hsb, t: f64) -> Hsb {
    // Interpolation de la teinte en tenant compte de la circularité (0-360)
    let h_diff = hsb2.h - hsb1.h;
    let h_short = if h_diff.abs() > 180.0 {
        if h_diff > 0.0 {
            h_diff - 360.0
        } else {
            h_diff + 360.0
        }
    } else {
        h_diff
    };
    let h = (hsb1.h + h_short * t) % 360.0;
    let h = if h < 0.0 { h + 360.0 } else { h };
    
    Hsb {
        h,
        s: hsb1.s + (hsb2.s - hsb1.s) * t,
        b: hsb1.b + (hsb2.b - hsb1.b) * t,
    }
}

/// Interpole entre deux couleurs LCH
pub fn interpolate_lch(lch1: Lch, lch2: Lch, t: f64) -> Lch {
    // Interpolation de la teinte en tenant compte de la circularité (0-360)
    let h_diff = lch2.h - lch1.h;
    let h_short = if h_diff.abs() > 180.0 {
        if h_diff > 0.0 {
            h_diff - 360.0
        } else {
            h_diff + 360.0
        }
    } else {
        h_diff
    };
    let h = (lch1.h + h_short * t) % 360.0;
    let h = if h < 0.0 { h + 360.0 } else { h };
    
    Lch {
        l: lch1.l + (lch2.l - lch1.l) * t,
        c: lch1.c + (lch2.c - lch1.c) * t,
        h,
    }
}
