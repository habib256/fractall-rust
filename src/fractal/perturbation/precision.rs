//! Politique de précision et sélection du fallback GMP de la perturbation.

use crate::fractal::FractalParams;
use rug::Float;

/// Spans HP convertis en FloatExp sans perdre les magnitudes sous la plage f64.
pub(crate) fn effective_spans_fexp(
    params: &FractalParams,
) -> (
    crate::fractal::perturbation::types::FloatExp,
    crate::fractal::perturbation::types::FloatExp,
) {
    use crate::fractal::perturbation::types::FloatExp;

    let from_hp_or_f64 = |hp: Option<&str>, fallback: f64| -> FloatExp {
        if fallback.is_finite() && fallback != 0.0 {
            return FloatExp::from_f64(fallback);
        }
        let Some(hp_value) = hp else {
            return FloatExp::from_f64(fallback);
        };
        let Ok(raw) = Float::parse(hp_value) else {
            return FloatExp::from_f64(fallback);
        };
        let value = Float::with_val(1024, raw);
        if value.is_zero() || !value.is_finite() {
            return FloatExp::from_f64(fallback);
        }

        let log2 = value.clone().abs().ln() / Float::with_val(1024, 2.0f64.ln());
        let exponent = log2.to_f64().floor();
        let mantissa_value = if exponent.abs() < 1023.0 {
            let power = Float::with_val(1024, 2.0f64.powf(exponent));
            value / power
        } else {
            let mut mantissa = value;
            let mut remaining = -exponent;
            while remaining.abs() >= 1000.0 {
                let step = if remaining > 0.0 { 1000.0 } else { -1000.0 };
                mantissa *= Float::with_val(1024, 2.0f64.powf(step));
                remaining -= step;
            }
            if remaining != 0.0 {
                mantissa *= Float::with_val(1024, 2.0f64.powf(remaining));
            }
            mantissa
        };
        let mantissa = mantissa_value.to_f64();
        if !mantissa.is_finite() || mantissa == 0.0 {
            return FloatExp::from_f64(fallback);
        }
        FloatExp::new(
            mantissa,
            exponent.clamp(i32::MIN as f64, i32::MAX as f64) as i32,
        )
    };

    (
        from_hp_or_f64(params.span_x_hp.as_deref(), params.span_x),
        from_hp_or_f64(params.span_y_hp.as_deref(), params.span_y),
    )
}

/// Spans haute précision convertis vers le tier double-double exponentiel.
pub(crate) fn effective_spans_dd(
    params: &FractalParams,
) -> (
    crate::fractal::perturbation::dd::DoubleDoubleExp,
    crate::fractal::perturbation::dd::DoubleDoubleExp,
) {
    use crate::fractal::perturbation::dd::{DoubleDouble, DoubleDoubleExp};

    let from_hp_or_f64 = |hp: Option<&str>, fallback: f64| -> DoubleDoubleExp {
        if let Some(hp_value) = hp {
            if let Ok(raw) = Float::parse(hp_value) {
                let value = Float::with_val(1024, raw);
                if !value.is_zero() && value.is_finite() {
                    let hi = value.to_f64();
                    if hi != 0.0 && hi.is_finite() {
                        let mut remainder = value;
                        remainder -= hi;
                        return DoubleDoubleExp::normalized(
                            DoubleDouble::new(hi, remainder.to_f64()),
                            0,
                        );
                    }
                }
            }
        }
        DoubleDoubleExp::from_f64(fallback)
    };

    (
        from_hp_or_f64(params.span_x_hp.as_deref(), params.span_x),
        from_hp_or_f64(params.span_y_hp.as_deref(), params.span_y),
    )
}

/// Taille de pixel effective, reconstruite depuis les spans HP lorsque le f64
/// sous-flue. Une sentinelle positive représente les profondeurs au-delà de la
/// plage f64 afin que les comparaisons de seuil restent correctes.
pub(crate) fn effective_pixel_size(params: &FractalParams) -> f64 {
    if params.width == 0 || params.height == 0 {
        return 0.0;
    }
    let pixel_size_f64 =
        (params.span_x.abs() / params.width as f64).max(params.span_y.abs() / params.height as f64);
    if pixel_size_f64.is_finite() && pixel_size_f64 > 0.0 {
        return pixel_size_f64;
    }

    let parse = |value: &str| -> Option<Float> {
        let raw = Float::parse(value).ok()?;
        Some(Float::with_val(1024, raw))
    };
    let sx_hp = params.span_x_hp.as_deref();
    let sy_hp = params.span_y_hp.as_deref().or(sx_hp);
    let (Some(sx), Some(sy)) = (sx_hp.and_then(parse), sy_hp.and_then(parse)) else {
        return 0.0;
    };
    let mut x_size = sx.abs();
    x_size /= Float::with_val(1024, params.width as f64);
    let mut y_size = sy.abs();
    y_size /= Float::with_val(1024, params.height as f64);
    let pixel = if x_size > y_size { x_size } else { y_size };
    if pixel.is_zero() || !pixel.is_finite() {
        return 0.0;
    }

    let log2 = pixel.ln() / Float::with_val(1024, 2.0f64.ln());
    let log2_f64 = log2.to_f64();
    if !log2_f64.is_finite() {
        return 0.0;
    }
    let result = 2.0_f64.powf(log2_f64);
    if result > 0.0 && result.is_finite() {
        result
    } else {
        f64::MIN_POSITIVE
    }
}

/// `log2(zoom)` HP-aware, avec `zoom = 4 / pixel_size`.
pub(crate) fn log2_zoom(params: &FractalParams) -> Option<f64> {
    if params.width == 0 || params.height == 0 {
        return None;
    }
    let log2_from_f64 = || -> Option<f64> {
        let pixel_size = (params.span_x.abs() / params.width as f64)
            .max(params.span_y.abs() / params.height as f64);
        if !pixel_size.is_finite() || pixel_size <= 0.0 {
            return None;
        }
        let zoom = 4.0 / pixel_size;
        (zoom.is_finite() && zoom > 1.0).then(|| zoom.log2())
    };
    let log2_from_hp = || -> Option<f64> {
        let sx = params.span_x_hp.as_deref()?;
        let sy = params.span_y_hp.as_deref().unwrap_or(sx);
        let parse = |value: &str| -> Option<Float> {
            let raw = Float::parse(value).ok()?;
            Some(Float::with_val(1024, raw))
        };
        let mut x_size = parse(sx)?.abs();
        x_size /= Float::with_val(1024, params.width as f64);
        let mut y_size = parse(sy)?.abs();
        y_size /= Float::with_val(1024, params.height as f64);
        let pixel = if x_size > y_size { x_size } else { y_size };
        if pixel.is_zero() || !pixel.is_finite() {
            return None;
        }
        let mut zoom = Float::with_val(1024, 4.0);
        zoom /= pixel;
        if zoom <= 1.0 {
            return None;
        }
        Some((zoom.ln() / Float::with_val(1024, 2.0f64.ln())).to_f64())
    };

    match log2_from_hp().or_else(log2_from_f64) {
        Some(value) if value.is_finite() && value > 0.0 => Some(value),
        _ => None,
    }
}

/// Borne technique imposée par le type de précision de `rug`, pas un plafond
/// fonctionnel choisi par le moteur.
pub(crate) const MAX_PERTURB_PRECISION_BITS: u32 = u32::MAX;

pub(crate) fn compute_perturbation_precision_bits(params: &FractalParams) -> u32 {
    if params.width == 0 || params.height == 0 {
        return params.precision_bits.max(128);
    }
    let log2_zoom = match log2_zoom(params) {
        Some(value) => value,
        None => return params.precision_bits.max(128),
    };

    let final_bits = if params.perturbation.use_reference_precision_formula {
        let log2_height = (params.height as f64).max(1.0).log2();
        let exp = (log2_zoom + log2_height).floor() as i64;
        let bits = if exp >= 0 { (24 + exp) as i64 } else { 24 } as u64;
        if bits > MAX_PERTURB_PRECISION_BITS as u64 {
            eprintln!(
                "[PRECISION] ⚠ zoom requiert ~{} bits > max u32 rug/MPFR : précision \
                 saturée au maximum du type.",
                bits
            );
        }
        bits.clamp(128, MAX_PERTURB_PRECISION_BITS as u64) as u32
    } else {
        let zoom_bits = log2_zoom.ceil() as i64;
        let safety_margin: i64 = if log2_zoom > 100.0 {
            200
        } else if log2_zoom > 66.0 {
            160
        } else if log2_zoom > 50.0 {
            128
        } else if log2_zoom > 33.0 {
            96
        } else if log2_zoom > 20.0 {
            80
        } else {
            64
        };
        let needed_bits = (zoom_bits + safety_margin).max(128) as u64;
        needed_bits.clamp(128, MAX_PERTURB_PRECISION_BITS as u64) as u32
    };

    final_bits.max(params.precision_bits.clamp(128, MAX_PERTURB_PRECISION_BITS))
}

/// Détermine si le zoom impose le fallback GMP complet par pixel.
pub fn should_use_full_gmp_perturbation(params: &FractalParams) -> bool {
    if params.width == 0 || params.height == 0 {
        return false;
    }
    if matches!(
        std::env::var("FRACTALL_FORCE_GMP_PERTURB").as_deref(),
        Ok("1" | "true")
    ) {
        return true;
    }
    // Le pixel-loop bytecode couvre les tiers f64/exp et évite le fallback
    // GMP par pixel lorsqu'il sait compiler la formule.
    if crate::fractal::perturbation::delta::bytecode_path_label(params).is_some() {
        return false;
    }
    let pixel_size = effective_pixel_size(params);
    if pixel_size <= 0.0 {
        return false;
    }

    // ComplexExp couvre les magnitudes usuelles du corpus. GMP complet reste
    // le garde-fou des formules hors bytecode au-delà de ce seuil extrême.
    pixel_size < 1e-300
}
