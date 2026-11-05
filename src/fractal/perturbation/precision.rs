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
        // `is_normal`, PAS `!= 0.0` : au-delà de ~1e308 le span f64 est
        // SOUS-NORMAL (1e321 → ~10 bits de mantisse, échelle fausse de ~0,2 %
        // ⇒ pixels de bord décalés de ~0,1 px : 16/1024 px faux vs GMP sur
        // liiiines, invisibles du traqueur de fiabilité — l'arithmétique est
        // juste, c'est l'ENTRÉE qui ne l'est pas). Sous-normal → string HP.
        if fallback.is_normal() {
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
    use crate::fractal::perturbation::dd::DoubleDoubleExp;

    // Exposant étendu via `from_gmp` : au-delà de ~1e308 le span HP
    // sous-dépassait `to_f64()` → repli `from_f64(span f64) = 0` → dc = 0
    // pour tous les pixels (tier dd uniforme à e1000, bug 2026-11-05).
    let from_hp_or_f64 = |hp: Option<&str>, fallback: f64| -> DoubleDoubleExp {
        if let Some(hp_value) = hp {
            if let Ok(raw) = Float::parse(hp_value) {
                let value = Float::with_val(1024, raw);
                if !value.is_zero() && value.is_finite() {
                    return DoubleDoubleExp::from_gmp(&value);
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
///
/// ⚠️ Appelée PAR PIXEL (`delta::try_bytecode_unified_path`, deux fois, dont
/// une via `bla_c_norm`). Tant que le span f64 est représentable c'est une
/// division ; au-delà de ~1e308 il vaut 0 et il faut repasser par les chaînes
/// HP — parse décimal GMP de centaines de chiffres + `ln` à 1024 bits, soit
/// plusieurs microsecondes. Répété à chaque pixel, c'était 45 % (e401) à 70 %
/// (e1000) du temps de la boucle pixel du tier exp : d'où le mémo ci-dessous.
pub(crate) fn effective_pixel_size(params: &FractalParams) -> f64 {
    if params.width == 0 || params.height == 0 {
        return 0.0;
    }
    let pixel_size_f64 =
        (params.span_x.abs() / params.width as f64).max(params.span_y.abs() / params.height as f64);
    if pixel_size_f64.is_finite() && pixel_size_f64 > 0.0 {
        return pixel_size_f64;
    }
    let Some(sx_hp) = params.span_x_hp.as_deref() else {
        return 0.0;
    };
    let sy_hp = params.span_y_hp.as_deref();

    // Mémo par thread, clé = CONTENU des chaînes HP + dimensions (jamais leur
    // adresse : un `String` libéré puis réalloué au même endroit mentirait).
    // La comparaison de contenu coûte un memcmp de quelques centaines d'octets,
    // contre un parse + ln GMP.
    HP_PIXEL_SIZE_MEMO.with(|cell| {
        let mut memo = cell.borrow_mut();
        if let Some(m) = memo.as_ref() {
            if m.width == params.width
                && m.height == params.height
                && m.span_x_hp == sx_hp
                && m.span_y_hp.as_deref() == sy_hp
            {
                return m.value;
            }
        }
        let value = pixel_size_from_hp(sx_hp, sy_hp.unwrap_or(sx_hp), params.width, params.height);
        *memo = Some(HpPixelSizeMemo {
            span_x_hp: sx_hp.to_owned(),
            span_y_hp: sy_hp.map(str::to_owned),
            width: params.width,
            height: params.height,
            value,
        });
        value
    })
}

struct HpPixelSizeMemo {
    span_x_hp: String,
    span_y_hp: Option<String>,
    width: u32,
    height: u32,
    value: f64,
}

thread_local! {
    static HP_PIXEL_SIZE_MEMO: std::cell::RefCell<Option<HpPixelSizeMemo>> =
        const { std::cell::RefCell::new(None) };
}

/// Chemin lent de [`effective_pixel_size`] : spans lus en GMP.
fn pixel_size_from_hp(sx_hp: &str, sy_hp: &str, width: u32, height: u32) -> f64 {
    let parse = |value: &str| -> Option<Float> {
        let raw = Float::parse(value).ok()?;
        Some(Float::with_val(1024, raw))
    };
    let (Some(sx), Some(sy)) = (parse(sx_hp), parse(sy_hp)) else {
        return 0.0;
    };
    let mut x_size = sx.abs();
    x_size /= Float::with_val(1024, width as f64);
    let mut y_size = sy.abs();
    y_size /= Float::with_val(1024, height as f64);
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

/// Zoom en notation scientifique (`1.02e401`) depuis son log2 — pour les
/// diagnostics : un f64 ne porte pas un zoom au-delà de 1e308. `None`
/// (zoom ≤ 1 ou vue dégénérée) → `0.00e0`, comme l'ancien affichage.
pub(crate) fn format_zoom_sci(log2_zoom: Option<f64>) -> String {
    let Some(l2) = log2_zoom.filter(|v| v.is_finite()) else {
        return "0.00e0".to_owned();
    };
    let l10 = l2 * std::f64::consts::LOG10_2;
    let mut exp = l10.floor();
    let mut mantissa = 10f64.powf(l10 - exp);
    // 9.996… s'arrondirait en « 10.00e49 ».
    if mantissa >= 9.995 {
        mantissa = 1.0;
        exp += 1.0;
    }
    format!("{mantissa:.2}e{}", exp as i64)
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

/// Marge ajoutée à la formule F3 `24 + floor(log2(zoom·height))` (G14.3,
/// 2026-11-05). Mesuré à 96² contre une orbite à précision ×4 (juge
/// convergé : ×2 == ×4 au pixel près) : à la précision F3 exacte, des scènes
/// du corpus ont 2 à 4 % de pixels FAUX quelle que soit l'arithmétique de
/// l'orbite (golden_spider 319-357 px, e227 226-229, evolution_trees
/// 241-250, dragon 6-8) ; avec +64 b elles rejoignent la vérité (0 px, 1 sur
/// e227). Coût ~+10 % sur l'orbite à 500-800 b, plus que compensé par le
/// fast-path d'itération (`orbit.rs::fast_sqr_scratch`, −35 %). Sans effet
/// sous ~1e70 (plancher `precision_bits` = 256 b).
pub(crate) const ORBIT_PRECISION_MARGIN_BITS: u64 = 64;

pub(crate) fn compute_perturbation_precision_bits(params: &FractalParams) -> u32 {
    if params.width == 0 || params.height == 0 {
        return params.engine.precision_bits.max(128);
    }
    let log2_zoom = match log2_zoom(params) {
        Some(value) => value,
        None => return params.engine.precision_bits.max(128),
    };

    let final_bits = if params.perturbation.use_reference_precision_formula {
        let log2_height = (params.height as f64).max(1.0).log2();
        let exp = (log2_zoom + log2_height).floor() as i64;
        let bits =
            if exp >= 0 { (24 + exp) as i64 } else { 24 } as u64 + ORBIT_PRECISION_MARGIN_BITS;
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

    final_bits.max(
        params
            .engine
            .precision_bits
            .clamp(128, MAX_PERTURB_PRECISION_BITS),
    )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::{default_params_for_type, FractalType};

    /// Span f64 nul, vérité dans les chaînes HP (le chemin lent). Les
    /// profondeurs du test restent représentables en f64 pour que les valeurs
    /// se DISTINGUENT : au-delà de 1e308 tout rend la même sentinelle
    /// `f64::MIN_POSITIVE` et un mémo collant passerait inaperçu.
    fn deep_params(span_hp: &str, width: u32, height: u32) -> FractalParams {
        let mut p = default_params_for_type(FractalType::Mandelbrot, width, height);
        p.span_x = 0.0;
        p.span_y = 0.0;
        p.span_x_hp = Some(span_hp.to_owned());
        p.span_y_hp = Some(span_hp.to_owned());
        p
    }

    /// Verrou du mémo de `effective_pixel_size` : la valeur mémorisée est celle
    /// du calcul GMP direct, et TOUT changement de la clé (chaîne HP, largeur,
    /// hauteur, absence du span y) la recalcule. Un mémo trop collant donnerait
    /// la taille de pixel de la vue précédente — tier et rayon BLA faux.
    #[test]
    fn hp_pixel_size_memo_follows_its_key() {
        let a = deep_params("4e-200", 256, 256);
        let direct_a = pixel_size_from_hp("4e-200", "4e-200", 256, 256);
        assert!(direct_a > 0.0);
        assert_eq!(effective_pixel_size(&a), direct_a);
        assert_eq!(effective_pixel_size(&a), direct_a, "second appel = mémo");

        // Autre profondeur, même longueur de chaîne.
        let b = deep_params("4e-250", 256, 256);
        let direct_b = pixel_size_from_hp("4e-250", "4e-250", 256, 256);
        assert_ne!(direct_a, direct_b);
        assert_eq!(effective_pixel_size(&b), direct_b);

        // Mêmes chaînes, autre taille d'image.
        let c = deep_params("4e-250", 512, 128);
        assert_eq!(
            effective_pixel_size(&c),
            pixel_size_from_hp("4e-250", "4e-250", 512, 128)
        );

        // Span y absent → repli sur x ; span y différent → il compte.
        let mut d = deep_params("4e-250", 512, 128);
        d.span_y_hp = None;
        assert_eq!(
            effective_pixel_size(&d),
            pixel_size_from_hp("4e-250", "4e-250", 512, 128)
        );
        d.span_y_hp = Some("4e-230".to_owned());
        assert_eq!(
            effective_pixel_size(&d),
            pixel_size_from_hp("4e-250", "4e-230", 512, 128)
        );

        // Retour à la première vue : pas de valeur périmée.
        assert_eq!(effective_pixel_size(&a), direct_a);
    }

    #[test]
    fn zoom_is_printed_beyond_the_f64_range() {
        assert_eq!(format_zoom_sci(None), "0.00e0");
        assert_eq!(format_zoom_sci(Some(50.0 / std::f64::consts::LOG10_2)), "1.00e50");
        // 1.0167e401 : hors plage f64.
        let l2 = (401.0 + 1.0167f64.log10()) / std::f64::consts::LOG10_2;
        assert_eq!(format_zoom_sci(Some(l2)), "1.02e401");
        // Mantisse qui s'arrondirait à 10.00.
        let l2 = (49.0 + 9.999f64.log10()) / std::f64::consts::LOG10_2;
        assert_eq!(format_zoom_sci(Some(l2)), "1.00e50");
        // Bout en bout sur une vue HP.
        let p = deep_params("4e-400", 256, 256);
        assert_eq!(format_zoom_sci(log2_zoom(&p)), "2.56e402");
    }

    /// Sans span HP et sans span f64 : zéro, sans passer par le mémo.
    #[test]
    fn pixel_size_without_any_span_is_zero() {
        let mut p = default_params_for_type(FractalType::Mandelbrot, 64, 64);
        p.span_x = 0.0;
        p.span_y = 0.0;
        p.span_x_hp = None;
        p.span_y_hp = None;
        assert_eq!(effective_pixel_size(&p), 0.0);
    }
}
