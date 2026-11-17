//! Coordonnées décimales de la vue : affichage compact (barre d'état) et
//! chaînes exactes (copie presse-papiers, commande CLI).
//!
//! Tout passe par une décomposition décimale EXACTE de la chaîne HP (`-5.0…e-1`
//! comme `-0.743…` ou `8.7288…e-5`) en partie entière / partie fractionnaire :
//! on tronque des chiffres, on n'arrondit jamais — un chiffre affiché est
//! toujours un chiffre vrai de la coordonnée.

use rug::Float;

use crate::fractal::FractalParams;

/// Nombre minimal de décimales affichées dans la barre d'état.
pub const MIN_DISPLAY_FRACTION_DIGITS: usize = 6;
/// Plafond des décimales affichées : au-delà, la barre d'état déborderait
/// (deux nombres de ~100 chiffres à 1e100). La valeur complète reste dans
/// l'infobulle et la copie.
pub const MAX_DISPLAY_FRACTION_DIGITS: usize = 20;
/// Chiffres de garde au-delà de la résolution pixel pour le centre émis dans
/// la commande CLI (erreur de troncature ≤ 1e-20 pixel).
pub const CLI_GUARD_DIGITS: usize = 20;
/// Chiffres significatifs du zoom émis (erreur relative du span ≤ 5e-21).
pub const ZOOM_SIGNIFICANT_DIGITS: usize = 20;

/// Garde-fou contre une chaîne hostile (`1e999999999`) : au-delà, pas de
/// décomposition (on retombe sur la chaîne brute).
const MAX_DECIMAL_EXPONENT: i64 = 1_000_000;

/// Valeur décimale exacte : `±int_part.frac_part`.
/// Invariants : `int_part` sans zéro de tête (ou `"0"`), `frac_part` sans zéro
/// de queue, `negative == false` pour zéro.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decimal {
    pub negative: bool,
    pub int_part: String,
    pub frac_part: String,
}

impl Decimal {
    /// Décompose une chaîne décimale (notation simple ou scientifique).
    pub fn parse(text: &str) -> Option<Self> {
        let s = text.trim();
        let (negative, s) = match s.as_bytes().first()? {
            b'-' => (true, &s[1..]),
            b'+' => (false, &s[1..]),
            _ => (false, s),
        };
        let (mantissa, exponent) = match s.find(['e', 'E']) {
            Some(i) => (&s[..i], s[i + 1..].parse::<i64>().ok()?),
            None => (s, 0),
        };
        if exponent.abs() > MAX_DECIMAL_EXPONENT {
            return None;
        }
        let (int_m, frac_m) = match mantissa.find('.') {
            Some(i) => (&mantissa[..i], &mantissa[i + 1..]),
            None => (mantissa, ""),
        };
        if int_m.is_empty() && frac_m.is_empty() {
            return None;
        }
        if !int_m.bytes().chain(frac_m.bytes()).all(|b| b.is_ascii_digit()) {
            return None;
        }
        let digits = format!("{int_m}{frac_m}");
        let point = int_m.len() as i64 + exponent;
        let (int_part, frac_part) = if point <= 0 {
            (String::new(), format!("{}{digits}", "0".repeat((-point) as usize)))
        } else if point as usize >= digits.len() {
            (
                format!("{digits}{}", "0".repeat(point as usize - digits.len())),
                String::new(),
            )
        } else {
            let (a, b) = digits.split_at(point as usize);
            (a.to_string(), b.to_string())
        };
        Some(Self::normalized(negative, &int_part, &frac_part))
    }

    fn normalized(negative: bool, int_part: &str, frac_part: &str) -> Self {
        let int_part = int_part.trim_start_matches('0');
        let int_part = if int_part.is_empty() { "0" } else { int_part };
        let frac_part = frac_part.trim_end_matches('0');
        let is_zero = int_part == "0" && frac_part.is_empty();
        Self {
            negative: negative && !is_zero,
            int_part: int_part.to_string(),
            frac_part: frac_part.to_string(),
        }
    }

    /// Valeur tronquée (vers zéro) à `digits` décimales ; `true` si des
    /// chiffres non nuls ont été retirés.
    pub fn truncated(&self, digits: usize) -> (Self, bool) {
        if self.frac_part.len() <= digits {
            return (self.clone(), false);
        }
        (
            Self::normalized(self.negative, &self.int_part, &self.frac_part[..digits]),
            true,
        )
    }

    /// Valeur arrondie au plus proche (demi vers l'infini) à `digits`
    /// décimales. Réservé à la commande CLI, jamais à l'affichage : évite
    /// qu'un -1.7619 saisi revienne en « -1.76189999… » (expansion binaire
    /// MPFR tronquée).
    pub fn rounded(&self, digits: usize) -> Self {
        if self.frac_part.len() <= digits {
            return self.clone();
        }
        let round_up = self.frac_part.as_bytes()[digits] >= b'5';
        let mut all: Vec<u8> = format!("{}{}", self.int_part, &self.frac_part[..digits]).into_bytes();
        if round_up {
            let mut i = all.len();
            loop {
                if i == 0 {
                    all.insert(0, b'1');
                    break;
                }
                i -= 1;
                if all[i] == b'9' {
                    all[i] = b'0';
                } else {
                    all[i] += 1;
                    break;
                }
            }
        }
        let all = String::from_utf8(all).expect("chiffres ASCII");
        let split = all.len() - digits;
        Self::normalized(self.negative, &all[..split], &all[split..])
    }

    /// Écriture exacte : notation simple, sauf si elle exigerait plus de
    /// `sci_threshold` zéros de tête (|x| petit) ou chiffres entiers (|x|
    /// grand) — notation scientifique `d.ddd…e±N` alors. Relue à l'identique
    /// par `Decimal::parse` et par le CLI (`rug::Float::parse`).
    pub fn to_canonical_string(&self, sci_threshold: usize) -> String {
        let sign = if self.negative { "-" } else { "" };
        let leading_zeros = self.frac_part.len() - self.frac_part.trim_start_matches('0').len();
        let use_sci = (self.int_part == "0"
            && !self.frac_part.is_empty()
            && leading_zeros > sci_threshold)
            || self.int_part.len() > sci_threshold + 1;
        if !use_sci {
            return if self.frac_part.is_empty() {
                format!("{sign}{}", self.int_part)
            } else {
                format!("{sign}{}.{}", self.int_part, self.frac_part)
            };
        }
        // Chiffres significatifs + exposant de la première décimale.
        let (digits, exponent) = if self.int_part == "0" {
            (
                self.frac_part[leading_zeros..].to_string(),
                -(leading_zeros as i64) - 1,
            )
        } else {
            (
                format!("{}{}", self.int_part, self.frac_part),
                self.int_part.len() as i64 - 1,
            )
        };
        let digits = digits.trim_end_matches('0');
        let (head, tail) = digits.split_at(1);
        if tail.is_empty() {
            format!("{sign}{head}e{exponent}")
        } else {
            format!("{sign}{head}.{tail}e{exponent}")
        }
    }

    /// Affichage compact en notation simple : au plus `digits` décimales,
    /// troncature marquée par `…`.
    pub fn to_compact_string(&self, digits: usize) -> String {
        let (shown, cut) = self.truncated(digits);
        // Une valeur tronquée à zéro garde son signe et ses zéros : « -0.000000… »
        // dit « petit négatif », pas « zéro ».
        let sign = if self.negative { "-" } else { "" };
        let frac = if cut {
            let mut f = self.frac_part[..digits].to_string();
            if f.is_empty() {
                f.push('0');
            }
            f
        } else {
            shown.frac_part.clone()
        };
        let mut out = if frac.is_empty() {
            format!("{sign}{}", shown.int_part)
        } else {
            format!("{sign}{}.{frac}", shown.int_part)
        };
        if cut {
            out.push('…');
        }
        out
    }
}

/// Chaîne décimale de la coordonnée : HP si présente, sinon le f64 (écriture
/// la plus courte qui le relit exactement).
fn coordinate_string(hp: &Option<String>, fallback: f64) -> String {
    hp.clone().unwrap_or_else(|| format!("{fallback:e}"))
}

pub fn center_strings(params: &FractalParams) -> (String, String) {
    (
        coordinate_string(&params.center_x_hp, params.center_x),
        coordinate_string(&params.center_y_hp, params.center_y),
    )
}

/// Span horizontal en `Float` (HP si présente : un span < 1e-308 n'existe pas
/// en f64).
fn span_x_float(params: &FractalParams) -> Option<Float> {
    let text = coordinate_string(&params.span_x_hp, params.span_x);
    let probe = Float::with_val(64, Float::parse(&text).ok()?);
    let precision = 128u32.saturating_add(probe.get_exp().map_or(0, |e| e.unsigned_abs()));
    let span = Float::with_val(precision, Float::parse(&text).ok()?).abs();
    (span.is_finite() && !span.is_zero()).then_some(span)
}

/// Décimales nécessaires pour distinguer deux pixels voisins :
/// `ceil(-log10(span_x / width))` (≥ 0). `None` si le span est inutilisable.
pub fn pixel_fraction_digits(params: &FractalParams) -> Option<usize> {
    let span = span_x_float(params)?;
    let pixel = span / Float::with_val(64, params.width.max(1));
    let digits = (-pixel.log10().to_f64()).ceil();
    digits.is_finite().then_some(digits.max(0.0) as usize)
}

/// Décimales affichées dans la barre d'état pour la vue courante.
pub fn display_fraction_digits(params: &FractalParams) -> usize {
    pixel_fraction_digits(params)
        .unwrap_or(MIN_DISPLAY_FRACTION_DIGITS)
        .clamp(MIN_DISPLAY_FRACTION_DIGITS, MAX_DISPLAY_FRACTION_DIGITS)
}

fn compact(value: &str, digits: usize) -> String {
    Decimal::parse(value)
        .map(|d| d.to_compact_string(digits))
        .unwrap_or_else(|| value.to_string())
}

/// Texte de la barre d'état : `Center: (re, im)` compact.
pub fn format_center_status(params: &FractalParams) -> String {
    let (x, y) = center_strings(params);
    let digits = display_fraction_digits(params);
    format!("Center: ({}, {})", compact(&x, digits), compact(&y, digits))
}

/// Écriture exacte d'une coordonnée (zéros de queue retirés).
pub fn exact_coordinate(value: &str) -> String {
    Decimal::parse(value)
        .map(|d| d.to_canonical_string(20))
        .unwrap_or_else(|| value.trim().to_string())
}

/// Zoom `4 / span_x` (convention CLI `--zoom` / TOML `zoom`) à
/// `ZOOM_SIGNIFICANT_DIGITS` chiffres significatifs.
pub fn zoom_string(params: &FractalParams) -> String {
    let Some(span) = span_x_float(params) else {
        return "1".to_string();
    };
    let precision = span.prec();
    let zoom = Float::with_val(precision, 4) / span;
    let (negative, digits, exp) = zoom.to_sign_string_exp(10, Some(ZOOM_SIGNIFICANT_DIGITS));
    let Some(exp) = exp else {
        return "1".to_string();
    };
    let text = format!("{}0.{digits}e{exp}", if negative { "-" } else { "" });
    Decimal::parse(&text)
        .map(|d| d.to_canonical_string(6))
        .unwrap_or(text)
}

/// Texte copié au clic sur les coordonnées : TOML léger (format
/// rust-fractal-core, relu par `fractall-cli --toml`), centre en précision
/// complète.
pub fn full_precision_clipboard_text(params: &FractalParams) -> String {
    let (x, y) = center_strings(params);
    format!(
        "# fractall view: save as view.toml, then\n\
         # fractall-cli --toml view.toml --type {} --output fractal.png\n\
         real = \"{}\"\nimag = \"{}\"\nzoom = \"{}\"\niterations = {}\n",
        params.fractal_type.id(),
        exact_coordinate(&x),
        exact_coordinate(&y),
        zoom_string(params),
        params.iteration_max,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::{default_params_for_type, FractalType};

    fn params_at(cx: &str, cy: &str, span_x: &str) -> FractalParams {
        let mut p = default_params_for_type(FractalType::Mandelbrot, 800, 600);
        p.center_x_hp = Some(cx.to_string());
        p.center_y_hp = Some(cy.to_string());
        p.span_x_hp = Some(span_x.to_string());
        p.center_x = cx.parse().unwrap();
        p.center_y = cy.parse().unwrap();
        p.span_x = span_x.parse().unwrap();
        p
    }

    #[test]
    fn decimal_parse_handles_plain_and_scientific() {
        let d = Decimal::parse("-5.000000000000000000000e-1").unwrap();
        assert_eq!((d.negative, &*d.int_part, &*d.frac_part), (true, "0", "5"));
        let d = Decimal::parse("8.7288e-5").unwrap();
        assert_eq!(d.frac_part, "000087288");
        let d = Decimal::parse("1.25E3").unwrap();
        assert_eq!((&*d.int_part, &*d.frac_part), ("1250", ""));
        let d = Decimal::parse("-0.000").unwrap();
        assert!(!d.negative);
        assert_eq!(d.to_canonical_string(20), "0");
        assert!(Decimal::parse("abc").is_none());
        assert!(Decimal::parse("1e99999999").is_none());
    }

    #[test]
    fn canonical_round_trips_exactly() {
        for s in [
            "-0.743643887037151007923796441482759390000",
            "8.72880000000000000000000000000000001e-5",
            "-1.5e-300",
            "123456789",
            "0",
        ] {
            let d = Decimal::parse(s).unwrap();
            let back = Decimal::parse(&d.to_canonical_string(20)).unwrap();
            assert_eq!(d, back, "{s}");
        }
        assert_eq!(Decimal::parse("-1.5e-300").unwrap().to_canonical_string(20), "-1.5e-300");
        assert_eq!(Decimal::parse("-5.0000e-1").unwrap().to_canonical_string(20), "-0.5");
    }

    #[test]
    fn rounding_carries_and_never_touches_short_values() {
        let r = |s: &str, n| Decimal::parse(s).unwrap().rounded(n).to_canonical_string(20);
        assert_eq!(r("-1.76189999999999999", 10), "-1.7619");
        assert_eq!(r("9.9996", 3), "10");
        assert_eq!(r("-0.00004", 4), "0");
        assert_eq!(r("0.12344", 4), "0.1234");
        assert_eq!(r("1.5", 10), "1.5");
    }

    #[test]
    fn default_view_is_short_plain_decimal() {
        // Chaîne telle que la GUI la produit (rug, 1024 b) pour -0.5.
        let cx = format!("-5.{}e-1", "0".repeat(300));
        let p = params_at(&cx, "0", "4");
        assert_eq!(format_center_status(&p), "Center: (-0.5, 0)");
    }

    #[test]
    fn f64_only_view_uses_mirrors() {
        let mut p = default_params_for_type(FractalType::Mandelbrot, 800, 600);
        p.center_x_hp = None;
        p.center_y_hp = None;
        p.span_x_hp = None;
        p.center_x = -0.743643887;
        p.center_y = 0.131825904;
        p.span_x = 4.0;
        assert_eq!(format_center_status(&p), "Center: (-0.743643…, 0.131825…)");
    }

    #[test]
    fn digits_follow_zoom_depth_at_1e15() {
        // span 4e-15 sur 800 px : pixel 5e-18 → 18 décimales.
        let x = "-0.743643887037151007923796441482759390001";
        let y = "0.131825904205311970493132056385139000001";
        let p = params_at(x, y, "4e-15");
        assert_eq!(
            format_center_status(&p),
            "Center: (-0.743643887037151007…, 0.131825904205311970…)"
        );
    }

    #[test]
    fn deep_zoom_is_capped_and_marked() {
        let x = format!("-1.86071995487372861{}", "7".repeat(80));
        let y = format!("8.7288{}e-5", "3".repeat(90));
        let p = params_at(&x, &y, "4e-64");
        assert_eq!(pixel_fraction_digits(&p), Some(67));
        assert_eq!(
            format_center_status(&p),
            "Center: (-1.86071995487372861777…, 0.00008728833333333333…)"
        );
    }

    #[test]
    fn span_below_f64_range_uses_hp_span() {
        let x = format!("0.{}12345", "0".repeat(330));
        let p = params_at(&x, "-0.25", "4e-300");
        assert_eq!(p.span_x, 4e-300);
        let mut p2 = p.clone();
        p2.span_x_hp = Some("4e-400".to_string());
        p2.span_x = 0.0; // sous-dépassement du miroir f64
        assert_eq!(pixel_fraction_digits(&p2), Some(403));
        assert_eq!(
            format_center_status(&p2),
            "Center: (0.00000000000000000000…, -0.25)"
        );
    }

    #[test]
    fn negative_values_truncate_toward_zero_without_rounding() {
        let p = params_at("-0.1999999999", "-1e-30", "4");
        // 0.1999999999 tronqué à 6 décimales : jamais « -0.2 ».
        assert_eq!(format_center_status(&p), "Center: (-0.199999…, -0.000000…)");
    }

    #[test]
    fn tiny_span_with_short_center_is_exact() {
        let p = params_at("-1.75", "0", "4e-50");
        assert_eq!(format_center_status(&p), "Center: (-1.75, 0)");
    }

    #[test]
    fn zoom_string_matches_cli_convention() {
        assert_eq!(zoom_string(&params_at("0", "0", "4")), "1");
        assert_eq!(zoom_string(&params_at("0", "0", "2.6666666666666666667")), "1.5");
        let z = zoom_string(&params_at("0", "0", "1.2732395447351626862e-64"));
        assert!(z.starts_with("3.14159265358979323"), "{z}");
        assert!(z.ends_with("e64"), "{z}");
        assert_eq!(zoom_string(&params_at("0", "0", "4e-5000")), "1e5000");
    }

    #[test]
    fn clipboard_text_is_loadable_light_toml() {
        let cx = format!("-5.{}e-1", "0".repeat(300));
        let p = params_at(&cx, "1.25e-3", "4e-10");
        let text = full_precision_clipboard_text(&p);
        let table: toml::Table = text.parse().unwrap();
        assert_eq!(table["real"].as_str(), Some("-0.5"));
        assert_eq!(table["imag"].as_str(), Some("0.00125"));
        assert_eq!(table["zoom"].as_str(), Some("1e10"));
        assert_eq!(table["iterations"].as_integer(), Some(p.iteration_max as i64));
    }

    #[test]
    fn center_status_keeps_hp_digits_up_to_pixel_resolution() {
        // Ancien verrou : la chaîne HP prime sur le miroir f64.
        let x = "-0.743643887037151007923796441482759390000";
        let y = "0.131825904205311970493132056385139000001";
        let p = params_at(x, y, "4");
        assert_eq!(format_center_status(&p), "Center: (-0.743643…, 0.131825…)");
        let p = params_at(x, y, "4e-30");
        assert_eq!(
            format_center_status(&p),
            "Center: (-0.74364388703715100792…, 0.13182590420531197049…)"
        );
    }

    #[test]
    fn center_status_reads_tiny_hp_with_zero_f64_mirrors() {
        let x = format!("0.{}123456789012345678900", "0".repeat(330));
        let y = format!("-0.{}987654321098765432100", "0".repeat(330));
        let mut p = params_at(&x, &y, "4");
        assert_eq!(p.center_x, 0.0);
        assert_eq!(p.center_y, 0.0);
        assert_eq!(format_center_status(&p), "Center: (0.000000…, -0.000000…)");
        // Copie pleine précision : rien de perdu.
        p.span_x_hp = Some("4e-340".to_string());
        let text = full_precision_clipboard_text(&p);
        assert!(text.contains("real = \"1.234567890123456789e-331\""), "{text}");
        assert!(text.contains("imag = \"-9.876543210987654321e-331\""), "{text}");
    }

    #[test]
    fn center_status_falls_back_per_coordinate_when_hp_pair_is_incomplete() {
        let mut p = default_params_for_type(FractalType::Mandelbrot, 800, 600);
        p.span_x_hp = None;
        p.span_x = 4.0;
        p.center_x = -0.743643887;
        p.center_y = 0.131825904;
        p.center_x_hp = Some("1.2345678901".to_string());
        p.center_y_hp = None;
        assert_eq!(format_center_status(&p), "Center: (1.234567…, 0.131825…)");
        p.center_x_hp = None;
        p.center_y_hp = Some("-9.876543210".to_string());
        assert_eq!(format_center_status(&p), "Center: (-0.743643…, -9.876543…)");
        p.center_x = 12345.0;
        p.center_y_hp = None;
        p.center_y = -0.125;
        assert_eq!(format_center_status(&p), "Center: (12345, -0.125)");
    }
}
