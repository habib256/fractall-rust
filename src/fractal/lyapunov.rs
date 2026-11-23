//! Algorithme de fractale de Lyapunov (Zircon City).
//!
//! Calcule l'exposant de Lyapunov de la carte logistique x_{n+1} = r * x * (1 - x)
//! où r alterne entre les paramètres a et b selon une séquence prédéfinie.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use num_complex::Complex64;
use rayon::prelude::*;
use rug::{Assign, Float};
use serde::{Deserialize, Serialize};

use crate::fractal::FractalParams;

/// Preset de Lyapunov prédéfini.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LyapunovPreset {
    /// Standard (Swallow): séquence "AB", plage [2.0, 4.0] x [2.0, 4.0]
    Standard,
    /// Zircon City: séquence "BBBBBBAAAAAA", plage A=[2.5, 3.4], B=[3.4, 4.0]
    #[default]
    ZirconCity,
    /// Jellyfish: séquence "AABAB", plage [3.4, 4.0] x [3.4, 4.0]
    Jellyfish,
    /// Asymmetric: séquence "AAB", plage [2.0, 4.0] x [2.0, 4.0]
    Asymmetric,
    /// Spaceship: séquence "ABBBA", plage [3.5, 3.9] x [3.5, 3.9]
    Spaceship,
    /// Heavy Blocks: séquence "BBBBBBBBAAAAAAAAA", plage A=[2.5, 3.4], B=[3.4, 4.0]
    HeavyBlocks,
}

impl LyapunovPreset {
    /// Retourne tous les presets disponibles.
    #[allow(dead_code)]
    pub fn all() -> &'static [LyapunovPreset] {
        &[
            LyapunovPreset::Standard,
            LyapunovPreset::ZirconCity,
            LyapunovPreset::Jellyfish,
            LyapunovPreset::Asymmetric,
            LyapunovPreset::Spaceship,
            LyapunovPreset::HeavyBlocks,
        ]
    }

    /// Retourne le nom lisible du preset.
    #[allow(dead_code)]
    pub fn name(self) -> &'static str {
        match self {
            LyapunovPreset::Standard => "Standard (Swallow)",
            LyapunovPreset::ZirconCity => "Zircon City",
            LyapunovPreset::Jellyfish => "Jellyfish",
            LyapunovPreset::Asymmetric => "Asymmetric",
            LyapunovPreset::Spaceship => "Spaceship",
            LyapunovPreset::HeavyBlocks => "Heavy Blocks",
        }
    }

    /// Retourne le nom en format CLI (minuscules avec tirets).
    #[allow(dead_code)]
    pub fn cli_name(self) -> &'static str {
        match self {
            LyapunovPreset::Standard => "standard",
            LyapunovPreset::ZirconCity => "zircon-city",
            LyapunovPreset::Jellyfish => "jellyfish",
            LyapunovPreset::Asymmetric => "asymmetric",
            LyapunovPreset::Spaceship => "spaceship",
            LyapunovPreset::HeavyBlocks => "heavy-blocks",
        }
    }

    /// Parse un nom CLI vers un preset.
    #[allow(dead_code)]
    pub fn from_cli_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "standard" => Some(LyapunovPreset::Standard),
            "zircon-city" | "zirconcity" => Some(LyapunovPreset::ZirconCity),
            "jellyfish" => Some(LyapunovPreset::Jellyfish),
            "asymmetric" => Some(LyapunovPreset::Asymmetric),
            "spaceship" => Some(LyapunovPreset::Spaceship),
            "heavy-blocks" | "heavyblocks" => Some(LyapunovPreset::HeavyBlocks),
            _ => None,
        }
    }
}

/// Configuration complète d'un preset Lyapunov.
#[derive(Clone, Debug)]
pub struct LyapunovConfig {
    #[allow(dead_code)]
    pub name: &'static str,
    pub sequence: Vec<bool>,
    pub xmin: f64,
    pub xmax: f64,
    pub ymin: f64,
    pub ymax: f64,
}

impl LyapunovConfig {
    /// Crée la configuration pour un preset donné.
    pub fn from_preset(preset: LyapunovPreset) -> Self {
        match preset {
            LyapunovPreset::Standard => LyapunovConfig {
                name: "Standard (Swallow)",
                sequence: sequence_from_str("AB"),
                xmin: 2.0,
                xmax: 4.0,
                ymin: 2.0,
                ymax: 4.0,
            },
            LyapunovPreset::ZirconCity => LyapunovConfig {
                name: "Zircon City",
                sequence: sequence_from_str("BBBBBBAAAAAA"),
                xmin: 2.5,
                xmax: 3.4,
                ymin: 3.4,
                ymax: 4.0,
            },
            LyapunovPreset::Jellyfish => LyapunovConfig {
                name: "Jellyfish",
                sequence: sequence_from_str("AABAB"),
                xmin: 3.4,
                xmax: 4.0,
                ymin: 3.4,
                ymax: 4.0,
            },
            LyapunovPreset::Asymmetric => LyapunovConfig {
                name: "Asymmetric",
                sequence: sequence_from_str("AAB"),
                xmin: 2.0,
                xmax: 4.0,
                ymin: 2.0,
                ymax: 4.0,
            },
            LyapunovPreset::Spaceship => LyapunovConfig {
                name: "Spaceship",
                sequence: sequence_from_str("ABBBA"),
                xmin: 3.5,
                xmax: 3.9,
                ymin: 3.5,
                ymax: 3.9,
            },
            LyapunovPreset::HeavyBlocks => LyapunovConfig {
                name: "Heavy Blocks",
                sequence: sequence_from_str("BBBBBBBBAAAAAAAAA"),
                xmin: 2.5,
                xmax: 3.4,
                ymin: 3.4,
                ymax: 4.0,
            },
        }
    }
}

/// Convertit une chaîne "AB..." en séquence de booléens.
/// 'A' -> true (utilise paramètre a), 'B' -> false (utilise paramètre b)
pub fn sequence_from_str(s: &str) -> Vec<bool> {
    s.chars().map(|c| c == 'A' || c == 'a').collect()
}

/// Séquence par défaut pour Zircon City: "BBBBBBAAAAAA"
const ZIRCON_SEQUENCE: &[bool] = &[
    false, false, false, false, false, false, // 6 B's (use b)
    true, true, true, true, true, true, // 6 A's (use a)
];

/// Constantes pour l'algorithme
const WARMUP_ITERATIONS: usize = 50;

fn parameter_sequence(sequence: &[bool]) -> impl Iterator<Item = bool> + Clone + '_ {
    let sequence = if sequence.is_empty() {
        ZIRCON_SEQUENCE
    } else {
        sequence
    };
    sequence.iter().copied().cycle()
}

/// Huit orbites voisines partagent la séquence A/B. Leurs récurrences sont
/// indépendantes, ce qui permet de vectoriser sans réordonner les produits.
const LYAPUNOV_LANES: usize = 8;

fn compute_lyapunov_exponent(a: f64, b: f64, iter_max: u32, sequence: &[bool]) -> f64 {
    compute_lyapunov_exponents([a], [b], iter_max, sequence)[0]
}

fn compute_lyapunov_exponents<const N: usize>(
    a: [f64; N],
    b: [f64; N],
    iter_max: u32,
    sequence: &[bool],
) -> [f64; N] {
    // Pour 0 ≤ r ≤ 4, la carte préserve [0, 1]. Même à r = 4, le
    // complément est exact pour x ≥ 1/2 ; en dessous, son arrondi déplace
    // le produit de moins d'une demi-ulp de 1. Les vues hors de cet
    // intervalle conservent le contrôle à chaque pas.
    if a.iter().chain(&b).all(|v| (0.0..=4.0).contains(v)) {
        iterate_logistic::<N, false>(a, b, iter_max, sequence)
    } else {
        iterate_logistic::<N, true>(a, b, iter_max, sequence)
    }
}

fn iterate_logistic<const N: usize, const CHECK_DOMAIN: bool>(
    a: [f64; N],
    b: [f64; N],
    iter_max: u32,
    sequence: &[bool],
) -> [f64; N] {
    let mut parameters = parameter_sequence(sequence).map(|is_a| if is_a { a } else { b });
    let mut x = [0.5; N];
    let advance = |x: [f64; N], r: [f64; N]| {
        std::array::from_fn(|i| {
            let next = r[i] * x[i] * (1.0 - x[i]);
            if CHECK_DOMAIN && !(0.0..=1.0).contains(&next) {
                0.5
            } else {
                next
            }
        })
    };

    for r in parameters.by_ref().take(WARMUP_ITERATIONS) {
        x = advance(x, r);
    }

    let mut product = [1.0_f64; N];
    let mut sum = [0.0; N];
    let derivatives = |x: [f64; N], r: [f64; N]| -> [f64; N] {
        std::array::from_fn(|i| (r[i] * (1.0 - 2.0 * x[i])).abs())
    };
    let checked_step = |x, r, product: &mut [f64; N], sum: &mut [f64; N]| {
        let derivative = derivatives(x, r);
        let next: [f64; N] = std::array::from_fn(|i| product[i] * derivative[i]);
        // Le ET sans court-circuit permet au compilateur de tester toutes
        // les voies ensemble. Une voie hors plage reprend ses deux facteurs.
        if next.iter().fold(true, |normal, v| normal & v.is_normal()) {
            *product = next;
        } else {
            for i in 0..N {
                if next[i].is_normal() {
                    product[i] = next[i];
                } else {
                    sum[i] += product[i].ln() + derivative[i].ln();
                    product[i] = 1.0;
                }
            }
        }
    };
    // Dans [0, 4], |r·(1−2x)| ≤ 4 à tout pas. Un produit sous-normal
    // ne peut donc pas remonter à MIN après au plus BLOCK multiplications,
    // même arrondies (multiplier par 4 est exact). Un débordement reste
    // infini ou NaN. Le test en fin de bloc couvre ainsi tous ses pas.
    const BLOCK: usize = 16;
    const MIN: f64 = f64::MIN_POSITIVE * (1u64 << (2 * BLOCK)) as f64;
    let mut remaining = iter_max as usize;
    if !CHECK_DOMAIN {
        while remaining >= BLOCK {
            let start = (x, product, parameters.clone());
            for r in parameters.by_ref().take(BLOCK) {
                let derivative = derivatives(x, r);
                product = std::array::from_fn(|i| product[i] * derivative[i]);
                x = advance(x, r);
            }
            if !product.iter().fold(true, |ok, v| ok & (*v >= MIN) & v.is_finite()) {
                // Retrouver le pas précis qui nécessite le passage au logarithme.
                let (mut prev_x, prev_product, steps) = start;
                product = prev_product;
                for r in steps.take(BLOCK) {
                    checked_step(prev_x, r, &mut product, &mut sum);
                    prev_x = advance(prev_x, r);
                }
                // Une dérivée exactement nulle fixe l'exposant à −∞.
                if sum.iter().all(|&value| value == f64::NEG_INFINITY) {
                    return [f64::NEG_INFINITY; N];
                }
            }
            remaining -= BLOCK;
        }
    }
    for r in parameters.take(remaining) {
        checked_step(x, r, &mut product, &mut sum);
        x = advance(x, r);
    }
    std::array::from_fn(|i| (sum[i] + product[i].ln()) / iter_max as f64)
}

/// Normalise l'exposant de Lyapunov vers [0, 1] pour la colorisation.
///
/// - Exposants négatifs (stabilité) -> [0, 0.85)
/// - Exposants positifs (chaos) -> [0.85, 1.0]
fn normalize_lyapunov(lyap: f64) -> f64 {
    if lyap < 0.0 {
        // Région stable
        let t = (-lyap).min(2.0);
        (t / 2.0) * 0.85
    } else {
        // Région chaotique
        let t = lyap.min(1.0);
        0.85 + t * 0.15
    }
}

fn compute_lyapunov_exponent_mpc(
    a: &Float,
    b: &Float,
    iter_max: u32,
    sequence: &[bool],
    prec: u32,
) -> Float {
    let twice_a = a.as_shl(1);
    let twice_b = b.as_shl(1);
    let mut parameters =
        parameter_sequence(sequence).map(|is_a| if is_a { (a, &*twice_a) } else { (b, &*twice_b) });
    let mut x = Float::with_val(prec, 0.5);
    let half = x.clone();
    let bounded = *a >= 0 && *a <= 4 && *b >= 0 && *b <= 4;
    let one = Float::with_val(prec, 1);
    let mut complement = Float::new(prec);
    let mut advance = |x: &mut Float, r: &Float| {
        complement.assign(&one - &*x);
        *x *= &complement;
        *x *= r;
        if !bounded && !(*x >= 0 && *x <= 1) {
            x.assign(0.5);
        }
    };

    for (r, _) in parameters.by_ref().take(WARMUP_ITERATIONS) {
        advance(&mut x, r);
    }

    // Séparer mantisse et exposant évite le sous-flux et ne demande qu’un
    // logarithme multiprécision du produit, même pour les longues orbites.
    let mut product = Float::with_val(prec, 1);
    let mut binary_exponent = 0_i64;
    let mut derivative = Float::new(prec);
    let min_exp = i64::from(rug::float::exp_min());
    let max_exp = i64::from(rug::float::exp_max());
    // 2r·(1/2−x) a le même arrondi que r·(1−2x) si 2r est
    // représentable et si la soustraction intermédiaire ne sous-flue pas.
    // L'écart non nul minimal près de 1/2 est 2^(-prec-1).
    let use_doubled = twice_a.is_finite() && twice_b.is_finite() && -i64::from(prec) >= min_exp;
    let evaluate = |derivative: &mut Float, x: &Float, r: &Float, twice_r: &Float| {
        if use_doubled {
            derivative.assign(&half - x);
            *derivative *= twice_r;
        } else {
            derivative.assign(x * -2);
            *derivative += &one;
            *derivative *= r;
        }
    };
    let mut multiply = |product: &mut Float, derivative: &Float| {
        // Le produit des deux mantisses est dans [1/4, 1). Ne séparer
        // leurs exposants que si le produit risque de sortir du domaine MPFR.
        let de = derivative.get_exp().unwrap_or(0);
        let exp = i64::from(product.get_exp().unwrap_or(0)) + i64::from(de);
        if exp <= min_exp || exp >= max_exp {
            binary_exponent += i64::from(product.frexp_mut()) + i64::from(de);
            *product *= &*derivative.as_shr(de);
        } else {
            *product *= derivative;
        }
    };
    let mut remaining = iter_max as usize;
    let period = if sequence.is_empty() {
        ZIRCON_SEQUENCE.len()
    } else {
        sequence.len()
    };
    // Quatre passages de la séquence couvrent aussi les deux premiers
    // doublements de période. Limiter le cache des séquences personnalisées
    // à 128 facteurs et à 64 Kio de mantisses par orbite.
    let block = if period <= 32 && 4 * period as u64 * u64::from(prec) <= 64 * 1024 * 8 {
        4 * period
    } else {
        remaining
    };
    let mut checkpoint = Float::new(prec);
    while remaining > 0 {
        checkpoint.assign(&x);
        let count = block.min(remaining);
        for (r, twice_r) in parameters.by_ref().take(count) {
            evaluate(&mut derivative, &x, r, twice_r);
            if derivative.is_zero() {
                return Float::with_val(prec, f64::NEG_INFINITY);
            }
            multiply(&mut product, &derivative);
            advance(&mut x, r);
        }
        remaining -= count;
        if remaining > block && x.as_ord() == checkpoint.as_ord() {
            // Même état MPFR à la même phase A/B : toutes les dérivées
            // suivantes se répètent exactement. Garder les multiplications
            // dans l'ordre initial, même si le dernier bloc est partiel.
            let mut cycle = Vec::with_capacity(block);
            for (r, twice_r) in parameters.by_ref().take(block) {
                evaluate(&mut derivative, &x, r, twice_r);
                cycle.push(derivative.clone());
                advance(&mut x, r);
            }
            for factor in cycle.iter().cycle().take(remaining) {
                multiply(&mut product, factor);
            }
            break;
        }
    }
    product.abs_mut();
    binary_exponent += i64::from(product.frexp_mut());
    product.ln_mut();
    let mut scale = Float::with_val(prec, rug::float::Constant::Log2);
    scale *= binary_exponent;
    product += scale;
    product /= iter_max;
    product
}

/// Version annulable du rendu Lyapunov.
pub fn render_lyapunov_cancellable(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    let width = params.width as usize;
    let height = params.height as usize;
    let mut iterations = vec![0u32; width * height];
    let mut zs = vec![Complex64::new(0.0, 0.0); width * height];

    if width == 0 || height == 0 {
        return Some((iterations, zs));
    }

    // Utiliser span directement au lieu de xmax-xmin pour éviter les problèmes de précision
    let iter_max = params.iteration_max;
    let sequence = &params.formula.lyapunov_sequence;
    let [dx, dy] = params.sampling.aa_subpixel_offset;
    let jitter = params.sampling.aa_jitter.map(|(k, scale)| {
        crate::fractal::jitter::pixel_offsets(width, k, scale)
    });
    let transform = params.transform_matrix();

    let cancelled = AtomicBool::new(false);

    iterations
        .par_chunks_mut(width)
        .zip(zs.par_chunks_mut(width))
        .enumerate()
        .for_each(|(j, (iter_row, z_row))| {
            // Vérifier l'annulation toutes les 16 lignes
            if j % 16 == 0 {
                if cancel.load(Ordering::Relaxed) {
                    cancelled.store(true, Ordering::Relaxed);
                    return;
                }
            }
            if cancelled.load(Ordering::Relaxed) {
                return;
            }

            let point = |i: usize| {
                let (jx, jy) = jitter.as_ref().map_or((dx, dy), |offset| offset(i, j));
                let x_ratio = (i as f64 + jx) / params.width as f64;
                let y_ratio = (j as f64 + jy) / params.height as f64;
                let dx = (x_ratio - 0.5) * params.span_x;
                let dy = (y_ratio - 0.5) * params.span_y;
                let (dx, dy) = match transform {
                    Some((a, b, c, d)) => (a * dx + b * dy, c * dx + d * dy),
                    None => (dx, dy),
                };
                (params.center_x + dx, params.center_y + dy)
            };
            let mut store = |i: usize, lyap| {
                let norm = normalize_lyapunov(lyap);
                iter_row[i] = (norm * iter_max as f64) as u32;
                z_row[i] = Complex64::new(norm * 2.0, 0.0);
            };
            let full_width = width / LYAPUNOV_LANES * LYAPUNOV_LANES;
            for first in (0..full_width).step_by(LYAPUNOV_LANES) {
                let points: [_; LYAPUNOV_LANES] = std::array::from_fn(|i| point(first + i));
                let values = compute_lyapunov_exponents(
                    points.map(|p| p.0),
                    points.map(|p| p.1),
                    iter_max,
                    sequence,
                );
                for (i, lyap) in values.into_iter().enumerate() {
                    store(first + i, lyap);
                }
            }
            for i in full_width..width {
                let (a, b) = point(i);
                store(i, compute_lyapunov_exponent(a, b, iter_max, sequence));
            }
        });

    if cancelled.load(Ordering::Relaxed) {
        None
    } else {
        Some((iterations, zs))
    }
}

/// Version annulable du rendu MPC Lyapunov.
pub fn render_lyapunov_mpc_cancellable(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    let width = params.width as usize;
    let height = params.height as usize;
    let mut iterations = vec![0u32; width * height];
    let mut zs = vec![Complex64::new(0.0, 0.0); width * height];

    if width == 0 || height == 0 {
        return Some((iterations, zs));
    }

    let prec = crate::fractal::perturbation::compute_perturbation_precision_bits(params)
        .max(params.engine.precision_bits)
        .max(64);
    let parse = |hp: Option<&String>, fallback: f64| {
        hp.and_then(|value| Float::parse(value).ok())
            .map(|value| Float::with_val(prec, value))
            .unwrap_or_else(|| Float::with_val(prec, fallback))
    };
    // Le centre et les spans viennent directement de la vue HP : les miroirs
    // f64 peuvent déjà être immobiles ou sous-fluer à ce niveau de zoom.
    let center_x = parse(params.center_x_hp.as_ref(), params.center_x);
    let center_y = parse(params.center_y_hp.as_ref(), params.center_y);
    let span_x = parse(params.span_x_hp.as_ref(), params.span_x);
    let span_y = parse(params.span_y_hp.as_ref(), params.span_y);

    let transform = params.transform_matrix();
    let pixel_delta = |value: &mut Float, index: usize, offset: f64, size: u32, span: &Float| {
        value.assign(index);
        *value += offset;
        *value /= size;
        *value -= 0.5;
        *value *= span;
    };

    let iter_max = params.iteration_max;
    let sequence = &params.formula.lyapunov_sequence;
    let [dx, dy] = params.sampling.aa_subpixel_offset;
    let jitter = params.sampling.aa_jitter.map(|(k, scale)| {
        crate::fractal::jitter::pixel_offsets(width, k, scale)
    });

    // Sur une grille alignée, chaque ligne réutilise les mêmes abscisses.
    // Limiter le cache des abscisses aux grilles où il reste de taille modeste.
    let column_bytes = std::mem::size_of::<Float>() as u64 + u64::from(prec).div_ceil(8);
    let columns = if transform.is_none() && jitter.is_none() && height > 1
        && u64::from(params.width) <= (4 * 1024 * 1024) / column_bytes
    {
        let mut columns = Vec::with_capacity(width);
        for i in 0..width {
            if i % 256 == 0 && cancel.load(Ordering::Relaxed) {
                return None;
            }
            let mut a = Float::new(prec);
            pixel_delta(&mut a, i, dx, params.width, &span_x);
            a += &center_x;
            columns.push(a);
        }
        Some(columns)
    } else {
        None
    };
    let cancelled = AtomicBool::new(false);

    iterations
        .par_chunks_mut(width)
        .zip(zs.par_chunks_mut(width))
        .enumerate()
        .for_each(|(j, (iter_row, z_row))| {
            if j % 16 == 0 {
                if cancel.load(Ordering::Relaxed) {
                    cancelled.store(true, Ordering::Relaxed);
                    return;
                }
            }
            if cancelled.load(Ordering::Relaxed) {
                return;
            }

            let mut row_x = Float::new(prec);
            let mut row_y = Float::new(prec);
            let prepare_y = |row_x: &mut Float, row_y: &mut Float, offset| {
                pixel_delta(row_y, j, offset, params.height, &span_y);
                if let Some((_, xy, _, yy)) = transform {
                    row_x.assign(&*row_y * xy);
                    *row_y *= yy;
                } else {
                    *row_y += &center_y;
                }
            };
            if jitter.is_none() {
                prepare_y(&mut row_x, &mut row_y, dy);
            }
            let mut sample_x = Float::new(prec);
            let mut sample_y = Float::new(prec);
            for (i, (iter, z)) in iter_row.iter_mut().zip(z_row.iter_mut()).enumerate() {
                let (a, b) = if let Some(columns) = &columns {
                    (&columns[i], &row_y)
                } else {
                    let jx = if let Some(offset) = &jitter {
                        let (jx, jy) = offset(i, j);
                        prepare_y(&mut row_x, &mut row_y, jy);
                        jx
                    } else {
                        dx
                    };
                    pixel_delta(&mut sample_x, i, jx, params.width, &span_x);
                    let b = if let Some((xx, _, yx, _)) = transform {
                        // Les produits et sommes gardent leurs arrondis MPFR distincts.
                        sample_y.assign(&sample_x * yx);
                        sample_y += &row_y;
                        sample_y += &center_y;
                        sample_x *= xx;
                        sample_x += &row_x;
                        &sample_y
                    } else {
                        &row_y
                    };
                    sample_x += &center_x;
                    (&sample_x, b)
                };
                let lyap = compute_lyapunov_exponent_mpc(a, b, iter_max, sequence, prec);
                let norm = normalize_lyapunov(lyap.to_f64());
                *iter = (norm * iter_max as f64) as u32;
                *z = Complex64::new(norm * 2.0, 0.0);
            }
        });

    if cancelled.load(Ordering::Relaxed) {
        None
    } else {
        Some((iterations, zs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Référence indépendante : un logarithme par terme et un index de phase
    // explicite. La carte conserve l'ordre des opérations du chemin f64.
    fn reference_exponent(a: f64, b: f64, n: u32, sequence: &[bool]) -> f64 {
        let mut x = 0.5;
        let mut sum = 0.0;
        for i in 0..WARMUP_ITERATIONS + n as usize {
            let r = if sequence[i % sequence.len()] { a } else { b };
            if i >= WARMUP_ITERATIONS {
                sum += (r * (1.0 - 2.0 * x)).abs().ln();
            }
            let next = r * x * (1.0 - x);
            x = if (0.0..=1.0).contains(&next) {
                next
            } else {
                0.5
            };
        }
        sum / n as f64
    }

    #[test]
    fn products_match_individual_logs_across_the_float_range() {
        let seq = sequence_from_str("AAB");
        for (a, b) in [
            (3.7, 3.95),
            (2.000001, 2.000001),
            (4.0, 4.0),
            (f64::from_bits(4.0_f64.to_bits() - 1), 4.0),
            (3.7, 5.25),
            (f64::from_bits(1), f64::from_bits(1)),
        ] {
            let expected = reference_exponent(a, b, 4000, &seq);
            let actual = compute_lyapunov_exponent(a, b, 4000, &seq);
            assert!(
                actual == expected || (actual - expected).abs() < 2e-13 * expected.abs().max(1.0)
            );
        }
    }

    #[test]
    fn checked_blocks_match_per_step_products_on_range_transitions() {
        let a = [4.0, 2.0, f64::MIN_POSITIVE, 3.99, 0.0, 2.000_001, 3.3, 4.0];
        let b = [4.0, 2.0, 3.95, f64::from_bits(1), 2.0, 2.000_001, 3.7, 3.8];
        for sequence in [&[true, false, true][..], &[]] {
            for n in [15, 16, 17, 511, 512, 513, 4097] {
                let block = iterate_logistic::<8, false>(a, b, n, sequence);
                let checked = iterate_logistic::<8, true>(a, b, n, sequence);
                assert_eq!(block.map(f64::to_bits), checked.map(f64::to_bits));
            }
        }
    }

    #[test]
    fn batched_exponents_match_single_pixels_at_phase_and_range_boundaries() {
        let tiny = f64::from_bits(1);
        let a = [3.7, 3.92, 2.0, tiny, 4.0, 2.000001, 3.65, 3.45];
        for last_b in [3.97, 5.25] {
            let b = [3.9, 3.74, 2.0, tiny, 4.0, 2.000001, 3.21, last_b];
            for sequence in [&[][..], &[true, false, false][..]] {
                for n in [0, 1, 17, 300, 4000] {
                    let values = compute_lyapunov_exponents(a, b, n, sequence);
                    for i in 0..LYAPUNOV_LANES {
                        let single = compute_lyapunov_exponent(a[i], b[i], n, sequence);
                        assert_eq!(values[i].to_bits(), single.to_bits(), "voie {i}, n={n}");
                    }
                }
            }
        }
    }

    #[test]
    fn render_samples_the_requested_position_on_both_axes() {
        use crate::fractal::{default_params_for_type, FractalType};
        let mut p = default_params_for_type(FractalType::Lyapunov, 11, 5);
        p.center_x = 3.14;
        p.center_y = 3.27;
        p.span_x = 0.17;
        p.span_y = 0.1;
        p.iteration_max = 73;
        let cancel = Arc::new(AtomicBool::new(false));
        let identity = [1.0, 0.0, 0.0, 1.0];
        let skew = [1.25, -0.5, 0.25, 0.75];
        for (uniform, jitter, rotation, matrix, [xx, xy, yx, yy]) in [
            ([0.0, 0.0], None, 0.0, None, identity),
            ([0.25, -0.5], None, 0.0, None, identity),
            ([0.25, -0.5], Some((3, 0.75)), 0.0, None, identity),
            ([0.25, -0.5], None, 90.0, None, [0.0, -1.0, 1.0, 0.0]),
            ([0.0, 0.0], Some((2, 0.5)), 90.0, Some(skew), skew),
        ] {
            p.rotation = rotation;
            p.transform_k = matrix;
            p.sampling.aa_subpixel_offset = uniform;
            p.sampling.aa_jitter = jitter;
            for render in [render_lyapunov_cancellable, render_lyapunov_mpc_cancellable] {
                let (_, zs) = render(&p, &cancel).unwrap();
                for (index, z) in zs.iter().enumerate() {
                    let i = index % 11;
                    let j = index / 11;
                    let (dx, dy) = match jitter {
                        Some((k, scale)) => crate::fractal::jitter::pixel_offset(11, i, j, k, scale),
                        None => (uniform[0], uniform[1]),
                    };
                    let x = ((i as f64 + dx) / 11.0 - 0.5) * p.span_x;
                    let y = ((j as f64 + dy) / 5.0 - 0.5) * p.span_y;
                    let a = p.center_x + (xx * x + xy * y);
                    let b = p.center_y + (yx * x + yy * y);
                    let expected =
                        reference_exponent(a, b, p.iteration_max, &p.formula.lyapunov_sequence);
                    assert!((z.re - 2.0 * normalize_lyapunov(expected)).abs() < 1e-11);
                }
            }
        }
    }

    #[test]
    fn mpc_resolves_the_narrow_axis_at_the_logistic_domain_boundary() {
        use crate::fractal::{default_params_for_type, FractalType, ViewHp};

        let cancel = Arc::new(AtomicBool::new(false));
        for (use_a, width, height, span_x, span_y, next_pixel) in [
            (true, 8, 2, "1e-100", "1", 1),
            (false, 2, 8, "1", "1e-100", 2),
        ] {
            let mut p = default_params_for_type(FractalType::Lyapunov, width, height);
            ViewHp::from_decimal_parts("4", "4", span_x, span_y, width, height, 128)
                .unwrap()
                .write_to_params(&mut p);
            p.formula.lyapunov_sequence = vec![use_a];
            p.iteration_max = 16;
            p.engine.precision_bits = 128;
            let rendered = render_lyapunov_mpc_cancellable(&p, &cancel).unwrap();
            p.engine.precision_bits = 768;
            let oracle = render_lyapunov_mpc_cancellable(&p, &cancel).unwrap();

            // À r = 4, l'orbite atteint zéro. Juste au-dessus, elle quitte
            // [0, 1] à chaque pas et revient à 0,5, où la dérivée est nulle.
            let center = ((height / 2) * width + width / 2) as usize;
            assert_eq!(oracle.1[center].re, 2.0);
            assert_eq!(oracle.1[center + next_pixel].re, 1.7);
            assert_eq!(rendered, oracle);
        }
    }

    fn mpc_exponent(a: f64, b: f64, iterations: u32, sequence: &[bool]) -> f64 {
        compute_lyapunov_exponent_mpc(
            &Float::with_val(128, a),
            &Float::with_val(128, b),
            iterations,
            sequence,
            128,
        )
        .to_f64()
    }

    #[test]
    fn mpc_products_preserve_derivatives_near_both_exponent_limits() {
        let prec = 128;
        let small = Float::with_val(prec, 1) << (rug::float::exp_min() + 4);
        let large = Float::with_val(prec, 1) << (rug::float::exp_max() - 4);
        let largest_power = Float::with_val(prec, 1) << (rug::float::exp_max() - 1);
        // Le premier bloc amène x à zéro, un point fixe. Chaque dérivée
        // mesurée dans le second bloc vaut donc exactement le paramètre b.
        let mut seq = vec![true; WARMUP_ITERATIONS];
        seq.extend([false; 16]);
        for b in [&small, &large, &largest_power] {
            let expected = b.clone().ln();
            let actual = compute_lyapunov_exponent_mpc(&small, b, 16, &seq, prec);
            assert_eq!(actual.to_f64(), expected.to_f64());
        }
    }

    #[test]
    fn mpc_replayed_derivatives_match_full_orbits_at_partial_periods() {
        let prec = 128;
        let sequence = sequence_from_str("AAB");
        // Même phase A/B à chaque itération. La séquence longue force
        // l'évaluation de toutes les dérivées, même après un retour exact.
        let expanded = sequence.repeat(32);
        let small = Float::with_val(prec, 1) << (rug::float::exp_min() + 4);
        for (a, b) in [
            (Float::with_val(prec, 3.2), Float::with_val(prec, 3.31)),
            (Float::with_val(prec, 3.95), Float::with_val(prec, 3.81)),
            (small.clone(), small),
        ] {
            for iterations in [1200, 1201, 1211] {
                let short = compute_lyapunov_exponent_mpc(&a, &b, iterations, &sequence, prec);
                let full = compute_lyapunov_exponent_mpc(&a, &b, iterations, &expanded, prec);
                assert_eq!(short, full);
            }
        }
    }

    #[test]
    fn repeating_a_sequence_preserves_its_phase() {
        let seq = sequence_from_str("AAB");
        let repeated = seq.repeat(18);
        for exponent in [compute_lyapunov_exponent, mpc_exponent] {
            assert_eq!(
                exponent(3.75, 3.2, 64, &seq),
                exponent(3.75, 3.2, 64, &repeated)
            );
        }
    }

    #[test]
    fn stable_fixed_point_preserves_small_derivatives() {
        let r = 2.000001_f64;
        let expected = (2.0 - r).abs().ln();
        for exponent in [compute_lyapunov_exponent, mpc_exponent] {
            assert!((exponent(r, r, 65, &[]) - expected).abs() < 1e-8);
            assert_eq!(exponent(2.0, 2.0, 65, &[]), f64::NEG_INFINITY);
        }
    }

    #[test]
    fn endpoint_orbit_remains_in_the_logistic_domain() {
        for exponent in [compute_lyapunov_exponent, mpc_exponent] {
            assert!((exponent(4.0, 4.0, 64, &[]) - 4.0_f64.ln()).abs() < 1e-12);
        }
    }

    #[test]
    fn test_lyapunov_exponent_chaotic() {
        // Point dans la région chaotique avec séquence par défaut
        let lyap = compute_lyapunov_exponent(3.0, 3.9, 1000, &[]);
        assert!(lyap < 0.0 || lyap > 0.0); // Just check it computes
    }

    #[test]
    fn test_lyapunov_exponent_custom_sequence() {
        // Test avec séquence personnalisée "AB"
        let seq = sequence_from_str("AB");
        let lyap = compute_lyapunov_exponent(3.5, 3.5, 1000, &seq);
        assert!(lyap < 0.0 || lyap > 0.0); // Just check it computes
    }

    #[test]
    fn test_normalize_negative() {
        let norm = normalize_lyapunov(-1.0);
        assert!(norm >= 0.0 && norm < 0.85);
    }

    #[test]
    fn test_normalize_positive() {
        let norm = normalize_lyapunov(0.5);
        assert!(norm >= 0.85 && norm <= 1.0);
    }

    #[test]
    fn test_sequence_from_str() {
        let seq = sequence_from_str("AABAB");
        assert_eq!(seq, vec![true, true, false, true, false]);
    }

    #[test]
    fn test_preset_all() {
        let presets = LyapunovPreset::all();
        assert_eq!(presets.len(), 6);
    }

    #[test]
    fn test_preset_cli_name_roundtrip() {
        for preset in LyapunovPreset::all() {
            let cli_name = preset.cli_name();
            let parsed = LyapunovPreset::from_cli_name(cli_name);
            assert_eq!(parsed, Some(*preset));
        }
    }
}
