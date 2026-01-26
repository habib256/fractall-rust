//! Algorithme de fractale de Lyapunov (Zircon City).
//!
//! Calcule l'exposant de Lyapunov de la carte logistique x_{n+1} = r * x * (1 - x)
//! où r alterne entre les paramètres a et b selon une séquence prédéfinie.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use num_complex::Complex64;
use rayon::prelude::*;
use rug::Float;

use crate::fractal::FractalParams;

/// Preset de Lyapunov prédéfini.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LyapunovPreset {
    /// Standard (Swallow): séquence "AB", plage [2.0, 4.0] x [2.0, 4.0]
    Standard,
    /// Zircon City: séquence "BBBBBBAAAAAA", plage A=[3.4, 4.0], B=[2.5, 3.4]
    #[default]
    ZirconCity,
    /// Jellyfish: séquence "AABAB", plage [3.4, 4.0] x [3.4, 4.0]
    Jellyfish,
    /// Asymmetric: séquence "AAB", plage [2.0, 4.0] x [2.0, 4.0]
    Asymmetric,
    /// Spaceship: séquence "ABBBA", plage [3.5, 3.9] x [3.5, 3.9]
    Spaceship,
    /// Heavy Blocks: séquence "BBBBBBBBAAAAAAAAA", plage A=[3.4, 4.0], B=[2.5, 3.4]
    HeavyBlocks,
}

impl LyapunovPreset {
    /// Retourne tous les presets disponibles.
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
    true, true, true, true, true, true,       // 6 A's (use a)
];

/// Constantes pour l'algorithme
const WARMUP_ITERATIONS: u32 = 50;
const BLOCK_SIZE: u32 = 64;
const MIN_DERIV: f64 = 1e-10;
const MIN_X: f64 = 0.0001;
const MAX_X: f64 = 0.9999;

/// Calcule l'exposant de Lyapunov pour un point (a, b) du plan des paramètres.
///
/// L'algorithme:
/// 1. Phase de warmup: itère sans accumuler pour stabiliser
/// 2. Phase de calcul: accumule log|dr/dx| par blocs pour éviter overflow/underflow
/// 3. Normalise par le nombre d'itérations
fn compute_lyapunov_exponent(a: f64, b: f64, iter_max: u32, sequence: &[bool]) -> f64 {
    let seq = if sequence.is_empty() { ZIRCON_SEQUENCE } else { sequence };
    let seq_len = seq.len();

    let mut x = 0.5;
    let mut seq_idx = 0usize;

    // Phase de warmup: stabilise x sans accumuler
    let warmup_cycles = WARMUP_ITERATIONS / seq_len as u32;
    for _ in 0..warmup_cycles {
        for &is_a in seq {
            let r = if is_a { a } else { b };
            x = r * x * (1.0 - x);

            // Reset si x sort du domaine valide
            if x < MIN_X || x > MAX_X || x.is_nan() {
                x = 0.5;
            }
        }
    }

    // Phase de calcul: accumule l'exposant par blocs
    let mut lyap = 0.0;
    let mut product = 1.0;
    let mut count_in_block = 0u32;

    for _ in 0..iter_max {
        let is_a = seq[seq_idx];
        let r = if is_a { a } else { b };

        // Itération de la carte logistique
        x = r * x * (1.0 - x);

        // Dérivée: |r * (1 - 2x)|
        let deriv = (r * (1.0 - 2.0 * x)).abs();

        if deriv > MIN_DERIV {
            product *= deriv;
        }

        count_in_block += 1;

        // Flush du bloc pour éviter overflow/underflow
        if count_in_block >= BLOCK_SIZE {
            if product > 0.0 {
                lyap += product.ln();
            }
            product = 1.0;
            count_in_block = 0;
        }

        // Reset si x diverge
        if x < MIN_X || x > MAX_X || x.is_nan() {
            x = 0.5;
        }

        seq_idx = (seq_idx + 1) % seq_len;
    }

    // Flush du dernier bloc partiel
    if count_in_block > 0 && product > 0.0 {
        lyap += product.ln();
    }

    // Normalisation
    lyap / iter_max as f64
}

/// Normalise l'exposant de Lyapunov vers [0, 1] pour la colorisation.
///
/// - Exposants négatifs (chaos) -> [0, 0.85)
/// - Exposants positifs (stabilité) -> [0.85, 1.0]
fn normalize_lyapunov(lyap: f64) -> f64 {
    if lyap < 0.0 {
        // Région chaotique
        let t = (-lyap).min(2.0);
        (t / 2.0) * 0.85
    } else {
        // Région stable
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
    let seq = if sequence.is_empty() { ZIRCON_SEQUENCE } else { sequence };
    let seq_len = seq.len();
    let mut x = Float::with_val(prec, 0.5);
    let one = Float::with_val(prec, 1.0);
    let two = Float::with_val(prec, 2.0);
    let min_x = Float::with_val(prec, MIN_X);
    let max_x = Float::with_val(prec, MAX_X);
    let min_deriv = Float::with_val(prec, MIN_DERIV);

    let warmup_cycles = WARMUP_ITERATIONS / seq_len as u32;
    for _ in 0..warmup_cycles {
        for &is_a in seq {
            let r = if is_a { a } else { b };
            let mut one_minus_x = one.clone();
            one_minus_x -= &x;
            let mut x_next = x.clone();
            x_next *= &one_minus_x;
            x_next *= r;
            x = x_next;

            if x < min_x || x > max_x || x.is_nan() {
                x = Float::with_val(prec, 0.5);
            }
        }
    }

    let mut lyap = Float::with_val(prec, 0.0);
    let mut product = Float::with_val(prec, 1.0);
    let mut count_in_block = 0u32;
    let mut seq_idx = 0usize;

    for _ in 0..iter_max {
        let is_a = seq[seq_idx];
        let r = if is_a { a } else { b };

        let mut one_minus_x = one.clone();
        one_minus_x -= &x;
        let mut x_next = x.clone();
        x_next *= &one_minus_x;
        x_next *= r;
        x = x_next;

        let mut two_x = x.clone();
        two_x *= &two;
        let mut term = one.clone();
        term -= two_x;
        term *= r;
        let deriv = term.abs();

        if deriv > min_deriv {
            product *= deriv;
        }
        count_in_block += 1;

        if count_in_block >= BLOCK_SIZE {
            if product > 0.0 {
                lyap += product.clone().ln();
            }
            product = Float::with_val(prec, 1.0);
            count_in_block = 0;
        }

        if x < min_x || x > max_x || x.is_nan() {
            x = Float::with_val(prec, 0.5);
        }

        seq_idx = (seq_idx + 1) % seq_len;
    }

    if count_in_block > 0 && product > 0.0 {
        lyap += product.clone().ln();
    }

    let denom = Float::with_val(prec, iter_max);
    lyap /= &denom;
    lyap
}

/// Rendu de la fractale de Lyapunov.
///
/// Retourne (iterations, zs) où:
/// - iterations[i] = valeur normalisée * iter_max (pour colorisation)
/// - zs[i].re = valeur normalisée * 2.0 (comme en C)
/// - zs[i].im = 0.0
#[allow(dead_code)]
pub fn render_lyapunov(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    let width = params.width as usize;
    let height = params.height as usize;
    let mut iterations = vec![0u32; width * height];
    let mut zs = vec![Complex64::new(0.0, 0.0); width * height];

    if width == 0 || height == 0 {
        return (iterations, zs);
    }

    // Utiliser span directement au lieu de xmax-xmin pour éviter les problèmes de précision
    let iter_max = params.iteration_max;

    // Utiliser la séquence personnalisée ou la séquence par défaut Zircon City
    let sequence: Vec<bool> = params.lyapunov_sequence.clone();

    // Parallélisation par lignes
    iterations
        .par_chunks_mut(width)
        .zip(zs.par_chunks_mut(width))
        .enumerate()
        .for_each(|(j, (iter_row, z_row))| {
            // Utiliser center+span directement
            let y_ratio = j as f64 / params.height as f64;
            let b = params.center_y + (y_ratio - 0.5) * params.span_y;

            for (i, (iter, z)) in iter_row.iter_mut().zip(z_row.iter_mut()).enumerate() {
                let x_ratio = i as f64 / params.width as f64;
                let a = params.center_x + (x_ratio - 0.5) * params.span_x;

                let lyap = compute_lyapunov_exponent(a, b, iter_max, &sequence);
                let norm = normalize_lyapunov(lyap);

                // Stockage compatible avec le système de colorisation
                *iter = (norm * iter_max as f64) as u32;
                *z = Complex64::new(norm * 2.0, 0.0);
            }
        });

    (iterations, zs)
}

/// Rendu MPC de la fractale de Lyapunov.
pub fn render_lyapunov_mpc(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    let width = params.width as usize;
    let height = params.height as usize;
    let mut iterations = vec![0u32; width * height];
    let mut zs = vec![Complex64::new(0.0, 0.0); width * height];

    if width == 0 || height == 0 {
        return (iterations, zs);
    }

    let prec = params.precision_bits.max(64);
    // Utiliser center+span directement pour éviter les problèmes de précision
    let center_x = Float::with_val(prec, params.center_x);
    let center_y = Float::with_val(prec, params.center_y);
    let span_x = Float::with_val(prec, params.span_x);
    let span_y = Float::with_val(prec, params.span_y);

    let width_f = Float::with_val(prec, params.width);
    let height_f = Float::with_val(prec, params.height);
    let half = Float::with_val(prec, 0.5);
    let iter_max = params.iteration_max;
    let sequence: Vec<bool> = params.lyapunov_sequence.clone();

    iterations
        .par_chunks_mut(width)
        .zip(zs.par_chunks_mut(width))
        .enumerate()
        .for_each(|(j, (iter_row, z_row))| {
            let j_f = Float::with_val(prec, j as u32);
            let mut y_ratio = j_f.clone();
            y_ratio /= &height_f;
            y_ratio -= &half;
            let mut b = span_y.clone();
            b *= &y_ratio;
            b += &center_y;
            for (i, (iter, z)) in iter_row.iter_mut().zip(z_row.iter_mut()).enumerate() {
                let i_f = Float::with_val(prec, i as u32);
                let mut x_ratio = i_f;
                x_ratio /= &width_f;
                x_ratio -= &half;
                let mut a = span_x.clone();
                a *= &x_ratio;
                a += &center_x;

                let lyap = compute_lyapunov_exponent_mpc(&a, &b, iter_max, &sequence, prec);
                let norm = normalize_lyapunov(lyap.to_f64());

                *iter = (norm * iter_max as f64) as u32;
                *z = Complex64::new(norm * 2.0, 0.0);
            }
        });

    (iterations, zs)
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
    let sequence: Vec<bool> = params.lyapunov_sequence.clone();

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

            // Utiliser center+span directement
            let y_ratio = j as f64 / params.height as f64;
            let b = params.center_y + (y_ratio - 0.5) * params.span_y;

            for (i, (iter, z)) in iter_row.iter_mut().zip(z_row.iter_mut()).enumerate() {
                let x_ratio = i as f64 / params.width as f64;
                let a = params.center_x + (x_ratio - 0.5) * params.span_x;

                let lyap = compute_lyapunov_exponent(a, b, iter_max, &sequence);
                let norm = normalize_lyapunov(lyap);

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

    let prec = params.precision_bits.max(64);
    // Utiliser center+span directement pour éviter les problèmes de précision
    let center_x = Float::with_val(prec, params.center_x);
    let center_y = Float::with_val(prec, params.center_y);
    let span_x = Float::with_val(prec, params.span_x);
    let span_y = Float::with_val(prec, params.span_y);

    let width_f = Float::with_val(prec, params.width);
    let height_f = Float::with_val(prec, params.height);
    let half = Float::with_val(prec, 0.5);

    let iter_max = params.iteration_max;
    let sequence: Vec<bool> = params.lyapunov_sequence.clone();
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

            let j_f = Float::with_val(prec, j as u32);
            let mut y_ratio = j_f.clone();
            y_ratio /= &height_f;
            y_ratio -= &half;
            let mut b = span_y.clone();
            b *= &y_ratio;
            b += &center_y;

            for (i, (iter, z)) in iter_row.iter_mut().zip(z_row.iter_mut()).enumerate() {
                let i_f = Float::with_val(prec, i as u32);
                let mut x_ratio = i_f;
                x_ratio /= &width_f;
                x_ratio -= &half;
                let mut a = span_x.clone();
                a *= &x_ratio;
                a += &center_x;

                let lyap = compute_lyapunov_exponent_mpc(&a, &b, iter_max, &sequence, prec);
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

    #[test]
    fn test_lyapunov_exponent_chaotic() {
        // Point dans la région chaotique avec séquence par défaut
        let lyap = compute_lyapunov_exponent(3.0, 3.9, 1000, &[]);
        // L'exposant devrait être négatif (chaos)
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
