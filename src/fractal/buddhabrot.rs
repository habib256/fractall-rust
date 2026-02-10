//! Algorithmes Buddhabrot et Nebulabrot.
//!
//! Buddhabrot: visualise la densité des trajectoires d'échappement de z²+c.
//! Nebulabrot: version RGB avec différentes limites d'itérations par canal.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use num_complex::Complex64;
use rayon::prelude::*;
use rug::Complex;
use rug::Float;

use crate::fractal::FractalParams;
use crate::fractal::gmp::complex_to_complex64;

/// Générateur de nombres pseudo-aléatoires simple (LCG).
struct Rng {
    seed: u32,
}

impl Rng {
    fn new(seed: u32) -> Self {
        Self { seed }
    }

    fn next(&mut self) -> u32 {
        self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
        self.seed
    }

    fn next_f64(&mut self) -> f64 {
        (self.next() & 0x7FFFFFFF) as f64 / 2147483647.0
    }
}

fn complex_norm_sqr_mpc(value: &Complex, prec: u32) -> Float {
    let mut re2 = value.real().clone();
    re2 *= value.real();
    let mut im2 = value.imag().clone();
    im2 *= value.imag();
    let mut sum = Float::with_val(prec, re2);
    sum += im2;
    sum
}

/// Rendu Buddhabrot.
///
/// Trace les trajectoires des points qui s'échappent et accumule
/// leur densité dans une matrice.
#[allow(dead_code)]
pub fn render_buddhabrot(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    let width = params.width as usize;
    let height = params.height as usize;
    let size = width * height;

    if width == 0 || height == 0 {
        return (vec![0; size], vec![Complex64::new(0.0, 0.0); size]);
    }

    // Utiliser span directement au lieu de xmax-xmin pour éviter les problèmes de précision
    let xrange = params.span_x;
    let yrange = params.span_y;
    let iter_max = params.iteration_max;
    let bailout_sq = params.bailout * params.bailout;

    // Nombre d'échantillons basé sur la résolution
    let pixels = width * height;
    let num_samples = if pixels <= 640 * 480 {
        pixels * 20
    } else if pixels <= 1024 * 768 {
        pixels * 10
    } else {
        pixels * 5
    }
    .max(1000)
    .min(50_000_000);

    // Matrice de densité atomique pour parallélisation
    let density: Vec<AtomicU32> = (0..size).map(|_| AtomicU32::new(0)).collect();

    // Seuil pour early-exit (points probablement dans l'ensemble)
    let early_exit_threshold = if iter_max < 50 { iter_max / 2 } else { 50 };

    // Traitement parallèle des échantillons
    (0..num_samples).into_par_iter().for_each(|sample_idx| {
        let mut rng = Rng::new((sample_idx as u32).wrapping_mul(12345).wrapping_add(42));

        // Point aléatoire dans le domaine : utiliser center+span directement
        // xg = center_x + (random - 0.5) * span_x
        let xg = params.center_x + (rng.next_f64() - 0.5) * xrange;
        let yg = params.center_y + (rng.next_f64() - 0.5) * yrange;
        let c = Complex64::new(xg, yg);

        // Trajectoire
        let mut trajectory = Vec::with_capacity(iter_max as usize);
        let mut z = Complex64::new(0.0, 0.0);
        let mut escaped = false;

        for iter in 0..iter_max {
            z = z * z + c;

            if z.re.is_nan() || z.im.is_nan() || z.re.is_infinite() || z.im.is_infinite() {
                break;
            }

            let mag2 = z.norm_sqr();

            // Early exit: si le point est encore petit après many iterations,
            // il est probablement dans l'ensemble
            if iter == early_exit_threshold && mag2 < 0.25 {
                break;
            }

            trajectory.push(z);

            if mag2 > bailout_sq {
                escaped = true;
                break;
            }
        }

        // Si le point s'est échappé, tracer sa trajectoire
        if escaped && !trajectory.is_empty() {
            let scale_x = width as f64 / xrange;
            let scale_y = height as f64 / yrange;

            for point in &trajectory {
                if point.re.is_nan() || point.im.is_nan() {
                    continue;
                }

                // Convertir en pixels en utilisant center+span directement
                // px = ((point.re - center_x + span_x/2) / span_x) * width
                let px = ((point.re - params.center_x + xrange * 0.5) * scale_x) as i32;
                let py = ((point.im - params.center_y + yrange * 0.5) * scale_y) as i32;

                if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                    let idx = py as usize * width + px as usize;
                    density[idx].fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    });

    // Trouver la densité maximale
    let max_density = density
        .iter()
        .map(|d| d.load(Ordering::Relaxed))
        .max()
        .unwrap_or(1)
        .max(1);

    let log_max = (1.0 + max_density as f64).ln();

    // Convertir en iterations/zs pour colorisation
    let iterations: Vec<u32> = density
        .par_iter()
        .map(|d| {
            let val = d.load(Ordering::Relaxed);
            let normalized = (1.0 + val as f64).ln() / log_max;
            (normalized * iter_max as f64) as u32
        })
        .collect();

    let zs: Vec<Complex64> = density
        .par_iter()
        .map(|d| {
            let val = d.load(Ordering::Relaxed);
            let normalized = (1.0 + val as f64).ln() / log_max;
            Complex64::new(normalized * 2.0, 0.0)
        })
        .collect();

    (iterations, zs)
}

/// Rendu Buddhabrot en précision MPC.
pub fn render_buddhabrot_mpc(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    let cancel = Arc::new(AtomicBool::new(false));
    render_buddhabrot_mpc_cancellable(params, &cancel)
        .unwrap_or_else(|| (Vec::new(), Vec::new()))
}

/// Version annulable du rendu Buddhabrot en MPC.
pub fn render_buddhabrot_mpc_cancellable(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    let width = params.width as usize;
    let height = params.height as usize;
    let size = width * height;

    if width == 0 || height == 0 {
        return Some((vec![0; size], vec![Complex64::new(0.0, 0.0); size]));
    }

    let prec = params.precision_bits.max(64);
    // Utiliser span directement au lieu de xmax-xmin pour éviter les problèmes de précision
    let xrange = params.span_x;
    let yrange = params.span_y;
    let iter_max = params.iteration_max;
    let bailout = Float::with_val(prec, params.bailout);
    let mut bailout_sq = bailout.clone();
    bailout_sq *= &bailout;
    let early_exit_threshold = if iter_max < 50 { iter_max / 2 } else { 50 };
    let early_exit_limit = Float::with_val(prec, 0.25f64);

    let pixels = width * height;
    let num_samples = if pixels <= 640 * 480 {
        pixels * 20
    } else if pixels <= 1024 * 768 {
        pixels * 10
    } else {
        pixels * 5
    }
    .max(1000)
    .min(50_000_000);

    let density: Vec<AtomicU32> = (0..size).map(|_| AtomicU32::new(0)).collect();
    let cancelled = AtomicBool::new(false);

    (0..num_samples).into_par_iter().for_each(|sample_idx| {
        if sample_idx % 10000 == 0 {
            if cancel.load(Ordering::Relaxed) {
                cancelled.store(true, Ordering::Relaxed);
                return;
            }
        }
        if cancelled.load(Ordering::Relaxed) {
            return;
        }

        let mut rng = Rng::new((sample_idx as u32).wrapping_mul(12345).wrapping_add(42));
        // Point aléatoire : utiliser center+span directement
        let xg = params.center_x + (rng.next_f64() - 0.5) * xrange;
        let yg = params.center_y + (rng.next_f64() - 0.5) * yrange;
        let c = Complex::with_val(prec, (xg, yg));

        let mut trajectory: Vec<Complex64> = Vec::with_capacity(iter_max as usize);
        let mut z = Complex::with_val(prec, (0.0, 0.0));
        let mut escaped = false;

        for iter in 0..iter_max {
            let mut z_next = z.clone();
            z_next *= &z;
            z_next += &c;
            z = z_next;

            if z.real().is_nan() || z.imag().is_nan() || z.real().is_infinite() || z.imag().is_infinite() {
                break;
            }

            let mag2 = complex_norm_sqr_mpc(&z, prec);
            if iter == early_exit_threshold && mag2 < early_exit_limit {
                break;
            }

            trajectory.push(complex_to_complex64(&z));

            if mag2 > bailout_sq {
                escaped = true;
                break;
            }
        }

        if escaped && !trajectory.is_empty() {
            let scale_x = width as f64 / xrange;
            let scale_y = height as f64 / yrange;

            for point in &trajectory {
                if point.re.is_nan() || point.im.is_nan() {
                    continue;
                }

                // Convertir en pixels en utilisant center+span directement
                let px = ((point.re - params.center_x + xrange * 0.5) * scale_x) as i32;
                let py = ((point.im - params.center_y + yrange * 0.5) * scale_y) as i32;

                if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                    let idx = py as usize * width + px as usize;
                    density[idx].fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    });

    if cancelled.load(Ordering::Relaxed) {
        return None;
    }

    let max_density = density
        .iter()
        .map(|d| d.load(Ordering::Relaxed))
        .max()
        .unwrap_or(1)
        .max(1);
    let log_max = (1.0 + max_density as f64).ln();

    let iterations: Vec<u32> = density
        .par_iter()
        .map(|d| {
            let val = d.load(Ordering::Relaxed);
            let normalized = (1.0 + val as f64).ln() / log_max;
            (normalized * iter_max as f64) as u32
        })
        .collect();

    let zs: Vec<Complex64> = density
        .par_iter()
        .map(|d| {
            let val = d.load(Ordering::Relaxed);
            let normalized = (1.0 + val as f64).ln() / log_max;
            Complex64::new(normalized * 2.0, 0.0)
        })
        .collect();

    Some((iterations, zs))
}

/// Rendu Nebulabrot (RGB).
///
/// Version colorée du Buddhabrot avec différentes limites d'itérations:
/// - Rouge: 50 itérations (structures fines)
/// - Vert: 500 itérations (structures moyennes)
/// - Bleu: 5000 itérations (structures profondes)
///
/// Retourne directement une image RGB au lieu de iterations/zs.
#[allow(dead_code)]
pub fn render_nebulabrot(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    let width = params.width as usize;
    let height = params.height as usize;
    let size = width * height;

    if width == 0 || height == 0 {
        return (vec![0; size], vec![Complex64::new(0.0, 0.0); size]);
    }

    // Utiliser span directement au lieu de xmax-xmin pour éviter les problèmes de précision
    let xrange = params.span_x;
    let yrange = params.span_y;
    let bailout_sq = params.bailout * params.bailout;

    // Limites d'itérations pour RGB
    const ITER_R: u32 = 50;
    const ITER_G: u32 = 500;
    const ITER_B: u32 = 5000;
    const ITER_MAX: u32 = ITER_B;

    // Nombre d'échantillons
    let pixels = width * height;
    let num_samples = if pixels <= 640 * 480 {
        pixels * 15
    } else if pixels <= 1024 * 768 {
        pixels * 8
    } else {
        pixels * 4
    }
    .max(1000)
    .min(30_000_000);

    // Matrices de densité par canal
    let density_r: Vec<AtomicU32> = (0..size).map(|_| AtomicU32::new(0)).collect();
    let density_g: Vec<AtomicU32> = (0..size).map(|_| AtomicU32::new(0)).collect();
    let density_b: Vec<AtomicU32> = (0..size).map(|_| AtomicU32::new(0)).collect();

    // Traitement parallèle
    (0..num_samples).into_par_iter().for_each(|sample_idx| {
        let mut rng = Rng::new((sample_idx as u32).wrapping_mul(12345).wrapping_add(42));

        // Point aléatoire : utiliser center+span directement
        let xg = params.center_x + (rng.next_f64() - 0.5) * xrange;
        let yg = params.center_y + (rng.next_f64() - 0.5) * yrange;
        let c = Complex64::new(xg, yg);

        let mut trajectory = Vec::with_capacity(ITER_MAX as usize);
        let mut z = Complex64::new(0.0, 0.0);
        let mut escaped = false;
        let mut escape_iter = 0u32;

        for iter in 0..ITER_MAX {
            z = z * z + c;

            if z.re.is_nan() || z.im.is_nan() || z.re.is_infinite() || z.im.is_infinite() {
                break;
            }

            trajectory.push(z);

            if z.norm_sqr() > bailout_sq {
                escaped = true;
                escape_iter = iter;
                break;
            }
        }

        if escaped && !trajectory.is_empty() {
            let scale_x = width as f64 / xrange;
            let scale_y = height as f64 / yrange;

            let contribute_r = escape_iter <= ITER_R;
            let contribute_g = escape_iter <= ITER_G;
            let contribute_b = escape_iter <= ITER_B;

            for point in trajectory {
                // Convertir en pixels en utilisant center+span directement
                let px = ((point.re - params.center_x + xrange * 0.5) * scale_x) as i32;
                let py = ((point.im - params.center_y + yrange * 0.5) * scale_y) as i32;

                if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                    let idx = py as usize * width + px as usize;
                    if contribute_r {
                        density_r[idx].fetch_add(1, Ordering::Relaxed);
                    }
                    if contribute_g {
                        density_g[idx].fetch_add(1, Ordering::Relaxed);
                    }
                    if contribute_b {
                        density_b[idx].fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
    });

    // Trouver les maxima par canal
    let max_r = density_r
        .iter()
        .map(|d| d.load(Ordering::Relaxed))
        .max()
        .unwrap_or(1)
        .max(1);
    let max_g = density_g
        .iter()
        .map(|d| d.load(Ordering::Relaxed))
        .max()
        .unwrap_or(1)
        .max(1);
    let max_b = density_b
        .iter()
        .map(|d| d.load(Ordering::Relaxed))
        .max()
        .unwrap_or(1)
        .max(1);

    let log_max_r = (1.0 + max_r as f64).ln();
    let log_max_g = (1.0 + max_g as f64).ln();
    let log_max_b = (1.0 + max_b as f64).ln();

    // Encoder les couleurs RGB dans iterations et zs
    // iterations[i] encode R et G (R dans les bits hauts, G dans les bits bas)
    // zs[i].re encode B normalisé
    let iterations: Vec<u32> = (0..size)
        .into_par_iter()
        .map(|i| {
            let r = density_r[i].load(Ordering::Relaxed);
            let g = density_g[i].load(Ordering::Relaxed);
            let r_norm = ((1.0 + r as f64).ln() / log_max_r * 255.0) as u32;
            let g_norm = ((1.0 + g as f64).ln() / log_max_g * 255.0) as u32;
            (r_norm << 16) | (g_norm << 8) // Encoder R et G
        })
        .collect();

    let zs: Vec<Complex64> = (0..size)
        .into_par_iter()
        .map(|i| {
            let b = density_b[i].load(Ordering::Relaxed);
            let b_norm = (1.0 + b as f64).ln() / log_max_b;
            Complex64::new(b_norm, 0.0) // B dans re
        })
        .collect();

    (iterations, zs)
}

/// Rendu Nebulabrot en précision MPC.
pub fn render_nebulabrot_mpc(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    let cancel = Arc::new(AtomicBool::new(false));
    render_nebulabrot_mpc_cancellable(params, &cancel)
        .unwrap_or_else(|| (Vec::new(), Vec::new()))
}

/// Version annulable du rendu Nebulabrot en MPC.
pub fn render_nebulabrot_mpc_cancellable(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    let width = params.width as usize;
    let height = params.height as usize;
    let size = width * height;

    if width == 0 || height == 0 {
        return Some((vec![0; size], vec![Complex64::new(0.0, 0.0); size]));
    }

    let prec = params.precision_bits.max(64);
    // Utiliser span directement au lieu de xmax-xmin pour éviter les problèmes de précision
    let xrange = params.span_x;
    let yrange = params.span_y;
    let bailout = Float::with_val(prec, params.bailout);
    let mut bailout_sq = bailout.clone();
    bailout_sq *= &bailout;

    const ITER_R: u32 = 50;
    const ITER_G: u32 = 500;
    const ITER_B: u32 = 5000;
    const ITER_MAX: u32 = ITER_B;

    let pixels = width * height;
    let num_samples = if pixels <= 640 * 480 {
        pixels * 15
    } else if pixels <= 1024 * 768 {
        pixels * 8
    } else {
        pixels * 4
    }
    .max(1000)
    .min(30_000_000);

    let density_r: Vec<AtomicU32> = (0..size).map(|_| AtomicU32::new(0)).collect();
    let density_g: Vec<AtomicU32> = (0..size).map(|_| AtomicU32::new(0)).collect();
    let density_b: Vec<AtomicU32> = (0..size).map(|_| AtomicU32::new(0)).collect();
    let cancelled = AtomicBool::new(false);

    (0..num_samples).into_par_iter().for_each(|sample_idx| {
        if sample_idx % 10000 == 0 {
            if cancel.load(Ordering::Relaxed) {
                cancelled.store(true, Ordering::Relaxed);
                return;
            }
        }
        if cancelled.load(Ordering::Relaxed) {
            return;
        }

        let mut rng = Rng::new((sample_idx as u32).wrapping_mul(12345).wrapping_add(42));
        // Point aléatoire : utiliser center+span directement
        let xg = params.center_x + (rng.next_f64() - 0.5) * xrange;
        let yg = params.center_y + (rng.next_f64() - 0.5) * yrange;
        let c = Complex::with_val(prec, (xg, yg));

        let mut trajectory: Vec<Complex64> = Vec::with_capacity(ITER_MAX as usize);
        let mut z = Complex::with_val(prec, (0.0, 0.0));
        let mut escaped = false;
        let mut escape_iter = 0u32;

        for iter in 0..ITER_MAX {
            let mut z_next = z.clone();
            z_next *= &z;
            z_next += &c;
            z = z_next;

            if z.real().is_nan() || z.imag().is_nan() || z.real().is_infinite() || z.imag().is_infinite() {
                break;
            }

            trajectory.push(complex_to_complex64(&z));

            if complex_norm_sqr_mpc(&z, prec) > bailout_sq {
                escaped = true;
                escape_iter = iter;
                break;
            }
        }

        if escaped && !trajectory.is_empty() {
            let scale_x = width as f64 / xrange;
            let scale_y = height as f64 / yrange;

            let contribute_r = escape_iter <= ITER_R;
            let contribute_g = escape_iter <= ITER_G;
            let contribute_b = escape_iter <= ITER_B;

            for point in trajectory {
                // Convertir en pixels en utilisant center+span directement
                let px = ((point.re - params.center_x + xrange * 0.5) * scale_x) as i32;
                let py = ((point.im - params.center_y + yrange * 0.5) * scale_y) as i32;

                if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                    let idx = py as usize * width + px as usize;
                    if contribute_r {
                        density_r[idx].fetch_add(1, Ordering::Relaxed);
                    }
                    if contribute_g {
                        density_g[idx].fetch_add(1, Ordering::Relaxed);
                    }
                    if contribute_b {
                        density_b[idx].fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
    });

    if cancelled.load(Ordering::Relaxed) {
        return None;
    }

    let max_r = density_r
        .iter()
        .map(|d| d.load(Ordering::Relaxed))
        .max()
        .unwrap_or(1)
        .max(1);
    let max_g = density_g
        .iter()
        .map(|d| d.load(Ordering::Relaxed))
        .max()
        .unwrap_or(1)
        .max(1);
    let max_b = density_b
        .iter()
        .map(|d| d.load(Ordering::Relaxed))
        .max()
        .unwrap_or(1)
        .max(1);

    let log_max_r = (1.0 + max_r as f64).ln();
    let log_max_g = (1.0 + max_g as f64).ln();
    let log_max_b = (1.0 + max_b as f64).ln();

    let iterations: Vec<u32> = (0..size)
        .into_par_iter()
        .map(|i| {
            let r = density_r[i].load(Ordering::Relaxed);
            let g = density_g[i].load(Ordering::Relaxed);
            let r_norm = ((1.0 + r as f64).ln() / log_max_r * 255.0) as u32;
            let g_norm = ((1.0 + g as f64).ln() / log_max_g * 255.0) as u32;
            (r_norm << 16) | (g_norm << 8)
        })
        .collect();

    let zs: Vec<Complex64> = (0..size)
        .into_par_iter()
        .map(|i| {
            let b = density_b[i].load(Ordering::Relaxed);
            let b_norm = (1.0 + b as f64).ln() / log_max_b;
            Complex64::new(b_norm, 0.0)
        })
        .collect();

    Some((iterations, zs))
}

/// Version annulable du rendu Buddhabrot.
pub fn render_buddhabrot_cancellable(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    let width = params.width as usize;
    let height = params.height as usize;
    let size = width * height;

    if width == 0 || height == 0 {
        return Some((vec![0; size], vec![Complex64::new(0.0, 0.0); size]));
    }

    // Utiliser span directement au lieu de xmax-xmin pour éviter les problèmes de précision
    let xrange = params.span_x;
    let yrange = params.span_y;
    let iter_max = params.iteration_max;
    let bailout_sq = params.bailout * params.bailout;

    let pixels = width * height;
    let num_samples = if pixels <= 640 * 480 {
        pixels * 20
    } else if pixels <= 1024 * 768 {
        pixels * 10
    } else {
        pixels * 5
    }
    .max(1000)
    .min(50_000_000);

    let density: Vec<AtomicU32> = (0..size).map(|_| AtomicU32::new(0)).collect();
    let early_exit_threshold = if iter_max < 50 { iter_max / 2 } else { 50 };
    let cancelled = AtomicBool::new(false);

    (0..num_samples).into_par_iter().for_each(|sample_idx| {
        // Vérifier l'annulation toutes les 10000 samples
        if sample_idx % 10000 == 0 {
            if cancel.load(Ordering::Relaxed) {
                cancelled.store(true, Ordering::Relaxed);
                return;
            }
        }
        if cancelled.load(Ordering::Relaxed) {
            return;
        }

        let mut rng = Rng::new((sample_idx as u32).wrapping_mul(12345).wrapping_add(42));
        // Point aléatoire : utiliser center+span directement
        let xg = params.center_x + (rng.next_f64() - 0.5) * xrange;
        let yg = params.center_y + (rng.next_f64() - 0.5) * yrange;
        let c = Complex64::new(xg, yg);

        let mut trajectory = Vec::with_capacity(iter_max as usize);
        let mut z = Complex64::new(0.0, 0.0);
        let mut escaped = false;

        for iter in 0..iter_max {
            z = z * z + c;

            if z.re.is_nan() || z.im.is_nan() || z.re.is_infinite() || z.im.is_infinite() {
                break;
            }

            if iter == early_exit_threshold && z.norm_sqr() < 0.25 {
                break;
            }

            trajectory.push(z);

            if z.norm_sqr() > bailout_sq {
                escaped = true;
                break;
            }
        }

        if escaped && !trajectory.is_empty() {
            let scale_x = width as f64 / xrange;
            let scale_y = height as f64 / yrange;

            for point in &trajectory {
                if point.re.is_nan() || point.im.is_nan() {
                    continue;
                }

                // Convertir en pixels en utilisant center+span directement
                let px = ((point.re - params.center_x + xrange * 0.5) * scale_x) as i32;
                let py = ((point.im - params.center_y + yrange * 0.5) * scale_y) as i32;

                if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                    let idx = py as usize * width + px as usize;
                    density[idx].fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    });

    if cancelled.load(Ordering::Relaxed) {
        return None;
    }

    let max_density = density
        .iter()
        .map(|d| d.load(Ordering::Relaxed))
        .max()
        .unwrap_or(1)
        .max(1);

    let log_max = (1.0 + max_density as f64).ln();

    let iterations: Vec<u32> = density
        .par_iter()
        .map(|d| {
            let val = d.load(Ordering::Relaxed);
            let normalized = (1.0 + val as f64).ln() / log_max;
            (normalized * iter_max as f64) as u32
        })
        .collect();

    let zs: Vec<Complex64> = density
        .par_iter()
        .map(|d| {
            let val = d.load(Ordering::Relaxed);
            let normalized = (1.0 + val as f64).ln() / log_max;
            Complex64::new(normalized * 2.0, 0.0)
        })
        .collect();

    Some((iterations, zs))
}

/// Version annulable du rendu Nebulabrot.
pub fn render_nebulabrot_cancellable(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    let width = params.width as usize;
    let height = params.height as usize;
    let size = width * height;

    if width == 0 || height == 0 {
        return Some((vec![0; size], vec![Complex64::new(0.0, 0.0); size]));
    }

    // Utiliser span directement au lieu de xmax-xmin pour éviter les problèmes de précision
    let xrange = params.span_x;
    let yrange = params.span_y;
    let bailout_sq = params.bailout * params.bailout;

    const ITER_R: u32 = 50;
    const ITER_G: u32 = 500;
    const ITER_B: u32 = 5000;
    const ITER_MAX: u32 = ITER_B;

    let pixels = width * height;
    let num_samples = if pixels <= 640 * 480 {
        pixels * 15
    } else if pixels <= 1024 * 768 {
        pixels * 8
    } else {
        pixels * 4
    }
    .max(1000)
    .min(30_000_000);

    let density_r: Vec<AtomicU32> = (0..size).map(|_| AtomicU32::new(0)).collect();
    let density_g: Vec<AtomicU32> = (0..size).map(|_| AtomicU32::new(0)).collect();
    let density_b: Vec<AtomicU32> = (0..size).map(|_| AtomicU32::new(0)).collect();
    let cancelled = AtomicBool::new(false);

    (0..num_samples).into_par_iter().for_each(|sample_idx| {
        if sample_idx % 10000 == 0 {
            if cancel.load(Ordering::Relaxed) {
                cancelled.store(true, Ordering::Relaxed);
                return;
            }
        }
        if cancelled.load(Ordering::Relaxed) {
            return;
        }

        let mut rng = Rng::new((sample_idx as u32).wrapping_mul(12345).wrapping_add(42));
        // Point aléatoire : utiliser center+span directement
        let xg = params.center_x + (rng.next_f64() - 0.5) * xrange;
        let yg = params.center_y + (rng.next_f64() - 0.5) * yrange;
        let c = Complex64::new(xg, yg);

        let mut trajectory = Vec::with_capacity(ITER_MAX as usize);
        let mut z = Complex64::new(0.0, 0.0);
        let mut escaped = false;
        let mut escape_iter = 0u32;

        for iter in 0..ITER_MAX {
            z = z * z + c;

            if z.re.is_nan() || z.im.is_nan() || z.re.is_infinite() || z.im.is_infinite() {
                break;
            }

            trajectory.push(z);

            if z.norm_sqr() > bailout_sq {
                escaped = true;
                escape_iter = iter;
                break;
            }
        }

        if escaped && !trajectory.is_empty() {
            let scale_x = width as f64 / xrange;
            let scale_y = height as f64 / yrange;

            let contribute_r = escape_iter <= ITER_R;
            let contribute_g = escape_iter <= ITER_G;
            let contribute_b = escape_iter <= ITER_B;

            for point in trajectory {
                // Convertir en pixels en utilisant center+span directement
                let px = ((point.re - params.center_x + xrange * 0.5) * scale_x) as i32;
                let py = ((point.im - params.center_y + yrange * 0.5) * scale_y) as i32;

                if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                    let idx = py as usize * width + px as usize;
                    if contribute_r {
                        density_r[idx].fetch_add(1, Ordering::Relaxed);
                    }
                    if contribute_g {
                        density_g[idx].fetch_add(1, Ordering::Relaxed);
                    }
                    if contribute_b {
                        density_b[idx].fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
    });

    if cancelled.load(Ordering::Relaxed) {
        return None;
    }

    let max_r = density_r.iter().map(|d| d.load(Ordering::Relaxed)).max().unwrap_or(1).max(1);
    let max_g = density_g.iter().map(|d| d.load(Ordering::Relaxed)).max().unwrap_or(1).max(1);
    let max_b = density_b.iter().map(|d| d.load(Ordering::Relaxed)).max().unwrap_or(1).max(1);

    let log_max_r = (1.0 + max_r as f64).ln();
    let log_max_g = (1.0 + max_g as f64).ln();
    let log_max_b = (1.0 + max_b as f64).ln();

    let iterations: Vec<u32> = (0..size)
        .into_par_iter()
        .map(|i| {
            let r = density_r[i].load(Ordering::Relaxed);
            let g = density_g[i].load(Ordering::Relaxed);
            let r_norm = ((1.0 + r as f64).ln() / log_max_r * 255.0) as u32;
            let g_norm = ((1.0 + g as f64).ln() / log_max_g * 255.0) as u32;
            (r_norm << 16) | (g_norm << 8)
        })
        .collect();

    let zs: Vec<Complex64> = (0..size)
        .into_par_iter()
        .map(|i| {
            let b = density_b[i].load(Ordering::Relaxed);
            let b_norm = (1.0 + b as f64).ln() / log_max_b;
            Complex64::new(b_norm, 0.0)
        })
        .collect();

    Some((iterations, zs))
}

// ─────────────────────────────────────────────────────────────────────────────
// Anti-Buddhabrot : accumule les orbites des points INTÉRIEURS (non-escapés).
// ─────────────────────────────────────────────────────────────────────────────

/// Rendu Anti-Buddhabrot.
///
/// Trace les trajectoires des points qui restent bornés (intérieurs à l'ensemble
/// de Mandelbrot) et accumule leur densité dans une matrice.
#[allow(dead_code)]
pub fn render_antibuddhabrot(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    let width = params.width as usize;
    let height = params.height as usize;
    let size = width * height;

    if width == 0 || height == 0 {
        return (vec![0; size], vec![Complex64::new(0.0, 0.0); size]);
    }

    let xrange = params.span_x;
    let yrange = params.span_y;
    let iter_max = params.iteration_max;
    let bailout_sq = params.bailout * params.bailout;

    let pixels = width * height;
    let num_samples = if pixels <= 640 * 480 {
        pixels * 20
    } else if pixels <= 1024 * 768 {
        pixels * 10
    } else {
        pixels * 5
    }
    .max(1000)
    .min(50_000_000);

    let density: Vec<AtomicU32> = (0..size).map(|_| AtomicU32::new(0)).collect();

    (0..num_samples).into_par_iter().for_each(|sample_idx| {
        let mut rng = Rng::new((sample_idx as u32).wrapping_mul(12345).wrapping_add(42));

        let xg = params.center_x + (rng.next_f64() - 0.5) * xrange;
        let yg = params.center_y + (rng.next_f64() - 0.5) * yrange;
        let c = Complex64::new(xg, yg);

        let mut trajectory = Vec::with_capacity(iter_max as usize);
        let mut z = Complex64::new(0.0, 0.0);
        let mut escaped = false;

        for _iter in 0..iter_max {
            z = z * z + c;

            if z.re.is_nan() || z.im.is_nan() || z.re.is_infinite() || z.im.is_infinite() {
                escaped = true;
                break;
            }

            let mag2 = z.norm_sqr();
            trajectory.push(z);

            if mag2 > bailout_sq {
                escaped = true;
                break;
            }
        }

        // Anti-Buddhabrot : tracer uniquement les trajectoires des points INTÉRIEURS
        if !escaped && !trajectory.is_empty() {
            let scale_x = width as f64 / xrange;
            let scale_y = height as f64 / yrange;

            for point in &trajectory {
                if point.re.is_nan() || point.im.is_nan() {
                    continue;
                }

                let px = ((point.re - params.center_x + xrange * 0.5) * scale_x) as i32;
                let py = ((point.im - params.center_y + yrange * 0.5) * scale_y) as i32;

                if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                    let idx = py as usize * width + px as usize;
                    density[idx].fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    });

    let max_density = density
        .iter()
        .map(|d| d.load(Ordering::Relaxed))
        .max()
        .unwrap_or(1)
        .max(1);

    let log_max = (1.0 + max_density as f64).ln();

    let iterations: Vec<u32> = density
        .par_iter()
        .map(|d| {
            let val = d.load(Ordering::Relaxed);
            let normalized = (1.0 + val as f64).ln() / log_max;
            (normalized * iter_max as f64) as u32
        })
        .collect();

    let zs: Vec<Complex64> = density
        .par_iter()
        .map(|d| {
            let val = d.load(Ordering::Relaxed);
            let normalized = (1.0 + val as f64).ln() / log_max;
            Complex64::new(normalized * 2.0, 0.0)
        })
        .collect();

    (iterations, zs)
}

/// Rendu Anti-Buddhabrot en précision MPC.
pub fn render_antibuddhabrot_mpc(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    let cancel = Arc::new(AtomicBool::new(false));
    render_antibuddhabrot_mpc_cancellable(params, &cancel)
        .unwrap_or_else(|| (Vec::new(), Vec::new()))
}

/// Version annulable du rendu Anti-Buddhabrot en MPC.
pub fn render_antibuddhabrot_mpc_cancellable(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    let width = params.width as usize;
    let height = params.height as usize;
    let size = width * height;

    if width == 0 || height == 0 {
        return Some((vec![0; size], vec![Complex64::new(0.0, 0.0); size]));
    }

    let prec = params.precision_bits.max(64);
    let xrange = params.span_x;
    let yrange = params.span_y;
    let iter_max = params.iteration_max;
    let bailout = Float::with_val(prec, params.bailout);
    let mut bailout_sq = bailout.clone();
    bailout_sq *= &bailout;

    let pixels = width * height;
    let num_samples = if pixels <= 640 * 480 {
        pixels * 20
    } else if pixels <= 1024 * 768 {
        pixels * 10
    } else {
        pixels * 5
    }
    .max(1000)
    .min(50_000_000);

    let density: Vec<AtomicU32> = (0..size).map(|_| AtomicU32::new(0)).collect();
    let cancelled = AtomicBool::new(false);

    (0..num_samples).into_par_iter().for_each(|sample_idx| {
        if sample_idx % 10000 == 0 {
            if cancel.load(Ordering::Relaxed) {
                cancelled.store(true, Ordering::Relaxed);
                return;
            }
        }
        if cancelled.load(Ordering::Relaxed) {
            return;
        }

        let mut rng = Rng::new((sample_idx as u32).wrapping_mul(12345).wrapping_add(42));
        let xg = params.center_x + (rng.next_f64() - 0.5) * xrange;
        let yg = params.center_y + (rng.next_f64() - 0.5) * yrange;
        let c = Complex::with_val(prec, (xg, yg));

        let mut trajectory: Vec<Complex64> = Vec::with_capacity(iter_max as usize);
        let mut z = Complex::with_val(prec, (0.0, 0.0));
        let mut escaped = false;

        for _iter in 0..iter_max {
            let mut z_next = z.clone();
            z_next *= &z;
            z_next += &c;
            z = z_next;

            if z.real().is_nan() || z.imag().is_nan() || z.real().is_infinite() || z.imag().is_infinite() {
                escaped = true;
                break;
            }

            trajectory.push(complex_to_complex64(&z));

            if complex_norm_sqr_mpc(&z, prec) > bailout_sq {
                escaped = true;
                break;
            }
        }

        if !escaped && !trajectory.is_empty() {
            let scale_x = width as f64 / xrange;
            let scale_y = height as f64 / yrange;

            for point in &trajectory {
                if point.re.is_nan() || point.im.is_nan() {
                    continue;
                }

                let px = ((point.re - params.center_x + xrange * 0.5) * scale_x) as i32;
                let py = ((point.im - params.center_y + yrange * 0.5) * scale_y) as i32;

                if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                    let idx = py as usize * width + px as usize;
                    density[idx].fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    });

    if cancelled.load(Ordering::Relaxed) {
        return None;
    }

    let max_density = density
        .iter()
        .map(|d| d.load(Ordering::Relaxed))
        .max()
        .unwrap_or(1)
        .max(1);
    let log_max = (1.0 + max_density as f64).ln();

    let iterations: Vec<u32> = density
        .par_iter()
        .map(|d| {
            let val = d.load(Ordering::Relaxed);
            let normalized = (1.0 + val as f64).ln() / log_max;
            (normalized * iter_max as f64) as u32
        })
        .collect();

    let zs: Vec<Complex64> = density
        .par_iter()
        .map(|d| {
            let val = d.load(Ordering::Relaxed);
            let normalized = (1.0 + val as f64).ln() / log_max;
            Complex64::new(normalized * 2.0, 0.0)
        })
        .collect();

    Some((iterations, zs))
}

/// Version annulable du rendu Anti-Buddhabrot (f64).
pub fn render_antibuddhabrot_cancellable(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    let width = params.width as usize;
    let height = params.height as usize;
    let size = width * height;

    if width == 0 || height == 0 {
        return Some((vec![0; size], vec![Complex64::new(0.0, 0.0); size]));
    }

    let xrange = params.span_x;
    let yrange = params.span_y;
    let iter_max = params.iteration_max;
    let bailout_sq = params.bailout * params.bailout;

    let pixels = width * height;
    let num_samples = if pixels <= 640 * 480 {
        pixels * 20
    } else if pixels <= 1024 * 768 {
        pixels * 10
    } else {
        pixels * 5
    }
    .max(1000)
    .min(50_000_000);

    let density: Vec<AtomicU32> = (0..size).map(|_| AtomicU32::new(0)).collect();
    let cancelled = AtomicBool::new(false);

    (0..num_samples).into_par_iter().for_each(|sample_idx| {
        if sample_idx % 10000 == 0 {
            if cancel.load(Ordering::Relaxed) {
                cancelled.store(true, Ordering::Relaxed);
                return;
            }
        }
        if cancelled.load(Ordering::Relaxed) {
            return;
        }

        let mut rng = Rng::new((sample_idx as u32).wrapping_mul(12345).wrapping_add(42));
        let xg = params.center_x + (rng.next_f64() - 0.5) * xrange;
        let yg = params.center_y + (rng.next_f64() - 0.5) * yrange;
        let c = Complex64::new(xg, yg);

        let mut trajectory = Vec::with_capacity(iter_max as usize);
        let mut z = Complex64::new(0.0, 0.0);
        let mut escaped = false;

        for _iter in 0..iter_max {
            z = z * z + c;

            if z.re.is_nan() || z.im.is_nan() || z.re.is_infinite() || z.im.is_infinite() {
                escaped = true;
                break;
            }

            trajectory.push(z);

            if z.norm_sqr() > bailout_sq {
                escaped = true;
                break;
            }
        }

        if !escaped && !trajectory.is_empty() {
            let scale_x = width as f64 / xrange;
            let scale_y = height as f64 / yrange;

            for point in &trajectory {
                if point.re.is_nan() || point.im.is_nan() {
                    continue;
                }

                let px = ((point.re - params.center_x + xrange * 0.5) * scale_x) as i32;
                let py = ((point.im - params.center_y + yrange * 0.5) * scale_y) as i32;

                if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                    let idx = py as usize * width + px as usize;
                    density[idx].fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    });

    if cancelled.load(Ordering::Relaxed) {
        return None;
    }

    let max_density = density
        .iter()
        .map(|d| d.load(Ordering::Relaxed))
        .max()
        .unwrap_or(1)
        .max(1);

    let log_max = (1.0 + max_density as f64).ln();

    let iterations: Vec<u32> = density
        .par_iter()
        .map(|d| {
            let val = d.load(Ordering::Relaxed);
            let normalized = (1.0 + val as f64).ln() / log_max;
            (normalized * iter_max as f64) as u32
        })
        .collect();

    let zs: Vec<Complex64> = density
        .par_iter()
        .map(|d| {
            let val = d.load(Ordering::Relaxed);
            let normalized = (1.0 + val as f64).ln() / log_max;
            Complex64::new(normalized * 2.0, 0.0)
        })
        .collect();

    Some((iterations, zs))
}
