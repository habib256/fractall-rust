//! Algorithmes Buddhabrot et Nebulabrot.
//!
//! Buddhabrot: visualise la densité des trajectoires d'échappement de z²+c.
//! Nebulabrot: version RGB avec différentes limites d'itérations par canal.

use num_complex::Complex64;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::fractal::FractalParams;

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

/// Rendu Buddhabrot.
///
/// Trace les trajectoires des points qui s'échappent et accumule
/// leur densité dans une matrice.
pub fn render_buddhabrot(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    let width = params.width as usize;
    let height = params.height as usize;
    let size = width * height;

    if width == 0 || height == 0 {
        return (vec![0; size], vec![Complex64::new(0.0, 0.0); size]);
    }

    let xmin = params.xmin;
    let xmax = params.xmax;
    let ymin = params.ymin;
    let ymax = params.ymax;
    let xrange = xmax - xmin;
    let yrange = ymax - ymin;
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
        let mut rng = Rng::new(42 + sample_idx as u32 * 12345);

        // Point aléatoire dans le domaine
        let xg = rng.next_f64() * xrange + xmin;
        let yg = rng.next_f64() * yrange + ymin;
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

                let px = ((point.re - xmin) * scale_x) as i32;
                let py = ((point.im - ymin) * scale_y) as i32;

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

/// Rendu Nebulabrot (RGB).
///
/// Version colorée du Buddhabrot avec différentes limites d'itérations:
/// - Rouge: 50 itérations (structures fines)
/// - Vert: 500 itérations (structures moyennes)
/// - Bleu: 5000 itérations (structures profondes)
///
/// Retourne directement une image RGB au lieu de iterations/zs.
pub fn render_nebulabrot(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    let width = params.width as usize;
    let height = params.height as usize;
    let size = width * height;

    if width == 0 || height == 0 {
        return (vec![0; size], vec![Complex64::new(0.0, 0.0); size]);
    }

    let xmin = params.xmin;
    let xmax = params.xmax;
    let ymin = params.ymin;
    let ymax = params.ymax;
    let xrange = xmax - xmin;
    let yrange = ymax - ymin;
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
        let mut rng = Rng::new(42 + sample_idx as u32 * 12345);

        let xg = rng.next_f64() * xrange + xmin;
        let yg = rng.next_f64() * yrange + ymin;
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
                let px = ((point.re - xmin) * scale_x) as i32;
                let py = ((point.im - ymin) * scale_y) as i32;

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
