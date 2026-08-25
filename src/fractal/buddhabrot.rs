//! Algorithmes Buddhabrot et Nebulabrot.
//!
//! Buddhabrot: visualise la densité des trajectoires d'échappement de z²+c.
//! Nebulabrot: version RGB avec différentes limites d'itérations par canal.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use num_complex::Complex64;
use rayon::prelude::*;
use rug::Complex;
use rug::Float;

use crate::fractal::FractalParams;

thread_local! {
    static TRAJ_BUF: RefCell<Vec<Complex64>> = RefCell::new(Vec::new());
    static MPC_PIXEL_BUF: RefCell<Vec<usize>> = RefCell::new(Vec::new());
}

struct MpcView {
    center_x: Float,
    center_y: Float,
    span_x: Float,
    span_y: Float,
    width: usize,
    height: usize,
    prec: u32,
}

impl MpcView {
    fn from_params(params: &FractalParams) -> Self {
        let prec = crate::fractal::perturbation::compute_perturbation_precision_bits(params)
            .max(params.precision_bits)
            .max(64);
        let parse = |hp: Option<&String>, fallback: f64| {
            hp.and_then(|value| Float::parse(value).ok())
                .map(|value| Float::with_val(prec, value))
                .unwrap_or_else(|| Float::with_val(prec, fallback))
        };
        Self {
            center_x: parse(params.center_x_hp.as_ref(), params.center_x),
            center_y: parse(params.center_y_hp.as_ref(), params.center_y),
            span_x: parse(params.span_x_hp.as_ref(), params.span_x),
            span_y: parse(params.span_y_hp.as_ref(), params.span_y),
            width: params.width as usize,
            height: params.height as usize,
            prec,
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn sample(&self, rx: f64, ry: f64) -> Complex {
        let mut x = self.span_x.clone();
        x *= rx - 0.5;
        x += &self.center_x;
        let mut y = self.span_y.clone();
        y *= ry - 0.5;
        y += &self.center_y;
        Complex::with_val(self.prec, (x, y))
    }

    fn pixel_index(&self, point: &Complex) -> Option<usize> {
        let mut x = Float::with_val(self.prec, point.real());
        x -= &self.center_x;
        x /= &self.span_x;
        x += 0.5;
        let mut y = Float::with_val(self.prec, point.imag());
        y -= &self.center_y;
        y /= &self.span_y;
        y += 0.5;
        let px = (x.to_f64() * self.width as f64).floor() as isize;
        let py = (y.to_f64() * self.height as f64).floor() as isize;
        (px >= 0 && px < self.width as isize && py >= 0 && py < self.height as isize)
            .then_some(py as usize * self.width + px as usize)
    }
}

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

/// Domaine d'échantillonnage des paramètres `c` des rendus de densité.
///
/// La densité Buddhabrot est définie sur le plan des `c` **entier** : la
/// fenêtre affichée ne sert qu'à PROJETER les trajectoires. Échantillonner les
/// `c` dans la vue (comportement historique) rendait toute navigation
/// dégénérée — dès un zoom ×100, les orbites quittent la fenêtre et n'y
/// reviennent jamais, donc l'image entière tombait à zéro.
const SAMPLE_CENTER_X: f64 = -0.5;
const SAMPLE_CENTER_Y: f64 = 0.0;
const SAMPLE_SPAN_X: f64 = 4.0;
const SAMPLE_SPAN_Y: f64 = 3.0;

/// Tire un `c` dans le domaine canonique (path f64).
#[inline]
fn sample_c_f64(rng: &mut Rng) -> Complex64 {
    let x = SAMPLE_CENTER_X + (rng.next_f64() - 0.5) * SAMPLE_SPAN_X;
    let y = SAMPLE_CENTER_Y + (rng.next_f64() - 0.5) * SAMPLE_SPAN_Y;
    Complex64::new(x, y)
}

/// Tire un `c` dans le domaine canonique (path MPC). L'ordre des opérations
/// reproduit celui de [`MpcView::sample`] pour que la vue par défaut — dont le
/// domaine EST la vue — reste bit-identique.
#[inline]
fn sample_c_mpc(rng: &mut Rng, prec: u32) -> Complex {
    let mut x = Float::with_val(prec, SAMPLE_SPAN_X);
    x *= rng.next_f64() - 0.5;
    x += SAMPLE_CENTER_X;
    let mut y = Float::with_val(prec, SAMPLE_SPAN_Y);
    y *= rng.next_f64() - 0.5;
    y += SAMPLE_CENTER_Y;
    Complex::with_val(prec, (x, y))
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

    let view = MpcView::from_params(params);
    let prec = view.prec;
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
        // Le `c` est tiré sur le domaine canonique, pas dans la vue.
        let c = sample_c_mpc(&mut rng, prec);

        MPC_PIXEL_BUF.with(|buf| {
            let mut trajectory = buf.borrow_mut();
            trajectory.clear();
            let mut z = Complex::with_val(prec, (0.0, 0.0));
            let mut escaped = false;

            for iter in 0..iter_max {
                let mut z_next = z.clone();
                z_next *= &z;
                z_next += &c;
                z = z_next;

                if z.real().is_nan()
                    || z.imag().is_nan()
                    || z.real().is_infinite()
                    || z.imag().is_infinite()
                {
                    break;
                }

                let mag2 = complex_norm_sqr_mpc(&z, prec);
                if iter == early_exit_threshold && mag2 < early_exit_limit {
                    break;
                }

                if let Some(idx) = view.pixel_index(&z) {
                    trajectory.push(idx);
                }

                if mag2 > bailout_sq {
                    escaped = true;
                    break;
                }
            }

            if escaped && !trajectory.is_empty() {
                for &idx in trajectory.iter() {
                    density[idx].fetch_add(1, Ordering::Relaxed);
                }
            }
        });
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

#[cfg(test)]
mod deep_zoom_tests {
    use super::*;
    use crate::fractal::{default_params_for_type, FractalType};

    #[test]
    fn mpc_density_view_preserves_sub_f64_pixels() {
        let mut params = default_params_for_type(FractalType::Buddhabrot, 100, 80);
        params.center_x_hp = Some("-0.743643887037158704752191506114774".into());
        params.center_y_hp = Some("0.131825904205311970493132056385139".into());
        params.span_x_hp = Some("4e-40".into());
        params.span_y_hp = Some("3.2e-40".into());
        params.center_x = -0.7436438870371587;
        params.center_y = 0.13182590420531198;
        params.span_x = 4e-40;
        params.span_y = 3.2e-40;

        let view = MpcView::from_params(&params);
        let left = view.sample(0.25, 0.5);
        let right = view.sample(0.75, 0.5);
        assert_ne!(left.real(), right.real());
        assert_eq!(view.pixel_index(&left), Some(40 * 100 + 25));
        assert_eq!(view.pixel_index(&right), Some(40 * 100 + 75));
    }

    /// Les deux tireurs de `c` portent les mêmes constantes de domaine : ils
    /// doivent produire le MÊME point pour une même graine, sinon les paths
    /// f64 et MPC échantillonneraient des champs différents.
    #[test]
    fn f64_and_mpc_samplers_share_the_canonical_domain() {
        for seed in [0u32, 1, 7, 4242] {
            let mut rng_f64 = Rng::new(seed);
            let mut rng_mpc = Rng::new(seed);
            let sampled_f64 = sample_c_f64(&mut rng_f64);
            let sampled_mpc = sample_c_mpc(&mut rng_mpc, 128);
            assert_eq!(sampled_mpc.real().to_f64(), sampled_f64.re, "seed {seed}");
            assert_eq!(sampled_mpc.imag().to_f64(), sampled_f64.im, "seed {seed}");
            assert!(
                (sampled_f64.re - SAMPLE_CENTER_X).abs() <= SAMPLE_SPAN_X / 2.0
                    && (sampled_f64.im - SAMPLE_CENTER_Y).abs() <= SAMPLE_SPAN_Y / 2.0,
                "seed {seed} : tirage hors du domaine canonique"
            );
        }
    }

    /// Le domaine d'échantillonnage ne doit dépendre d'AUCUN paramètre de vue :
    /// c'est ce découplage qui rend la navigation possible.
    #[test]
    fn sampling_domain_ignores_the_rendered_view() {
        assert_eq!(
            (
                SAMPLE_CENTER_X,
                SAMPLE_CENTER_Y,
                SAMPLE_SPAN_X,
                SAMPLE_SPAN_Y
            ),
            (-0.5, 0.0, 4.0, 3.0),
            "le domaine canonique doit rester la vue par défaut des types densité"
        );
        let defaults = default_params_for_type(FractalType::Buddhabrot, 64, 48);
        assert_eq!(
            (
                defaults.center_x,
                defaults.center_y,
                defaults.span_x,
                defaults.span_y
            ),
            (
                SAMPLE_CENTER_X,
                SAMPLE_CENTER_Y,
                SAMPLE_SPAN_X,
                SAMPLE_SPAN_Y
            ),
            "vue par défaut ≠ domaine : le rendu par défaut ne serait plus complet"
        );
    }
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

    let view = MpcView::from_params(params);
    let prec = view.prec;
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
        // Le `c` est tiré sur le domaine canonique, pas dans la vue.
        let c = sample_c_mpc(&mut rng, prec);

        MPC_PIXEL_BUF.with(|buf| {
            let mut trajectory = buf.borrow_mut();
            trajectory.clear();
            let mut z = Complex::with_val(prec, (0.0, 0.0));
            let mut escaped = false;
            let mut escape_iter = 0u32;

            for iter in 0..ITER_MAX {
                let mut z_next = z.clone();
                z_next *= &z;
                z_next += &c;
                z = z_next;

                if z.real().is_nan()
                    || z.imag().is_nan()
                    || z.real().is_infinite()
                    || z.imag().is_infinite()
                {
                    break;
                }

                if let Some(idx) = view.pixel_index(&z) {
                    trajectory.push(idx);
                }

                if complex_norm_sqr_mpc(&z, prec) > bailout_sq {
                    escaped = true;
                    escape_iter = iter;
                    break;
                }
            }

            if escaped && !trajectory.is_empty() {
                let contribute_r = escape_iter <= ITER_R;
                let contribute_g = escape_iter <= ITER_G;
                let contribute_b = escape_iter <= ITER_B;

                for &idx in trajectory.iter() {
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
        });
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
        // Le `c` est tiré sur le domaine canonique, pas dans la vue.
        let c = sample_c_f64(&mut rng);

        TRAJ_BUF.with(|buf| {
            let mut trajectory = buf.borrow_mut();
            trajectory.clear();
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

                for point in trajectory.iter() {
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
        // Le `c` est tiré sur le domaine canonique, pas dans la vue.
        let c = sample_c_f64(&mut rng);

        TRAJ_BUF.with(|buf| {
            let mut trajectory = buf.borrow_mut();
            trajectory.clear();
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

                for &point in trajectory.iter() {
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

// ─────────────────────────────────────────────────────────────────────────────
// Anti-Buddhabrot : accumule les orbites des points INTÉRIEURS (non-escapés).
// ─────────────────────────────────────────────────────────────────────────────

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

    let view = MpcView::from_params(params);
    let prec = view.prec;
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
        let c = sample_c_mpc(&mut rng, prec);

        MPC_PIXEL_BUF.with(|buf| {
            let mut trajectory = buf.borrow_mut();
            trajectory.clear();
            let mut z = Complex::with_val(prec, (0.0, 0.0));
            let mut escaped = false;

            for _iter in 0..iter_max {
                let mut z_next = z.clone();
                z_next *= &z;
                z_next += &c;
                z = z_next;

                if z.real().is_nan()
                    || z.imag().is_nan()
                    || z.real().is_infinite()
                    || z.imag().is_infinite()
                {
                    escaped = true;
                    break;
                }

                if let Some(idx) = view.pixel_index(&z) {
                    trajectory.push(idx);
                }

                if complex_norm_sqr_mpc(&z, prec) > bailout_sq {
                    escaped = true;
                    break;
                }
            }

            if !escaped && !trajectory.is_empty() {
                for &idx in trajectory.iter() {
                    density[idx].fetch_add(1, Ordering::Relaxed);
                }
            }
        });
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
        let c = sample_c_f64(&mut rng);

        TRAJ_BUF.with(|buf| {
            let mut trajectory = buf.borrow_mut();
            trajectory.clear();
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

                for point in trajectory.iter() {
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
