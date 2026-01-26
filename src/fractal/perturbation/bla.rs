use num_complex::Complex64;

use crate::fractal::{FractalParams, FractalType};

#[derive(Clone, Copy, Debug)]
pub struct BlaNode {
    pub a: Complex64,
    pub b: Complex64,
    pub c: Complex64,
    pub validity_radius: f64,
}

#[derive(Clone, Debug)]
pub struct BlaTable {
    pub levels: Vec<Vec<BlaNode>>,
}

impl BlaTable {
    pub fn empty() -> Self {
        Self { levels: Vec::new() }
    }
}

/// Compute BLA coefficients (A, B, C) for a single iteration at z_ref.
/// For z^d + c iteration: δ' = A·δ + B·dc + C·δ²
/// where A = d·z^(d-1), B = 1 (Mandelbrot) or 0 (Julia), C = d(d-1)/2·z^(d-2)
fn compute_bla_coefficients(
    z: Complex64,
    fractal_type: FractalType,
    power: f64,
) -> (Complex64, Complex64, Complex64) {
    let is_julia = fractal_type == FractalType::Julia;

    match fractal_type {
        FractalType::Mandelbrot | FractalType::Julia => {
            // Standard z² + c: A = 2z, C = 1
            let a = z * 2.0;
            let b = if is_julia {
                Complex64::new(0.0, 0.0)
            } else {
                Complex64::new(1.0, 0.0)
            };
            let c = Complex64::new(1.0, 0.0);
            (a, b, c)
        }
        FractalType::Multibrot => {
            // z^d + c: A = d·z^(d-1), C = d(d-1)/2·z^(d-2)
            let d = power;
            let z_norm = z.norm();

            // For very small z, use simplified coefficients to avoid numerical issues
            if z_norm < 1e-15 {
                let a = Complex64::new(0.0, 0.0);
                let b = Complex64::new(1.0, 0.0);
                // For d > 2, z^(d-2) → ∞ as z → 0, but δ² term is negligible
                let c = Complex64::new(0.0, 0.0);
                return (a, b, c);
            }

            // A = d · z^(d-1)
            let a = z.powf(d - 1.0) * d;

            let b = Complex64::new(1.0, 0.0);

            // C = d(d-1)/2 · z^(d-2)
            let c_coeff = d * (d - 1.0) / 2.0;
            let c = if (d - 2.0).abs() < 1e-10 {
                // d ≈ 2: z^0 = 1
                Complex64::new(c_coeff, 0.0)
            } else {
                z.powf(d - 2.0) * c_coeff
            };

            // Validate coefficients
            let a = if a.re.is_finite() && a.im.is_finite() {
                a
            } else {
                z * 2.0 // Fallback to Mandelbrot
            };
            let c = if c.re.is_finite() && c.im.is_finite() {
                c
            } else {
                Complex64::new(1.0, 0.0)
            };

            (a, b, c)
        }
        _ => {
            // Fallback: use Mandelbrot coefficients
            let a = z * 2.0;
            let b = Complex64::new(1.0, 0.0);
            let c = Complex64::new(1.0, 0.0);
            (a, b, c)
        }
    }
}

pub fn build_bla_table(ref_orbit: &[Complex64], params: &FractalParams) -> BlaTable {
    let supports_bla = matches!(
        params.fractal_type,
        FractalType::Mandelbrot | FractalType::Julia | FractalType::Multibrot
    );
    if !supports_bla {
        return BlaTable::empty();
    }

    let base_len = ref_orbit.len().saturating_sub(1);
    if base_len == 0 {
        return BlaTable::empty();
    }

    let mut levels: Vec<Vec<BlaNode>> = Vec::new();
    let mut level0 = Vec::with_capacity(base_len);
    let base_threshold = params.bla_threshold.max(1e-16);
    let validity_scale = params.bla_validity_scale.clamp(0.1, 100.0);
    let power = params.multibrot_power;

    // Cap maximum validity at a reasonable multiple of base_threshold
    // This allows larger radii when z_ref is small, while preventing unbounded growth
    let max_validity = base_threshold * validity_scale * 10.0;

    for &z in ref_orbit.iter().take(base_len) {
        let (a, b, c) = compute_bla_coefficients(z, params.fractal_type, power);
        let a_norm = a.norm();

        // When a_norm is small, validity can exceed base_threshold
        // The formula: threshold / (1 + |a|) scales inversely with derivative magnitude
        let mut validity = (base_threshold * validity_scale) / (1.0 + a_norm);
        if !validity.is_finite() {
            validity = base_threshold * validity_scale;
        }
        // Allow validity to exceed base_threshold when z_ref is small (a_norm small)
        // but cap at max_validity to prevent numerical issues
        validity = validity.min(max_validity);

        level0.push(BlaNode {
            a,
            b,
            c,
            validity_radius: validity,
        });
    }
    levels.push(level0);

    let max_level = 16usize;
    for level in 1..=max_level {
        let step = 1usize << (level - 1);
        let prev = &levels[level - 1];
        if prev.len() <= step {
            break;
        }
        let mut current = Vec::with_capacity(prev.len() - step);
        for i in 0..(prev.len() - step) {
            let node1 = prev[i];
            let node2 = prev[i + step];
            let a_new = node2.a * node1.a;
            let b_new = node2.a * node1.b + node2.b;
            let c_new = node2.a * node1.c + node2.c * (node1.a * node1.a);
            let a_norm = node1.a.norm();
            let scaled = node2.validity_radius / (1.0 + a_norm);
            current.push(BlaNode {
                a: a_new,
                b: b_new,
                c: c_new,
                validity_radius: node1.validity_radius.min(scaled),
            });
        }
        levels.push(current);
    }

    BlaTable { levels }
}
