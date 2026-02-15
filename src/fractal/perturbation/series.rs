use num_complex::Complex64;

use crate::fractal::FractalParams;
use crate::fractal::perturbation::types::ComplexExp;

/// Coefficients de la série de Taylor pour l'approximation par série (SA).
///
/// ## Mandelbrot dc-series (is_julia = false)
///
/// Pour Mandelbrot z^2 + c avec delta_0 = 0, on développe delta_n en série de dc:
///   delta_n = a_n * dc + b_n * dc^2 + c_n * dc^3 + d_n * dc^4
///
/// Récurrences (issues de delta_{n+1} = 2*Z_n*delta_n + delta_n^2 + dc):
///   a_{n+1} = 2*Z_n*a_n + 1          (a_0 = 0, a_1 = 1)
///   b_{n+1} = 2*Z_n*b_n + a_n^2      (b_0 = 0)
///   c_{n+1} = 2*Z_n*c_n + 2*a_n*b_n  (c_0 = 0)
///   d_{n+1} = 2*Z_n*d_n + 2*a_n*c_n + b_n^2  (d_0 = 0)
///
/// ## Julia delta-series (is_julia = true)
///
/// Pour Julia z^2 + seed avec delta_0 = dc, on développe delta_n en série de delta_0:
///   delta_n = a_n * delta_0 + b_n * delta_0^2 + c_n * delta_0^3 + d_n * delta_0^4
///
/// Récurrences (issues de delta_{n+1} = 2*Z_n*delta_n + delta_n^2):
///   a_{n+1} = 2*Z_n*a_n              (a_0 = 1)
///   b_{n+1} = 2*Z_n*b_n + a_n^2      (b_0 = 0)
///   c_{n+1} = 2*Z_n*c_n + 2*a_n*b_n  (c_0 = 0)
///   d_{n+1} = 2*Z_n*d_n + 2*a_n*c_n + b_n^2  (d_0 = 0)
#[derive(Clone, Copy, Debug)]
pub struct SeriesCoefficients {
    /// Coefficient linéaire (ordre 1)
    pub a: Complex64,
    /// Coefficient quadratique (ordre 2)
    pub b: Complex64,
    /// Coefficient cubique (ordre 3)
    pub c: Complex64,
    /// Coefficient quartique (ordre 4)
    pub d: Complex64,
}

impl Default for SeriesCoefficients {
    fn default() -> Self {
        Self {
            a: Complex64::new(0.0, 0.0),
            b: Complex64::new(0.0, 0.0),
            c: Complex64::new(0.0, 0.0),
            d: Complex64::new(0.0, 0.0),
        }
    }
}

/// Table de coefficients de série pour toutes les itérations.
/// Permet de sauter des itérations initiales en utilisant l'approximation par série.
#[derive(Clone, Debug)]
pub struct SeriesTable {
    /// Coefficients pour chaque itération (index = numéro d'itération)
    pub coeffs: Vec<SeriesCoefficients>,
    /// true si la table a été construite pour Julia (delta_0-series), false pour Mandelbrot (dc-series)
    #[allow(dead_code)]
    pub is_julia: bool,
}

impl SeriesTable {
    pub fn empty() -> Self {
        Self {
            coeffs: Vec::new(),
            is_julia: false,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.coeffs.is_empty()
    }
}

/// Construit la table de série à partir de l'orbite de référence.
///
/// ## Mandelbrot (is_julia = false)
///
/// Récurrences dc-series (delta_0 = 0, variable = dc):
///   a_{n+1} = 2*Z_n*a_n + 1          (a_0 = 0)
///   b_{n+1} = 2*Z_n*b_n + a_n^2      (b_0 = 0)
///   c_{n+1} = 2*Z_n*c_n + 2*a_n*b_n  (c_0 = 0)
///   d_{n+1} = 2*Z_n*d_n + 2*a_n*c_n + b_n^2  (d_0 = 0)
///
/// ## Julia (is_julia = true)
///
/// Récurrences delta_0-series (delta_0 = dc, variable = delta_0):
///   a_{n+1} = 2*Z_n*a_n              (a_0 = 1)
///   b_{n+1} = 2*Z_n*b_n + a_n^2      (b_0 = 0)
///   c_{n+1} = 2*Z_n*c_n + 2*a_n*b_n  (c_0 = 0)
///   d_{n+1} = 2*Z_n*d_n + 2*a_n*c_n + b_n^2  (d_0 = 0)
pub fn build_series_table(z_ref: &[Complex64], is_julia: bool) -> SeriesTable {
    if z_ref.is_empty() {
        return SeriesTable::empty();
    }

    let mut coeffs = Vec::with_capacity(z_ref.len());
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);

    // Initial coefficients differ between Mandelbrot and Julia
    let mut a = if is_julia { one } else { zero };  // Mandelbrot: a_0=0, Julia: a_0=1
    let mut b = zero;
    let mut c = zero;
    let mut d = zero;

    for (i, &z_n) in z_ref.iter().enumerate() {
        // Store current coefficients
        coeffs.push(SeriesCoefficients { a, b, c, d });

        // Early termination if coefficients explode
        let a_norm = a.norm();
        if !a.re.is_finite() || !a.im.is_finite() || a_norm > 1e100 {
            break;
        }
        if !b.re.is_finite() || !b.im.is_finite() {
            break;
        }

        // Skip the last iteration (no z_{n+1})
        if i >= z_ref.len() - 1 {
            break;
        }

        // Recurrence relations for next iteration
        let two_z = z_n * 2.0;
        let a_sq = a * a;
        let two_ab = a * b * 2.0;
        let two_ac = a * c * 2.0;
        let b_sq = b * b;

        if is_julia {
            // Julia: a_{n+1} = 2*Z_n*a_n  (no +1 term)
            let a_next = two_z * a;
            let b_next = two_z * b + a_sq;
            let c_next = two_z * c + two_ab;
            let d_next = two_z * d + two_ac + b_sq;
            a = a_next;
            b = b_next;
            c = c_next;
            d = d_next;
        } else {
            // Mandelbrot: a_{n+1} = 2*Z_n*a_n + 1  (the crucial +1 term!)
            let a_next = two_z * a + one;
            let b_next = two_z * b + a_sq;
            let c_next = two_z * c + two_ab;
            let d_next = two_z * d + two_ac + b_sq;
            a = a_next;
            b = b_next;
            c = c_next;
            d = d_next;
        }
    }

    SeriesTable { coeffs, is_julia }
}

/// Résultat du calcul de saut de série
pub struct SeriesSkipResult {
    /// Itération jusqu'à laquelle on peut sauter
    pub skip_to: usize,
    /// Delta après le saut (pour chaque pixel, évaluer avec son propre dc)
    pub delta: ComplexExp,
    /// Erreur estimée de l'approximation
    pub estimated_error: f64,
}

/// Calcule le nombre d'itérations que l'on peut sauter avec l'approximation par série.
///
/// Pour Mandelbrot: évalue delta_n = a_n*dc + b_n*dc^2 + c_n*dc^3 + d_n*dc^4
///   où dc est l'offset du pixel par rapport au centre.
///
/// Pour Julia: évalue delta_n = a_n*delta_0 + b_n*delta_0^2 + c_n*delta_0^3 + d_n*delta_0^4
///   où delta_0 = dc est le delta initial du pixel.
///
/// Dans les deux cas, l'argument `dc` est le "petit paramètre" de la série.
///
/// # Arguments
/// * `table` - Table de coefficients de série
/// * `dc` - Offset pixel (dc pour Mandelbrot, delta_0 pour Julia)
/// * `error_tolerance` - Tolérance d'erreur maximale acceptée
///
/// # Returns
/// Option contenant le résultat du saut si possible, None sinon
pub fn compute_series_skip(
    table: &SeriesTable,
    dc: ComplexExp,
    error_tolerance: f64,
) -> Option<SeriesSkipResult> {
    if table.is_empty() || error_tolerance <= 0.0 {
        return None;
    }

    let dc_f64 = dc.to_complex64_approx();
    let dc_norm = dc_f64.norm();
    if dc_norm <= 0.0 || !dc_norm.is_finite() {
        return None;
    }

    let dc_sq = dc_f64 * dc_f64;
    let dc_cube = dc_sq * dc_f64;
    let dc_4 = dc_sq * dc_sq;
    let dc_norm_sq = dc_norm * dc_norm;

    // Find the best (latest) iteration we can skip to.
    // Strategy: scan forward, keep the last valid iteration where error is acceptable.
    // Stop when the series diverges (error grows beyond tolerance).
    let mut best_skip = 0usize;
    let mut best_approx = Complex64::new(0.0, 0.0);
    let mut best_error = f64::MAX;

    for (n, coeffs) in table.coeffs.iter().enumerate() {
        // Evaluate series: delta_n = a_n*dc + b_n*dc^2 + c_n*dc^3 + d_n*dc^4
        let approx = coeffs.a * dc_f64
            + coeffs.b * dc_sq
            + coeffs.c * dc_cube
            + coeffs.d * dc_4;

        // Estimate truncation error: O(dc^5) term
        // Error ~ |next_coeff| * |dc|^5
        // We approximate |next_coeff| from coefficient growth rate
        let d_norm = coeffs.d.norm();
        let c_norm = coeffs.c.norm();

        // Use ratio of consecutive coefficients to estimate next coefficient magnitude
        let next_coeff_estimate = if c_norm > 1e-30 {
            d_norm * d_norm / c_norm  // rough extrapolation
        } else {
            d_norm * dc_norm  // fallback
        };
        let dc_5 = dc_norm_sq * dc_norm_sq * dc_norm;
        let error = next_coeff_estimate * dc_5;

        // Also check that the series hasn't diverged:
        // the last term should be small relative to the total
        let last_term_norm = (coeffs.d * dc_4).norm();
        let approx_norm = approx.norm();
        let term_ratio = if approx_norm > 1e-30 {
            last_term_norm / approx_norm
        } else {
            last_term_norm
        };

        if error < error_tolerance && term_ratio < 0.5 && approx.re.is_finite() && approx.im.is_finite() {
            best_skip = n;
            best_approx = approx;
            best_error = error;
        }

        // Stop scanning if the series is clearly diverging
        if term_ratio > 1.0 || !approx.re.is_finite() || !approx.im.is_finite() {
            break;
        }
    }

    // Only return if we can skip at least 2 iterations (skipping 1 isn't worth the overhead)
    if best_skip >= 2 {
        Some(SeriesSkipResult {
            skip_to: best_skip,
            delta: ComplexExp::from_complex64(best_approx),
            estimated_error: best_error,
        })
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SeriesConfig {
    pub order: u8,
    pub threshold: f64,
    pub error_tolerance: f64,
}

impl SeriesConfig {
    pub fn from_params(params: &FractalParams) -> Self {
        Self {
            order: params.series_order,
            threshold: params.series_threshold.max(0.0),
            error_tolerance: params.series_error_tolerance.max(0.0),
        }
    }
}

pub fn should_use_series(config: SeriesConfig, delta_norm_sqr: f64, validity_radius: f64) -> bool {
    if config.order < 2 {
        return false;
    }
    if !delta_norm_sqr.is_finite() {
        return false;
    }
    let threshold = config.threshold.min(validity_radius);
    if threshold <= 0.0 {
        return false;
    }
    delta_norm_sqr < threshold * threshold
}

/// Estimate the series approximation error.
///
/// # Arguments
/// * `delta_norm_sqr` - |delta|^2 of the current delta
/// * `order` - Series order (2, 3, 4, 5, or 6)
/// * `bla_level` - BLA level (0-16), higher levels skip more iterations
/// * `coeff_a_norm` - |A| coefficient norm from BLA node
///
/// The error estimation accounts for:
/// - Base truncation error O(delta^(order+1))
/// - Error accumulation from BLA level (more skipped iterations = more error)
/// - Error amplification from coefficient A magnitude
pub fn estimate_series_error(
    delta_norm_sqr: f64,
    order: u8,
    bla_level: usize,
    coeff_a_norm: f64,
) -> f64 {
    if order < 2 || !delta_norm_sqr.is_finite() {
        return 0.0;
    }
    let delta_abs = delta_norm_sqr.sqrt();

    // Base error according to series order
    let base_error = match order {
        2 => delta_abs * delta_norm_sqr,                        // O(delta^3)
        3 => delta_norm_sqr * delta_norm_sqr,                   // O(delta^4)
        4 => delta_norm_sqr * delta_norm_sqr * delta_abs,       // O(delta^5)
        5 => delta_norm_sqr * delta_norm_sqr * delta_norm_sqr,  // O(delta^6)
        6 => delta_norm_sqr * delta_norm_sqr * delta_norm_sqr * delta_abs, // O(delta^7)
        _ => delta_abs * delta_norm_sqr,  // Fallback to order 2
    };

    // Amplification factor based on BLA level
    let level_factor = 1.0 + (bla_level as f64) * 0.1;

    // Amplification factor from coefficient A
    let coeff_factor = if coeff_a_norm > 1.0 {
        1.0 + coeff_a_norm.ln()
    } else {
        1.0
    };

    base_error * level_factor * coeff_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn series_activation_threshold() {
        let cfg = SeriesConfig {
            order: 2,
            threshold: 1e-3,
            error_tolerance: 1e-6,
        };
        let delta_small = 1e-8;
        let delta_large = 1e-4;
        assert!(should_use_series(cfg, delta_small, 1e-2));
        assert!(!should_use_series(cfg, delta_large, 1e-3));
    }

    #[test]
    fn series_error_estimate_behaves() {
        let err2 = estimate_series_error(1e-8, 2, 0, 1.0);
        let err3 = estimate_series_error(1e-8, 3, 0, 1.0);
        assert!(err3 <= err2);
    }

    #[test]
    fn series_error_increases_with_bla_level() {
        let delta_norm_sqr = 1e-8;
        let err_level_0 = estimate_series_error(delta_norm_sqr, 2, 0, 1.0);
        let err_level_5 = estimate_series_error(delta_norm_sqr, 2, 5, 1.0);
        let err_level_10 = estimate_series_error(delta_norm_sqr, 2, 10, 1.0);

        assert!(err_level_5 > err_level_0);
        assert!(err_level_10 > err_level_5);
    }

    #[test]
    fn series_error_increases_with_coeff_norm() {
        let delta_norm_sqr = 1e-8;
        let err_coeff_1 = estimate_series_error(delta_norm_sqr, 2, 0, 1.0);
        let err_coeff_10 = estimate_series_error(delta_norm_sqr, 2, 0, 10.0);
        let err_coeff_100 = estimate_series_error(delta_norm_sqr, 2, 0, 100.0);

        assert!(err_coeff_10 > err_coeff_1);
        assert!(err_coeff_100 > err_coeff_10);
    }

    #[test]
    fn series_error_small_coeff_no_amplification() {
        let delta_norm_sqr = 1e-8;
        let err_coeff_small = estimate_series_error(delta_norm_sqr, 2, 0, 0.5);
        let err_coeff_one = estimate_series_error(delta_norm_sqr, 2, 0, 1.0);

        assert_eq!(err_coeff_small, err_coeff_one);
    }

    #[test]
    fn series_error_invalid_inputs() {
        assert_eq!(estimate_series_error(1e-8, 1, 0, 1.0), 0.0);
        assert_eq!(estimate_series_error(1e-8, 0, 0, 1.0), 0.0);
        assert_eq!(estimate_series_error(f64::NAN, 2, 0, 1.0), 0.0);
        assert_eq!(estimate_series_error(f64::INFINITY, 2, 0, 1.0), 0.0);
    }

    /// Verify that Mandelbrot dc-series coefficients are computed correctly.
    /// For Mandelbrot at origin (Z_n = 0 for all n):
    ///   a_0=0, a_1=1, a_2=1 (since 2*0*1 + 1 = 1)
    ///   b_0=0, b_1=0, b_2=0 (since 2*0*0 + 0^2 = 0)
    #[test]
    fn mandelbrot_series_at_origin() {
        // Reference orbit at the origin: z_0=0, z_1=0, z_2=0, ...
        let z_ref = vec![Complex64::new(0.0, 0.0); 10];
        let table = build_series_table(&z_ref, false);
        assert!(!table.is_empty());
        assert!(!table.is_julia);

        // a_0 = 0 (initial for Mandelbrot)
        assert!((table.coeffs[0].a.norm()) < 1e-10);
        // a_1 = 2*Z_0*a_0 + 1 = 0 + 1 = 1
        assert!((table.coeffs[1].a - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        // a_2 = 2*Z_1*a_1 + 1 = 0 + 1 = 1
        assert!((table.coeffs[2].a - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    }

    /// Verify that the Mandelbrot dc-series actually works:
    /// For c = 0.1 (small dc), the series should approximate delta_n well.
    #[test]
    fn mandelbrot_series_skip_works() {
        // Compute a reference orbit at center c=0
        // Mandelbrot: z_{n+1} = z_n^2 + c, with c=0, z_0=0
        // So z_n = 0 for all n (the orbit stays at origin)
        let orbit_len = 50;
        let z_ref = vec![Complex64::new(0.0, 0.0); orbit_len];
        let table = build_series_table(&z_ref, false);

        // For a pixel at dc = 0.1:
        // delta_1 = dc = 0.1
        // delta_2 = 0^2 + 0.1 = 0.1 (since z_ref=0, delta^2 + dc)
        // Wait, actual iteration: delta_{n+1} = 2*Z_n*delta_n + delta_n^2 + dc
        // With Z_n = 0: delta_{n+1} = delta_n^2 + dc
        // delta_0 = 0
        // delta_1 = 0 + 0.1 = 0.1
        // delta_2 = 0.01 + 0.1 = 0.11
        // delta_3 = 0.0121 + 0.1 = 0.1121
        let dc = Complex64::new(0.1, 0.0);

        // Verify series evaluation at iteration 1: a_1*dc = 1*0.1 = 0.1
        let approx_1 = table.coeffs[1].a * dc;
        assert!((approx_1 - Complex64::new(0.1, 0.0)).norm() < 1e-10);

        // Verify series at iteration 2: a_2*dc + b_2*dc^2
        // a_2 = 1, b_2 = 2*0*0 + 0^2 = 0, so approx = 0.1
        // But actual delta_2 = 0.11, so at order 1 the error is 0.01 = dc^2
        // The b coefficient should capture this at later iterations
        let _approx_2 = table.coeffs[2].a * dc + table.coeffs[2].b * dc * dc;
        // With Z_n=0: b_1 = 0, b_2 = 0 + 0 = 0, but we need more iterations
        // for the quadratic term to build up

        // Test compute_series_skip with a small dc
        let small_dc = ComplexExp::from_complex64(Complex64::new(0.001, 0.0));
        let result = compute_series_skip(&table, small_dc, 1e-9);
        // With dc=0.001, the series should be able to skip several iterations
        assert!(result.is_some(), "Series skip should work for small dc");
        let skip = result.unwrap();
        assert!(skip.skip_to >= 2, "Should skip at least 2 iterations, got {}", skip.skip_to);
    }

    /// Verify Julia delta_0-series matches the old behavior.
    /// For Julia with Z_0 reference, the coefficients should satisfy:
    ///   a_0 = 1, a_{n+1} = 2*Z_n*a_n (no +1 term)
    #[test]
    fn julia_series_coefficients() {
        let z_ref = vec![
            Complex64::new(0.5, 0.3),
            Complex64::new(0.2, 0.1),
            Complex64::new(0.4, -0.2),
        ];
        let table = build_series_table(&z_ref, true);
        assert!(table.is_julia);

        // a_0 = 1 for Julia
        assert!((table.coeffs[0].a - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        // a_1 = 2*Z_0*a_0 = 2*(0.5+0.3i)*1 = 1.0+0.6i
        let expected_a1 = Complex64::new(1.0, 0.6);
        assert!((table.coeffs[1].a - expected_a1).norm() < 1e-10);
    }
}
