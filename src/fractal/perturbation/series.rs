use num_complex::Complex64;

use crate::fractal::FractalParams;
use crate::fractal::perturbation::types::ComplexExp;

/// Coefficients de série de Taylor pour une itération.
/// Pour z' = z² + c, la série est: δ'_n = A_n·δ + B_n·δ² + C_n·δ³
/// où les coefficients évoluent selon les récurrences:
/// - A_{n+1} = 2·z_n·A_n (init A_0 = 1)
/// - B_{n+1} = 2·z_n·B_n + A_n² (init B_0 = 0)
/// - C_{n+1} = 2·z_n·C_n + 2·A_n·B_n (init C_0 = 0)
#[derive(Clone, Copy, Debug)]
pub struct SeriesCoefficients {
    /// Coefficient linéaire A: δ'_n ≈ A·δ
    pub a: Complex64,
    /// Coefficient quadratique B: δ'_n ≈ A·δ + B·δ²
    pub b: Complex64,
    /// Coefficient cubique C: δ'_n ≈ A·δ + B·δ² + C·δ³
    pub c: Complex64,
}

impl Default for SeriesCoefficients {
    fn default() -> Self {
        Self {
            a: Complex64::new(1.0, 0.0),
            b: Complex64::new(0.0, 0.0),
            c: Complex64::new(0.0, 0.0),
        }
    }
}

/// Table de coefficients de série pour toutes les itérations.
/// Permet de sauter des itérations initiales en utilisant l'approximation par série.
#[derive(Clone, Debug)]
pub struct SeriesTable {
    /// Coefficients pour chaque itération (index = numéro d'itération)
    pub coeffs: Vec<SeriesCoefficients>,
    /// Rayon de validité estimé pour chaque itération
    pub validity_radii: Vec<f64>,
}

impl SeriesTable {
    pub fn empty() -> Self {
        Self {
            coeffs: Vec::new(),
            validity_radii: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.coeffs.is_empty()
    }
}

/// Construit la table de série à partir de l'orbite de référence.
///
/// Les coefficients évoluent selon les récurrences:
/// - A_{n+1} = 2·z_n·A_n
/// - B_{n+1} = 2·z_n·B_n + A_n²
/// - C_{n+1} = 2·z_n·C_n + 2·A_n·B_n
pub fn build_series_table(z_ref: &[Complex64]) -> SeriesTable {
    if z_ref.is_empty() {
        return SeriesTable::empty();
    }

    let mut coeffs = Vec::with_capacity(z_ref.len());
    let mut validity_radii = Vec::with_capacity(z_ref.len());

    // Initial coefficients: A_0 = 1, B_0 = 0, C_0 = 0
    let mut a = Complex64::new(1.0, 0.0);
    let mut b = Complex64::new(0.0, 0.0);
    let mut c = Complex64::new(0.0, 0.0);

    for (i, &z_n) in z_ref.iter().enumerate() {
        // Store current coefficients
        coeffs.push(SeriesCoefficients { a, b, c });

        // Estimate validity radius based on coefficient growth
        // When |A| grows too large, the series becomes unreliable
        let a_norm = a.norm();
        let b_norm = b.norm();
        let validity = if a_norm > 1e-10 {
            // Validity radius is approximately 1/|A| scaled by error tolerance
            // We want |B·δ²| << |A·δ|, so |δ| << |A|/|B|
            if b_norm > 1e-10 {
                (a_norm / b_norm).min(1.0) * 0.1
            } else {
                1.0
            }
        } else {
            1.0
        };
        validity_radii.push(validity.min(1.0).max(1e-20));

        // Early termination if coefficients explode
        if !a.re.is_finite() || !a.im.is_finite() || a_norm > 1e100 {
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

        // A_{n+1} = 2·z_n·A_n
        let a_next = two_z * a;
        // B_{n+1} = 2·z_n·B_n + A_n²
        let b_next = two_z * b + a_sq;
        // C_{n+1} = 2·z_n·C_n + 2·A_n·B_n
        let c_next = two_z * c + two_ab;

        a = a_next;
        b = b_next;
        c = c_next;
    }

    SeriesTable { coeffs, validity_radii }
}

/// Résultat du calcul de saut de série
pub struct SeriesSkipResult {
    /// Itération jusqu'à laquelle on peut sauter
    pub skip_to: usize,
    /// Delta après le saut
    pub delta: ComplexExp,
    /// Erreur estimée de l'approximation
    pub estimated_error: f64,
}

/// Calcule le nombre d'itérations que l'on peut sauter avec l'approximation par série.
///
/// # Arguments
/// * `table` - Table de coefficients de série
/// * `delta_norm` - Norme du delta initial |δ|
/// * `error_tolerance` - Tolérance d'erreur maximale acceptée
/// * `dc` - Offset du pixel par rapport au centre (pour Mandelbrot)
/// * `is_julia` - true si c'est un ensemble de Julia
///
/// # Returns
/// Option contenant le résultat du saut si possible, None sinon
pub fn compute_series_skip(
    table: &SeriesTable,
    delta: ComplexExp,
    dc: ComplexExp,
    error_tolerance: f64,
    is_julia: bool,
) -> Option<SeriesSkipResult> {
    if table.is_empty() || error_tolerance <= 0.0 {
        return None;
    }

    let delta_norm = delta.norm_sqr_approx().sqrt();
    if delta_norm <= 0.0 || !delta_norm.is_finite() {
        return None;
    }

    // Find the maximum iteration we can skip to
    let mut best_skip = 0usize;
    let mut best_delta = delta;
    let mut best_error = f64::MAX;

    for (n, (coeffs, &validity)) in table.coeffs.iter().zip(table.validity_radii.iter()).enumerate() {
        // Check if delta is within validity radius
        if delta_norm > validity {
            break;
        }

        // Compute approximated delta at iteration n:
        // δ_n ≈ A_n·δ + B_n·δ² + C_n·δ³
        let delta_f64 = delta.to_complex64_approx();
        let delta_sq = delta_f64 * delta_f64;
        let delta_cube = delta_sq * delta_f64;

        let mut approx = coeffs.a * delta_f64 + coeffs.b * delta_sq + coeffs.c * delta_cube;

        // For Mandelbrot, add contribution from dc
        // This is a simplified model; full integration would need dc coefficients
        if !is_julia && n > 0 {
            // Accumulate dc contribution: sum of A_k for k=0..n-1
            let dc_f64 = dc.to_complex64_approx();
            // Approximate dc contribution as A_n * dc (simplified)
            approx += coeffs.a * dc_f64 * (n as f64);
        }

        // Estimate error: O(δ^4) for cubic approximation
        let delta_4 = delta_norm * delta_norm * delta_norm * delta_norm;
        let a_norm = coeffs.a.norm();
        let error = delta_4 * (1.0 + a_norm);

        if error < error_tolerance && error < best_error {
            best_skip = n;
            best_delta = ComplexExp::from_complex64(approx);
            best_error = error;
        }

        // Stop if error is growing too fast
        if error > error_tolerance * 100.0 {
            break;
        }
    }

    // Only return if we can skip at least 1 iteration
    if best_skip > 0 {
        Some(SeriesSkipResult {
            skip_to: best_skip,
            delta: best_delta,
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
/// * `delta_norm_sqr` - |δ|² of the current delta
/// * `order` - Series order (2, 3, 4, 5, or 6)
/// * `bla_level` - BLA level (0-16), higher levels skip more iterations
/// * `coeff_a_norm` - |A| coefficient norm from BLA node
///
/// The error estimation accounts for:
/// - Base truncation error O(δ^(order+1))
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
    // L'erreur est O(δ^(order+1))
    let base_error = match order {
        2 => delta_abs * delta_norm_sqr,                        // O(δ³)
        3 => delta_norm_sqr * delta_norm_sqr,                   // O(δ⁴)
        4 => delta_norm_sqr * delta_norm_sqr * delta_abs,       // O(δ⁵)
        5 => delta_norm_sqr * delta_norm_sqr * delta_norm_sqr,  // O(δ⁶)
        6 => delta_norm_sqr * delta_norm_sqr * delta_norm_sqr * delta_abs, // O(δ⁷)
        _ => delta_abs * delta_norm_sqr,  // Fallback to order 2
    };

    // Amplification factor based on BLA level
    // Higher levels skip more iterations, accumulating more error
    let level_factor = 1.0 + (bla_level as f64) * 0.1;

    // Amplification factor from coefficient A
    // Large coefficients amplify errors more
    let coeff_factor = if coeff_a_norm > 1.0 {
        1.0 + coeff_a_norm.ln()
    } else {
        1.0
    };

    base_error * level_factor * coeff_factor
}

#[cfg(test)]
mod tests {
    use super::{estimate_series_error, should_use_series, SeriesConfig};

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
        // With level=0 and coeff_a_norm=1.0, behavior should be similar to before
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

        // Higher BLA levels should produce higher error estimates
        assert!(err_level_5 > err_level_0);
        assert!(err_level_10 > err_level_5);
    }

    #[test]
    fn series_error_increases_with_coeff_norm() {
        let delta_norm_sqr = 1e-8;
        let err_coeff_1 = estimate_series_error(delta_norm_sqr, 2, 0, 1.0);
        let err_coeff_10 = estimate_series_error(delta_norm_sqr, 2, 0, 10.0);
        let err_coeff_100 = estimate_series_error(delta_norm_sqr, 2, 0, 100.0);

        // Larger coefficient norms should produce higher error estimates
        assert!(err_coeff_10 > err_coeff_1);
        assert!(err_coeff_100 > err_coeff_10);
    }

    #[test]
    fn series_error_small_coeff_no_amplification() {
        let delta_norm_sqr = 1e-8;
        // Coefficients <= 1 should not amplify (coeff_factor = 1.0)
        let err_coeff_small = estimate_series_error(delta_norm_sqr, 2, 0, 0.5);
        let err_coeff_one = estimate_series_error(delta_norm_sqr, 2, 0, 1.0);

        assert_eq!(err_coeff_small, err_coeff_one);
    }

    #[test]
    fn series_error_invalid_inputs() {
        // Order < 2 should return 0
        assert_eq!(estimate_series_error(1e-8, 1, 0, 1.0), 0.0);
        assert_eq!(estimate_series_error(1e-8, 0, 0, 1.0), 0.0);

        // Non-finite delta should return 0
        assert_eq!(estimate_series_error(f64::NAN, 2, 0, 1.0), 0.0);
        assert_eq!(estimate_series_error(f64::INFINITY, 2, 0, 1.0), 0.0);
    }
}
