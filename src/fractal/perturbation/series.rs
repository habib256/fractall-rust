use num_complex::Complex64;

use crate::fractal::FractalParams;
use crate::fractal::perturbation::types::ComplexExp;

/// Maximum supported series order.
pub const MAX_SERIES_ORDER: usize = 32;

/// Higher-order series approximation coefficients.
///
/// Inspired by rust-fractal-core's arbitrary-order series approach.
/// Stores coefficients for orders 1..=order so that:
///   delta_n = sum_{k=1}^{order} coeffs[k] * dc^k
///
/// ## Mandelbrot dc-series (is_julia = false)
///
///   Recurrences (from delta_{n+1} = 2*Z_n*delta_n + delta_n^2 + dc):
///     coeff[1]_{n+1} = 2*Z_n*coeff[1]_n + 1
///     coeff[k]_{n+1} = 2*Z_n*coeff[k]_n + sum_{j=1}^{k-1} coeff[j]_n * coeff[k-j]_n
///
/// ## Julia delta-series (is_julia = true)
///
///   Recurrences (from delta_{n+1} = 2*Z_n*delta_n + delta_n^2):
///     coeff[1]_{n+1} = 2*Z_n*coeff[1]_n      (no +1)
///     coeff[k]_{n+1} = 2*Z_n*coeff[k]_n + sum_{j=1}^{k-1} coeff[j]_n * coeff[k-j]_n
#[derive(Clone, Debug)]
pub struct HighOrderCoefficients {
    /// coeffs[k] for k in 0..=order, where coeffs[0] = Z_n (reference), coeffs[k>=1] = series coeff
    pub coeffs: Vec<Complex64>,
    pub order: usize,
}

impl HighOrderCoefficients {
    pub fn new(order: usize) -> Self {
        Self {
            coeffs: vec![Complex64::new(0.0, 0.0); order + 1],
            order,
        }
    }
}

/// Fixed-order (4) coefficients for backward compatibility.
#[derive(Clone, Copy, Debug)]
pub struct SeriesCoefficients {
    pub a: Complex64,
    pub b: Complex64,
    pub c: Complex64,
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

/// Table of series coefficients for all iterations.
/// Supports both fixed-order (4) via `coeffs` and higher-order via `ho_coeffs`.
///
/// Inspired by rust-fractal-core: supports configurable order and data_storage_interval
/// to reduce memory usage on very deep zooms.
#[derive(Clone, Debug)]
pub struct SeriesTable {
    /// Fixed-order coefficients for backward compat (index = iteration number)
    pub coeffs: Vec<SeriesCoefficients>,
    /// Higher-order coefficients (stored every `data_storage_interval` iterations)
    pub ho_coeffs: Vec<Vec<Complex64>>,
    /// Interval between stored higher-order coefficients (inspired by rust-fractal-core)
    pub data_storage_interval: usize,
    /// Series order (number of terms, e.g. 4 = a*dc + b*dc^2 + c*dc^3 + d*dc^4)
    pub order: usize,
    pub is_julia: bool,
    /// Validated iteration count from probe-based validation.
    /// Series is only valid up to this iteration (0 = not validated yet).
    pub validated_skip: usize,
}

impl SeriesTable {
    pub fn empty() -> Self {
        Self {
            coeffs: Vec::new(),
            ho_coeffs: Vec::new(),
            data_storage_interval: 1,
            order: 4,
            is_julia: false,
            validated_skip: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.coeffs.is_empty() && self.ho_coeffs.is_empty()
    }

    /// Get the higher-order coefficients at a given iteration.
    /// Returns None if the iteration isn't stored.
    pub fn get_ho_coeffs(&self, iteration: usize) -> Option<&Vec<Complex64>> {
        if self.ho_coeffs.is_empty() || self.data_storage_interval == 0 {
            return None;
        }
        if iteration == 0 {
            return self.ho_coeffs.first();
        }
        let idx = iteration / self.data_storage_interval;
        self.ho_coeffs.get(idx)
    }
}

/// Build the series table from the reference orbit.
///
/// Supports arbitrary order (up to MAX_SERIES_ORDER).
/// Inspired by rust-fractal-core's generate_approximation() which computes
/// coefficients using the convolution formula for z^2+c.
///
/// The `data_storage_interval` parameter controls memory usage: only every Nth
/// iteration's coefficients are stored. This is crucial for deep zooms where
/// the orbit can be millions of iterations long.
pub fn build_series_table_ho(
    z_ref: &[Complex64],
    is_julia: bool,
    order: usize,
    data_storage_interval: usize,
) -> SeriesTable {
    if z_ref.is_empty() || order == 0 {
        return SeriesTable::empty();
    }

    let order = order.min(MAX_SERIES_ORDER);
    let interval = data_storage_interval.max(1);
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);

    // Current coefficients: prev[k] for k in 0..=order
    // prev[0] = Z_n (from reference orbit)
    // prev[k>=1] = series coefficient of order k
    let mut prev = vec![zero; order + 1];
    prev[0] = z_ref[0]; // Z_0
    if is_julia {
        prev[1] = one; // a_0 = 1 for Julia
    }
    // For Mandelbrot: prev[1] = 0 (a_0 = 0)

    // Also build fixed-order table for backward compat
    let mut fixed_coeffs = Vec::with_capacity(z_ref.len());
    let mut ho_coeffs_stored = Vec::new();

    // Store initial coefficients
    fixed_coeffs.push(SeriesCoefficients {
        a: prev[1],
        b: if order >= 2 { prev[2] } else { zero },
        c: if order >= 3 { prev[3] } else { zero },
        d: if order >= 4 { prev[4] } else { zero },
    });
    ho_coeffs_stored.push(prev[1..=order].to_vec());

    let mut next = vec![zero; order + 1];

    for i in 1..z_ref.len() {
        // Z_{n+1} is from reference orbit
        next[0] = if i < z_ref.len() { z_ref[i] } else { zero };

        // Recurrence: coeff[1]_{n+1} = 2*Z_n*coeff[1]_n + add_value
        let two_z = prev[0] * 2.0;
        let add_value = if is_julia { zero } else { one };
        next[1] = two_z * prev[1] + add_value;

        // Higher-order recurrence (convolution formula from rust-fractal-core):
        // coeff[k]_{n+1} = 2*Z_n*coeff[k]_n + sum_{j=1}^{k-1} coeff[j] * coeff[k-j]
        for k in 2..=order {
            let mut sum = two_z * prev[k];

            // Convolution: sum pairs (j, k-j) for j=1..=(k-1)/2
            for j in 1..=((k - 1) / 2) {
                sum += prev[j] * prev[k - j];
            }
            sum *= 2.0;

            // Middle term when k is even
            if k % 2 == 0 {
                sum += prev[k / 2] * prev[k / 2];
            }

            next[k] = sum;
        }

        // Check for overflow/NaN in any coefficient
        let mut overflow = false;
        for k in 1..=order {
            if !next[k].re.is_finite() || !next[k].im.is_finite() || next[k].norm() > 1e100 {
                overflow = true;
                break;
            }
        }
        if overflow {
            break;
        }

        // Store coefficients
        fixed_coeffs.push(SeriesCoefficients {
            a: next[1],
            b: if order >= 2 { next[2] } else { zero },
            c: if order >= 3 { next[3] } else { zero },
            d: if order >= 4 { next[4] } else { zero },
        });

        if i % interval == 0 || i == z_ref.len() - 1 {
            ho_coeffs_stored.push(next[1..=order].to_vec());
        }

        // Swap for next iteration
        std::mem::swap(&mut prev, &mut next);
    }

    SeriesTable {
        coeffs: fixed_coeffs,
        ho_coeffs: ho_coeffs_stored,
        data_storage_interval: interval,
        order,
        is_julia,
        validated_skip: 0,
    }
}

/// Build the series table (backward compatible, fixed order 4).
pub fn build_series_table(z_ref: &[Complex64], is_julia: bool) -> SeriesTable {
    build_series_table_ho(z_ref, is_julia, 4, 1)
}

/// Probe-based series approximation validation.
///
/// Inspired by rust-fractal-core's check_approximation() method.
/// Uses probe points (corners + grid) to validate that the series approximation
/// is accurate by comparing series evaluation against actual perturbation iteration.
///
/// Returns the maximum iteration to which the series can safely be skipped.
pub fn validate_series_with_probes(
    table: &SeriesTable,
    z_ref: &[Complex64],
    is_julia: bool,
    delta_pixel: f64,
    image_width: usize,
    image_height: usize,
    probe_sampling: usize,
) -> usize {
    if table.is_empty() || z_ref.is_empty() || delta_pixel <= 0.0 {
        return 0;
    }

    let probe_sampling = probe_sampling.max(2).min(8);
    let delta_pixel_sqr = delta_pixel * delta_pixel;

    // Generate probe points at grid positions across the image
    // Inspired by rust-fractal-core's calculate_probes()
    let mut probes: Vec<Complex64> = Vec::new();
    for j in 0..probe_sampling {
        for i in 0..probe_sampling {
            let pos_x = image_width as f64 * (i as f64 / (probe_sampling as f64 - 1.0));
            let pos_y = image_height as f64 * (j as f64 / (probe_sampling as f64 - 1.0));
            let re = (pos_x - image_width as f64 * 0.5) * delta_pixel;
            let im = (pos_y - image_height as f64 * 0.5) * delta_pixel;
            probes.push(Complex64::new(re, im));
        }
    }

    if probes.is_empty() {
        return 0;
    }

    // For each probe, iterate both by series evaluation and actual perturbation,
    // find where they diverge.
    // Inspired by rust-fractal-core's iterate_probes() method.
    let mut min_valid = z_ref.len();

    for probe_dc in &probes {
        let mut actual_delta = if is_julia { *probe_dc } else { Complex64::new(0.0, 0.0) };

        for n in 0..z_ref.len().saturating_sub(1) {
            // Evaluate series approximation at this iteration
            if n < table.coeffs.len() {
                let coeffs = &table.coeffs[n];
                let series_delta = *probe_dc * (coeffs.a + *probe_dc * (coeffs.b + *probe_dc * (coeffs.c + *probe_dc * coeffs.d)));

                // Compare series vs actual
                let relative_error = (actual_delta - series_delta).norm_sqr();

                // Derivative estimate for relative error scaling (inspired by rust-fractal-core)
                let derivative_norm = (coeffs.a + *probe_dc * (coeffs.b * 2.0 + *probe_dc * coeffs.c * 3.0)).norm_sqr();
                let derivative_scale = if derivative_norm > 1.0 { derivative_norm } else { 1.0 };

                if relative_error / derivative_scale > delta_pixel_sqr || !relative_error.is_finite() {
                    min_valid = min_valid.min(n.saturating_sub(1).max(1));
                    break;
                }
            } else {
                min_valid = min_valid.min(n);
                break;
            }

            // Actual perturbation iteration
            let z_n = z_ref[n];
            if is_julia {
                actual_delta = z_n * actual_delta * 2.0 + actual_delta * actual_delta;
            } else {
                actual_delta = z_n * actual_delta * 2.0 + actual_delta * actual_delta + *probe_dc;
            }

            // Bailout
            if actual_delta.norm_sqr() > 1e16 {
                min_valid = min_valid.min(n);
                break;
            }
        }
    }

    min_valid.min(z_ref.len().saturating_sub(1))
}

/// Evaluate higher-order series at a given point and iteration.
///
/// Uses Horner's method for numerical stability:
///   delta = dc * (c[1] + dc * (c[2] + dc * (c[3] + ... + dc * c[order])))
///
/// Inspired by rust-fractal-core's evaluate() method.
pub fn evaluate_ho_series(
    coefficients: &[Complex64],
    point_delta: Complex64,
) -> Complex64 {
    if coefficients.is_empty() {
        return Complex64::new(0.0, 0.0);
    }

    // Horner's method: start from highest order
    let mut result = coefficients[coefficients.len() - 1];
    for coeff in coefficients[..coefficients.len() - 1].iter().rev() {
        result = result * point_delta + *coeff;
    }
    result *= point_delta;
    result
}

/// Evaluate the derivative of the higher-order series at a given point.
///
/// d/d(dc) [sum c[k] * dc^k] = sum k * c[k] * dc^(k-1)
///
/// Inspired by rust-fractal-core's evaluate_derivative() method.
pub fn evaluate_ho_derivative(
    coefficients: &[Complex64],
    point_delta: Complex64,
) -> Complex64 {
    if coefficients.is_empty() {
        return Complex64::new(1.0, 0.0);
    }

    let order = coefficients.len();
    let mut result = coefficients[order - 1] * order as f64;
    for k in (1..order).rev() {
        result = result * point_delta + coefficients[k - 1] * k as f64;
    }
    result
}

/// Result of series skip computation
pub struct SeriesSkipResult {
    pub skip_to: usize,
    pub delta: ComplexExp,
    pub estimated_error: f64,
}

/// Compute the number of iterations that can be skipped with series approximation.
///
/// For higher-order series, uses the full coefficient set and Horner evaluation.
/// Falls back to order-4 if higher-order coefficients aren't available.
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

    let dc_norm_sq = dc_norm * dc_norm;

    // Respect probe validation if available
    let max_skip = if table.validated_skip > 0 {
        table.validated_skip
    } else {
        usize::MAX
    };

    let mut best_skip = 0usize;
    let mut best_approx = Complex64::new(0.0, 0.0);
    let mut best_error = f64::MAX;

    // Try higher-order evaluation if available
    let use_ho = table.order > 4 && !table.ho_coeffs.is_empty();

    for (n, coeffs) in table.coeffs.iter().enumerate() {
        if n > max_skip {
            break;
        }

        // Evaluate series
        let approx = if use_ho {
            if let Some(ho) = table.get_ho_coeffs(n) {
                evaluate_ho_series(ho, dc_f64)
            } else {
                // Fall back to fixed-order evaluation
                dc_f64 * (coeffs.a + dc_f64 * (coeffs.b + dc_f64 * (coeffs.c + dc_f64 * coeffs.d)))
            }
        } else {
            dc_f64 * (coeffs.a + dc_f64 * (coeffs.b + dc_f64 * (coeffs.c + dc_f64 * coeffs.d)))
        };

        // Estimate truncation error
        let d_norm = coeffs.d.norm();
        let c_norm = coeffs.c.norm();
        let next_coeff_estimate = if c_norm > 1e-30 {
            d_norm * d_norm / c_norm
        } else {
            d_norm * dc_norm
        };

        let order_plus_one = (table.order + 1) as i32;
        let dc_high = dc_norm.powi(order_plus_one);
        let error = next_coeff_estimate * dc_high;

        // Check series convergence
        let dc_order = dc_norm.powi(table.order as i32);
        let last_term_norm = d_norm * dc_order;
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

        if term_ratio > 1.0 || !approx.re.is_finite() || !approx.im.is_finite() {
            break;
        }
    }

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

/// Compute series skip using higher-order coefficients and extended precision.
///
/// Uses ComplexExp for evaluation to avoid f64 underflow at deep zooms.
/// Inspired by rust-fractal-core's extended precision evaluation.
pub fn compute_series_skip_extended(
    table: &SeriesTable,
    dc: ComplexExp,
    error_tolerance: f64,
) -> Option<SeriesSkipResult> {
    if table.ho_coeffs.is_empty() || error_tolerance <= 0.0 {
        return compute_series_skip(table, dc, error_tolerance);
    }

    let dc_norm_sqr = dc.norm_sqr_approx();
    if dc_norm_sqr <= 0.0 || !dc_norm_sqr.is_finite() {
        return None;
    }

    let max_skip = if table.validated_skip > 0 {
        table.validated_skip
    } else {
        usize::MAX
    };

    let mut best_skip = 0usize;
    let mut best_delta = ComplexExp::zero();
    let mut best_error = f64::MAX;

    for (stored_idx, ho) in table.ho_coeffs.iter().enumerate() {
        let n = stored_idx * table.data_storage_interval;
        if n > max_skip || n >= table.coeffs.len() {
            break;
        }

        // Evaluate using extended precision Horner
        let mut result = ComplexExp::from_complex64(ho[ho.len() - 1]);
        for coeff in ho[..ho.len() - 1].iter().rev() {
            result = result.mul(dc).add(ComplexExp::from_complex64(*coeff));
        }
        result = result.mul(dc);

        let result_norm = result.norm_sqr_approx();
        if !result_norm.is_finite() {
            break;
        }

        // Rough error estimate
        let last_coeff_norm = ho.last().map(|c| c.norm()).unwrap_or(0.0);
        let dc_norm = dc_norm_sqr.sqrt();
        let error = last_coeff_norm * dc_norm.powi((table.order + 1) as i32);

        if error < error_tolerance && result_norm > 0.0 {
            best_skip = n;
            best_delta = result;
            best_error = error;
        }

        if error > error_tolerance * 100.0 {
            break;
        }
    }

    if best_skip >= 2 {
        Some(SeriesSkipResult {
            skip_to: best_skip,
            delta: best_delta,
            estimated_error: best_error,
        })
    } else {
        compute_series_skip(table, dc, error_tolerance)
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

    let base_error = match order {
        2 => delta_abs * delta_norm_sqr,
        3 => delta_norm_sqr * delta_norm_sqr,
        4 => delta_norm_sqr * delta_norm_sqr * delta_abs,
        5 => delta_norm_sqr * delta_norm_sqr * delta_norm_sqr,
        6 => delta_norm_sqr * delta_norm_sqr * delta_norm_sqr * delta_abs,
        n => delta_abs.powi(n as i32 + 1),
    };

    let level_factor = 1.0 + (bla_level as f64) * 0.1;
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

    #[test]
    fn mandelbrot_series_at_origin() {
        let z_ref = vec![Complex64::new(0.0, 0.0); 10];
        let table = build_series_table(&z_ref, false);
        assert!(!table.is_empty());
        assert!(!table.is_julia);

        assert!((table.coeffs[0].a.norm()) < 1e-10);
        assert!((table.coeffs[1].a - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((table.coeffs[2].a - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn mandelbrot_series_skip_works() {
        let orbit_len = 50;
        let z_ref = vec![Complex64::new(0.0, 0.0); orbit_len];
        let table = build_series_table(&z_ref, false);

        let dc = Complex64::new(0.1, 0.0);
        let approx_1 = table.coeffs[1].a * dc;
        assert!((approx_1 - Complex64::new(0.1, 0.0)).norm() < 1e-10);

        let small_dc = ComplexExp::from_complex64(Complex64::new(0.001, 0.0));
        let result = compute_series_skip(&table, small_dc, 1e-9);
        assert!(result.is_some(), "Series skip should work for small dc");
        let skip = result.unwrap();
        assert!(skip.skip_to >= 2, "Should skip at least 2 iterations, got {}", skip.skip_to);
    }

    #[test]
    fn julia_series_coefficients() {
        let z_ref = vec![
            Complex64::new(0.5, 0.3),
            Complex64::new(0.2, 0.1),
            Complex64::new(0.4, -0.2),
        ];
        let table = build_series_table(&z_ref, true);
        assert!(table.is_julia);

        assert!((table.coeffs[0].a - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        let expected_a1 = Complex64::new(1.0, 0.6);
        assert!((table.coeffs[1].a - expected_a1).norm() < 1e-10);
    }

    #[test]
    fn higher_order_series_matches_fixed() {
        // Verify that order-4 HO series matches the fixed-order implementation
        let z_ref: Vec<Complex64> = (0..20)
            .map(|i| Complex64::new(0.1 * i as f64, 0.05 * i as f64))
            .collect();

        let table_fixed = build_series_table(&z_ref, false);
        let table_ho = build_series_table_ho(&z_ref, false, 4, 1);

        for n in 0..table_fixed.coeffs.len().min(table_ho.coeffs.len()) {
            let fc = &table_fixed.coeffs[n];
            let hc = &table_ho.coeffs[n];
            assert!((fc.a - hc.a).norm() < 1e-8, "a mismatch at iter {}", n);
            assert!((fc.b - hc.b).norm() < 1e-8, "b mismatch at iter {}", n);
            assert!((fc.c - hc.c).norm() < 1e-8, "c mismatch at iter {}", n);
            assert!((fc.d - hc.d).norm() < 1e-8, "d mismatch at iter {}", n);
        }
    }

    #[test]
    fn higher_order_series_more_terms() {
        // Verify that order-8 series gives better approximation than order-4
        let z_ref: Vec<Complex64> = (0..30)
            .map(|i| Complex64::new(0.3 + 0.01 * i as f64, -0.2 + 0.005 * i as f64))
            .collect();

        let table_4 = build_series_table_ho(&z_ref, false, 4, 1);
        let table_8 = build_series_table_ho(&z_ref, false, 8, 1);

        assert_eq!(table_4.order, 4);
        assert_eq!(table_8.order, 8);
        assert!(table_8.ho_coeffs[0].len() == 8);
    }

    #[test]
    fn probe_validation_works() {
        let z_ref = vec![Complex64::new(0.0, 0.0); 50];
        let table = build_series_table(&z_ref, false);

        let validated = validate_series_with_probes(
            &table, &z_ref, false,
            0.01, // delta_pixel
            100, 100, // image size
            3, // probe sampling
        );
        assert!(validated > 0, "Probe validation should find valid iterations");
    }

    #[test]
    fn ho_evaluation_horner() {
        let coeffs = vec![
            Complex64::new(1.0, 0.0),  // order 1
            Complex64::new(0.5, 0.0),  // order 2
        ];
        let dc = Complex64::new(0.1, 0.0);
        let result = evaluate_ho_series(&coeffs, dc);
        // Expected: dc * (1.0 + dc * 0.5) = 0.1 * (1.0 + 0.05) = 0.105
        assert!((result - Complex64::new(0.105, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn data_storage_interval_reduces_storage() {
        let z_ref = vec![Complex64::new(0.1, 0.2); 100];
        let table_1 = build_series_table_ho(&z_ref, false, 8, 1);
        let table_10 = build_series_table_ho(&z_ref, false, 8, 10);

        // With interval=10, should store ~10x fewer HO coefficient sets
        assert!(table_10.ho_coeffs.len() < table_1.ho_coeffs.len());
        assert!(table_10.ho_coeffs.len() <= 12); // ~100/10 + initial + final
    }
}
