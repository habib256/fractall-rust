use crate::fractal::FractalParams;

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
/// * `order` - Series order (2 or 3)
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
    let base_error = match order {
        2 => delta_abs * delta_norm_sqr,      // O(δ³)
        3 => delta_norm_sqr * delta_norm_sqr, // O(δ⁴)
        _ => delta_abs * delta_norm_sqr,
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
