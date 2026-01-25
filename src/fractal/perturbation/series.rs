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

pub fn estimate_series_error(delta_norm_sqr: f64, order: u8) -> f64 {
    if order < 2 || !delta_norm_sqr.is_finite() {
        return 0.0;
    }
    let delta_abs = delta_norm_sqr.sqrt();
    match order {
        2 => delta_abs * delta_norm_sqr,
        3 => delta_norm_sqr * delta_norm_sqr,
        _ => delta_abs * delta_norm_sqr,
    }
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
        let err2 = estimate_series_error(1e-8, 2);
        let err3 = estimate_series_error(1e-8, 3);
        assert!(err3 <= err2);
    }
}
