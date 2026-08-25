//! Validation des buffers de réutilisation entre passes progressives.

use num_complex::Complex64;

use crate::fractal::{FractalParams, OutColoringMode};

pub(super) struct ReuseData<'a> {
    pub(super) iterations: &'a [u32],
    pub(super) zs: &'a [Complex64],
    pub(super) width: u32,
    pub(super) ratio: u32,
}

pub(super) fn build_reuse<'a>(
    params: &FractalParams,
    reuse: Option<(&'a [u32], &'a [Complex64], u32, u32)>,
) -> Option<ReuseData<'a>> {
    let needs_extra_data = matches!(
        params.color.out_coloring_mode,
        OutColoringMode::Distance
            | OutColoringMode::DistanceAO
            | OutColoringMode::Distance3D
            | OutColoringMode::OrbitTraps
            | OutColoringMode::Wings
    );
    if needs_extra_data {
        return None;
    }

    let (iterations, zs, width, height) = reuse?;
    if width == 0 || height == 0 {
        return None;
    }
    let expected_len = (width * height) as usize;
    if iterations.len() != expected_len || zs.len() != expected_len {
        return None;
    }
    if params.width % width != 0 || params.height % height != 0 {
        return None;
    }
    let ratio_x = params.width / width;
    let ratio_y = params.height / height;
    if ratio_x < 2 || ratio_y < 2 || ratio_x != ratio_y {
        return None;
    }

    Some(ReuseData {
        iterations,
        zs,
        width,
        ratio: ratio_x,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::definitions::default_params_for_type;
    use crate::fractal::FractalType;

    #[test]
    fn accepts_aligned_integer_scale_reuse() {
        let params = default_params_for_type(FractalType::Mandelbrot, 8, 4);
        let iterations = vec![0; 8];
        let zs = vec![Complex64::new(0.0, 0.0); 8];

        let reuse = build_reuse(&params, Some((&iterations, &zs, 4, 2))).unwrap();

        assert_eq!(reuse.width, 4);
        assert_eq!(reuse.ratio, 2);
    }

    #[test]
    fn rejects_reuse_when_coloring_needs_missing_channels() {
        let mut params = default_params_for_type(FractalType::Mandelbrot, 8, 4);
        params.color.out_coloring_mode = OutColoringMode::Distance;
        let iterations = vec![0; 8];
        let zs = vec![Complex64::new(0.0, 0.0); 8];

        assert!(build_reuse(&params, Some((&iterations, &zs, 4, 2))).is_none());
    }
}
