//! Résolution de l'échantillonnage et calcul des offsets pixel GMP.

use rug::{Complex, Float};

use crate::fractal::perturbation::orbit::ReferenceOrbitCache;
use crate::fractal::FractalParams;

/// Constantes pré-calculées pour le calcul GMP de l'offset pixel.
#[derive(Clone)]
pub struct DcGmpContext {
    pub inv_width: Float,
    pub inv_height: Float,
    pub half: Float,
    pub x_range: Float,
    pub y_range: Float,
    pub prec: u32,
    pub rot: Option<(f64, f64, f64, f64)>,
    pub aa_uniform: [f64; 2],
    pub aa_jitter: Option<(u64, f64)>,
    pub width: usize,
    /// `centre_vue - centre_reference`, pour les consommateurs relatifs.
    pub ref_offset: Option<(Float, Float)>,
}

/// Plan d'échantillonnage complété après sélection de la référence.
#[derive(Clone)]
pub struct ResolvedSamplingPlan {
    pub sampling: crate::fractal::wisdom::SamplingPlan,
    pub ref_offset: Option<(Float, Float)>,
}

impl ResolvedSamplingPlan {
    pub fn with_reference(params: &FractalParams, prec: u32, cache: &ReferenceOrbitCache) -> Self {
        let sampling = crate::fractal::wisdom::sampling_plan(params);
        let parse = |s: &str| Float::parse(s).ok().map(|p| Float::with_val(prec, p));
        let cx = match params.center_x_hp.as_deref() {
            Some(s) => parse(s),
            None => Some(Float::with_val(prec, params.center_x)),
        };
        let cy = match params.center_y_hp.as_deref() {
            Some(s) => parse(s),
            None => Some(Float::with_val(prec, params.center_y)),
        };
        let ref_offset = match (
            cx,
            cy,
            parse(&cache.center_x_gmp),
            parse(&cache.center_y_gmp),
        ) {
            (Some(cx), Some(cy), Some(rx), Some(ry)) => {
                let ox = Float::with_val(prec, &cx - &rx);
                let oy = Float::with_val(prec, &cy - &ry);
                (!ox.is_zero() || !oy.is_zero()).then_some((ox, oy))
            }
            _ => None,
        };
        Self {
            sampling,
            ref_offset,
        }
    }
}

impl DcGmpContext {
    /// Contexte relatif à la référence : `compute_dc_ref` ajoute l'écart entre
    /// le centre de vue et le centre de la référence réutilisée.
    pub fn with_reference(params: &FractalParams, prec: u32, cache: &ReferenceOrbitCache) -> Self {
        let mut ctx = Self::new(params, prec);
        ctx.ref_offset = ResolvedSamplingPlan::with_reference(params, prec, cache).ref_offset;
        ctx
    }

    pub fn new(params: &FractalParams, prec: u32) -> Self {
        let sampling = crate::fractal::wisdom::sampling_plan(params);
        let inv_width = Float::with_val(prec, 1.0) / Float::with_val(prec, params.width as f64);
        let inv_height = Float::with_val(prec, 1.0) / Float::with_val(prec, params.height as f64);
        let parse_span = |hp: Option<&String>, fallback| match hp {
            Some(value) => Float::parse(value)
                .map(|parsed| Float::with_val(prec, parsed))
                .unwrap_or_else(|_| Float::with_val(prec, fallback)),
            None => Float::with_val(prec, fallback),
        };

        Self {
            inv_width,
            inv_height,
            half: Float::with_val(prec, 0.5),
            x_range: parse_span(params.span_x_hp.as_ref(), params.span_x),
            y_range: parse_span(params.span_y_hp.as_ref(), params.span_y),
            prec,
            rot: sampling.transform,
            aa_uniform: sampling.aa_uniform,
            aa_jitter: sampling.aa_jitter,
            width: params.width as usize,
            ref_offset: None,
        }
    }

    pub fn compute_dc_ref(&self, i: usize, j: usize) -> Complex {
        let mut dc = self.compute_dc(i, j);
        if let Some((ox, oy)) = &self.ref_offset {
            let (mut re, mut im) = dc.into_real_imag();
            re += ox;
            im += oy;
            dc = Complex::with_val(self.prec, (re, im));
        }
        dc
    }

    pub fn compute_dc(&self, i: usize, j: usize) -> Complex {
        let (jx, jy) = match self.aa_jitter {
            Some((k, scale)) => crate::fractal::jitter::pixel_offset(self.width, i, j, k, scale),
            None => (self.aa_uniform[0], self.aa_uniform[1]),
        };
        let mut i_float = Float::with_val(self.prec, i as f64);
        i_float += &self.half;
        if jx != 0.0 {
            i_float += jx;
        }
        let mut j_float = Float::with_val(self.prec, j as f64);
        j_float += &self.half;
        if jy != 0.0 {
            j_float += jy;
        }
        let mut x_ratio = Float::with_val(self.prec, &i_float * &self.inv_width);
        let mut y_ratio = Float::with_val(self.prec, &j_float * &self.inv_height);
        x_ratio -= &self.half;
        y_ratio -= &self.half;
        let dx = Float::with_val(self.prec, &x_ratio * &self.x_range);
        let dy = Float::with_val(self.prec, &y_ratio * &self.y_range);
        match self.rot {
            Some((a, b, c, d)) => {
                let dx_r =
                    Float::with_val(self.prec, &dx * a) + Float::with_val(self.prec, &dy * b);
                let dy_r =
                    Float::with_val(self.prec, &dx * c) + Float::with_val(self.prec, &dy * d);
                Complex::with_val(self.prec, (dx_r, dy_r))
            }
            None => Complex::with_val(self.prec, (dx, dy)),
        }
    }
}
