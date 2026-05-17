use num_complex::Complex64;
use rug::{Complex, Float};
use std::cell::RefCell;
use std::sync::OnceLock;

use crate::fractal::bytecode::bla_dual::BlaTableUnified;
use crate::fractal::bytecode::pixel_loop_exp::iterate_pixel_unified_exp;
use crate::fractal::bytecode::{build_bla_table_for_formula, compile_formula, Formula};
use crate::fractal::{FractalParams, FractalType};
use crate::fractal::perturbation::bla::BlaTable;
use crate::fractal::perturbation::orbit::{ReferenceOrbit, HybridBlaReferences};
use crate::fractal::perturbation::types::{ComplexExp, FloatExp};
use crate::fractal::gmp::complex_norm_sqr;
use crate::fractal::perturbation::series::{
    SeriesConfig, SeriesTable, should_use_series, estimate_series_error,
    compute_series_skip,
};

/// Compute smooth (fractional) iteration count for continuous coloring.
///
/// Inspired by rust-fractal-core's smooth iteration formula:
///   smooth = n + 1 - log2(log2(|z|))
///
/// For power-d fractals (Multibrot), the formula generalizes to:
///   smooth = n + 1 - log_d(log2(|z|))
///
/// Returns `iteration as f64` if the point didn't escape or if the formula fails.
#[inline]
pub fn compute_smooth_iteration(iteration: u32, z_final: Complex64, bailout: f64, power: f64) -> f64 {
    let norm_sqr = z_final.norm_sqr();
    if !norm_sqr.is_finite() || norm_sqr <= 0.0 || norm_sqr <= bailout * bailout {
        return iteration as f64;
    }
    let log_zn = norm_sqr.ln() * 0.5; // ln(|z|) = 0.5 * ln(|z|²)
    if log_zn <= 0.0 || !log_zn.is_finite() {
        return iteration as f64;
    }
    let log_log_zn = log_zn.ln(); // ln(ln(|z|))
    if !log_log_zn.is_finite() {
        return iteration as f64;
    }
    let log_power = power.ln(); // ln(d) for Multibrot
    if log_power <= 0.0 || !log_power.is_finite() {
        return iteration as f64;
    }
    let smooth = iteration as f64 + 1.0 - log_log_zn / log_power;
    if smooth.is_finite() && smooth >= 0.0 {
        smooth
    } else {
        iteration as f64
    }
}

/// Compute adaptive batch size based on fractal power.
/// Inspired by rust-fractal-core's `iterations_before_check = 400 / power`.
/// For standard Mandelbrot (power=2): 200 iterations per batch.
/// For higher powers: smaller batches to check escape more frequently.
#[inline]
fn adaptive_batch_size(power: f64) -> u32 {
    if power <= 0.0 || !power.is_finite() {
        return 256;
    }
    (400.0 / power).round().clamp(64.0, 512.0) as u32
}

// Cache thread-local de la BlaTableUnified par worker rayon.
// Évite la reconstruction par pixel (O(M log M) en taille d'orbite).
// Stocke l'identité de l'orbite (ptr de `z_ref_f64` + len) + le type/power
// pour invalider quand on change de render.
thread_local! {
    static BLA_UNIFIED_CACHE: RefCell<Option<BlaUnifiedCacheEntry>> = const { RefCell::new(None) };
}

struct BlaUnifiedCacheEntry {
    orbit_ptr: usize,
    orbit_len: usize,
    fractal_type: FractalType,
    multibrot_power: f64,
    formula: Formula,
    tables: Vec<BlaTableUnified>,
}

/// Tente le path bytecode unifié si toutes les conditions sont remplies.
/// Renvoie `Some(DeltaResult)` si le path a été appliqué, `None` sinon
/// (le caller fallback sur le path historique).
fn try_bytecode_unified_path(
    params: &FractalParams,
    ref_orbit: &ReferenceOrbit,
    delta0: &ComplexExp,
    dc: &ComplexExp,
) -> Option<DeltaResult> {
    // Le path bytecode supporte désormais distance/interior/orbit_traps
    // en perturbation via ddelta tracking (cf. UnifiedOptions).
    let formula = compile_formula(params.fractal_type, params.multibrot_power)?;
    if formula.phases.len() != 1 {
        // Multi-phase pas encore supporté.
        return None;
    }
    // Dispatch selon pixel_size :
    // - pixel_size > 1e-13 : path f64 (rapide, mantissa f64 suffit)
    // - 1e-150 < pixel_size < 1e-13 : path ComplexExp (mantissa+exp pour delta)
    // - pixel_size < 1e-150 : fallback GMP legacy (z_ref_f64 underflow)
    let pixel_size = (params.span_x.abs() / params.width.max(1) as f64)
        .max(params.span_y.abs() / params.height.max(1) as f64);
    let use_exp_path = pixel_size < 1e-13;
    if pixel_size < 1e-150 {
        return None;
    }

    // Préparer / recycler la table BLA depuis le cache thread-local.
    let orbit_ptr = ref_orbit.z_ref_f64.as_ptr() as usize;
    let orbit_len = ref_orbit.z_ref_f64.len();
    let c_ref = ref_orbit.cref;
    let c_norm = (c_ref.re * c_ref.re + c_ref.im * c_ref.im).sqrt();

    let result = BLA_UNIFIED_CACHE.with(|cache_cell| {
        let mut cache = cache_cell.borrow_mut();
        // Vérifier si l'entrée actuelle correspond à ce render.
        let needs_rebuild = match cache.as_ref() {
            None => true,
            Some(entry) => {
                entry.orbit_ptr != orbit_ptr
                    || entry.orbit_len != orbit_len
                    || entry.fractal_type != params.fractal_type
                    || (entry.multibrot_power - params.multibrot_power).abs() > 1e-12
            }
        };
        if needs_rebuild {
            let tables =
                build_bla_table_for_formula(&formula, &ref_orbit.z_ref_f64, c_norm, 6e-8)?;
            *cache = Some(BlaUnifiedCacheEntry {
                orbit_ptr,
                orbit_len,
                fractal_type: params.fractal_type,
                multibrot_power: params.multibrot_power,
                formula: formula.clone(),
                tables,
            });
        }
        let entry = cache.as_ref()?;
        let bla = &entry.tables[0];

        // Pour Julia : delta_init = pixel - cref (caller fournit delta0
        // non-zéro), c_for_add = seed (constant), dc_for_add = 0.
        // Pour Mandelbrot : delta_init = 0, c_for_add = cref, dc_for_add = dc.
        let is_julia = Formula::is_julia_for(params.fractal_type);

        if use_exp_path {
            // Path ComplexExp pour deep zoom > 1e13.
            let (c_for_add, dc_for_add_exp) = if is_julia {
                (
                    Complex64::new(params.seed.re, params.seed.im),
                    ComplexExp::zero(),
                )
            } else {
                (c_ref, *dc)
            };
            let res_exp = iterate_pixel_unified_exp(
                ref_orbit,
                bla,
                &entry.formula,
                c_for_add,
                dc_for_add_exp,
                *delta0,
                params.iteration_max,
                params.bailout,
            );
            // Conversion vers UnifiedPixelResult (même shape, juste typage).
            return Some(crate::fractal::bytecode::pixel_loop::UnifiedPixelResult {
                iteration: res_exp.iteration,
                z_final: res_exp.z_final,
                rebase_count: res_exp.rebase_count,
                bla_steps: res_exp.bla_steps,
                orbit: None,
                distance: None,
                is_interior: false,
            });
        }

        // Path f64 standard (rapide).
        let delta_init = delta0.to_complex64_approx();
        let dc_approx = dc.to_complex64_approx();
        let (c_for_add, dc_for_add) = if is_julia {
            (
                Complex64::new(params.seed.re, params.seed.im),
                Complex64::new(0.0, 0.0),
            )
        } else {
            (c_ref, dc_approx)
        };

        let options = crate::fractal::bytecode::pixel_loop::UnifiedOptions {
            orbit_trap: if params.enable_orbit_traps {
                Some(params.orbit_trap_type)
            } else {
                None
            },
            enable_distance: params.enable_distance_estimation,
            enable_interior: params.enable_interior_detection,
            interior_threshold: params.interior_threshold,
            is_julia,
        };
        let pixel_result = crate::fractal::bytecode::pixel_loop::iterate_pixel_unified_full(
            ref_orbit,
            bla,
            &entry.formula,
            c_for_add,
            dc_for_add,
            delta_init,
            params.iteration_max,
            params.bailout,
            options,
        );

        Some(pixel_result)
    })?;

    // Convertir UnifiedPixelResult → DeltaResult attendu par le pipeline.
    let smooth_iteration = if result.iteration < params.iteration_max {
        // Smooth coloring standard : n + 1 - log2(log|z|)
        let z_norm_sqr = result.z_final.norm_sqr().max(1.0);
        let log_z = z_norm_sqr.ln() * 0.5;
        let nu = (log_z.ln() / std::f64::consts::LN_2).max(0.0);
        (result.iteration as f64 + 1.0 - nu).max(0.0)
    } else {
        result.iteration as f64
    };

    Some(DeltaResult {
        iteration: result.iteration,
        z_final: result.z_final,
        glitched: false,
        suspect: false,
        distance: result.distance.unwrap_or(f64::INFINITY),
        is_interior: result.is_interior,
        phase_changed: false,
        smooth_iteration,
    })
}

fn rebase_stride() -> u32 {
    static STRIDE: OnceLock<u32> = OnceLock::new();
    *STRIDE.get_or_init(|| {
        let raw = std::env::var("FRACTALL_PERTURB_REBASE_STRIDE")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(1);
        raw.clamp(1, 64)
    })
}

pub struct DeltaResult {
    pub iteration: u32,
    pub z_final: Complex64,
    pub glitched: bool,
    pub suspect: bool,
    /// Distance estimation (if computed). f64::INFINITY if not computed or invalid.
    /// Can be used for distance-based coloring or 3D rendering.
    pub distance: f64,
    /// Whether the point is in the interior of the set. Only valid if interior detection was enabled.
    /// Can be used for interior coloring (e.g., black for interior points).
    pub is_interior: bool,
    /// Whether the phase changed during rebasing (for Hybrid BLA).
    /// If true, the caller should switch to the new phase reference.
    pub phase_changed: bool,
    /// Smooth (fractional) iteration count for continuous coloring.
    ///
    /// Inspired by rust-fractal-core's smooth iteration output:
    /// For escaped points: `n + 1 - log2(log2(|z_final|))`
    /// For non-escaped points: same as `iteration` (integer).
    ///
    /// This avoids banding artifacts in coloring by providing a continuous
    /// iteration count. The caller can use this directly for palette mapping
    /// instead of computing smooth iteration separately.
    pub smooth_iteration: f64,
}

/// Result of a fast f64 batch perturbation loop.
enum BatchResult {
    /// Batch completed, continue with BLA/next batch
    Continue,
    /// Pixel escaped
    Escaped { iteration: u32, z_final: Complex64 },
    /// Pixel glitched or NaN
    Glitched { iteration: u32, z_final: Complex64 },
    /// Rebasing needed: delta has grown larger than z_curr.
    /// The outer loop should set delta = z_curr and n = 0.
    NeedRebase { z_curr: Complex64 },
}

/// Fast scaled perturbation for Mandelbrot z²+c (and Julia).
///
/// Inspired by rust-fractal-core's scaled delta approach: uses ldexp/frexp scaling
/// to keep the delta mantissa in f64 range while tracking the exponent separately.
/// This avoids the overhead of full ComplexExp operations (mantissa+exponent) in the
/// inner loop while maintaining precision across a much wider range than plain f64.
///
/// The key insight from rust-fractal-core is:
///   scale_factor = 2^(delta.exponent)
///   scaled_delta = delta.mantissa (in f64)
///   z_curr = z_ref + scale_factor * scaled_delta
///
/// After each iteration, the scale_factor is updated via reduce() (re-normalize).
/// At extended iterations (where z_ref underflows f64), we fall back to ComplexExtended.
///
/// # When to use
/// - Mandelbrot or Julia (standard z²+c perturbation)
/// - NOT Burning Ship, Tricorn, or Multibrot (different formulas)
/// - NOT when distance estimation or interior detection is enabled
#[inline(never)]
fn fast_mandelbrot_batch_f64<const IS_JULIA: bool>(
    ref_orbit: &ReferenceOrbit,
    n: &mut u32,
    delta: &mut ComplexExp,
    dc_f64: Complex64,
    bailout_sqr: f64,
    glitch_tolerance_sqr: f64,
    max_iter: u32,
    _effective_len: u32,
    suspect: &mut bool,
    iters_ptb: &mut u32,
) -> BatchResult {
    // Adaptive batch size: 400/power for power=2 → 200
    const BATCH_SIZE: u32 = 200;

    // Scaled delta approach (inspired by rust-fractal-core):
    // Keep delta as mantissa * 2^exponent, but do arithmetic on mantissa only,
    // using scale_factor = 2^exponent to convert between delta and z_ref space.
    let mut d_re = delta.re.mantissa;
    let mut d_im = delta.im.mantissa;
    let delta_exp = delta.re.exponent.max(delta.im.exponent);
    let scale_factor = crate::fractal::perturbation::types::pow2i(delta_exp);

    // Scale dc into delta space: dc_scaled = dc * 2^(-delta_exp)
    let inv_scale = if delta_exp > -500 && delta_exp < 500 {
        crate::fractal::perturbation::types::pow2i(-delta_exp)
    } else {
        1.0
    };
    let dc_re_scaled = dc_f64.re * inv_scale;
    let dc_im_scaled = dc_f64.im * inv_scale;

    // If scale factor is degenerate, fall back to plain f64
    let use_scaling = scale_factor.is_finite() && scale_factor > 0.0
        && inv_scale.is_finite() && delta_exp.abs() < 500;

    if !use_scaling {
        // Fall back to plain f64 path (no scaling)
        d_re = delta.re.to_f64();
        d_im = delta.im.to_f64();
    }

    let dc_re = if use_scaling { dc_re_scaled } else { dc_f64.re };
    let dc_im = if use_scaling { dc_im_scaled } else { dc_f64.im };
    let sf = if use_scaling { scale_factor } else { 1.0 };

    let orbit_f64 = &ref_orbit.z_ref_f64;
    let phase_offset = ref_orbit.phase_offset as usize;

    let next_extended = {
        let idx = ref_orbit.extended_iterations.partition_point(|&iter| iter <= *n);
        ref_orbit.extended_iterations.get(idx).copied().unwrap_or(u32::MAX)
    };
    let batch_end = (*n + BATCH_SIZE).min(max_iter).min(next_extended);

    // Optimization from rust-fractal-core: when delta exponent is very negative (< -500),
    // the delta is so small that escape is impossible. Skip bailout/glitch/rebase checks
    // and just do the perturbation math for the entire batch. This is a significant
    // performance win for interior points at moderate zoom levels.
    if delta_exp < -500 && use_scaling {
        while *n < batch_end {
            let idx = *n as usize + phase_offset;
            if idx >= orbit_f64.len() {
                break;
            }
            let z_ref = orbit_f64[idx];

            let two_zr_re = 2.0 * z_ref.re * inv_scale;
            let two_zr_im = 2.0 * z_ref.im * inv_scale;
            let new_re = two_zr_re * d_re - two_zr_im * d_im + sf * (d_re * d_re - d_im * d_im);
            let new_im = two_zr_re * d_im + two_zr_im * d_re + sf * (2.0 * d_re * d_im);
            if IS_JULIA {
                d_re = new_re;
                d_im = new_im;
            } else {
                d_re = new_re + dc_re;
                d_im = new_im + dc_im;
            }
            *n += 1;
            *iters_ptb += 1;
        }
    } else {

    while *n < batch_end {
        let idx = *n as usize + phase_offset;
        if idx >= orbit_f64.len() {
            break;
        }
        let z_ref = orbit_f64[idx];

        // z_curr = z_ref + scale_factor * delta_mantissa
        let z_curr_re = z_ref.re + sf * d_re;
        let z_curr_im = z_ref.im + sf * d_im;
        let z_curr_norm_sqr = z_curr_re * z_curr_re + z_curr_im * z_curr_im;

        if z_curr_norm_sqr > bailout_sqr {
            *delta = if use_scaling {
                ComplexExp {
                    re: FloatExp::new(d_re, delta_exp),
                    im: FloatExp::new(d_im, delta_exp),
                }
            } else {
                ComplexExp::from_complex64(Complex64::new(d_re, d_im))
            };
            return BatchResult::Escaped {
                iteration: *n,
                z_final: Complex64::new(z_curr_re, z_curr_im),
            };
        }

        // Glitch check (scaled): |sf * delta|² > tolerance² * max(|z_ref|², 1e-6)
        let d_norm_sqr = (sf * d_re) * (sf * d_re) + (sf * d_im) * (sf * d_im);
        let z_ref_norm_sqr = (z_ref.re * z_ref.re + z_ref.im * z_ref.im).max(1e-6);
        if !d_norm_sqr.is_finite() || d_norm_sqr > glitch_tolerance_sqr * z_ref_norm_sqr {
            *delta = if use_scaling {
                ComplexExp {
                    re: FloatExp::new(d_re, delta_exp),
                    im: FloatExp::new(d_im, delta_exp),
                }
            } else {
                ComplexExp::from_complex64(Complex64::new(d_re, d_im))
            };
            return BatchResult::Glitched {
                iteration: *n,
                z_final: Complex64::new(z_curr_re, z_curr_im),
            };
        }

        // Rebasing check
        if z_curr_norm_sqr > 0.0 && d_norm_sqr > 0.0 && z_curr_norm_sqr < d_norm_sqr {
            *delta = ComplexExp::from_complex64(Complex64::new(z_curr_re, z_curr_im));
            return BatchResult::NeedRebase {
                z_curr: Complex64::new(z_curr_re, z_curr_im),
            };
        }

        // Perturbation step in scaled space:
        // delta' = 2*z_ref*delta + delta^2 * scale_factor + dc_scaled
        // (note: delta^2 needs to be multiplied by sf because delta is in scaled space)
        let two_zr_re = 2.0 * z_ref.re * inv_scale;
        let two_zr_im = 2.0 * z_ref.im * inv_scale;
        let new_re = two_zr_re * d_re - two_zr_im * d_im + sf * (d_re * d_re - d_im * d_im);
        let new_im = two_zr_re * d_im + two_zr_im * d_re + sf * (2.0 * d_re * d_im);
        if IS_JULIA {
            d_re = new_re;
            d_im = new_im;
        } else {
            d_re = new_re + dc_re;
            d_im = new_im + dc_im;
        }
        *n += 1;
        *iters_ptb += 1;

        if !d_re.is_finite() || !d_im.is_finite() {
            *delta = if use_scaling {
                ComplexExp {
                    re: FloatExp::new(d_re, delta_exp),
                    im: FloatExp::new(d_im, delta_exp),
                }
            } else {
                ComplexExp::from_complex64(Complex64::new(d_re, d_im))
            };
            *suspect = true;
            return BatchResult::Glitched {
                iteration: *n,
                z_final: Complex64::new(d_re, d_im),
            };
        }
    }

    } // close else (escape-check-enabled path)

    // Write back delta from mantissa and normalize.
    // Periodic reduce() inspired by rust-fractal-core: re-normalize mantissa after
    // each batch to prevent gradual precision loss during long iteration sequences.
    *delta = if use_scaling {
        let mut d = ComplexExp {
            re: FloatExp::new(d_re, delta_exp),
            im: FloatExp::new(d_im, delta_exp),
        };
        d.reduce();
        d
    } else {
        ComplexExp::from_complex64(Complex64::new(d_re, d_im))
    };

    // Periodic re-normalization after batch (inspired by rust-fractal-core).
    // During long iteration sequences, floating-point mantissas can drift from
    // the canonical [0.5, 1.0) range, causing gradual precision loss.
    // Re-normalizing after each batch keeps mantissas well-conditioned.
    if *iters_ptb % 256 == 0 {
        delta.reduce();
    }

    // Extended iteration handling (inspired by rust-fractal-core):
    // If the batch ended at an extended iteration (where z_ref underflows f64),
    // perform one step using ComplexExtended arithmetic, then continue.
    // This avoids the overhead of re-entering the outer BLA loop just for one iteration.
    if *n == next_extended && *n < max_iter {
        let idx = *n as usize + phase_offset;
        if idx < ref_orbit.z_ref.len() {
            let z_ref_ext = ref_orbit.z_ref[idx];

            // Check bailout using extended arithmetic
            let z_curr = delta.add(z_ref_ext);
            let z_curr_norm_sqr = z_curr.norm_sqr_approx();

            if z_curr_norm_sqr > bailout_sqr {
                let z_curr_f64 = z_curr.to_complex64_approx();
                return BatchResult::Escaped {
                    iteration: *n,
                    z_final: z_curr_f64,
                };
            }

            // Rebasing check
            let delta_norm_sqr = delta.norm_sqr_approx();
            if z_curr_norm_sqr > 0.0 && delta_norm_sqr > 0.0 && z_curr_norm_sqr < delta_norm_sqr {
                *delta = ComplexExp::from_complex64(z_curr.to_complex64_approx());
                return BatchResult::NeedRebase {
                    z_curr: z_curr.to_complex64_approx(),
                };
            }

            // Perturbation step using ComplexExtended (full exponent tracking)
            let z_ref_2 = ComplexExp {
                re: FloatExp::new(z_ref_ext.re.mantissa * 2.0, z_ref_ext.re.exponent),
                im: FloatExp::new(z_ref_ext.im.mantissa * 2.0, z_ref_ext.im.exponent),
            };
            let linear = delta.mul(z_ref_2);
            let nonlinear = delta.mul(*delta);
            *delta = if IS_JULIA {
                linear.add(nonlinear)
            } else {
                linear.add(nonlinear).add(ComplexExp {
                    re: FloatExp::from_f64(dc_f64.re),
                    im: FloatExp::from_f64(dc_f64.im),
                })
            };
            delta.reduce();
            *n += 1;
            *iters_ptb += 1;
        }
    }

    BatchResult::Continue
}

/// Check if rebasing is beneficial using an improved heuristic.
///
/// Inspired by rust-fractal-core's rebasing strategy: instead of only checking
/// |z_curr| < |delta| (which can trigger premature rebasing when delta and z_ref
/// nearly cancel), also verify that the new delta (z_curr) has meaningfully smaller
/// norm. This avoids "ping-pong" rebasing where delta oscillates between two
/// similar-magnitude values.
///
/// The condition from rust-fractal-core is:
///   rebase when |z_curr|² < |delta|² * REBASE_THRESHOLD
/// where REBASE_THRESHOLD < 1.0 provides hysteresis.
///
/// Additionally, avoid rebasing when |z_ref| is very small (near a zero of the
/// reference orbit), as this is a natural behavior and not a glitch condition.
#[inline(always)]
pub fn should_rebase(z_curr_norm_sqr: f64, delta_norm_sqr: f64, z_ref_norm_sqr: f64) -> bool {
    // Hysteresis factor configurable via env. Defaults to 1.0 (F3-strict: z_curr < delta).
    // Valeurs <1.0 introduisent une marge anti-oscillation (style rust-fractal-core).
    static HYS: OnceLock<f64> = OnceLock::new();
    let hys = *HYS.get_or_init(|| {
        std::env::var("FRACTALL_REBASE_HYSTERESIS")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .unwrap_or(1.0)
            .clamp(0.01, 1.0)
    });

    if z_curr_norm_sqr <= 0.0 || delta_norm_sqr <= 0.0 {
        return false;
    }

    if z_ref_norm_sqr < 1e-20 {
        return false;
    }

    z_curr_norm_sqr < delta_norm_sqr * hys
}

/// Calcule diffabs(c, d) = |c + d| - |c| de manière stable.
///
/// Inspired by rust-fractal-core's `diff_abs()` function.
/// Used for Burning Ship perturbation where we need the variation of |x|.
///
/// This avoids catastrophic cancellation when computing `|a + b| - |a|`
/// for values where `|b| << |a|`.
#[inline]
pub fn diffabs(c: f64, d: f64) -> f64 {
    let cd = c + d;
    let c2d = 2.0 * c + d;
    if c >= 0.0 {
        if cd >= 0.0 {
            d
        } else {
            -c2d
        }
    } else {
        if cd > 0.0 {
            c2d
        } else {
            -d
        }
    }
}

/// Fast scaled perturbation for Burning Ship |(Re(z))|, |Im(z)|)² + c.
///
/// Inspired by rust-fractal-core's `perturb_function` for FRACTAL_TYPE=1 (Burning Ship).
/// Key technique: uses `diffabs()` to compute `|Z_ref + delta| - |Z_ref|` stably,
/// avoiding catastrophic cancellation in the absolute value computation.
///
/// The formula from rust-fractal-core:
///   delta_re' = (2*Z_re + d_re*sf) * d_re - (2*Z_im + d_im*sf) * d_im + dc_re
///   delta_im' = 2 * diffabs(Z_re*Z_im/sf, Z_re*d_im + d_re*(Z_im + d_im*sf))
///             + dc_im
#[inline(never)]
fn fast_burning_ship_batch_f64(
    ref_orbit: &ReferenceOrbit,
    n: &mut u32,
    delta: &mut ComplexExp,
    dc_f64: Complex64,
    bailout_sqr: f64,
    glitch_tolerance_sqr: f64,
    max_iter: u32,
    suspect: &mut bool,
    iters_ptb: &mut u32,
) -> BatchResult {
    // Adaptive batch size: 400/power for power=2 → 200
    const BATCH_SIZE: u32 = 200;

    let mut d_re = delta.re.mantissa;
    let mut d_im = delta.im.mantissa;
    let delta_exp = delta.re.exponent.max(delta.im.exponent);
    let sf = crate::fractal::perturbation::types::pow2i(delta_exp);
    let inv_sf = if delta_exp > -500 && delta_exp < 500 {
        crate::fractal::perturbation::types::pow2i(-delta_exp)
    } else {
        1.0
    };

    let use_scaling = sf.is_finite() && sf > 0.0
        && inv_sf.is_finite() && delta_exp.abs() < 500;

    if !use_scaling {
        d_re = delta.re.to_f64();
        d_im = delta.im.to_f64();
    }

    let (dc_re, dc_im, scale, inv_scale) = if use_scaling {
        (dc_f64.re * inv_sf, dc_f64.im * inv_sf, sf, inv_sf)
    } else {
        (dc_f64.re, dc_f64.im, 1.0, 1.0)
    };

    let orbit_f64 = &ref_orbit.z_ref_f64;
    let phase_offset = ref_orbit.phase_offset as usize;

    let next_extended = {
        let idx = ref_orbit.extended_iterations.partition_point(|&iter| iter <= *n);
        ref_orbit.extended_iterations.get(idx).copied().unwrap_or(u32::MAX)
    };
    let batch_end = (*n + BATCH_SIZE).min(max_iter).min(next_extended);

    let make_delta = |d_re: f64, d_im: f64| -> ComplexExp {
        if use_scaling {
            ComplexExp {
                re: FloatExp::new(d_re, delta_exp),
                im: FloatExp::new(d_im, delta_exp),
            }
        } else {
            ComplexExp::from_complex64(Complex64::new(d_re, d_im))
        }
    };

    // Optimization from rust-fractal-core: when delta exponent is very negative (< -500),
    // the delta is so small that escape is impossible. Skip bailout/glitch/rebase checks
    // and just do the perturbation math for the entire batch.
    if delta_exp < -500 && use_scaling {
        while *n < batch_end {
            let idx = *n as usize + phase_offset;
            if idx >= orbit_f64.len() {
                break;
            }
            let z_ref = orbit_f64[idx];

            let temp_re = d_re;
            d_re = (2.0 * z_ref.re * inv_scale + temp_re * scale * inv_scale) * temp_re
                 - (2.0 * z_ref.im * inv_scale + d_im * scale * inv_scale) * d_im
                 + dc_re;
            d_im = 2.0 * diffabs(
                z_ref.re * z_ref.im * inv_scale,
                z_ref.re * d_im + temp_re * (z_ref.im * inv_scale + d_im * scale * inv_scale),
            ) + dc_im;
            *n += 1;
            *iters_ptb += 1;
        }
    } else {

    while *n < batch_end {
        let idx = *n as usize + phase_offset;
        if idx >= orbit_f64.len() {
            break;
        }
        let z_ref = orbit_f64[idx];

        let z_curr_re = z_ref.re + scale * d_re;
        let z_curr_im = z_ref.im + scale * d_im;
        let z_curr_norm_sqr = z_curr_re * z_curr_re + z_curr_im * z_curr_im;

        if z_curr_norm_sqr > bailout_sqr {
            *delta = make_delta(d_re, d_im);
            return BatchResult::Escaped {
                iteration: *n,
                z_final: Complex64::new(z_curr_re, z_curr_im),
            };
        }

        let d_norm_sqr = (scale * d_re) * (scale * d_re) + (scale * d_im) * (scale * d_im);
        let z_ref_norm_sqr = (z_ref.re * z_ref.re + z_ref.im * z_ref.im).max(1e-6);
        if !d_norm_sqr.is_finite() || d_norm_sqr > glitch_tolerance_sqr * z_ref_norm_sqr {
            *delta = make_delta(d_re, d_im);
            return BatchResult::Glitched {
                iteration: *n,
                z_final: Complex64::new(z_curr_re, z_curr_im),
            };
        }

        if z_curr_norm_sqr > 0.0 && d_norm_sqr > 0.0 && z_curr_norm_sqr < d_norm_sqr {
            *delta = ComplexExp::from_complex64(Complex64::new(z_curr_re, z_curr_im));
            return BatchResult::NeedRebase {
                z_curr: Complex64::new(z_curr_re, z_curr_im),
            };
        }

        // Burning Ship perturbation formula from rust-fractal-core:
        // The key insight is using diffabs to stably compute |Z + delta| - |Z|
        let temp_re = d_re;
        d_re = (2.0 * z_ref.re * inv_scale + temp_re * scale * inv_scale) * temp_re
             - (2.0 * z_ref.im * inv_scale + d_im * scale * inv_scale) * d_im
             + dc_re;
        d_im = 2.0 * diffabs(
            z_ref.re * z_ref.im * inv_scale,
            z_ref.re * d_im + temp_re * (z_ref.im * inv_scale + d_im * scale * inv_scale),
        ) + dc_im;

        *n += 1;
        *iters_ptb += 1;

        if !d_re.is_finite() || !d_im.is_finite() {
            *delta = make_delta(d_re, d_im);
            *suspect = true;
            return BatchResult::Glitched {
                iteration: *n,
                z_final: Complex64::new(d_re, d_im),
            };
        }
    }

    } // close else (escape-check-enabled path)

    // Periodic reduce() after batch: re-normalize mantissa to prevent precision drift.
    *delta = make_delta(d_re, d_im);
    // Periodic re-normalization after batch to prevent mantissa drift
    if *iters_ptb % 256 == 0 {
        delta.reduce();
    }
    BatchResult::Continue
}

/// Fast scaled perturbation for Tricorn: conj(z)² + c.
///
/// Inspired by rust-fractal-core's approach for non-conformal fractals.
/// For Tricorn, the perturbation formula is:
///   z_{n+1} = conj(Z_n + delta_n)² + C + dc
///   delta_{n+1} = z_{n+1} - Z_{n+1}
///
/// Since conjugation is non-conformal, we compute the full z_curr and subtract
/// the reference at the next iteration.
#[inline(never)]
fn fast_tricorn_batch_f64(
    ref_orbit: &ReferenceOrbit,
    n: &mut u32,
    delta: &mut ComplexExp,
    dc_f64: Complex64,
    bailout_sqr: f64,
    glitch_tolerance_sqr: f64,
    max_iter: u32,
    suspect: &mut bool,
    iters_ptb: &mut u32,
) -> BatchResult {
    // Adaptive batch size: 400/power for power=2 → 200
    const BATCH_SIZE: u32 = 200;

    let orbit_f64 = &ref_orbit.z_ref_f64;
    let phase_offset = ref_orbit.phase_offset as usize;
    let c_pixel = ref_orbit.cref + dc_f64;

    let next_extended = {
        let idx = ref_orbit.extended_iterations.partition_point(|&iter| iter <= *n);
        ref_orbit.extended_iterations.get(idx).copied().unwrap_or(u32::MAX)
    };
    let batch_end = (*n + BATCH_SIZE).min(max_iter).min(next_extended);

    // Check if delta is very small (optimization from rust-fractal-core).
    // For Tricorn, delta is tracked as z_curr - z_ref, not scaled mantissa,
    // so we check exponent of the initial delta.
    let delta_exp = delta.re.exponent.max(delta.im.exponent);

    // Optimization from rust-fractal-core: when delta exponent is very negative (< -500),
    // the delta is so small that escape is impossible. Skip bailout/glitch/rebase checks
    // and just do the perturbation math.
    if delta_exp < -500 {
        while *n < batch_end {
            let idx = *n as usize + phase_offset;
            if idx >= orbit_f64.len() {
                break;
            }
            let z_ref = orbit_f64[idx];
            let d_f64 = delta.to_complex64_approx();
            let z_curr_re = z_ref.re + d_f64.re;
            let z_curr_im = z_ref.im + d_f64.im;

            let z_next_re = z_curr_re * z_curr_re - z_curr_im * z_curr_im + c_pixel.re;
            let z_next_im = -2.0 * z_curr_re * z_curr_im + c_pixel.im;

            *n += 1;
            *iters_ptb += 1;

            let next_idx = *n as usize + phase_offset;
            if next_idx >= orbit_f64.len() {
                *delta = ComplexExp::from_complex64(Complex64::new(z_next_re, z_next_im));
                return BatchResult::NeedRebase {
                    z_curr: Complex64::new(z_next_re, z_next_im),
                };
            }
            let z_ref_next = orbit_f64[next_idx];
            *delta = ComplexExp::from_complex64(Complex64::new(
                z_next_re - z_ref_next.re,
                z_next_im - z_ref_next.im,
            ));
        }
    } else {

    while *n < batch_end {
        let idx = *n as usize + phase_offset;
        if idx >= orbit_f64.len() {
            break;
        }
        let z_ref = orbit_f64[idx];
        let delta_f64 = delta.to_complex64_approx();

        let z_curr_re = z_ref.re + delta_f64.re;
        let z_curr_im = z_ref.im + delta_f64.im;
        let z_curr_norm_sqr = z_curr_re * z_curr_re + z_curr_im * z_curr_im;

        if z_curr_norm_sqr > bailout_sqr {
            return BatchResult::Escaped {
                iteration: *n,
                z_final: Complex64::new(z_curr_re, z_curr_im),
            };
        }

        let d_norm_sqr = delta_f64.norm_sqr();
        let z_ref_norm_sqr = (z_ref.re * z_ref.re + z_ref.im * z_ref.im).max(1e-6);
        if !d_norm_sqr.is_finite() || d_norm_sqr > glitch_tolerance_sqr * z_ref_norm_sqr {
            return BatchResult::Glitched {
                iteration: *n,
                z_final: Complex64::new(z_curr_re, z_curr_im),
            };
        }

        if z_curr_norm_sqr > 0.0 && d_norm_sqr > 0.0 && z_curr_norm_sqr < d_norm_sqr {
            *delta = ComplexExp::from_complex64(Complex64::new(z_curr_re, z_curr_im));
            return BatchResult::NeedRebase {
                z_curr: Complex64::new(z_curr_re, z_curr_im),
            };
        }

        // Tricorn: z' = conj(z)² + c
        // conj(z_curr) = (z_curr_re, -z_curr_im)
        // conj(z_curr)² = (z_curr_re² - z_curr_im², -2*z_curr_re*z_curr_im)
        let z_next_re = z_curr_re * z_curr_re - z_curr_im * z_curr_im + c_pixel.re;
        let z_next_im = -2.0 * z_curr_re * z_curr_im + c_pixel.im;

        *n += 1;
        *iters_ptb += 1;

        let next_idx = *n as usize + phase_offset;
        if next_idx >= orbit_f64.len() {
            // Rebase needed
            *delta = ComplexExp::from_complex64(Complex64::new(z_next_re, z_next_im));
            return BatchResult::NeedRebase {
                z_curr: Complex64::new(z_next_re, z_next_im),
            };
        }
        let z_ref_next = orbit_f64[next_idx];
        *delta = ComplexExp::from_complex64(Complex64::new(
            z_next_re - z_ref_next.re,
            z_next_im - z_ref_next.im,
        ));

        let d = delta.to_complex64_approx();
        if !d.re.is_finite() || !d.im.is_finite() {
            *suspect = true;
            return BatchResult::Glitched {
                iteration: *n,
                z_final: Complex64::new(z_next_re, z_next_im),
            };
        }
    }

    } // close else (escape-check-enabled path)

    BatchResult::Continue
}

/// Generate Pascal's triangle coefficients (binomial coefficients) for Multibrot power.
///
/// Inspired by rust-fractal-core's `generate_pascal_coefficients()`.
/// For power d, generates [C(d,0), C(d,1), ..., C(d,d)] as f64.
/// Used in the perturbation formula for z^d + c where d > 2:
///   delta_{n+1} = sum_{k=1}^{d} C(d,k) * Z_ref^(d-k) * delta^k + dc
pub fn generate_pascal_coefficients(power: usize) -> Vec<f64> {
    let mut coeffs = vec![0.0f64; power + 1];
    coeffs[0] = 1.0;
    for i in 1..=power {
        coeffs[i] = coeffs[i - 1] * (power - i + 1) as f64 / i as f64;
    }
    coeffs
}

/// Fast scaled perturbation for Multibrot z^d + c with Pascal coefficients.
///
/// Inspired by rust-fractal-core's generic power perturbation using binomial expansion:
///   (Z + delta)^d = sum_{k=0}^{d} C(d,k) * Z^(d-k) * delta^k
///   delta_{n+1} = (Z + delta)^d - Z^d + dc
///               = sum_{k=1}^{d} C(d,k) * Z^(d-k) * delta^k + dc
///
/// For power 2: delta' = 2*Z*delta + delta² + dc (standard Mandelbrot)
/// For power 3: delta' = 3*Z²*delta + 3*Z*delta² + delta³ + dc
/// For power d: uses Pascal coefficients for the binomial expansion
///
/// Uses scaled delta approach (ldexp-based) like Mandelbrot/BurningShip batches
/// for better numerical stability. Previously used `to_complex64_approx()` per
/// iteration which was slow and imprecise at deep zooms.
#[inline(never)]
fn fast_multibrot_batch_f64(
    ref_orbit: &ReferenceOrbit,
    n: &mut u32,
    delta: &mut ComplexExp,
    dc_f64: Complex64,
    bailout_sqr: f64,
    glitch_tolerance_sqr: f64,
    max_iter: u32,
    power: usize,
    pascal: &[f64],
    suspect: &mut bool,
    iters_ptb: &mut u32,
) -> BatchResult {
    // Adaptive batch size based on fractal power (rust-fractal-core: 400/power)
    let batch_size = adaptive_batch_size(power as f64);

    // Scaled delta approach (inspired by rust-fractal-core):
    // scale_factor = 2^exponent, keep mantissa in f64 range
    let delta_exp = delta.re.exponent.max(delta.im.exponent);
    let scale_factor = crate::fractal::perturbation::types::pow2i(delta_exp);
    let inv_scale = if delta_exp > -500 && delta_exp < 500 {
        crate::fractal::perturbation::types::pow2i(-delta_exp)
    } else {
        1.0
    };

    let use_scaling = scale_factor.is_finite() && scale_factor > 0.0
        && inv_scale.is_finite() && delta_exp.abs() < 500;

    let mut d_re;
    let mut d_im;
    if use_scaling {
        d_re = delta.re.mantissa;
        d_im = delta.im.mantissa;
        // Scale to align exponents
        if delta.re.exponent != delta_exp {
            d_re *= crate::fractal::perturbation::types::pow2i(delta.re.exponent - delta_exp);
        }
        if delta.im.exponent != delta_exp {
            d_im *= crate::fractal::perturbation::types::pow2i(delta.im.exponent - delta_exp);
        }
    } else {
        d_re = delta.re.to_f64();
        d_im = delta.im.to_f64();
    }

    let sf = if use_scaling { scale_factor } else { 1.0 };

    let orbit_f64 = &ref_orbit.z_ref_f64;
    let phase_offset = ref_orbit.phase_offset as usize;

    let next_extended = {
        let idx = ref_orbit.extended_iterations.partition_point(|&iter| iter <= *n);
        ref_orbit.extended_iterations.get(idx).copied().unwrap_or(u32::MAX)
    };
    let batch_end = (*n + batch_size).min(max_iter).min(next_extended);

    let make_delta = |d_re: f64, d_im: f64| -> ComplexExp {
        if use_scaling {
            ComplexExp {
                re: FloatExp::new(d_re, delta_exp),
                im: FloatExp::new(d_im, delta_exp),
            }
        } else {
            ComplexExp::from_complex64(Complex64::new(d_re, d_im))
        }
    };

    while *n < batch_end {
        let idx = *n as usize + phase_offset;
        if idx >= orbit_f64.len() {
            break;
        }
        let z_ref = orbit_f64[idx];

        // z_curr = z_ref + scale_factor * delta_mantissa
        let z_curr_re = z_ref.re + sf * d_re;
        let z_curr_im = z_ref.im + sf * d_im;
        let z_curr_norm_sqr = z_curr_re * z_curr_re + z_curr_im * z_curr_im;

        if z_curr_norm_sqr > bailout_sqr {
            *delta = make_delta(d_re, d_im);
            return BatchResult::Escaped {
                iteration: *n,
                z_final: Complex64::new(z_curr_re, z_curr_im),
            };
        }

        let d_norm_sqr = (sf * d_re) * (sf * d_re) + (sf * d_im) * (sf * d_im);
        let z_ref_norm_sqr = (z_ref.re * z_ref.re + z_ref.im * z_ref.im).max(1e-6);
        if !d_norm_sqr.is_finite() || d_norm_sqr > glitch_tolerance_sqr * z_ref_norm_sqr {
            *delta = make_delta(d_re, d_im);
            return BatchResult::Glitched {
                iteration: *n,
                z_final: Complex64::new(z_curr_re, z_curr_im),
            };
        }

        if z_curr_norm_sqr > 0.0 && d_norm_sqr > 0.0 && z_curr_norm_sqr < d_norm_sqr {
            *delta = ComplexExp::from_complex64(Complex64::new(z_curr_re, z_curr_im));
            return BatchResult::NeedRebase {
                z_curr: Complex64::new(z_curr_re, z_curr_im),
            };
        }

        // Generic power perturbation using binomial expansion (rust-fractal-core approach):
        // delta' = delta * (C(d,1)*Z^(d-1) + delta * (C(d,2)*Z^(d-2) + ... + delta))
        // Work in scaled space: delta_f64 = sf * (d_re, d_im), z_ref in absolute space
        let delta_f64 = Complex64::new(sf * d_re, sf * d_im);
        let mut sum = Complex64::new(pascal[1], 0.0) * z_ref + delta_f64;
        let mut z_p = z_ref;
        for i in 2..power {
            sum = sum * delta_f64;
            z_p = z_p * z_ref;
            sum = sum + Complex64::new(pascal[i], 0.0) * z_p;
        }
        let new_delta = delta_f64 * sum + dc_f64;

        // Store result back in scaled space
        if use_scaling {
            d_re = new_delta.re * inv_scale;
            d_im = new_delta.im * inv_scale;
        } else {
            d_re = new_delta.re;
            d_im = new_delta.im;
        }
        *n += 1;
        *iters_ptb += 1;

        if !new_delta.re.is_finite() || !new_delta.im.is_finite() {
            *delta = make_delta(d_re, d_im);
            *suspect = true;
            return BatchResult::Glitched {
                iteration: *n,
                z_final: Complex64::new(z_curr_re, z_curr_im),
            };
        }
    }

    *delta = make_delta(d_re, d_im);
    // Periodic re-normalization after batch for Multibrot
    if *iters_ptb % 256 == 0 {
        delta.reduce();
    }
    BatchResult::Continue
}

/// Calcule la tolérance de glitch adaptative basée sur le niveau de zoom.
///
/// Plus le zoom est profond, plus la tolérance peut être relaxée car les erreurs
/// numériques sont plus importantes mais moins visibles à grande échelle.
///
/// # Arguments
/// * `pixel_size` - Taille d'un pixel dans l'espace complexe
/// * `user_tolerance` - Tolérance définie par l'utilisateur (1e-4 par défaut)
///
/// # Returns
/// La tolérance adaptative à utiliser pour la détection des glitches.
pub fn compute_adaptive_glitch_tolerance(pixel_size: f64, user_tolerance: f64) -> f64 {
    // Si l'utilisateur a défini une tolérance personnalisée (différente de 1e-4),
    // respecter son choix
    const DEFAULT_TOLERANCE: f64 = 1e-4;
    if (user_tolerance - DEFAULT_TOLERANCE).abs() > 1e-10 {
        return user_tolerance;
    }

    // Calculer le niveau de zoom: log10(4 / pixel_size)
    // À pixel_size = 4.0 (vue complète), zoom_level ≈ 0
    // À pixel_size = 4e-14, zoom_level ≈ 14
    let zoom_level = if pixel_size > 0.0 && pixel_size.is_finite() {
        (4.0 / pixel_size).log10().max(0.0)
    } else {
        0.0
    };

    // Continuous adaptive tolerance scaling (inspired by rust-fractal-core).
    // Instead of discrete steps, use a smooth logarithmic ramp:
    //   tolerance = 10^(-5 + zoom_level * slope)
    // This avoids discontinuities at zoom level boundaries and provides
    // a smoother glitch detection experience across all zoom depths.
    //
    // Clamped to [1e-6, 1e-1] range.
    let slope = 0.1; // tolerance increases by 10x every 10 zoom levels
    let log_tol = -5.0 + zoom_level * slope;
    let tolerance = 10.0f64.powf(log_tol.clamp(-6.0, -1.0));
    tolerance
}

/// Iterate a pixel using perturbation theory (Section 2 of deep zoom theory).
///
/// # Section 2: Perturbation
///
/// Low precision deltas relative to high precision orbit.
///
/// ## Formules mathématiques
///
/// - **Pixel orbit**: `Z_m + z_n` (où `Z_m` est l'orbite de référence haute précision, `z_n` le delta basse précision)
/// - **Point C du pixel**: `C + c` (où `C` est le centre, `c` l'offset du pixel)
/// - **Formule de perturbation**: `z_{n+1} = 2·Z_m·z_n + z_n² + c`
///
/// ## Initialisation
///
/// - `m` et `n` commencent à 0 (`m = 0`, `n = 0`)
/// - `z_0 = 0` (delta initial = 0 pour Mandelbrot)
///
/// **Note**: Dans le code, `m` et `n` sont représentés par une seule variable `n` qui est toujours
/// synchronisée (`m = n`). Pour Julia, l'initialisation diffère: `z_0 = c` (delta initial = offset du pixel).
///
/// ## Rebasing
///
/// Rebasing to avoid glitches: when `|Z_m + z_n| < |z_n|`, replace `z_n` with `Z_m + z_n`
/// and reset the reference iteration count `m` to 0.
///
/// ## Notation dans le code
///
/// **Important**: Dans le code, il n'y a qu'une seule variable `n` qui représente à la fois
/// `m` (l'index de l'orbite de référence) et `n` (l'itération du delta). Ils sont toujours
/// synchronisés (`m = n` dans notre implémentation).
///
/// - `Z_m` ↔ `ref_orbit.z_ref_f64[n]` (orbite de référence haute précision à l'itération `n`)
/// - `z_n` ↔ `delta` (delta de perturbation basse précision, type `ComplexExp`)
/// - `C` ↔ `ref_orbit.cref` (point de référence, centre de l'image)
/// - `c` ↔ `dc` (offset du pixel par rapport au centre, type `ComplexExp`)
/// - `m` et `n` ↔ `n` (une seule variable dans le code, toujours synchronisée)
///
/// # Arguments
///
/// * `params` - Paramètres de la fractale
/// * `ref_orbit` - Orbite de référence haute précision calculée au centre (ou référence de phase pour Hybrid BLA)
/// * `bla_table` - Table BLA (Bivariate Linear Approximation) pour sauter des itérations.
///   Approxime `l` itérations par `z_{n+l} = A_{n,l}·z_n + B_{n,l}·c` quand valide.
///   For Hybrid BLA: one BLA table per reference (one per phase).
/// * `series_table` - Table de séries pour approximation (optionnelle)
/// * `delta0` - Delta initial (`z_0`): `ComplexExp::zero()` pour Mandelbrot, `dc` pour Julia
/// * `dc` - Offset du pixel par rapport au centre (`c` dans la formule)
///
/// # Hybrid BLA
///
/// For a hybrid loop with multiple phases, this function is called with the reference
/// for the current phase. Rebasing switches to the reference for the current phase.
///
/// # Parameters
///
/// * `current_phase` - Current phase (for Hybrid BLA). Updated when rebasing occurs.
/// * `hybrid_refs` - Hybrid BLA references (for Hybrid BLA). Used to calculate new phase on rebasing.
pub fn iterate_pixel(
    params: &FractalParams,
    ref_orbit: &ReferenceOrbit,
    bla_table: &BlaTable,
    series_table: Option<&SeriesTable>,
    delta0: ComplexExp,
    dc: ComplexExp,
    mut current_phase: Option<&mut u32>,
    hybrid_refs: Option<&HybridBlaReferences>,
) -> DeltaResult {
    let rebase_stride = rebase_stride();

    // P3.1 : path bytecode unifié (BLA mat2 + delta-form + rebasing F3).
    // Supporte désormais distance/interior/orbit_traps via ddelta tracking.
    // Si activé et type supporté, dispatch vers le pixel loop unifié.
    if params.use_bytecode_engine {
        if let Some(result) =
            try_bytecode_unified_path(params, ref_orbit, &delta0, &dc)
        {
            return result;
        }
    }

    // Distance estimation et interior detection sont supportés par le
    // path bytecode (cf. dispatch ci-dessus). Si bytecode désactivé,
    // ces features ne sont plus disponibles en perturbation.
    if (params.enable_distance_estimation || params.enable_interior_detection)
        && !params.use_bytecode_engine
    {
        eprintln!(
            "[WARN] distance_estimation/interior_detection nécessite use_bytecode_engine \
             (passer --bytecode ou retirer --no-bytecode). Rendu sans ces features."
        );
    }

    // Standard path without dual numbers
    // Compteurs alignés C++ Fraktaler-3: iters_ptb (itérations perturbation), steps_bla (pas BLA).
    // Limites séparées max_perturb_iterations / max_bla_steps (0 = illimité).
    let mut n = 0u32;
    let mut iters_ptb = 1u32;  // C++: iters_ptb = 1 au départ
    let mut steps_bla = 0u32;  // C++: steps_bla = 0 au départ
    let mut delta = delta0;
    // Use z_ref_f64 for fast path iteration (z_ref is high-precision Vec<ComplexExp>)
    // Hybrid BLA: account for phase offset in effective length
    let effective_len = ref_orbit.effective_len() as u32;
    let max_iter = params.iteration_max.min(effective_len.saturating_sub(1));
    let bailout_sqr = params.bailout * params.bailout;
    let mut phase_changed = false;

    // Calculer le pixel_size pour la tolérance adaptative
    let pixel_size = params.span_x / params.width as f64;
    
    // DÉSACTIVÉ: Optimisation pour le centre exact qui causait des artefacts circulaires visibles.
    // La perturbation standard fonctionne correctement même au centre exact.
    // Variables conservées pour référence future mais non utilisées:
    // let dc_norm_sqr = dc.norm_sqr_approx();
    // let delta_norm_sqr_initial = delta.norm_sqr_approx();
    // let pixel_size_sqr = pixel_size * pixel_size;
    // let center_threshold = pixel_size_sqr * 0.01;
    // let delta_threshold = pixel_size_sqr * 1e-8;
    // let is_center_like = dc_norm_sqr < center_threshold && delta_norm_sqr_initial < delta_threshold;
    let adaptive_tolerance = compute_adaptive_glitch_tolerance(pixel_size, params.glitch_tolerance);
    let glitch_tolerance_sqr = adaptive_tolerance * adaptive_tolerance;
    let is_julia = params.fractal_type == FractalType::Julia;
    let is_burning_ship = params.fractal_type == FractalType::BurningShip;
    let is_multibrot = params.fractal_type == FractalType::Multibrot;
    let is_tricorn = params.fractal_type == FractalType::Tricorn;
    let multibrot_power = params.multibrot_power;
    let smooth_power = if is_multibrot { params.multibrot_power } else { 2.0 };
    let series_config = SeriesConfig::from_params(params);
    let mut suspect = false;

    // Use high-precision orbit when pixel size is very small (deep zoom)
    // This helps avoid precision loss when z_ref values are at extreme ranges
    let use_high_precision = pixel_size < 1e-14;

    // Try standalone series skip BEFORE BLA (if enabled and series table is available).
    // For Mandelbrot: the series variable is dc (pixel offset), since delta_0 = 0.
    // For Julia: the series variable is also dc (= delta_0), since delta_0 = dc.
    // In both cases, dc is the "small parameter" that the series is expanded in.
    //
    // Enhanced with per-pixel tiled validation (inspired by rust-fractal-core's
    // check_approximation() tiled mode): each pixel can skip a different number
    // of iterations based on its position in the image. Pixels near the center
    // can often skip more iterations than edge pixels.
    if let Some(table) = series_table {
        if params.series_standalone && !is_burning_ship && !is_multibrot && !is_tricorn {
            // Compute per-pixel series skip using tiled validation if available
            let pixel_max_skip = if let Some(ref tiled) = table.tiled_validation {
                // Estimate pixel position from dc
                let dc_approx = dc.to_complex64_approx();
                let px = if params.span_x != 0.0 && params.span_x.is_finite() {
                    ((dc_approx.re / params.span_x + 0.5) * params.width as f64) as usize
                } else {
                    params.width as usize / 2
                };
                let py = if params.span_y != 0.0 && params.span_y.is_finite() {
                    ((dc_approx.im / params.span_y + 0.5) * params.height as f64) as usize
                } else {
                    params.height as usize / 2
                };
                tiled.valid_iteration_for_pixel(px, py, params.width as usize, params.height as usize)
            } else {
                table.validated_skip
            };

            if let Some(skip_result) = compute_series_skip(
                table,
                dc,
                series_config.error_tolerance,
            ) {
                // Use the minimum of the computed skip and the per-pixel validated skip
                let effective_skip = if pixel_max_skip > 0 {
                    skip_result.skip_to.min(pixel_max_skip)
                } else {
                    skip_result.skip_to
                };
                if effective_skip >= 2 {
                    n = effective_skip as u32;
                    // Re-evaluate series at the effective skip point if it differs
                    if effective_skip == skip_result.skip_to {
                        delta = skip_result.delta;
                    } else {
                        // Re-evaluate the series at the clamped iteration
                        let dc_f64_tmp = dc.to_complex64_approx();
                        if effective_skip < table.coeffs.len() {
                            let coeffs = &table.coeffs[effective_skip];
                            let approx = dc_f64_tmp * (coeffs.a + dc_f64_tmp * (coeffs.b + dc_f64_tmp * (coeffs.c + dc_f64_tmp * coeffs.d)));
                            delta = ComplexExp::from_complex64(approx);
                        } else {
                            delta = skip_result.delta;
                        }
                    }
                    if skip_result.estimated_error > series_config.error_tolerance * 0.5 {
                        suspect = true;
                    }
                }
            }
        }
    }

    // Pre-cache dc as f64 (loop invariant, avoids repeated ComplexExp -> f64 conversion)
    let dc_f64 = dc.to_complex64_approx();

    // Determine if we can use the fast f64 batch path.
    // Conditions: not deep zoom needing high precision.
    // We now have fast batch paths for Mandelbrot, Julia, Burning Ship, Tricorn, and Multibrot.
    let can_use_fast_f64 = !use_high_precision;

    // Pre-compute Pascal coefficients for Multibrot (inspired by rust-fractal-core)
    let pascal = if is_multibrot {
        generate_pascal_coefficients(multibrot_power as usize)
    } else {
        Vec::new()
    };

    // Main iteration loop: BLA Table Lookup algorithm
    // Based on fraktaler-3 implementation with nested BLA loop for consecutive steps
    // For each iteration:
    // 1. Nested BLA loop: apply consecutive BLA steps until no more valid BLA is found
    //    - Check rebasing at each BLA iteration
    // 2. If no BLA was applied, do a perturbation iteration (fast f64 batch when possible)
    // 3. Check for rebasing opportunities after perturbation step
    // Limites séparées (C++: iters_ptb < PerturbIterations && steps_bla < BLASteps)
    let limit_ptb = params.max_perturb_iterations;
    let limit_bla = params.max_bla_steps;

    // BLA level hint: track last successful level to avoid scanning from the top each time.
    // As delta grows, the valid BLA level tends to decrease monotonically, so starting
    // the search near the last successful level amortizes to O(1).
    let mut last_nc_bla_level: usize = if bla_table.nc_num_levels() > 0 { bla_table.nc_num_levels() - 1 } else { 0 };
    let mut last_conf_bla_level: usize = if bla_table.num_levels() > 0 { bla_table.num_levels() - 1 } else { 0 };
    'outer: while n < max_iter
        && (limit_ptb == 0 || iters_ptb < limit_ptb)
        && (limit_bla == 0 || steps_bla < limit_bla)
    {
        // Nested BLA loop: apply consecutive BLA steps (like C++ do...while(b))
        // This allows multiple BLA steps to be applied in sequence, improving performance
        let mut bla_applied = false;
        // Cache de la norme carrée pour éviter les recalculs (optimisation 2)
        let mut delta_norm_sqr_cached = delta.norm_sqr_approx();
        // Cache de la conversion ComplexExp → Complex64 (optimisation 1)
        let mut delta_approx_cached = delta.to_complex64_approx();
        loop {
            // C++: break if limits exceeded (lines 293-294)
            if limit_ptb != 0 && iters_ptb >= limit_ptb {
                break;
            }
            if limit_bla != 0 && steps_bla >= limit_bla {
                break;
            }
            // Optimisation 4: Vérifier rebasing seulement si nécessaire (fin d'orbite effective)
            // Les autres vérifications de rebasing sont faites après application BLA pour améliorer les performances
            if n >= effective_len {
                // Reached end of effective orbit: rebase
                let last_idx = effective_len.saturating_sub(1) as usize;
                let z_ref = ref_orbit.get_z_ref_f64(last_idx as u32).unwrap_or_else(|| {
                    ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
                });
                // Utiliser le cache de delta_approx (optimisation 1)
                let z_curr = z_ref + delta_approx_cached;
                delta = ComplexExp::from_complex64(z_curr);
                // Mettre à jour les caches après rebasing
                delta_approx_cached = delta.to_complex64_approx();
                delta_norm_sqr_cached = delta.norm_sqr_approx();
                
                // Hybrid BLA: change phase on rebasing: phase = (phase + n) % cycle_period
                if let Some(ref mut phase) = current_phase {
                    if let Some(refs) = hybrid_refs {
                        if refs.cycle_period > 0 {
                            **phase = (**phase + n) % refs.cycle_period;
                            phase_changed = true;
                        }
                    }
                }
                n = 0;
                // Reset BLA level hints after rebase (delta is small again)
                last_nc_bla_level = if bla_table.nc_num_levels() > 0 { bla_table.nc_num_levels() - 1 } else { 0 };
                last_conf_bla_level = if bla_table.num_levels() > 0 { bla_table.num_levels() - 1 } else { 0 };
            }

            // Try to find and apply a BLA step
            let mut stepped = false;

            // Try non-conformal BLA first for Tricorn and Burning Ship.
            // Burning Ship uses non-conformal BLA (2×2 matrices) because the absolute
            // value operation makes the Jacobian anti-conformal in the 2nd/4th quadrant.
            // Complex multiplication cannot correctly compose these transformations
            // across BLA levels.
            if is_tricorn || is_burning_ship {
                if bla_table.nc_num_levels() > 0 {
                    {
                        // Utiliser le cache de delta_approx (optimisation 1)
                        let delta_vec = (delta_approx_cached.re, delta_approx_cached.im);
                        let delta_norm_sqr_check = delta_vec.0 * delta_vec.0 + delta_vec.1 * delta_vec.1;

                        // BLA level hint: start from last successful level + 1 (clamped)
                        let start_nc_level = last_nc_bla_level.min(bla_table.nc_num_levels() - 1);
                        for level in (0..=start_nc_level).rev() {
                            if (n as usize) >= bla_table.nc_level_len(level) {
                                continue;
                            }
                            let node = bla_table.get_nc_node(level, n as usize).unwrap();

                            if delta_norm_sqr_check < node.validity_radius * node.validity_radius {
                                // Apply non-conformal BLA: z_{n+l} = A_{n,l}·z_n + B_{n,l}·c
                                // For non-conformal fractals (Tricorn), A and B are 2×2 real matrices
                                // instead of complex numbers, as angles are not preserved.
                                let dc_approx = dc.to_complex64_approx();
                                let dc_vec = (dc_approx.re, dc_approx.im);

                                // Linear term: A_{n,l}·z_n
                                let linear_vec = node.a.mul_vector(delta_vec.0, delta_vec.1);

                                // Add dc term: B_{n,l}·c
                                let dc_term_vec = node.b.mul_vector(dc_vec.0, dc_vec.1);

                                let next_vec = (linear_vec.0 + dc_term_vec.0, linear_vec.1 + dc_term_vec.1);

                                // Convert back to ComplexExp
                                delta = ComplexExp::from_complex64(Complex64::new(next_vec.0, next_vec.1));
                                n += 1u32 << level;  // Skip l = 2^level iterations
                                stepped = true;
                                bla_applied = true;
                                steps_bla += 1;  // C++: steps_bla++ après chaque pas BLA
                                // Update BLA level hint for next iteration
                                last_nc_bla_level = level;
                                // Mettre à jour les caches après application BLA
                                delta_approx_cached = delta.to_complex64_approx();
                                delta_norm_sqr_cached = delta.norm_sqr_approx();
                                
                                // Rebasing post-BLA avec hysteresis via should_rebase().
                                if n < effective_len && (rebase_stride == 1 || (n % rebase_stride) == 0) {
                                    let z_ref_check = ref_orbit.get_z_ref_f64(n).unwrap_or_else(|| {
                                        ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
                                    });
                                    let z_curr_check = z_ref_check + delta_approx_cached;
                                    let z_curr_norm_sqr_check = z_curr_check.norm_sqr();
                                    let z_ref_norm_sqr_check = z_ref_check.norm_sqr();

                                    if should_rebase(z_curr_norm_sqr_check, delta_norm_sqr_cached, z_ref_norm_sqr_check) {
                                        delta = ComplexExp::from_complex64(z_curr_check);
                                        delta_approx_cached = delta.to_complex64_approx();
                                        delta_norm_sqr_cached = delta.norm_sqr_approx();

                                        if let Some(ref mut phase) = current_phase {
                                            if let Some(refs) = hybrid_refs {
                                                if refs.cycle_period > 0 {
                                                    **phase = (**phase + n) % refs.cycle_period;
                                                    phase_changed = true;
                                                }
                                            }
                                        }
                                        n = 0;
                                        last_nc_bla_level = if bla_table.nc_num_levels() > 0 { bla_table.nc_num_levels() - 1 } else { 0 };
                                        last_conf_bla_level = if bla_table.num_levels() > 0 { bla_table.num_levels() - 1 } else { 0 };
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
            }

            // BLA Table Lookup:
            // Find the BLA starting from iteration m (n in our code) that has the largest skip l
            // satisfying |z| < R. If there is none, do a perturbation iteration.
            // Check for rebasing opportunities after each BLA application or perturbation step.
            //
            // Bivariate Linear Approximation (BLA):
            // Sometimes, l iterations starting at n can be approximated by bivariate linear function:
            // z_{n+l} = A_{n,l}·z_n + B_{n,l}·c
            // This is valid when the non-linear part of the full perturbation iterations is so small
            // that omitting it would cause fewer problems than the rounding error of the low precision data type.
            //
            // Note: BLA is automatically disabled for very deep zooms (>10^15) where f64 precision
            // is insufficient. In such cases, the BLA table will be empty and full GMP perturbation is used.
            //
            // Search strategy: iterate levels in reverse order (largest skip first) to find the largest
            // valid skip l = 2^level satisfying |z_n| < R_{n,l}.
            //
            // Optimization inspired by rust-fractal-core: track the last successful BLA level
            // and start searching from there instead of always from the top. When delta is growing
            // (which happens as perturbation proceeds), the valid level tends to decrease monotonically.
            // Conformal BLA: only for Mandelbrot/Julia/Multibrot (NOT Burning Ship or Tricorn).
            // Burning Ship and Tricorn use the non-conformal BLA path above.
            if !stepped && !is_burning_ship && !is_tricorn && bla_table.num_levels() > 0 {
                // Utiliser le cache de la norme au lieu de recalculer (optimisation 2)
                let delta_norm_sqr = delta_norm_sqr_cached;
                // Search from last successful level (BLA level hint) to lowest level
                let start_conf_level = last_conf_bla_level.min(bla_table.num_levels() - 1);
                for level in (0..=start_conf_level).rev() {
                    if (n as usize) >= bla_table.level_len(level) {
                        continue;
                    }
                    let node = bla_table.get_node(level, n as usize).unwrap();
                    // Single Step BLA validity condition: |z_n| < R_{n,l}
                    // Derived from: |z_n²| << |2·Z_n·z_n + c|
                    // Assuming negligibility of c: |z_n| << |2·Z_n| = |A_{n,1}|
                    // Therefore: |z_n| < R_{n,l} where R_{n,l} = ε·|A_{n,l}|
                    if delta_norm_sqr < node.validity_radius * node.validity_radius {
                        // Note: Burning Ship/Tricorn use the non-conformal BLA path,
                        // so work_delta == delta here (no sign transformation needed).
                        let work_delta = delta;

                        if should_use_series(series_config, delta_norm_sqr, node.validity_radius) {
                            // Use series approximation with higher-order terms
                            let delta_sq = work_delta.mul(work_delta);
                            let mut next_delta = work_delta.mul_complex64(node.a);  // A_{n,l}·z_n
                            if !is_julia {
                                next_delta = next_delta.add(dc.mul_complex64(node.b));  // + B_{n,l}·c
                            }
                            // Terme quadratique (ordre 2): C·z_n²
                            next_delta = next_delta.add(delta_sq.mul_complex64(node.c));

                            // Termes d'ordre supérieur pour Multibrot (ordre 4-6)
                            if series_config.order >= 4 && is_multibrot {
                                // Terme cubique δ³
                                let delta_cube = delta_sq.mul(work_delta);
                                if node.d.norm() > 1e-20 {
                                    next_delta = next_delta.add(delta_cube.mul_complex64(node.d));
                                }

                                // Terme quartique δ⁴ (ordre 5-6)
                                if series_config.order >= 5 && node.e.norm() > 1e-20 {
                                    let delta_4 = delta_sq.mul(delta_sq);
                                    next_delta = next_delta.add(delta_4.mul_complex64(node.e));
                                }
                            }

                            let coeff_a_norm = node.a.norm();
                            let err = estimate_series_error(
                                delta_norm_sqr,
                                series_config.order,
                                level,
                                coeff_a_norm,
                            );
                            if err > series_config.error_tolerance {
                                suspect = true;
                            }
                            delta = next_delta;
                        } else {
                            // Apply BLA: z_{n+l} = A_{n,l}·z_n + B_{n,l}·c
                            let mut next_delta = work_delta.mul_complex64(node.a);  // A_{n,l}·z_n
                            if !is_julia {
                                next_delta = next_delta.add(dc.mul_complex64(node.b));  // + B_{n,l}·c
                            }
                            delta = next_delta;
                        }
                        // Skip l = 2^level iterations: z_{n+l} has been computed
                        n += 1u32 << level;
                        stepped = true;
                        bla_applied = true;
                        steps_bla += 1;  // C++: steps_bla++ après chaque pas BLA
                        // Update BLA level hint for next iteration
                        last_conf_bla_level = level;
                        // Mettre à jour les caches après application BLA
                        delta_approx_cached = delta.to_complex64_approx();
                        delta_norm_sqr_cached = delta.norm_sqr_approx();

                        // Rebasing post-BLA avec hysteresis via should_rebase().
                        if n < effective_len && (rebase_stride == 1 || (n % rebase_stride) == 0) {
                            let z_ref_check = ref_orbit.get_z_ref_f64(n).unwrap_or_else(|| {
                                ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
                            });
                            let z_curr_check = z_ref_check + delta_approx_cached;
                            let z_curr_norm_sqr_check = z_curr_check.norm_sqr();
                            let z_ref_norm_sqr_check = z_ref_check.norm_sqr();

                            if should_rebase(z_curr_norm_sqr_check, delta_norm_sqr_cached, z_ref_norm_sqr_check) {
                                delta = ComplexExp::from_complex64(z_curr_check);
                                delta_approx_cached = delta.to_complex64_approx();
                                delta_norm_sqr_cached = delta.norm_sqr_approx();

                                if let Some(ref mut phase) = current_phase {
                                    if let Some(refs) = hybrid_refs {
                                        if refs.cycle_period > 0 {
                                            **phase = (**phase + n) % refs.cycle_period;
                                            phase_changed = true;
                                        }
                                    }
                                }
                                n = 0;
                                last_nc_bla_level = if bla_table.nc_num_levels() > 0 { bla_table.nc_num_levels() - 1 } else { 0 };
                                last_conf_bla_level = if bla_table.num_levels() > 0 { bla_table.num_levels() - 1 } else { 0 };
                            }
                        }
                        break; // Found a BLA step, continue nested loop to try another
                    }
                }
            }

            // If no BLA step was found, exit the nested loop
            if !stepped {
                break;
            }
        }

        // If no BLA was applied, do perturbation iteration(s)
        if !bla_applied {
            // Fast f64 batch path: processes up to 256 iterations using pure f64
            // arithmetic, avoiding ComplexExp overhead. Now supports Mandelbrot, Julia,
            // Burning Ship, Tricorn, and Multibrot (inspired by rust-fractal-core).
            if can_use_fast_f64 {
                let batch_result = if is_burning_ship {
                    fast_burning_ship_batch_f64(
                        ref_orbit, &mut n, &mut delta, dc_f64,
                        bailout_sqr, glitch_tolerance_sqr, max_iter,
                        &mut suspect, &mut iters_ptb,
                    )
                } else if is_tricorn {
                    fast_tricorn_batch_f64(
                        ref_orbit, &mut n, &mut delta, dc_f64,
                        bailout_sqr, glitch_tolerance_sqr, max_iter,
                        &mut suspect, &mut iters_ptb,
                    )
                } else if is_multibrot {
                    fast_multibrot_batch_f64(
                        ref_orbit, &mut n, &mut delta, dc_f64,
                        bailout_sqr, glitch_tolerance_sqr, max_iter,
                        multibrot_power as usize, &pascal,
                        &mut suspect, &mut iters_ptb,
                    )
                } else {
                    if is_julia {
                        fast_mandelbrot_batch_f64::<true>(
                            ref_orbit, &mut n, &mut delta, dc_f64,
                            bailout_sqr, glitch_tolerance_sqr, max_iter,
                            effective_len, &mut suspect, &mut iters_ptb,
                        )
                    } else {
                        fast_mandelbrot_batch_f64::<false>(
                            ref_orbit, &mut n, &mut delta, dc_f64,
                            bailout_sqr, glitch_tolerance_sqr, max_iter,
                            effective_len, &mut suspect, &mut iters_ptb,
                        )
                    }
                };
                match batch_result {
                    BatchResult::Escaped { iteration, z_final } => {
                        return DeltaResult {
                            iteration,
                            z_final,
                            glitched: false,
                            suspect,
                            distance: f64::INFINITY,
                            is_interior: false,
                            phase_changed,
                            smooth_iteration: compute_smooth_iteration(iteration, z_final, params.bailout, smooth_power),
                        };
                    }
                    BatchResult::Glitched { iteration, z_final } => {
                        return DeltaResult {
                            iteration,
                            z_final,
                            glitched: true,
                            suspect,
                            distance: f64::INFINITY,
                            is_interior: false,
                            phase_changed,
                        smooth_iteration: 0.0,
                        };
                    }
                    BatchResult::Continue => {
                        // Batch completed normally, update caches
                        // and fall through to rebasing/bailout/glitch checks below
                        delta_approx_cached = delta.to_complex64_approx();
                        delta_norm_sqr_cached = delta.norm_sqr_approx();
                    }
                    BatchResult::NeedRebase { z_curr } => {
                        // Rebasing: replace delta with z_curr and reset n to 0
                        delta = ComplexExp::from_complex64(z_curr);
                        delta_approx_cached = z_curr;
                        delta_norm_sqr_cached = z_curr.norm_sqr();
                        // Hybrid BLA: change phase on rebasing
                        if let Some(ref mut phase) = current_phase {
                            if let Some(refs) = hybrid_refs {
                                if refs.cycle_period > 0 {
                                    **phase = (**phase + n) % refs.cycle_period;
                                    phase_changed = true;
                                }
                            }
                        }
                        n = 0;
                        // Reset BLA level hints after rebase
                        last_nc_bla_level = if bla_table.nc_num_levels() > 0 { bla_table.nc_num_levels() - 1 } else { 0 };
                        last_conf_bla_level = if bla_table.num_levels() > 0 { bla_table.num_levels() - 1 } else { 0 };
                        continue 'outer;
                    }
                }
            } else {

            // Single-step paths for non-standard fractal types
            let z_ref = ref_orbit.get_z_ref_f64(n).unwrap_or_else(|| {
                ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
            });

            if is_burning_ship {
                let delta_approx = delta_approx_cached;
                let z_curr = z_ref + delta_approx;

                let ref_sign_re = if z_ref.re >= 0.0 { 1.0 } else { -1.0 };
                let ref_sign_im = if z_ref.im >= 0.0 { 1.0 } else { -1.0 };
                let curr_sign_re = if z_curr.re >= 0.0 { 1.0 } else { -1.0 };
                let curr_sign_im = if z_curr.im >= 0.0 { 1.0 } else { -1.0 };

                let quadrant_stable = ref_sign_re == curr_sign_re && ref_sign_im == curr_sign_im;

                if quadrant_stable {
                    let delta_diffabs_re = diffabs(z_ref.re, delta_approx.re);
                    let delta_diffabs_im = diffabs(z_ref.im, delta_approx.im);
                    let delta_diffabs = ComplexExp::from_complex64(Complex64::new(delta_diffabs_re, delta_diffabs_im));
                    let z_abs = Complex64::new(z_ref.re.abs(), z_ref.im.abs());
                    let linear = delta_diffabs.mul_complex64(z_abs * 2.0);
                    let nonlinear = delta_diffabs.mul(delta_diffabs);
                    delta = linear.add(nonlinear).add(dc);
                } else {
                    let re = z_curr.re.abs();
                    let im = z_curr.im.abs();
                    let mut z_temp = Complex64::new(re, im);
                    z_temp = z_temp * z_temp;
                    let c_pixel = ref_orbit.cref + dc_f64;
                    let z_next = z_temp + c_pixel;
                    let next_index = n + 1;
                    let z_ref_next = match ref_orbit.get_z_ref_f64(next_index) {
                        Some(z) => z,
                        None => break,
                    };
                    delta = ComplexExp::from_complex64(z_next - z_ref_next);
                }
                n += 1;
            } else if is_multibrot {
                let d = multibrot_power;
                let z_norm = z_ref.norm();
                if z_norm > 1e-15 {
                    let a = z_ref.powf(d - 1.0) * d;
                    let linear = delta.mul_complex64(a);
                    let c_coeff = d * (d - 1.0) / 2.0;
                    let c = z_ref.powf(d - 2.0) * c_coeff;
                    let nonlinear = delta.mul(delta).mul_complex64(c);
                    delta = linear.add(nonlinear).add(dc);
                } else {
                    delta = dc;
                }
                n += 1;
            } else if is_tricorn {
                let coeffs = crate::fractal::perturbation::nonconformal::compute_tricorn_bla_coefficients(z_ref);
                let delta_vec = (delta_approx_cached.re, delta_approx_cached.im);
                let linear_vec = coeffs.a.mul_vector(delta_vec.0, delta_vec.1);
                let delta_conj_sq_re = delta_vec.0 * delta_vec.0 - delta_vec.1 * delta_vec.1;
                let delta_conj_sq_im = -2.0 * delta_vec.0 * delta_vec.1;
                let dc_vec = (dc_f64.re, dc_f64.im);
                let dc_term_vec = coeffs.b.mul_vector(dc_vec.0, dc_vec.1);
                let next_vec = (
                    linear_vec.0 + delta_conj_sq_re + dc_term_vec.0,
                    linear_vec.1 + delta_conj_sq_im + dc_term_vec.1,
                );
                delta = ComplexExp::from_complex64(Complex64::new(next_vec.0, next_vec.1));
                n += 1;
            } else if use_high_precision {
                let z_ref_hp = ref_orbit.get_z_ref(n).unwrap_or_else(|| {
                    ref_orbit.z_ref[ref_orbit.z_ref.len().saturating_sub(1)]
                });
                let z_ref_2 = ComplexExp {
                    re: crate::fractal::perturbation::types::FloatExp::new(
                        z_ref_hp.re.mantissa * 2.0,
                        z_ref_hp.re.exponent,
                    ),
                    im: crate::fractal::perturbation::types::FloatExp::new(
                        z_ref_hp.im.mantissa * 2.0,
                        z_ref_hp.im.exponent,
                    ),
                };
                let linear = delta.mul(z_ref_2);
                let nonlinear = delta.mul(delta);
                delta = if is_julia {
                    linear.add(nonlinear)
                } else {
                    linear.add(nonlinear).add(dc)
                };
                n += 1;
            } else {
                // Fallback single-step (should rarely be reached for Mandelbrot/Julia
                // since can_use_fast_f64 would be true)
                let z_ref_2 = Complex64::new(z_ref.re * 2.0, z_ref.im * 2.0);
                let linear = delta.mul_complex64(z_ref_2);
                let nonlinear = delta.mul(delta);
                delta = if is_julia {
                    linear.add(nonlinear)
                } else {
                    linear.add(nonlinear).add(dc)
                };
                n += 1;
            }
            // Periodic reduce() to re-normalize mantissa and prevent gradual precision
            // loss during long iteration sequences. Inspired by rust-fractal-core which
            // calls reduce() every ~250 iterations to keep mantissas well-conditioned.
            if iters_ptb % 250 == 0 {
                delta.reduce();
            }
            // Update caches after single-step perturbation
            delta_approx_cached = delta.to_complex64_approx();
            delta_norm_sqr_cached = delta.norm_sqr_approx();
            iters_ptb += 1;
            } // close else (single-step paths)
        }

        // Check rebasing after perturbation iteration
        // Note: Rebasing is also checked at the start of the BLA loop, but we check again here
        // after a perturbation step to ensure we catch all rebasing opportunities
        // For high-precision path, use ComplexExp for z_curr calculation
        let (z_curr, z_ref_norm_sqr) = if use_high_precision && !is_burning_ship && !is_multibrot && !is_tricorn {
            let z_ref_hp = ref_orbit.get_z_ref(n).unwrap_or_else(|| {
                ref_orbit.z_ref[ref_orbit.z_ref.len().saturating_sub(1)]
            });
            let z_curr_hp = z_ref_hp.add(delta);
            let z_curr = z_curr_hp.to_complex64_approx();
            let z_ref_norm = z_ref_hp.norm_sqr_approx();
            (z_curr, z_ref_norm)
        } else {
            let z_ref = ref_orbit.get_z_ref_f64(n).unwrap_or_else(|| {
                ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
            });
            // Utiliser le cache de delta_approx (optimisation 1)
            let z_curr = z_ref + delta_approx_cached;
            (z_curr, z_ref.norm_sqr())
        };

        // ====================================================================================
        // Check for rebasing opportunities after each BLA application or perturbation step
        // ====================================================================================
        //
        // Rebasing to avoid glitches: when |Z_m + z_n| < |z_n|, replace z_n with Z_m + z_n
        // and reset the reference iteration count m to 0.
        //
        // This check is performed after each BLA application or perturbation iteration step
        // as specified in the BLA Table Lookup algorithm.
        //
        // Implémentation:
        // - z_curr = Z_m + z_n (où Z_m = z_ref[n] et z_n = delta)
        //   Note: Dans le code, m et n sont représentés par la même variable n (m = n toujours)
        // - Condition: |z_curr| < |delta| (équivalent à |Z_m + z_n| < |z_n|)
        // - Si la condition est vraie:
        //   * delta ← z_curr (replace z_n with Z_m + z_n)
        //   * n ← 0 (reset m to 0, car m = n dans notre implémentation)
        let z_curr_norm_sqr = z_curr.norm_sqr();
        // Recalculer delta_norm_sqr pour la vérification de rebasing
        let delta_norm_sqr_check = delta.norm_sqr_approx();
        
        // Improved rebasing check inspired by rust-fractal-core:
        // Use should_rebase() which adds hysteresis to prevent oscillating rebases
        // and avoids rebasing near reference orbit zeros.
        if should_rebase(z_curr_norm_sqr, delta_norm_sqr_check, z_ref_norm_sqr) {
            // Rebasing: replace z_n with Z_m + z_n and reset m to 0
            delta = ComplexExp::from_complex64(z_curr);  // replace z_n with Z_m + z_n
            
            // Hybrid BLA: change phase on rebasing: phase = (phase + n) % cycle_period
            // This must be done for all rebasing, not just in the BLA loop
            if let Some(ref mut phase) = current_phase {
                if let Some(refs) = hybrid_refs {
                    if refs.cycle_period > 0 {
                        **phase = (**phase + n) % refs.cycle_period;
                        phase_changed = true;
                    }
                }
            }
            n = 0;  // reset m to 0 (car m = n)
            // Reset BLA level hints after rebase
            last_nc_bla_level = if bla_table.nc_num_levels() > 0 { bla_table.nc_num_levels() - 1 } else { 0 };
            last_conf_bla_level = if bla_table.num_levels() > 0 { bla_table.num_levels() - 1 } else { 0 };
            continue;
        }

        if !z_curr.re.is_finite() || !z_curr.im.is_finite() {
            return DeltaResult {
                iteration: n,
                z_final: z_curr,
                glitched: true,
                suspect,
                distance: f64::INFINITY,
                is_interior: false,
                phase_changed,
            smooth_iteration: 0.0,
            };
        }
        if z_curr.norm_sqr() > bailout_sqr {
            return DeltaResult {
                iteration: n,
                z_final: z_curr,
                glitched: false,
                suspect,
                distance: f64::INFINITY, // Distance estimation not computed for escaped points
                is_interior: false,
                phase_changed,
                smooth_iteration: compute_smooth_iteration(n, z_curr, params.bailout, smooth_power),
            };
        }

        // Pauldelbrot glitch criterion: |δ|² > G² · |Z_ref|²
        // Use max(|Z_ref|², 1e-6) instead of |Z_ref|² + 1.0 for proper scaling.
        // The +1.0 made detection too lenient when |Z_ref| < 1.
        let glitch_scale = z_ref_norm_sqr.max(1e-6);
        if !delta_norm_sqr_check.is_finite() || delta_norm_sqr_check > glitch_tolerance_sqr * glitch_scale {
            return DeltaResult {
                iteration: n,
                z_final: z_curr,
                glitched: true,
                suspect,
                distance: f64::INFINITY,
                is_interior: false,
                phase_changed,
            smooth_iteration: 0.0,
            };
        }
    }

    let effective_len = ref_orbit.effective_len() as u32;
    let final_index = n.min(effective_len.saturating_sub(1));
    let z_curr = if use_high_precision && !is_burning_ship && !is_multibrot && !is_tricorn {
        let z_ref_hp = ref_orbit.get_z_ref(final_index).unwrap_or_else(|| {
            ref_orbit.z_ref[ref_orbit.z_ref.len().saturating_sub(1)]
        });
        z_ref_hp.add(delta).to_complex64_approx()
    } else {
        let z_ref = ref_orbit.get_z_ref_f64(final_index).unwrap_or_else(|| {
            ref_orbit.z_ref_f64[ref_orbit.z_ref_f64.len().saturating_sub(1)]
        });
        z_ref + delta.to_complex64_approx()
    };
    DeltaResult {
        iteration: final_index,
        z_final: z_curr,
        glitched: false,
        suspect,
        distance: f64::INFINITY, // Distance estimation not computed by default
        is_interior: false, // Interior detection not computed by default
        phase_changed,
    smooth_iteration: 0.0,
    }
}

/// Iterate a pixel using perturbation theory with full GMP precision.
/// This function is used for very deep zooms (>10^15) where f64/ComplexExp precision is insufficient.
///
/// # Arguments
///
/// * `params` - Paramètres de la fractale
/// * `ref_orbit` - Orbite de référence haute précision calculée au centre (avec z_ref_gmp)
/// * `dc_gmp` - Offset du pixel par rapport au centre en GMP (`c` dans la formule)
/// * `prec` - Précision GMP à utiliser
///
/// # Returns
///
/// DeltaResult avec le nombre d'itérations et la valeur finale de z
pub fn iterate_pixel_gmp(
    params: &FractalParams,
    ref_orbit: &ReferenceOrbit,
    dc_gmp: &Complex,
    prec: u32,
) -> DeltaResult {
    let mut n = 0u32;
    let effective_len = ref_orbit.effective_len() as u32;
    let max_iter = params.iteration_max.min(effective_len.saturating_sub(1));
    
    let bailout = Float::with_val(prec, params.bailout);
    let mut bailout_sqr = bailout.clone();
    bailout_sqr *= &bailout;
    
    // Initialisation selon le type de fractale
    // IMPORTANT: S'assurer que delta utilise la même précision que prec
    let mut delta = match params.fractal_type {
        FractalType::Julia => {
            // Julia: delta initial = dc (car z_0 = C + c pour Julia)
            // Créer une nouvelle valeur avec la précision explicite
            Complex::with_val(prec, (dc_gmp.real(), dc_gmp.imag()))
        }
        _ => {
            // Mandelbrot/BurningShip: delta initial = 0 (car z_0 = seed)
            Complex::with_val(prec, (0, 0))
        }
    };
    
    let is_julia = params.fractal_type == FractalType::Julia;
    let is_burning_ship = params.fractal_type == FractalType::BurningShip;
    let is_tricorn = params.fractal_type == FractalType::Tricorn;
    let smooth_power = if params.fractal_type == FractalType::Multibrot { params.multibrot_power } else { 2.0 };

    // Precompute glitch tolerance outside the loop to avoid repeated GMP allocations
    let pixel_size_gmp = params.span_x / params.width as f64;
    let adaptive_tolerance_gmp = compute_adaptive_glitch_tolerance(pixel_size_gmp, params.glitch_tolerance);
    let glitch_tolerance_sqr_gmp = Float::with_val(prec, adaptive_tolerance_gmp * adaptive_tolerance_gmp);
    let min_scale_gmp = Float::with_val(prec, 1e-6);

    // Main iteration loop with full GMP precision
    while n < max_iter {
        // Get reference point at iteration n
        let z_ref = match ref_orbit.get_z_ref_gmp(n) {
            Some(z) => z,
            None => break, // End of effective orbit
        };
        
        // Apply perturbation formula depending on fractal type.
        // Burning Ship and Tricorn have their own formulas and must NOT use the
        // standard Mandelbrot perturbation (which would corrupt delta before
        // their type-specific handling).
        if is_burning_ship {
            // Burning Ship: z' = (|Re(z)|, |Im(z)|)² + c
            // z_curr = z_ref + delta
            let mut z_curr = Complex::with_val(prec, z_ref);
            z_curr += &delta;

            // Apply abs() to real and imaginary parts
            let re_abs = z_curr.real().clone().abs();
            let im_abs = z_curr.imag().clone().abs();
            let z_abs_val = Complex::with_val(prec, (re_abs, im_abs));
            let mut z_next = z_abs_val.clone();
            z_next *= &z_abs_val;

            // Add cref + dc
            z_next += &ref_orbit.cref_gmp;
            if !is_julia {
                z_next += dc_gmp;
            }

            // Calculate delta for next iteration: z_next - z_ref_next
            if (n + 1) >= effective_len {
                delta = z_next;
                n = 0;
                continue;
            }

            let z_ref_next = match ref_orbit.get_z_ref_gmp(n + 1) {
                Some(z) => z,
                None => break,
            };
            delta = z_next - Complex::with_val(prec, z_ref_next);
        } else if is_tricorn {
            // Tricorn: z' = conj(z)² + c
            let mut z_curr = Complex::with_val(prec, z_ref);
            z_curr += &delta;
            let z_conj = z_curr.conj();
            let mut z_temp = z_conj.clone();
            z_temp *= &z_conj;
            z_temp += &ref_orbit.cref_gmp;
            if !is_julia {
                z_temp += dc_gmp;
            }

            if (n + 1) >= effective_len {
                delta = z_temp;
                n = 0;
                continue;
            }

            let z_ref_next = match ref_orbit.get_z_ref_gmp(n + 1) {
                Some(z) => z,
                None => break,
            };
            delta = z_temp - Complex::with_val(prec, z_ref_next);
        } else {
            // Standard Mandelbrot/Julia: delta_{n+1} = 2·z_ref·delta + delta² + dc
            // Horner form: delta * (2*z_ref + delta) + dc
            let mut sum = Complex::with_val(prec, z_ref);
            sum *= 2;
            sum += &delta;

            // Multiply by delta in-place
            let mut next_delta = delta.clone();
            next_delta *= &sum;

            // Add dc for Mandelbrot
            if !is_julia {
                next_delta += dc_gmp;
            }

            delta = next_delta;
        }
        
        // Advance iteration counter: delta now holds delta_{n+1}
        n += 1;

        // For Mandelbrot standard path, handle orbit end (BS/Tricorn already handled above).
        // Note: This is normally unreachable since max_iter <= effective_len - 1, but kept
        // as a defensive guard. If hit, rebase instead of breaking (matches f64 path behavior).
        if !is_burning_ship && !is_tricorn && n >= effective_len {
            // Can't compute z_curr without z_ref[n], so just break
            break;
        }

        // Check bailout using z_ref[n] (the NEW n, i.e. the next reference point)
        // IMPORTANT: After computing delta_{n+1}, the correct full z is z_ref[n+1] + delta_{n+1}
        let z_ref_next = match ref_orbit.get_z_ref_gmp(n) {
            Some(z) => z,
            None => break,
        };
        let mut z_curr = Complex::with_val(prec, z_ref_next);
        z_curr += &delta;
        let z_curr_norm_sqr = complex_norm_sqr(&z_curr, prec);

        if !z_curr.real().is_finite() || !z_curr.imag().is_finite() {
            return DeltaResult {
                iteration: n,
                z_final: crate::fractal::gmp::complex_to_complex64(&z_curr),
                glitched: true,
                suspect: false,
                distance: f64::INFINITY,
                is_interior: false,
                phase_changed: false,
            smooth_iteration: 0.0,
            };
        }

        if z_curr_norm_sqr > bailout_sqr {
            let z_final = crate::fractal::gmp::complex_to_complex64(&z_curr);
            return DeltaResult {
                iteration: n,
                z_final,
                glitched: false,
                suspect: false,
                distance: f64::INFINITY,
                is_interior: false,
                phase_changed: false,
                smooth_iteration: compute_smooth_iteration(n, z_final, params.bailout, smooth_power),
            };
        }

        // Check for rebasing: when |Z_m + z_n| < |z_n|
        let delta_norm_sqr = complex_norm_sqr(&delta, prec);
        if z_curr_norm_sqr.is_sign_positive()
            && delta_norm_sqr.is_sign_positive()
            && z_curr_norm_sqr < delta_norm_sqr {
            // Rebasing: replace z_n with Z_m + z_n and reset m to 0
            delta = z_curr;
            n = 0;
            continue;
        }

        // Check for glitch: delta is too large relative to z_ref at current iteration
        let z_ref_norm_sqr = complex_norm_sqr(z_ref_next, prec);
        // Pauldelbrot glitch criterion: |δ|² > G² · max(|Z_ref|², 1e-6)
        let glitch_scale = if z_ref_norm_sqr < min_scale_gmp { min_scale_gmp.clone() } else { z_ref_norm_sqr };
        let mut glitch_threshold = glitch_tolerance_sqr_gmp.clone();
        glitch_threshold *= &glitch_scale;

        // Check if delta_norm_sqr is too large (glitch detected)
        if !delta_norm_sqr.is_finite() || delta_norm_sqr > glitch_threshold {
            return DeltaResult {
                iteration: n,
                z_final: crate::fractal::gmp::complex_to_complex64(&z_curr),
                glitched: true,
                suspect: false,
                distance: f64::INFINITY,
                is_interior: false,
                phase_changed: false,
            smooth_iteration: 0.0,
            };
        }
    }
    
    // Final result
    // IMPORTANT: S'assurer que toutes les opérations utilisent la même précision prec
    let final_index = n.min(effective_len.saturating_sub(1));
    let z_ref = match ref_orbit.get_z_ref_gmp(final_index) {
        Some(z) => z,
        None => match ref_orbit.z_ref_gmp.last() {
            Some(z) => z,
            None => {
                // Vecteur vide - retourner un résultat glitch
                return DeltaResult {
                    iteration: 0,
                    z_final: Complex64::new(0.0, 0.0),
                    glitched: true,
                    suspect: false,
                    distance: f64::INFINITY,
                    is_interior: false,
                    phase_changed: false,
                smooth_iteration: 0.0,
                };
            }
        },
    };
    let z_ref_prec = Complex::with_val(prec, (z_ref.real(), z_ref.imag()));
    let delta_prec = Complex::with_val(prec, (delta.real(), delta.imag()));
    let mut z_curr = z_ref_prec.clone();
    z_curr += &delta_prec;
    
    // Final glitch check: verify delta is reasonable (reuse precomputed tolerance)
    let z_ref_norm_sqr = complex_norm_sqr(&z_ref_prec, prec);
    let delta_norm_sqr = complex_norm_sqr(&delta_prec, prec);
    // Pauldelbrot glitch criterion: |δ|² > G² · max(|Z_ref|², 1e-6)
    let glitch_scale = if z_ref_norm_sqr < min_scale_gmp { min_scale_gmp.clone() } else { z_ref_norm_sqr };
    let mut glitch_threshold = glitch_tolerance_sqr_gmp.clone();
    glitch_threshold *= &glitch_scale;
    let is_glitched = !delta_norm_sqr.is_finite() || delta_norm_sqr > glitch_threshold;
    
    DeltaResult {
        iteration: final_index,
        z_final: crate::fractal::gmp::complex_to_complex64(&z_curr),
        glitched: is_glitched,
        suspect: false,
        distance: f64::INFINITY,
        is_interior: false,
        phase_changed: false,
    smooth_iteration: 0.0,
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::definitions::default_params_for_type;
    use crate::fractal::{AlgorithmMode, FractalParams, FractalType};

    #[allow(dead_code)]
    fn test_params() -> FractalParams {
        let mut p = default_params_for_type(FractalType::Mandelbrot, 100, 100);
        p.span_x = 4.0;
        p.span_y = 4.0;
        p.iteration_max = 100;
        p.precision_bits = 192;
        p.algorithm_mode = AlgorithmMode::Perturbation;
        p.bla_threshold = 1e-6;
        p.glitch_neighbor_pass = false;
        p
    }

    #[test]
    fn delta_result_has_new_fields() {
        let result = DeltaResult {
            iteration: 10,
            z_final: Complex64::new(1.0, 2.0),
            glitched: false,
            suspect: false,
            distance: f64::INFINITY,
            is_interior: false,
            phase_changed: false,
        smooth_iteration: 0.0,
        };
        assert_eq!(result.iteration, 10);
        assert_eq!(result.distance, f64::INFINITY);
        assert_eq!(result.is_interior, false);
        assert_eq!(result.phase_changed, false);
    }

    #[test]
    fn diffabs_stable_computation() {
        // When c >= 0 and c + d >= 0: result = d
        assert_eq!(diffabs(5.0, 1.0), 1.0);
        // When c >= 0 and c + d < 0: result = -(2c + d)
        assert_eq!(diffabs(1.0, -3.0), -(-1.0)); // |1-3|-|1| = 2-1 = 1, but diffabs gives -(2+(-3))=1
        assert_eq!(diffabs(1.0, -3.0), 1.0);
        // When c < 0 and c + d > 0: result = 2c + d
        assert_eq!(diffabs(-1.0, 3.0), 1.0); // |-1+3|-|-1| = 2-1 = 1
        // When c < 0 and c + d <= 0: result = -d
        assert_eq!(diffabs(-5.0, 1.0), -1.0); // |-5+1|-|-5| = 4-5 = -1
    }

    #[test]
    fn pascal_coefficients_power_2() {
        let pascal = generate_pascal_coefficients(2);
        assert_eq!(pascal.len(), 3);
        assert!((pascal[0] - 1.0).abs() < 1e-12);
        assert!((pascal[1] - 2.0).abs() < 1e-12);
        assert!((pascal[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pascal_coefficients_power_3() {
        let pascal = generate_pascal_coefficients(3);
        assert_eq!(pascal.len(), 4);
        assert!((pascal[0] - 1.0).abs() < 1e-12); // C(3,0) = 1
        assert!((pascal[1] - 3.0).abs() < 1e-12); // C(3,1) = 3
        assert!((pascal[2] - 3.0).abs() < 1e-12); // C(3,2) = 3
        assert!((pascal[3] - 1.0).abs() < 1e-12); // C(3,3) = 1
    }

    #[test]
    fn pascal_coefficients_power_4() {
        let pascal = generate_pascal_coefficients(4);
        assert_eq!(pascal.len(), 5);
        assert!((pascal[0] - 1.0).abs() < 1e-12);
        assert!((pascal[1] - 4.0).abs() < 1e-12);
        assert!((pascal[2] - 6.0).abs() < 1e-12);
        assert!((pascal[3] - 4.0).abs() < 1e-12);
        assert!((pascal[4] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn adaptive_glitch_tolerance_scales_with_zoom() {
        // Shallow zoom: stricter tolerance
        let t_shallow = compute_adaptive_glitch_tolerance(1.0, 1e-4);
        // Deep zoom: more relaxed
        let t_deep = compute_adaptive_glitch_tolerance(1e-20, 1e-4);
        assert!(t_deep > t_shallow);
    }

    #[test]
    fn smooth_iteration_escaped_point() {
        // z = 3.0 + 0i, bailout = 2.0, power = 2.0 (standard Mandelbrot)
        let z = Complex64::new(3.0, 0.0);
        let smooth = compute_smooth_iteration(10, z, 2.0, 2.0);
        // Should be close to 10 but fractional (> 10 because |z| > bailout)
        assert!(smooth > 9.0 && smooth < 12.0, "smooth={}", smooth);
        assert!(smooth != 10.0, "Should be fractional, not integer");
    }

    #[test]
    fn smooth_iteration_non_escaped_point() {
        // z = 0.5 + 0i, bailout = 2.0 => |z| < bailout, not escaped
        let z = Complex64::new(0.5, 0.0);
        let smooth = compute_smooth_iteration(100, z, 2.0, 2.0);
        assert_eq!(smooth, 100.0);
    }

    #[test]
    fn smooth_iteration_large_z() {
        // Very large z (deeply escaped)
        let z = Complex64::new(1e10, 0.0);
        let smooth = compute_smooth_iteration(50, z, 2.0, 2.0);
        assert!(smooth < 50.0, "Large |z| should give smooth < iteration");
        assert!(smooth > 40.0, "Should be reasonable, smooth={}", smooth);
    }

    #[test]
    fn smooth_iteration_multibrot_power_3() {
        // Multibrot with power 3
        let z = Complex64::new(3.0, 0.0);
        let smooth_p2 = compute_smooth_iteration(10, z, 2.0, 2.0);
        let smooth_p3 = compute_smooth_iteration(10, z, 2.0, 3.0);
        // Higher power should give different smooth values
        assert!((smooth_p2 - smooth_p3).abs() > 0.01);
    }

}

