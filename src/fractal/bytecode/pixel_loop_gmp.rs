//! Itération GMP par pixel du moteur bytecode.

use num_complex::Complex64;
use rug::{Complex, Float};

use crate::fractal::gmp::complex_norm_sqr;
use crate::fractal::perturbation::counter::PixelCounter;
use crate::fractal::perturbation::orbit::ReferenceOrbit;
use crate::fractal::perturbation::pixel_math::{
    compute_adaptive_glitch_tolerance, compute_smooth_iteration,
};
use crate::fractal::perturbation::types::DeltaResult;
use crate::fractal::{FractalParams, FractalType};

/// Contrat d'un pixel de correction GMP. Contrairement à `PixelLoopLimits`,
/// il ne porte aucun cap d'accélération : ce chemin doit produire l'oracle
/// exact jusqu'à `params.iteration_max`.
pub struct GmpPixelRequest<'a> {
    pub params: &'a FractalParams,
    pub ref_orbit: &'a ReferenceOrbit,
    pub dc: &'a Complex,
    pub precision: u32,
}

/// Iterate a pixel using perturbation theory with full GMP precision.
/// This function is used for very deep zooms (>10^15) where f64/ComplexExp precision is insufficient.
///
/// # Arguments
///
/// La requête nommée porte paramètres, orbite de référence, offset GMP et
/// précision de calcul.
///
/// # Returns
///
/// DeltaResult avec le nombre d'itérations et la valeur finale de z
pub fn iterate_pixel_gmp(request: GmpPixelRequest<'_>) -> DeltaResult {
    let GmpPixelRequest {
        params,
        ref_orbit,
        dc: dc_gmp,
        precision: prec,
    } = request;
    // Compteur COMMUN (G5 `PixelCounter`) : `n` = itération ABSOLUE (compte
    // renvoyé, borne de boucle), `m` = index dans l'orbite de référence
    // (remis à 0 au rebase). Avant 2026-08-23 un seul compteur servait aux
    // deux : chaque rebase remettait la borne de boucle à 0 → pixel intérieur
    // qui rebase = boucle INFINIE (hang observé), et compte faux après
    // rebase. Le type rend la classe impossible (rebase ne touche pas n).
    let mut c = PixelCounter::new();
    let effective_len = ref_orbit.effective_len() as u32;
    let max_m = effective_len.saturating_sub(1);

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
    let smooth_power = if params.fractal_type == FractalType::Multibrot {
        params.multibrot_power
    } else {
        2.0
    };

    // Precompute glitch tolerance outside the loop to avoid repeated GMP allocations
    let pixel_size_gmp = params.span_x / params.width as f64;
    let adaptive_tolerance_gmp =
        compute_adaptive_glitch_tolerance(pixel_size_gmp, params.glitch_tolerance);
    let glitch_tolerance_sqr_gmp =
        Float::with_val(prec, adaptive_tolerance_gmp * adaptive_tolerance_gmp);
    let min_scale_gmp = Float::with_val(prec, 1e-6);

    // Main iteration loop with full GMP precision
    while c.keep_iterating(params.iteration_max) && c.m() < max_m {
        // Get reference point at iteration m
        let z_ref = match ref_orbit.get_z_ref_gmp(c.m()) {
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
            if (c.m() + 1) >= effective_len {
                delta = z_next;
                c.step();
                c.rebase();
                continue;
            }

            let z_ref_next = match ref_orbit.get_z_ref_gmp(c.m() + 1) {
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

            if (c.m() + 1) >= effective_len {
                delta = z_temp;
                c.step();
                c.rebase();
                continue;
            }

            let z_ref_next = match ref_orbit.get_z_ref_gmp(c.m() + 1) {
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

        // Advance iteration counters: delta now holds delta_{n+1}
        c.step();

        // For Mandelbrot standard path, handle orbit end (BS/Tricorn already handled above).
        // Note: This is normally unreachable since max_iter <= effective_len - 1, but kept
        // as a defensive guard. If hit, rebase instead of breaking (matches f64 path behavior).
        if !is_burning_ship && !is_tricorn && c.m() >= effective_len {
            // Can't compute z_curr without z_ref[n], so just break
            break;
        }

        // Check bailout using z_ref[n] (the NEW n, i.e. the next reference point)
        // IMPORTANT: After computing delta_{n+1}, the correct full z is z_ref[n+1] + delta_{n+1}
        let z_ref_next = match ref_orbit.get_z_ref_gmp(c.m()) {
            Some(z) => z,
            None => break,
        };
        let mut z_curr = Complex::with_val(prec, z_ref_next);
        z_curr += &delta;
        let z_curr_norm_sqr = complex_norm_sqr(&z_curr, prec);

        if !z_curr.real().is_finite() || !z_curr.imag().is_finite() {
            return DeltaResult {
                iteration: c.n(),
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
                iteration: c.n(),
                z_final,
                glitched: false,
                suspect: false,
                distance: f64::INFINITY,
                is_interior: false,
                phase_changed: false,
                smooth_iteration: compute_smooth_iteration(
                    c.n(),
                    z_final,
                    params.bailout,
                    smooth_power,
                ),
            };
        }

        // Check for rebasing: when |Z_m + z_n| < |z_n|
        let delta_norm_sqr = complex_norm_sqr(&delta, prec);
        if z_curr_norm_sqr.is_sign_positive()
            && delta_norm_sqr.is_sign_positive()
            && z_curr_norm_sqr < delta_norm_sqr
        {
            // Rebasing: replace z_n with Z_m + z_n and reset m to 0
            delta = z_curr;
            c.rebase();
            continue;
        }

        // Check for glitch: delta is too large relative to z_ref at current iteration
        let z_ref_norm_sqr = complex_norm_sqr(z_ref_next, prec);
        // Pauldelbrot glitch criterion: |δ|² > G² · max(|Z_ref|², 1e-6)
        let glitch_scale = if z_ref_norm_sqr < min_scale_gmp {
            min_scale_gmp.clone()
        } else {
            z_ref_norm_sqr
        };
        let mut glitch_threshold = glitch_tolerance_sqr_gmp.clone();
        glitch_threshold *= &glitch_scale;

        // Check if delta_norm_sqr is too large (glitch detected)
        if !delta_norm_sqr.is_finite() || delta_norm_sqr > glitch_threshold {
            return DeltaResult {
                iteration: c.n(),
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
    let final_index = c.m().min(effective_len.saturating_sub(1));
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
    let glitch_scale = if z_ref_norm_sqr < min_scale_gmp {
        min_scale_gmp.clone()
    } else {
        z_ref_norm_sqr
    };
    let mut glitch_threshold = glitch_tolerance_sqr_gmp.clone();
    glitch_threshold *= &glitch_scale;
    let is_glitched = !delta_norm_sqr.is_finite() || delta_norm_sqr > glitch_threshold;

    // Continuation via pure per-pixel GMP iteration when the reference orbit
    // was exhausted before the user's iteration cap (centers outside the M-set
    // produce short non-periodic reference orbits). Without this, every pixel
    // that outlives the reference inherits z_ref[effective_len-1] + delta,
    // yielding identical (iter, z) for spatially distinct pixels.
    let ref_exhausted = c.n() < params.iteration_max && c.m() >= max_m;
    let z_curr_norm_sqr = complex_norm_sqr(&z_curr, prec);
    if ref_exhausted && !is_glitched && z_curr_norm_sqr < bailout_sqr {
        let c_mandel = {
            let mut c = Complex::with_val(prec, &ref_orbit.cref_gmp);
            c += dc_gmp;
            c
        };
        let seed_complex = Complex::with_val(prec, (params.seed.re, params.seed.im));
        let multibrot_power = params.multibrot_power;
        while c.keep_iterating(params.iteration_max) {
            let z_new = if is_burning_ship {
                let re_abs = z_curr.real().clone().abs();
                let im_abs = z_curr.imag().clone().abs();
                let z_abs = Complex::with_val(prec, (re_abs, im_abs));
                let mut zn = z_abs.clone();
                zn *= &z_abs;
                zn += &c_mandel;
                zn
            } else if is_tricorn {
                let z_conj = z_curr.clone().conj();
                let mut zn = z_conj.clone();
                zn *= &z_conj;
                zn += &c_mandel;
                zn
            } else if params.fractal_type == FractalType::Multibrot {
                let mut zn = crate::fractal::gmp::pow_f64_mpc(&z_curr, multibrot_power, prec);
                zn += &c_mandel;
                zn
            } else {
                // Mandelbrot / Julia: z_new = z² + c
                let mut zn = z_curr.clone();
                zn *= &z_curr;
                if is_julia {
                    zn += &seed_complex;
                } else {
                    zn += &c_mandel;
                }
                zn
            };
            z_curr = z_new;
            c.step();
            if !z_curr.real().is_finite() || !z_curr.imag().is_finite() {
                break;
            }
            let zn2 = complex_norm_sqr(&z_curr, prec);
            if zn2 > bailout_sqr {
                break;
            }
        }
        return DeltaResult {
            iteration: c.n(),
            z_final: crate::fractal::gmp::complex_to_complex64(&z_curr),
            glitched: false,
            suspect: false,
            distance: f64::INFINITY,
            is_interior: false,
            phase_changed: false,
            smooth_iteration: 0.0,
        };
    }

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
