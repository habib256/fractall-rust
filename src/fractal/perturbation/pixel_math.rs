//! Calculs scalaires partagés par les moteurs de perturbation par pixel.

use num_complex::Complex64;

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
pub fn compute_smooth_iteration(
    iteration: u32,
    z_final: Complex64,
    bailout: f64,
    power: f64,
) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::*;

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
