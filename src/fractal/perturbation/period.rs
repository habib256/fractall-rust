//! Period detection for Mandelbrot-family fractals.
//!
//! Inspired by rust-fractal-core's `root_finding.rs`:
//! - **Box period detection**: Tracks 4 corner points of a bounding box through
//!   perturbation iterations. When the box surrounds the origin, a periodic cycle
//!   has been found. This is the fastest method for detecting periods.
//! - **Atom domain period detection**: Finds the iteration where `|Z_n|` is minimized,
//!   which corresponds to the atom domain period. Simpler but less robust than box method.
//!
//! Period detection is useful for:
//! - Automatic iteration limit adjustment (set max_iter >= period * 2)
//! - Nucleus finding (Newton's method converges faster with known period)
//! - Interior detection (points inside minibrots have periodic orbits)

use num_complex::Complex64;

use crate::fractal::perturbation::orbit::ReferenceOrbit;
use crate::fractal::perturbation::types::ComplexExp;

/// Box-based period detection.
///
/// Tracks 4 corner points of a bounding box around the center through perturbation
/// iterations. When the iterated box surrounds the origin (winding number test),
/// a periodic orbit has been detected.
///
/// Inspired by rust-fractal-core's `BoxPeriod` struct and `find_period()` method.
pub struct BoxPeriod {
    /// The 4 corner deltas (relative to center)
    pub corners_z: [ComplexExp; 4],
    /// The initial corner positions (for perturbation dc values)
    pub corners_c: [ComplexExp; 4],
    /// Detected period (0 if not yet detected)
    pub period: usize,
}

impl BoxPeriod {
    /// Create a new box period detector from 4 corner deltas.
    ///
    /// The corners should be arranged as a box around the center point:
    /// corner[0] = top-left, corner[1] = top-right,
    /// corner[2] = bottom-left, corner[3] = bottom-right
    pub fn new(corners: [ComplexExp; 4]) -> Self {
        BoxPeriod {
            corners_z: corners,
            corners_c: corners,
            period: 0,
        }
    }

    /// Create a box period detector from image parameters.
    ///
    /// Uses the 4 corners of the image as the bounding box.
    pub fn from_image_bounds(delta_pixel: f64, width: usize, height: usize) -> Self {
        let half_w = width as f64 * 0.5 * delta_pixel;
        let half_h = height as f64 * 0.5 * delta_pixel;

        let corners = [
            ComplexExp::from_complex64(Complex64::new(-half_w, -half_h)),
            ComplexExp::from_complex64(Complex64::new(half_w, -half_h)),
            ComplexExp::from_complex64(Complex64::new(-half_w, half_h)),
            ComplexExp::from_complex64(Complex64::new(half_w, half_h)),
        ];

        Self::new(corners)
    }

    /// Check if a line segment from `a` to `b` crosses the positive real axis
    /// from below to above (contributing +1 to winding number).
    ///
    /// Inspired by rust-fractal-core's `crosses_origin()`.
    fn crosses_origin(a: Complex64, b: Complex64) -> usize {
        if a.im.signum() as i32 != b.im.signum() as i32 {
            let d = b - a;
            let s = d.im.signum() as i32;
            let t = (d.im * a.re - d.re * a.im).signum() as i32;
            (s == t) as usize
        } else {
            0
        }
    }

    /// Check if the 4 iterated corner points surround the origin.
    ///
    /// Uses the winding number test: count how many edges of the quadrilateral
    /// cross the positive real axis. If the count is odd, the origin is inside.
    ///
    /// The points are offset by `z_ref` (the reference orbit value at this iteration)
    /// to get their absolute positions.
    fn points_surround_origin(&self, z_ref: Complex64) -> bool {
        let a = self.corners_z[0].to_complex64_approx() + z_ref;
        let b = self.corners_z[1].to_complex64_approx() + z_ref;
        let c = self.corners_z[2].to_complex64_approx() + z_ref;
        let d = self.corners_z[3].to_complex64_approx() + z_ref;

        let crossings = Self::crosses_origin(a, b)
            + Self::crosses_origin(b, c)
            + Self::crosses_origin(c, d)
            + Self::crosses_origin(d, a);

        crossings & 1 == 1
    }

    /// Find the period by iterating the box corners using perturbation.
    ///
    /// Inspired by rust-fractal-core's `BoxPeriod::find_period()`.
    ///
    /// At each iteration, the 4 corner deltas are advanced using the Mandelbrot
    /// perturbation formula: `delta' = 2 * Z_ref * delta + delta^2 + dc`.
    /// When the iterated box surrounds the origin, the period has been found.
    ///
    /// Returns the detected period, or 0 if none found within the reference orbit length.
    pub fn find_period(&mut self, ref_orbit: &ReferenceOrbit) -> usize {
        self.period = 0;
        let orbit_len = ref_orbit.z_ref_f64.len();

        for n in 1..orbit_len {
            let z_ref = ref_orbit.z_ref_f64[n - 1];

            // Check if box surrounds origin at this iteration
            if self.points_surround_origin(ref_orbit.z_ref_f64[n]) {
                self.period = n;
                return n;
            }

            // Advance all 4 corners using perturbation: delta' = 2*Z*delta + delta^2 + dc
            for i in 0..4 {
                let delta = self.corners_z[i].to_complex64_approx();
                let dc = self.corners_c[i].to_complex64_approx();
                let new_delta = 2.0 * z_ref * delta + delta * delta + dc;
                self.corners_z[i] = ComplexExp::from_complex64(new_delta);

                // Periodic reduce to prevent precision loss (every 250 iterations)
                if n % 250 == 0 {
                    self.corners_z[i].reduce();
                }
            }
        }

        0
    }
}

/// Atom domain period detection.
///
/// Finds the iteration `n` where `|Z_ref[n]|` is minimized. This iteration
/// corresponds to the atom domain period - the orbit comes closest to the
/// critical point, indicating the period of the nearest minibrot.
///
/// Inspired by rust-fractal-core's `find_atom_domain_period()`.
///
/// This is simpler than box period detection but less robust:
/// - Works well when the center is close to a minibrot nucleus
/// - May give incorrect results for points far from any nucleus
/// - Does not require corner points, only the reference orbit
///
/// # Arguments
/// * `ref_orbit` - Reference orbit computed at the center
///
/// # Returns
/// The detected period (iteration of minimum |Z_n|), or 0 if none found.
pub fn find_atom_domain_period(ref_orbit: &ReferenceOrbit) -> usize {
    let orbit_len = ref_orbit.z_ref_f64.len();
    if orbit_len < 2 {
        return 0;
    }

    let mut min_norm_sqr = f64::MAX;
    let mut min_at = 0;

    for n in 1..orbit_len {
        let z = ref_orbit.z_ref_f64[n];
        let norm_sqr = z.norm_sqr();

        // Skip escaped points
        if norm_sqr > 1e16 {
            break;
        }

        if norm_sqr < min_norm_sqr {
            min_norm_sqr = norm_sqr;
            min_at = n;
        }
    }

    min_at
}

/// Suggest maximum iteration count based on detected period.
///
/// When a period is detected, the maximum iteration count should be at least
/// 2x the period to ensure we can properly resolve the minibrot structure.
///
/// # Arguments
/// * `current_max_iter` - Current maximum iteration count
/// * `period` - Detected period (from box method or atom domain)
///
/// # Returns
/// Suggested maximum iteration count (may be higher than current)
pub fn suggest_iterations_from_period(current_max_iter: u32, period: usize) -> u32 {
    if period == 0 {
        return current_max_iter;
    }

    // Ensure at least 2x period, with a minimum margin of 500
    let suggested = (period as u32)
        .saturating_mul(2)
        .max(period as u32 + 500);

    current_max_iter.max(suggested)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn crosses_origin_basic() {
        // Segment crossing positive real axis from below to above
        let a = Complex64::new(0.5, -1.0);
        let b = Complex64::new(0.5, 1.0);
        assert_eq!(BoxPeriod::crosses_origin(a, b), 1);

        // Segment not crossing
        let a = Complex64::new(0.5, 1.0);
        let b = Complex64::new(0.5, 2.0);
        assert_eq!(BoxPeriod::crosses_origin(a, b), 0);
    }

    #[test]
    fn atom_domain_basic() {
        // Simple orbit that approaches origin at iteration 3
        let ref_orbit = ReferenceOrbit {
            cref: Complex64::new(-1.0, 0.0),
            z_ref: vec![
                ComplexExp::from_complex64(Complex64::new(0.0, 0.0)),
                ComplexExp::from_complex64(Complex64::new(-1.0, 0.0)),
                ComplexExp::from_complex64(Complex64::new(0.0, 0.0)),
                ComplexExp::from_complex64(Complex64::new(-1.0, 0.0)),
            ],
            z_ref_f64: vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
            z_ref_gmp: Vec::new(),
            cref_gmp: rug::Complex::new(53),
            phase_offset: 0,
            extended_iterations: Vec::new(),
            high_precision_data: Vec::new(),
            data_storage_interval: 1,
        };

        let period = find_atom_domain_period(&ref_orbit);
        // z_ref[2] = (0,0) has smallest norm, but z_ref[0] is also 0
        // z_ref[1] = (-1,0) has norm 1
        // Minimum is at iteration 1 (z_ref[0] is skipped because we start at n=1)
        assert!(period > 0);
    }

    #[test]
    fn suggest_iterations_basic() {
        assert_eq!(suggest_iterations_from_period(1000, 0), 1000);
        assert!(suggest_iterations_from_period(500, 1000) >= 2000);
        assert_eq!(suggest_iterations_from_period(5000, 200), 5000);
    }

    #[test]
    fn box_period_from_image() {
        let bp = BoxPeriod::from_image_bounds(0.01, 100, 100);
        assert_eq!(bp.period, 0);
        // Check corners are symmetric
        let c0 = bp.corners_c[0].to_complex64_approx();
        let c3 = bp.corners_c[3].to_complex64_approx();
        assert!((c0.re + c3.re).abs() < 1e-10);
        assert!((c0.im + c3.im).abs() < 1e-10);
    }
}
