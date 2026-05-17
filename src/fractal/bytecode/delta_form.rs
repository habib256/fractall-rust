//! Interpréteur bytecode en **forme delta** pour la perturbation.
//!
//! Pour chaque opcode, propage à la fois la valeur absolue `Z` (reference)
//! et le delta `δ` (perturbation) selon la règle de la chaîne :
//! étant donné `z = Z + δ`, on calcule `op(z) - op(Z)` qui devient le
//! nouveau `δ_op`.
//!
//! Permet de traiter tous les types escape-time (Mandelbrot, Burning Ship,
//! Tricorn, Celtic, Buffalo, Perpendicular Burning Ship, Multibrot puissance
//! entière) via un seul interpréteur, vs un per-type hardcoded.
//!
//! Règles par opcode (Z scalaire complexe, δ scalaire complexe) :
//! - `Sqr`   : `δ' = 2·Z·δ + δ²`
//! - `Mul`   : `δ' = stored_Z·δ + Z·stored_δ + δ·stored_δ`
//! - `Store` : `stored_Z := Z`, `stored_δ := δ`
//! - `AbsX`  : `(Z+δ).re_abs - Z.re_abs = diffabs(Z.re, δ.re)`
//!             et `Z.re := |Z.re|`
//! - `AbsY`  : symétrique sur l'imaginaire
//! - `NegX`  : `δ.re := -δ.re`, `Z.re := -Z.re`
//! - `NegY`  : `δ.im := -δ.im`, `Z.im := -Z.im`
//! - `Add`   : `δ += dc`, `Z += c_ref`
//!
//! `diffabs(c, d) = |c + d| - |c|` calculé de manière stable (cf.
//! `delta.rs::diffabs`). Évite la cancellation catastrophique pour
//! BurningShip/Celtic etc. en perturbation.

use num_complex::Complex64;

use super::{Op, Phase};
use crate::fractal::perturbation::delta::diffabs;

/// État de l'interpréteur delta-form.
#[derive(Clone, Copy, Debug)]
pub struct DeltaState {
    /// Reference value Z (mutée par chaque opcode dans la phase).
    pub z_ref: Complex64,
    /// Delta δ (mutée par chaque opcode selon la règle de chaîne).
    pub delta: Complex64,
    /// `Store` snapshot du Z courant.
    pub stored_z: Complex64,
    /// `Store` snapshot du δ courant.
    pub stored_delta: Complex64,
}

impl DeltaState {
    pub fn new(z_ref: Complex64, delta: Complex64) -> Self {
        Self {
            z_ref,
            delta,
            stored_z: z_ref,
            stored_delta: delta,
        }
    }

    /// Applique une phase complète. À la fin, `z_ref` et `delta` représentent
    /// la valeur à la fin de cette itération.
    ///
    /// `c_ref` = constante ajoutée à la reference (cref pour Mandelbrot-like,
    /// seed pour Julia-like).
    /// `dc` = constante ajoutée au delta = c_pixel - c_ref.
    pub fn step(&mut self, phase: &Phase, c_ref: Complex64, dc: Complex64) {
        for op in &phase.ops {
            match op {
                Op::Sqr => {
                    // δ' = 2·Z·δ + δ²
                    let two_z = self.z_ref * 2.0;
                    let new_delta = two_z * self.delta + self.delta * self.delta;
                    // Z' = Z²
                    let new_z = self.z_ref * self.z_ref;
                    self.delta = new_delta;
                    self.z_ref = new_z;
                }
                Op::Mul => {
                    // δ' = stored_Z·δ + Z·stored_δ + δ·stored_δ
                    let new_delta = self.stored_z * self.delta
                        + self.z_ref * self.stored_delta
                        + self.delta * self.stored_delta;
                    // Z' = Z·stored_Z
                    let new_z = self.z_ref * self.stored_z;
                    self.delta = new_delta;
                    self.z_ref = new_z;
                }
                Op::Store => {
                    self.stored_z = self.z_ref;
                    self.stored_delta = self.delta;
                }
                Op::AbsX => {
                    // (Z+δ).re_abs - Z.re_abs = diffabs(Z.re, δ.re)
                    let new_delta_re = diffabs(self.z_ref.re, self.delta.re);
                    self.delta = Complex64::new(new_delta_re, self.delta.im);
                    self.z_ref = Complex64::new(self.z_ref.re.abs(), self.z_ref.im);
                }
                Op::AbsY => {
                    let new_delta_im = diffabs(self.z_ref.im, self.delta.im);
                    self.delta = Complex64::new(self.delta.re, new_delta_im);
                    self.z_ref = Complex64::new(self.z_ref.re, self.z_ref.im.abs());
                }
                Op::NegX => {
                    self.delta = Complex64::new(-self.delta.re, self.delta.im);
                    self.z_ref = Complex64::new(-self.z_ref.re, self.z_ref.im);
                }
                Op::NegY => {
                    self.delta = Complex64::new(self.delta.re, -self.delta.im);
                    self.z_ref = Complex64::new(self.z_ref.re, -self.z_ref.im);
                }
                Op::Add => {
                    self.delta += dc;
                    self.z_ref += c_ref;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::bytecode::compile_formula;
    use crate::fractal::FractalType;

    /// La somme `Z + δ` à la fin doit toujours égaler l'itération complète
    /// `op(z, c_pixel)` (mod arrondi). C'est l'invariant fondamental de la
    /// perturbation.
    fn assert_invariant_after_step(
        ft: FractalType,
        multibrot_power: f64,
        z_ref_initial: Complex64,
        delta_initial: Complex64,
        c_ref: Complex64,
        dc: Complex64,
        tol: f64,
    ) {
        let formula = compile_formula(ft, multibrot_power).unwrap();
        let phase = &formula.phases[0];

        // Path delta-form : applique la phase à (Z, δ).
        let mut state = DeltaState::new(z_ref_initial, delta_initial);
        state.step(phase, c_ref, dc);
        let z_plus_delta_after = state.z_ref + state.delta;

        // Path absolu : applique la phase directement sur z = Z + δ avec c = c_ref + dc.
        let c_pixel = c_ref + dc;
        let mut z_abs = z_ref_initial + delta_initial;
        let mut stored = z_abs;
        for op in &phase.ops {
            match op {
                Op::Sqr => z_abs = z_abs * z_abs,
                Op::Mul => z_abs = z_abs * stored,
                Op::Store => stored = z_abs,
                Op::AbsX => z_abs = Complex64::new(z_abs.re.abs(), z_abs.im),
                Op::AbsY => z_abs = Complex64::new(z_abs.re, z_abs.im.abs()),
                Op::NegX => z_abs = Complex64::new(-z_abs.re, z_abs.im),
                Op::NegY => z_abs = Complex64::new(z_abs.re, -z_abs.im),
                Op::Add => z_abs += c_pixel,
            }
        }

        let diff = (z_plus_delta_after - z_abs).norm();
        assert!(
            diff < tol,
            "Invariant Z+δ violé pour {:?}: Z+δ={:?}, z_abs={:?}, diff={}",
            ft,
            z_plus_delta_after,
            z_abs,
            diff
        );
    }

    #[test]
    fn invariant_mandelbrot() {
        // Mandelbrot : Z=0.3+0.4i, δ=0.001-0.002i, c_ref=-0.5, dc=0.001+0.001i.
        assert_invariant_after_step(
            FractalType::Mandelbrot,
            2.0,
            Complex64::new(0.3, 0.4),
            Complex64::new(0.001, -0.002),
            Complex64::new(-0.5, 0.0),
            Complex64::new(0.001, 0.001),
            1e-14,
        );
    }

    #[test]
    fn invariant_burning_ship_q1() {
        // BS premier quadrant : Z et Z+δ tous deux dans Re>0, Im>0.
        assert_invariant_after_step(
            FractalType::BurningShip,
            2.0,
            Complex64::new(0.5, 0.3),
            Complex64::new(0.001, -0.001),
            Complex64::new(-1.7, -0.1),
            Complex64::new(0.001, 0.001),
            1e-14,
        );
    }

    #[test]
    fn invariant_burning_ship_q2_crossing() {
        // BS : Z dans Q1 (Re>0), Z+δ dans Q2 (Re<0) → diffabs gère le flip.
        assert_invariant_after_step(
            FractalType::BurningShip,
            2.0,
            Complex64::new(0.001, 0.5),
            Complex64::new(-0.01, 0.001),
            Complex64::new(-1.7, -0.1),
            Complex64::new(0.001, 0.001),
            1e-12,
        );
    }

    #[test]
    fn invariant_tricorn() {
        assert_invariant_after_step(
            FractalType::Tricorn,
            2.0,
            Complex64::new(0.3, -0.2),
            Complex64::new(0.001, 0.001),
            Complex64::new(-0.5, 0.5),
            Complex64::new(0.001, 0.001),
            1e-14,
        );
    }

    #[test]
    fn invariant_celtic_positive_re() {
        // Re(z²) doit être positif pour rester du même côté du flip AbsX.
        assert_invariant_after_step(
            FractalType::Celtic,
            2.0,
            Complex64::new(1.0, 0.5),
            Complex64::new(0.001, -0.001),
            Complex64::new(-0.5, 0.1),
            Complex64::new(0.001, 0.001),
            1e-12,
        );
    }

    #[test]
    fn invariant_buffalo() {
        assert_invariant_after_step(
            FractalType::Buffalo,
            2.0,
            Complex64::new(0.7, 0.4),
            Complex64::new(0.001, 0.002),
            Complex64::new(-0.3, 0.0),
            Complex64::new(0.001, 0.001),
            1e-12,
        );
    }

    #[test]
    fn invariant_perpendicular_burning_ship() {
        assert_invariant_after_step(
            FractalType::PerpendicularBurningShip,
            2.0,
            Complex64::new(0.4, -0.3),
            Complex64::new(0.001, 0.001),
            Complex64::new(-0.7, 0.0),
            Complex64::new(0.001, 0.001),
            1e-12,
        );
    }

    #[test]
    fn invariant_multibrot_pow3() {
        assert_invariant_after_step(
            FractalType::Multibrot,
            3.0,
            Complex64::new(0.3, 0.4),
            Complex64::new(0.001, -0.002),
            Complex64::new(-0.5, 0.0),
            Complex64::new(0.001, 0.001),
            1e-13,
        );
    }

    #[test]
    fn invariant_multibrot_pow5() {
        // pow 5 = Store, Sqr, Sqr, Mul, Add — exerce Mul/Store en delta-form.
        assert_invariant_after_step(
            FractalType::Multibrot,
            5.0,
            Complex64::new(0.2, 0.3),
            Complex64::new(0.0001, 0.0001),
            Complex64::new(-0.5, 0.0),
            Complex64::new(0.0001, 0.0001),
            1e-12,
        );
    }
}
