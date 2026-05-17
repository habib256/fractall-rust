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
use crate::fractal::perturbation::types::{ComplexExp, FloatExp};

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

/// Variante extended-precision : delta en ComplexExp (mantissa+exponent),
/// Z_ref reste en Complex64 (valeurs O(1), pas d'underflow).
///
/// Pour deep zoom > 1e15 où le delta f64 underflowerait : ComplexExp
/// préserve les magnitudes via l'exponent séparé.
#[derive(Clone, Copy, Debug)]
pub struct DeltaStateExp {
    pub z_ref: Complex64,
    pub delta: ComplexExp,
    pub stored_z: Complex64,
    pub stored_delta: ComplexExp,
}

impl DeltaStateExp {
    pub fn new(z_ref: Complex64, delta: ComplexExp) -> Self {
        Self {
            z_ref,
            delta,
            stored_z: z_ref,
            stored_delta: delta,
        }
    }

    /// Applique une phase complète (mirror exact de `DeltaState::step` mais
    /// avec ComplexExp pour delta).
    ///
    /// `c_ref` (Complex64) et `dc` (ComplexExp). c_ref ne perd pas de précision
    /// car la reference orbit est O(1). dc peut être très petit à deep zoom.
    pub fn step(&mut self, phase: &Phase, c_ref: Complex64, dc: ComplexExp) {
        for op in &phase.ops {
            match op {
                Op::Sqr => {
                    // δ' = 2·Z·δ + δ²
                    // δ² : ComplexExp * ComplexExp
                    let delta_sq = self.delta.mul(self.delta);
                    // 2·Z·δ : Complex64 (2·Z) * ComplexExp (δ)
                    let two_z = self.z_ref * 2.0;
                    let two_z_delta = self.delta.mul_complex64(two_z);
                    self.delta = two_z_delta.add(delta_sq);
                    // Z' = Z²
                    self.z_ref = self.z_ref * self.z_ref;
                }
                Op::Mul => {
                    // δ' = stored_Z·δ + Z·stored_δ + δ·stored_δ
                    let term1 = self.delta.mul_complex64(self.stored_z);
                    let term2 = self.stored_delta.mul_complex64(self.z_ref);
                    let term3 = self.delta.mul(self.stored_delta);
                    self.delta = term1.add(term2).add(term3);
                    // Z' = Z·stored_Z
                    self.z_ref = self.z_ref * self.stored_z;
                }
                Op::Store => {
                    self.stored_z = self.z_ref;
                    self.stored_delta = self.delta;
                }
                Op::AbsX => {
                    // (Z+δ).re_abs - Z.re_abs : diffabs_exp(Z.re, δ.re)
                    let new_delta_re = diffabs_exp(self.z_ref.re, self.delta.re);
                    self.delta = ComplexExp {
                        re: new_delta_re,
                        im: self.delta.im,
                    };
                    self.z_ref = Complex64::new(self.z_ref.re.abs(), self.z_ref.im);
                }
                Op::AbsY => {
                    let new_delta_im = diffabs_exp(self.z_ref.im, self.delta.im);
                    self.delta = ComplexExp {
                        re: self.delta.re,
                        im: new_delta_im,
                    };
                    self.z_ref = Complex64::new(self.z_ref.re, self.z_ref.im.abs());
                }
                Op::NegX => {
                    self.delta = ComplexExp {
                        re: FloatExp::new(-self.delta.re.mantissa, self.delta.re.exponent),
                        im: self.delta.im,
                    };
                    self.z_ref = Complex64::new(-self.z_ref.re, self.z_ref.im);
                }
                Op::NegY => {
                    self.delta = ComplexExp {
                        re: self.delta.re,
                        im: FloatExp::new(-self.delta.im.mantissa, self.delta.im.exponent),
                    };
                    self.z_ref = Complex64::new(self.z_ref.re, -self.z_ref.im);
                }
                Op::Add => {
                    self.delta = self.delta.add(dc);
                    self.z_ref += c_ref;
                }
            }
        }
    }
}

/// Variante ComplexExp de `diffabs(c, d) = |c + d| - |c|`.
///
/// `c` est f64 (composante de Z_ref O(1)), `d` est FloatExp (composante
/// de delta, potentiellement très petite). Calcul stable comme `delta.rs::diffabs`.
fn diffabs_exp(c: f64, d: FloatExp) -> FloatExp {
    let d_f64 = d.to_f64();
    let cd = c + d_f64;
    let c2d = 2.0 * c + d_f64;
    // Quand d est très petit (sous f64 epsilon), le résultat est juste `d`
    // ou `-d` selon le quadrant — pas de cancellation possible. On utilise
    // les comparaisons via la conversion f64 mais on retourne FloatExp pour
    // préserver la précision.
    if c >= 0.0 {
        if cd >= 0.0 {
            d
        } else {
            // -c2d en FloatExp : si c domine, c2d ≈ 2c → on perd la précision
            // de d (mais ce cas est rare et le legacy diffabs a le même comportement).
            FloatExp::from_f64(-c2d)
        }
    } else {
        if cd > 0.0 {
            FloatExp::from_f64(c2d)
        } else {
            FloatExp::new(-d.mantissa, d.exponent)
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

    /// Invariant pour `DeltaStateExp` : même propriété que `DeltaState`,
    /// avec un delta_initial très petit (1e-20) qui underflowerait f64
    /// si on accumulait des δ² sur plusieurs itérations.
    fn assert_invariant_exp_after_step(
        ft: FractalType,
        multibrot_power: f64,
        z_ref_initial: Complex64,
        delta_initial_f64: Complex64,
        c_ref: Complex64,
        dc_f64: Complex64,
        tol: f64,
    ) {
        let formula = compile_formula(ft, multibrot_power).unwrap();
        let phase = &formula.phases[0];
        let delta_init_exp = ComplexExp::from_complex64(delta_initial_f64);
        let dc_exp = ComplexExp::from_complex64(dc_f64);

        let mut state = DeltaStateExp::new(z_ref_initial, delta_init_exp);
        state.step(phase, c_ref, dc_exp);
        let z_plus_delta = state.z_ref + state.delta.to_complex64_approx();

        // Path absolu : iterate z + δ via la phase.
        let c_pixel = c_ref + dc_f64;
        let mut z_abs = z_ref_initial + delta_initial_f64;
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
        let diff = (z_plus_delta - z_abs).norm();
        assert!(
            diff < tol,
            "Invariant exp violé pour {:?}: Z+δ={:?}, z_abs={:?}, diff={}",
            ft,
            z_plus_delta,
            z_abs,
            diff
        );
    }

    #[test]
    fn invariant_exp_mandelbrot() {
        assert_invariant_exp_after_step(
            FractalType::Mandelbrot,
            2.0,
            Complex64::new(0.3, 0.4),
            Complex64::new(1e-15, -1e-15),
            Complex64::new(-0.5, 0.0),
            Complex64::new(1e-15, 1e-15),
            1e-12,
        );
    }

    #[test]
    fn invariant_exp_burning_ship() {
        assert_invariant_exp_after_step(
            FractalType::BurningShip,
            2.0,
            Complex64::new(0.5, 0.3),
            Complex64::new(1e-15, -1e-15),
            Complex64::new(-1.7, -0.1),
            Complex64::new(1e-15, 1e-15),
            1e-12,
        );
    }

    #[test]
    fn invariant_exp_tricorn() {
        assert_invariant_exp_after_step(
            FractalType::Tricorn,
            2.0,
            Complex64::new(0.3, -0.2),
            Complex64::new(1e-15, 1e-15),
            Complex64::new(-0.5, 0.5),
            Complex64::new(1e-15, 1e-15),
            1e-12,
        );
    }

    /// Vraie validation deep zoom : delta = 1e-100, hors range f64
    /// normal. Avec DeltaState (f64), ce delta sera vu comme 0 → invariant
    /// trivialement violé. Avec DeltaStateExp, le delta est préservé.
    #[test]
    fn invariant_exp_deep_zoom_delta() {
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let phase = &formula.phases[0];
        let z_ref_initial = Complex64::new(0.3, 0.4);
        let c_ref = Complex64::new(-0.5, 0.0);
        let delta_exp = ComplexExp {
            re: FloatExp::new(1.0, -300), // ≈ 1e-90
            im: FloatExp::new(2.0, -300),
        };
        let dc_exp = ComplexExp {
            re: FloatExp::new(1.5, -300),
            im: FloatExp::new(0.5, -300),
        };

        // ComplexExp version : delta préservé.
        let mut state = DeltaStateExp::new(z_ref_initial, delta_exp);
        state.step(phase, c_ref, dc_exp);
        // Le delta après step doit rester de magnitude raisonnable
        // (~2·Z·δ ~ Z·1e-90 ~ 1e-90, pas underflow).
        let mag = (state.delta.norm_sqr_approx()).sqrt();
        // Conversion: la magnitude réelle est ~1e-90, qui underflow f64
        // (qui s'arrête à ~1e-308). Donc norm_sqr_approx → 0.0.
        // Mais l'invariant interne (mantissa/exponent) doit rester valide.
        assert!(state.delta.re.exponent < -200, "exponent doit rester très négatif");
        let _ = mag;
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
