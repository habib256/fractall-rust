//! Interpréteur GMP du bytecode hybride pour la construction de l'orbite référence.
//!
//! Utilise des buffers réutilisables pour éviter les allocations GMP par opcode
//! (cf. bugfix #12 sur la tolérance glitch GMP recalculée — même principe).
//! Conçu pour être appelé en boucle serrée sur des centaines de milliers d'itérations.

use rug::{Assign, Complex, Float};

use super::{Formula, Op};

/// Buffers GMP réutilisables pour l'interpréteur. Allouer une fois, réutiliser
/// à chaque itération. Sans ça : ~5 allocations GMP par opcode = très lent.
pub struct GmpInterpState {
    pub z: Complex,
    pub stored: Complex,
    pub phase: usize,
    /// Scratch pour les opérations sur les parties réelle/imaginaire.
    re_scratch: Float,
    im_scratch: Float,
    /// Scratch complexe pour `Sqr` (z = z * z).
    cplx_scratch: Complex,
    #[allow(dead_code)]
    prec: u32,
}

impl GmpInterpState {
    pub fn new(prec: u32, z0: Complex) -> Self {
        Self {
            stored: z0.clone(),
            z: z0,
            phase: 0,
            re_scratch: Float::with_val(prec, 0),
            im_scratch: Float::with_val(prec, 0),
            cplx_scratch: Complex::with_val(prec, (0, 0)),
            prec,
        }
    }

    /// Exécute une itération complète (toutes les ops d'une phase, incrément phase).
    ///
    /// `c` est la constante ajoutée par `Op::Add` (cref pour Mandelbrot-like,
    /// seed pour Julia-like).
    pub fn step(&mut self, formula: &Formula, c: &Complex) {
        let phase = &formula.phases[self.phase];
        for op in &phase.ops {
            match op {
                Op::Sqr => {
                    // z = z * z via scratch pour éviter une alloc
                    self.cplx_scratch.assign(&self.z);
                    self.z *= &self.cplx_scratch;
                }
                Op::Mul => {
                    self.z *= &self.stored;
                }
                Op::Store => {
                    self.stored.assign(&self.z);
                }
                Op::AbsX => {
                    self.re_scratch.assign(self.z.real());
                    self.re_scratch.abs_mut();
                    // SAFETY de rug : on assigne directement via mutable_real()
                    self.z.mut_real().assign(&self.re_scratch);
                }
                Op::AbsY => {
                    self.im_scratch.assign(self.z.imag());
                    self.im_scratch.abs_mut();
                    self.z.mut_imag().assign(&self.im_scratch);
                }
                Op::NegX => {
                    self.re_scratch.assign(self.z.real());
                    self.re_scratch = -self.re_scratch.clone();
                    self.z.mut_real().assign(&self.re_scratch);
                }
                Op::NegY => {
                    self.im_scratch.assign(self.z.imag());
                    self.im_scratch = -self.im_scratch.clone();
                    self.z.mut_imag().assign(&self.im_scratch);
                }
                Op::Add => {
                    self.z += c;
                }
            }
        }
        // Cycle de phase
        let n_phases = formula.phases.len();
        if n_phases > 1 {
            self.phase = (self.phase + 1) % n_phases;
        }
    }

    #[allow(dead_code)]
    pub fn prec(&self) -> u32 {
        self.prec
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::bytecode::compile_formula;
    use crate::fractal::FractalType;
    use num_complex::Complex64;
    use rug::Complex;

    fn gmp_to_c64(z: &Complex) -> Complex64 {
        Complex64::new(z.real().to_f64(), z.imag().to_f64())
    }

    /// Vérifie que l'interpréteur GMP produit la même orbite que l'arithmétique
    /// f64 (modulo l'arrondi) pour Mandelbrot sur un point régulier.
    #[test]
    fn mandelbrot_gmp_matches_f64() {
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let prec = 256u32;
        let c = Complex::with_val(prec, (-0.5, 0.3));
        let z0 = Complex::with_val(prec, (0, 0));
        let mut state = GmpInterpState::new(prec, z0);

        let mut z_f64 = Complex64::new(0.0, 0.0);
        let c_f64 = Complex64::new(-0.5, 0.3);
        for _ in 0..30 {
            state.step(&formula, &c);
            z_f64 = z_f64 * z_f64 + c_f64;
            let diff = (gmp_to_c64(&state.z) - z_f64).norm();
            assert!(diff < 1e-10, "divergence GMP vs f64: {}", diff);
        }
    }

    #[test]
    fn burning_ship_gmp_matches_f64() {
        let formula = compile_formula(FractalType::BurningShip, 2.0).unwrap();
        let prec = 256u32;
        // Point intérieur du BS pour orbite bornée (sinon la divergence chaotique
        // amplifie l'écart de rounding).
        let c = Complex::with_val(prec, (-1.62, -0.03));
        let z0 = Complex::with_val(prec, (0, 0));
        let mut state = GmpInterpState::new(prec, z0);

        let mut z_f64 = Complex64::new(0.0, 0.0);
        let c_f64 = Complex64::new(-1.62, -0.03);
        for _ in 0..20 {
            state.step(&formula, &c);
            let z_abs = Complex64::new(z_f64.re.abs(), z_f64.im.abs());
            z_f64 = z_abs * z_abs + c_f64;
            let diff = (gmp_to_c64(&state.z) - z_f64).norm();
            assert!(diff < 1e-10, "BS divergence: {}", diff);
        }
    }

    #[test]
    fn tricorn_gmp_matches_f64() {
        let formula = compile_formula(FractalType::Tricorn, 2.0).unwrap();
        let prec = 256u32;
        // Point intérieur du Tricorn pour orbite bornée.
        let c = Complex::with_val(prec, (-0.1, 0.1));
        let z0 = Complex::with_val(prec, (0, 0));
        let mut state = GmpInterpState::new(prec, z0);

        let mut z_f64 = Complex64::new(0.0, 0.0);
        let c_f64 = Complex64::new(-0.1, 0.1);
        for _ in 0..20 {
            state.step(&formula, &c);
            let conj = Complex64::new(z_f64.re, -z_f64.im);
            z_f64 = conj * conj + c_f64;
            let diff = (gmp_to_c64(&state.z) - z_f64).norm();
            assert!(diff < 1e-10, "Tricorn divergence: {}", diff);
        }
    }

    #[test]
    fn multibrot_pow3_gmp_matches_f64() {
        let formula = compile_formula(FractalType::Multibrot, 3.0).unwrap();
        let prec = 256u32;
        let c = Complex::with_val(prec, (0.4, 0.1));
        let z0 = Complex::with_val(prec, (0, 0));
        let mut state = GmpInterpState::new(prec, z0);

        let mut z_f64 = Complex64::new(0.0, 0.0);
        let c_f64 = Complex64::new(0.4, 0.1);
        for _ in 0..30 {
            state.step(&formula, &c);
            let z_sq = z_f64 * z_f64;
            z_f64 = z_sq * z_f64 + c_f64;
            let diff = (gmp_to_c64(&state.z) - z_f64).norm();
            assert!(diff < 1e-10, "Multibrot pow3 divergence: {}", diff);
        }
    }
}
