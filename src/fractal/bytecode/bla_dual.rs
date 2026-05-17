//! Construction BLA unifiée via dual-numbers walking le bytecode (style F3).
//!
//! Mirrors `hybrid_bla()` de Fraktaler-3 (cf. `docs/fraktaler-3-analysis.md` §4).
//! Produit pour une itération unique :
//! - Une matrice 2×2 réelle `A` = Jacobien de z' par rapport à δ
//! - Un rayon de validité `r²` (carré, comme F3 stocke directement r²)
//! - Le nombre d'itérations sautées (toujours 1 pour single-step)
//!
//! B est implicitement l'identité car `Op::Add` est toujours en fin de phase
//! et `z := z + c` propage c via identité.
//!
//! Pour un step on traverse les opcodes en propageant un `DualComplex2`
//! `{ value: (x, y), jac: Mat2 }` initialisé à `value = Z_ref, jac = I`.
//! Chaque opcode met à jour value et jac selon la règle de la chaîne.

use crate::fractal::bytecode::{Op, Phase};

/// Matrice 2×2 réelle (compatible avec `nonconformal::Matrix2x2`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat2 {
    pub m00: f64,
    pub m01: f64,
    pub m10: f64,
    pub m11: f64,
}

impl Mat2 {
    pub const ZERO: Self = Self { m00: 0.0, m01: 0.0, m10: 0.0, m11: 0.0 };
    pub const IDENTITY: Self = Self { m00: 1.0, m01: 0.0, m10: 0.0, m11: 1.0 };

    /// Multiplication matricielle `self · rhs`.
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            m00: self.m00 * rhs.m00 + self.m01 * rhs.m10,
            m01: self.m00 * rhs.m01 + self.m01 * rhs.m11,
            m10: self.m10 * rhs.m00 + self.m11 * rhs.m10,
            m11: self.m10 * rhs.m01 + self.m11 * rhs.m11,
        }
    }

    /// Multiplication par scalaire.
    pub fn scale(self, s: f64) -> Self {
        Self {
            m00: self.m00 * s,
            m01: self.m01 * s,
            m10: self.m10 * s,
            m11: self.m11 * s,
        }
    }

    /// Norme opérateur 2 (plus grande valeur singulière).
    /// Formule fermée pour 2×2 via les valeurs propres de Mᵀ·M.
    pub fn sup_norm(self) -> f64 {
        let mtm_00 = self.m00 * self.m00 + self.m10 * self.m10;
        let mtm_01 = self.m00 * self.m01 + self.m10 * self.m11;
        let mtm_11 = self.m01 * self.m01 + self.m11 * self.m11;
        let trace = mtm_00 + mtm_11;
        let det = mtm_00 * mtm_11 - mtm_01 * mtm_01; // Mᵀ·M est symétrique → m10 = m01
        let disc = (trace * trace - 4.0 * det).max(0.0);
        ((trace + disc.sqrt()) * 0.5).max(0.0).sqrt()
    }
}

/// État dual : `value` (point z courant) + `jac` (∂z/∂δ).
#[derive(Clone, Copy, Debug)]
pub struct DualComplex2 {
    pub value_x: f64,
    pub value_y: f64,
    pub jac: Mat2,
}

impl DualComplex2 {
    /// Initialise comme `z = z0 + δ` (jac = I).
    pub fn from_value(z0_x: f64, z0_y: f64) -> Self {
        Self { value_x: z0_x, value_y: z0_y, jac: Mat2::IDENTITY }
    }

    /// `z := z²` (multiplication complexe).
    /// `(x + iy)² = (x² - y²) + i·2xy`
    /// Chain rule : nouvelle jac = `2 · mat2(x, -y; y, x) · jac`.
    fn sqr(&mut self) {
        let x = self.value_x;
        let y = self.value_y;
        let m = Mat2 { m00: x, m01: -y, m10: y, m11: x }.scale(2.0);
        self.jac = m.mul(self.jac);
        // value := value²
        let new_x = x * x - y * y;
        let new_y = 2.0 * x * y;
        self.value_x = new_x;
        self.value_y = new_y;
    }

    /// `z := z · stored` (multiplication complexe).
    /// Le `stored` a sa propre Jacobienne : la dérivée du produit est
    /// `d(z·s) = dz·s + z·ds` (règle de Leibniz). Donc :
    /// `new_jac = mat_complex(s) · jac + mat_complex(z) · stored.jac`.
    fn mul(&mut self, stored: &DualComplex2) {
        let zx = self.value_x;
        let zy = self.value_y;
        let sx = stored.value_x;
        let sy = stored.value_y;
        let ms = Mat2 { m00: sx, m01: -sy, m10: sy, m11: sx };
        let mz = Mat2 { m00: zx, m01: -zy, m10: zy, m11: zx };
        // Σ : jac' = ms · jac + mz · stored.jac
        let jac_a = ms.mul(self.jac);
        let jac_b = mz.mul(stored.jac);
        self.jac = Mat2 {
            m00: jac_a.m00 + jac_b.m00,
            m01: jac_a.m01 + jac_b.m01,
            m10: jac_a.m10 + jac_b.m10,
            m11: jac_a.m11 + jac_b.m11,
        };
        // value := z · s = (zx·sx - zy·sy) + i(zx·sy + zy·sx)
        let new_x = zx * sx - zy * sy;
        let new_y = zx * sy + zy * sx;
        self.value_x = new_x;
        self.value_y = new_y;
    }

    /// `z.re := |z.re|`. Si re < 0, on flip la ligne 0 de la Jacobienne.
    fn abs_x(&mut self) {
        if self.value_x < 0.0 {
            self.value_x = -self.value_x;
            self.jac.m00 = -self.jac.m00;
            self.jac.m01 = -self.jac.m01;
        }
    }

    fn abs_y(&mut self) {
        if self.value_y < 0.0 {
            self.value_y = -self.value_y;
            self.jac.m10 = -self.jac.m10;
            self.jac.m11 = -self.jac.m11;
        }
    }

    /// `z.re := -z.re` ; flip la ligne 0 de la Jacobienne.
    fn neg_x(&mut self) {
        self.value_x = -self.value_x;
        self.jac.m00 = -self.jac.m00;
        self.jac.m01 = -self.jac.m01;
    }

    fn neg_y(&mut self) {
        self.value_y = -self.value_y;
        self.jac.m10 = -self.jac.m10;
        self.jac.m11 = -self.jac.m11;
    }

    fn abs_value(&self) -> f64 {
        (self.value_x * self.value_x + self.value_y * self.value_y).sqrt()
    }
}

/// BLA single-step calculé pour un point de référence et une phase.
#[derive(Clone, Copy, Debug)]
pub struct BlaSingleStep {
    /// Jacobien A = ∂z'/∂δ.
    pub a: Mat2,
    /// Rayon de validité au carré (F3 stocke directement r²).
    pub r2: f64,
}

/// Construit le BLA single-step en walking le bytecode avec dual-numbers.
///
/// Suit `hybrid_bla()` de F3. `z_ref` est la valeur de la référence à l'itération
/// courante (avant application de la phase). `epsilon` est le facteur de précision
/// (typiquement `2^(-prec_bits)`, par exemple `2^-24 ≈ 6e-8` pour f32).
pub fn build_bla_single_step(
    z_ref_x: f64,
    z_ref_y: f64,
    phase: &Phase,
    epsilon: f64,
) -> BlaSingleStep {
    let mut w = DualComplex2::from_value(z_ref_x, z_ref_y);
    let mut stored = w; // valeur arbitraire ; Store l'écrasera si appelé
    let z_abs = w.abs_value();

    // Rayon de validité initial. F3 utilise `e * |Z| * degree / (degree*(degree-1)/2)`
    // mais notre boucle prend le min sur les ops donc le seed initial peut être grand.
    let mut r = f64::INFINITY;

    for op in &phase.ops {
        // Sauvegarder l'état AVANT l'op (utile pour le calcul du rayon).
        let w_before = w;
        let stored_before = stored;
        let sup_a_before = w.jac.sup_norm();

        match op {
            Op::Sqr => {
                // r ← min(r, ε · |W0| / sup(A0))
                if sup_a_before > 1e-300 {
                    let candidate = epsilon * w_before.abs_value() / sup_a_before;
                    r = r.min(candidate);
                }
                w.sqr();
            }
            Op::Mul => {
                // r ← min(r, ε · min(|W0|, |W0_stored|) / sup(A0))
                if sup_a_before > 1e-300 {
                    let m = w_before.abs_value().min(stored_before.abs_value());
                    let candidate = epsilon * m / sup_a_before;
                    r = r.min(candidate);
                }
                w.mul(&stored);
            }
            Op::Store => {
                stored = w;
                // pas de contrainte sur r
            }
            Op::AbsX => {
                // r ← min(r, |W0.x| / 2 / sup(A0))
                if sup_a_before > 1e-300 {
                    let candidate = w_before.value_x.abs() / 2.0 / sup_a_before;
                    r = r.min(candidate);
                }
                w.abs_x();
            }
            Op::AbsY => {
                if sup_a_before > 1e-300 {
                    let candidate = w_before.value_y.abs() / 2.0 / sup_a_before;
                    r = r.min(candidate);
                }
                w.abs_y();
            }
            Op::NegX => {
                w.neg_x();
            }
            Op::NegY => {
                w.neg_y();
            }
            Op::Add => {
                // Fin de phase : on retourne A et r².
                let _ = z_abs; // évite unused
                let r_clamped = if r.is_finite() { r } else { 0.0 };
                return BlaSingleStep {
                    a: w.jac,
                    r2: r_clamped * r_clamped,
                };
            }
        }
    }

    // Si pas d'Op::Add (bytecode mal formé), on retourne quand même.
    BlaSingleStep {
        a: w.jac,
        r2: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::bytecode::compile_formula;
    use crate::fractal::FractalType;

    /// Compare deux mat2 avec tolérance.
    fn mat2_close(a: Mat2, b: Mat2, tol: f64) -> bool {
        (a.m00 - b.m00).abs() < tol
            && (a.m01 - b.m01).abs() < tol
            && (a.m10 - b.m10).abs() < tol
            && (a.m11 - b.m11).abs() < tol
    }

    /// Mandelbrot : A devrait être 2·[Zx, -Zy; Zy, Zx] (multiplication complexe par 2z).
    #[test]
    fn mandelbrot_bla_matches_complex_2z() {
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let (zx, zy) = (0.3, -0.4);
        let bla = build_bla_single_step(zx, zy, &formula.phases[0], 1e-6);
        let expected = Mat2 {
            m00: 2.0 * zx,
            m01: -2.0 * zy,
            m10: 2.0 * zy,
            m11: 2.0 * zx,
        };
        assert!(
            mat2_close(bla.a, expected, 1e-10),
            "got {:?}, expected {:?}",
            bla.a,
            expected
        );
    }

    /// Tricorn : (X-iY)². A = [[2X, -2Y], [-2Y, -2X]] (cf. nonconformal::compute_tricorn_bla_coefficients).
    #[test]
    fn tricorn_bla_matches_existing_formula() {
        let formula = compile_formula(FractalType::Tricorn, 2.0).unwrap();
        let (zx, zy) = (2.0, 3.0);
        let bla = build_bla_single_step(zx, zy, &formula.phases[0], 1e-6);
        let expected = Mat2 {
            m00: 2.0 * zx,
            m01: -2.0 * zy,
            m10: -2.0 * zy,
            m11: -2.0 * zx,
        };
        assert!(
            mat2_close(bla.a, expected, 1e-10),
            "got {:?}, expected {:?}",
            bla.a,
            expected
        );
    }

    /// Burning Ship 1er quadrant : A = [[2X, -2Y], [2Y, 2X]] (conformal).
    #[test]
    fn burning_ship_q1_matches_existing() {
        let formula = compile_formula(FractalType::BurningShip, 2.0).unwrap();
        let (zx, zy) = (2.0, 3.0);
        let bla = build_bla_single_step(zx, zy, &formula.phases[0], 1e-6);
        let expected = Mat2 {
            m00: 2.0 * zx,
            m01: -2.0 * zy,
            m10: 2.0 * zy,
            m11: 2.0 * zx,
        };
        assert!(
            mat2_close(bla.a, expected, 1e-10),
            "got {:?}, expected {:?}",
            bla.a,
            expected
        );
    }

    /// Burning Ship 2e quadrant (X<0, Y>=0) : A = [[2X, -2Y], [-2Y, 2|X|]].
    /// Référence nonconformal::compute_burning_ship_bla_coefficients :
    ///   m10 = 2·sign(X)·|Y| = -2|Y|·sign si X<0
    ///   m11 = 2·|X|·sign(Y) = 2|X| si Y>=0
    #[test]
    fn burning_ship_q2_matches_existing() {
        let formula = compile_formula(FractalType::BurningShip, 2.0).unwrap();
        let (zx, zy) = (-2.0, 3.0);
        let bla = build_bla_single_step(zx, zy, &formula.phases[0], 1e-6);
        let expected = Mat2 {
            m00: 2.0 * zx,         // -4
            m01: -2.0 * zy,        // -6
            m10: 2.0 * (-1.0) * zy.abs(), // -6
            m11: 2.0 * zx.abs() * 1.0,    //  4
        };
        assert!(
            mat2_close(bla.a, expected, 1e-10),
            "got {:?}, expected {:?}",
            bla.a,
            expected
        );
    }

    /// Burning Ship 3e quadrant (X<0, Y<0) : A = [[2X, -2Y], [2Y, 2X]] (conformal de signe opposé).
    #[test]
    fn burning_ship_q3_matches_existing() {
        let formula = compile_formula(FractalType::BurningShip, 2.0).unwrap();
        let (zx, zy) = (-2.0, -3.0);
        let bla = build_bla_single_step(zx, zy, &formula.phases[0], 1e-6);
        // m10 = 2·sign(X)·|Y| = 2·(-1)·3 = -6
        // m11 = 2·|X|·sign(Y) = 2·2·(-1) = -4
        let expected = Mat2 {
            m00: -4.0,
            m01: 6.0,
            m10: -6.0,
            m11: -4.0,
        };
        assert!(
            mat2_close(bla.a, expected, 1e-10),
            "got {:?}, expected {:?}",
            bla.a,
            expected
        );
    }

    /// Multibrot puissance 3 : z³ = Store; Sqr; Mul; Add.
    /// A = 3·z²·I (au sens matrice complexe = mat2 de mul par 3z²).
    #[test]
    fn multibrot_pow3_jacobian() {
        let formula = compile_formula(FractalType::Multibrot, 3.0).unwrap();
        let (zx, zy) = (0.5, 0.3);
        let bla = build_bla_single_step(zx, zy, &formula.phases[0], 1e-6);
        // dz³/dz = 3z². z² = (zx²-zy² + i·2zx·zy). 3z² = (3(zx²-zy²) + i·6zx·zy).
        // mat2 de mul par 3z² = [[Re, -Im], [Im, Re]] de 3z².
        let re_3z2 = 3.0 * (zx * zx - zy * zy);
        let im_3z2 = 3.0 * 2.0 * zx * zy;
        let expected = Mat2 {
            m00: re_3z2,
            m01: -im_3z2,
            m10: im_3z2,
            m11: re_3z2,
        };
        assert!(
            mat2_close(bla.a, expected, 1e-10),
            "got {:?}, expected {:?}",
            bla.a,
            expected
        );
    }

    /// Multibrot puissance 4 : z⁴ = Sqr; Sqr; Add.
    /// A = 4z³.
    #[test]
    fn multibrot_pow4_jacobian() {
        let formula = compile_formula(FractalType::Multibrot, 4.0).unwrap();
        let (zx, zy) = (0.5, 0.3);
        let bla = build_bla_single_step(zx, zy, &formula.phases[0], 1e-6);
        // 4z³ : on calcule z² = (zx²-zy², 2zx·zy), puis z³ = z·z² (mul complexe), puis 4·z³.
        let zsq_re = zx * zx - zy * zy;
        let zsq_im = 2.0 * zx * zy;
        let z3_re = zx * zsq_re - zy * zsq_im;
        let z3_im = zx * zsq_im + zy * zsq_re;
        let re_4z3 = 4.0 * z3_re;
        let im_4z3 = 4.0 * z3_im;
        let expected = Mat2 {
            m00: re_4z3,
            m01: -im_4z3,
            m10: im_4z3,
            m11: re_4z3,
        };
        assert!(
            mat2_close(bla.a, expected, 1e-10),
            "got {:?}, expected {:?}",
            bla.a,
            expected
        );
    }

    /// Celtic : z² puis |Re|. A devrait incorporer le signe de Re(z²).
    /// Re(z²) = Zx² - Zy², donc si Zx² > Zy² alors AbsX est no-op (Re>=0),
    /// sinon flip de la ligne 0.
    #[test]
    fn celtic_q_positive_re_zsq() {
        let formula = compile_formula(FractalType::Celtic, 2.0).unwrap();
        // Choisir zx tel que zx² > zy² → Re(z²) > 0
        let (zx, zy) = (1.0, 0.5);
        let bla = build_bla_single_step(zx, zy, &formula.phases[0], 1e-6);
        // Re(z²) = 1 - 0.25 = 0.75 > 0 → AbsX no-op → A = Jacobien de z² = 2·[Zx, -Zy; Zy, Zx]
        let expected = Mat2 {
            m00: 2.0 * zx,
            m01: -2.0 * zy,
            m10: 2.0 * zy,
            m11: 2.0 * zx,
        };
        assert!(
            mat2_close(bla.a, expected, 1e-10),
            "Celtic Re(z²)>0: got {:?}, expected {:?}",
            bla.a,
            expected
        );
    }

    #[test]
    fn celtic_q_negative_re_zsq() {
        let formula = compile_formula(FractalType::Celtic, 2.0).unwrap();
        // Choisir zx tel que zx² < zy² → Re(z²) < 0 → AbsX flip ligne 0
        let (zx, zy) = (0.5, 1.0);
        let bla = build_bla_single_step(zx, zy, &formula.phases[0], 1e-6);
        // Jac après z² = 2·[Zx, -Zy; Zy, Zx]. AbsX flip ligne 0 → [-2Zx, 2Zy; 2Zy, 2Zx]
        let expected = Mat2 {
            m00: -2.0 * zx,
            m01: 2.0 * zy,
            m10: 2.0 * zy,
            m11: 2.0 * zx,
        };
        assert!(
            mat2_close(bla.a, expected, 1e-10),
            "Celtic Re(z²)<0: got {:?}, expected {:?}",
            bla.a,
            expected
        );
    }

    /// Le rayon de validité est positif et fini pour des points raisonnables.
    #[test]
    fn validity_radius_finite() {
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let bla = build_bla_single_step(0.3, -0.4, &formula.phases[0], 1e-6);
        assert!(bla.r2.is_finite());
        assert!(bla.r2 > 0.0);
    }

    #[test]
    fn mat2_sup_norm_identity_is_1() {
        assert!((Mat2::IDENTITY.sup_norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mat2_sup_norm_scaled() {
        let m = Mat2::IDENTITY.scale(3.5);
        assert!((m.sup_norm() - 3.5).abs() < 1e-10);
    }
}
