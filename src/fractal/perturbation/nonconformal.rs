use num_complex::Complex64;

/// Matrice 2×2 réelle pour les formules non-conformes (comme Tricorn/Mandelbar).
/// Pour les formules non-conformes, les dérivées ne peuvent pas être représentées
/// par des nombres complexes, donc on utilise des matrices 2×2 réelles.
#[derive(Clone, Copy, Debug)]
pub struct Matrix2x2 {
    pub m00: f64,
    pub m01: f64,
    pub m10: f64,
    pub m11: f64,
}

impl Matrix2x2 {
    #[allow(dead_code)]
    pub fn zero() -> Self {
        Self {
            m00: 0.0,
            m01: 0.0,
            m10: 0.0,
            m11: 0.0,
        }
    }

    pub fn identity() -> Self {
        Self {
            m00: 1.0,
            m01: 0.0,
            m10: 0.0,
            m11: 1.0,
        }
    }

    /// Multiplie la matrice par un vecteur 2D (re, im)
    pub fn mul_vector(self, re: f64, im: f64) -> (f64, f64) {
        (
            self.m00 * re + self.m01 * im,
            self.m10 * re + self.m11 * im,
        )
    }

    /// Multiplie deux matrices
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            m00: self.m00 * rhs.m00 + self.m01 * rhs.m10,
            m01: self.m00 * rhs.m01 + self.m01 * rhs.m11,
            m10: self.m10 * rhs.m00 + self.m11 * rhs.m10,
            m11: self.m10 * rhs.m01 + self.m11 * rhs.m11,
        }
    }

    /// Additionne deux matrices
    pub fn add(self, rhs: Self) -> Self {
        Self {
            m00: self.m00 + rhs.m00,
            m01: self.m01 + rhs.m01,
            m10: self.m10 + rhs.m10,
            m11: self.m11 + rhs.m11,
        }
    }

    /// Multiplie la matrice par un scalaire
    #[allow(dead_code)]
    pub fn scale(self, s: f64) -> Self {
        Self {
            m00: self.m00 * s,
            m01: self.m01 * s,
            m10: self.m10 * s,
            m11: self.m11 * s,
        }
    }

    /// Calcule la plus grande valeur singulière (sup|M|)
    pub fn sup_norm(self) -> f64 {
        // For a 2x2 matrix, the largest singular value can be computed as:
        // σ_max = sqrt((trace(M^T M) + sqrt(trace(M^T M)^2 - 4*det(M^T M))) / 2)
        let mtm_00 = self.m00 * self.m00 + self.m10 * self.m10;
        let mtm_01 = self.m00 * self.m01 + self.m10 * self.m11;
        let mtm_10 = self.m01 * self.m00 + self.m11 * self.m10;
        let mtm_11 = self.m01 * self.m01 + self.m11 * self.m11;
        
        let trace = mtm_00 + mtm_11;
        let det = mtm_00 * mtm_11 - mtm_01 * mtm_10;
        
        let discriminant = trace * trace - 4.0 * det;
        if discriminant < 0.0 {
            // Fallback: use Frobenius norm
            (mtm_00 + mtm_01 + mtm_10 + mtm_11).sqrt()
        } else {
            ((trace + discriminant.sqrt()) / 2.0).sqrt().max(0.0)
        }
    }

    /// Calcule la plus petite valeur singulière (inf|M|)
    pub fn inf_norm(self) -> f64 {
        // For a 2x2 matrix, the smallest singular value:
        // σ_min = sqrt((trace(M^T M) - sqrt(trace(M^T M)^2 - 4*det(M^T M))) / 2)
        let mtm_00 = self.m00 * self.m00 + self.m10 * self.m10;
        let mtm_01 = self.m00 * self.m01 + self.m10 * self.m11;
        let mtm_10 = self.m01 * self.m00 + self.m11 * self.m10;
        let mtm_11 = self.m01 * self.m01 + self.m11 * self.m11;
        
        let trace = mtm_00 + mtm_11;
        let det = mtm_00 * mtm_11 - mtm_01 * mtm_10;
        
        let discriminant = trace * trace - 4.0 * det;
        if discriminant < 0.0 {
            // Matrix is singular or near-singular
            0.0
        } else {
            let sqrt_disc = discriminant.sqrt();
            let min_singular = ((trace - sqrt_disc) / 2.0).sqrt().max(0.0);
            // Avoid division by zero
            min_singular.max(1e-20)
        }
    }
}

/// Coefficients BLA non-conformes pour une itération.
/// Pour Tricorn (Mandelbar): z' = conj(z)² + c = (X - iY)² + C
/// Les dérivées sont représentées par des matrices 2×2.
#[derive(Clone, Copy, Debug)]
pub struct BlaCoefficientsNonConformal {
    /// Coefficient linéaire A (matrice 2×2)
    pub a: Matrix2x2,
    /// Coefficient dc B (matrice 2×2)
    pub b: Matrix2x2,
}

impl BlaCoefficientsNonConformal {
    #[allow(dead_code)]
    pub fn zero() -> Self {
        Self {
            a: Matrix2x2::zero(),
            b: Matrix2x2::zero(),
        }
    }
}

/// Calcule les coefficients BLA non-conformes pour Tricorn.
/// Tricorn: z' = (X - iY)² + C = (X² - Y² - 2iXY) + C
/// 
/// Pour z = X + iY, on a:
/// z' = X' + iY' où:
/// X' = X² - Y² + C_re
/// Y' = -2XY + C_im
///
/// Les dérivées partielles:
/// dX'/dX = 2X, dX'/dY = -2Y
/// dY'/dX = -2Y, dY'/dY = -2X
///
/// Donc A = [[2X, -2Y], [-2Y, -2X]]
pub fn compute_tricorn_bla_coefficients(z: Complex64) -> BlaCoefficientsNonConformal {
    let x = z.re;
    let y = z.im;
    
    // A = [[2X, -2Y], [-2Y, -2X]]
    let a = Matrix2x2 {
        m00: 2.0 * x,
        m01: -2.0 * y,
        m10: -2.0 * y,
        m11: -2.0 * x,
    };
    
    // B = I (matrice identité) car dc contribue directement
    let b = Matrix2x2::identity();
    
    BlaCoefficientsNonConformal { a, b }
}

/// Calcule le rayon de validité pour un BLA non-conforme single-step (Section 4.1 of deep zoom theory).
/// Formule: R = ε·inf|A| - sup|B|·|c| / inf|A|
/// For non-conformal formulas, we use inf|A| and sup|B| (largest/smallest singular values)
pub fn compute_nonconformal_validity_radius(
    a: Matrix2x2,
    b: Matrix2x2,
    epsilon: f64,
    c_norm: f64,
) -> f64 {
    let inf_a = a.inf_norm();
    let sup_b = b.sup_norm();
    
    if inf_a < 1e-20 {
        return 0.0;
    }
    
    let term1 = epsilon * inf_a;
    let term2 = sup_b * c_norm / inf_a;
    
    (term1 - term2).max(0.0)
}

/// Fusionne deux BLAs non-conformes (Section 4.1 of deep zoom theory).
/// Si Tx skips lx iterations from mx when |z| < Rx
/// and Ty skips ly iterations from mx+lx when |z| < Ry,
/// then Tz = Ty ∘ Tx skips lx+ly iterations from mx when |z| < Rz.
///
/// Formule fusion: Rz = max{0, min{Rx, Ry - sup|Bx|·|c| / sup|Ax|}}
/// Note: Uses sup|Ax| in denominator as per theory (not inf|Ax|)
pub fn merge_nonconformal_bla(
    ax: Matrix2x2,
    bx: Matrix2x2,
    rx: f64,
    ay: Matrix2x2,
    by: Matrix2x2,
    ry: f64,
    c_norm: f64,
) -> (Matrix2x2, Matrix2x2, f64) {
    // Composition: Az = Ay·Ax, Bz = Ay·Bx + By
    let az = ay.mul(ax);
    let bz = ay.mul(bx).add(by);
    
    // Validity radius: Rz = max{0, min{Rx, Ry - sup|Bx|·|c| / sup|Ax|}}
    let sup_ax = ax.sup_norm();
    let sup_bx = bx.sup_norm();
    
    let rz = if sup_ax < 1e-20 {
        rx.min(ry).max(0.0)
    } else {
        let adjustment = sup_bx * c_norm / sup_ax;
        rx.min(ry - adjustment).max(0.0)
    };
    
    (az, bz, rz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_multiplication() {
        let m1 = Matrix2x2 {
            m00: 1.0,
            m01: 2.0,
            m10: 3.0,
            m11: 4.0,
        };
        let m2 = Matrix2x2 {
            m00: 5.0,
            m01: 6.0,
            m10: 7.0,
            m11: 8.0,
        };
        let result = m1.mul(m2);
        // [1 2] [5 6]   [19 22]
        // [3 4] [7 8] = [43 50]
        assert!((result.m00 - 19.0).abs() < 1e-10);
        assert!((result.m01 - 22.0).abs() < 1e-10);
        assert!((result.m10 - 43.0).abs() < 1e-10);
        assert!((result.m11 - 50.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_vector_multiplication() {
        let m = Matrix2x2 {
            m00: 1.0,
            m01: 2.0,
            m10: 3.0,
            m11: 4.0,
        };
        let (re, im) = m.mul_vector(5.0, 6.0);
        // [1 2] [5]   [17]
        // [3 4] [6] = [39]
        assert!((re - 17.0).abs() < 1e-10);
        assert!((im - 39.0).abs() < 1e-10);
    }

    #[test]
    fn tricorn_coefficients() {
        let z = Complex64::new(2.0, 3.0);
        let coeffs = compute_tricorn_bla_coefficients(z);
        // A = [[2X, -2Y], [-2Y, -2X]] = [[4, -6], [-6, -4]]
        assert!((coeffs.a.m00 - 4.0).abs() < 1e-10);
        assert!((coeffs.a.m01 - (-6.0)).abs() < 1e-10);
        assert!((coeffs.a.m10 - (-6.0)).abs() < 1e-10);
        assert!((coeffs.a.m11 - (-4.0)).abs() < 1e-10);
        // B should be identity
        assert!((coeffs.b.m00 - 1.0).abs() < 1e-10);
        assert!((coeffs.b.m11 - 1.0).abs() < 1e-10);
    }
}
