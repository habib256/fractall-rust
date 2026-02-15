use num_complex::Complex64;

/// Nombre dual complexe étendu avec 4 parties duales.
/// Utilisé pour suivre simultanément:
/// - Les dérivées pour distance estimation (dual_re, dual_im par rapport aux coordonnées pixel)
/// - Les dérivées pour interior detection (dual_z1_re, dual_z1_im par rapport à Z₁+z₁)
#[derive(Clone, Copy, Debug)]
pub struct ExtendedDualComplex {
    /// Valeur complexe
    pub value: Complex64,
    /// Dérivée par rapport à coordonnée X pixel (pour distance estimation)
    pub dual_re: Complex64,
    /// Dérivée par rapport à coordonnée Y pixel (pour distance estimation)
    pub dual_im: Complex64,
    /// Dérivée par rapport à Z₁+z₁ (partie réelle) pour interior detection
    pub dual_z1_re: Complex64,
    /// Dérivée par rapport à Z₁+z₁ (partie imaginaire) pour interior detection
    pub dual_z1_im: Complex64,
}

impl ExtendedDualComplex {
    #[allow(dead_code)] // Part of complete API, may be used in future
    pub fn zero() -> Self {
        Self {
            value: Complex64::new(0.0, 0.0),
            dual_re: Complex64::new(0.0, 0.0),
            dual_im: Complex64::new(0.0, 0.0),
            dual_z1_re: Complex64::new(0.0, 0.0),
            dual_z1_im: Complex64::new(0.0, 0.0),
        }
    }

    /// Crée un ExtendedDualComplex à partir d'un Complex64 avec dérivées nulles
    pub fn from_complex(value: Complex64) -> Self {
        Self {
            value,
            dual_re: Complex64::new(0.0, 0.0),
            dual_im: Complex64::new(0.0, 0.0),
            dual_z1_re: Complex64::new(0.0, 0.0),
            dual_z1_im: Complex64::new(0.0, 0.0),
        }
    }

    /// Initialise avec dzdz1 = 1+0i au départ (point critique)
    #[allow(dead_code)]
    pub fn from_critical_point(value: Complex64) -> Self {
        Self {
            value,
            dual_re: Complex64::new(0.0, 0.0),
            dual_im: Complex64::new(0.0, 0.0),
            dual_z1_re: Complex64::new(1.0, 0.0), // dzdz1 = 1+0i initialement
            dual_z1_im: Complex64::new(0.0, 0.0),
        }
    }

    /// Addition de deux ExtendedDualComplex
    pub fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            dual_re: self.dual_re + rhs.dual_re,
            dual_im: self.dual_im + rhs.dual_im,
            dual_z1_re: self.dual_z1_re + rhs.dual_z1_re,
            dual_z1_im: self.dual_z1_im + rhs.dual_z1_im,
        }
    }

    /// Multiplication de deux ExtendedDualComplex
    /// Pour z²: d(z²)/dz1 = 2z·dz/dz1
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            value: self.value * rhs.value,
            // Distance estimation derivatives
            dual_re: self.value * rhs.dual_re + rhs.value * self.dual_re,
            dual_im: self.value * rhs.dual_im + rhs.value * self.dual_im,
            // Interior detection derivatives: d(z₁·z₂)/dz1 = z₂·dz₁/dz1 + z₁·dz₂/dz1
            dual_z1_re: self.value * rhs.dual_z1_re + rhs.value * self.dual_z1_re,
            dual_z1_im: self.value * rhs.dual_z1_im + rhs.value * self.dual_z1_im,
        }
    }

    /// Carré d'un ExtendedDualComplex
    /// Pour z²: d(z²)/dz1 = 2z·dz/dz1
    pub fn square(self) -> Self {
        Self {
            value: self.value * self.value,
            // Distance estimation: d(z²)/dk = 2z·dz/dk
            dual_re: self.value * self.dual_re * 2.0,
            dual_im: self.value * self.dual_im * 2.0,
            // Interior detection: d(z²)/dz1 = 2z·dz/dz1
            dual_z1_re: self.value * self.dual_z1_re * 2.0,
            dual_z1_im: self.value * self.dual_z1_im * 2.0,
        }
    }

    /// Multiplication par un scalaire réel
    #[allow(dead_code)]
    pub fn scale(self, s: f64) -> Self {
        Self {
            value: self.value * s,
            dual_re: self.dual_re * s,
            dual_im: self.dual_im * s,
            dual_z1_re: self.dual_z1_re * s,
            dual_z1_im: self.dual_z1_im * s,
        }
    }

    /// Apply sign transformation for Burning Ship BLA perturbation.
    /// Multiplies real parts by sign_re and imaginary parts by sign_im
    /// for both value and all dual components.
    pub fn mul_signed(self, sign_re: f64, sign_im: f64) -> Self {
        Self {
            value: Complex64::new(self.value.re * sign_re, self.value.im * sign_im),
            dual_re: Complex64::new(self.dual_re.re * sign_re, self.dual_re.im * sign_im),
            dual_im: Complex64::new(self.dual_im.re * sign_re, self.dual_im.im * sign_im),
            dual_z1_re: Complex64::new(self.dual_z1_re.re * sign_re, self.dual_z1_re.im * sign_im),
            dual_z1_im: Complex64::new(self.dual_z1_im.re * sign_re, self.dual_z1_im.im * sign_im),
        }
    }

    /// Norme au carré de la valeur
    pub fn norm_sqr(self) -> f64 {
        self.value.norm_sqr()
    }

    /// Norme de la valeur
    #[allow(dead_code)] // Part of complete API, may be used in future
    pub fn norm(self) -> f64 {
        self.value.norm()
    }

    /// Calcule la norme de la dérivée pour interior detection: |dz/dz1|
    /// Utilise la norme opérateur pour les matrices 2×2 (valeurs singulières)
    pub fn interior_derivative_norm(self) -> f64 {
        // Pour les formules complexes-analytiques, dz/dz1 est un nombre complexe
        // Pour les formules non-analytiques, on utiliserait une matrice 2×2
        // Ici on suppose conforme, donc on utilise simplement la norme complexe
        // |dz/dz1| = sqrt(|dual_z1_re|² + |dual_z1_im|²)
        (self.dual_z1_re.norm_sqr() + self.dual_z1_im.norm_sqr()).sqrt()
    }
}

/// Vérifie si un point est dans l'intérieur d'un ensemble de Mandelbrot/Julia.
/// 
/// Un point est dans l'intérieur si |dz/dz1| < threshold, où dz/dz1 est la dérivée
/// de Z+z par rapport à Z₁+z₁ (point critique, généralement 0).
/// 
/// # Arguments
/// * `dual` - ExtendedDualComplex avec dérivées calculées
/// * `threshold` - Seuil pour la détection (par défaut 0.001)
/// 
/// # Returns
/// `true` si le point est dans l'intérieur
pub fn is_interior(dual: ExtendedDualComplex, threshold: f64) -> bool {
    let dz_norm = dual.interior_derivative_norm();
    dz_norm < threshold && dz_norm > 0.0 && dz_norm.is_finite()
}

/// Calcule la propagation de la dérivée dzdz1 pour une itération de perturbation.
/// 
/// Pour Mandelbrot z²+c: d(z²+c)/dz1 = 2z·dz/dz1
/// Pour Julia z²+c: d(z²+c)/dz1 = 2z·dz/dz1 (c est constant)
/// 
/// # Arguments
/// * `z` - Valeur complexe actuelle
/// * `dzdz1` - Dérivée précédente dz/dz1
/// 
/// # Returns
/// Nouvelle dérivée après une itération
#[allow(dead_code)]
pub fn propagate_interior_derivative(z: Complex64, dzdz1: Complex64) -> Complex64 {
    // Pour z²+c: d(z²+c)/dz1 = 2z·dz/dz1
    z * dzdz1 * 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extended_dual_square() {
        let z = ExtendedDualComplex {
            value: Complex64::new(2.0, 3.0),
            dual_re: Complex64::new(0.0, 0.0),
            dual_im: Complex64::new(0.0, 0.0),
            dual_z1_re: Complex64::new(1.0, 0.0),
            dual_z1_im: Complex64::new(0.0, 0.0),
        };
        let sq = z.square();
        // (2+3i)² = -5+12i
        assert!((sq.value.re - (-5.0)).abs() < 1e-10);
        assert!((sq.value.im - 12.0).abs() < 1e-10);
        // d(z²)/dz1 = 2z·dz/dz1 = 2*(2+3i)*1 = 4+6i
        assert!((sq.dual_z1_re.re - 4.0).abs() < 1e-10);
        assert!((sq.dual_z1_re.im - 6.0).abs() < 1e-10);
    }

    #[test]
    fn interior_detection() {
        // Point avec petite dérivée (intérieur)
        let interior = ExtendedDualComplex {
            value: Complex64::new(0.1, 0.1),
            dual_re: Complex64::new(0.0, 0.0),
            dual_im: Complex64::new(0.0, 0.0),
            dual_z1_re: Complex64::new(0.0005, 0.0),
            dual_z1_im: Complex64::new(0.0, 0.0),
        };
        assert!(is_interior(interior, 0.001));

        // Point avec grande dérivée (extérieur)
        let exterior = ExtendedDualComplex {
            value: Complex64::new(2.0, 2.0),
            dual_re: Complex64::new(0.0, 0.0),
            dual_im: Complex64::new(0.0, 0.0),
            dual_z1_re: Complex64::new(10.0, 0.0),
            dual_z1_im: Complex64::new(0.0, 0.0),
        };
        assert!(!is_interior(exterior, 0.001));
    }
}
