use num_complex::Complex64;

/// Nombre dual complexe pour la différentiation automatique.
/// Utilisé pour suivre les dérivées de Z + z par rapport aux coordonnées pixel.
/// 
/// Pour la distance estimation, on suit dz/dk où k sont les coordonnées pixel.
/// value: la valeur complexe
/// dual_re: dérivée par rapport à la coordonnée X du pixel
/// dual_im: dérivée par rapport à la coordonnée Y du pixel
#[derive(Clone, Copy, Debug)]
pub struct DualComplex {
    pub value: Complex64,
    pub dual_re: Complex64,
    pub dual_im: Complex64,
}

impl DualComplex {
    #[allow(dead_code)]
    pub fn zero() -> Self {
        Self {
            value: Complex64::new(0.0, 0.0),
            dual_re: Complex64::new(0.0, 0.0),
            dual_im: Complex64::new(0.0, 0.0),
        }
    }

    /// Crée un DualComplex à partir d'un Complex64 avec des dérivées nulles
    #[allow(dead_code)]
    pub fn from_complex(value: Complex64) -> Self {
        Self {
            value,
            dual_re: Complex64::new(0.0, 0.0),
            dual_im: Complex64::new(0.0, 0.0),
        }
    }

    /// Crée un DualComplex pour une coordonnée pixel avec dérivée unitaire
    /// pixel_re: coordonnée X du pixel avec dual_re = 1+0i
    /// pixel_im: coordonnée Y du pixel avec dual_im = 0+1i
    #[allow(dead_code)]
    pub fn from_pixel_coords(pixel_re: f64, pixel_im: f64) -> Self {
        Self {
            value: Complex64::new(pixel_re, pixel_im),
            dual_re: Complex64::new(1.0, 0.0),
            dual_im: Complex64::new(0.0, 1.0),
        }
    }

    /// Addition de deux nombres duals complexes
    #[allow(dead_code)]
    pub fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            dual_re: self.dual_re + rhs.dual_re,
            dual_im: self.dual_im + rhs.dual_im,
        }
    }

    /// Multiplication de deux nombres duals complexes
    /// (a + ε·da) * (b + ε·db) = a*b + ε·(a*db + b*da)
    #[allow(dead_code)]
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            value: self.value * rhs.value,
            dual_re: self.value * rhs.dual_re + rhs.value * self.dual_re,
            dual_im: self.value * rhs.dual_im + rhs.value * self.dual_im,
        }
    }

    /// Multiplication par un scalaire réel
    #[allow(dead_code)]
    pub fn scale(self, s: f64) -> Self {
        Self {
            value: self.value * s,
            dual_re: self.dual_re * s,
            dual_im: self.dual_im * s,
        }
    }

    /// Carré d'un nombre dual complexe
    /// (z + ε·dz)² = z² + ε·2z·dz
    #[allow(dead_code)]
    pub fn square(self) -> Self {
        Self {
            value: self.value * self.value,
            dual_re: self.value * self.dual_re * 2.0,
            dual_im: self.value * self.dual_im * 2.0,
        }
    }

    /// Norme au carré de la valeur
    #[allow(dead_code)] // Used in tests, part of complete API
    pub fn norm_sqr(self) -> f64 {
        self.value.norm_sqr()
    }

    /// Norme de la valeur
    pub fn norm(self) -> f64 {
        self.value.norm()
    }

    /// Calcule la norme de la dérivée directionnelle |dz/dk|
    /// où k est la direction du pixel
    pub fn derivative_norm(self) -> f64 {
        // |dz/dk| = sqrt(|dual_re|² + |dual_im|²)
        (self.dual_re.norm_sqr() + self.dual_im.norm_sqr()).sqrt()
    }
}

/// Calcule la distance estimation à partir d'un DualComplex.
/// 
/// Formule: distance = |z|·ln|z| / |dz/dk|
/// où z est la valeur complexe et dz/dk est la dérivée par rapport aux coordonnées pixel.
pub fn compute_distance_estimate(dual: DualComplex) -> f64 {
    let z_norm = dual.norm();
    let dz_norm = dual.derivative_norm();
    
    if z_norm <= 0.0 || dz_norm <= 0.0 || !z_norm.is_finite() || !dz_norm.is_finite() {
        return f64::INFINITY;
    }
    
    // Distance estimation: |z|·ln|z| / |dz/dk|
    // Le dz/dk est déjà préscalé par l'espacement des pixels, ce qui aide à éviter l'overflow
    z_norm * z_norm.ln() / dz_norm
}

/// Transforme les coordonnées pixel en coordonnées complexes avec propagation des dérivées.
/// 
/// # Arguments
/// * `pixel_re` - Coordonnée X du pixel (0..width)
/// * `pixel_im` - Coordonnée Y du pixel (0..height)
/// * `center_x` - Centre X du plan complexe
/// * `center_y` - Centre Y du plan complexe
/// * `span_x` - Étendue X du plan complexe
/// * `span_y` - Étendue Y du plan complexe
/// * `width` - Largeur de l'image en pixels
/// * `height` - Hauteur de l'image en pixels
/// 
/// # Returns
/// DualComplex représentant C + c dans le plan complexe avec dérivées préservées
pub fn transform_pixel_to_complex(
    pixel_re: f64,
    pixel_im: f64,
    center_x: f64,
    center_y: f64,
    span_x: f64,
    span_y: f64,
    width: f64,
    height: f64,
) -> DualComplex {
    // Coordonnées pixel normalisées (0..1)
    let norm_re = pixel_re / width;
    let norm_im = pixel_im / height;
    
    // Offset depuis le centre (-0.5..0.5)
    let offset_re = norm_re - 0.5;
    let offset_im = norm_im - 0.5;
    
    // Coordonnées complexes
    let c_re = center_x + offset_re * span_x;
    let c_im = center_y + offset_im * span_y;
    
    // Dérivées: dc/dpixel_re = span_x / width, dc/dpixel_im = span_y / height
    let dc_dre = span_x / width;
    let dc_dim = span_y / height;
    
    DualComplex {
        value: Complex64::new(c_re, c_im),
        dual_re: Complex64::new(dc_dre, 0.0),
        dual_im: Complex64::new(0.0, dc_dim),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dual_complex_add() {
        let a = DualComplex {
            value: Complex64::new(1.0, 2.0),
            dual_re: Complex64::new(3.0, 4.0),
            dual_im: Complex64::new(5.0, 6.0),
        };
        let b = DualComplex {
            value: Complex64::new(7.0, 8.0),
            dual_re: Complex64::new(9.0, 10.0),
            dual_im: Complex64::new(11.0, 12.0),
        };
        let sum = a.add(b);
        assert!((sum.value.re - 8.0).abs() < 1e-10);
        assert!((sum.value.im - 10.0).abs() < 1e-10);
        assert!((sum.dual_re.re - 12.0).abs() < 1e-10);
        assert!((sum.dual_re.im - 14.0).abs() < 1e-10);
    }

    #[test]
    fn dual_complex_mul() {
        let a = DualComplex {
            value: Complex64::new(1.0, 2.0),
            dual_re: Complex64::new(1.0, 0.0),
            dual_im: Complex64::new(0.0, 1.0),
        };
        let b = DualComplex {
            value: Complex64::new(3.0, 4.0),
            dual_re: Complex64::new(1.0, 0.0),
            dual_im: Complex64::new(0.0, 1.0),
        };
        let prod = a.mul(b);
        // (1+2i)*(3+4i) = -5+10i
        assert!((prod.value.re - (-5.0)).abs() < 1e-10);
        assert!((prod.value.im - 10.0).abs() < 1e-10);
    }

    #[test]
    fn dual_complex_square() {
        let z = DualComplex {
            value: Complex64::new(2.0, 3.0),
            dual_re: Complex64::new(1.0, 0.0),
            dual_im: Complex64::new(0.0, 1.0),
        };
        let sq = z.square();
        // (2+3i)² = -5+12i
        assert!((sq.value.re - (-5.0)).abs() < 1e-10);
        assert!((sq.value.im - 12.0).abs() < 1e-10);
        // d(z²)/dz = 2z, donc dual_re = 2*(2+3i)*1 = 4+6i
        assert!((sq.dual_re.re - 4.0).abs() < 1e-10);
        assert!((sq.dual_re.im - 6.0).abs() < 1e-10);
    }
}
