//! Table BLA à coefficients **double-double** (~106 bits) pour le tier dd
//! Mandelbrot (`pixel_loop_dd`).
//!
//! La BLA f64 (`bla_dual::BlaTableUnified`) stocke `A`/`B` en `Mat2<f64>` :
//! appliquer un pas BLA `δ' = A·δ + B·dc` avec `A` f64 arrondit `δ` à 53 bits,
//! ré-introduisant le plancher que le tier dd cherche à lever — c'est pourquoi
//! `pixel_loop_dd` itérait SANS BLA (correct mais ~4-10× plus lent). Cette table
//! stocke `A`/`B` en `Mat2Dd` (dd) : le pas BLA reste ~106 bits, on retrouve le
//! skip d'itérations sans perte de précision (équivalent du BLA float128 de F3).
//!
//! Spécialisé Mandelbrot (phase `[Sqr, Add]`) — le seul type du tier dd :
//! single-step `A = 2·[Zx,-Zy; Zy,Zx]`, `r² = (ε·|Z|)²` (cf.
//! `bla_dual::build_bla_single_step`, cas Sqr avec jac initial = I). Le merge et
//! le lookup multi-niveaux miment `BlaTableUnified` ; seuls les coefficients
//! passent en dd (les rayons `r²` restent f64 — magnitudes).

use crate::fractal::bytecode::bla_dual::BLA_SKIP_LEVELS;
use crate::fractal::perturbation::dd::{ComplexDDExp, DoubleDouble};
use crate::fractal::perturbation::types::FloatExp;

/// Matrice 2×2 à coefficients `DoubleDouble` (~106 b). Range f64 (pas
/// d'exposant séparé) : suffit car les `A` réellement CONSULTÉS restent bornés
/// (`|A| ≲ 1/|δ|`), les merges hauts qui débordent saturent à ±inf → `r²=0` →
/// rejetés au lookup (comme la BLA f64).
#[derive(Clone, Copy, Debug)]
pub struct Mat2Dd {
    pub m00: DoubleDouble,
    pub m01: DoubleDouble,
    pub m10: DoubleDouble,
    pub m11: DoubleDouble,
}

impl Mat2Dd {
    pub const IDENTITY: Self = Self {
        m00: DoubleDouble { hi: 1.0, lo: 0.0 },
        m01: DoubleDouble::ZERO,
        m10: DoubleDouble::ZERO,
        m11: DoubleDouble { hi: 1.0, lo: 0.0 },
    };

    /// Produit matriciel `self · rhs` en dd.
    #[inline]
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            m00: self.m00.mul(rhs.m00).add(self.m01.mul(rhs.m10)),
            m01: self.m00.mul(rhs.m01).add(self.m01.mul(rhs.m11)),
            m10: self.m10.mul(rhs.m00).add(self.m11.mul(rhs.m10)),
            m11: self.m10.mul(rhs.m01).add(self.m11.mul(rhs.m11)),
        }
    }

    /// Somme matricielle en dd.
    #[inline]
    pub fn add(self, rhs: Self) -> Self {
        Self {
            m00: self.m00.add(rhs.m00),
            m01: self.m01.add(rhs.m01),
            m10: self.m10.add(rhs.m10),
            m11: self.m11.add(rhs.m11),
        }
    }

    /// Norme opérateur 2 (plus grande valeur singulière), calculée en f64 sur
    /// les composantes hautes — c'est une magnitude pour les rayons `r²`, la
    /// précision dd y est superflue.
    #[inline]
    pub fn sup_norm_f64(self) -> f64 {
        let (a, b, c, d) = (
            self.m00.to_f64(),
            self.m01.to_f64(),
            self.m10.to_f64(),
            self.m11.to_f64(),
        );
        let mtm_00 = a * a + c * c;
        let mtm_01 = a * b + c * d;
        let mtm_11 = b * b + d * d;
        let trace = mtm_00 + mtm_11;
        let det = mtm_00 * mtm_11 - mtm_01 * mtm_01;
        let disc = (trace * trace - 4.0 * det).max(0.0);
        ((trace + disc.sqrt()) * 0.5).max(0.0).sqrt()
    }
}

/// Nœud BLA dd : `δ' = A·δ + B·dc`, valide pour `|δ|² < r2`, saute `l` iters.
#[derive(Clone, Copy, Debug)]
pub struct BlaStepDd {
    pub a: Mat2Dd,
    pub b: Mat2Dd,
    pub r2: f64,
    pub l: u32,
}

impl BlaStepDd {
    /// Single-step Mandelbrot au point de référence `Z` (dd) : `A = 2·[Zx,-Zy;
    /// Zy,Zx]`, `B = I`, `r² = (ε·|Z|)²` (jac initial = I ⇒ sup=1).
    #[inline]
    fn single_mandelbrot(z: ComplexDDExp, epsilon: f64) -> Self {
        let zx = z.re.to_dd();
        let zy = z.im.to_dd();
        let two_zx = zx.mul_f64(2.0);
        let two_zy = zy.mul_f64(2.0);
        let a = Mat2Dd {
            m00: two_zx,
            m01: two_zy.neg(),
            m10: two_zy,
            m11: two_zx,
        };
        // |Z| en f64 (Z borné O(1)) pour le rayon.
        let zf = z.to_complex64_approx();
        let r = epsilon * (zf.re * zf.re + zf.im * zf.im).sqrt();
        Self {
            a,
            b: Mat2Dd::IDENTITY,
            r2: r * r,
            l: 1,
        }
    }

    /// Compose deux BLAs adjacents `T_z = T_y ∘ T_x` (cf. `BlaMultiStep::merge`,
    /// mêmes formules F3 `bla.h:33-37`, coefficients en dd, rayons en f64).
    fn merge(x: Self, y: Self, c: f64) -> Self {
        let az = y.a.mul(x.a);
        let bz = y.a.mul(x.b).add(y.b);
        let sup_ax = x.a.sup_norm_f64();
        let sup_bx = x.b.sup_norm_f64();
        let rx = x.r2.sqrt();
        let ry = y.r2.sqrt();
        let rz = if sup_ax < 1e-20 {
            rx.min(ry).max(0.0)
        } else {
            let inner = (ry - sup_bx * c).max(0.0) / sup_ax;
            rx.min(inner).max(0.0)
        };
        Self {
            a: az,
            b: bz,
            r2: rz * rz,
            l: x.l + y.l,
        }
    }
}

/// Table BLA dd multi-niveaux (mirror de `BlaTableUnified`, coefficients dd).
#[derive(Clone, Debug)]
pub struct BlaTableDd {
    pub levels: Vec<Vec<BlaStepDd>>,
}

impl BlaTableDd {
    /// Construit la table pour une orbite référence dd Mandelbrot. `c_norm` =
    /// rayon image en espace-c (`max |δc|`) ; `epsilon` = facteur de précision.
    pub fn build_mandelbrot(z_ref_dd: &[ComplexDDExp], c_norm: f64, epsilon: f64) -> Self {
        let m = z_ref_dd.len().saturating_sub(1);
        if m == 0 {
            return Self { levels: Vec::new() };
        }
        let level0: Vec<BlaStepDd> = (0..m)
            .map(|i| BlaStepDd::single_mandelbrot(z_ref_dd[i], epsilon))
            .collect();
        let mut levels = vec![level0];
        while levels.last().unwrap().len() > 1 {
            let prev = levels.last().unwrap();
            let mut next: Vec<BlaStepDd> = Vec::with_capacity(prev.len().div_ceil(2));
            let mut i = 0;
            while i + 1 < prev.len() {
                next.push(BlaStepDd::merge(prev[i], prev[i + 1], c_norm));
                i += 2;
            }
            if i < prev.len() {
                next.push(prev[i]);
            }
            levels.push(next);
        }
        // Vider les niveaux < BLA_SKIP_LEVELS (jamais consultés — cf. bla_dual).
        for l in 0..BLA_SKIP_LEVELS.min(levels.len()) {
            levels[l] = Vec::new();
            levels[l].shrink_to_fit();
        }
        Self { levels }
    }

    /// Lookup FloatExp-aware (mirror de `BlaTableUnified::lookup_fexp`) : plus
    /// grand `l` valide à partir de `m` quand `|δ|² < r²`. `None` sinon.
    pub fn lookup_fexp(&self, m: usize, delta_norm_sqr_fexp: FloatExp) -> Option<&BlaStepDd> {
        let nlevels = self.levels.len();
        if nlevels <= BLA_SKIP_LEVELS {
            return None;
        }
        let top = if m == 0 {
            nlevels - 1
        } else {
            (m.trailing_zeros() as usize).min(nlevels - 1)
        };
        let mut level = top;
        while level >= BLA_SKIP_LEVELS {
            let nodes = &self.levels[level];
            let idx = m >> level;
            if idx < nodes.len() {
                let node = &nodes[idx];
                let r2_fexp = FloatExp::from_f64(node.r2);
                if delta_norm_sqr_fexp < r2_fexp {
                    return Some(node);
                }
            }
            if level == 0 {
                break;
            }
            level -= 1;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::perturbation::dd::DoubleDoubleExp;
    use num_complex::Complex64;

    fn cdd(re: f64, im: f64) -> ComplexDDExp {
        ComplexDDExp {
            re: DoubleDoubleExp::from_f64(re),
            im: DoubleDoubleExp::from_f64(im),
        }
    }

    /// Single-step Mandelbrot : A = 2·[Zx,-Zy; Zy,Zx].
    #[test]
    fn single_step_matches_2z() {
        let z = cdd(0.3, -0.4);
        let s = BlaStepDd::single_mandelbrot(z, 1e-6);
        assert!((s.a.m00.to_f64() - 0.6).abs() < 1e-12);
        assert!((s.a.m01.to_f64() - 0.8).abs() < 1e-12); // -2·(-0.4) = 0.8
        assert!((s.a.m10.to_f64() - (-0.8)).abs() < 1e-12);
        assert!((s.a.m11.to_f64() - 0.6).abs() < 1e-12);
        assert!(s.r2 > 0.0 && s.r2.is_finite());
        assert_eq!(s.l, 1);
    }

    /// Merge de 2 single-steps Mandelbrot : A = 4·Z0·Z1 (mult complexe), l=2.
    /// Compare à la BLA f64 (`BlaMultiStep::merge`) — mêmes formules.
    #[test]
    fn merge_matches_f64_bla() {
        use crate::fractal::bytecode::bla_dual::{build_bla_single_step, BlaMultiStep};
        use crate::fractal::bytecode::compile_formula;
        use crate::fractal::FractalType;

        let z0 = Complex64::new(0.2, 0.1);
        let z1 = Complex64::new(-0.3, 0.4);
        // dd
        let sd0 = BlaStepDd::single_mandelbrot(cdd(z0.re, z0.im), 1e-6);
        let sd1 = BlaStepDd::single_mandelbrot(cdd(z1.re, z1.im), 1e-6);
        let md = BlaStepDd::merge(sd0, sd1, 0.5);
        // f64
        let phase = &compile_formula(FractalType::Mandelbrot, 2.0).unwrap().phases[0];
        let mf = BlaMultiStep::merge(
            // z_land arbitraire : non lu par ce test (compare A/B/r² seulement).
            BlaMultiStep::from_single(build_bla_single_step(z0.re, z0.im, phase, 1e-6), z1),
            BlaMultiStep::from_single(build_bla_single_step(z1.re, z1.im, phase, 1e-6), z0),
            0.5,
        );
        assert_eq!(md.l, 2);
        assert!((md.a.m00.to_f64() - mf.a.m00).abs() < 1e-12);
        assert!((md.a.m01.to_f64() - mf.a.m01).abs() < 1e-12);
        assert!((md.b.m00.to_f64() - mf.b.m00).abs() < 1e-12);
        assert!((md.r2 - mf.r2).abs() < 1e-12 * mf.r2.max(1.0));
    }

    /// Table sur une orbite Mandelbrot : structure niveaux + lookup cohérent.
    #[test]
    fn table_build_and_lookup() {
        let c = Complex64::new(-0.7, 0.3);
        let mut z = Complex64::new(0.0, 0.0);
        let mut orbit = vec![cdd(0.0, 0.0)];
        for _ in 0..40 {
            z = z * z + c;
            orbit.push(cdd(z.re, z.im));
        }
        let table = BlaTableDd::build_mandelbrot(&orbit, 1e-9, 1e-6);
        // m=8 aligné niveau 3 : delta minuscule → BLA l ≥ 8.
        let tiny = FloatExp::from_f64(1e-30);
        let res = table.lookup_fexp(8, tiny);
        assert!(res.is_some());
        assert!(res.unwrap().l >= 8);
        // delta énorme → aucun BLA valide.
        assert!(table.lookup_fexp(8, FloatExp::from_f64(1e30)).is_none());
    }
}
