//! Mirror **FloatExp** (mantisse + exposant) de `bla_dual.rs`.
//!
//! Construction BLA unifiée via dual-numbers walking le bytecode, mais avec
//! toute l'arithmétique en `FloatExp`/`ComplexExp` au lieu de `f64`/`Complex64`.
//! Nécessaire pour le path atom-tronqué deep zoom (`FRACTALL_ATOM_PERIOD=1`) où
//! les valeurs de graze (~1e-8000) et les coefficients BLA `A=∏2Z` (~1e444)
//! under/overflow f64 → la BLA f64 produit des coefficients faux.
//!
//! ⚠️ Mirror MÉCANIQUE fidèle : mêmes formules, mêmes constantes
//! (`BLA_SKIP_LEVELS`), même math merge/validité que `bla_dual.rs`. Voir ce
//! fichier pour les explications algorithmiques détaillées.
//!
//! N'est atteignable que quand `FRACTALL_ATOM_PERIOD=1` (construction gated dans
//! `delta.rs`). Ne touche pas le path f64 par défaut.

use super::bla_dual::BLA_SKIP_LEVELS;
use super::{Op, Phase};
use crate::fractal::perturbation::types::{ComplexExp, FloatExp};

/// Négation FloatExp (pas de `Neg` sur FloatExp ; `new` re-normalise et gère
/// `-0.0` → zéro).
#[inline(always)]
fn fneg(f: FloatExp) -> FloatExp {
    FloatExp::new(-f.mantissa, f.exponent)
}

/// Matrice 2×2 en FloatExp (mirror de `bla_dual::Mat2`).
#[derive(Clone, Copy, Debug)]
pub struct Mat2Exp {
    pub m00: FloatExp,
    pub m01: FloatExp,
    pub m10: FloatExp,
    pub m11: FloatExp,
}

impl Mat2Exp {
    #[allow(dead_code)]
    pub fn zero() -> Self {
        let z = FloatExp::zero();
        Self { m00: z, m01: z, m10: z, m11: z }
    }

    pub fn identity() -> Self {
        let one = FloatExp::from_f64(1.0);
        let z = FloatExp::zero();
        Self { m00: one, m01: z, m10: z, m11: one }
    }

    /// Multiplication matricielle `self · rhs`.
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            m00: self.m00 * rhs.m00 + self.m01 * rhs.m10,
            m01: self.m00 * rhs.m01 + self.m01 * rhs.m11,
            m10: self.m10 * rhs.m00 + self.m11 * rhs.m10,
            m11: self.m10 * rhs.m01 + self.m11 * rhs.m11,
        }
    }

    /// Multiplication par scalaire FloatExp.
    pub fn scale(self, s: FloatExp) -> Self {
        Self {
            m00: self.m00 * s,
            m01: self.m01 * s,
            m10: self.m10 * s,
            m11: self.m11 * s,
        }
    }

    /// Norme opérateur 2 (plus grande valeur singulière). Même formule fermée
    /// 2×2 que `bla_dual::Mat2::sup_norm`, en FloatExp.
    pub fn sup_norm(self) -> FloatExp {
        let zero = FloatExp::zero();
        let mtm_00 = self.m00 * self.m00 + self.m10 * self.m10;
        let mtm_01 = self.m00 * self.m01 + self.m10 * self.m11;
        let mtm_11 = self.m01 * self.m01 + self.m11 * self.m11;
        let trace = mtm_00 + mtm_11;
        let det = mtm_00 * mtm_11 - mtm_01 * mtm_01;
        let disc = (trace * trace - 4.0 * det).max(zero);
        ((trace + disc.sqrt()) * 0.5).max(zero).sqrt()
    }
}

/// État dual FloatExp : `value` (point z courant) + `jac` (∂z/∂δ).
#[derive(Clone, Copy, Debug)]
pub struct DualComplex2Exp {
    pub value: ComplexExp,
    pub jac: Mat2Exp,
}

impl DualComplex2Exp {
    /// Initialise comme `z = z0 + δ` (jac = I).
    pub fn from_value(z0: ComplexExp) -> Self {
        Self { value: z0, jac: Mat2Exp::identity() }
    }

    /// `z := z²` (multiplication complexe). Chain rule identique à `DualComplex2`.
    fn sqr(&mut self) {
        let x = self.value.re;
        let y = self.value.im;
        let two = FloatExp::from_f64(2.0);
        let m = Mat2Exp { m00: x, m01: fneg(y), m10: y, m11: x }.scale(two);
        self.jac = m.mul(self.jac);
        let new_x = x * x - y * y;
        let new_y = x * y * 2.0;
        self.value = ComplexExp { re: new_x, im: new_y };
    }

    /// `z := z · stored` (multiplication complexe, règle de Leibniz).
    fn mul(&mut self, stored: &DualComplex2Exp) {
        let zx = self.value.re;
        let zy = self.value.im;
        let sx = stored.value.re;
        let sy = stored.value.im;
        let ms = Mat2Exp { m00: sx, m01: fneg(sy), m10: sy, m11: sx };
        let mz = Mat2Exp { m00: zx, m01: fneg(zy), m10: zy, m11: zx };
        let jac_a = ms.mul(self.jac);
        let jac_b = mz.mul(stored.jac);
        self.jac = Mat2Exp {
            m00: jac_a.m00 + jac_b.m00,
            m01: jac_a.m01 + jac_b.m01,
            m10: jac_a.m10 + jac_b.m10,
            m11: jac_a.m11 + jac_b.m11,
        };
        let new_x = zx * sx - zy * sy;
        let new_y = zx * sy + zy * sx;
        self.value = ComplexExp { re: new_x, im: new_y };
    }

    /// `z.re := |z.re|`. Si re < 0, flip la ligne 0 de la Jacobienne.
    fn abs_x(&mut self) {
        if self.value.re.mantissa < 0.0 {
            self.value.re = fneg(self.value.re);
            self.jac.m00 = fneg(self.jac.m00);
            self.jac.m01 = fneg(self.jac.m01);
        }
    }

    fn abs_y(&mut self) {
        if self.value.im.mantissa < 0.0 {
            self.value.im = fneg(self.value.im);
            self.jac.m10 = fneg(self.jac.m10);
            self.jac.m11 = fneg(self.jac.m11);
        }
    }

    fn neg_x(&mut self) {
        self.value.re = fneg(self.value.re);
        self.jac.m00 = fneg(self.jac.m00);
        self.jac.m01 = fneg(self.jac.m01);
    }

    fn neg_y(&mut self) {
        self.value.im = fneg(self.value.im);
        self.jac.m10 = fneg(self.jac.m10);
        self.jac.m11 = fneg(self.jac.m11);
    }

    /// `z := z · (cos + sin·i)`. Rotation linéaire (isométrique). `cos`/`sin`
    /// restent f64 (issus de `Op::Rot`).
    fn rot(&mut self, cos_theta: f64, sin_theta: f64) {
        let c = cos_theta;
        let s = sin_theta;
        let new_x = c * self.value.re - s * self.value.im;
        let new_y = s * self.value.re + c * self.value.im;
        self.value = ComplexExp { re: new_x, im: new_y };
        let jac = self.jac;
        self.jac = Mat2Exp {
            m00: c * jac.m00 - s * jac.m10,
            m01: c * jac.m01 - s * jac.m11,
            m10: s * jac.m00 + c * jac.m10,
            m11: s * jac.m01 + c * jac.m11,
        };
    }

    fn abs_value(&self) -> FloatExp {
        (self.value.re.sqr() + self.value.im.sqr()).sqrt()
    }
}

/// BLA single-step FloatExp (mirror de `BlaSingleStep`).
#[derive(Clone, Copy, Debug)]
pub struct BlaSingleStepExp {
    pub a: Mat2Exp,
    pub r2: FloatExp,
}

/// `min` d'un rayon `Option<FloatExp>` (None = +∞) avec un candidat.
#[inline(always)]
fn min_radius(r: Option<FloatExp>, candidate: FloatExp) -> FloatExp {
    match r {
        Some(existing) => existing.min(candidate),
        None => candidate,
    }
}

/// Degré de la phase (produit des puissances). Port de F3 `opcodes_degree`
/// (`param.cc`). Mandelbrot `[Sqr, Add]` → 2.
fn phase_degree(phase: &Phase) -> i64 {
    let mut deg_stored: i64 = 0;
    let mut deg: i64 = 1;
    for op in &phase.ops {
        match op {
            Op::Store => deg_stored = deg,
            Op::Mul => deg += deg_stored,
            Op::Sqr => deg <<= 1,
            _ => {}
        }
    }
    deg
}

/// Construit le BLA single-step FloatExp en walking le bytecode. Mirror de
/// `build_bla_single_step`, mais le **rayon de validité initial** suit F3
/// (`hybrid.h:161`) : `r = e·|Z|·degree / (degree·(degree-1)/2)` au lieu de
/// l'`INFINITY` de `bla_dual.rs`.
///
/// ⚠️ DÉVIATION VOLONTAIRE vs le mirror f64 : `bla_dual.rs` initialise `r` à
/// l'infini et ne survit qu'au grâce à l'underflow f64 de `W` → 0 aux grazes
/// (~1e-8000) qui annule le rayon. En FloatExp `W` reste ~1e-8000 (non nul) →
/// l'infini laisserait le rayon op-borné (`e/2`, indépendant de |Z|) trop
/// large → BLA sur-skippe au graze (drop de δ² là où 2Zδ≈0) → itérations
/// fausses. Le seed F3 `2·e·|Z|` s'effondre à ~1e-8000 au graze → nœud jamais
/// valide → pas direct → CORRECT. Pour les points normaux (|Z|~O(1)) le seed
/// (~1.2e-7) est dominé par la contrainte op (`e/2`) → aucun changement.
pub fn build_bla_single_step_exp(
    z_ref: ComplexExp,
    phase: &Phase,
    epsilon: FloatExp,
) -> BlaSingleStepExp {
    let two = FloatExp::from_f64(2.0);
    let mut w = DualComplex2Exp::from_value(z_ref);
    let mut stored = w;
    let z_abs = w.abs_value();

    // Rayon de validité initial F3 : `e·|Z|·degree / (degree·(degree-1)/2)`.
    // (`degree·(degree-1)/2` = C(degree,2), clampé ≥ 1 pour degree < 2.)
    let degree = phase_degree(phase);
    let binom = ((degree * (degree - 1)) / 2).max(1);
    let factor = degree as f64 / binom as f64;
    let mut r: Option<FloatExp> = Some((epsilon * z_abs) * factor);

    for op in &phase.ops {
        let w_before = w;
        let stored_before = stored;
        let sup_a_before = w.jac.sup_norm();
        // Le guard f64 `sup_a_before > 1e-300` devient : sup_a_before non nul
        // (FloatExp ne underflow pas → seul le zéro exact est à écarter).
        let sup_a_nonzero = sup_a_before.mantissa != 0.0;

        match op {
            Op::Sqr => {
                if sup_a_nonzero {
                    let candidate = (epsilon * w_before.abs_value()).div(sup_a_before);
                    r = Some(min_radius(r, candidate));
                }
                w.sqr();
            }
            Op::Mul => {
                if sup_a_nonzero {
                    let m = w_before.abs_value().min(stored_before.abs_value());
                    let candidate = (epsilon * m).div(sup_a_before);
                    r = Some(min_radius(r, candidate));
                }
                w.mul(&stored);
            }
            Op::Store => {
                stored = w;
            }
            Op::AbsX => {
                if sup_a_nonzero {
                    let candidate = w_before.value.re.abs().div(two).div(sup_a_before);
                    r = Some(min_radius(r, candidate));
                }
                w.abs_x();
            }
            Op::AbsY => {
                if sup_a_nonzero {
                    let candidate = w_before.value.im.abs().div(two).div(sup_a_before);
                    r = Some(min_radius(r, candidate));
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
                let r2 = match r {
                    Some(rr) => rr.sqr(),
                    None => FloatExp::zero(),
                };
                return BlaSingleStepExp { a: w.jac, r2 };
            }
            Op::Rot { cos_theta, sin_theta } => {
                w.rot(*cos_theta, *sin_theta);
            }
        }
    }

    BlaSingleStepExp { a: w.jac, r2: FloatExp::zero() }
}

/// BLA multi-step FloatExp (mirror de `BlaMultiStep`).
#[derive(Clone, Copy, Debug)]
pub struct BlaMultiStepExp {
    pub a: Mat2Exp,
    pub b: Mat2Exp,
    pub r2: FloatExp,
    pub l: u32,
}

impl BlaMultiStepExp {
    /// Promotion d'un single-step (avec `B = I`, `l = 1`).
    pub fn from_single(s: BlaSingleStepExp) -> Self {
        Self {
            a: s.a,
            b: Mat2Exp::identity(),
            r2: s.r2,
            l: 1,
        }
    }

    /// Compose deux BLAs adjacents `T_z = T_y ∘ T_x`. Formules F3 identiques à
    /// `BlaMultiStep::merge`, arithmétique FloatExp.
    pub fn merge(x: BlaMultiStepExp, y: BlaMultiStepExp, c: FloatExp) -> Self {
        let zero = FloatExp::zero();
        let az_x = y.a.mul(x.a);
        let bz = {
            let ay_bx = y.a.mul(x.b);
            Mat2Exp {
                m00: ay_bx.m00 + y.b.m00,
                m01: ay_bx.m01 + y.b.m01,
                m10: ay_bx.m10 + y.b.m10,
                m11: ay_bx.m11 + y.b.m11,
            }
        };
        let sup_ax = x.a.sup_norm();
        let sup_bx = x.b.sup_norm();
        let rx = x.r2.sqrt();
        let ry = y.r2.sqrt();
        // Formule F3 EXACTE (`bla.h:merge`) : `r = min(rx, max(0, (ry − xB·c)/xA))`.
        // ⚠️ DÉVIATION vs `bla_dual.rs` : celui-ci ajoute un guard `xA < 1e-20 →
        // min(rx,ry)` (hack anti-div-par-zéro f64 quand `xA` underflow). En
        // FloatExp `xA` ne underflow jamais → on suit F3 sans guard (la division
        // par un `xA` contractant < 1 GONFLE le rayon, ce qui est correct : un
        // pas x contractant admet un delta d'entrée plus grand). Le guard f64
        // faussait le rayon aux grazes (xA ~ 1e-9991) → over-skip par-dessus les
        // points de rebase. On ne garde que la garde `xA == 0` (matrice A nulle
        // exacte, ex. itér. 0 Z=0) pour éviter inf.
        let inner = if sup_ax.mantissa == 0.0 {
            rx
        } else {
            (ry - sup_bx * c).max(zero).div(sup_ax)
        };
        let rz = rx.min(inner);
        Self {
            a: az_x,
            b: bz,
            r2: rz.sqr(),
            l: x.l + y.l,
        }
    }
}

/// Table BLA unifiée multi-niveaux FloatExp (mirror de `BlaTableUnified`).
#[derive(Clone, Debug)]
pub struct BlaTableUnifiedExp {
    pub levels: Vec<Vec<BlaMultiStepExp>>,
}

impl BlaTableUnifiedExp {
    /// Construit la table BLA FloatExp pour une phase à partir de l'orbite
    /// référence `ComplexExp`. Mirror serial de `BlaTableUnified::build`
    /// (même structure de niveaux, même free des bas niveaux `< BLA_SKIP_LEVELS`).
    pub fn build(
        z_ref: &[ComplexExp],
        phase: &Phase,
        c_norm: FloatExp,
        epsilon: FloatExp,
    ) -> Self {
        let m = z_ref.len().saturating_sub(1);
        if m == 0 {
            return Self { levels: Vec::new() };
        }

        // Level 0 : un single-step par itération de référence.
        let level0: Vec<BlaMultiStepExp> = (0..m)
            .map(|i| BlaMultiStepExp::from_single(build_bla_single_step_exp(z_ref[i], phase, epsilon)))
            .collect();
        let mut levels = vec![level0];

        // Niveaux supérieurs : merge adjacents (tail impair promu tel quel).
        while levels.last().unwrap().len() > 1 {
            let prev = levels.last().unwrap();
            let n_pairs = prev.len() / 2;
            let mut next: Vec<BlaMultiStepExp> = (0..n_pairs)
                .map(|k| BlaMultiStepExp::merge(prev[2 * k], prev[2 * k + 1], c_norm))
                .collect();
            if prev.len() % 2 == 1 {
                next.push(prev[prev.len() - 1]);
            }
            levels.push(next);
        }

        // Libère les niveaux < BLA_SKIP_LEVELS (jamais consultés, cf. bla_dual).
        for l in 0..BLA_SKIP_LEVELS.min(levels.len()) {
            levels[l] = Vec::new();
            levels[l].shrink_to_fit();
        }
        Self { levels }
    }

    /// Cherche le BLA valide (le plus grand `l`) à partir de `m` quand
    /// `delta_norm_sqr < r2`. Mirror de `BlaTableUnified::lookup_fexp`, `r2` déjà
    /// en FloatExp.
    pub fn lookup_fexp(
        &self,
        m: usize,
        delta_norm_sqr_fexp: FloatExp,
    ) -> Option<&BlaMultiStepExp> {
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
                if delta_norm_sqr_fexp < node.r2 {
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
    use crate::fractal::bytecode::compile_formula;
    use crate::fractal::FractalType;
    use num_complex::Complex64;

    fn cx(re: f64, im: f64) -> ComplexExp {
        ComplexExp::from_complex64(Complex64::new(re, im))
    }

    fn close(a: FloatExp, b: f64, tol: f64) -> bool {
        (a.to_f64() - b).abs() < tol
    }

    /// Mandelbrot : A = 2·[Zx, -Zy; Zy, Zx] — même oracle que le sibling f64.
    #[test]
    fn mandelbrot_bla_matches_complex_2z() {
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let (zx, zy) = (0.3, -0.4);
        let eps = FloatExp::from_f64(1e-6);
        let bla = build_bla_single_step_exp(cx(zx, zy), &formula.phases[0], eps);
        assert!(close(bla.a.m00, 2.0 * zx, 1e-10));
        assert!(close(bla.a.m01, -2.0 * zy, 1e-10));
        assert!(close(bla.a.m10, 2.0 * zy, 1e-10));
        assert!(close(bla.a.m11, 2.0 * zx, 1e-10));
    }

    /// Identity sup_norm == 1.
    #[test]
    fn mat2exp_sup_norm_identity() {
        assert!(close(Mat2Exp::identity().sup_norm(), 1.0, 1e-10));
    }

    #[test]
    fn mat2exp_sup_norm_scaled() {
        let m = Mat2Exp::identity().scale(FloatExp::from_f64(3.5));
        assert!(close(m.sup_norm(), 3.5, 1e-10));
    }

    /// La construction de table doit produire les mêmes niveaux que le f64,
    /// avec les bas niveaux vidés.
    #[test]
    fn table_build_levels_8_iterations() {
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let phase = &formula.phases[0];
        let orbit: Vec<ComplexExp> = (0..9).map(|i| cx(i as f64 * 0.1, 0.0)).collect();
        let table = BlaTableUnifiedExp::build(
            &orbit,
            phase,
            FloatExp::from_f64(0.5),
            FloatExp::from_f64(1e-6),
        );
        assert_eq!(table.levels.len(), 4);
        for l in 0..BLA_SKIP_LEVELS.min(4) {
            assert_eq!(table.levels[l].len(), 0);
        }
        assert_eq!(table.levels[3].len(), 1);
        assert_eq!(table.levels[3][0].l, 8);
    }

    /// Merge de 2 single-steps Mandelbrot : A_merged = 4·Z_0·Z_1 (mat2 complexe),
    /// bit-parité avec l'oracle f64.
    #[test]
    fn merge_mandelbrot_two_steps() {
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let phase = &formula.phases[0];
        let z0 = Complex64::new(0.2, 0.1);
        let z1 = Complex64::new(-0.3, 0.4);
        let eps = FloatExp::from_f64(1e-6);
        let s0 = build_bla_single_step_exp(cx(z0.re, z0.im), phase, eps);
        let s1 = build_bla_single_step_exp(cx(z1.re, z1.im), phase, eps);
        let merged = BlaMultiStepExp::merge(
            BlaMultiStepExp::from_single(s0),
            BlaMultiStepExp::from_single(s1),
            FloatExp::from_f64(0.5),
        );
        let z0z1 = z0 * z1;
        assert!(close(merged.a.m00, 4.0 * z0z1.re, 1e-10));
        assert!(close(merged.a.m01, -4.0 * z0z1.im, 1e-10));
        assert!(close(merged.a.m10, 4.0 * z0z1.im, 1e-10));
        assert!(close(merged.a.m11, 4.0 * z0z1.re, 1e-10));
        assert_eq!(merged.l, 2);
    }

    /// Le lookup retourne un BLA de niveau ≥ BLA_SKIP_LEVELS pour un delta
    /// minuscule à m aligné, et None pour un delta énorme.
    #[test]
    fn lookup_returns_valid() {
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let phase = &formula.phases[0];
        let c = Complex64::new(-0.7, 0.3);
        let mut z = Complex64::new(0.0, 0.0);
        let mut orbit = vec![cx(0.0, 0.0)];
        for _ in 0..40 {
            z = z * z + c;
            orbit.push(cx(z.re, z.im));
        }
        let table = BlaTableUnifiedExp::build(
            &orbit,
            phase,
            FloatExp::from_f64(1e-9),
            FloatExp::from_f64(1e-6),
        );
        let huge = FloatExp::new(1.0, 100);
        assert!(table.lookup_fexp(8, huge).is_none());
        let tiny = FloatExp::new(1.0, -100);
        let res = table.lookup_fexp(8, tiny);
        assert!(res.is_some());
        assert!(res.unwrap().l >= 8);
        assert!(table.lookup_fexp(1, tiny).is_none());
    }
}
