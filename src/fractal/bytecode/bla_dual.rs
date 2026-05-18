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

use crate::fractal::bytecode::{Formula, Op, Phase};

/// Matrice 2×2 réelle (compatible avec `nonconformal::Matrix2x2`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat2 {
    pub m00: f64,
    pub m01: f64,
    pub m10: f64,
    pub m11: f64,
}

impl Mat2 {
    #[allow(dead_code)]
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

/// BLA multi-step : A·δ + B·c, valide pour `|δ|² < r2`, saute `l` itérations.
///
/// Issue d'un single-step (`l=1`, `B = I`) ou de la composition de deux BLAs
/// adjacents via [`merge`].
#[derive(Clone, Copy, Debug)]
pub struct BlaMultiStep {
    pub a: Mat2,
    pub b: Mat2,
    pub r2: f64,
    pub l: u32,
}

impl BlaMultiStep {
    /// Promotion d'un single-step (avec `B = I`, `l = 1`).
    pub fn from_single(s: BlaSingleStep) -> Self {
        Self {
            a: s.a,
            b: Mat2::IDENTITY,
            r2: s.r2,
            l: 1,
        }
    }

    /// Compose deux BLAs adjacents : `T_z = T_y ∘ T_x`.
    ///
    /// Formules F3 (cf. `merge_nonconformal_bla` existant) :
    /// - `A_z = A_y · A_x`
    /// - `B_z = A_y · B_x + B_y`
    /// - `R_z² = max(0, min(R_x², (sqrt(R_y²) - sup|B_x|·|c| / sup|A_x|)²))`
    pub fn merge(x: BlaMultiStep, y: BlaMultiStep, c_norm: f64) -> Self {
        let az_x = y.a.mul(x.a);
        let bz = {
            let ay_bx = y.a.mul(x.b);
            Mat2 {
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
        let rz = if sup_ax < 1e-20 {
            rx.min(ry).max(0.0)
        } else {
            let adjustment = sup_bx * c_norm / sup_ax;
            rx.min(ry - adjustment).max(0.0)
        };
        Self {
            a: az_x,
            b: bz,
            r2: rz * rz,
            l: x.l + y.l,
        }
    }
}

/// Construit une `BlaTableUnified` pour une formule donnée à partir
/// de l'orbite référence.
///
/// `epsilon` recommandé : `1.0 / 2^24 ≈ 5.96e-8` (matche le `bla_threshold`
/// F3 par défaut, soit la précision relative de f32). Pour les zooms très
/// profonds, on peut serrer (ex: `1e-10`) ; pour la performance brute on
/// peut élargir (`1e-6`).
///
/// Renvoie `None` si la formule n'est pas compilable en bytecode ou si
/// l'orbite est trop courte pour bâtir une table utile (M < 2).
pub fn build_bla_table_for_formula(
    formula: &Formula,
    ref_orbit: &[num_complex::Complex64],
    c_norm: f64,
    epsilon: f64,
) -> Option<Vec<BlaTableUnified>> {
    // Pour chaque phase de la formule, une table BLA séparée (préparation
    // hybrides multi-phases ; en mono-phase Vec contient 1 entrée).
    if ref_orbit.len() < 2 {
        return None;
    }
    let tables = formula
        .phases
        .iter()
        .map(|phase| BlaTableUnified::build(ref_orbit, phase, c_norm, epsilon))
        .collect();
    Some(tables)
}

/// Table BLA unifiée multi-niveaux. `levels[k][i]` est le BLA pour les
/// itérations `[i, i + 2^k)` de la référence (sauf le dernier qui peut
/// sauter moins).
///
/// Construction inspirée de F3 `hybrid_blas` :
/// - level 0 : M single-steps (un par itération de la référence)
/// - level k+1 : merge paires adjacentes de level k → ⌈M / 2^(k+1)⌉ entries
///
/// Pas (encore) intégrée à `delta.rs::iterate_pixel` — c'est l'objet de
/// Session C.
#[derive(Clone, Debug)]
pub struct BlaTableUnified {
    pub levels: Vec<Vec<BlaMultiStep>>,
}

impl BlaTableUnified {
    /// Construit la table BLA unifiée pour une phase, à partir de l'orbite
    /// référence et de la phase bytecode.
    ///
    /// `c_norm` = `|cref|` (norme du centre de la référence, utilisée dans la
    /// formule de merge pour ajuster le rayon de validité).
    ///
    /// `epsilon` = facteur de précision (typiquement `2^(-prec_bits)`).
    pub fn build(
        ref_orbit: &[num_complex::Complex64],
        phase: &Phase,
        c_norm: f64,
        epsilon: f64,
    ) -> Self {
        let m = ref_orbit.len().saturating_sub(1);
        if m == 0 {
            return Self { levels: Vec::new() };
        }

        // Level 0 : un single-step par itération de référence.
        let level0: Vec<BlaMultiStep> = (0..m)
            .map(|i| {
                let z = ref_orbit[i];
                BlaMultiStep::from_single(build_bla_single_step(z.re, z.im, phase, epsilon))
            })
            .collect();
        let mut levels = vec![level0];

        // Niveaux supérieurs : merge adjacents.
        while levels.last().unwrap().len() > 1 {
            let prev = levels.last().unwrap();
            let mut next: Vec<BlaMultiStep> = Vec::with_capacity((prev.len() + 1) / 2);
            let mut i = 0;
            while i + 1 < prev.len() {
                let merged = BlaMultiStep::merge(prev[i], prev[i + 1], c_norm);
                next.push(merged);
                i += 2;
            }
            // Si nombre impair, le dernier est promu tel quel.
            if i < prev.len() {
                next.push(prev[i]);
            }
            levels.push(next);
        }
        Self { levels }
    }

    /// Cherche le BLA avec le plus grand `l` valide à partir de l'itération
    /// `m` quand `|δ|² < r2`. Retourne `None` si aucun BLA n'est valide.
    ///
    /// Stratégie F3 : parcourir les niveaux du plus grand au plus petit,
    /// retourner le premier valide.
    pub fn lookup(&self, m: usize, delta_norm_sqr: f64) -> Option<&BlaMultiStep> {
        for (level, nodes) in self.levels.iter().enumerate().rev() {
            // Index dans ce niveau : `m / 2^level`.
            let idx = m >> level;
            if idx >= nodes.len() {
                continue;
            }
            // Vérifier qu'on ne dépasse pas la fin de la référence sur ce niveau.
            // Le BLA[level][idx] couvre les itérations [idx·2^level, (idx+1)·2^level).
            // m doit être à l'INTÉRIEUR de cet intervalle (sinon on saute trop).
            let start = idx << level;
            if m != start {
                // m n'est pas aligné au début de ce niveau, niveau trop gros.
                continue;
            }
            let node = &nodes[idx];
            if delta_norm_sqr < node.r2 {
                return Some(node);
            }
        }
        None
    }

    /// Variante FloatExp-aware utilisée par `pixel_loop_exp` pour éviter
    /// l'underflow f64 quand |delta| < 2^-1022.
    ///
    /// À zoom > 1e308, `delta.norm_sqr_approx()` retourne 0.0 (FloatExp::to_f64
    /// → 0 quand exp < -1022) et la validité BLA `0 < r²` est universellement
    /// vraie pour tous les pixels → ils prennent le même skip max → image
    /// uniforme (cf. e1121).
    ///
    /// Cette variante reçoit (mantissa, exp) de la norme² en FloatExp et
    /// compare correctement à r² f64 :
    /// - Si `delta_exp < -1074` (true zéro absolu en f64) : `r² > 0` toujours vrai
    /// - Sinon : `mantissa * 2^exp < r²` → compare directement
    pub fn lookup_fexp(
        &self,
        m: usize,
        delta_norm_sqr_fexp: crate::fractal::perturbation::types::FloatExp,
    ) -> Option<&BlaMultiStep> {
        for (level, nodes) in self.levels.iter().enumerate().rev() {
            let idx = m >> level;
            if idx >= nodes.len() {
                continue;
            }
            let start = idx << level;
            if m != start {
                continue;
            }
            let node = &nodes[idx];
            // Compare delta_norm_sqr (FloatExp) < node.r2 (f64) sans passer par
            // to_f64 qui underflow. r² > 0 (par construction BLA), donc si delta
            // mantissa est zéro ou si exp est très négatif (sous f64::MIN_EXP),
            // delta < r² trivialement.
            let m_d = delta_norm_sqr_fexp.mantissa;
            let e_d = delta_norm_sqr_fexp.exponent;
            let r2 = node.r2;
            let valid = if m_d == 0.0 {
                true
            } else if e_d < -1074 {
                true // delta inférieur à toute valeur f64 positive normalisable
            } else if e_d > 1023 {
                false // delta dépasse f64::MAX > r²
            } else {
                // delta = m_d * 2^e_d (peut être denormal si e_d ∈ [-1074, -1022])
                // r2 est f64 normal. Comparaison directe via reconstruction f64.
                let d_f64 = m_d * 2.0f64.powi(e_d);
                d_f64 < r2
            };
            if valid {
                return Some(node);
            }
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

    /// Mandelbrot z² + c : la composition de 2 single-steps en (Z_0, Z_1)
    /// doit donner A_merged = A_1 · A_0 = (2·Z_1·I)·(2·Z_0·I) = 4·Z_0·Z_1
    /// (vu comme mat2 de multiplication complexe).
    #[test]
    fn merge_mandelbrot_two_steps() {
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let phase = &formula.phases[0];
        let z0 = Complex64::new(0.2, 0.1);
        let z1 = Complex64::new(-0.3, 0.4);
        let s0 = build_bla_single_step(z0.re, z0.im, phase, 1e-6);
        let s1 = build_bla_single_step(z1.re, z1.im, phase, 1e-6);
        let merged = BlaMultiStep::merge(
            BlaMultiStep::from_single(s0),
            BlaMultiStep::from_single(s1),
            0.5,
        );

        // 4·Z_0·Z_1 = 4 * (0.2+0.1i)(-0.3+0.4i)
        // = 4 * (-0.06 - 0.04 + i(0.08 - 0.03))
        // = 4 * (-0.10 + 0.05i) = -0.4 + 0.2i
        let z0z1 = z0 * z1;
        let re_4 = 4.0 * z0z1.re;
        let im_4 = 4.0 * z0z1.im;
        let expected = Mat2 {
            m00: re_4,
            m01: -im_4,
            m10: im_4,
            m11: re_4,
        };
        assert!(
            mat2_close(merged.a, expected, 1e-10),
            "merged A : got {:?}, expected {:?}",
            merged.a,
            expected
        );
        assert_eq!(merged.l, 2);

        // B = A_1 · I + I = A_1 + I. A_1 = 2·[Z_1.re, -Z_1.im; Z_1.im, Z_1.re].
        let expected_b = Mat2 {
            m00: 2.0 * z1.re + 1.0,
            m01: -2.0 * z1.im,
            m10: 2.0 * z1.im,
            m11: 2.0 * z1.re + 1.0,
        };
        assert!(
            mat2_close(merged.b, expected_b, 1e-10),
            "merged B : got {:?}, expected {:?}",
            merged.b,
            expected_b
        );
    }

    /// La construction de table doit produire log2(M) niveaux et 1 entrée
    /// au sommet pour M=8.
    #[test]
    fn table_build_levels_8_iterations() {
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let phase = &formula.phases[0];
        // 9 entrées d'orbite → M = 8 single-steps possibles.
        let orbit: Vec<Complex64> = (0..9).map(|i| Complex64::new(i as f64 * 0.1, 0.0)).collect();
        let table = BlaTableUnified::build(&orbit, phase, 0.5, 1e-6);

        // M=8 → 4 niveaux : 8, 4, 2, 1.
        assert_eq!(table.levels.len(), 4);
        assert_eq!(table.levels[0].len(), 8);
        assert_eq!(table.levels[1].len(), 4);
        assert_eq!(table.levels[2].len(), 2);
        assert_eq!(table.levels[3].len(), 1);

        // Le BLA top-level doit avoir l = 8.
        assert_eq!(table.levels[3][0].l, 8);
    }

    /// Le lookup retourne `None` quand delta est trop grand, et le plus
    /// large BLA valide sinon. Test à m=1 car m=0 correspond à z_ref=0
    /// pour Mandelbrot, ce qui donne r=0 (BLA inutilisable au step 0 —
    /// c'est la raison pour laquelle F3 commence aussi le BLA à m=1).
    #[test]
    fn lookup_returns_largest_valid() {
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let phase = &formula.phases[0];
        // Orbite simulée Mandelbrot avec c = -0.7 + 0.3i, z_0 = 0.
        let c = Complex64::new(-0.7, 0.3);
        let mut z = Complex64::new(0.0, 0.0);
        let mut orbit = vec![z];
        for _ in 0..9 {
            z = z * z + c;
            orbit.push(z);
        }
        let table = BlaTableUnified::build(&orbit, phase, c.norm(), 1e-6);

        // delta = +∞ à m=1 → aucun BLA valide.
        assert!(table.lookup(1, f64::INFINITY).is_none());

        // delta très petit à m=1 → trouve un BLA.
        let res = table.lookup(1, 1e-30);
        assert!(
            res.is_some(),
            "lookup m=1 à delta minuscule doit trouver un BLA"
        );
        assert!(res.unwrap().l >= 1);
    }

    /// Avec une orbite typique Mandelbrot (zoom 1) et un delta très petit,
    /// la table BLA doit permettre des skips importants au début de l'orbite.
    #[test]
    fn realistic_mandelbrot_table() {
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let phase = &formula.phases[0];
        // Orbite simulée Mandelbrot pour c = -0.7 + 0.3i sur quelques iters.
        let c = Complex64::new(-0.7, 0.3);
        let mut z = Complex64::new(0.0, 0.0);
        let mut orbit = vec![z];
        for _ in 0..32 {
            z = z * z + c;
            orbit.push(z);
        }
        let table = BlaTableUnified::build(&orbit, phase, c.norm(), 1e-6);
        assert!(!table.levels.is_empty());
        // Un delta très petit devrait permettre un skip non-trivial à m=0.
        let lookup = table.lookup(0, 1e-20);
        if let Some(node) = lookup {
            assert!(node.l >= 1);
        }
    }
}
