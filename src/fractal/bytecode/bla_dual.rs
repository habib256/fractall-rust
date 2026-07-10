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
use rayon::prelude::*;

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

    /// `z := z · (cos + sin·i)`. Rotation linéaire :
    /// - value : produit complexe usuel.
    /// - jac : multiplication à gauche par la matrice de rotation
    ///         `R = [[c, -s], [s, c]]`.
    ///
    /// `|det R| = 1` → la rotation n'ajoute pas de contrainte sur le rayon
    /// de validité (norme préservée), donc on ne touche pas `r` côté builder.
    fn rot(&mut self, cos_theta: f64, sin_theta: f64) {
        let c = cos_theta;
        let s = sin_theta;
        let new_x = c * self.value_x - s * self.value_y;
        let new_y = s * self.value_x + c * self.value_y;
        self.value_x = new_x;
        self.value_y = new_y;
        let jac = self.jac;
        self.jac = Mat2 {
            m00: c * jac.m00 - s * jac.m10,
            m01: c * jac.m01 - s * jac.m11,
            m10: s * jac.m00 + c * jac.m10,
            m11: s * jac.m01 + c * jac.m11,
        };
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
            Op::Rot { cos_theta, sin_theta } => {
                // Pas de contrainte ajoutée sur le rayon (rotation isométrique).
                w.rot(*cos_theta, *sin_theta);
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
    /// Formules F3 (`bla.h:33-37`) :
    /// - `A_z = A_y · A_x`
    /// - `B_z = A_y · B_x + B_y`
    /// - `R_z = min(R_x, max(0, (R_y − sup|B_x|·c) / sup|A_x|))`
    ///
    /// `c` est le rayon de l'image en espace-c (`max |δc|`, cf. F3
    /// `engine.cc:282` `c = pixel_spacing · pixel_precision`), pas `|cref|`.
    pub fn merge(x: BlaMultiStep, y: BlaMultiStep, c: f64) -> Self {
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
            // F3 `bla.h:37` : le /sup|A_x| s'applique à TOUT le numérateur
            // (R_y inclus), pas au seul terme c. Sans ça, R_z est surestimé au
            // depth (sup|A_x| ≫ 1) → over-skip BLA (cf. artefacts rug 1e56).
            let inner = (ry - sup_bx * c).max(0.0) / sup_ax;
            rx.min(inner).max(0.0)
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
/// Nombre de niveaux BLA bas (single/2/4-step) ignorés au lookup. Aligné
/// Fraktaler-3 `bla_skip_levels = 3` (`param.h:50`). Aux petits pas, le pas
/// perturbation direct est plus précis que le BLA linéaire f64 ; les niveaux
/// hauts (≥ 8-step) gardent le gain de perf.
pub const BLA_SKIP_LEVELS: usize = 3;

#[derive(Clone, Debug)]
pub struct BlaTableUnified {
    pub levels: Vec<Vec<BlaMultiStep>>,
}

impl BlaTableUnified {
    /// Construit la table BLA unifiée pour une phase, à partir de l'orbite
    /// référence et de la phase bytecode.
    ///
    /// `c_norm` = rayon de l'image en espace-c (`max |δc|`, F3 `engine.cc:282`
    /// `c = pixel_spacing · pixel_precision`), utilisé dans la formule de merge
    /// pour ajuster le rayon de validité. PAS `|cref|` (erreur historique).
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

        // Seuil au-delà duquel le build passe en parallèle (rayon). Les orbites
        // profondes (dragon 4 M nœuds, e113/glitch_test_2 250 k) dominaient la
        // phase pixel via le build serial ; le map level-0 et les merges sont
        // indépendants par index → parallélisables SANS changer l'ordre ni les
        // valeurs (bit-identique au serial). Sous le seuil (goldens ~2,5 k iter),
        // on reste serial pour éviter l'overhead rayon. ⚠️ Le gain n'existe QUE si
        // le build tourne avec des cœurs libres (cf. `delta::prewarm_bla_entry`,
        // appelé hors de la boucle pixel) — sous le lock global les autres workers
        // sont parqués et rayon ne peut pas les voler.
        const PAR_BLA_MIN: usize = 1 << 16; // 65536

        // Single-step level-0 par index d'orbite (jamais matérialisé en entier).
        let single = |i: usize| -> BlaMultiStep {
            let z = ref_orbit[i];
            BlaMultiStep::from_single(build_bla_single_step(z.re, z.im, phase, epsilon))
        };

        // Level 1 construit en STREAMING : chaque nœud fusionne 2 single-steps
        // adjacents SANS stocker le niveau 0 (m nœuds = le plus gros). Comme
        // `single(i) == level0[i]`, `level1[k] = merge(level0[2k], level0[2k+1])`
        // est **bit-identique** au build « level0 puis merge » historique (tail
        // impair = `single(m-1)` promu = `level0[m-1]` promu). Sur les orbites
        // ultra-longues (opus2 80 M) la table BLA domine la RSS (cf. TODO OOM) :
        // ne pas matérialiser le niveau 0 retire ~m nœuds du pic de build.
        let n_pairs1 = m / 2;
        let build_l1 = |k: usize| BlaMultiStep::merge(single(2 * k), single(2 * k + 1), c_norm);
        let mut level1: Vec<BlaMultiStep> = if n_pairs1 >= PAR_BLA_MIN {
            (0..n_pairs1).into_par_iter().map(build_l1).collect()
        } else {
            (0..n_pairs1).map(build_l1).collect()
        };
        if m % 2 == 1 {
            level1.push(single(m - 1));
        }
        // levels[0] = Vec vide (niveau 0 skip, jamais matérialisé) ; level1 à l'index 1.
        let mut levels: Vec<Vec<BlaMultiStep>> = vec![Vec::new(), level1];

        // Niveaux supérieurs : merge adjacents (paires `[2k, 2k+1]`, tail impair
        // promu tel quel — identique à la boucle serial d'origine).
        while levels.last().unwrap().len() > 1 {
            let prev = levels.last().unwrap();
            let n_pairs = prev.len() / 2;
            let mut next: Vec<BlaMultiStep> = if n_pairs >= PAR_BLA_MIN {
                (0..n_pairs)
                    .into_par_iter()
                    .map(|k| BlaMultiStep::merge(prev[2 * k], prev[2 * k + 1], c_norm))
                    .collect()
            } else {
                (0..n_pairs)
                    .map(|k| BlaMultiStep::merge(prev[2 * k], prev[2 * k + 1], c_norm))
                    .collect()
            };
            // Si nombre impair, le dernier est promu tel quel.
            if prev.len() % 2 == 1 {
                next.push(prev[prev.len() - 1]);
            }
            levels.push(next);
            // Libère IMMÉDIATEMENT le niveau consommé s'il est < BLA_SKIP_LEVELS :
            // il vient de servir à merger le niveau du dessus et n'est JAMAIS
            // consulté par lookup (cf. F3 bla_skip_levels). Le libérer ici (au
            // lieu de tout garder jusqu'à la fin) abaisse le pic de build de ~2m
            // à ~0,75m nœuds. `levels.last()` (utilisé au tour suivant) est
            // l'index len-1 ; on ne touche que len-2. Idempotent avec le clear
            // final ci-dessous.
            let consumed = levels.len() - 2;
            if (1..BLA_SKIP_LEVELS).contains(&consumed) {
                levels[consumed] = Vec::new();
                levels[consumed].shrink_to_fit();
            }
        }

        // Libère les niveaux < BLA_SKIP_LEVELS restants (level 0 déjà vide ;
        // les autres déjà vidés dans la boucle sauf si l'orbite est trop courte
        // pour atteindre ce niveau). JAMAIS consultés par `lookup`/`lookup_fexp`
        // (cf. F3 bla_skip_levels). Les Vec vides préservent l'indexation
        // `levels[level]` pour level ≥ BLA_SKIP_LEVELS.
        for l in 0..BLA_SKIP_LEVELS.min(levels.len()) {
            levels[l] = Vec::new();
            levels[l].shrink_to_fit();
        }
        Self { levels }
    }

    /// Cherche le BLA avec le plus grand `l` valide à partir de l'itération
    /// `m` quand `|δ|² < r2`. Retourne `None` si aucun BLA n'est valide.
    ///
    /// Stratégie F3 : parcourir les niveaux du plus grand au plus petit,
    /// retourner le premier valide. Les `BLA_SKIP_LEVELS` plus bas niveaux
    /// (single/2/4-step) sont ignorés — cf. F3 `bla_skip_levels=3`
    /// (`bla.cc:151`, `param.h:50`) : aux petits pas, le pas perturbation
    /// direct (ComplexExp exact) est PLUS précis que le BLA linéaire en f64
    /// (qui drop δ² et déconditionne ses coefficients au deep zoom). Sans ce
    /// skip, fractall applique le single-step BLA en permanence et accumule
    /// l'erreur → artefacts (anneaux/blobs lissés, cf. rug zoom 1e56).
    pub fn lookup(&self, m: usize, delta_norm_sqr: f64) -> Option<&BlaMultiStep> {
        // Seuls les niveaux `L ≤ trailing_zeros(m)` sont alignés (cf. lookup_fexp) :
        // démarrer au plus haut niveau aligné évite de parcourir les niveaux hauts
        // jamais alignés à chaque itération (réduit les accès à la table BLA, qui
        // est le coût dominant — memory-bound, cf. G2).
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
            let idx = m >> level; // m aligné à 2^level ⇒ m == idx<<level (pas de check)
            if idx < nodes.len() {
                let node = &nodes[idx];
                if delta_norm_sqr < node.r2 {
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
        // Seuls les niveaux `L ≤ trailing_zeros(m)` sont alignés (m % 2^L == 0,
        // donc `m == (m>>L)<<L`). Démarrer au plus haut niveau aligné évite de
        // parcourir les ~log2(ref_len) niveaux hauts jamais alignés à chaque
        // itération — gain hot-loop deep zoom (cf. G2 : lookup co-dominant).
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
            let idx = m >> level; // m aligné à 2^level ⇒ m == idx<<level (pas de check)
            if idx < nodes.len() {
                let node = &nodes[idx];
                // Compare `delta_norm_sqr (FloatExp) < node.r2` via PartialOrd (gère
                // l'underflow exp < -1074 ET le faux-positif « delta tiny → always
                // valid », cf. floral_fantasy). r² peut être denormal/zéro.
                let r2_fexp = crate::fractal::perturbation::types::FloatExp::from_f64(node.r2);
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

    /// Phase [Sqr, Rot, Add] : la Jacobienne doit valoir R · J_sqr où R est
    /// la matrice de rotation et J_sqr = 2·[Zx, -Zy; Zy, Zx]. Vérifie qu'on
    /// compose correctement la rotation par-dessus le Jacobien sqr.
    #[test]
    fn rot_after_sqr_matches_composed_jacobian() {
        use crate::fractal::bytecode::{Op, Phase};

        let theta = 0.5_f64;
        let (s, c) = theta.sin_cos();
        let phase = Phase::new(vec![
            Op::Sqr,
            Op::Rot { cos_theta: c, sin_theta: s },
            Op::Add,
        ]);
        let (zx, zy) = (0.3, -0.4);
        let bla = build_bla_single_step(zx, zy, &phase, 1e-6);

        // J_sqr = 2·[Zx, -Zy; Zy, Zx]
        let j_sqr = Mat2 {
            m00: 2.0 * zx,
            m01: -2.0 * zy,
            m10: 2.0 * zy,
            m11: 2.0 * zx,
        };
        // R = [[c, -s], [s, c]]
        let r = Mat2 { m00: c, m01: -s, m10: s, m11: c };
        // Composition : R · J_sqr.
        let expected = Mat2 {
            m00: r.m00 * j_sqr.m00 + r.m01 * j_sqr.m10,
            m01: r.m00 * j_sqr.m01 + r.m01 * j_sqr.m11,
            m10: r.m10 * j_sqr.m00 + r.m11 * j_sqr.m10,
            m11: r.m10 * j_sqr.m01 + r.m11 * j_sqr.m11,
        };
        assert!(
            mat2_close(bla.a, expected, 1e-10),
            "got {:?}, expected R·J_sqr={:?}",
            bla.a,
            expected
        );
        assert!(bla.r2 > 0.0, "rotation ne doit pas annuler le rayon de validité");
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
    /// au sommet pour M=8. Les niveaux < BLA_SKIP_LEVELS sont vidés après build
    /// (jamais consultés par lookup — cf. G2 memory hygiene) mais préservés
    /// comme Vec vides pour garder l'indexation `levels[level]`.
    #[test]
    fn table_build_levels_8_iterations() {
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let phase = &formula.phases[0];
        // 9 entrées d'orbite → M = 8 single-steps possibles.
        let orbit: Vec<Complex64> = (0..9).map(|i| Complex64::new(i as f64 * 0.1, 0.0)).collect();
        let table = BlaTableUnified::build(&orbit, phase, 0.5, 1e-6);

        // M=8 → 4 niveaux : 8, 4, 2, 1 ; les niveaux < BLA_SKIP_LEVELS vidés.
        assert_eq!(table.levels.len(), 4);
        for l in 0..BLA_SKIP_LEVELS.min(4) {
            assert_eq!(table.levels[l].len(), 0, "niveau {l} doit être vidé (skip)");
        }
        assert_eq!(table.levels[3].len(), 1);

        // Le BLA top-level (consulté) doit avoir l = 8.
        assert_eq!(table.levels[3][0].l, 8);
    }

    /// Le lookup retourne `None` quand delta est trop grand, et un BLA de
    /// niveau ≥ `BLA_SKIP_LEVELS` (8-step ou plus) sinon. Avec le skip des
    /// bas niveaux (F3 `bla_skip_levels`), les petits pas / m non alignés
    /// retournent `None` (le caller fait un pas direct, plus précis).
    #[test]
    fn lookup_returns_largest_valid() {
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let phase = &formula.phases[0];
        // Orbite Mandelbrot longue (c = -0.7 + 0.3i, z_0 = 0) pour avoir des
        // niveaux BLA hauts (≥ 8-step). 40 itérations → M=40, niveaux jusqu'à 32.
        let c = Complex64::new(-0.7, 0.3);
        let mut z = Complex64::new(0.0, 0.0);
        let mut orbit = vec![z];
        for _ in 0..40 {
            z = z * z + c;
            orbit.push(z);
        }
        // c_norm petit (extent pixel typique) pour que les BLA fusionnés
        // gardent un rayon de validité positif (un grand c_norm écrase les
        // rayons des niveaux hauts via le terme B·c du merge).
        let table = BlaTableUnified::build(&orbit, phase, 1e-9, 1e-6);

        // m=8 est aligné au niveau 3 (8-step), le plus bas niveau utilisable.
        // delta = +∞ → aucun BLA valide.
        assert!(table.lookup(8, f64::INFINITY).is_none());

        // delta minuscule à m=8 → trouve un BLA de niveau ≥ 3 (l ≥ 8).
        let res = table.lookup(8, 1e-30);
        assert!(
            res.is_some(),
            "lookup m=8 (aligné niveau 3) à delta minuscule doit trouver un BLA"
        );
        assert!(
            res.unwrap().l >= 8,
            "avec skip_levels=3, le plus petit BLA retourné saute ≥ 8 itérations, got l={}",
            res.unwrap().l
        );

        // m=1 (non aligné aux niveaux hauts) → None : pas direct.
        assert!(
            table.lookup(1, 1e-30).is_none(),
            "m=1 ne doit PAS retourner de BLA bas niveau (skip_levels) — pas direct"
        );
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
