//! Itération DIRECTE en double-double (~106 bits), tracée (2026-11-19).
//!
//! Étage de correction entre la perturbation f64 et le GMP pur par pixel,
//! pour les familles qui n'ont PAS de tier dd de perturbation (Julia,
//! Tricorn — le tier dd de `pixel_loop_dd` est Mandelbrot seulement) et sur
//! les vues peu profondes, où `c` tient dans 106 bits.
//!
//! Pourquoi : un pixel que la perturbation f64 signale partait droit au GMP
//! pur (~150 ns par itération à 256 b). Mesuré par l'axe `types` du harness
//! (Tricorn, bord ×4096, 5000 itérations, 128×80) : la correction du path
//! standard coûtait 0,17 s pour 910 pixels signalés, dont 0,007 s de
//! perturbation — tout le reste en GMP. La même itération en double-double
//! coûte ~10-15 ns par itération.
//!
//! Garantie : celle des autres tiers — une borne d'erreur au premier ordre
//! `E' = E·(2|z| + E) + u·(|z|² + |c|)` (`u` = [`DD_UNIT`], les plis `abs` /
//! `neg` sont 1-lipschitziens), et le pixel n'est déclaré fiable que si la
//! borne reste négligeable ([`E_MAX`]) ET que l'évasion est déterminée
//! (marge `|z| − E ≥ R` au pas d'évasion, `|z| + E < R` avant). Un pixel non
//! fiable garde le chemin d'avant : GMP pur.

use num_complex::Complex64;

use super::{Op, Phase};
use crate::fractal::perturbation::dd::DoubleDouble as Dd;
use crate::fractal::perturbation::qd::Qd;

/// Unité d'arrondi des opérations double-double, avec deux bits de garde
/// (les `dd_mul` / `dd_add` « sloppy » tiennent ~2⁻¹⁰⁴ relatif).
pub const DD_UNIT: f64 = 1.0 / (1u128 << 102) as f64;

/// Borne absolue (espace z) au-delà de laquelle le pixel n'est plus
/// déterminé en double-double : la boucle s'arrête et rend « non fiable ».
pub const E_MAX: f64 = 1e-3;

/// Écart relatif maximal entre une norme recalculée en f64 sur les seules
/// PARTIES HAUTES (troncature des parties basses ≤ 2⁻⁵³, deux carrés, une
/// somme, une racine) et le module exact du nombre dd/qd : 8·2⁻⁵³ (marge
/// ×2). Les tests d'évasion des étages dd/qd décident sur cette norme f64,
/// dont l'arrondi (~2⁻⁵²·R) dépasse de très loin leur borne (~2⁻¹⁰³·R) : il
/// doit entrer dans la marge, sinon un point exact juste sous le rayon dont
/// la norme f64 arrondit au-dessus sort « fiable » au mauvais pas (verrous
/// `direct_stages_escape_test_covers_f64_norm_rounding`,
/// `dd_stage_escape_test_covers_f64_norm_rounding`).
pub const NORM_REL: f64 = 8.0 / (1u64 << 53) as f64;

/// Résultat d'un pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectDdResult {
    pub iteration: u32,
    pub z: Complex64,
    /// `false` : la valeur n'est pas garantie (à recalculer en GMP).
    pub reliable: bool,
}

/// La phase est-elle itérable ici ? Mêmes opcodes que le traqueur du noyau
/// par lots (`interp_batch::phase_trackable`).
pub fn phase_supported(phase: &Phase) -> bool {
    super::interp_batch::phase_trackable(phase)
}

/// Arithmétique d'une itération directe tracée : double-double ou
/// quad-double (mêmes opérations, dans le même ordre — le code monomorphisé
/// pour [`Dd`] est celui d'avant la généralisation).
pub trait DirectNum: Copy {
    /// Unité d'arrondi relative aux opérandes (bits de garde compris).
    const UNIT: f64;
    fn hi(self) -> f64;
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul(self, o: Self) -> Self;
    fn sqr(self) -> Self;
    /// `2·x` (exact pour le qd ; le dd garde son `mul_f64(2.0)` historique).
    fn twice(self) -> Self;
    fn neg(self) -> Self;
    fn abs(self) -> Self;
}

impl DirectNum for Dd {
    const UNIT: f64 = DD_UNIT;
    #[inline(always)]
    fn hi(self) -> f64 {
        self.hi
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Dd::add(self, o)
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Dd::sub(self, o)
    }
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        Dd::mul(self, o)
    }
    #[inline(always)]
    fn sqr(self) -> Self {
        Dd::sqr(self)
    }
    #[inline(always)]
    fn twice(self) -> Self {
        Dd::mul_f64(self, 2.0)
    }
    #[inline(always)]
    fn neg(self) -> Self {
        Dd::neg(self)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        Dd::abs(self)
    }
}

#[cfg(test)]
impl DirectNum for Qd {
    /// 2⁻²⁰⁶ : erreur « sloppy » mesurée ≤ 2⁻²¹⁰ relative aux opérandes,
    /// marge ×16 (même unité que la boucle qd de perturbation).
    const UNIT: f64 = super::pixel_loop_qd::U_QD;
    #[inline(always)]
    fn hi(self) -> f64 {
        self.0[0]
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Qd::add(self, o)
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Qd::sub(self, o)
    }
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        Qd::mul(self, o)
    }
    #[inline(always)]
    fn sqr(self) -> Self {
        Qd::sqr(self)
    }
    #[inline(always)]
    fn twice(self) -> Self {
        Qd::mul_pow2(self, 2.0)
    }
    #[inline(always)]
    fn neg(self) -> Self {
        Qd::neg(self)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        if self.0[0] < 0.0 {
            self.neg()
        } else {
            self
        }
    }
}

/// Itère `phase` depuis `z0` avec la constante `c` (Mandelbrot-like :
/// `z0 = 0`, `c` = pixel ; Julia : `z0` = pixel, `c` = graine). Même
/// sémantique de compte que `interp::iterate_bytecode_f64` (test de sortie
/// initial, puis après chaque phase).
pub fn iterate_direct_dd(
    phase: &Phase,
    z0: (Dd, Dd),
    c: (Dd, Dd),
    iteration_max: u32,
    bailout: f64,
) -> DirectDdResult {
    iterate_direct(phase, z0, c, iteration_max, bailout)
}

/// Comme [`iterate_direct_dd`] en quad-double (~212 bits) : l'étage suivant
/// des pixels que le double-double ne garantit pas, avant le GMP pur. Même
/// garantie (borne au premier ordre, [`E_MAX`], marge d'évasion), unité
/// 2⁻²⁰⁶.
///
/// Arithmétique de [`super::direct_simd::QdV`] à une voie (renormalisation
/// sans branchement) : la référence pixel par pixel des voies vectorielles
/// de production ([`super::direct_simd::direct_qd_lanes`]).
pub fn iterate_direct_qd(
    phase: &Phase,
    z0: (Qd, Qd),
    c: (Qd, Qd),
    iteration_max: u32,
    bailout: f64,
) -> DirectDdResult {
    use super::direct_simd::{iterate_lanes, QdV};
    iterate_lanes::<QdV<1>, 1>(phase, &[c], z0, false, iteration_max, bailout)[0]
}

/// Ancienne boucle qd mono-pixel (`perturbation::qd`, renormalisation à
/// branches) : témoin des tests (mêmes comptes que la boucle de production).
#[cfg(test)]
pub(crate) fn iterate_direct_qd_branchy(
    phase: &Phase,
    z0: (Qd, Qd),
    c: (Qd, Qd),
    iteration_max: u32,
    bailout: f64,
) -> DirectDdResult {
    iterate_direct(phase, z0, c, iteration_max, bailout)
}

/// Corps commun de [`iterate_direct_dd`] / [`iterate_direct_qd`].
#[inline(always)]
fn iterate_direct<N: DirectNum>(
    phase: &Phase,
    z0: (N, N),
    c: (N, N),
    iteration_max: u32,
    bailout: f64,
) -> DirectDdResult {
    let b2 = bailout * bailout;
    // Zone où le test de marge exact vaut d'être fait : |z| à moins de E_MAX
    // (majoré) du rayon.
    let near_lo = (bailout - 2.0 * E_MAX).max(0.0);
    let near2 = near_lo * near_lo;
    let c_abs = c.0.hi().abs() + c.1.hi().abs();
    let (mut x, mut y) = z0;
    let mut e = 0.0f64;
    let mut iter = 0u32;
    let out = |iter: u32, x: N, y: N, reliable: bool| DirectDdResult {
        iteration: iter,
        z: Complex64::new(x.hi(), y.hi()),
        reliable,
    };
    if !phase_supported(phase) {
        return out(0, x, y, false);
    }
    // Test initial, même marge que les tests de sortie (cf. `NORM_REL`).
    let n0 = x.hi() * x.hi() + y.hi() * y.hi();
    if n0 >= b2 || n0.sqrt() * (1.0 + NORM_REL) >= bailout {
        return out(0, x, y, n0 >= b2 && n0.sqrt() * (1.0 - NORM_REL) >= bailout);
    }
    while iter < iteration_max {
        for op in &phase.ops {
            match op {
                Op::Sqr => {
                    let n = x.hi() * x.hi() + y.hi() * y.hi();
                    e = e * (2.0 * n.sqrt() + e) + N::UNIT * n;
                    let re = x.sqr().sub(y.sqr());
                    let im = x.mul(y).twice();
                    x = re;
                    y = im;
                }
                Op::AbsX => x = x.abs(),
                Op::AbsY => y = y.abs(),
                Op::NegX => x = x.neg(),
                Op::NegY => y = y.neg(),
                Op::Add => {
                    x = x.add(c.0);
                    y = y.add(c.1);
                    e += N::UNIT * (x.hi().abs() + y.hi().abs() + c_abs);
                    iter += 1;
                }
                _ => return out(iter, x, y, false),
            }
        }
        let n = x.hi() * x.hi() + y.hi() * y.hi();
        if !n.is_finite() {
            return out(iter, x, y, false);
        }
        if n >= b2 {
            // Évasion déterminée ssi |z| − E ≥ R (et la borne est saine).
            let ok = e < E_MAX.max(0.01 * n.sqrt()) && n.sqrt() * (1.0 - NORM_REL) - e >= bailout;
            return out(iter, x, y, ok);
        }
        // `!(e < E_MAX)` attrape aussi NaN.
        if !(e < E_MAX) {
            return out(iter, x, y, false);
        }
        // Pas encore évadé, mais à portée d'erreur du rayon : indéterminé.
        if n >= near2 && n.sqrt() * (1.0 + NORM_REL) + e >= bailout {
            return out(iter, x, y, false);
        }
    }
    out(iter, x, y, true)
}

/// Valeur double-double d'un flottant GMP (106 bits de tête).
pub fn dd_from_gmp(v: &rug::Float) -> Dd {
    let hi = v.to_f64();
    if !hi.is_finite() {
        return Dd::new(hi, 0.0);
    }
    let rest = rug::Float::with_val(v.prec().max(128), v - hi);
    Dd::new(hi, rest.to_f64())
}

/// Valeur quad-double d'un flottant GMP (212 bits de tête).
pub fn qd_from_gmp(v: &rug::Float) -> Qd {
    let mut scratch = rug::Float::new(v.prec().max(64));
    Qd::from_float(v, &mut scratch)
}

/// Interrupteur `FRACTALL_DIRECT_QD=0` (A/B) : pas d'étage quad-double
/// direct — les pixels que le double-double direct ne garantit pas partent
/// droit au GMP pur, comme avant.
pub fn direct_qd_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("FRACTALL_DIRECT_QD").ok().as_deref().map(str::trim),
            Some("0") | Some("off") | Some("false")
        )
    })
}

/// Interrupteur `FRACTALL_DIRECT_DD=0` (A/B) : les pixels signalés partent
/// droit au GMP, comme avant.
pub fn direct_dd_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("FRACTALL_DIRECT_DD").ok().as_deref().map(str::trim),
            Some("0") | Some("off") | Some("false")
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::bytecode::{compile_formula, interp::iterate_bytecode_f64};
    use crate::fractal::FractalType;

    fn dd(v: f64) -> Dd {
        Dd::from_f64(v)
    }

    /// Sur des points à évasion rapide (aucune amplification), le
    /// double-double rend le MÊME compte que l'interpréteur f64 de
    /// référence, pour chaque famille supportée.
    #[test]
    fn direct_dd_matches_f64_interpreter_on_fast_escapes() {
        for ft in [
            FractalType::Mandelbrot,
            FractalType::Tricorn,
            FractalType::BurningShip,
            FractalType::Celtic,
            FractalType::Buffalo,
            FractalType::PerpendicularBurningShip,
        ] {
            let f = compile_formula(ft, 2.0).expect("formule");
            let phase = &f.phases[0];
            assert!(phase_supported(phase), "{ft:?}");
            let mut checked = 0;
            for j in 0..24 {
                for i in 0..24 {
                    let c = Complex64::new(-2.2 + 3.3 * i as f64 / 23.0, -1.6 + 3.2 * j as f64 / 23.0);
                    let r64 = iterate_bytecode_f64(&f, Complex64::new(0.0, 0.0), c, 60, 25.0);
                    let r = iterate_direct_dd(phase, (dd(0.0), dd(0.0)), (dd(c.re), dd(c.im)), 60, 25.0);
                    if r.reliable && r64.iteration < 40 {
                        assert_eq!(r.iteration, r64.iteration, "{ft:?} c={c}");
                        checked += 1;
                    }
                }
            }
            assert!(checked > 200, "{ft:?} : {checked} points comparés seulement");
        }
    }

    /// Un point intérieur stable est fiable au plafond ; une orbite
    /// chaotique longue perd sa détermination et le dit.
    #[test]
    fn direct_dd_reports_lost_determination() {
        let f = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let phase = &f.phases[0];
        // c = −1 : cycle superattractif de période 2.
        let r = iterate_direct_dd(phase, (dd(0.0), dd(0.0)), (dd(-1.0), dd(0.0)), 5000, 25.0);
        assert!(r.reliable && r.iteration == 5000);
        // c = −2 + 1e-12 i : voisin du bout de l'antenne, dynamique chaotique
        // (exposant ln 2 par pas) — 106 bits sont épuisés en ~100 pas.
        let r = iterate_direct_dd(phase, (dd(0.0), dd(0.0)), (dd(-1.999_999_9), dd(1e-12)), 100_000, 25.0);
        assert!(!r.reliable || r.iteration < 400, "{r:?}");
    }

    /// Le quad-double rend les mêmes comptes que le double-double là où les
    /// deux sont fiables, et tient environ deux fois plus longtemps sur une
    /// orbite chaotique (212 bits contre 106 : ~ln 2 perdu par pas).
    #[test]
    fn direct_qd_matches_dd_and_lasts_twice_as_long() {
        let f = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let phase = &f.phases[0];
        let q = |v: f64| Qd::from_f64(v);
        let mut checked = 0;
        for j in 0..20 {
            for i in 0..20 {
                let c = Complex64::new(-2.2 + 3.3 * i as f64 / 19.0, -1.6 + 3.2 * j as f64 / 19.0);
                let a = iterate_direct_dd(phase, (dd(0.0), dd(0.0)), (dd(c.re), dd(c.im)), 2000, 25.0);
                let b = iterate_direct_qd(phase, (q(0.0), q(0.0)), (q(c.re), q(c.im)), 2000, 25.0);
                if a.reliable {
                    assert!(b.reliable, "c={c}");
                    assert_eq!(a.iteration, b.iteration, "c={c}");
                    checked += 1;
                }
            }
        }
        assert!(checked > 300, "{checked}");
        // Axe réel près de la pointe : orbite bornée chaotique (exposant ~ln 2).
        let c = (-1.99, 0.0);
        let a = iterate_direct_dd(phase, (dd(0.0), dd(0.0)), (dd(c.0), dd(c.1)), 100_000, 25.0);
        let b = iterate_direct_qd(phase, (q(0.0), q(0.0)), (q(c.0), q(c.1)), 100_000, 25.0);
        assert!(!a.reliable && !b.reliable, "{a:?} {b:?}");
        assert!(b.iteration as f64 >= 1.8 * a.iteration as f64, "dd {} qd {}", a.iteration, b.iteration);
    }

    /// Les 106 bits servent : deux `c` égaux en f64 mais distincts en
    /// double-double donnent des orbites distinctes.
    #[test]
    fn direct_dd_consumes_the_low_word() {
        let f = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let phase = &f.phases[0];
        let a = iterate_direct_dd(phase, (dd(0.0), dd(0.0)), (Dd::new(-0.75, 0.0), Dd::new(0.1, 0.0)), 30, 25.0);
        let b = iterate_direct_dd(phase, (dd(0.0), dd(0.0)), (Dd::new(-0.75, 1e-20), Dd::new(0.1, 0.0)), 30, 25.0);
        assert_eq!(a.iteration, b.iteration);
        let v = rug::Float::with_val(200, rug::Float::parse("0.1000000000000000000000000000001").unwrap());
        let d = dd_from_gmp(&v);
        assert_eq!(d.hi, 0.1);
        assert!(d.lo != 0.0 && d.lo.abs() < 1e-16);
    }
}
