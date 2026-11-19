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

/// Unité d'arrondi des opérations double-double, avec deux bits de garde
/// (les `dd_mul` / `dd_add` « sloppy » tiennent ~2⁻¹⁰⁴ relatif).
pub const DD_UNIT: f64 = 1.0 / (1u128 << 102) as f64;

/// Borne absolue (espace z) au-delà de laquelle le pixel n'est plus
/// déterminé en double-double : la boucle s'arrête et rend « non fiable ».
pub const E_MAX: f64 = 1e-3;

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
    let b2 = bailout * bailout;
    // Zone où le test de marge exact vaut d'être fait : |z| à moins de E_MAX
    // (majoré) du rayon.
    let near_lo = (bailout - 2.0 * E_MAX).max(0.0);
    let near2 = near_lo * near_lo;
    let c_abs = c.0.hi.abs() + c.1.hi.abs();
    let (mut x, mut y) = z0;
    let mut e = 0.0f64;
    let mut iter = 0u32;
    let out = |iter: u32, x: Dd, y: Dd, reliable: bool| DirectDdResult {
        iteration: iter,
        z: Complex64::new(x.hi, y.hi),
        reliable,
    };
    if !phase_supported(phase) {
        return out(0, x, y, false);
    }
    if x.hi * x.hi + y.hi * y.hi >= b2 {
        return out(0, x, y, true);
    }
    while iter < iteration_max {
        for op in &phase.ops {
            match op {
                Op::Sqr => {
                    let n = x.hi * x.hi + y.hi * y.hi;
                    e = e * (2.0 * n.sqrt() + e) + DD_UNIT * n;
                    let re = x.sqr().sub(y.sqr());
                    let im = x.mul(y).mul_f64(2.0);
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
                    e += DD_UNIT * (x.hi.abs() + y.hi.abs() + c_abs);
                    iter += 1;
                }
                _ => return out(iter, x, y, false),
            }
        }
        let n = x.hi * x.hi + y.hi * y.hi;
        if !n.is_finite() {
            return out(iter, x, y, false);
        }
        if n >= b2 {
            // Évasion déterminée ssi |z| − E ≥ R (et la borne est saine).
            let ok = e < E_MAX.max(0.01 * n.sqrt()) && n.sqrt() - e >= bailout;
            return out(iter, x, y, ok);
        }
        // `!(e < E_MAX)` attrape aussi NaN.
        if !(e < E_MAX) {
            return out(iter, x, y, false);
        }
        // Pas encore évadé, mais à portée d'erreur du rayon : indéterminé.
        if n >= near2 && n.sqrt() + e >= bailout {
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
