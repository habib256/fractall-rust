//! Traqueur de fiabilité du tier **dd au tier exp** (G14.1 i).
//!
//! [`ShadowTrackerDd`] tient sa borne et ses entrées en f64 : sous ~2⁻¹⁰²² δ,
//! dc et les Z de frôlement atom-domain valent 0, chaque terme ajoute 0 et la
//! borne reste nulle — le traqueur dd était AVEUGLE au tier exp (pixel <
//! 1e-280), où la correction dd est pourtant appelée (pixels flaggés par
//! `ShadowTrackerExp`) et où `--dd-tier` rend des frames entières.
//!
//! [`ShadowTrackerDdExp`] : même récurrence, mêmes constantes (`U_DD`,
//! `GAMMA_DD`, ε BLA dd, troncature `l·ε`, borne collante), mais borne ET
//! magnitudes en `FloatExp` (mantisse f64 + exposant `i32`) : rien ne
//! sous-déborde. Les magnitudes sont lues sur les mantisses dd arrondies à
//! 53 b (une magnitude n'a pas besoin de 106 b) : écart relatif ≤ 2⁻⁵² par
//! terme avec le traqueur f64 là où celui-ci ne sous-déborde pas (verrou
//! `dd_exp_tracker_agrees_with_f64_tracker_where_f64_does_not_underflow`).
//!
//! Le trait [`DdTrack`] rend `ddexp_loop` générique : l'implémentation pour
//! `ShadowTrackerDd` refait EXACTEMENT les calculs f64 historiques (bornes et
//! itérations bit-identiques hors tier exp).

use super::bla_dd::Mat2Dd;
use super::reliability::{ShadowTrackerDd, GAMMA_DD, U_DD};
use crate::fractal::perturbation::dd::{ComplexDDExp, DoubleDoubleExp};
use crate::fractal::perturbation::types::FloatExp;

/// Exposant `frexp` maximal de dc en dessous duquel le pixel est suivi par
/// [`ShadowTrackerDdExp`] : |dc| < 2⁻⁹⁰⁰ ≈ 1e-271. C'est EXACTEMENT la garde
/// d'entrée du chemin rapide dd (`fast::run`, `MAX_SCALE`) : ces pixels
/// passaient déjà tous par `ddexp_loop`, le chemin rapide est intact. Le
/// tier exp (pixel < 1e-280) est inclus ; la frange 1e-280..1e-271 du tier
/// f64 aussi — là le traqueur f64 a déjà des termes locaux à ≤ 70 binades du
/// sous-normal (`GAMMA_DD·|dc|` ≈ 2⁻¹⁰⁰²), le traqueur à exposant y est le
/// plus juste des deux.
pub const DD_EXP_TRACK_MAX_DC_EXPONENT: i32 = -900;

/// Le pixel de `dc` doit-il être suivi par le traqueur à exposant ?
/// (`dc = 0` : non — pixel sur la référence, comportement historique.)
pub fn wants_exp_tracker(dc: ComplexDDExp) -> bool {
    [dc.re, dc.im]
        .iter()
        .filter(|x| x.mantissa.hi != 0.0)
        .map(|x| x.exponent)
        .max()
        .is_some_and(|e| e < DD_EXP_TRACK_MAX_DC_EXPONENT)
}

/// Interface du traqueur consommée par la boucle dd de référence.
pub trait DdTrack: Copy {
    /// Rebase `δ ← Z[m] + δ`.
    fn on_rebase(&mut self, z_m0: ComplexDDExp, zz: ComplexDDExp);
    /// Saut BLA de `l` pas (`delta` = δ AVANT le saut).
    fn on_bla(&mut self, a: &Mat2Dd, b: &Mat2Dd, delta: ComplexDDExp, dc: ComplexDDExp, l: u32);
    /// Pas direct : `z_abs` = Z[m] + δ_old, `z_m` = Z[m], `delta` = δ_old.
    fn on_direct(&mut self, z_abs: ComplexDDExp, z_m: ComplexDDExp, delta: ComplexDDExp);
    fn runaway(&self, threshold: f64) -> bool;
    fn error_bound(&self) -> f32;
}

/// Borne L1 (≥ module) d'un complexe dd, en f64 — identique à
/// `pixel_loop_dd::l1c`.
#[inline(always)]
fn l1c(z: ComplexDDExp) -> f64 {
    z.re.to_f64().abs() + z.im.to_f64().abs()
}

impl DdTrack for ShadowTrackerDd {
    #[inline(always)]
    fn on_rebase(&mut self, z_m0: ComplexDDExp, zz: ComplexDDExp) {
        self.rebase(l1c(z_m0), l1c(zz));
    }
    #[inline(always)]
    fn on_bla(&mut self, a: &Mat2Dd, b: &Mat2Dd, delta: ComplexDDExp, dc: ComplexDDExp, l: u32) {
        let sa = a.m00.to_f64().hypot(a.m10.to_f64());
        let sb = b.m00.to_f64().hypot(b.m10.to_f64());
        self.bla(sa, sa * l1c(delta), sb * l1c(dc), l);
    }
    #[inline(always)]
    fn on_direct(&mut self, z_abs: ComplexDDExp, z_m: ComplexDDExp, delta: ComplexDDExp) {
        let za = z_abs.to_complex64_approx().norm();
        let d = delta.to_complex64_approx();
        self.direct(za, 2.0 * l1c(z_m) * (d.re.abs() + d.im.abs()), d.norm_sqr());
    }
    #[inline(always)]
    fn runaway(&self, threshold: f64) -> bool {
        ShadowTrackerDd::runaway(self, threshold)
    }
    #[inline(always)]
    fn error_bound(&self) -> f32 {
        ShadowTrackerDd::error_bound(self)
    }
}

/// |x| en `FloatExp` (mantisse dd arrondie à 53 b, exposant conservé).
#[inline(always)]
fn fx(x: DoubleDoubleExp) -> FloatExp {
    FloatExp::new(x.mantissa.to_f64().abs(), x.exponent)
}

/// Borne L1 d'un complexe dd en `FloatExp` (sans sous-débordement).
#[inline(always)]
pub fn l1x(z: ComplexDDExp) -> FloatExp {
    fx(z.re) + fx(z.im)
}

/// Traqueur dd à exposant étendu (cf. doc du module).
#[derive(Clone, Copy, Debug)]
pub struct ShadowTrackerDdExp {
    pub err: FloatExp,
    dc_abs: FloatExp,
    bla_epsilon: f64,
    cap: FloatExp,
    /// Borne emballée (collante) : `error_bound() = +∞`.
    stuck: bool,
}

impl ShadowTrackerDdExp {
    pub fn new(dc: ComplexDDExp, bla_epsilon: f64, cap: f64) -> Self {
        let re = fx(dc.re);
        let im = fx(dc.im);
        Self {
            err: FloatExp::zero(),
            // |dc| exact au 53 b près (même grandeur que `norm()` du f64).
            dc_abs: (re.sqr() + im.sqr()).sqrt(),
            bla_epsilon,
            cap: FloatExp::from_f64(cap),
            stuck: false,
        }
    }

    #[inline(always)]
    fn stick(&mut self) {
        // NaN ⇒ comparaison fausse ⇒ emballée, comme le f64.
        let below = matches!(
            self.err.partial_cmp(&self.cap),
            Some(std::cmp::Ordering::Less)
        ) && self.err.mantissa.is_finite();
        if !below {
            self.stuck = true;
        }
    }

    fn direct_terms(&mut self, z_abs: FloatExp, two_z_delta_l1: FloatExp, delta_norm_sqr: FloatExp) {
        if self.stuck {
            return;
        }
        let local = (two_z_delta_l1 + delta_norm_sqr + self.dc_abs) * GAMMA_DD;
        self.err = self.err * (z_abs * 2.0 + self.err) + local;
        self.stick();
    }
}

impl DdTrack for ShadowTrackerDdExp {
    #[inline(always)]
    fn on_rebase(&mut self, z_m0: ComplexDDExp, zz: ComplexDDExp) {
        if self.stuck {
            return;
        }
        self.err = self.err + (l1x(z_m0) + l1x(zz)) * U_DD;
        self.stick();
    }
    #[inline(always)]
    fn on_bla(&mut self, a: &Mat2Dd, b: &Mat2Dd, delta: ComplexDDExp, dc: ComplexDDExp, l: u32) {
        if self.stuck {
            return;
        }
        // Coefficients dd sans exposant (plage f64) : σ₁ conforme en f64,
        // comme le traqueur f64 ; seuls les opérandes ont besoin d'exposant.
        let sa = a.m00.to_f64().hypot(a.m10.to_f64());
        let sb = b.m00.to_f64().hypot(b.m10.to_f64());
        let local = (l1x(delta) * sa + l1x(dc) * sb) * (self.bla_epsilon * l as f64 + GAMMA_DD);
        self.err = self.err * sa + local;
        self.stick();
    }
    #[inline(always)]
    fn on_direct(&mut self, z_abs: ComplexDDExp, z_m: ComplexDDExp, delta: ComplexDDExp) {
        let (zr, zi) = (fx(z_abs.re), fx(z_abs.im));
        let za = (zr.sqr() + zi.sqr()).sqrt();
        let (dr, di) = (fx(delta.re), fx(delta.im));
        let two_z_delta = l1x(z_m) * (dr + di) * 2.0;
        self.direct_terms(za, two_z_delta, dr.sqr() + di.sqr());
    }
    #[inline(always)]
    fn runaway(&self, threshold: f64) -> bool {
        self.stuck
            || !matches!(
                self.err.partial_cmp(&FloatExp::from_f64(threshold)),
                Some(std::cmp::Ordering::Less)
            )
    }
    fn error_bound(&self) -> f32 {
        if self.stuck {
            return f32::INFINITY;
        }
        let v = self.err.to_f64();
        if v.is_nan() {
            f32::INFINITY
        } else {
            v as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::bytecode::bla_dd::DD_BLA_EPSILON;
    use crate::fractal::perturbation::dd::DoubleDouble;

    fn cdd(re: f64, im: f64, e: i32) -> ComplexDDExp {
        ComplexDDExp {
            re: DoubleDoubleExp::normalized(DoubleDouble::from_f64(re), e),
            im: DoubleDoubleExp::normalized(DoubleDouble::from_f64(im), e),
        }
    }

    /// Même suite d'événements (pas directs, saut BLA, rebase) rejouée à
    /// l'échelle 2^shift : à shift = 0 (aucun sous-débordement f64) les deux
    /// traqueurs s'accordent ; à shift = -3000 (tier exp) le f64 rend 0.
    fn replay<T: DdTrack>(t: &mut T, shift: i32) {
        let z = cdd(0.3, -0.2, 0); // Z de référence O(1)
        let a = Mat2Dd {
            m00: DoubleDouble::from_f64(1.7),
            m01: DoubleDouble::from_f64(-0.4),
            m10: DoubleDouble::from_f64(0.4),
            m11: DoubleDouble::from_f64(1.7),
        };
        let b = Mat2Dd {
            m00: DoubleDouble::from_f64(3.0),
            m01: DoubleDouble::from_f64(-1.0),
            m10: DoubleDouble::from_f64(1.0),
            m11: DoubleDouble::from_f64(3.0),
        };
        let dc = cdd(0.6, 0.7, -40 + shift);
        let mut d = cdd(0.5, 0.25, -30 + shift);
        for k in 0..50 {
            let zabs = ComplexDDExp {
                re: z.re.add(d.re),
                im: z.im.add(d.im),
            };
            t.on_direct(zabs, z, d);
            if k % 10 == 9 {
                t.on_bla(&a, &b, d, dc, 7);
            }
            if k == 25 {
                // Rebase sur une référence en frôlement atom-domain.
                let zt = cdd(0.9, 0.1, -20 + shift);
                t.on_rebase(zt, ComplexDDExp { re: zt.re.add(d.re), im: zt.im.add(d.im) });
            }
            d = cdd(0.5 + 0.001 * k as f64, 0.25, -30 + shift);
        }
    }

    /// Fin de l'aveuglement : à l'échelle du tier exp (δ ~ 2⁻³⁰³⁰ ≈ 1e-912,
    /// dc ~ 1e-915, Z de frôlement ~ 1e-909) la borne du traqueur f64 est
    /// EXACTEMENT 0 ; celle du traqueur à exposant est > 0, et vaut la borne
    /// à l'échelle 1 multipliée par 2⁻³⁰⁰⁰ (récurrence linéaire tant que
    /// E ≪ |Z|, homogène en l'échelle).
    #[test]
    fn dd_exp_tracker_sees_what_the_f64_dd_tracker_misses_at_exp_tier() {
        let mut f = ShadowTrackerDd::new(cdd(0.6, 0.7, -3040).to_complex64_approx().norm(), DD_BLA_EPSILON);
        replay(&mut f, -3000);
        assert_eq!(f.err, 0.0, "le traqueur f64 devait être aveugle ici");
        let mut x = ShadowTrackerDdExp::new(cdd(0.6, 0.7, -3040), DD_BLA_EPSILON, 1e30);
        replay(&mut x, -3000);
        assert!(x.err > FloatExp::zero(), "borne exp nulle : toujours aveugle");
        let mut x0 = ShadowTrackerDdExp::new(cdd(0.6, 0.7, -40), DD_BLA_EPSILON, 1e30);
        replay(&mut x0, 0);
        let ratio = x.err.mantissa / x0.err.mantissa;
        let de = x.err.exponent - x0.err.exponent;
        // Écarts d'homogénéité : |Z+δ| et δ² (δ/Z ~ 2⁻³⁰ à l'échelle 1) — 1e-6.
        let r = ratio * 2f64.powi(de + 3000);
        assert!((r - 1.0).abs() < 1e-6, "borne exp non homogène : {r}");
    }

    /// Cohérence : là où le f64 ne sous-déborde pas, les deux bornes
    /// s'accordent à 1e-12 relatif (les magnitudes du traqueur à exposant
    /// sont les mêmes mantisses arrondies à 53 b : seul l'ordre
    /// d'arrondi diffère, quelques ulp ; |dc| exact des deux côtés).
    #[test]
    fn dd_exp_tracker_agrees_with_f64_tracker_where_f64_does_not_underflow() {
        let mut f = ShadowTrackerDd::new(cdd(0.6, 0.7, -40).to_complex64_approx().norm(), DD_BLA_EPSILON);
        replay(&mut f, 0);
        let mut x = ShadowTrackerDdExp::new(cdd(0.6, 0.7, -40), DD_BLA_EPSILON, 1e30);
        replay(&mut x, 0);
        assert!(f.err > 0.0);
        let rel = (x.err.to_f64() - f.err).abs() / f.err;
        assert!(rel < 1e-12, "f64 {} vs exp {} (rel {rel})", f.err, x.err.to_f64());
    }

    /// Borne emballée collante : au-delà du seuil elle reste `+∞` même si un
    /// saut BLA contractant (σ₁ < 1) suit.
    #[test]
    fn dd_exp_tracker_runaway_is_sticky() {
        let mut x = ShadowTrackerDdExp::new(cdd(0.6, 0.7, -3040), DD_BLA_EPSILON, 1e30);
        x.err = FloatExp::from_f64(1e31);
        x.stick();
        let small = Mat2Dd {
            m00: DoubleDouble::from_f64(1e-40),
            m01: DoubleDouble::from_f64(0.0),
            m10: DoubleDouble::from_f64(0.0),
            m11: DoubleDouble::from_f64(1e-40),
        };
        x.on_bla(&small, &small, cdd(0.5, 0.5, -3000), cdd(0.5, 0.5, -3040), 1);
        assert!(x.runaway(1e30));
        assert_eq!(x.error_bound(), f32::INFINITY);
    }

    #[test]
    fn exp_tracker_selection_matches_fast_path_entry_guard() {
        assert!(!wants_exp_tracker(cdd(0.5, 0.5, -899)));
        assert!(wants_exp_tracker(cdd(0.5, 0.5, -901)));
        assert!(!wants_exp_tracker(cdd(0.0, 0.0, 0)));
    }
}
