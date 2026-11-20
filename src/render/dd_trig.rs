//! `sin`/`cos`/`sinh`/`cosh` en double-double (~106 b) pour l'étage dd des
//! types Sin du suivi de fiabilité (`scalar_reliability`) — 2026-11-19.
//!
//! Aucune fonction transcendante dd n'existait dans le repo. Schéma classique
//! (QD de Bailey) avec des constantes calculées UNE fois en GMP (`OnceLock`) :
//! - `sin`/`cos` : réduction `x = k·π/64 + t` (`|t| ≤ π/128`), table de
//!   `sin(kπ/64)`, `cos(kπ/64)` (k mod 128), séries de Taylor de `t` jusqu'à
//!   `t¹³` / `t¹²` (reste < 10⁻³³) ;
//! - `exp` : `a = m·ln 2 + j/256 + r` (`|r| ≤ 1/512`), table de
//!   `exp(j/256)`, Taylor jusqu'à `r⁹` (reste < 10⁻³³), puis `2^m` exact ;
//!   `sinh`/`cosh = (e^a ∓ e^−a)/2` — erreur ABSOLUE ≤ `CH·u·cosh`, qui suffit
//!   au traqueur (la borne est absolue).
//!
//! Bornes d'erreur ([`SIN_ERR`], [`HYP_ERR`]) vérifiées contre MPFR 256 b
//! avec une marge ×8 (verrou `dd_trig_matches_mpfr_within_bounds`).

use std::sync::OnceLock;

use rug::float::Constant;
use rug::Float;

use crate::fractal::bytecode::direct_dd::dd_from_gmp;
use crate::fractal::perturbation::dd::DoubleDouble as Dd;

/// `|sin x − ŝ|, |cos x − ĉ| ≤ SIN_ERR·DD_UNIT·(1 + |x|)` (réduction en
/// `|x|·2⁻¹⁰⁶`, table et série en quelques `2⁻¹⁰⁶`).
pub const SIN_ERR: f64 = 16.0;
/// `|sinh y − ŝh|, |cosh y − ĉh| ≤ HYP_ERR·DD_UNIT·cosh y`.
pub const HYP_ERR: f64 = 16.0;
/// Au-delà, pas de résultat (`None`) : le pixel part au GMP.
const SIN_MAX_ARG: f64 = 1e9;
const HYP_MAX_ARG: f64 = 700.0;
/// `|s| ≤ ln 2 / 2` ⇒ `|j| = |round(256·s)| ≤ 89`.
const EXP_J: usize = 90;

/// Addition double-double « rapide » (QD sloppy) : un seul `two_sum`, erreur
/// ≤ 2⁻¹⁰⁴·(|a| + |b|) — relative aux OPÉRANDES, pas au résultat (contre
/// deux `two_sum` et une erreur relative au résultat pour `Dd::add`). Les
/// bornes des pas qui l'emploient comptent donc l'arrondi de chaque somme sur
/// ses opérandes (séries de Taylor ici : termes décroissants, même ordre que
/// le résultat).
#[inline(always)]
pub(crate) fn dd_add_fast(a: Dd, b: Dd) -> Dd {
    let s = a.hi + b.hi;
    let bb = s - a.hi;
    let e = (a.hi - (s - bb)) + (b.hi - bb) + (a.lo + b.lo);
    let hi = s + e;
    Dd::new(hi, e - (hi - s))
}

pub struct DdTrig {
    pi64: Dd,
    sin_tab: [Dd; 128],
    cos_tab: [Dd; 128],
    /// `(−1)ⁱ/(2i+1)!`, i = 0..=6.
    sin_coef: [Dd; 7],
    /// `(−1)ⁱ/(2i)!`, i = 0..=6.
    cos_coef: [Dd; 7],
    ln2: Dd,
    /// `exp(j/256)`, j = −EXP_J..=EXP_J.
    exp_tab: [Dd; 2 * EXP_J + 1],
    /// `1/n!`, n = 0..=9.
    exp_coef: [Dd; 10],
}

impl DdTrig {
    pub fn get() -> &'static DdTrig {
        static T: OnceLock<DdTrig> = OnceLock::new();
        T.get_or_init(DdTrig::build)
    }

    fn build() -> DdTrig {
        const P: u32 = 320;
        let pi = Float::with_val(P, Constant::Pi);
        let pi64 = Float::with_val(P, &pi / 64u32);
        let sin_tab = std::array::from_fn(|k| {
            dd_from_gmp(&Float::with_val(P, &pi64 * k as u32).sin())
        });
        let cos_tab = std::array::from_fn(|k| {
            dd_from_gmp(&Float::with_val(P, &pi64 * k as u32).cos())
        });
        let fact = |n: u32| -> Float { Float::with_val(P, rug::Integer::from(rug::Integer::factorial(n))) };
        let inv = |n: u32, neg: bool| -> Dd {
            let mut v = Float::with_val(P, 1u32) / fact(n);
            if neg {
                v = -v;
            }
            dd_from_gmp(&v)
        };
        let sin_coef = std::array::from_fn(|i| inv(2 * i as u32 + 1, i % 2 == 1));
        let cos_coef = std::array::from_fn(|i| inv(2 * i as u32, i % 2 == 1));
        let exp_coef = std::array::from_fn(|n| inv(n as u32, false));
        let ln2 = dd_from_gmp(&Float::with_val(P, Constant::Log2));
        let exp_tab = std::array::from_fn(|j| {
            let x = Float::with_val(P, j as i32 - EXP_J as i32) / 256u32;
            dd_from_gmp(&x.exp())
        });
        DdTrig { pi64: dd_from_gmp(&pi64), sin_tab, cos_tab, sin_coef, cos_coef, ln2, exp_tab, exp_coef }
    }

    /// `(sin x, cos x)`, `None` si `|x| ≥ 1e9` ou non fini.
    #[inline(always)]
    pub fn sin_cos(&self, x: Dd) -> Option<(Dd, Dd)> {
        if !(x.hi.abs() < SIN_MAX_ARG) {
            return None;
        }
        let k = (x.hi * (64.0 / std::f64::consts::PI)).round();
        let t = x.sub(self.pi64.mul_f64(k));
        let j = (k as i64).rem_euclid(128) as usize;
        let t2 = t.sqr();
        let mut s = self.sin_coef[6];
        let mut c = self.cos_coef[6];
        for i in (0..6).rev() {
            s = dd_add_fast(s.mul(t2), self.sin_coef[i]);
            c = dd_add_fast(c.mul(t2), self.cos_coef[i]);
        }
        let s = s.mul(t);
        let (sj, cj) = (self.sin_tab[j], self.cos_tab[j]);
        Some((sj.mul(c).add(cj.mul(s)), cj.mul(c).sub(sj.mul(s))))
    }

    /// `exp(a)` pour `0 ≤ a < 700`.
    #[inline(always)]
    fn exp(&self, a: Dd) -> Dd {
        let m = (a.hi * std::f64::consts::LOG2_E).round();
        let s = a.sub(self.ln2.mul_f64(m));
        let j = (s.hi * 256.0).round().clamp(-(EXP_J as f64), EXP_J as f64);
        let r = s.sub(Dd::from_f64(j / 256.0));
        let mut e = self.exp_coef[9];
        for n in (0..9).rev() {
            e = dd_add_fast(e.mul(r), self.exp_coef[n]);
        }
        e.mul(self.exp_tab[(j as i32 + EXP_J as i32) as usize]).mul_f64(2f64.powi(m as i32))
    }

    /// `(sinh y, cosh y)` (sinh signé comme y), `None` si `|y| ≥ 700`.
    #[inline(always)]
    pub fn sinh_cosh(&self, y: Dd) -> Option<(Dd, Dd)> {
        let a = y.abs();
        if !(a.hi < HYP_MAX_ARG) {
            return None;
        }
        if a.hi == 0.0 {
            return Some((y, Dd::from_f64(1.0)));
        }
        let e = self.exp(a);
        let ie = Dd::from_f64(1.0).div(e);
        let sh = e.sub(ie).mul_f64(0.5);
        let ch = e.add(ie).mul_f64(0.5);
        Some((if y.hi < 0.0 { sh.neg() } else { sh }, ch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::bytecode::direct_dd::DD_UNIT;

    fn to_gmp(x: Dd) -> Float {
        let mut v = Float::with_val(256, x.hi);
        v += x.lo;
        v
    }

    fn err(x: Dd, exact: &Float) -> f64 {
        let mut d = to_gmp(x);
        d -= exact;
        d.to_f64().abs()
    }

    /// Erreurs mesurées contre MPFR 256 b ≤ borne / 8, sur des arguments dd
    /// (partie basse non nulle), proches des multiples de π/16 compris.
    #[test]
    fn dd_trig_matches_mpfr_within_bounds() {
        let t = DdTrig::get();
        let (mut worst_s, mut worst_h) = (0.0f64, 0.0f64);
        for i in -2000..=2000 {
            let hi = i as f64 * 0.0251 + (i as f64 * 0.37).sin() * 1e-3;
            let x = Dd::new(hi, hi * 1.3e-17);
            let gx = to_gmp(x);
            let (s, c) = t.sin_cos(x).unwrap();
            let bound = SIN_ERR * DD_UNIT * (1.0 + hi.abs());
            let e = err(s, &Float::with_val(256, gx.sin_ref())).max(err(c, &Float::with_val(256, gx.cos_ref())));
            worst_s = worst_s.max(e / bound);
            let y = Dd::new(hi * 0.4, -hi * 2.1e-17);
            let gy = to_gmp(y);
            let (sh, ch) = t.sinh_cosh(y).unwrap();
            let gch = Float::with_val(256, gy.cosh_ref());
            let bound = HYP_ERR * DD_UNIT * gch.to_f64();
            let e = err(sh, &Float::with_val(256, gy.sinh_ref())).max(err(ch, &gch));
            worst_h = worst_h.max(e / bound);
        }
        // Multiples de π/16 à l'ulp près (t ≈ 0) et grands arguments.
        for k in [1i32, 7, 8, 16, 31, 32, 63, 64, 127, 128, 400, 4001] {
            let x = dd_from_gmp(&(Float::with_val(256, Constant::Pi) * k / 64u32));
            let gx = to_gmp(x);
            let (s, c) = t.sin_cos(x).unwrap();
            let bound = SIN_ERR * DD_UNIT * (1.0 + x.hi.abs());
            let e = err(s, &Float::with_val(256, gx.sin_ref())).max(err(c, &Float::with_val(256, gx.cos_ref())));
            worst_s = worst_s.max(e / bound);
        }
        eprintln!("dd trig : pire erreur / borne : sin_cos {worst_s:.4}, sinh_cosh {worst_h:.4}");
        assert!(worst_s < 0.125, "sin_cos : {worst_s}");
        assert!(worst_h < 0.125, "sinh_cosh : {worst_h}");
        assert!(t.sin_cos(Dd::from_f64(2e9)).is_none());
        assert!(t.sinh_cosh(Dd::from_f64(-800.0)).is_none());
        let (sh, ch) = t.sinh_cosh(Dd::from_f64(-0.0)).unwrap();
        assert_eq!((sh.hi, ch.hi), (0.0, 1.0));
    }
}
