//! Traqueur de fiabilité exp « léger » de la boucle scaled-double.
//!
//! Même récurrence que [`ShadowTrackerExp`] (`E ← E·(2|z| + E) + γ·local` au
//! pas direct, `E ← σ₁(A)·E + (l·ε + γ)·local` au saut BLA), mais la borne est
//! portée comme `em · 2^ee` (f64 + exposant propre) renormalisée SEULEMENT
//! quand `em` sort de `[2⁻²⁰⁰, 2²⁰⁰]` — plus de `frexp` ni de gardes
//! d'exactitude à chaque pas. La borne n'est plus identique au bit à celle de
//! la boucle FloatExp (quelques ulp relatifs) ; les DÉCISIONS (emballement
//! ≥ 1e30, `escape_undetermined`) le restent tant que E n'est pas à ~1e-10
//! relatif d'un seuil : vérifié sur le corpus (comptes `flagged` et PNG).
//!
//! Le facteur de propagation (2|z| + E, σ₁) est calculé comme dans le hook
//! exact ; seuls les arrondis diffèrent. Rebases, tours lents et régime
//! emballé (E absolu > 2⁶⁰⁰) passent par le traqueur exact
//! (synchronisation `em·2^ee` ↔ `FloatExp`).

use num_complex::Complex64;

use super::ScaledObs;
use crate::fractal::bytecode::bla_dual_exp::Mat2Exp;
use crate::fractal::bytecode::reliability::{ShadowTrackerExp, StepObserverExp, EXP_TRACK_GAMMA};
use crate::fractal::perturbation::types::{ComplexExp, FloatExp};

#[inline(always)]
fn p2(k: i32) -> f64 {
    if k < -1022 {
        0.0
    } else if k > 1023 {
        f64::INFINITY
    } else {
        f64::from_bits(((k + 1023) as u64) << 52)
    }
}

#[inline(always)]
fn ex(x: FloatExp) -> i32 {
    if x.mantissa == 0.0 {
        -(1 << 28)
    } else {
        x.exponent
    }
}

const M_LO: f64 = f64::from_bits(((-200 + 1023) as u64) << 52);
const M_HI: f64 = f64::from_bits(((200 + 1023) as u64) << 52);
/// Au-delà, E absolu > 2⁶⁰⁰ (emballé) : traqueur exact.
const EE_MAX: i32 = 600;

/// Traqueur scaled : la borne vit dans `(em, ee)` ; `trk.err` n'est à jour
/// qu'après [`ScaledTracker::sync`].
#[derive(Clone, Copy, Debug)]
pub struct ScaledTracker {
    pub trk: ShadowTrackerExp,
    em: f64,
    ee: i32,
    dcm: f64,
    dce: i32,
    exact: bool,
}

impl ScaledTracker {
    pub fn new(trk: ShadowTrackerExp) -> Self {
        let dc = trk.dc_l1();
        let mut t = Self {
            trk,
            em: 0.0,
            ee: 0,
            dcm: dc.mantissa,
            dce: ex(dc),
            exact: false,
        };
        t.reload();
        t
    }

    /// `(em, ee)` → `trk.err`.
    #[inline]
    pub fn sync(&mut self) {
        if !self.exact {
            self.trk.set_err(self.em, self.ee);
        }
    }

    /// `trk.err` → `(em, ee)` ; régime emballé ⇒ reste sur le traqueur exact.
    #[inline]
    fn reload(&mut self) {
        let e = self.trk.err;
        self.em = e.mantissa;
        self.ee = e.exponent;
        self.exact = e.mantissa != 0.0 && e.exponent > EE_MAX;
    }

    /// `E ← E·mul·2^mul_e + lf·2^e`, puis renormalisation paresseuse.
    #[inline(always)]
    fn combine(&mut self, mul: f64, mul_e: i32, lf: f64, e: i32) {
        if self.em == 0.0 {
            self.em = lf;
            self.ee = e;
        } else {
            let mut em = self.em * mul;
            let mut ee = self.ee + mul_e;
            if e > ee {
                em *= p2(ee - e);
                ee = e;
            }
            self.em = em + lf * p2(e - ee);
            self.ee = ee;
        }
        let a = self.em.abs();
        if !(M_LO..=M_HI).contains(&a) {
            let biased = ((a.to_bits() >> 52) & 0x7ff) as i32;
            if a == 0.0 {
                self.ee = 0;
            } else if biased == 0 || biased == 0x7ff {
                // sous-normal / non fini : retour au traqueur exact
                let f = FloatExp::from_f64(self.em);
                self.trk.set_err(f.mantissa, f.exponent + self.ee);
                self.reload();
                self.exact = true;
                return;
            } else {
                let k = biased - 1023;
                self.em *= p2(-k);
                self.ee += k;
            }
        }
        if self.ee > EE_MAX {
            self.sync();
            self.reload();
            self.exact = true;
        }
    }
}

impl StepObserverExp for ScaledTracker {
    #[inline]
    fn direct(&mut self, z: ComplexExp, two_z_delta: ComplexExp, delta_old: ComplexExp) {
        self.sync();
        self.trk.direct(z, two_z_delta, delta_old);
        self.reload();
    }

    #[inline]
    fn bla(&mut self, a: &Mat2Exp, b: &Mat2Exp, d: ComplexExp, dc: ComplexExp, l: u32) {
        self.sync();
        self.trk.bla(a, b, d, dc, l);
        self.reload();
    }

    #[inline]
    fn rebase(&mut self, z_ref: ComplexExp, z_new: ComplexExp) {
        self.sync();
        self.trk.rebase(z_ref, z_new);
        self.reload();
    }
}

impl ScaledObs for ScaledTracker {
    const TRACKS: bool = true;

    /// Pas direct : `z = Z + δ`, `local = (|2Zδ|₁ + |δ|² + |dc|₁)·γ` (cadre 2^e).
    #[inline(always)]
    fn direct_scaled(&mut self, z: Complex64, d: Complex64, e: i32, tzd: Complex64) -> bool {
        if self.exact || e > 100 || (self.dcm != 0.0 && self.dce - e > 900) {
            return false;
        }
        let se = p2(e);
        let (zr, zi) = (z.re + d.re * se, z.im + d.im * se);
        let z_abs = (zr * zr + zi * zi).sqrt();
        let err_abs = self.em * p2(self.ee);
        let w = z_abs * 2.0 + err_abs;
        let l1 = tzd.re.abs() + tzd.im.abs();
        let nd = (d.re * d.re + d.im * d.im) * se;
        let dcf = self.dcm * p2(self.dce - e);
        let lf = (l1 + nd + dcf) * EXP_TRACK_GAMMA;
        self.combine(w, 0, lf, e);
        true
    }

    /// Saut BLA : `E ← σ₁(A)·E + (|Aδ|₁ + |B·dc|₁)·(l·ε + γ)` (cadre 2^ep).
    #[inline(always)]
    fn bla_scaled(&mut self, a: &Mat2Exp, ad: Complex64, bt: [f64; 4], ep: i32, l: u32) -> bool {
        if self.exact {
            return false;
        }
        let conformal = a.m00 == a.m11 && a.m01 == FloatExp::zero() - a.m10;
        let es = if conformal {
            ex(a.m00).max(ex(a.m10))
        } else {
            ex(a.m00).max(ex(a.m01)).max(ex(a.m10)).max(ex(a.m11))
        };
        if es < -(1 << 27) {
            return false;
        }
        let f = |x: FloatExp| x.mantissa * p2(ex(x) - es);
        let n = if conformal {
            let (x0, x1) = (f(a.m00), f(a.m10));
            x0 * x0 + x1 * x1
        } else {
            let (x0, x1, x2, x3) = (f(a.m00), f(a.m01), f(a.m10), f(a.m11));
            x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3
        };
        let sf = n.sqrt();
        let l1a = ad.re.abs() + ad.im.abs();
        let l1b = (bt[0] + bt[1]).abs() + (bt[2] + bt[3]).abs();
        let trunc = self.trk.bla_epsilon() * l as f64 + EXP_TRACK_GAMMA;
        let lf = (l1a + l1b) * trunc;
        self.combine(sf, es, lf, ep);
        true
    }
}
