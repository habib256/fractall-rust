//! RÉFÉRENCE de test : le traqueur scaled léger tel qu'il était avant le
//! chemin chaud du pas direct et le σ₁ précalculé (2026-11-06, G14.16) —
//! code verbatim, seul σ₁ est désormais reçu du nœud (son calcul d'origine
//! est [`sigma_reference`], comparé nœud par nœud au précalcul de la table).
//! Le verrou `light_tracker_matches_reference_bit_for_bit` exige la même
//! borne au bit près (mantisse ET exposant) que ce code.

use num_complex::Complex64;

use super::super::ScaledObs;
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
const EE_MAX: i32 = 600;

/// σ₁(A) tel que le calculait le hook `bla_scaled` à chaque saut.
pub fn sigma_reference(a: &Mat2Exp) -> Option<(f64, i32)> {
    let conformal = a.m00 == a.m11 && a.m01 == FloatExp::zero() - a.m10;
    let es = if conformal {
        ex(a.m00).max(ex(a.m10))
    } else {
        ex(a.m00).max(ex(a.m01)).max(ex(a.m10)).max(ex(a.m11))
    };
    if es < -(1 << 27) {
        return None;
    }
    let f = |x: FloatExp| x.mantissa * p2(ex(x) - es);
    let n = if conformal {
        let (x0, x1) = (f(a.m00), f(a.m10));
        x0 * x0 + x1 * x1
    } else {
        let (x0, x1, x2, x3) = (f(a.m00), f(a.m01), f(a.m10), f(a.m11));
        x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3
    };
    Some((n.sqrt(), es))
}

#[derive(Clone, Copy, Debug)]
pub struct RefTracker {
    pub trk: ShadowTrackerExp,
    em: f64,
    ee: i32,
    dcm: f64,
    dce: i32,
    exact: bool,
}

impl RefTracker {
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

    pub fn sync(&mut self) {
        if !self.exact {
            self.trk.set_err(self.em, self.ee);
        }
    }

    fn reload(&mut self) {
        let e = self.trk.err;
        self.em = e.mantissa;
        self.ee = e.exponent;
        self.exact = e.mantissa != 0.0 && e.exponent > EE_MAX;
    }

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

impl StepObserverExp for RefTracker {
    fn direct(&mut self, z: ComplexExp, two_z_delta: ComplexExp, delta_old: ComplexExp) {
        self.sync();
        self.trk.direct(z, two_z_delta, delta_old);
        self.reload();
    }

    fn bla(&mut self, a: &Mat2Exp, b: &Mat2Exp, d: ComplexExp, dc: ComplexExp, l: u32) {
        self.sync();
        self.trk.bla(a, b, d, dc, l);
        self.reload();
    }

    fn rebase(&mut self, z_ref: ComplexExp, z_new: ComplexExp) {
        self.sync();
        self.trk.rebase(z_ref, z_new);
        self.reload();
    }
}

impl ScaledObs for RefTracker {
    const TRACKS: bool = true;

    fn direct_scaled(&mut self, z: Complex64, d: Complex64, _nd: f64, e: i32, tzd: Complex64) -> bool {
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

    fn bla_scaled(&mut self, sigma: (f64, i32), ad: Complex64, bt: [f64; 4], ep: i32, l: u32) -> bool {
        if self.exact {
            return false;
        }
        let (sf, es) = sigma;
        if es < -(1 << 27) {
            return false;
        }
        let l1a = ad.re.abs() + ad.im.abs();
        let l1b = (bt[0] + bt[1]).abs() + (bt[2] + bt[3]).abs();
        let trunc = self.trk.bla_epsilon() * l as f64 + EXP_TRACK_GAMMA;
        let lf = (l1a + l1b) * trunc;
        self.combine(sf, es, lf, ep);
        true
    }
}
