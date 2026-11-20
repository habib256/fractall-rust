//! Étage **quad-double** (~212 b) de la correction des boucles SCALAIRES
//! suivies (`scalar_reliability`) — 2026-11-20.
//!
//! Chaîne de correction de ces types : f64 tracé → double-double tracé →
//! **quad-double tracé** → GMP pur. Les pixels que le dd ne détermine pas
//! (bord Magnet Mandelbrot 256×160 : 2,5 % ; Mandelbrot Sin : 1,8 %)
//! partaient tous au GMP pur, qui faisait l'essentiel de la correction
//! (Mandelbrot Sin : ~15 µs/itération, 11,8 s cumulées sur 13,9). Le qd en
//! règle l'écrasante majorité pour une fraction du prix ; le GMP ne garde
//! que ce qu'il ne détermine pas non plus.
//!
//! Mêmes pas, mêmes bornes que les étages dd (`*_dd_step`), avec :
//! - l'arithmétique `perturbation::qd` (variantes « sloppy » : erreur
//!   relative aux OPÉRANDES de chaque opération — c'est le modèle des bornes,
//!   écrites sur les opérandes) ;
//! - l'unité [`UQ`] = 2⁻²⁰⁰, soit 64× l'erreur mesurée des opérations qd
//!   (< 2⁻²⁰⁸ contre `rug`, verrou `perturbation::qd`) : la marge absorbe
//!   les bornes complexes grossières (normes L1 contre modules) au prix de
//!   6 bits sur les ~100 que le qd a de plus que le dd ;
//! - pour les types Sin, `sin/cos/sinh/cosh` qd ([`QdTrig`]) : mêmes
//!   réductions que `dd_trig`, plus de termes, constantes GMP (`OnceLock`),
//!   bornes vérifiées contre MPFR 512 b (`qd_trig_matches_mpfr_within_bounds`).
//!
//! Le point exact du pixel est arrondi en qd DEPUIS le GMP (`qd_from_gmp`
//! sur centre HP + dc GMP), à la précision de l'étage GMP qui suit.

use num_complex::Complex64;
use rayon::prelude::*;
use rug::float::Constant;
use rug::Float;

use crate::fractal::bytecode::direct_dd::{qd_from_gmp, NORM_REL};
use crate::fractal::cycle::CycleCut;
use crate::fractal::perturbation::qd::Qd;
use crate::fractal::{FractalParams, FractalType};

/// Unité d'arrondi des bornes qd (2⁻²⁰⁰, cf. doc du module).
pub const UQ: f64 = 6.223_015_277_861_142e-61;
/// Plafond de la borne (même valeur que l'étage dd : bornes non linéarisées).
const E_MAX_NL: f64 = 0.25;
/// Plafond des bornes LINÉARISÉES (Barnsley), comme `scalar_reliability::E_MAX`.
const E_MAX_LIN: f64 = 1e-3;
const SIN_FLOOR: f64 = 1e-60;

// --- arithmétique qd complémentaire (sans toucher `perturbation::qd`) -------

#[inline(always)]
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

/// Renormalisation de quatre composantes déjà ordonnées (chaîne de
/// `quick_two_sum` sans branchement : une composante nulle donne une somme
/// exacte, la valeur est conservée).
#[inline(always)]
fn renorm4(c0: f64, c1: f64, c2: f64, c3: f64) -> Qd {
    let (s, c3) = quick_two_sum(c2, c3);
    let (s, c2) = quick_two_sum(c1, s);
    let (c0, c1) = quick_two_sum(c0, s);
    let (s1, s2) = quick_two_sum(c1, c2);
    let (s2, s3) = quick_two_sum(s2, c3);
    Qd([c0, s1, s2, s3])
}

#[inline(always)]
pub fn qd_abs(a: Qd) -> Qd {
    if a.0[0] < 0.0 {
        a.neg()
    } else {
        a
    }
}

/// Division qd (QD `sloppy_div` : quatre quotients partiels, restes exacts au
/// modèle sloppy près). Erreur relative mesurée < 2⁻²⁰⁴ (verrou
/// `qd_div_matches_rug`).
#[inline(always)]
pub fn qd_div(a: Qd, b: Qd) -> Qd {
    let q0 = a.0[0] / b.0[0];
    let r = a.sub(b.mul_f64(q0));
    let q1 = r.0[0] / b.0[0];
    let r = r.sub(b.mul_f64(q1));
    let q2 = r.0[0] / b.0[0];
    let r = r.sub(b.mul_f64(q2));
    let q3 = r.0[0] / b.0[0];
    renorm4(q0, q1, q2, q3)
}

/// Complexe quad-double des pas de correction.
#[derive(Clone, Copy, Debug)]
pub struct Cqd {
    pub re: Qd,
    pub im: Qd,
}

impl Cqd {
    pub const ZERO: Cqd = Cqd { re: Qd::ZERO, im: Qd::ZERO };

    #[inline(always)]
    fn sqr(self) -> Self {
        Cqd { re: self.re.sqr().sub(self.im.sqr()), im: self.re.mul(self.im).mul_pow2(2.0) }
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Cqd { re: self.re.add(o.re), im: self.im.add(o.im) }
    }
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        Cqd {
            re: self.re.mul(o.re).sub(self.im.mul(o.im)),
            im: self.re.mul(o.im).add(self.im.mul(o.re)),
        }
    }
    #[inline(always)]
    pub fn abs_hi(self) -> f64 {
        (self.re.0[0] * self.re.0[0] + self.im.0[0] * self.im.0[0]).sqrt()
    }
    #[inline(always)]
    pub(crate) fn hi(self) -> Complex64 {
        Complex64::new(self.re.0[0], self.im.0[0])
    }
}

// --- voies entrelacées génériques --------------------------------------------

/// Nombre complexe d'un étage de correction à voies entrelacées : partie
/// haute et état complet (pour le raccourci des cycles exacts : `S` mots,
/// borne comprise).
pub(crate) trait LaneZ<const S: usize>: Copy {
    fn hi_c(self) -> Complex64;
    fn abs_h(self) -> f64;
    fn state(self, e: f64) -> [f64; S];
}

impl LaneZ<9> for Cqd {
    #[inline(always)]
    fn hi_c(self) -> Complex64 {
        self.hi()
    }
    #[inline(always)]
    fn abs_h(self) -> f64 {
        self.abs_hi()
    }
    #[inline(always)]
    fn state(self, e: f64) -> [f64; 9] {
        let (r, i) = (self.re.0, self.im.0);
        [r[0], r[1], r[2], r[3], i[0], i[1], i[2], i[3], e]
    }
}

/// Fin d'un pas tracé : `(continuer, fiable)` — borne petite, évasion
/// déterminée (même test que l'étage dd).
///
/// ⚠️ La sortie est décidée sur la norme f64 des parties hautes : son
/// arrondi (~2⁻⁵²·R) dépasse de très loin la borne dd/qd (~2⁻¹⁰³·R). Sans
/// le terme `NORM_REL·n`, un point exact juste SOUS le rayon dont la norme
/// f64 arrondit au-dessus sortait « fiable » au mauvais pas (verrou
/// `dd_stage_escape_test_covers_f64_norm_rounding`).
#[inline(always)]
pub(crate) fn step_ok(n: f64, bailout: f64, e: f64) -> bool {
    n.is_finite() && (n - bailout).abs() > e + NORM_REL * n && e < E_MAX_NL
}

/// `LANES` pixels de front, chaque voie rechargée dès qu'elle finit. Chaque
/// voie fait exactement la suite d'opérations d'une boucle mono-pixel
/// (pas, test de sortie, raccourci de cycle sur l'état complet).
///
/// `julia` : le point du pixel est le `z` de départ (borne initiale
/// `unit·|z₀|`) et le pas ignore son troisième argument ; sinon `z` part de
/// `seed` et le point est le `c` du pas.
pub(crate) fn lanes<Z: LaneZ<S>, const S: usize, const LANES: usize>(
    points: &[Z],
    seed: Z,
    julia: bool,
    unit: f64,
    iteration_max: u32,
    bailout: f64,
    step: impl Fn(Z, f64, Z) -> (Z, f64, f64),
) -> Vec<(u32, Complex64, bool)> {
    const FREE: usize = usize::MAX;
    let b2 = bailout * bailout;
    let mut out = vec![(0u32, Complex64::new(0.0, 0.0), false); points.len()];
    let mut next = 0usize;
    let mut idx = [FREE; LANES];
    let mut z = [seed; LANES];
    let mut c = [seed; LANES];
    let mut e = [0.0f64; LANES];
    let mut n = [0.0f64; LANES];
    let mut it = [0u32; LANES];
    let mut cyc: [CycleCut<S>; LANES] = std::array::from_fn(|_| CycleCut::new());
    loop {
        for l in 0..LANES {
            loop {
                if idx[l] != FREE {
                    let h = z[l].hi_c();
                    let n2 = h.re * h.re + h.im * h.im;
                    // Point de départ : sa sortie (ou non) doit être
                    // déterminée comme celle d'un pas (`step_ok`).
                    let ok0 = it[l] > 0 || step_ok(n2.sqrt(), bailout, e[l]);
                    if ok0 && it[l] < iteration_max && n2 < b2 {
                        break;
                    }
                    out[idx[l]] = (it[l], h, ok0);
                    idx[l] = FREE;
                }
                if next == points.len() {
                    break;
                }
                idx[l] = next;
                c[l] = points[next];
                cyc[l] = CycleCut::new();
                if julia {
                    z[l] = points[next];
                    e[l] = unit * points[next].abs_h();
                } else {
                    z[l] = seed;
                    e[l] = 0.0;
                }
                it[l] = 0;
                next += 1;
            }
        }
        if idx.iter().all(|&k| k == FREE) {
            break;
        }
        for l in 0..LANES {
            let (nz, ne, nn) = step(z[l], e[l], c[l]);
            z[l] = nz;
            e[l] = ne;
            n[l] = nn;
        }
        for l in 0..LANES {
            if idx[l] != FREE {
                it[l] += 1;
                if !step_ok(n[l], bailout, e[l]) {
                    out[idx[l]] = (it[l], z[l].hi_c(), false);
                    idx[l] = FREE;
                } else {
                    cyc[l].observe(&mut it[l], iteration_max, z[l].state(e[l]));
                }
            }
        }
    }
    out
}

/// [`lanes`] en parallèle par paquets (monomorphisé sur le pas).
pub(crate) fn lanes_par<Z, const S: usize, const LANES: usize, F>(
    points: &[Z],
    seed: Z,
    julia: bool,
    unit: f64,
    iteration_max: u32,
    bailout: f64,
    step: F,
) -> Vec<(u32, Complex64, bool)>
where
    Z: LaneZ<S> + Send + Sync,
    F: Fn(Z, f64, Z) -> (Z, f64, f64) + Sync + Copy,
{
    // Paquets plus petits que l'étage dd : peu de pixels, chers.
    const CHUNK: usize = 8;
    points
        .par_chunks(CHUNK)
        .flat_map_iter(|ch| lanes::<Z, S, LANES>(ch, seed, julia, unit, iteration_max, bailout, step))
        .collect()
}

// --- fonctions transcendantes qd ---------------------------------------------

/// `|sin x − ŝ|, |cos x − ĉ| ≤ SIN_ERR·UQ·(1 + |x|)`.
pub const SIN_ERR: f64 = 4.0;
/// `|sinh y − ŝh|, |cosh y − ĉh| ≤ HYP_ERR·UQ·(1 + |y|)·cosh y`.
pub const HYP_ERR: f64 = 4.0;
const SIN_MAX_ARG: f64 = 1e9;
const HYP_MAX_ARG: f64 = 700.0;
/// `|s| ≤ ln 2 / 2` ⇒ `|j₁| = |round(256·s)| ≤ 89`.
const EXP_J1: usize = 90;
/// `|s₁| ≤ 1/512` ⇒ `|j₂| = |round(65536·s₁)| ≤ 128`.
const EXP_J2: usize = 129;
/// `|s₂| ≤ 2⁻¹⁷` ⇒ `|j₃| = |round(2²⁴·s₂)| ≤ 128`.
const EXP_J3: usize = 129;
const SIN_TERMS: usize = 13;
/// `1/n!`, n ≤ 7 : reste `r⁸/8!` < 2⁻²²⁹ pour `|r| ≤ 2⁻²⁵`.
const EXP_TERMS: usize = 8;

/// Polynôme `Σ aᵢ·uⁱ` à la Estrin : chaînes de dépendance en `log₂ N`
/// (Horner : `N`), mêmes opérations qd.
#[inline(always)]
fn estrin<const N: usize>(a: &[Qd; N], u: Qd) -> Qd {
    let mut b = *a;
    let mut n = N;
    let mut p = u;
    while n > 1 {
        let half = n / 2;
        for i in 0..half {
            b[i] = b[2 * i].add(b[2 * i + 1].mul(p));
        }
        if n % 2 == 1 {
            b[half] = b[n - 1];
            n = half + 1;
        } else {
            n = half;
        }
        if n > 1 {
            p = p.sqr();
        }
    }
    b[0]
}

pub struct QdTrig {
    pi64: Qd,
    sin_tab: [Qd; 128],
    cos_tab: [Qd; 128],
    /// `(−1)ⁱ/(2i+1)!`, `(−1)ⁱ/(2i)!`, i < SIN_TERMS (reste < 10⁻⁶⁶ pour
    /// `|t| ≤ π/128`).
    sin_coef: [Qd; SIN_TERMS],
    cos_coef: [Qd; SIN_TERMS],
    /// `ln 2 = ln2_hi + ln2_lo` (Cody-Waite : `m·ln2_hi` exact).
    ln2_hi: f64,
    ln2_lo: Qd,
    /// `exp(j/256)`, `exp(j/65536)`, `exp(j/2²⁴)`.
    exp_tab1: [Qd; 2 * EXP_J1 + 1],
    exp_tab2: [Qd; 2 * EXP_J2 + 1],
    exp_tab3: [Qd; 2 * EXP_J3 + 1],
    /// Coefficients pairs `1/(2i)!` et impairs `1/(2i+1)!`.
    exp_even: [Qd; EXP_TERMS / 2],
    exp_odd: [Qd; EXP_TERMS / 2],
}

impl QdTrig {
    pub fn get() -> &'static QdTrig {
        static T: std::sync::OnceLock<QdTrig> = std::sync::OnceLock::new();
        T.get_or_init(QdTrig::build)
    }

    fn build() -> QdTrig {
        const P: u32 = 512;
        let pi = Float::with_val(P, Constant::Pi);
        let pi64 = Float::with_val(P, &pi / 64u32);
        let sin_tab = std::array::from_fn(|k| qd_from_gmp(&Float::with_val(P, &pi64 * k as u32).sin()));
        let cos_tab = std::array::from_fn(|k| qd_from_gmp(&Float::with_val(P, &pi64 * k as u32).cos()));
        let inv = |n: u32, neg: bool| -> Qd {
            let f = Float::with_val(P, rug::Integer::from(rug::Integer::factorial(n)));
            let mut v = Float::with_val(P, 1u32) / f;
            if neg {
                v = -v;
            }
            qd_from_gmp(&v)
        };
        let sin_coef = std::array::from_fn(|i| inv(2 * i as u32 + 1, i % 2 == 1));
        let cos_coef = std::array::from_fn(|i| inv(2 * i as u32, i % 2 == 1));
        let exp_even = std::array::from_fn(|i| inv(2 * i as u32, false));
        let exp_odd = std::array::from_fn(|i| inv(2 * i as u32 + 1, false));
        let ln2 = Float::with_val(P, Constant::Log2);
        let ln2_hi = Float::with_val(40, &ln2).to_f64();
        let exp_tab1 = std::array::from_fn(|j| {
            qd_from_gmp(&(Float::with_val(P, j as i32 - EXP_J1 as i32) / 256u32).exp())
        });
        let exp_tab2 = std::array::from_fn(|j| {
            qd_from_gmp(&(Float::with_val(P, j as i32 - EXP_J2 as i32) / 65536u32).exp())
        });
        let exp_tab3 = std::array::from_fn(|j| {
            qd_from_gmp(&(Float::with_val(P, j as i32 - EXP_J3 as i32) / 16_777_216u32).exp())
        });
        QdTrig {
            pi64: qd_from_gmp(&pi64),
            sin_tab,
            cos_tab,
            sin_coef,
            cos_coef,
            ln2_hi,
            ln2_lo: qd_from_gmp(&Float::with_val(P, &ln2 - ln2_hi)),
            exp_tab1,
            exp_tab2,
            exp_tab3,
            exp_even,
            exp_odd,
        }
    }

    /// `(sin x, cos x)`, `None` si `|x| ≥ 1e9` ou non fini.
    #[inline(always)]
    pub fn sin_cos(&self, x: Qd) -> Option<(Qd, Qd)> {
        if !(x.0[0].abs() < SIN_MAX_ARG) {
            return None;
        }
        let k = (x.0[0] * (64.0 / std::f64::consts::PI)).round();
        let t = x.sub(self.pi64.mul_f64(k));
        let j = (k as i64).rem_euclid(128) as usize;
        let t2 = t.sqr();
        let s = estrin(&self.sin_coef, t2).mul(t);
        let c = estrin(&self.cos_coef, t2);
        let (sj, cj) = (self.sin_tab[j], self.cos_tab[j]);
        Some((sj.mul(c).add(cj.mul(s)), cj.mul(c).sub(sj.mul(s))))
    }

    /// `(exp(a), exp(−a))` pour `0 ≤ a < 700` : un seul polynôme en parties
    /// paire et impaire (`E ± O`), sans division.
    #[inline(always)]
    fn exp_pm(&self, a: Qd) -> (Qd, Qd) {
        let m = (a.0[0] * std::f64::consts::LOG2_E).round();
        let s = a.sub(Qd::from_f64(m * self.ln2_hi)).sub(self.ln2_lo.mul_f64(m));
        let j1 = (s.0[0] * 256.0).round().clamp(-(EXP_J1 as f64), EXP_J1 as f64);
        let s1 = s.sub(Qd::from_f64(j1 / 256.0));
        let j2 = (s1.0[0] * 65536.0).round().clamp(-(EXP_J2 as f64), EXP_J2 as f64);
        let s2 = s1.sub(Qd::from_f64(j2 / 65536.0));
        let j3 = (s2.0[0] * 16_777_216.0).round().clamp(-(EXP_J3 as f64), EXP_J3 as f64);
        let r = s2.sub(Qd::from_f64(j3 / 16_777_216.0));
        let r2 = r.sqr();
        let ev = estrin(&self.exp_even, r2);
        let od = estrin(&self.exp_odd, r2).mul(r);
        let (j1, j2, j3) = (j1 as i32, j2 as i32, j3 as i32);
        let (e1, e2, e3) = (EXP_J1 as i32, EXP_J2 as i32, EXP_J3 as i32);
        let t = |sg: i32| {
            self.exp_tab1[(e1 + sg * j1) as usize]
                .mul(self.exp_tab2[(e2 + sg * j2) as usize])
                .mul(self.exp_tab3[(e3 + sg * j3) as usize])
        };
        let mi = m as i32;
        (
            ev.add(od).mul(t(1)).mul_pow2(2f64.powi(mi)),
            ev.sub(od).mul(t(-1)).mul_pow2(2f64.powi(-mi)),
        )
    }

    /// `(sinh y, cosh y)` (sinh signé comme y), `None` si `|y| ≥ 700`.
    #[inline(always)]
    pub fn sinh_cosh(&self, y: Qd) -> Option<(Qd, Qd)> {
        let a = qd_abs(y);
        if !(a.0[0] < HYP_MAX_ARG) {
            return None;
        }
        if a.0[0] == 0.0 {
            return Some((y, Qd::from_f64(1.0)));
        }
        let (e, ie) = self.exp_pm(a);
        let sh = e.sub(ie).mul_pow2(0.5);
        let ch = e.add(ie).mul_pow2(0.5);
        Some((if y.0[0] < 0.0 { sh.neg() } else { sh }, ch))
    }
}

// --- pas tracés qd (mêmes bornes que les pas dd, unité UQ) -------------------

/// Alpha Mandelbrot (cf. `alpha_dd_step`).
#[inline(always)]
fn alpha_qd_step(z: Cqd, e: f64, c: Cqd) -> (Cqd, f64, f64) {
    let c_abs = c.abs_hi();
    let e_c = UQ * c_abs;
    let a = z.abs_hi();
    let s = z.sqr();
    let m = s.add(c);
    let m2 = m.sqr();
    let z = s.add(m2).add(c);
    let es = e * (2.0 * a + e) + 2.0 * UQ * a * a;
    let bm = m.abs_hi();
    let dm = e_c + UQ * (a * a + c_abs);
    let em = es + dm;
    let (pr, pi) = (1.0 + 2.0 * m.re.0[0], 2.0 * m.im.0[0]);
    let f = (pr * pr + pi * pi).sqrt() + em;
    let e = es * f + dm * (2.0 * bm + em) + e_c + 2.0 * UQ * (a * a + bm * bm + c_abs);
    (z, e, z.abs_hi())
}

/// Mandelbulb puissance 8 (cf. `mandelbulb_dd_step`).
#[inline(always)]
fn mandelbulb_qd_step(z: Cqd, e: f64, c: Cqd) -> (Cqd, f64, f64) {
    let c_abs = c.abs_hi();
    let e_c = UQ * c_abs;
    let a = z.abs_hi();
    let z = z.sqr().sqr().sqr().add(c);
    let t = a + e;
    let t2 = t * t;
    let t7 = t2 * t2 * t2 * t;
    let a2 = a * a;
    let a8 = a2 * a2 * a2 * a2;
    let n = z.abs_hi();
    (z, 8.0 * e * t7 + 16.0 * UQ * a8 + e_c + UQ * (a8 + c_abs + n), n)
}

/// Magnet (cf. `magnet_dd_step`) ; arrondis relatifs aux OPÉRANDES (sommes
/// sloppy : `n = z² + s₁` paie `|z|² + |s₁|`, pas `|n|`).
#[inline(always)]
pub(crate) fn magnet_qd_step(z: Cqd, e: f64, s1: Cqd, s2: Cqd, e_s: f64) -> (Cqd, f64, f64) {
    let n = z.sqr().add(s1);
    let q = Cqd { re: z.re.mul_pow2(2.0), im: z.im.mul_pow2(2.0) }.add(s2);
    let qq = q.re.sqr().add(q.im.sqr());
    let nr = n.re.mul(q.re).add(n.im.mul(q.im));
    let ni = n.im.mul(q.re).sub(n.re.mul(q.im));
    // Une seule division : 1/|q|² puis deux produits (erreur relative de
    // quelques 2⁻²¹¹ — dans le terme 8·UQ·|r|).
    let iqq = qd_div(Qd::from_f64(1.0), qq);
    let r = Cqd { re: nr.mul(iqq), im: ni.mul(iqq) };
    let zn = r.sqr();
    let q_abs = qq.0[0].sqrt();
    let r_abs = r.abs_hi();
    let z2 = z.re.0[0] * z.re.0[0] + z.im.0[0] * z.im.0[0];
    let s1a = s1.abs_hi();
    let s2a = s2.abs_hi();
    let rho_n = UQ * (3.0 * z2 + n.abs_hi() + s1a) + e_s;
    let rho_q = UQ * (2.0 * z2.sqrt() + q_abs + s2a) + e_s;
    let den = q_abs - 2.0 * e - rho_q;
    let (wr, wi) = (z.re.0[0] - r.re.0[0], z.im.0[0] - r.im.0[0]);
    let w = (wr * wr + wi * wi).sqrt();
    let er = (e * (2.0 * w + e) + rho_n + r_abs * rho_q + 8.0 * UQ * r_abs * (2.0 * e + rho_q)) / den
        + 8.0 * UQ * r_abs;
    let e2 = if den > 1e-10 && qq.0[0] >= 1e-20 {
        er * (2.0 * r_abs + er) + 2.0 * UQ * r_abs * r_abs
    } else {
        f64::INFINITY
    };
    (zn, e2, zn.abs_hi())
}

/// Sin : `z' = k·sin z` (cf. `sin_dd_step`), erreurs de [`QdTrig`].
#[inline(always)]
fn sin_qd_step(trig: &QdTrig, z: Cqd, e: f64, k: Cqd, e_k: f64) -> (Cqd, f64, f64) {
    let (Some((sx, cx)), Some((sh, ch))) = (trig.sin_cos(z.re), trig.sinh_cosh(z.im)) else {
        return (z, f64::INFINITY, f64::NAN);
    };
    let zn = k.mul(Cqd { re: sx.mul(ch), im: cx.mul(sh) });
    let (s, c, shf, chf) = (sx.0[0], cx.0[0], sh.0[0], ch.0[0]);
    let cos_abs = (c * c + shf * shf).sqrt();
    let sin1 = (s.abs() * chf + c.abs() * shf.abs()).max(SIN_FLOOR);
    let eps_sin = 2.0
        * (SIN_ERR * (1.0 + z.re.0[0].abs()) + HYP_ERR * (1.0 + z.im.0[0].abs()) + 2.0)
        * UQ
        * chf;
    let k_abs = k.abs_hi();
    let k1 = k.re.0[0].abs() + k.im.0[0].abs();
    let e2 = ((k_abs + e_k) * e * (cos_abs + 0.5 * e * chf * (1.0 + 2.0 * e))
        + (e_k + 4.0 * UQ * k1) * (sin1 + eps_sin)
        + k_abs * eps_sin)
        * (1.0 + 1e-12);
    (zn, e2, zn.abs_hi())
}

/// Barnsley (cf. `barnsley_dd_step`) : borne linéarisée, plafond 1e-3.
#[inline(always)]
fn barnsley_qd_step(z: Cqd, e: f64, c: Cqd, e_c: f64) -> (Cqd, f64, f64) {
    if z.re.0[0].abs() <= e {
        return (z, f64::INFINITY, f64::NAN);
    }
    let one = Qd::from_f64(1.0);
    let wx = if z.re.0[0] >= 0.0 { z.re.sub(one) } else { z.re.add(one) };
    let w = Cqd { re: wx, im: z.im };
    let zn = w.mul(c);
    let w1 = w.re.0[0].abs() + w.im.0[0].abs();
    let c1 = c.re.0[0].abs() + c.im.0[0].abs();
    let e2 = c.abs_hi() * e + w1 * (e_c + 4.0 * UQ * c1);
    (zn, if e2 < E_MAX_LIN { e2 } else { f64::INFINITY }, zn.abs_hi())
}

/// Voies de l'étage qd.
const QD_LANES: usize = 4;

/// Étage quad-double sur `points` (point exact de chaque pixel, arrondi en
/// qd). `None` pour un type sans étage qd.
pub fn qd_stage(params: &FractalParams, points: &[Cqd]) -> Option<Vec<(u32, Complex64, bool)>> {
    let (n, r) = (params.iteration_max, params.bailout);
    let seed = Cqd { re: Qd::from_f64(params.seed.re), im: Qd::from_f64(params.seed.im) };
    macro_rules! go {
        ($julia:expr, $z0:expr, $step:expr) => {
            Some(lanes_par::<Cqd, 9, QD_LANES, _>(points, $z0, $julia, UQ, n, r, $step))
        };
    }
    match params.fractal_type {
        // Sin et Magnet : voies vectorielles (`scalar_lanes`).
        FractalType::MandelbrotSin | FractalType::JuliaSin | FractalType::MagnetMandelbrot | FractalType::MagnetJulia
            if crate::render::scalar_lanes::enabled() =>
        {
            use crate::fractal::bytecode::direct_simd::QdV;
            use crate::render::scalar_lanes::{lanes_par, LaneKind};
            let s = (seed.re, seed.im);
            let (kind, julia, other) = match params.fractal_type {
                FractalType::MandelbrotSin => (LaneKind::Sin, false, s),
                FractalType::JuliaSin => (LaneKind::Sin, true, s),
                FractalType::MagnetMandelbrot => (LaneKind::Magnet, false, (Qd::ZERO, Qd::ZERO)),
                _ => (LaneKind::Magnet, true, s),
            };
            let pts: Vec<(Qd, Qd)> = points.iter().map(|p| (p.re, p.im)).collect();
            Some(lanes_par::<QdV<4>, 4>(kind, &pts, other, julia, n, r))
        }
        FractalType::BarnsleyJulia => go!(true, seed, move |z: Cqd, e: f64, _: Cqd| barnsley_qd_step(z, e, seed, 0.0)),
        FractalType::BarnsleyMandelbrot => {
            go!(true, seed, |z: Cqd, e: f64, c: Cqd| barnsley_qd_step(z, e, c, UQ * c.abs_hi()))
        }
        FractalType::AlphaMandelbrot => go!(false, seed, alpha_qd_step),
        FractalType::Mandelbulb => go!(false, seed, mandelbulb_qd_step),
        FractalType::MagnetJulia => {
            let s1 = Cqd { re: seed.re.sub(Qd::from_f64(1.0)), im: seed.im };
            let s2 = Cqd { re: seed.re.sub(Qd::from_f64(2.0)), im: seed.im };
            go!(true, seed, move |z: Cqd, e: f64, _: Cqd| magnet_qd_step(z, e, s1, s2, 0.0))
        }
        FractalType::MagnetMandelbrot => go!(false, Cqd::ZERO, |z: Cqd, e: f64, c: Cqd| {
            let s1 = Cqd { re: c.re.sub(Qd::from_f64(1.0)), im: c.im };
            let s2 = Cqd { re: c.re.sub(Qd::from_f64(2.0)), im: c.im };
            magnet_qd_step(z, e, s1, s2, UQ * c.abs_hi())
        }),
        FractalType::MandelbrotSin => {
            let trig = QdTrig::get();
            go!(false, seed, move |z: Cqd, e: f64, c: Cqd| sin_qd_step(trig, z, e, c, UQ * c.abs_hi()))
        }
        FractalType::JuliaSin => {
            let trig = QdTrig::get();
            go!(true, seed, move |z: Cqd, e: f64, _: Cqd| sin_qd_step(trig, z, e, seed, 0.0))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u32 = 512;

    fn to_gmp(x: Qd) -> Float {
        x.to_float(P)
    }

    fn err(x: Qd, exact: &Float) -> f64 {
        let mut d = to_gmp(x);
        d -= exact;
        d.to_f64().abs()
    }

    fn qd_of(v: f64, k: u32) -> Qd {
        // Argument à quatre composantes non nulles.
        let mut f = Float::with_val(P, v);
        f += v * 1.3e-17 * (k as f64 + 1.0).sin();
        f += v * 2.1e-34;
        f += v * 3.7e-51;
        qd_from_gmp(&f)
    }

    /// Erreurs contre MPFR 512 b ≤ borne / 8 (bornes déclarées de [`QdTrig`]).
    #[test]
    fn qd_trig_matches_mpfr_within_bounds() {
        let t = QdTrig::get();
        let (mut worst_s, mut worst_h) = (0.0f64, 0.0f64);
        for i in -3000..=3000 {
            let hi = i as f64 * 0.0251 + (i as f64 * 0.37).sin() * 1e-3;
            let x = qd_of(hi, i as u32);
            let gx = to_gmp(x);
            let (s, c) = t.sin_cos(x).unwrap();
            let bound = SIN_ERR * UQ * (1.0 + hi.abs());
            let e = err(s, &Float::with_val(P, gx.sin_ref())).max(err(c, &Float::with_val(P, gx.cos_ref())));
            worst_s = worst_s.max(e / bound);
            // sinh/cosh jusqu'à |y| = 60 (et petits arguments).
            for y in [hi * 0.8, hi * 1e-5, hi * 1e-12] {
                let y = qd_of(y, i as u32 + 7);
                let gy = to_gmp(y);
                let (sh, ch) = t.sinh_cosh(y).unwrap();
                let gch = Float::with_val(P, gy.cosh_ref());
                let bound = HYP_ERR * UQ * (1.0 + y.0[0].abs()) * gch.to_f64();
                let e = err(sh, &Float::with_val(P, gy.sinh_ref())).max(err(ch, &gch));
                worst_h = worst_h.max(e / bound);
            }
        }
        for k in [1i32, 7, 8, 16, 31, 32, 63, 64, 127, 128, 400, 4001, 1_000_001] {
            let x = qd_from_gmp(&(Float::with_val(P, Constant::Pi) * k / 64u32));
            let gx = to_gmp(x);
            let (s, c) = t.sin_cos(x).unwrap();
            let bound = SIN_ERR * UQ * (1.0 + x.0[0].abs());
            let e = err(s, &Float::with_val(P, gx.sin_ref())).max(err(c, &Float::with_val(P, gx.cos_ref())));
            worst_s = worst_s.max(e / bound);
        }
        for y in [300.0, 699.0, -650.5] {
            let y = qd_of(y, 3);
            let gy = to_gmp(y);
            let (sh, ch) = t.sinh_cosh(y).unwrap();
            let gch = Float::with_val(P, gy.cosh_ref());
            let rel = |x: Qd, g: &Float| {
                let mut d = to_gmp(x);
                d -= g;
                d /= &gch;
                d.to_f64().abs()
            };
            let e = rel(sh, &Float::with_val(P, gy.sinh_ref())).max(rel(ch, &gch));
            worst_h = worst_h.max(e / (HYP_ERR * UQ * (1.0 + y.0[0].abs())));
        }
        eprintln!("qd trig : pire erreur / borne : sin_cos {worst_s:.5}, sinh_cosh {worst_h:.5}");
        assert!(worst_s < 0.125, "sin_cos : {worst_s}");
        assert!(worst_h < 0.125, "sinh_cosh : {worst_h}");
        assert!(t.sin_cos(Qd::from_f64(2e9)).is_none());
        assert!(t.sinh_cosh(Qd::from_f64(-800.0)).is_none());
        let (sh, ch) = t.sinh_cosh(Qd::from_f64(-0.0)).unwrap();
        assert_eq!((sh.0[0], ch.0[0]), (0.0, 1.0));
    }

    /// Division qd : erreur relative < 2⁻²⁰⁴ (UQ / 16).
    #[test]
    fn qd_div_matches_rug() {
        let mut worst = 0.0f64;
        for i in 1..5000u32 {
            let a = qd_of((i as f64 * 0.731).sin() * 10f64.powi((i % 9) as i32 - 4), i);
            let b = qd_of((i as f64 * 1.37).cos() * 10f64.powi((i % 7) as i32 - 3) + 1e-9, i + 3);
            let want = Float::with_val(P, to_gmp(a) / to_gmp(b));
            let mut d = to_gmp(qd_div(a, b));
            d -= &want;
            d /= &want;
            worst = worst.max(d.to_f64().abs());
        }
        eprintln!("qd div : {:.1} b", -worst.log2());
        assert!(worst < UQ / 16.0, "{worst:e}");
    }
}
