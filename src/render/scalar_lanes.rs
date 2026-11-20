//! Étages double-double et quad-double des types **Sin** (Mandelbrot Sin,
//! Julia Sin) et **Magnet** (Mandelbrot, Julia) en **voies vectorielles** —
//! 2026-11-20. Un pas par type ([`sin_step`], [`magnet_step`]) sur un
//! pilote commun ([`drive`]).
//!
//! Magnet : même schéma, la division `n/q` devient UNE réciproque `1/|q|²`
//! en voies ([`div`]) ; qd scalaire ~600 ns/itération.
//!
//! Le pas Sin (`z' = k·sin z`) coûte un `sin_cos` et un `sinh_cosh` : en dd
//! scalaire ~190 ns/itération, en qd scalaire ~2 µs — une longue suite
//! d'opérations dd/qd, bornée par le DÉBIT d'un seul pixel à la fois
//! (bord Mandelbrot Sin 256×160 : 3,4 s de dd et 2,3 s de qd sur 7 s).
//! Ici `L` pixels avancent de front, chaque composante étant un vecteur
//! [`V<L>`] (arithmétique sans branchement de `bytecode::direct_simd` :
//! [`DdV`], [`QdV`]) ; les fonctions transcendantes sont écrites pour des
//! voies (réduction d'argument par voie, tables lues voie par voie).
//!
//! Mêmes réductions que `dd_trig` / `scalar_qd::QdTrig` (tables GMP,
//! `ln 2` de Cody-Waite, `e^±a` d'un seul polynôme en parties paire et
//! impaire, Estrin), mêmes bornes que `sin_dd_step` / `sin_qd_step` — les
//! constantes d'erreur sont vérifiées pour CETTE arithmétique contre MPFR
//! 512 b (`sin_lanes_trig_matches_mpfr_within_bounds`) : les bits diffèrent
//! des versions scalaires (renormalisation qd sans branchement, ordre des
//! produits), pas le modèle d'erreur.
//!
//! Un pixel déclaré fiable a un compte EXACT (borne rigoureuse), donc le
//! compte du GMP pur ; son z final n'est garanti qu'à la borne près.

use num_complex::Complex64;
use rayon::prelude::*;
use rug::float::Constant;
use rug::Float;

use crate::fractal::bytecode::direct_dd::{dd_from_gmp, qd_from_gmp, DD_UNIT};
use crate::fractal::bytecode::direct_simd::{DdV, LaneNum, QdV, V};
use crate::fractal::cycle::CycleCut;
use crate::fractal::perturbation::dd::DoubleDouble as Dd;
use crate::fractal::perturbation::qd::Qd;
use crate::render::scalar_qd::{step_ok, UQ};

const SIN_MAX_ARG: f64 = 1e9;
const HYP_MAX_ARG: f64 = 700.0;
const SIN_FLOOR: f64 = 1e-60;
/// Coefficients au plus (qd : 13).
const MAX_COEF: usize = 13;

/// Tables d'un type de nombre (construites une fois en GMP 512 b).
pub(crate) struct SinTables<S> {
    pi64: S,
    sin_tab: Vec<S>,
    cos_tab: Vec<S>,
    /// `(−1)ⁱ/(2i+1)!`, `(−1)ⁱ/(2i)!`.
    sin_coef: Vec<S>,
    cos_coef: Vec<S>,
    ln2_hi: f64,
    ln2_lo: S,
    /// Niveaux de réduction de l'exponentielle : `(échelle 2^k, J, exp(j/2^k)
    /// indexé j + J)`.
    exp_levels: Vec<(f64, i32, Vec<S>)>,
    /// `1/(2i)!`, `1/(2i+1)!`.
    exp_even: Vec<S>,
    exp_odd: Vec<S>,
}

fn build_tables<S>(conv: impl Fn(&Float) -> S, sin_terms: usize, level_bits: &[u32], exp_terms: usize) -> SinTables<S> {
    const P: u32 = 512;
    let pi = Float::with_val(P, Constant::Pi);
    let pi64 = Float::with_val(P, &pi / 64u32);
    let inv = |n: u32, neg: bool| -> S {
        let f = Float::with_val(P, rug::Integer::from(rug::Integer::factorial(n)));
        let mut v = Float::with_val(P, 1u32) / f;
        if neg {
            v = -v;
        }
        conv(&v)
    };
    let ln2 = Float::with_val(P, Constant::Log2);
    let ln2_hi = Float::with_val(40, &ln2).to_f64();
    let mut exp_levels = Vec::new();
    let mut prev = 0.0f64;
    for &b in level_bits {
        let scale = 2f64.powi(b as i32);
        // |s| ≤ ln2/2 au premier niveau, ≤ 1/(2·échelle précédente) ensuite.
        let jmax = if prev == 0.0 { (scale * 0.3466).ceil() } else { scale / (2.0 * prev) } as i32 + 1;
        let tab = (-jmax..=jmax)
            .map(|j| conv(&Float::with_val(P, Float::with_val(P, j) / scale).exp()))
            .collect();
        exp_levels.push((scale, jmax, tab));
        prev = scale;
    }
    SinTables {
        pi64: conv(&pi64),
        sin_tab: (0..128u32).map(|k| conv(&Float::with_val(P, &pi64 * k).sin())).collect(),
        cos_tab: (0..128u32).map(|k| conv(&Float::with_val(P, &pi64 * k).cos())).collect(),
        sin_coef: (0..sin_terms).map(|i| inv(2 * i as u32 + 1, i % 2 == 1)).collect(),
        cos_coef: (0..sin_terms).map(|i| inv(2 * i as u32, i % 2 == 1)).collect(),
        ln2_hi,
        ln2_lo: conv(&Float::with_val(P, &ln2 - ln2_hi)),
        exp_levels,
        exp_even: (0..exp_terms / 2).map(|i| inv(2 * i as u32, false)).collect(),
        exp_odd: (0..exp_terms / 2).map(|i| inv(2 * i as u32 + 1, false)).collect(),
    }
}

/// Nombre en voies des étages Sin : unité et constantes d'erreur des
/// fonctions transcendantes, tables.
pub(crate) trait SinNum<const L: usize>: LaneNum<L> + Send + Sync
where
    Self::Scalar: Send + Sync + 'static,
{
    /// Unité d'arrondi des bornes (marge comprise).
    const U: f64;
    /// `|sin x − ŝ|, |cos x − ĉ| ≤ SIN_ERR·U·(1 + |x|)`.
    const SIN_ERR: f64;
    /// `|sinh y − ŝh|, |cosh y − ĉh| ≤ HYP_ERR·U·(1 + |y|)·cosh y`.
    const HYP_ERR: f64;
    fn from_hi(v: V<L>) -> Self;
    /// Composantes de la voie `l` (dd : `[hi, lo, 0, 0]`).
    fn words(&self, l: usize) -> [f64; 4];
    fn tables() -> &'static SinTables<Self::Scalar>;
    fn scalar_zero() -> Self::Scalar;
    /// Composantes d'un nombre (dd : 2, qd : 4).
    const WORDS: usize;
}

impl<const L: usize> SinNum<L> for DdV<L> {
    const U: f64 = DD_UNIT;
    const WORDS: usize = 2;
    const SIN_ERR: f64 = 16.0;
    const HYP_ERR: f64 = 16.0;
    #[inline(always)]
    fn from_hi(v: V<L>) -> Self {
        DdV::from_parts(v, V::splat(0.0))
    }
    #[inline(always)]
    fn words(&self, l: usize) -> [f64; 4] {
        [self.hi().0[l], self.lo().0[l], 0.0, 0.0]
    }
    fn scalar_zero() -> Dd {
        Dd::ZERO
    }
    fn tables() -> &'static SinTables<Dd> {
        static T: std::sync::OnceLock<SinTables<Dd>> = std::sync::OnceLock::new();
        T.get_or_init(|| build_tables(dd_from_gmp, 7, &[8, 16], 6))
    }
}

impl<const L: usize> SinNum<L> for QdV<L> {
    const U: f64 = UQ;
    const WORDS: usize = 4;
    const SIN_ERR: f64 = 4.0;
    const HYP_ERR: f64 = 4.0;
    #[inline(always)]
    fn from_hi(v: V<L>) -> Self {
        let z = V::splat(0.0);
        QdV([v, z, z, z])
    }
    #[inline(always)]
    fn words(&self, l: usize) -> [f64; 4] {
        [self.0[0].0[l], self.0[1].0[l], self.0[2].0[l], self.0[3].0[l]]
    }
    fn scalar_zero() -> Qd {
        Qd::ZERO
    }
    fn tables() -> &'static SinTables<Qd> {
        static T: std::sync::OnceLock<SinTables<Qd>> = std::sync::OnceLock::new();
        T.get_or_init(|| build_tables(qd_from_gmp, 13, &[8, 16, 24], 8))
    }
}

#[inline(always)]
fn vmap<const L: usize>(v: V<L>, f: impl Fn(f64) -> f64) -> V<L> {
    let mut r = [0.0; L];
    for l in 0..L {
        r[l] = f(v.0[l]);
    }
    V(r)
}

/// `Σ aᵢ·uⁱ` à la Estrin.
#[inline(always)]
fn estrin<N: LaneNum<L>, const L: usize>(a: &[N::Scalar], u: N) -> N {
    let mut b: [N; MAX_COEF] = [N::splat(a[0]); MAX_COEF];
    for (i, &c) in a.iter().enumerate() {
        b[i] = N::splat(c);
    }
    let mut n = a.len();
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

/// Table lue voie par voie.
#[inline(always)]
fn gather<N: LaneNum<L>, const L: usize>(tab: &[N::Scalar], idx: [usize; L]) -> N {
    let mut r = N::splat(tab[idx[0]]);
    for l in 1..L {
        r.set_lane(l, tab[idx[l]]);
    }
    r
}

/// `(sin x, cos x)` par voie ; voies hors domaine (`|x| ≥ 1e9`) : `ok = false`.
#[inline(always)]
pub(crate) fn sin_cos<N: SinNum<L>, const L: usize>(x: N) -> (N, N, [bool; L])
where
    N::Scalar: Send + Sync + 'static,
{
    let t = N::tables();
    let xh = x.hi();
    let mut ok = [true; L];
    let mut k = [0.0; L];
    let mut j = [0usize; L];
    for l in 0..L {
        ok[l] = xh.0[l].abs() < SIN_MAX_ARG;
        k[l] = if ok[l] { (xh.0[l] * (64.0 / std::f64::consts::PI)).round() } else { 0.0 };
        j[l] = (k[l] as i64).rem_euclid(128) as usize;
    }
    let tt = x.sub(N::splat(t.pi64).mul(N::from_hi(V(k))));
    let u = tt.sqr();
    let s = estrin::<N, L>(&t.sin_coef, u).mul(tt);
    let c = estrin::<N, L>(&t.cos_coef, u);
    let sj: N = gather(&t.sin_tab, j);
    let cj: N = gather(&t.cos_tab, j);
    (sj.mul(c).add(cj.mul(s)), cj.mul(c).sub(sj.mul(s)), ok)
}

/// `(exp a, exp −a)` par voie, `0 ≤ a < 700`.
#[inline(always)]
fn exp_pm<N: SinNum<L>, const L: usize>(a: N) -> (N, N)
where
    N::Scalar: Send + Sync + 'static,
{
    let t = N::tables();
    let m = vmap(a.hi(), |h| (h * std::f64::consts::LOG2_E).round());
    let mut s = a
        .sub(N::from_hi(vmap(m, |m| m * t.ln2_hi)))
        .sub(N::splat(t.ln2_lo).mul(N::from_hi(m)));
    let mut pos: Option<(N, N)> = None;
    for (scale, jmax, tab) in &t.exp_levels {
        let jf = vmap(s.hi(), |h| (h * scale).round().clamp(-(*jmax as f64), *jmax as f64));
        s = s.sub(N::from_hi(vmap(jf, |j| j / scale)));
        let mut ip = [0usize; L];
        let mut im = [0usize; L];
        for l in 0..L {
            ip[l] = (jmax + jf.0[l] as i32) as usize;
            im[l] = (jmax - jf.0[l] as i32) as usize;
        }
        let (tp, tm): (N, N) = (gather(tab, ip), gather(tab, im));
        pos = Some(match pos {
            None => (tp, tm),
            Some((p, q)) => (p.mul(tp), q.mul(tm)),
        });
    }
    let (tp, tm) = pos.expect("au moins un niveau");
    let r2 = s.sqr();
    let ev = estrin::<N, L>(&t.exp_even, r2);
    let od = estrin::<N, L>(&t.exp_odd, r2).mul(s);
    (
        ev.add(od).mul(tp).mul(N::from_hi(vmap(m, |m| 2f64.powi(m as i32)))),
        ev.sub(od).mul(tm).mul(N::from_hi(vmap(m, |m| 2f64.powi(-(m as i32))))),
    )
}

/// `(sinh y, cosh y)` par voie ; voies hors domaine (`|y| ≥ 700`) : `ok = false`.
#[inline(always)]
pub(crate) fn sinh_cosh<N: SinNum<L>, const L: usize>(y: N) -> (N, N, [bool; L])
where
    N::Scalar: Send + Sync + 'static,
{
    let yh = y.hi();
    let mut a = y.abs();
    let mut ok = [true; L];
    let mut sign = [1.0; L];
    let ah = a.hi();
    for l in 0..L {
        ok[l] = ah.0[l] < HYP_MAX_ARG;
        if !ok[l] {
            a.set_lane(l, N::scalar_zero());
        }
        if yh.0[l] < 0.0 {
            sign[l] = -1.0;
        }
    }
    let (e, ie) = exp_pm::<N, L>(a);
    let half = N::from_hi(V::splat(0.5));
    let sh = e.sub(ie).mul(half).mul(N::from_hi(V(sign)));
    let ch = e.add(ie).mul(half);
    (sh, ch, ok)
}

/// Division par voie `a / b` : quotients partiels successifs (`WORDS` pour
/// un nombre à `WORDS` composantes), restes exacts au modèle d'erreur de
/// l'arithmétique près. Erreur relative vérifiée contre MPFR
/// (`lane_div_matches_mpfr`).
#[inline(always)]
pub(crate) fn div<N: SinNum<L>, const L: usize>(a: N, b: N) -> N
where
    N::Scalar: Send + Sync + 'static,
{
    let bh = b.hi();
    let mut q = N::from_hi(a.hi().zip_div(bh));
    let mut r = a.sub(b.mul(q));
    for _ in 1..N::WORDS {
        let qk = N::from_hi(r.hi().zip_div(bh));
        q = q.add(qk);
        r = r.sub(b.mul(qk));
    }
    q
}

trait ZipDiv {
    fn zip_div(self, o: Self) -> Self;
}
impl<const L: usize> ZipDiv for V<L> {
    #[inline(always)]
    fn zip_div(self, o: Self) -> Self {
        let mut r = [0.0; L];
        for l in 0..L {
            r[l] = self.0[l] / o.0[l];
        }
        V(r)
    }
}

/// Pas Sin vectoriel `z' = k·sin z` (cf. `sin_dd_step` / `sin_qd_step`) :
/// met à jour `(x, y)` et la borne `e`, renvoie `|z'|` par voie (NaN hors
/// domaine des fonctions).
#[inline(always)]
fn sin_step<N: SinNum<L>, const L: usize>(
    x: &mut N,
    y: &mut N,
    kx: &N,
    ky: &N,
    e: &mut [f64; L],
    e_k: &[f64; L],
) -> [f64; L]
where
    N::Scalar: Send + Sync + 'static,
{
    let u = N::U;
    let (sx, cx, ok1) = sin_cos::<N, L>(*x);
    let (sh, ch, ok2) = sinh_cosh::<N, L>(*y);
    let wr = sx.mul(ch);
    let wi = cx.mul(sh);
    let nx = kx.mul(wr).sub(ky.mul(wi));
    let ny = kx.mul(wi).add(ky.mul(wr));
    let (xh, yh) = (x.hi(), y.hi());
    let (sv, cv, shv, chv) = (sx.hi(), cx.hi(), sh.hi(), ch.hi());
    let (kxh, kyh) = (kx.hi(), ky.hi());
    let (nxh, nyh) = (nx.hi(), ny.hi());
    *x = nx;
    *y = ny;
    let mut norm = [0.0; L];
    for l in 0..L {
        let (s, c, shf, chf) = (sv.0[l], cv.0[l], shv.0[l], chv.0[l]);
        let cos_abs = (c * c + shf * shf).sqrt();
        let sin1 = (s.abs() * chf + c.abs() * shf.abs()).max(SIN_FLOOR);
        let eps_sin = 2.0
            * (N::SIN_ERR * (1.0 + xh.0[l].abs()) + N::HYP_ERR * (1.0 + yh.0[l].abs()) + 2.0)
            * u
            * chf;
        let (kr, ki) = (kxh.0[l], kyh.0[l]);
        let k_abs = (kr * kr + ki * ki).sqrt();
        let k1 = kr.abs() + ki.abs();
        let el = e[l];
        let ok = ok1[l] && ok2[l];
        e[l] = if ok {
            ((k_abs + e_k[l]) * el * (cos_abs + 0.5 * el * chf * (1.0 + 2.0 * el))
                + (e_k[l] + 4.0 * u * k1) * (sin1 + eps_sin)
                + k_abs * eps_sin)
                * (1.0 + 1e-12)
        } else {
            f64::INFINITY
        };
        let (h, v) = (nxh.0[l], nyh.0[l]);
        norm[l] = if ok { (h * h + v * v).sqrt() } else { f64::NAN };
    }
    norm
}

/// Pas Magnet vectoriel (cf. `magnet_dd_step` / `scalar_qd::magnet_qd_step`,
/// arrondis relatifs aux OPÉRANDES) : `n = z² + (k − 1)`, `q = 2z + (k − 2)`,
/// `z' = (n/q)²` avec UNE division (`1/|q|²`). `k` = la graine (Julia) ou le
/// pixel (Mandelbrot, erreur `e_k` additive dans `k − 1`, `k − 2`).
/// `E' = ∞` si `|q|` n'est pas séparé de 0 (le GMP décide la sortie au pôle).
#[inline(always)]
fn magnet_step<N: SinNum<L>, const L: usize>(
    x: &mut N,
    y: &mut N,
    kx: &N,
    ky: &N,
    e: &mut [f64; L],
    e_k: &[f64; L],
) -> [f64; L]
where
    N::Scalar: Send + Sync + 'static,
{
    let u = N::U;
    let one = N::from_hi(V::splat(1.0));
    let two = N::from_hi(V::splat(2.0));
    let (s1x, s2x) = (kx.sub(one), kx.sub(two));
    let nr = x.sqr().sub(y.sqr()).add(s1x);
    let ni = x.mul(*y).twice().add(*ky);
    let qr = x.twice().add(s2x);
    let qi = y.twice().add(*ky);
    let qq = qr.sqr().add(qi.sqr());
    let pr = nr.mul(qr).add(ni.mul(qi));
    let pi = ni.mul(qr).sub(nr.mul(qi));
    let iqq = div::<N, L>(one, qq);
    let rr = pr.mul(iqq);
    let ri = pi.mul(iqq);
    let nx = rr.sqr().sub(ri.sqr());
    let ny = rr.mul(ri).twice();
    let (xh, yh) = (x.hi(), y.hi());
    let (nrh, nih, qqh) = (nr.hi(), ni.hi(), qq.hi());
    let (s1h, s2h, kyh) = (s1x.hi(), s2x.hi(), ky.hi());
    let (rrh, rih) = (rr.hi(), ri.hi());
    let (nxh, nyh) = (nx.hi(), ny.hi());
    *x = nx;
    *y = ny;
    let mut norm = [0.0; L];
    for l in 0..L {
        let (zr, zi) = (xh.0[l], yh.0[l]);
        let z2 = zr * zr + zi * zi;
        let n_abs = (nrh.0[l] * nrh.0[l] + nih.0[l] * nih.0[l]).sqrt();
        let s1a = (s1h.0[l] * s1h.0[l] + kyh.0[l] * kyh.0[l]).sqrt();
        let s2a = (s2h.0[l] * s2h.0[l] + kyh.0[l] * kyh.0[l]).sqrt();
        let q_abs = qqh.0[l].sqrt();
        let (r0, r1) = (rrh.0[l], rih.0[l]);
        let r_abs = (r0 * r0 + r1 * r1).sqrt();
        let el = e[l];
        let rho_n = u * (3.0 * z2 + n_abs + s1a) + e_k[l];
        let rho_q = u * (2.0 * z2.sqrt() + q_abs + s2a) + e_k[l];
        let den = q_abs - 2.0 * el - rho_q;
        let (wr, wi) = (zr - r0, zi - r1);
        let w = (wr * wr + wi * wi).sqrt();
        let er = (el * (2.0 * w + el) + rho_n + r_abs * rho_q + 8.0 * u * r_abs * (2.0 * el + rho_q)) / den
            + 8.0 * u * r_abs;
        e[l] = if den > 1e-10 && qqh.0[l] >= 1e-20 {
            er * (2.0 * r_abs + er) + 2.0 * u * r_abs * r_abs
        } else {
            f64::INFINITY
        };
        let (h, v) = (nxh.0[l], nyh.0[l]);
        norm[l] = (h * h + v * v).sqrt();
    }
    norm
}

/// Itération tracée de `points`, `L` pixels de front, par le pas `step`.
/// `julia` : le point est `z₀` (borne initiale `U·|z₀|`) et `k` = `other` ;
/// sinon `z₀` = `other` et `k` = le point (erreur `e_k = U·|k|`).
/// Sémantique d'un pixel : celle des pas scalaires dans `scalar_qd::lanes`
/// (sortie, fiabilité, raccourci de cycle sur l'état complet borne comprise).
fn drive<N, const L: usize>(
    points: &[(N::Scalar, N::Scalar)],
    other: (N::Scalar, N::Scalar),
    julia: bool,
    iteration_max: u32,
    bailout: f64,
    step: impl Fn(&mut N, &mut N, &N, &N, &mut [f64; L], &[f64; L]) -> [f64; L],
) -> Vec<(u32, Complex64, bool)>
where
    N: SinNum<L>,
    N::Scalar: Send + Sync + 'static,
{
    const FREE: usize = usize::MAX;
    let u = N::U;
    let b2 = bailout * bailout;
    let mut out = vec![(0u32, Complex64::new(0.0, 0.0), false); points.len()];
    let mut next = 0usize;
    let mut idx = [FREE; L];
    let mut x = N::splat(other.0);
    let mut y = N::splat(other.1);
    let mut kx = N::splat(other.0);
    let mut ky = N::splat(other.1);
    let mut e = [0.0f64; L];
    let mut e_k = [0.0f64; L];
    let mut it = [0u32; L];
    let mut cyc: [CycleCut<9>; L] = std::array::from_fn(|_| CycleCut::new());
    loop {
        for l in 0..L {
            loop {
                if idx[l] != FREE {
                    let (h, v) = (x.hi().0[l], y.hi().0[l]);
                    if it[l] < iteration_max && h * h + v * v < b2 {
                        break;
                    }
                    out[idx[l]] = (it[l], Complex64::new(h, v), true);
                    idx[l] = FREE;
                }
                if next == points.len() {
                    break;
                }
                let p = points[next];
                idx[l] = next;
                cyc[l] = CycleCut::new();
                it[l] = 0;
                let ph = (N::splat(p.0).hi().0[0], N::splat(p.1).hi().0[0]);
                let pa = (ph.0 * ph.0 + ph.1 * ph.1).sqrt();
                if julia {
                    x.set_lane(l, p.0);
                    y.set_lane(l, p.1);
                    e[l] = u * pa;
                    e_k[l] = 0.0;
                } else {
                    x.set_lane(l, other.0);
                    y.set_lane(l, other.1);
                    kx.set_lane(l, p.0);
                    ky.set_lane(l, p.1);
                    e[l] = 0.0;
                    e_k[l] = u * pa;
                }
                next += 1;
            }
        }
        if idx.iter().all(|&k| k == FREE) {
            break;
        }
        // Un pas sur toutes les voies (une voie libre calcule pour rien).
        let norm = step(&mut x, &mut y, &kx, &ky, &mut e, &e_k);
        for l in 0..L {
            if idx[l] == FREE {
                continue;
            }
            it[l] += 1;
            if !step_ok(norm[l], bailout, e[l]) {
                out[idx[l]] = (it[l], Complex64::new(x.hi().0[l], y.hi().0[l]), false);
                idx[l] = FREE;
            } else {
                let (a, b) = (x.words(l), y.words(l));
                cyc[l].observe(&mut it[l], iteration_max, [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], e[l]]);
            }
        }
    }
    out
}

/// Type traité en voies.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum LaneKind {
    Sin,
    Magnet,
}

/// Itération tracée de `points` par le pas `kind` (cf. [`drive`]).
pub(crate) fn kind_lanes<N, const L: usize>(
    kind: LaneKind,
    points: &[(N::Scalar, N::Scalar)],
    other: (N::Scalar, N::Scalar),
    julia: bool,
    iteration_max: u32,
    bailout: f64,
) -> Vec<(u32, Complex64, bool)>
where
    N: SinNum<L>,
    N::Scalar: Send + Sync + 'static,
{
    match kind {
        LaneKind::Sin => drive::<N, L>(points, other, julia, iteration_max, bailout, sin_step::<N, L>),
        LaneKind::Magnet => drive::<N, L>(points, other, julia, iteration_max, bailout, magnet_step::<N, L>),
    }
}

/// `FRACTALL_SIN_LANES=0` : étages Sin et Magnet scalaires (`*_dd_step` /
/// `*_qd_step`) au lieu des voies vectorielles (A/B). Lu une fois.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FRACTALL_SIN_LANES").map_or(true, |v| v != "0"))
}

/// [`kind_lanes`] en parallèle par paquets de `4·L` pixels.
pub(crate) fn lanes_par<N, const L: usize>(
    kind: LaneKind,
    points: &[(N::Scalar, N::Scalar)],
    other: (N::Scalar, N::Scalar),
    julia: bool,
    iteration_max: u32,
    bailout: f64,
) -> Vec<(u32, Complex64, bool)>
where
    N: SinNum<L>,
    N::Scalar: Send + Sync + 'static,
{
    // Paquets LONGS : une voie qui finit tard (pixel plafonné, 5000 pas)
    // laisse les autres voies vides en fin de paquet — à 4·L pixels par
    // paquet, l'étage dd Magnet vectoriel était 2× plus LENT que le scalaire.
    // Assez de paquets pour occuper les fils.
    let chunk = (points.len() / (2 * rayon::current_num_threads().max(1))).clamp(4 * L, 32 * L);
    points
        .par_chunks(chunk)
        .flat_map_iter(|ch| kind_lanes::<N, L>(kind, ch, other, julia, iteration_max, bailout))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u32 = 512;

    fn lane_float<N: SinNum<L>, const L: usize>(v: &N, l: usize) -> Float
    where
        N::Scalar: Send + Sync + 'static,
    {
        let mut f = Float::with_val(P, 0);
        for w in v.words(l) {
            f += w;
        }
        f
    }

    /// Erreurs des fonctions en voies contre MPFR 512 b ≤ borne / 8.
    fn check<N, const L: usize>(conv: impl Fn(&Float) -> N::Scalar) -> (f64, f64)
    where
        N: SinNum<L>,
        N::Scalar: Send + Sync + 'static,
    {
        let (mut ws, mut wh) = (0.0f64, 0.0f64);
        let arg = |i: i32| -> Float {
            let hi = i as f64 * 0.0251 + (i as f64 * 0.37).sin() * 1e-3;
            let mut f = Float::with_val(P, hi);
            f += hi * 1.3e-17 * (i as f64).cos();
            f += hi * 2.1e-34;
            f += hi * 3.7e-51;
            f
        };
        let mut args: Vec<Float> = (-3000..=3000).map(arg).collect();
        for k in [1i32, 7, 8, 16, 31, 32, 63, 64, 127, 128, 400, 4001, 1_000_001] {
            args.push(Float::with_val(P, Float::with_val(P, Constant::Pi) * k / 64u32));
        }
        for ch in args.chunks(L) {
            let mut x = N::splat(conv(&ch[0]));
            for (l, a) in ch.iter().enumerate() {
                x.set_lane(l, conv(a));
            }
            let (s, c, _) = sin_cos::<N, L>(x);
            for l in 0..ch.len() {
                let gx = lane_float::<N, L>(&x, l);
                let bound = N::SIN_ERR * N::U * (1.0 + gx.to_f64().abs());
                for (got, want) in [(&s, Float::with_val(P, gx.sin_ref())), (&c, Float::with_val(P, gx.cos_ref()))] {
                    let mut d = lane_float::<N, L>(got, l);
                    d -= &want;
                    ws = ws.max(d.to_f64().abs() / bound);
                }
            }
            // sinh/cosh : 0,8·x (jusqu'à |y| ≈ 60), petits et grands arguments.
            for f in [0.8, 1e-5, 1e-12, 8.0] {
                let mut y = N::splat(conv(&ch[0]));
                for (l, a) in ch.iter().enumerate() {
                    y.set_lane(l, conv(&Float::with_val(P, a * f)));
                }
                let (sh, chh, ok) = sinh_cosh::<N, L>(y);
                for l in 0..ch.len() {
                    if !ok[l] {
                        continue;
                    }
                    let gy = lane_float::<N, L>(&y, l);
                    let gch = Float::with_val(P, gy.cosh_ref());
                    let bound = N::HYP_ERR * N::U * (1.0 + gy.to_f64().abs()) * gch.to_f64();
                    for (got, want) in [(&sh, Float::with_val(P, gy.sinh_ref())), (&chh, gch.clone())] {
                        let mut d = lane_float::<N, L>(got, l);
                        d -= &want;
                        wh = wh.max(d.to_f64().abs() / bound);
                    }
                }
            }
        }
        (ws, wh)
    }

    /// Division en voies : erreur relative < U/4 (le pas Magnet en compte 8U
    /// pour tout le quotient).
    #[test]
    fn lane_div_matches_mpfr() {
        fn worst<N: SinNum<L>, const L: usize>(conv: impl Fn(&Float) -> N::Scalar) -> f64
        where
            N::Scalar: Send + Sync + 'static,
        {
            let mut w = 0.0f64;
            let val = |i: u32, s: f64| -> Float {
                let hi = (i as f64 * 0.731 + s).sin() * 10f64.powi((i % 9) as i32 - 4) + 1e-9;
                let mut f = Float::with_val(P, hi);
                f += hi * 1.7e-17 * (i as f64).cos();
                f += hi * 2.3e-34;
                f += hi * 3.1e-51;
                f
            };
            for i in (1..4000u32).step_by(L) {
                let (mut a, mut b) = (N::splat(conv(&val(i, 0.0))), N::splat(conv(&val(i, 1.0))));
                for l in 0..L {
                    a.set_lane(l, conv(&val(i + l as u32, 0.0)));
                    b.set_lane(l, conv(&val(i + l as u32 * 3, 2.0)));
                }
                let q = div::<N, L>(a, b);
                for l in 0..L {
                    let want = Float::with_val(P, lane_float::<N, L>(&a, l) / lane_float::<N, L>(&b, l));
                    let mut d = lane_float::<N, L>(&q, l);
                    d -= &want;
                    d /= &want;
                    w = w.max(d.to_f64().abs() / N::U);
                }
            }
            w
        }
        let (d, q) = (worst::<DdV<8>, 8>(dd_from_gmp), worst::<QdV<4>, 4>(qd_from_gmp));
        eprintln!("division en voies : pire erreur / U : dd {d:.4}, qd {q:.6}");
        assert!(d < 0.25 && q < 0.25, "dd {d} qd {q}");
    }

    #[test]
    fn sin_lanes_trig_matches_mpfr_within_bounds() {
        let (s, h) = check::<DdV<8>, 8>(dd_from_gmp);
        eprintln!("dd voies : pire erreur / borne sin_cos {s:.5}, sinh_cosh {h:.5}");
        assert!(s < 0.125 && h < 0.125, "dd : {s} {h}");
        let (s, h) = check::<QdV<4>, 4>(qd_from_gmp);
        eprintln!("qd voies : pire erreur / borne sin_cos {s:.5}, sinh_cosh {h:.5}");
        assert!(s < 0.125 && h < 0.125, "qd : {s} {h}");
    }
}
