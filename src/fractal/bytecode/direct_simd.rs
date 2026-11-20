//! Étages DIRECTS double-double / quad-double en **voies vectorielles**
//! (2026-11-19).
//!
//! L'itération directe d'un pixel en dd/qd est une longue chaîne dépendante
//! (`two_prod`, `two_sum`, renormalisation) : un pixel à la fois, elle est
//! bornée par la LATENCE et laisse les unités vectorielles vides. Ici `L`
//! pixels avancent de front, chaque composante d'un nombre étant un vecteur
//! [`V<L>`] de `L` flottants (une voie = un pixel, rechargée dès qu'elle
//! sort) : chaque opération dd/qd s'applique aux `L` voies à la fois et LLVM
//! la compile en instructions vectorielles (AVX2/AVX-512 en `target-cpu=
//! native`, SSE2 en baseline x86-64).
//!
//! Exigence de la vectorisation : AUCUN branchement dépendant des données
//! dans l'arithmétique.
//! - Double-double : les opérations de `perturbation::dd` n'en ont pas — les
//!   voies dd font EXACTEMENT ses opérations, résultats identiques au bit
//!   près à la boucle mono-pixel [`super::direct_dd::iterate_direct_dd`]
//!   (verrou `dd_lanes_match_direct_dd_bit_for_bit`).
//! - Quad-double : la renormalisation de QD (`perturbation::qd::renorm5`)
//!   branche sur les composantes NULLES. [`QdV`] utilise la variante sans
//!   branchement (toujours le chemin « composantes non nulles ») : quand une
//!   composante est nulle, `quick_two_sum(0, b) = (b, 0)` est exact, la
//!   valeur représentée est conservée (plus exactement que le chemin à
//!   branches, qui arrondissait le dernier terme dans le précédent) — même
//!   modèle d'erreur « sloppy » relatif aux opérandes, vérifié contre `rug`
//!   (`qdv_ops_match_rug_within_qd_unit`, opérandes à composantes nulles
//!   compris). Les bits de z peuvent donc différer de `perturbation::qd`
//!   dans ces cas ; la borne tracée (`U_QD` = 2⁻²⁰⁶, marge ×16 sur l'erreur
//!   mesurée) couvre l'arithmétique réellement faite.
//!
//! `two_prod` (cf. [`two_prod`] et `crate::fractal::eft`) : FMA en ligne si
//! le binaire est compilé avec `fma` ; dans un binaire baseline x86-64 sur
//! CPU à AVX2 + FMA, les étages passent par une copie compilée avec
//! `avx2,fma` (aiguillage à l'exécution, [`direct_dd_lanes`]) où le FMA est en
//! ligne et vectorisé — éléphants 1e7 60 000 it. : étages directs 6,0 → 3,2 s ;
//! sans FMA, découpage de Dekker. Tous EXACTS (produit sans erreur) : mêmes
//! bits (verrou `two_prod_modes_give_identical_lanes`). Sans `fma` à la
//! compilation, `f64::mul_add` est un appel de bibliothèque — ni inliné ni
//! vectorisé, et logiciel sur un CPU sans FMA.

use num_complex::Complex64;

use super::direct_dd::{phase_supported, DirectDdResult, DD_UNIT, E_MAX};
use super::{Op, Phase};
use crate::fractal::eft;
use crate::fractal::perturbation::dd::DoubleDouble as Dd;
use crate::fractal::perturbation::qd::Qd;

/// `L` flottants traités ensemble (une voie par pixel).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V<const L: usize>(pub [f64; L]);

impl<const L: usize> V<L> {
    #[inline(always)]
    pub fn splat(v: f64) -> Self {
        V([v; L])
    }
    #[inline(always)]
    fn zip(self, o: Self, f: impl Fn(f64, f64) -> f64) -> Self {
        let mut r = [0.0; L];
        for l in 0..L {
            r[l] = f(self.0[l], o.0[l]);
        }
        V(r)
    }
    #[inline(always)]
    fn map(self, f: impl Fn(f64) -> f64) -> Self {
        let mut r = [0.0; L];
        for l in 0..L {
            r[l] = f(self.0[l]);
        }
        V(r)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        self.map(f64::sqrt)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        self.map(f64::abs)
    }
    /// `-x` sur les voies où `sign < 0`, `x` ailleurs (sans branchement).
    #[inline(always)]
    fn neg_where_negative(self, sign: Self) -> Self {
        let mut r = [0.0; L];
        for l in 0..L {
            r[l] = if sign.0[l] < 0.0 { -self.0[l] } else { self.0[l] };
        }
        V(r)
    }
}

impl<const L: usize> std::ops::Add for V<L> {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        self.zip(o, |a, b| a + b)
    }
}
impl<const L: usize> std::ops::Sub for V<L> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        self.zip(o, |a, b| a - b)
    }
}
impl<const L: usize> std::ops::Mul for V<L> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        self.zip(o, |a, b| a * b)
    }
}
impl<const L: usize> std::ops::Neg for V<L> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        self.map(|a| -a)
    }
}

/// Somme sans erreur (Knuth).
#[inline(always)]
fn two_sum<const L: usize>(a: V<L>, b: V<L>) -> (V<L>, V<L>) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// Somme sans erreur rapide, `|a| ≥ |b|` (ou `a = 0` : exacte aussi).
#[inline(always)]
fn quick_two_sum<const L: usize>(a: V<L>, b: V<L>) -> (V<L>, V<L>) {
    let s = a + b;
    (s, b - (s - a))
}

/// Mode du produit sans erreur des voies (paramètre `M` de [`DdV`] /
/// [`QdV`]) : choix à l'exécution (défaut de production).
pub const TP_AUTO: u8 = 0;
/// `mul_add` en ligne : variante compilée avec `avx2,fma` (cf.
/// [`direct_dd_lanes`]), où c'est l'instruction `vfmadd`, vectorisée.
pub const TP_FMA: u8 = 1;
/// Découpage de Dekker forcé (chemin des CPU sans FMA ; tests).
pub const TP_DEKKER: u8 = 2;

/// Produit sans erreur, mêmes bits dans tous les modes :
/// - [`TP_FMA`] ou FMA connu à la compilation : `mul_add` en ligne ;
/// - [`TP_AUTO`], baseline x86_64 sur CPU à FMA (sans AVX2, sinon la
///   variante `avx2,fma` est prise en amont) : appel de `eft::fma_call` voie
///   par voie — mesuré plus rapide que Dekker vectorisé en SSE2 (éléphants
///   1e7, étages directs 5,6 s contre 9,7 s) ;
/// - CPU sans FMA ([`TP_AUTO`]) ou [`TP_DEKKER`] : Dekker sans branchement
///   sur toutes les voies ; une voie hors du domaine exact de Dekker
///   (`eft::dekker_exact` — en pratique une voie libre partie à l'infini) est
///   refaite par `mul_add`, hors ligne.
#[inline(always)]
fn two_prod<const L: usize, const M: u8>(a: V<L>, b: V<L>) -> (V<L>, V<L>) {
    let p = a * b;
    let mut e = [0.0; L];
    if M == TP_FMA || (M == TP_AUTO && eft::fma_inline()) {
        for l in 0..L {
            e[l] = a.0[l].mul_add(b.0[l], -p.0[l]);
        }
    } else if M == TP_AUTO && eft::fma_runtime() {
        for l in 0..L {
            e[l] = eft::fma_call(a.0[l], b.0[l], -p.0[l]);
        }
    } else {
        let mut ok = true;
        for l in 0..L {
            e[l] = eft::dekker_err(a.0[l], b.0[l], p.0[l]);
            ok &= eft::dekker_exact(a.0[l], b.0[l], p.0[l]);
        }
        if !ok {
            two_prod_fixup(&a.0, &b.0, &p.0, &mut e);
        }
    }
    (p, V(e))
}

/// Voies hors domaine de Dekker : erreur par `mul_add` (exact).
#[cold]
#[inline(never)]
fn two_prod_fixup<const L: usize>(a: &[f64; L], b: &[f64; L], p: &[f64; L], e: &mut [f64; L]) {
    for l in 0..L {
        if !eft::dekker_exact(a[l], b[l], p[l]) {
            e[l] = a[l].mul_add(b[l], -p[l]);
        }
    }
}

/// Arithmétique d'un étage direct en voies vectorielles.
pub trait LaneNum<const L: usize>: Copy {
    /// Nombre d'une voie (entrée des points).
    type Scalar: Copy;
    /// Unité d'arrondi relative aux opérandes (bits de garde compris).
    const UNIT: f64;
    fn splat(s: Self::Scalar) -> Self;
    fn set_lane(&mut self, l: usize, s: Self::Scalar);
    fn hi(&self) -> V<L>;
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul(self, o: Self) -> Self;
    fn sqr(self) -> Self;
    fn twice(self) -> Self;
    fn neg(self) -> Self;
    fn abs(self) -> Self;
}

/// Double-double en voies : mêmes opérations que `perturbation::dd`.
#[derive(Clone, Copy, Debug)]
pub struct DdV<const L: usize, const M: u8 = TP_AUTO> {
    hi: V<L>,
    lo: V<L>,
}

impl<const L: usize, const M: u8> LaneNum<L> for DdV<L, M> {
    type Scalar = Dd;
    const UNIT: f64 = DD_UNIT;
    #[inline(always)]
    fn splat(s: Dd) -> Self {
        DdV { hi: V::splat(s.hi), lo: V::splat(s.lo) }
    }
    #[inline(always)]
    fn set_lane(&mut self, l: usize, s: Dd) {
        self.hi.0[l] = s.hi;
        self.lo.0[l] = s.lo;
    }
    #[inline(always)]
    fn hi(&self) -> V<L> {
        self.hi
    }
    /// `dd::add` (QD `dd_add`).
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        let (s1, s2) = two_sum(self.hi, o.hi);
        let (t1, t2) = two_sum(self.lo, o.lo);
        let s2 = s2 + t1;
        let (s1, s2) = quick_two_sum(s1, s2);
        let s2 = s2 + t2;
        let (hi, lo) = quick_two_sum(s1, s2);
        DdV { hi, lo }
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        self.add(o.neg())
    }
    /// `dd::mul` (QD `dd_mul`).
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        let (p1, p2) = two_prod::<L, M>(self.hi, o.hi);
        let p2 = p2 + (self.hi * o.lo + self.lo * o.hi);
        let (hi, lo) = quick_two_sum(p1, p2);
        DdV { hi, lo }
    }
    /// `dd::sqr`.
    #[inline(always)]
    fn sqr(self) -> Self {
        let (p1, p2) = two_prod::<L, M>(self.hi, self.hi);
        let p2 = p2 + V::splat(2.0) * (self.hi * self.lo);
        let (hi, lo) = quick_two_sum(p1, p2);
        DdV { hi, lo }
    }
    /// `dd::mul_f64(2.0)` (celui de la boucle mono-pixel).
    #[inline(always)]
    fn twice(self) -> Self {
        let two = V::splat(2.0);
        let (p1, p2) = two_prod::<L, M>(self.hi, two);
        let p2 = p2 + self.lo * two;
        let (hi, lo) = quick_two_sum(p1, p2);
        DdV { hi, lo }
    }
    #[inline(always)]
    fn neg(self) -> Self {
        DdV { hi: -self.hi, lo: -self.lo }
    }
    #[inline(always)]
    fn abs(self) -> Self {
        DdV { hi: self.hi.neg_where_negative(self.hi), lo: self.lo.neg_where_negative(self.hi) }
    }
}

/// Renormalisation QD à 5 entrées SANS branchement (cf. doc du module).
#[inline(always)]
fn renorm5<const L: usize>(c0: V<L>, c1: V<L>, c2: V<L>, c3: V<L>, c4: V<L>) -> [V<L>; 4] {
    let (s0, c4) = quick_two_sum(c3, c4);
    let (s0, c3) = quick_two_sum(c2, s0);
    let (s0, c2) = quick_two_sum(c1, s0);
    let (c0, c1) = quick_two_sum(c0, s0);
    let (s1, s2) = quick_two_sum(c1, c2);
    let (s2, s3) = quick_two_sum(s2, c3);
    [c0, s1, s2, s3 + c4]
}

/// `three_sum` QD.
#[inline(always)]
fn three_sum<const L: usize>(a: &mut V<L>, b: &mut V<L>, c: &mut V<L>) {
    let (t1, t2) = two_sum(*a, *b);
    let (a2, t3) = two_sum(*c, t1);
    let (b2, c2) = two_sum(t2, t3);
    *a = a2;
    *b = b2;
    *c = c2;
}

/// `three_sum2` QD.
#[inline(always)]
fn three_sum2<const L: usize>(a: &mut V<L>, b: &mut V<L>, c: V<L>) {
    let (t1, t2) = two_sum(*a, *b);
    let (a2, t3) = two_sum(c, t1);
    *a = a2;
    *b = t2 + t3;
}

/// Quad-double en voies : algorithmes « sloppy » de QD (Hida, Li, Bailey),
/// renormalisation sans branchement.
#[derive(Clone, Copy, Debug)]
pub struct QdV<const L: usize, const M: u8 = TP_AUTO>(pub [V<L>; 4]);

impl<const L: usize, const M: u8> LaneNum<L> for QdV<L, M> {
    type Scalar = Qd;
    /// 2⁻²⁰⁶ (même unité que la boucle qd de perturbation).
    const UNIT: f64 = super::pixel_loop_qd::U_QD;
    #[inline(always)]
    fn splat(s: Qd) -> Self {
        QdV(s.0.map(V::splat))
    }
    #[inline(always)]
    fn set_lane(&mut self, l: usize, s: Qd) {
        for k in 0..4 {
            self.0[k].0[l] = s.0[k];
        }
    }
    #[inline(always)]
    fn hi(&self) -> V<L> {
        self.0[0]
    }
    /// QD `sloppy_add`.
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        let (a, b) = (self.0, o.0);
        let (s0, t0) = two_sum(a[0], b[0]);
        let (s1, t1) = two_sum(a[1], b[1]);
        let (mut s2, t2) = two_sum(a[2], b[2]);
        let (mut s3, t3) = two_sum(a[3], b[3]);
        let (s1, mut t0) = two_sum(s1, t0);
        let mut t1 = t1;
        three_sum(&mut s2, &mut t0, &mut t1);
        three_sum2(&mut s3, &mut t0, t2);
        let t0 = t0 + t1 + t3;
        QdV(renorm5(s0, s1, s2, s3, t0))
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        self.add(o.neg())
    }
    /// QD `sloppy_mul`.
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        let (a, b) = (self.0, o.0);
        let (p0, q0) = two_prod::<L, M>(a[0], b[0]);
        let (mut p1, q1) = two_prod::<L, M>(a[0], b[1]);
        let (mut p2, q2) = two_prod::<L, M>(a[1], b[0]);
        let (mut p3, q3) = two_prod::<L, M>(a[0], b[2]);
        let (mut p4, q4) = two_prod::<L, M>(a[1], b[1]);
        let (mut p5, q5) = two_prod::<L, M>(a[2], b[0]);
        let mut q0 = q0;
        three_sum(&mut p1, &mut p2, &mut q0);
        let (mut q1, mut q2) = (q1, q2);
        three_sum(&mut p2, &mut q1, &mut q2);
        three_sum(&mut p3, &mut p4, &mut p5);
        let (s0, t0) = two_sum(p2, p3);
        let (s1, t1) = two_sum(q1, p4);
        let mut s2 = q2 + p5;
        let (mut s1, t0) = two_sum(s1, t0);
        s2 = s2 + (t0 + t1);
        s1 = s1
            + (a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0] + q0 + q3 + q4 + q5);
        QdV(renorm5(p0, p1, s0, s1, s2))
    }
    /// QD `sqr`.
    #[inline(always)]
    fn sqr(self) -> Self {
        let a = self.0;
        let two = V::splat(2.0);
        let (p0, q0) = two_prod::<L, M>(a[0], a[0]);
        let (p1, q1) = two_prod::<L, M>(two * a[0], a[1]);
        let (p2, q2) = two_prod::<L, M>(two * a[0], a[2]);
        let (p3, q3) = two_prod::<L, M>(a[1], a[1]);
        let (p1, q0) = two_sum(q0, p1);
        let (q0, q1) = two_sum(q0, q1);
        let (p2, p3) = two_sum(p2, p3);
        let (s0, t0) = two_sum(q0, p2);
        let (s1, t1) = two_sum(q1, p3);
        let (s1, t0) = two_sum(s1, t0);
        let t0 = t0 + t1;
        let (s1, t0) = quick_two_sum(s1, t0);
        let (p2, t1) = quick_two_sum(s0, s1);
        let (p3, q0) = quick_two_sum(t1, t0);
        let p4 = two * a[0] * a[3];
        let p5 = two * a[1] * a[2];
        let (p4, p5) = two_sum(p4, p5);
        let (q2, q3) = two_sum(q2, q3);
        let (t0, t1) = two_sum(p4, q2);
        let t1 = t1 + p5 + q3;
        let (p3, p4) = two_sum(p3, t0);
        let p4 = p4 + q0 + t1;
        QdV(renorm5(p0, p1, p2, p3, p4))
    }
    /// `2·x`, exact.
    #[inline(always)]
    fn twice(self) -> Self {
        let two = V::splat(2.0);
        QdV(self.0.map(|c| c * two))
    }
    #[inline(always)]
    fn neg(self) -> Self {
        QdV(self.0.map(|c| -c))
    }
    #[inline(always)]
    fn abs(self) -> Self {
        let s = self.0[0];
        QdV(self.0.map(|c| c.neg_where_negative(s)))
    }
}

/// Itération directe tracée de `points`, `L` pixels de front. Sémantique de
/// [`super::direct_dd::iterate_direct_dd`] par voie (mêmes opérations, même
/// ordre, mêmes tests de sortie).
///
/// `julia` : le point du pixel est le `z` de départ et `other` la constante
/// `c` ; sinon `z` part de `other` et le point est `c`.
#[inline(always)]
pub fn iterate_lanes<N: LaneNum<L>, const L: usize>(
    phase: &Phase,
    points: &[(N::Scalar, N::Scalar)],
    other: (N::Scalar, N::Scalar),
    julia: bool,
    iteration_max: u32,
    bailout: f64,
) -> Vec<DirectDdResult> {
    const FREE: usize = usize::MAX;
    let start = |p: (N::Scalar, N::Scalar)| if julia { (p, other) } else { (other, p) };
    let res = |iter: u32, x: f64, y: f64, reliable: bool| DirectDdResult {
        iteration: iter,
        z: Complex64::new(x, y),
        reliable,
    };
    let hi1 = |s: N::Scalar| N::splat(s).hi().0[0];
    if !phase_supported(phase) {
        return points
            .iter()
            .map(|&p| {
                let (z0, _) = start(p);
                res(0, hi1(z0.0), hi1(z0.1), false)
            })
            .collect();
    }
    let b2 = bailout * bailout;
    let near_lo = (bailout - 2.0 * E_MAX).max(0.0);
    let near2 = near_lo * near_lo;
    let mut out = vec![res(0, 0.0, 0.0, false); points.len()];
    let mut next = 0usize;
    let mut idx = [FREE; L];
    let mut x = N::splat(other.0);
    let mut y = N::splat(other.1);
    let mut cx = x;
    let mut cy = y;
    let mut c_abs = V::<L>::splat(0.0);
    let mut e = V::<L>::splat(0.0);
    let mut it = [0u32; L];
    loop {
        // Sorties par la condition de boucle, puis recharge des voies libres.
        for l in 0..L {
            loop {
                if idx[l] != FREE {
                    if it[l] < iteration_max {
                        break;
                    }
                    out[idx[l]] = res(it[l], x.hi().0[l], y.hi().0[l], true);
                    idx[l] = FREE;
                }
                if next == points.len() {
                    break;
                }
                let (z0, c) = start(points[next]);
                let (zx, zy) = (hi1(z0.0), hi1(z0.1));
                if zx * zx + zy * zy >= b2 {
                    out[next] = res(0, zx, zy, true);
                    next += 1;
                    continue;
                }
                idx[l] = next;
                x.set_lane(l, z0.0);
                y.set_lane(l, z0.1);
                cx.set_lane(l, c.0);
                cy.set_lane(l, c.1);
                c_abs.0[l] = hi1(c.0).abs() + hi1(c.1).abs();
                e.0[l] = 0.0;
                it[l] = 0;
                next += 1;
            }
        }
        if idx.iter().all(|&k| k == FREE) {
            break;
        }
        // Une itération sur toutes les voies (une voie libre calcule pour
        // rien : sa sortie n'est jamais lue).
        for op in &phase.ops {
            match op {
                Op::Sqr => {
                    let (xh, yh) = (x.hi(), y.hi());
                    let n = xh * xh + yh * yh;
                    e = e * (V::splat(2.0) * n.sqrt() + e) + V::splat(N::UNIT) * n;
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
                    x = x.add(cx);
                    y = y.add(cy);
                    e = e + V::splat(N::UNIT) * (x.hi().abs() + y.hi().abs() + c_abs);
                    for k in it.iter_mut() {
                        *k += 1;
                    }
                }
                // Exclu par `phase_supported`.
                _ => unreachable!("opcode hors phase_supported"),
            }
        }
        let (xh, yh) = (x.hi(), y.hi());
        let n = xh * xh + yh * yh;
        for l in 0..L {
            if idx[l] == FREE {
                continue;
            }
            let (n, el) = (n.0[l], e.0[l]);
            let verdict = if !n.is_finite() {
                Some(false)
            } else if n >= b2 {
                Some(el < E_MAX.max(0.01 * n.sqrt()) && n.sqrt() - el >= bailout)
            } else if !(el < E_MAX) || (n >= near2 && n.sqrt() + el >= bailout) {
                Some(false)
            } else {
                None
            };
            if let Some(ok) = verdict {
                out[idx[l]] = res(it[l], xh.0[l], yh.0[l], ok);
                idx[l] = FREE;
            }
        }
    }
    out
}

/// Voies de l'étage double-double direct.
pub const DD_LANES: usize = 8;
/// Voies de l'étage quad-double direct.
pub const QD_LANES: usize = 8;

/// Étage double-double direct sur `points`, en voies : résultats identiques
/// au bit près à [`super::direct_dd::iterate_direct_dd`] pixel par pixel.
pub fn direct_dd_lanes(
    phase: &Phase,
    points: &[(Dd, Dd)],
    other: (Dd, Dd),
    julia: bool,
    iteration_max: u32,
    bailout: f64,
) -> Vec<DirectDdResult> {
    #[cfg(all(target_arch = "x86_64", not(all(target_feature = "avx2", target_feature = "fma"))))]
    if eft::avx2_fma_dispatch() {
        // SAFETY : la variante n'exige que `avx2` et `fma`, présents
        // (`avx2_fma_dispatch` = détection `cpuid`, lue une fois).
        return unsafe { direct_dd_lanes_avx2(phase, points, other, julia, iteration_max, bailout) };
    }
    iterate_lanes::<DdV<DD_LANES>, DD_LANES>(phase, points, other, julia, iteration_max, bailout)
}

/// [`direct_dd_lanes`] compilée avec `avx2,fma` (binaire baseline sur CPU
/// récent) : `two_prod` y est `vfmadd` en ligne, les voies en `ymm`. Mêmes
/// opérations IEEE (Rust ne contracte jamais `a·b + c` implicitement) :
/// sortie identique au bit près (verrou `avx2_lanes_match_baseline_lanes`).
#[cfg(all(target_arch = "x86_64", not(all(target_feature = "avx2", target_feature = "fma"))))]
#[target_feature(enable = "avx2,fma")]
fn direct_dd_lanes_avx2(
    phase: &Phase,
    points: &[(Dd, Dd)],
    other: (Dd, Dd),
    julia: bool,
    iteration_max: u32,
    bailout: f64,
) -> Vec<DirectDdResult> {
    iterate_lanes::<DdV<DD_LANES, TP_FMA>, DD_LANES>(phase, points, other, julia, iteration_max, bailout)
}

/// Étage quad-double direct sur `points`, en voies : résultats identiques au
/// bit près à [`super::direct_dd::iterate_direct_qd`] pixel par pixel.
pub fn direct_qd_lanes(
    phase: &Phase,
    points: &[(Qd, Qd)],
    other: (Qd, Qd),
    julia: bool,
    iteration_max: u32,
    bailout: f64,
) -> Vec<DirectDdResult> {
    #[cfg(all(target_arch = "x86_64", not(all(target_feature = "avx2", target_feature = "fma"))))]
    if eft::avx2_fma_dispatch() {
        // SAFETY : cf. `direct_dd_lanes`.
        return unsafe { direct_qd_lanes_avx2(phase, points, other, julia, iteration_max, bailout) };
    }
    iterate_lanes::<QdV<QD_LANES>, QD_LANES>(phase, points, other, julia, iteration_max, bailout)
}

/// [`direct_qd_lanes`] compilée avec `avx2,fma` (cf. [`direct_dd_lanes_avx2`]).
#[cfg(all(target_arch = "x86_64", not(all(target_feature = "avx2", target_feature = "fma"))))]
#[target_feature(enable = "avx2,fma")]
fn direct_qd_lanes_avx2(
    phase: &Phase,
    points: &[(Qd, Qd)],
    other: (Qd, Qd),
    julia: bool,
    iteration_max: u32,
    bailout: f64,
) -> Vec<DirectDdResult> {
    iterate_lanes::<QdV<QD_LANES, TP_FMA>, QD_LANES>(phase, points, other, julia, iteration_max, bailout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::bytecode::compile_formula;
    use crate::fractal::bytecode::direct_dd::{iterate_direct_dd, iterate_direct_qd, iterate_direct_qd_branchy};
    use crate::fractal::FractalType;

    fn dd(v: f64) -> Dd {
        Dd::from_f64(v)
    }
    fn qd(v: f64) -> Qd {
        Qd::from_f64(v)
    }

    /// Points du bord parabolique des éléphants (1e7) : `w×h` pixels de la
    /// vue, en f64 (les tests et le banc ne jugent pas la vérité).
    fn elephant_points(w: usize, h: usize) -> Vec<Complex64> {
        let (cx, cy, span) = (0.276_316_922_752_587_96, 0.0085, 4e-7);
        let sy = span * h as f64 / w as f64;
        (0..w * h)
            .map(|k| {
                let (i, j) = (k % w, k / w);
                Complex64::new(
                    cx + ((i as f64 + 0.5) / w as f64 - 0.5) * span,
                    cy + ((j as f64 + 0.5) / h as f64 - 0.5) * sy,
                )
            })
            .collect()
    }

    /// Bord des éléphants (orbites longues, dd refusés) + grille large.
    fn mixed_points() -> Vec<Complex64> {
        let mut pts = elephant_points(12, 8);
        pts.extend(
            (0..64).map(|k| Complex64::new(-2.2 + 3.3 * (k % 8) as f64 / 7.0, -1.6 + 3.2 * (k / 8) as f64 / 7.0)),
        );
        pts
    }

    const FAMILIES: [FractalType; 5] = [
        FractalType::Mandelbrot,
        FractalType::BurningShip,
        FractalType::Celtic,
        FractalType::Tricorn,
        FractalType::Julia,
    ];

    fn same(a: &DirectDdResult, b: &DirectDdResult) -> bool {
        a.iteration == b.iteration
            && a.reliable == b.reliable
            && a.z.re.to_bits() == b.z.re.to_bits()
            && a.z.im.to_bits() == b.z.im.to_bits()
    }

    fn lanes_at<N: LaneNum<L>, const L: usize>(
        phase: &Phase,
        pts: &[(N::Scalar, N::Scalar)],
        other: (N::Scalar, N::Scalar),
        julia: bool,
        n: u32,
    ) -> Vec<DirectDdResult> {
        iterate_lanes::<N, L>(phase, pts, other, julia, n, 25.0)
    }

    /// Voies dd == boucle mono-pixel `iterate_direct_dd` AU BIT PRÈS (compte,
    /// z, verdict), pour toute largeur de voie, Mandelbrot-like, Julia et
    /// familles à pliage.
    #[test]
    fn dd_lanes_match_direct_dd_bit_for_bit() {
        let pts = mixed_points();
        let mut refused = 0;
        for ft in FAMILIES {
            let f = compile_formula(ft, 2.0).unwrap();
            let phase = &f.phases[0];
            let julia = ft == FractalType::Julia;
            let n = if ft == FractalType::Mandelbrot { 20_000 } else { 2000 };
            let p: Vec<(Dd, Dd)> = pts.iter().map(|c| (dd(c.re), dd(c.im))).collect();
            let o = if julia { (dd(-0.8), dd(0.156)) } else { (dd(0.0), dd(0.0)) };
            let want: Vec<DirectDdResult> = p
                .iter()
                .map(|&q| if julia { iterate_direct_dd(phase, q, o, n, 25.0) } else { iterate_direct_dd(phase, o, q, n, 25.0) })
                .collect();
            refused += want.iter().filter(|r| !r.reliable).count();
            for (l, got) in [
                (1, lanes_at::<DdV<1>, 1>(phase, &p, o, julia, n)),
                (3, lanes_at::<DdV<3>, 3>(phase, &p, o, julia, n)),
                (4, lanes_at::<DdV<4>, 4>(phase, &p, o, julia, n)),
                (8, lanes_at::<DdV<8>, 8>(phase, &p, o, julia, n)),
                (16, lanes_at::<DdV<16>, 16>(phase, &p, o, julia, n)),
            ] {
                for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(same(g, w), "{ft:?} L={l} pixel {k} : {g:?} ≠ {w:?}");
                }
            }
        }
        assert!(refused > 10, "anti-vacuité : {refused} pixels dd refusés");
    }

    /// Les trois réalisations du produit sans erreur (choix à l'exécution,
    /// Dekker forcé, variante `avx2,fma` de production) rendent les mêmes
    /// résultats AU BIT PRÈS sur les étages dd et qd complets.
    #[test]
    fn two_prod_modes_give_identical_lanes() {
        let pts = mixed_points();
        #[cfg(all(target_arch = "x86_64", not(all(target_feature = "avx2", target_feature = "fma"))))]
        let avx = std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");
        let mut checked = 0;
        for ft in FAMILIES {
            let f = compile_formula(ft, 2.0).unwrap();
            let phase = &f.phases[0];
            let julia = ft == FractalType::Julia;
            let n = if ft == FractalType::Mandelbrot { 20_000 } else { 2000 };
            let p: Vec<(Dd, Dd)> = pts.iter().map(|c| (dd(c.re), dd(c.im))).collect();
            let o = if julia { (dd(-0.8), dd(0.156)) } else { (dd(0.0), dd(0.0)) };
            let auto = lanes_at::<DdV<8>, 8>(phase, &p, o, julia, n);
            let mut others = vec![("dekker", lanes_at::<DdV<8, TP_DEKKER>, 8>(phase, &p, o, julia, n))];
            #[cfg(all(target_arch = "x86_64", not(all(target_feature = "avx2", target_feature = "fma"))))]
            if avx {
                // SAFETY : AVX2 + FMA détectés.
                others.push(("avx2", unsafe { direct_dd_lanes_avx2(phase, &p, o, julia, n, 25.0) }));
            }
            others.push(("dispatch", direct_dd_lanes(phase, &p, o, julia, n, 25.0)));
            for (name, got) in &others {
                for (k, (g, w)) in got.iter().zip(&auto).enumerate() {
                    assert!(same(g, w), "dd {name} {ft:?} pixel {k} : {g:?} ≠ {w:?}");
                    checked += 1;
                }
            }
            let q: Vec<(Qd, Qd)> = pts.iter().map(|c| (qd(c.re), qd(c.im))).collect();
            let oq = if julia { (qd(-0.8), qd(0.156)) } else { (qd(0.0), qd(0.0)) };
            let auto = lanes_at::<QdV<8>, 8>(phase, &q, oq, julia, n);
            let mut others = vec![("dekker", lanes_at::<QdV<8, TP_DEKKER>, 8>(phase, &q, oq, julia, n))];
            #[cfg(all(target_arch = "x86_64", not(all(target_feature = "avx2", target_feature = "fma"))))]
            if avx {
                // SAFETY : AVX2 + FMA détectés.
                others.push(("avx2", unsafe { direct_qd_lanes_avx2(phase, &q, oq, julia, n, 25.0) }));
            }
            others.push(("dispatch", direct_qd_lanes(phase, &q, oq, julia, n, 25.0)));
            for (name, got) in &others {
                for (k, (g, w)) in got.iter().zip(&auto).enumerate() {
                    assert!(same(g, w), "qd {name} {ft:?} pixel {k} : {g:?} ≠ {w:?}");
                    checked += 1;
                }
            }
        }
        assert!(checked > 2000);
    }

    /// Voies qd == une voie (`iterate_direct_qd`) au bit près ; et mêmes
    /// comptes / verdicts que l'ancienne boucle qd à renormalisation à
    /// branches (`perturbation::qd`).
    #[test]
    fn qd_lanes_match_single_lane_and_branchy_counts() {
        let pts = mixed_points();
        let mut checked = 0;
        for ft in FAMILIES {
            let f = compile_formula(ft, 2.0).unwrap();
            let phase = &f.phases[0];
            let julia = ft == FractalType::Julia;
            let n = if ft == FractalType::Mandelbrot { 20_000 } else { 2000 };
            let p: Vec<(Qd, Qd)> = pts.iter().map(|c| (qd(c.re), qd(c.im))).collect();
            let o = if julia { (qd(-0.8), qd(0.156)) } else { (Qd::ZERO, Qd::ZERO) };
            let one = |q: (Qd, Qd)| {
                if julia {
                    iterate_direct_qd(phase, q, o, n, 25.0)
                } else {
                    iterate_direct_qd(phase, o, q, n, 25.0)
                }
            };
            let want: Vec<DirectDdResult> = p.iter().map(|&q| one(q)).collect();
            for (l, got) in [
                (4, lanes_at::<QdV<4>, 4>(phase, &p, o, julia, n)),
                (8, lanes_at::<QdV<8>, 8>(phase, &p, o, julia, n)),
                (16, lanes_at::<QdV<16>, 16>(phase, &p, o, julia, n)),
            ] {
                for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(same(g, w), "{ft:?} L={l} pixel {k} : {g:?} ≠ {w:?}");
                }
            }
            for (k, &q) in p.iter().enumerate() {
                let b = if julia {
                    iterate_direct_qd_branchy(phase, q, o, n, 25.0)
                } else {
                    iterate_direct_qd_branchy(phase, o, q, n, 25.0)
                };
                let w = &want[k];
                assert_eq!((w.iteration, w.reliable), (b.iteration, b.reliable), "{ft:?} pixel {k}");
                checked += 1;
            }
        }
        assert!(checked > 500);
    }

    /// Dekker == FMA en voies : produit sans erreur, mêmes bits, voies hors
    /// domaine (infinis, sous-dépassement) comprises.
    #[test]
    fn lane_two_prod_dekker_matches_fma() {
        let mut s = 0x2545F4914F6CDD1Du64;
        let mut r = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let m = (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
            m * 2f64.powi(((s % 200) as i32) - 100)
        };
        for k in 0..50_000 {
            let mut a = V::<4>([r(), r(), r(), r()]);
            let b = V::<4>([r(), r(), r(), r()]);
            if k % 7 == 0 {
                a.0[k % 4] = [f64::INFINITY, 1e-320, 0.0, 2f64.powi(1000)][k % 4];
            }
            let (p, e) = two_prod::<4, TP_DEKKER>(a, b);
            for l in 0..4 {
                let (pf, ef) = eft::two_prod_fma(a.0[l], b.0[l]);
                assert_eq!(p.0[l].to_bits(), pf.to_bits());
                assert!(e.0[l].to_bits() == ef.to_bits() || (ef.is_nan() && e.0[l].is_nan()));
            }
        }
    }

    /// Erreur des opérations qd sans branchement contre `rug` à 512 b,
    /// relative aux OPÉRANDES (modèle « sloppy ») : < 2⁻²⁰⁸ — y compris en
    /// annulation et avec des composantes NULLES (entrées f64, z = 0) : le
    /// cas où la renormalisation diffère de `perturbation::qd`.
    #[test]
    fn qdv_ops_match_rug_within_qd_unit() {
        use rug::Float;
        const P: u32 = 512;
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let mut gen = |scale: f64, ncomp: i32| {
            let mut f = Float::with_val(P, next() * scale);
            for k in 1..ncomp {
                f += next() * scale * 2f64.powi(-53 * k);
            }
            let mut t = Float::new(P);
            Qd::from_float(&f, &mut t)
        };
        let one = |q: Qd| QdV::<1>(q.0.map(|c| V([c])));
        let back = |q: QdV<1>| {
            let mut f = Float::with_val(P, q.0[0].0[0]);
            for c in &q.0[1..] {
                f += c.0[0];
            }
            f
        };
        let rel = |got: QdV<1>, want: &Float, den: &Float| {
            if *den == 0 {
                return if back(got) == *want { 0.0 } else { 1.0 };
            }
            let mut d = back(got);
            d -= want;
            (d.abs() / den).to_f64()
        };
        let mut worst = [0.0f64; 4];
        for i in 0..20000 {
            // 1..=5 composantes « pleines » (1 = un f64, 3 composantes nulles).
            let na = 1 + (i % 5);
            let nb = 1 + ((i / 5) % 5);
            let a = gen(2f64.powi((i % 40) - 20), na);
            let mut b = gen(2f64.powi(((i / 40) % 40) - 20), nb);
            if i % 3 == 0 {
                b = Qd::from_f64(-a.0[0]).add(b.mul_pow2(2f64.powi(-60)));
            }
            if i % 97 == 0 {
                b = Qd::ZERO;
            }
            let (fa, fb) = (a.to_float(P), b.to_float(P));
            let (va, vb) = (one(a), one(b));
            let den = Float::with_val(P, fa.clone().abs() + fb.clone().abs());
            worst[0] = worst[0].max(rel(va.add(vb), &Float::with_val(P, &fa + &fb), &den));
            worst[1] = worst[1].max(rel(va.sub(vb), &Float::with_val(P, &fa - &fb), &den));
            let prod = Float::with_val(P, &fa * &fb);
            worst[2] = worst[2].max(rel(va.mul(vb), &prod, &Float::with_val(P, prod.clone().abs())));
            let sq = Float::with_val(P, fa.square_ref());
            worst[3] = worst[3].max(rel(va.sqr(), &sq, &sq));
        }
        eprintln!("qdv worst bits: {:?}", worst.map(|w| -w.log2()));
        for (k, w) in worst.iter().enumerate() {
            assert!(*w < 2f64.powi(-208), "op {k}: erreur relative {w:e} ({:.1} b)", -w.log2());
        }
    }

    /// Banc mono-fil des étages directs (ns/itération), bord des éléphants
    /// 1e7 à 60 000 itérations : dd sur des pixels longs, qd sur les pixels
    /// que le dd refuse. Min de 7 tours alternés.
    /// `cargo test --release --lib bench_direct_simd -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_direct_simd() {
        use rayon::prelude::*;
        use std::time::Instant;
        let f = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let phase = &f.phases[0];
        let n = 60_000;
        let pts = elephant_points(128, 80);
        let ddp: Vec<(Dd, Dd)> = pts.iter().map(|c| (dd(c.re), dd(c.im))).collect();
        let z = (dd(0.0), dd(0.0));
        let cls: Vec<DirectDdResult> =
            ddp.par_chunks(32).flat_map_iter(|ch| direct_dd_lanes(phase, ch, z, false, n, 25.0)).collect();
        let long: Vec<(Dd, Dd)> =
            (0..pts.len()).filter(|&k| cls[k].iteration > 2000).map(|k| ddp[k]).step_by(7).take(120).collect();
        let refused: Vec<(Qd, Qd)> = (0..pts.len())
            .filter(|&k| !cls[k].reliable)
            .map(|k| (qd(pts[k].re), qd(pts[k].im)))
            .step_by(3)
            .take(64)
            .collect();
        eprintln!("dd longs {} ; dd refusés (qd) {}", long.len(), refused.len());
        let zq = (Qd::ZERO, Qd::ZERO);
        type Run<'a> = (&'a str, Box<dyn Fn() -> u64 + 'a>);
        let iters = |v: Vec<DirectDdResult>| v.iter().map(|r| r.iteration as u64).sum::<u64>();
        let runs: Vec<Run> = vec![
            ("dd mono", Box::new(|| long.iter().map(|&p| iterate_direct_dd(phase, z, p, n, 25.0).iteration as u64).sum())),
            ("dd L4", Box::new(|| iters(lanes_at::<DdV<4>, 4>(phase, &long, z, false, n)))),
            ("dd L8", Box::new(|| iters(lanes_at::<DdV<8>, 8>(phase, &long, z, false, n)))),
            ("dd L16", Box::new(|| iters(lanes_at::<DdV<16>, 16>(phase, &long, z, false, n)))),
            ("qd old", Box::new(|| refused.iter().map(|&p| iterate_direct_qd_branchy(phase, zq, p, n, 25.0).iteration as u64).sum())),
            ("qd L1", Box::new(|| iters(lanes_at::<QdV<1>, 1>(phase, &refused, zq, false, n)))),
            ("qd L4", Box::new(|| iters(lanes_at::<QdV<4>, 4>(phase, &refused, zq, false, n)))),
            ("qd L8", Box::new(|| iters(lanes_at::<QdV<8>, 8>(phase, &refused, zq, false, n)))),
            ("qd L16", Box::new(|| iters(lanes_at::<QdV<16>, 16>(phase, &refused, zq, false, n)))),
        ];
        // Estimation d'un étage triple-double : la même boucle qd avec
        // l'unité d'un td (~2⁻¹⁵³ marge comprise) — part des pixels refusés
        // par le dd que le td garantirait.
        #[derive(Clone, Copy)]
        struct TdUnit<const L: usize>(QdV<L>);
        impl<const L: usize> LaneNum<L> for TdUnit<L> {
            type Scalar = Qd;
            const UNIT: f64 = 1.0 / (1u128 << 102) as f64 / (1u64 << 51) as f64;
            fn splat(s: Qd) -> Self {
                TdUnit(QdV::splat(s))
            }
            fn set_lane(&mut self, l: usize, s: Qd) {
                self.0.set_lane(l, s)
            }
            fn hi(&self) -> V<L> {
                self.0.hi()
            }
            fn add(self, o: Self) -> Self {
                TdUnit(self.0.add(o.0))
            }
            fn sub(self, o: Self) -> Self {
                TdUnit(self.0.sub(o.0))
            }
            fn mul(self, o: Self) -> Self {
                TdUnit(self.0.mul(o.0))
            }
            fn sqr(self) -> Self {
                TdUnit(self.0.sqr())
            }
            fn twice(self) -> Self {
                TdUnit(self.0.twice())
            }
            fn neg(self) -> Self {
                TdUnit(self.0.neg())
            }
            fn abs(self) -> Self {
                TdUnit(self.0.abs())
            }
        }
        let all_refused: Vec<(Qd, Qd)> =
            (0..pts.len()).filter(|&k| !cls[k].reliable).map(|k| (qd(pts[k].re), qd(pts[k].im))).collect();
        let q: Vec<DirectDdResult> = all_refused
            .par_chunks(16)
            .flat_map_iter(|ch| lanes_at::<QdV<8>, 8>(phase, ch, zq, false, n))
            .collect();
        let t: Vec<DirectDdResult> = all_refused
            .par_chunks(16)
            .flat_map_iter(|ch| lanes_at::<TdUnit<8>, 8>(phase, ch, zq, false, n))
            .collect();
        let qok = q.iter().filter(|r| r.reliable).count();
        let tok: Vec<usize> = (0..q.len()).filter(|&k| t[k].reliable).collect();
        let qit: u64 = q.iter().map(|r| r.iteration as u64).sum();
        let tit: u64 = tok.iter().map(|&k| q[k].iteration as u64).sum();
        eprintln!(
            "td (estim.) : refusés dd {} ; qd fiables {qok} ; td fiables {} ({} it. sur {qit} de l'étage qd)",
            all_refused.len(),
            tok.len(),
            tit
        );
        let mut best = vec![f64::INFINITY; runs.len()];
        for _ in 0..7 {
            for (k, (_, r)) in runs.iter().enumerate() {
                let t = Instant::now();
                let it = r();
                best[k] = best[k].min(t.elapsed().as_nanos() as f64 / it as f64);
            }
        }
        for (k, (name, _)) in runs.iter().enumerate() {
            eprintln!("{name:8} {:.2} ns/it", best[k]);
        }
    }
}
