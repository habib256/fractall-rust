//! Arithmétique double-double (~106 bits de mantisse) en Rust pur.
//!
//! Motivation (P1.6.e, parité Fraktaler-3) : le delta de perturbation est
//! stocké en `ComplexExp` dont la mantisse est un `f64` (~53 bits). Sur les
//! orbites quasi-périodiques intérieures (cf. glitch_test_1/5), l'erreur du
//! delta s'accumule itération après itération et le pixel escape à tort. F3
//! sélectionne alors un type plus précis (`float128`, 113 bits) via son
//! wisdom file. `DoubleDouble` est l'équivalent pur-Rust : deux `f64`
//! (`hi` + `lo`) non chevauchants donnant ~106 bits de mantisse, sans FFI.
//!
//! Algorithmes : transformations sans erreur (Dekker `two_prod`, Knuth
//! `two_sum`) telles qu'utilisées par la bibliothèque QD (Bailey et al.).
//! `f64::mul_add` (FMA matériel) rend `two_prod` exact en deux opérations.
//!
//! Ce module est volontairement autonome et exhaustivement testé : il sera
//! ensuite étendu en `DoubleDoubleExp` (mantisse double-double + exposant
//! `i32`) puis branché dans `pixel_loop_exp` comme alternative haute
//! précision à `ComplexExp`. Aucune intégration au hot-loop ici.
//!
//! `allow(dead_code)` au niveau module : le kernel est livré et testé avant
//! son câblage (P1.6.e étape 1/N).
#![allow(dead_code)]

/// Somme sans erreur de deux f64 (Knuth `two_sum`) : renvoie `(s, e)` tels que
/// `s = fl(a + b)` et `a + b = s + e` exactement, sans hypothèse sur les
/// magnitudes relatives.
#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}

/// Somme sans erreur rapide (Dekker `fast_two_sum`) : exige `|a| >= |b|`.
/// `s = fl(a + b)`, `a + b = s + e`.
#[inline(always)]
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let err = b - (s - a);
    (s, err)
}

/// Produit sans erreur (Dekker `two_prod` via FMA) : `p = fl(a*b)`,
/// `a*b = p + e` exactement.
#[inline(always)]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let err = a.mul_add(b, -p);
    (p, err)
}

/// Nombre double-double : valeur = `hi + lo`, avec `|lo| <= 0.5 ulp(hi)`.
/// Donne ~106 bits de mantisse effective.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoubleDouble {
    pub hi: f64,
    pub lo: f64,
}

impl DoubleDouble {
    pub const ZERO: Self = Self { hi: 0.0, lo: 0.0 };

    #[inline(always)]
    pub fn new(hi: f64, lo: f64) -> Self {
        Self { hi, lo }
    }

    #[inline(always)]
    pub fn from_f64(value: f64) -> Self {
        Self { hi: value, lo: 0.0 }
    }

    /// Conversion vers f64 (composante haute, l'erreur sur lo est sous l'ulp).
    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.hi.is_finite() && self.lo.is_finite()
    }

    #[inline(always)]
    pub fn neg(self) -> Self {
        Self { hi: -self.hi, lo: -self.lo }
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        if self.hi < 0.0 {
            self.neg()
        } else {
            self
        }
    }

    /// Addition double-double + double-double (QD `dd_add`, précision ~106 bits).
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        let (s1, s2) = two_sum(self.hi, other.hi);
        let (t1, t2) = two_sum(self.lo, other.lo);
        let s2 = s2 + t1;
        let (s1, s2) = quick_two_sum(s1, s2);
        let s2 = s2 + t2;
        let (hi, lo) = quick_two_sum(s1, s2);
        Self { hi, lo }
    }

    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        self.add(other.neg())
    }

    /// Multiplication double-double × double-double (QD `dd_mul`).
    #[inline(always)]
    pub fn mul(self, other: Self) -> Self {
        let (p1, p2) = two_prod(self.hi, other.hi);
        // p2 += hi*lo' + lo*hi' (les termes lo*lo' sont sous la précision).
        let p2 = p2 + (self.hi * other.lo + self.lo * other.hi);
        let (hi, lo) = quick_two_sum(p1, p2);
        Self { hi, lo }
    }

    /// Multiplication par un f64 simple (chemin rapide, exact via FMA).
    #[inline(always)]
    pub fn mul_f64(self, b: f64) -> Self {
        let (p1, p2) = two_prod(self.hi, b);
        let p2 = p2 + self.lo * b;
        let (hi, lo) = quick_two_sum(p1, p2);
        Self { hi, lo }
    }

    /// Carré (raccourci de `mul` avec self).
    #[inline(always)]
    pub fn sqr(self) -> Self {
        let (p1, p2) = two_prod(self.hi, self.hi);
        let p2 = p2 + 2.0 * (self.hi * self.lo);
        let (hi, lo) = quick_two_sum(p1, p2);
        Self { hi, lo }
    }

    /// Division double-double / double-double (QD `dd_div`, algo de Bailey :
    /// 3 quotients successifs corrigés par les résidus dd). Précision ~106 bits.
    #[inline(always)]
    pub fn div(self, other: Self) -> Self {
        let q1 = self.hi / other.hi;
        let r = self.sub(other.mul_f64(q1)); // r = a - q1·b
        let q2 = r.hi / other.hi;
        let r = r.sub(other.mul_f64(q2)); // r = r - q2·b
        let q3 = r.hi / other.hi;
        let (hi, lo) = quick_two_sum(q1, q2);
        Self { hi, lo }.add(Self::from_f64(q3))
    }
}

/// Nombre double-double + exposant `i32` : valeur = `mantissa · 2^exponent`,
/// avec `mantissa.hi` normalisé dans `[0.5, 1)` (comme `FloatExp`, mais avec
/// ~106 bits de mantisse au lieu de ~53). Couvre à la fois la précision
/// (double-double) ET la dynamique deep-zoom (exposant débordant `f64`),
/// équivalent pur-Rust du `float128`-avec-exposant de F3.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoubleDoubleExp {
    pub mantissa: DoubleDouble,
    pub exponent: i32,
}

impl DoubleDoubleExp {
    pub const ZERO: Self = Self {
        mantissa: DoubleDouble::ZERO,
        exponent: 0,
    };

    /// Normalise pour que `mantissa.hi ∈ [0.5, 1)`, exposant ajusté.
    #[inline(always)]
    pub fn normalized(mantissa: DoubleDouble, exponent: i32) -> Self {
        if mantissa.hi == 0.0 {
            return Self::ZERO;
        }
        let (_, e) = super::types::frexp(mantissa.hi);
        let scale = super::types::pow2i(-e);
        Self {
            mantissa: DoubleDouble {
                hi: mantissa.hi * scale,
                lo: mantissa.lo * scale,
            },
            exponent: exponent + e,
        }
    }

    #[inline(always)]
    pub fn from_f64(value: f64) -> Self {
        if value == 0.0 {
            return Self::ZERO;
        }
        Self::normalized(DoubleDouble::from_f64(value), 0)
    }

    #[inline(always)]
    pub fn from_dd(value: DoubleDouble) -> Self {
        Self::normalized(value, 0)
    }

    /// Depuis un `FloatExp` (mantisse f64 53 bits + exposant `i32`). Préserve
    /// l'exposant (pas d'underflow deep-zoom, contrairement à `to_f64`) ; la
    /// partie `lo` reste nulle (on n'a que 53 bits en entrée). Sert à convertir
    /// `dc`/`delta` du path `ComplexExp` vers le tier dd.
    #[inline(always)]
    pub fn from_floatexp(fe: super::types::FloatExp) -> Self {
        Self::normalized(DoubleDouble::from_f64(fe.mantissa), fe.exponent)
    }

    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        if self.mantissa.hi == 0.0 {
            return 0.0;
        }
        if self.exponent > 1023 {
            return f64::INFINITY.copysign(self.mantissa.hi);
        }
        if self.exponent < -1074 {
            return 0.0;
        }
        self.mantissa.to_f64() * super::types::pow2i(self.exponent)
    }

    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.mantissa.is_finite()
    }

    #[inline(always)]
    pub fn neg(self) -> Self {
        Self {
            mantissa: self.mantissa.neg(),
            exponent: self.exponent,
        }
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        Self {
            mantissa: self.mantissa.abs(),
            exponent: self.exponent,
        }
    }

    /// Addition : aligne les exposants puis additionne les mantisses dd.
    #[inline(always)]
    pub fn add(self, rhs: Self) -> Self {
        if self.mantissa.hi == 0.0 {
            return rhs;
        }
        if rhs.mantissa.hi == 0.0 {
            return self;
        }
        let diff = self.exponent - rhs.exponent;
        // Au-delà de ~106 bits d'écart, le plus petit terme disparaît.
        if diff >= 108 {
            return self;
        }
        if diff <= -108 {
            return rhs;
        }
        if diff >= 0 {
            let scale = super::types::pow2i(-diff);
            let scaled = DoubleDouble {
                hi: rhs.mantissa.hi * scale,
                lo: rhs.mantissa.lo * scale,
            };
            Self::normalized(self.mantissa.add(scaled), self.exponent)
        } else {
            let scale = super::types::pow2i(diff);
            let scaled = DoubleDouble {
                hi: self.mantissa.hi * scale,
                lo: self.mantissa.lo * scale,
            };
            Self::normalized(scaled.add(rhs.mantissa), rhs.exponent)
        }
    }

    #[inline(always)]
    pub fn sub(self, rhs: Self) -> Self {
        self.add(rhs.neg())
    }

    /// Multiplication : produit des mantisses dd, somme des exposants.
    #[inline(always)]
    pub fn mul(self, rhs: Self) -> Self {
        Self::normalized(self.mantissa.mul(rhs.mantissa), self.exponent + rhs.exponent)
    }

    #[inline(always)]
    pub fn mul_f64(self, rhs: f64) -> Self {
        if rhs == 0.0 {
            return Self::ZERO;
        }
        Self::normalized(self.mantissa.mul_f64(rhs), self.exponent)
    }

    #[inline(always)]
    pub fn sqr(self) -> Self {
        Self::normalized(self.mantissa.sqr(), self.exponent * 2)
    }

    /// Division : quotient des mantisses dd, différence des exposants.
    #[inline(always)]
    pub fn div(self, rhs: Self) -> Self {
        if self.mantissa.hi == 0.0 {
            return Self::ZERO;
        }
        Self::normalized(self.mantissa.div(rhs.mantissa), self.exponent - rhs.exponent)
    }

    /// Re-normalise (par sécurité après de longues séquences).
    #[inline(always)]
    pub fn reduce(&mut self) {
        *self = Self::normalized(self.mantissa, self.exponent);
    }
}

impl PartialOrd for DoubleDoubleExp {
    /// Ordre sign-aware sur valeurs normalisées (mantissa.hi ∈ [0.5,1) ou
    /// (-1,-0.5], zéro = hi 0). Évite `to_f64` qui sature au-delà de 2^1023.
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering::*;
        let s = self.mantissa.hi.signum();
        let o = other.mantissa.hi.signum();
        if self.mantissa.hi == 0.0 && other.mantissa.hi == 0.0 {
            return Some(Equal);
        }
        if self.mantissa.hi == 0.0 {
            return Some(if other.mantissa.hi > 0.0 { Less } else { Greater });
        }
        if other.mantissa.hi == 0.0 {
            return Some(if self.mantissa.hi > 0.0 { Greater } else { Less });
        }
        if s != o {
            return self.mantissa.hi.partial_cmp(&other.mantissa.hi);
        }
        let mag_cmp = match self.exponent.cmp(&other.exponent) {
            Less => Less,
            Greater => Greater,
            Equal => {
                // Même exposant : comparer les mantisses dd (hi puis lo).
                let a = self.mantissa.abs();
                let b = other.mantissa.abs();
                match a.hi.partial_cmp(&b.hi)? {
                    Equal => a.lo.partial_cmp(&b.lo)?,
                    c => c,
                }
            }
        };
        Some(if s > 0.0 { mag_cmp } else { mag_cmp.reverse() })
    }
}

/// Nombre complexe à composantes `DoubleDoubleExp` : équivalent haute
/// précision de `ComplexExp` (qui a des composantes `FloatExp`/53 bits).
/// Mêmes opérations que `ComplexExp` pour servir de remplacement direct du
/// delta de perturbation dans `pixel_loop_exp` (P1.6.e étape 4, à venir).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComplexDDExp {
    pub re: DoubleDoubleExp,
    pub im: DoubleDoubleExp,
}

impl ComplexDDExp {
    pub const ZERO: Self = Self {
        re: DoubleDoubleExp::ZERO,
        im: DoubleDoubleExp::ZERO,
    };

    #[inline(always)]
    pub fn from_complex64(value: num_complex::Complex64) -> Self {
        Self {
            re: DoubleDoubleExp::from_f64(value.re),
            im: DoubleDoubleExp::from_f64(value.im),
        }
    }

    /// Depuis un `ComplexExp` (composantes `FloatExp` 53 bits). Préserve les
    /// exposants (deep-zoom safe) ; les mantisses restent à 53 bits (on ne peut
    /// pas inventer les bits manquants). Utilisé pour porter `dc`/`delta` du
    /// path `ComplexExp` vers le tier dd au dispatch.
    #[inline(always)]
    pub fn from_complex_exp(ce: super::types::ComplexExp) -> Self {
        Self {
            re: DoubleDoubleExp::from_floatexp(ce.re),
            im: DoubleDoubleExp::from_floatexp(ce.im),
        }
    }

    #[inline(always)]
    pub fn to_complex64_approx(self) -> num_complex::Complex64 {
        num_complex::Complex64::new(self.re.to_f64(), self.im.to_f64())
    }

    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    #[inline(always)]
    pub fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re.add(rhs.re),
            im: self.im.add(rhs.im),
        }
    }

    /// Produit complexe : (a+bi)(c+di) = (ac-bd) + (ad+bc)i.
    #[inline(always)]
    pub fn mul(self, rhs: Self) -> Self {
        let re = self.re.mul(rhs.re).sub(self.im.mul(rhs.im));
        let im = self.re.mul(rhs.im).add(self.im.mul(rhs.re));
        Self { re, im }
    }

    /// Produit par un `Complex64` (cas chaud : 2·Z·δ où Z est la référence f64).
    #[inline(always)]
    pub fn mul_complex64(self, rhs: num_complex::Complex64) -> Self {
        let re = self.re.mul_f64(rhs.re).sub(self.im.mul_f64(rhs.im));
        let im = self.re.mul_f64(rhs.im).add(self.im.mul_f64(rhs.re));
        Self { re, im }
    }

    /// Carré complexe : (a+bi)² = (a²-b²) + 2abi.
    #[inline(always)]
    pub fn sqr(self) -> Self {
        let re = self.re.sqr().sub(self.im.sqr());
        let im = self.re.mul(self.im).mul_f64(2.0);
        Self { re, im }
    }

    /// Norme² en `DoubleDoubleExp` (pas d'underflow deep zoom, comme
    /// `ComplexExp::norm_sqr_fexp`).
    #[inline(always)]
    pub fn norm_sqr(self) -> DoubleDoubleExp {
        self.re.sqr().add(self.im.sqr())
    }

    #[inline(always)]
    pub fn reduce(&mut self) {
        self.re.reduce();
        self.im.reduce();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper : valeur f64 de référence d'un DD.
    fn val(dd: DoubleDouble) -> f64 {
        dd.to_f64()
    }

    #[test]
    fn from_to_f64_roundtrip() {
        for &x in &[0.0, 1.0, -1.0, 3.14159, 1e300, 1e-300, -2.5e17] {
            assert_eq!(DoubleDouble::from_f64(x).to_f64(), x);
        }
    }

    #[test]
    fn two_sum_is_error_free() {
        // a + b == s + e exactement, vérifié en ré-additionnant.
        let a = 1.0;
        let b = 1e-20;
        let (s, e) = two_sum(a, b);
        // s perd b (sous l'ulp de 1.0) ; e capture exactement b.
        assert_eq!(s, 1.0);
        assert_eq!(e, 1e-20);
    }

    #[test]
    fn two_prod_is_error_free() {
        // Produit avec une erreur d'arrondi non nulle.
        let a = 1.0 + 2f64.powi(-30);
        let b = 1.0 + 2f64.powi(-30);
        let (p, e) = two_prod(a, b);
        // p + e doit reconstituer a*b exactement (a*b a besoin de >53 bits).
        // a*b = 1 + 2^-29 + 2^-60. p ≈ 1 + 2^-29, e ≈ 2^-60.
        let reconstructed = p + e;
        assert!((reconstructed - (1.0 + 2f64.powi(-29) + 2f64.powi(-60))).abs() < 1e-30);
        assert!(e != 0.0, "two_prod doit capturer l'erreur d'arrondi");
    }

    #[test]
    fn dd_add_beats_f64_precision() {
        // (1 + 1e-20) + 1e-20 : en f64 pur, 1+1e-20 == 1, donc le résultat
        // serait 1.0. En double-double on garde les 1e-20.
        let one = DoubleDouble::from_f64(1.0);
        let tiny = DoubleDouble::from_f64(1e-20);
        let sum = one.add(tiny).add(tiny);
        // Valeur attendue : 1 + 2e-20. La partie lo doit porter le 2e-20.
        assert!((sum.hi - 1.0).abs() < 1e-30);
        assert!((sum.lo - 2e-20).abs() < 1e-35, "lo={}", sum.lo);
        // f64 pur perdrait tout :
        let f64_sum = 1.0_f64 + 1e-20 + 1e-20;
        assert_eq!(f64_sum, 1.0);
    }

    #[test]
    fn dd_mul_carries_extra_bits() {
        // (1 + 2^-30)² = 1 + 2^-29 + 2^-60. Le terme 2^-60 dépasse f64 (53 bits)
        // mais tient dans double-double (~106 bits).
        let x = DoubleDouble::new(1.0 + 2f64.powi(-30), 0.0);
        let sq = x.sqr();
        let expected_hi = 1.0 + 2f64.powi(-29);
        assert!((sq.hi - expected_hi).abs() < 1e-15);
        // Le bit 2^-60 doit apparaître dans lo.
        assert!((sq.lo - 2f64.powi(-60)).abs() < 1e-20, "lo={:e}", sq.lo);
    }

    #[test]
    fn dd_mul_matches_mul_f64_for_simple() {
        let x = DoubleDouble::from_f64(3.0);
        let y = DoubleDouble::from_f64(7.0);
        assert_eq!(val(x.mul(y)), 21.0);
        assert_eq!(val(x.mul_f64(7.0)), 21.0);
    }

    #[test]
    fn dd_div_reciprocal_precision() {
        // 1/3 en dd doit être plus précis que 1/3 en f64. On vérifie que
        // (1/3)·3 == 1 à ~106 bits (le résidu dépasse la précision f64).
        let one = DoubleDouble::from_f64(1.0);
        let three = DoubleDouble::from_f64(3.0);
        let third = one.div(three);
        let back = third.mul(three);
        // back doit reconstituer 1.0 à mieux que 1e-30 (bien sous 1e-16 f64).
        assert!((back.hi - 1.0).abs() < 1e-30, "back.hi={}", back.hi);
        assert!(back.lo.abs() < 1e-30, "back.lo={:e}", back.lo);
        // Le f64 pur : (1.0/3.0)*3.0 laisse une erreur ~1e-16 (souvent != 1).
    }

    #[test]
    fn ddexp_div_matches_mul_inverse() {
        // (a/b)·b == a en DoubleDoubleExp à travers les exposants.
        let a = DoubleDoubleExp::normalized(DoubleDouble::new(0.7, 1e-20), 40);
        let b = DoubleDoubleExp::normalized(DoubleDouble::from_f64(0.6), -25);
        let q = a.div(b);
        let back = q.mul(b);
        assert_eq!(back.exponent, a.exponent);
        assert!((back.mantissa.hi - a.mantissa.hi).abs() < 1e-14);
    }

    #[test]
    fn dd_sub_self_is_zero() {
        let x = DoubleDouble::new(1.0 + 2f64.powi(-40), 2f64.powi(-90));
        let z = x.sub(x);
        assert_eq!(z.hi, 0.0);
        assert_eq!(z.lo, 0.0);
    }

    #[test]
    fn dd_accumulation_stays_accurate() {
        // Accumuler 1e6 fois 1e-10 dans 1.0. f64 pur dérive ; DD reste exact.
        let mut acc = DoubleDouble::from_f64(1.0);
        let inc = DoubleDouble::from_f64(1e-10);
        for _ in 0..1_000_000 {
            acc = acc.add(inc);
        }
        // Attendu : 1.0 + 1e6 * 1e-10 = 1.0001 exactement.
        let expected = 1.0 + 1e6 * 1e-10;
        assert!((acc.to_f64() - expected).abs() < 1e-13, "got {}", acc.to_f64());
    }

    #[test]
    fn abs_and_neg() {
        let x = DoubleDouble::new(-2.0, -1e-20);
        let a = x.abs();
        assert_eq!(a.hi, 2.0);
        assert_eq!(a.lo, 1e-20);
        let n = x.neg();
        assert_eq!(n.hi, 2.0);
        assert_eq!(n.lo, 1e-20);
    }

    // --- DoubleDoubleExp ---

    #[test]
    fn ddexp_roundtrip_in_range() {
        for &x in &[1.5, -3.25, 1e100, 1e-100, 42.0, -0.001] {
            let v = DoubleDoubleExp::from_f64(x);
            assert!((v.to_f64() - x).abs() <= x.abs() * 1e-15 + 1e-300, "x={}", x);
        }
    }

    #[test]
    fn ddexp_survives_f64_overflow_range() {
        // 2^2000 déborde f64 (max ~2^1024) mais tient en DoubleDoubleExp.
        // Normalisation : 1.0·2^2000 = 0.5·2^2001 (mantissa.hi ∈ [0.5,1)).
        let big = DoubleDoubleExp::normalized(DoubleDouble::from_f64(1.0), 2000);
        assert!(big.is_finite());
        assert_eq!(big.exponent, 2001);
        assert_eq!(big.mantissa.hi, 0.5);
        // 2^-2000 sous-déborde f64 mais reste représentable.
        let tiny = DoubleDoubleExp::normalized(DoubleDouble::from_f64(1.0), -2000);
        assert!(tiny.is_finite());
        assert_eq!(tiny.exponent, -1999);
        // Produit = 1.0 (0.5·2^2001 × 0.5·2^-1999 = 0.25·2^2 = 1.0).
        let prod = big.mul(tiny);
        assert!((prod.to_f64() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ddexp_add_keeps_106bit_precision() {
        // (1 + 2^-80) où 2^-80 est SOUS la précision f64 (53 bits) mais DANS
        // la précision double-double (~106 bits). FloatExp perdrait le terme.
        let one = DoubleDoubleExp::from_f64(1.0);
        let tiny = DoubleDoubleExp::normalized(DoubleDouble::from_f64(1.0), -80);
        let sum = one.add(tiny);
        // Le terme 2^-80 doit survivre dans mantissa.lo.
        let recovered = sum.sub(one);
        let expected = 2f64.powi(-80);
        assert!(
            (recovered.to_f64() - expected).abs() < expected * 1e-10,
            "recovered={:e} expected={:e}",
            recovered.to_f64(),
            expected
        );
    }

    #[test]
    fn ddexp_mul_adds_exponents() {
        let a = DoubleDoubleExp::normalized(DoubleDouble::from_f64(1.5), 100);
        let b = DoubleDoubleExp::normalized(DoubleDouble::from_f64(2.0), 200);
        let p = a.mul(b);
        // 1.5·2^100 × 2·2^200 = 3·2^300 = 1.5·2^301.
        assert!((p.mantissa.hi - 0.75).abs() < 1e-15); // normalisé dans [0.5,1)
        assert_eq!(p.exponent, 302); // 3·2^300 = 0.75·2^302
    }

    #[test]
    fn ddexp_sqr_matches_mul_self() {
        let x = DoubleDoubleExp::normalized(DoubleDouble::new(0.6, 1e-20), 37);
        let s = x.sqr();
        let m = x.mul(x);
        assert_eq!(s.exponent, m.exponent);
        assert!((s.mantissa.hi - m.mantissa.hi).abs() < 1e-15);
    }

    #[test]
    fn ddexp_ordering_across_exponents() {
        let small = DoubleDoubleExp::normalized(DoubleDouble::from_f64(1.0), -2000);
        let big = DoubleDoubleExp::normalized(DoubleDouble::from_f64(1.0), 2000);
        assert!(small < big);
        assert!(big > small);
        let neg_big = big.neg();
        assert!(neg_big < small);
        // Égalité réflexive.
        assert_eq!(small.partial_cmp(&small), Some(std::cmp::Ordering::Equal));
    }

    #[test]
    fn ddexp_accumulation_beats_floatexp() {
        // Accumuler 2^-70 un million de fois dans 1.0. Avec une mantisse f64
        // (FloatExp) chaque ajout serait perdu (1 + 2^-70 == 1). En dd la
        // somme s'accumule : 1 + 1e6·2^-70.
        let mut acc = DoubleDoubleExp::from_f64(1.0);
        let inc = DoubleDoubleExp::normalized(DoubleDouble::from_f64(1.0), -70);
        for _ in 0..1_000_000 {
            acc = acc.add(inc);
        }
        let delta = acc.sub(DoubleDoubleExp::from_f64(1.0));
        let expected = 1.0e6 * 2f64.powi(-70);
        assert!(
            (delta.to_f64() - expected).abs() < expected * 1e-6,
            "delta={:e} expected={:e}",
            delta.to_f64(),
            expected
        );
    }

    // --- ComplexDDExp ---

    use num_complex::Complex64;

    #[test]
    fn cddexp_mul_matches_complex64() {
        // (1+2i)·(3+4i) = (3-8) + (4+6)i = -5 + 10i.
        let a = ComplexDDExp::from_complex64(Complex64::new(1.0, 2.0));
        let b = ComplexDDExp::from_complex64(Complex64::new(3.0, 4.0));
        let p = a.mul(b).to_complex64_approx();
        assert!((p.re - (-5.0)).abs() < 1e-13 && (p.im - 10.0).abs() < 1e-13, "{:?}", p);
        // mul_complex64 doit donner le même résultat.
        let p2 = a.mul_complex64(Complex64::new(3.0, 4.0)).to_complex64_approx();
        assert!((p2.re - (-5.0)).abs() < 1e-13 && (p2.im - 10.0).abs() < 1e-13);
    }

    #[test]
    fn cddexp_sqr_matches_mul_self() {
        let z = ComplexDDExp::from_complex64(Complex64::new(0.6, -0.4));
        let s = z.sqr().to_complex64_approx();
        let m = z.mul(z).to_complex64_approx();
        assert!((s.re - m.re).abs() < 1e-14 && (s.im - m.im).abs() < 1e-14);
        // (0.6-0.4i)² = 0.36-0.16 + 2·0.6·(-0.4)i = 0.20 - 0.48i.
        assert!((s.re - 0.20).abs() < 1e-13 && (s.im - (-0.48)).abs() < 1e-13, "{:?}", s);
    }

    #[test]
    fn cddexp_perturbation_step_keeps_precision() {
        // Mimique un pas Mandelbrot δ' = 2·Z·δ + δ² + dc avec dc minuscule
        // (deep zoom). Z = O(1), δ ~ 2^-90 : f64 perdrait δ² et l'addition,
        // dd la conserve. On vérifie juste que δ ne s'annule pas.
        let z = Complex64::new(0.3, 0.4);
        let mut delta = ComplexDDExp {
            re: DoubleDoubleExp::normalized(DoubleDouble::from_f64(1.0), -90),
            im: DoubleDoubleExp::normalized(DoubleDouble::from_f64(1.0), -90),
        };
        let dc = ComplexDDExp {
            re: DoubleDoubleExp::normalized(DoubleDouble::from_f64(1.0), -90),
            im: DoubleDoubleExp::ZERO,
        };
        for _ in 0..50 {
            let two_z_delta = delta.mul_complex64(z * 2.0);
            let delta_sq = delta.sqr();
            delta = two_z_delta.add(delta_sq).add(dc);
            delta.reduce();
        }
        // δ doit rester fini et non nul (l'information à 2^-90 a survécu).
        assert!(delta.is_finite());
        assert!(delta.re.mantissa.hi != 0.0 || delta.im.mantissa.hi != 0.0);
    }

    #[test]
    fn cddexp_norm_sqr_deep_zoom_no_underflow() {
        // δ ~ 2^-2000 : norm_sqr en f64 collapse à 0 ; en DoubleDoubleExp non.
        let delta = ComplexDDExp {
            re: DoubleDoubleExp::normalized(DoubleDouble::from_f64(1.0), -2000),
            im: DoubleDoubleExp::ZERO,
        };
        let n = delta.norm_sqr();
        assert!(n.mantissa.hi != 0.0, "norm_sqr ne doit pas underflow");
        // (2^-2000)² = 2^-4000.
        assert_eq!(n.exponent, -3999); // 1.0·2^-4000 normalisé = 0.5·2^-3999
    }
}
