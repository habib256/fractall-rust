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
}
