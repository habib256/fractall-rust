//! Arithmétique **quad-double** (~212 bits de mantisse) en Rust pur, pour le
//! tier qd de la correction de fiabilité (TODO G14.1 h) : les pixels dont la
//! borne d'erreur du double-double s'emballe passaient DIRECTEMENT au GMP par
//! pixel (~1-3 µs/itération à 600-860 b, allocations comprises).
//!
//! Algorithmes de la bibliothèque QD (Hida, Li, Bailey — `qd_inline.h`) dans
//! leurs variantes « sloppy » (addition et multiplication sans tri des
//! composantes), celles que QD compile par défaut. Valeur = `x[0] + x[1] +
//! x[2] + x[3]`, composantes non chevauchantes après `renorm`.
//!
//! Erreur : la variante sloppy n'est PAS un arrondi correct relatif au
//! résultat ; elle est bornée relativement aux OPÉRANDES (≈ quelques 2⁻²¹¹ ·
//! (|a|+|b|) pour l'addition, · |a|·|b| pour le produit) — le modèle d'erreur
//! que suit le traqueur de la boucle qd (unité `U_QD`, mesurée par les tests
//! contre `rug::Float` à 512 b, plus une marge).
//!
//! Pas d'exposant étendu : plage f64. La boucle qd travaille dans un cadre
//! mis à l'échelle 2^s (cf. `bytecode::pixel_loop_qd`) qui garde δ et dc loin
//! du sous-normal.

/// Somme sans erreur (Knuth).
#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// Somme sans erreur rapide, `|a| ≥ |b|` (ou a = 0).
#[inline(always)]
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

/// Produit sans erreur (FMA).
#[inline(always)]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

/// `three_sum` QD : (a, b, c) ← somme renormalisée à trois termes.
#[inline(always)]
fn three_sum(a: &mut f64, b: &mut f64, c: &mut f64) {
    let (t1, t2) = two_sum(*a, *b);
    let (a2, t3) = two_sum(*c, t1);
    let (b2, c2) = two_sum(t2, t3);
    *a = a2;
    *b = b2;
    *c = c2;
}

/// `three_sum2` QD : (a, b) ← somme à deux termes de (a, b, c).
#[inline(always)]
fn three_sum2(a: &mut f64, b: &mut f64, c: f64) {
    let (t1, t2) = two_sum(*a, *b);
    let (a2, t3) = two_sum(c, t1);
    *a = a2;
    *b = t2 + t3;
}

/// Renormalisation QD à 5 entrées → 4 composantes non chevauchantes.
#[inline(always)]
fn renorm5(c0: f64, c1: f64, c2: f64, c3: f64, c4: f64) -> Qd {
    if !c0.is_finite() {
        return Qd([c0, 0.0, 0.0, 0.0]);
    }
    let (s0, c4) = quick_two_sum(c3, c4);
    let (s0, c3) = quick_two_sum(c2, s0);
    let (s0, c2) = quick_two_sum(c1, s0);
    let (c0, c1) = quick_two_sum(c0, s0);
    let (mut s0, mut s1, mut s2, mut s3) = (c0, c1, 0.0, 0.0);
    if s1 != 0.0 {
        (s1, s2) = quick_two_sum(s1, c2);
        if s2 != 0.0 {
            (s2, s3) = quick_two_sum(s2, c3);
            if s3 != 0.0 {
                s3 += c4;
            } else {
                s2 += c4;
            }
        } else {
            (s1, s2) = quick_two_sum(s1, c3);
            if s2 != 0.0 {
                (s2, s3) = quick_two_sum(s2, c4);
            } else {
                (s1, s2) = quick_two_sum(s1, c4);
            }
        }
    } else {
        (s0, s1) = quick_two_sum(s0, c2);
        if s1 != 0.0 {
            (s1, s2) = quick_two_sum(s1, c3);
            if s2 != 0.0 {
                (s2, s3) = quick_two_sum(s2, c4);
            } else {
                (s1, s2) = quick_two_sum(s1, c4);
            }
        } else {
            (s0, s1) = quick_two_sum(s0, c3);
            if s1 != 0.0 {
                (s1, s2) = quick_two_sum(s1, c4);
            } else {
                (s0, s1) = quick_two_sum(s0, c4);
            }
        }
    }
    Qd([s0, s1, s2, s3])
}

/// Nombre quad-double : `x[0] + x[1] + x[2] + x[3]`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Qd(pub [f64; 4]);

impl Qd {
    pub const ZERO: Qd = Qd([0.0; 4]);

    #[inline(always)]
    pub fn from_f64(v: f64) -> Qd {
        Qd([v, 0.0, 0.0, 0.0])
    }

    /// Partie haute (approximation f64).
    #[inline(always)]
    pub fn hi(self) -> f64 {
        self.0[0]
    }

    /// Approximation double-double `(x0, x1 + x2)` en f64 — pour des
    /// comparaisons de magnitude plus fines que la partie haute seule.
    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        self.0[0] + (self.0[1] + (self.0[2] + self.0[3]))
    }

    #[inline(always)]
    pub fn neg(self) -> Qd {
        Qd([-self.0[0], -self.0[1], -self.0[2], -self.0[3]])
    }

    /// Multiplication par 2^k EXACTE (tant qu'aucune composante ne devient
    /// sous-normale ou infinie).
    #[inline(always)]
    pub fn mul_pow2(self, f: f64) -> Qd {
        Qd([self.0[0] * f, self.0[1] * f, self.0[2] * f, self.0[3] * f])
    }

    /// Addition (QD `sloppy_add`).
    #[inline(always)]
    pub fn add(self, b: Qd) -> Qd {
        let a = self.0;
        let b = b.0;
        let (s0, t0) = two_sum(a[0], b[0]);
        let (s1, t1) = two_sum(a[1], b[1]);
        let (mut s2, t2) = two_sum(a[2], b[2]);
        let (mut s3, t3) = two_sum(a[3], b[3]);
        let (s1, mut t0) = two_sum(s1, t0);
        let mut t1 = t1;
        three_sum(&mut s2, &mut t0, &mut t1);
        three_sum2(&mut s3, &mut t0, t2);
        let t0 = t0 + t1 + t3;
        renorm5(s0, s1, s2, s3, t0)
    }

    #[inline(always)]
    pub fn sub(self, b: Qd) -> Qd {
        self.add(b.neg())
    }

    /// Multiplication (QD `sloppy_mul`).
    #[inline(always)]
    pub fn mul(self, b: Qd) -> Qd {
        let a = self.0;
        let b = b.0;
        let (p0, q0) = two_prod(a[0], b[0]);
        let (mut p1, q1) = two_prod(a[0], b[1]);
        let (mut p2, q2) = two_prod(a[1], b[0]);
        let (mut p3, q3) = two_prod(a[0], b[2]);
        let (mut p4, q4) = two_prod(a[1], b[1]);
        let (mut p5, q5) = two_prod(a[2], b[0]);
        let mut q0 = q0;
        three_sum(&mut p1, &mut p2, &mut q0);
        let (mut q1, mut q2) = (q1, q2);
        three_sum(&mut p2, &mut q1, &mut q2);
        three_sum(&mut p3, &mut p4, &mut p5);
        let (s0, t0) = two_sum(p2, p3);
        let (s1, t1) = two_sum(q1, p4);
        let mut s2 = q2 + p5;
        let (mut s1, t0) = two_sum(s1, t0);
        s2 += t0 + t1;
        s1 += a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0] + q0 + q3 + q4 + q5;
        renorm5(p0, p1, s0, s1, s2)
    }

    /// Multiplication par un f64 (QD `mul_qd_d`).
    #[inline(always)]
    pub fn mul_f64(self, b: f64) -> Qd {
        let a = self.0;
        let (p0, q0) = two_prod(a[0], b);
        let (p1, q1) = two_prod(a[1], b);
        let (p2, q2) = two_prod(a[2], b);
        let p3 = a[3] * b;
        let s0 = p0;
        let (s1, mut s2) = two_sum(q0, p1);
        let (mut q1, mut p2) = (q1, p2);
        three_sum(&mut s2, &mut q1, &mut p2);
        let mut q2 = q2;
        three_sum2(&mut q1, &mut q2, p3);
        let s3 = q1;
        let s4 = q2 + p2;
        renorm5(s0, s1, s2, s3, s4)
    }

    /// Carré (QD `sqr`).
    #[inline(always)]
    pub fn sqr(self) -> Qd {
        let a = self.0;
        let (p0, q0) = two_prod(a[0], a[0]);
        let (p1, q1) = two_prod(2.0 * a[0], a[1]);
        let (p2, q2) = two_prod(2.0 * a[0], a[2]);
        let (p3, q3) = two_prod(a[1], a[1]);
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
        let p4 = 2.0 * a[0] * a[3];
        let p5 = 2.0 * a[1] * a[2];
        let (p4, p5) = two_sum(p4, p5);
        let (q2, q3) = two_sum(q2, q3);
        let (t0, t1) = two_sum(p4, q2);
        let t1 = t1 + p5 + q3;
        let (p3, p4) = two_sum(p3, t0);
        let p4 = p4 + q0 + t1;
        renorm5(p0, p1, p2, p3, p4)
    }

    /// Arrondi d'un `rug::Float` en qd : quatre soustractions successives de
    /// la partie haute (chacune exacte à la précision du tampon `scratch`,
    /// de précision ≥ celle de `v`). Hors plage f64 (|v| ≥ 2¹⁰²⁴) : ∞ ;
    /// les bits sous 2⁻¹⁰⁷⁴ sont perdus (erreur ABSOLUE ≤ 2⁻¹⁰⁷⁴).
    pub fn from_float(v: &rug::Float, scratch: &mut rug::Float) -> Qd {
        use rug::Assign;
        scratch.assign(v);
        let mut out = [0.0; 4];
        for o in out.iter_mut() {
            let h = scratch.to_f64();
            *o = h;
            if h == 0.0 || !h.is_finite() {
                break;
            }
            *scratch -= h;
        }
        Qd(out)
    }

    /// Valeur exacte en `rug::Float` (tests).
    pub fn to_float(self, prec: u32) -> rug::Float {
        let mut f = rug::Float::with_val(prec, self.0[0]);
        for &x in &self.0[1..] {
            f += x;
        }
        f
    }
}

/// Complexe quad-double.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ComplexQd {
    pub re: Qd,
    pub im: Qd,
}

impl ComplexQd {
    pub const ZERO: ComplexQd = ComplexQd { re: Qd::ZERO, im: Qd::ZERO };

    #[inline(always)]
    pub fn add(self, o: ComplexQd) -> ComplexQd {
        ComplexQd { re: self.re.add(o.re), im: self.im.add(o.im) }
    }

    /// Produit complexe (4 produits qd).
    #[inline(always)]
    pub fn mul(self, o: ComplexQd) -> ComplexQd {
        ComplexQd {
            re: self.re.mul(o.re).sub(self.im.mul(o.im)),
            im: self.re.mul(o.im).add(self.im.mul(o.re)),
        }
    }

    #[inline(always)]
    pub fn mul_pow2(self, f: f64) -> ComplexQd {
        ComplexQd { re: self.re.mul_pow2(f), im: self.im.mul_pow2(f) }
    }

    /// Approximation f64.
    #[inline(always)]
    pub fn hi(self) -> num_complex::Complex64 {
        num_complex::Complex64::new(self.re.to_f64(), self.im.to_f64())
    }

    pub fn from_complex(z: &rug::Complex, scratch: &mut rug::Float) -> ComplexQd {
        ComplexQd {
            re: Qd::from_float(z.real(), scratch),
            im: Qd::from_float(z.imag(), scratch),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Float;

    const P: u32 = 512;

    /// Générateur déterministe (xorshift) de qd « pleins » : 4 composantes
    /// non chevauchantes d'exposants décroissants.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        }
        fn qd(&mut self, scale: f64) -> Qd {
            let mut f = Float::with_val(P, self.next() * scale);
            for k in 1..5 {
                f += self.next() * scale * 2f64.powi(-53 * k);
            }
            let mut s = Float::new(P);
            Qd::from_float(&f, &mut s)
        }
    }

    fn rel_err(got: Qd, want: &Float, denom: &Float) -> f64 {
        let mut d = got.to_float(P);
        d -= want;
        (d.abs() / denom).to_f64()
    }

    #[test]
    fn from_float_round_trips_212_bits() {
        let mut s = Float::new(P);
        let mut f = Float::with_val(P, 1);
        f /= 3;
        let q = Qd::from_float(&f, &mut s);
        let back = q.to_float(P);
        let err = ((back - &f) / &f).to_f64().abs();
        assert!(err < 2f64.powi(-210), "err {err:e}");
    }

    /// Erreur de add/mul/sqr contre rug à 512 b, relative aux opérandes
    /// (modèle de la variante sloppy) : < 2⁻²⁰⁸, y compris en annulation.
    #[test]
    fn ops_match_rug_within_qd_unit() {
        let mut r = Rng(0x9E3779B97F4A7C15);
        let mut worst = [0.0f64; 3];
        for i in 0..20000 {
            let a = r.qd(2f64.powi((i % 40) - 20));
            let mut b = r.qd(2f64.powi(((i / 40) % 40) - 20));
            if i % 3 == 0 {
                // Annulation forcée : b ≈ −a à 2⁻⁶⁰ près.
                b = a.neg().add(b.mul_pow2(2f64.powi(-60)));
            }
            let (fa, fb) = (a.to_float(P), b.to_float(P));
            let sum = Float::with_val(P, &fa + &fb);
            let den = Float::with_val(P, fa.clone().abs() + fb.clone().abs());
            worst[0] = worst[0].max(rel_err(a.add(b), &sum, &den));
            let prod = Float::with_val(P, &fa * &fb);
            let den = Float::with_val(P, prod.clone().abs());
            worst[1] = worst[1].max(rel_err(a.mul(b), &prod, &den));
            let sq = Float::with_val(P, fa.square_ref());
            worst[2] = worst[2].max(rel_err(a.sqr(), &sq, &sq));
        }
        eprintln!("qd worst bits: {:?}", worst.map(|w| -w.log2()));
        for (k, w) in worst.iter().enumerate() {
            assert!(*w < 2f64.powi(-208), "op {k}: erreur relative {w:e} ({:.1} b)", -w.log2());
        }
    }

    #[test]
    fn mul_f64_and_pow2_exact() {
        let mut r = Rng(42);
        for _ in 0..1000 {
            let a = r.qd(1.0);
            let b = r.next();
            let want = Float::with_val(P, a.to_float(P) * b);
            let den = want.clone().abs();
            assert!(rel_err(a.mul_f64(b), &want, &den) < 2f64.powi(-208));
            let s = a.mul_pow2(2f64.powi(-700));
            let want = Float::with_val(P, a.to_float(P) >> 700);
            assert_eq!(s.to_float(P), want);
        }
    }
}
