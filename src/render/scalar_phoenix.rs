//! Fiabilité de **Phoenix** (boucle scalaire, hors bytecode) — 2026-11-20.
//!
//! Phoenix itère un COUPLE : `z' = z² + p₁ + p₂·y`, `y' = z` (`p₁ = 0,56667`,
//! `p₂ = −½`, `z₀` = le pixel, `y₀ = 0` ; `iterations::phoenix`). L'erreur
//! `e = (Δz, Δy)` évolue par le jacobien `J = [[2ẑ, p₂], [1, 0]]`. Une borne
//! en MODULES (`[[2|z|, ½], [1, 0]]`) a un rayon spectral
//! `|z| + √(|z|² + ½) > 1` dès `|z| > ¼` : elle s'emballait sur 100 % du bord
//! (double-double compris) alors que le vrai produit des jacobiens contracte
//! — les PHASES de `2ẑ` se compensent d'un pas à l'autre.
//!
//! **Borne ellipsoïdale** : on propage une matrice hermitienne 2×2
//! `P = [[a, b], [b̄, d]]` (`a, d` réels, `b` complexe : 4 réels) telle que
//! `e·eᴴ ⪯ P`. Pour `e' = J·e + w` (`w` = terme non linéaire + arrondi, sur
//! la seule composante `z` : `y' = ẑ` est exact),
//! `e'·e'ᴴ ⪯ (1 + δ)·J·P·Jᴴ + (1 + 1/δ)·diag(|w|², 0)`, soit
//!
//! ```text
//! a' = (1+δ)·(4|ẑ|²·a + 4p₂·Re(ẑ·b) + p₂²·d) + (1+1/δ)·r²
//! b' = (1+δ)·(2ẑ·a + p₂·b̄)
//! d' = (1+δ)·a
//! ```
//!
//! La partie linéaire est EXACTE (l'ellipsoïde est transporté par `J`, phases
//! comprises) ; le terme non linéaire `Δz²` vérifie `|Δz|² ≤ a` (`r = a + ρ`,
//! `ρ` l'arrondi du pas). Aucune racine. Pixel signalé si `√a` dépasse
//! [`E_MAX_NL`] ou si l'évasion n'est pas déterminée (`|ẑ|` à moins de `√a`
//! du rayon). Correction : même boucle en double-double (voies entrelacées,
//! raccourci de cycle sur l'état complet, ellipsoïde compris), puis GMP pur
//! sous le budget du path standard (`scalar_reliability::correct`).
//!
//! Mesuré (juge GMP pur 256 b ≡ 512 b) : bord de l'axe `types` ×4096
//! 64×40 2 pixels faux → 0 (128×80 : 13 → 0), 7,6 % de pixels signalés,
//! tous rendus fiables par le double-double ; accueil 0,1-0,2 % signalés.
//! Balayages de 50 + 48 vues de bord aléatoires (×30 à ×1e11, plafonds
//! 2000-20 000) : 283 pixels faux en f64, AUCUN non signalé, 0 faux après
//! correction, 4 % / 16 % signalés (`sweep_phoenix_edges`).
//! Coût : traqueur seul +55 % (mono-fil, `bench_phoenix_tracker`) ; rendu
//! 720p complet +24 % à l'accueil, +47 % au bord (correction dd comprise).
//!
//! Leçons :
//! - un ellipsoïde est EXACT sur la partie linéaire, une borne en modules non
//!   (rayon spectral du jacobien en modules > 1 alors que le vrai contracte) ;
//! - la boucle Phoenix est si légère (~11 opérations flottantes par pas) que
//!   le traqueur se paie au DÉBIT : produits-sommes fusionnés ([`fma`], sans
//!   jamais appeler la `fma` logicielle), carrés partagés avec la norme,
//!   `b` stocké mis à l'échelle, `d' = K_d·a` (marge d'arrondi hors
//!   diagonale reportée sur `a` par complément de Schur) ;
//! - le terme non linéaire `a²` n'est PAS négligeable au seuil : le pondérer
//!   ×17 pour alléger le bruit (×17/16) signale plus (7,6 → 8,3 %), et le
//!   linéariser sous un plafond `√a ≤ 1e-3` signale 100 % du bord — la borne
//!   de pixels justes vaut couramment 1e-3..0,25 ;
//! - majorer `|y|` par `√2·R` (constante) au lieu de `(1 + |y|²)/√2` : 9,3 %
//!   de pixels signalés au lieu de 7,6 %.

use num_complex::Complex64;
use rayon::prelude::*;

use crate::fractal::perturbation::dd::DoubleDouble as Dd;

/// Constantes de la boucle de production (`iterations::phoenix`, et
/// `gmp::phoenix_mpc` qui les forme depuis les MÊMES littéraux f64).
pub const P1: f64 = 0.56667;
pub const P2: f64 = -0.5;

/// Unité d'arrondi f64, avec un bit de garde.
const U64: f64 = 1.0 / (1u64 << 52) as f64;
/// Plafond de `√a` (borne non linéarisée, rigoureuse à grande erreur : seul
/// le test d'évasion juge), comme les autres boucles scalaires.
const E_MAX_NL: f64 = 0.25;
/// δ du découpage de Minkowski `(1 + δ)·A + (1 + 1/δ)·B`.
const SPLIT: f64 = 1.0 / 1024.0;
const SPLIT_X: f64 = 1.0 + SPLIT;
/// `1 + 1/δ`, majoré de l'arrondi de son propre produit.
const SPLIT_Y: f64 = (1.0 + 1.0 / SPLIT) * (1.0 + 1e-12);
/// Marge des arrondis du calcul de `P` lui-même, par unité de `a + d`
/// (cf. [`slack_coef`]).
const KAPPA: f64 = 16.0 * U64;

/// Coefficient `k2` de la marge `k2·(a + d)` ajoutée à `a'` : elle couvre
/// les arrondis du calcul de `P'` (une erreur `ε_b` hors diagonale est
/// couverte par `σ_a ≥ |ε_b|²/(η·a)`, `σ_d ≥ η·a` — Schur, `η = u` — ce qui
/// laisse `d' = K_d·a` sans terme en `d`).
/// Tant que la boucle tourne `|ẑ| < R`, et `P` est semi-définie positive
/// (`|b|₁ ≤ √2·√(ad) ≤ a + d`) : `|ε_a| ≤ 16u·(4R² + 5R + 3/2)·(a + d)`,
/// `|ε_b| ≤ 4u·(3R·a + |b|₁)` d'où `|ε_b|²/(u·a) ≤ 32u·(9R²·a + 2d)`.
#[inline(always)]
fn slack_coef(bailout: f64) -> f64 {
    KAPPA * (4.0 * bailout * bailout + 5.0 * bailout + 1.5) + 32.0 * U64 * (9.0 * bailout * bailout + 2.0)
}

/// `K_d` de `d' = K_d·a` : `(1 + δ)` plus `|ε_d| ≤ u·a`, `η = u` et
/// l'arrondi du produit.
const K_D: f64 = SPLIT_X + 8.0 * U64;

/// `a·b + c` : FMA matériel quand la cible l'a (`target-cpu=native`, ARM64),
/// sinon produit puis somme — PAS `f64::mul_add` sans FMA, qui appelle la
/// `fma` logicielle de la libm (binaires de release x86-64 de base).
/// Les deux formes majorent pareil : un arrondi de moins ou de plus est
/// couvert par la marge des arrondis de [`Ell::step`].
#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(any(target_feature = "fma", target_arch = "aarch64"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(any(target_feature = "fma", target_arch = "aarch64")))]
    {
        a * b + c
    }
}

/// Ellipsoïde d'erreur `P = [[a, b], [b̄, d]]` du couple `(Δz, Δy)`.
#[derive(Clone, Copy, Debug)]
struct Ell {
    a: f64,
    br: f64,
    bi: f64,
    d: f64,
}

impl Ell {
    /// Erreur initiale `e0` sur `z₀` ; `y₀ = 0` exact.
    #[inline(always)]
    fn start(e0: f64) -> Self {
        Ell { a: e0 * e0, br: 0.0, bi: 0.0, d: 0.0 }
    }

    /// Un pas : `P' = (1+δ)·J·P·Jᴴ + (1+1/δ)·|w|²·e₁e₁ᴴ` + marge des
    /// arrondis ([`slack_coef`]), `J = [[2ẑ, p₂], [1, 0]]` évalué en
    /// `(zr, zi)` (`n2p = zr² + zi²`). Le terme additif vérifie
    /// `|w| ≤ a + w_ext` (`|Δz²| ≤ a`), majoré par `|w|² ≤ 2a² + 2·w_ext²` ;
    /// l'appelant passe `w = 2(1+1/δ)·w_ext²` (termes positifs : la marge
    /// relative 1e-12 de `SPLIT_Y` couvre leurs arrondis).
    ///
    /// `b` est stocké divisé par `2(1+δ)` (`br`, `bi` = `b̃`) : `b̃' = ẑ·a +
    /// (1+δ)p₂·conj(b̃)`, le facteur passe dans la constante de `Re(ẑ·b)`.
    ///
    /// La boucle f64 tracée est bornée par le DÉBIT : chaque opération se
    /// paie (premier jet +155 %, puis +82 % avec une chaîne de `a` réduite à
    /// `a·(c + 2(1+1/δ)·a) + reste`, +64 % en produits-sommes [`fma`] et
    /// carrés partagés, puis `b̃`, `d' = K_d·a` et `w` préformé).
    #[inline(always)]
    fn step(self, zr: f64, zi: f64, n2p: f64, w: f64, k2: f64) -> Self {
        let Ell { a, br, bi, d } = self;
        // Re(ẑ·b̃) = zr·br − zi·bi.
        let re_zb = fma(zr, br, -(zi * bi));
        let rest = fma(
            SPLIT_X * P2 * P2 + k2,
            d,
            fma(4.0 * SPLIT_X * P2 * (2.0 * SPLIT_X), re_zb, w),
        );
        let na = fma(a, fma(2.0 * SPLIT_Y, a, fma(4.0 * SPLIT_X, n2p, k2)), rest);
        Ell {
            a: na,
            br: fma(zr, a, (SPLIT_X * P2) * br),
            bi: fma(zi, a, -((SPLIT_X * P2) * bi)),
            d: K_D * a,
        }
    }

    /// Le pas qui vient de finir laisse-t-il le pixel indéterminé ? `n2` :
    /// `|ẑ|²` tel que la boucle le compare à `b2`, à `n2_err` près ;
    /// `|Δz| ≤ √a`, et `|z| ∈ [R − √a, R + √a]` implique
    /// `||z|² − R²| ≤ √a·(2R + √a)` — testé au carré, sans racine.
    #[inline(always)]
    fn undetermined(&self, n2: f64, b2: f64, two_r_emax_sq: f64, n2_err: f64) -> bool {
        let m = (n2 - b2).abs() - n2_err;
        m <= 0.0 || m * m <= self.a * two_r_emax_sq || !(self.a < E_MAX_NL * E_MAX_NL)
    }
}

/// Phoenix f64 tracé : compte, `z` final et cycle exact IDENTIQUES au bit
/// près à `iterations::phoenix` (verrou). `e0` : erreur du point de départ
/// (le pixel) par rapport au point exact du juge. Renvoie
/// `(itérations, z, non_fiable)`.
#[inline]
pub fn phoenix_f64(z0: Complex64, e0: f64, iteration_max: u32, bailout: f64) -> (u32, Complex64, bool) {
    let b2 = bailout * bailout;
    let two_r_emax_sq = (2.0 * bailout + E_MAX_NL) * (2.0 * bailout + E_MAX_NL);
    let p1 = P1;
    let p2 = P2;
    let mut z = z0;
    let mut y = Complex64::new(0.0, 0.0);
    let mut i = 0u32;
    let mut cyc = crate::fractal::cycle::CycleCut::<4>::new();
    let k2 = slack_coef(bailout);
    // Arrondi du pas (par composante ≤ u·|termes|) : carré ≤ 2u·|z|²
    // (|z²|₁ ≤ 2|z|²), + p₁, produit par p₂ exact, somme finale
    // ≤ u·|z'|₁ ≤ 1,01·u·(2|z|² + p₁ + ½|y|₁), et |y|₁ ≤ √2·|y|
    // ≤ (1 + |y|²)/√2 : ρ = c₁·|z|² + c_y·|y|² + c₀, sans valeur absolue ni
    // racine (un majorant constant de |y| — √2·R — signalait 9,3 % du bord
    // ×4096 au lieu de 7,6 %). Préformé par √(2(1+1/δ)) (majoré) : le bruit
    // du pas est ρ'² directement.
    let sw = (2.0 * SPLIT_Y).sqrt() * (1.0 + 1e-12);
    let rho_1 = sw * (4.02 * U64 * 2.0);
    let rho_y = sw * (4.02 * U64 * 0.3536);
    let rho_0 = sw * (4.02 * U64 * (p1 + 0.3536));
    let mut n2y = 0.0f64; // |y|²
    // Sous ce rayon (au carré) le pas est déterminé dès que √a < E_MAX_NL :
    // |z*| ≤ |ẑ| + √a < R. Le test complet n'y est pas fait ; la condition
    // √a < E_MAX_NL est vérifiée sur le MAXIMUM de `a` en fin de boucle.
    let b2_safe = if bailout > E_MAX_NL { (bailout - E_MAX_NL) * (bailout - E_MAX_NL) } else { 0.0 };
    let mut ell = Ell::start(e0);
    let mut a_max = 0.0f64;
    // Mêmes arrondis que `iterations::phoenix`, en partageant les carrés :
    // `z * z` (num-complex) forme `zr·zr − zi·zi` et `zr·zi + zi·zr`, et
    // `norm_sqr` `zr·zr + zi·zi`. Les deux produits croisés sont le même
    // flottant, leur somme est son double EXACT ; les carrés servent à la
    // norme du test de boucle ET au pas suivant. Trois produits de moins par
    // pas, rendus au traqueur (boucle bornée par le débit).
    let mut sr = z.re * z.re;
    let mut si = z.im * z.im;
    let mut n2 = sr + si;
    let mut bad = false;
    while i < iteration_max && n2 < b2 {
        let t = z.re * z.im;
        let z_temp = Complex64::new((sr - si + p1) + y.re * p2, (t + t) + y.im * p2);
        let zc = z;
        let n2p = n2;
        y = z;
        z = z_temp;
        i += 1;
        sr = z.re * z.re;
        si = z.im * z.im;
        n2 = sr + si;
        if !bad {
            let rho = fma(rho_1, n2p, fma(rho_y, n2y, rho_0));
            n2y = n2p;
            ell = ell.step(zc.re, zc.im, n2p, rho * rho, k2);
            a_max = a_max.max(ell.a);
            // L'arrondi de la norme f64 de la boucle (≤ 2u·|z|²) est couvert :
            // √a ≥ √(1+1/δ)·ρ ≫ 2u·|z|²/(2R).
            if n2 >= b2_safe && ell.undetermined(n2, b2, two_r_emax_sq, 0.0) {
                bad = true;
            }
        }
        cyc.observe(&mut i, iteration_max, [z.re, z.im, y.re, y.im]);
    }
    // `max` propage NaN comme une borne finie : NaN n'est jamais < E².
    bad |= !(a_max < E_MAX_NL * E_MAX_NL) || ell.a.is_nan();
    (i, z, bad)
}

// --- Étage double-double -----------------------------------------------------

/// Unité d'arrondi double-double, avec deux bits de garde.
const UDD: f64 = crate::fractal::bytecode::direct_dd::DD_UNIT;

#[derive(Clone, Copy)]
struct Cdd {
    re: Dd,
    im: Dd,
}

#[inline(always)]
fn hi(z: Cdd) -> Complex64 {
    Complex64::new(z.re.hi, z.im.hi)
}

/// État d'une voie de l'étage dd : `z`, `y`, l'ellipsoïde.
#[derive(Clone, Copy)]
struct DdState {
    z: Cdd,
    y: Cdd,
    ell: Ell,
}

impl DdState {
    #[inline(always)]
    fn start(z0: Cdd) -> Self {
        let e0 = UDD * (z0.re.hi.abs() + z0.im.hi.abs());
        DdState { z: z0, y: Cdd { re: Dd::ZERO, im: Dd::ZERO }, ell: Ell::start(e0) }
    }

    #[inline(always)]
    fn n2(&self) -> f64 {
        self.z.re.hi * self.z.re.hi + self.z.im.hi * self.z.im.hi
    }

    /// État complet pour le raccourci des cycles exacts : `z`, `y` (deux mots
    /// par composante) ET l'ellipsoïde — le pas suivant lit tout, un état
    /// répété répète tout l'avenir, borne comprise.
    #[inline(always)]
    fn cycle_bits(&self) -> [f64; 12] {
        let Ell { a, br, bi, d } = self.ell;
        [
            self.z.re.hi, self.z.re.lo, self.z.im.hi, self.z.im.lo, self.y.re.hi, self.y.re.lo,
            self.y.im.hi, self.y.im.lo, a, br, bi, d,
        ]
    }

    /// Un pas dd tracé : `(état', |z'|² en f64 sur les parties hautes)`.
    #[inline(always)]
    fn step(self, k2: f64) -> (Self, f64) {
        let DdState { z, y, ell } = self;
        let re = z.re.sqr().sub(z.im.sqr()).add(Dd::from_f64(P1)).add(y.re.mul_f64(P2));
        let im = z.re.mul(z.im).mul_f64(2.0).add(y.im.mul_f64(P2));
        let n2p = z.re.hi * z.re.hi + z.im.hi * z.im.hi;
        let n2 = re.hi * re.hi + im.hi * im.hi;
        // Arrondis dd (sommes exactes à ~2⁻¹⁰⁴ près des opérandes).
        let rho = 2.0
            * UDD
            * (2.0 * n2p + P1 + 0.5 * (y.re.hi.abs() + y.im.hi.abs()) + re.hi.abs() + im.hi.abs());
        // Le jacobien est évalué sur les parties HAUTES : l'écart exact 2·ẑ_lo
        // agit sur Δz comme un terme additif de carré ≤ 4|ẑ_lo|²·a, d'où
        // w_ext = ρ + q, w_ext² ≤ 2ρ² + 2q².
        let q2 = 4.0 * (z.re.lo * z.re.lo + z.im.lo * z.im.lo) * ell.a;
        let w = (4.0 * SPLIT_Y) * (rho * rho + q2) * (1.0 + 1e-12);
        let ell = ell.step(z.re.hi, z.im.hi, n2p, w, k2);
        (DdState { z: Cdd { re, im }, y: z, ell }, n2)
    }
}

/// Fin d'un pas dd : le pixel reste-t-il déterminé ? La norme de la boucle
/// est calculée en f64 sur les parties hautes (erreur ≤ 4u₆₄·|z|²).
#[inline(always)]
fn dd_reliable(st: &DdState, n2: f64, b2: f64, two_r_emax_sq: f64) -> bool {
    n2.is_finite() && !st.ell.undetermined(n2, b2, two_r_emax_sq, 4.0 * U64 * n2)
}

/// Un pixel dd tracé (référence des voies de [`dd_lanes`]) :
/// `(itérations, z, fiable)`.
#[cfg(test)]
fn dd_pixel(z0: Cdd, iteration_max: u32, bailout: f64) -> (u32, Complex64, bool) {
    let b2 = bailout * bailout;
    let two_r_emax_sq = (2.0 * bailout + E_MAX_NL) * (2.0 * bailout + E_MAX_NL);
    let k2 = slack_coef(bailout);
    let mut st = DdState::start(z0);
    let mut i = 0u32;
    let mut cyc = crate::fractal::cycle::CycleCut::<12>::new();
    while i < iteration_max && st.n2() < b2 {
        let (ns, n2) = st.step(k2);
        st = ns;
        i += 1;
        if !dd_reliable(&st, n2, b2, two_r_emax_sq) {
            return (i, hi(st.z), false);
        }
        cyc.observe(&mut i, iteration_max, st.cycle_bits());
    }
    (i, hi(st.z), true)
}

/// Voies entrelacées (même schéma que les autres types suivis : le pas dd
/// est une chaîne dépendante de `two_sum`/`two_prod` bornée par sa latence).
/// Chaque voie fait la suite d'opérations de [`dd_pixel`] : résultats
/// identiques (verrou `phoenix_dd_lanes_match_single_pixel_loop`).
const DD_LANES: usize = 4;

fn dd_lanes(points: &[Cdd], iteration_max: u32, bailout: f64) -> Vec<(u32, Complex64, bool)> {
    const FREE: usize = usize::MAX;
    let b2 = bailout * bailout;
    let two_r_emax_sq = (2.0 * bailout + E_MAX_NL) * (2.0 * bailout + E_MAX_NL);
    let k2 = slack_coef(bailout);
    let zero = Cdd { re: Dd::ZERO, im: Dd::ZERO };
    let mut out = vec![(0u32, Complex64::new(0.0, 0.0), false); points.len()];
    let mut next = 0usize;
    let mut idx = [FREE; DD_LANES];
    let mut st = [DdState::start(zero); DD_LANES];
    let mut n2 = [0.0f64; DD_LANES];
    let mut it = [0u32; DD_LANES];
    let mut cyc: [crate::fractal::cycle::CycleCut<12>; DD_LANES] =
        std::array::from_fn(|_| crate::fractal::cycle::CycleCut::new());
    loop {
        // Sorties par la condition de boucle, puis recharge des voies libres.
        for l in 0..DD_LANES {
            loop {
                if idx[l] != FREE {
                    let n2 = st[l].n2();
                    // Point de départ : sortie déterminée comme celle d'un pas.
                    let ok0 = it[l] > 0 || dd_reliable(&st[l], n2, b2, two_r_emax_sq);
                    if ok0 && it[l] < iteration_max && n2 < b2 {
                        break;
                    }
                    out[idx[l]] = (it[l], hi(st[l].z), ok0);
                    idx[l] = FREE;
                }
                if next == points.len() {
                    break;
                }
                idx[l] = next;
                st[l] = DdState::start(points[next]);
                cyc[l] = crate::fractal::cycle::CycleCut::new();
                it[l] = 0;
                next += 1;
            }
        }
        if idx.iter().all(|&k| k == FREE) {
            break;
        }
        // Un pas sur toutes les voies (une voie libre calcule pour rien).
        for l in 0..DD_LANES {
            let (ns, nn) = st[l].step(k2);
            st[l] = ns;
            n2[l] = nn;
        }
        for l in 0..DD_LANES {
            if idx[l] != FREE {
                it[l] += 1;
                if !dd_reliable(&st[l], n2[l], b2, two_r_emax_sq) {
                    out[idx[l]] = (it[l], hi(st[l].z), false);
                    idx[l] = FREE;
                } else {
                    cyc[l].observe(&mut it[l], iteration_max, st[l].cycle_bits());
                }
            }
        }
    }
    out
}

/// Étage double-double de la correction : `points` = point exact de chaque
/// pixel (centre HP + dc en GMP), arrondi en dd. `(itérations, z, fiable)`.
pub(crate) fn dd_stage(points: &[(Dd, Dd)], iteration_max: u32, bailout: f64) -> Vec<(u32, Complex64, bool)> {
    const CHUNK: usize = 64;
    let pts: Vec<Cdd> = points.iter().map(|&(re, im)| Cdd { re, im }).collect();
    pts.par_chunks(CHUNK).flat_map_iter(|ch| dd_lanes(ch, iteration_max, bailout)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::iterations::iterate_point;
    use crate::fractal::{default_params_for_type, FractalParams, FractalType};

    fn scene(view: Option<(f64, f64, f64)>, w: usize, h: usize, n: u32) -> FractalParams {
        let mut p = default_params_for_type(FractalType::Phoenix, w as u32, h as u32);
        if let Some((cx, cy, zoom)) = view {
            p.center_x = cx;
            p.center_y = cy;
            p.span_x = 4.0 / zoom;
            p.span_y = p.span_x * h as f64 / w as f64;
        }
        p.iteration_max = n;
        p
    }

    fn points(p: &FractalParams, w: usize, h: usize) -> Vec<Complex64> {
        (0..w * h)
            .map(|k| {
                let (i, j) = (k % w, k / w);
                Complex64::new(
                    p.center_x + ((i as f64 + 0.5) / w as f64 - 0.5) * p.span_x,
                    p.center_y + ((j as f64 + 0.5) / h as f64 - 0.5) * p.span_y,
                )
            })
            .collect()
    }

    fn pixel_error(p: &FractalParams, z: Complex64) -> f64 {
        U64 * (z.re.abs() + z.im.abs() + p.span_x.abs() + p.span_y.abs())
    }

    /// Scène de bord de l'axe `types` (`harness/type-scenes.json`).
    const EDGE: (f64, f64, f64) = (0.151763916015625, -0.415435791015625, 4096.0);

    /// Juge GMP pur (le GMP par pixel du path standard, `use_gmp`).
    fn gmp_iters(p: &FractalParams, bits: u32) -> Vec<u32> {
        let mut g = p.clone();
        g.engine.algorithm_mode = crate::fractal::AlgorithmMode::ReferenceGmp;
        g.engine.use_gmp = true;
        g.engine.precision_bits = bits;
        crate::render::render_escape_time(&g).iterations
    }

    #[test]
    fn phoenix_dd_lanes_match_single_pixel_loop() {
        let (cx, cy, span) = (EDGE.0, EDGE.1, 4e-3);
        let pts: Vec<Cdd> = (0..103)
            .map(|k| {
                let (i, j) = ((k % 11) as f64, (k / 11) as f64);
                let re = Dd::from_f64(cx + span * (i / 10.0 - 0.5)).add(Dd::new(1e-20 * i, 0.0));
                Cdd { re, im: Dd::from_f64(cy + span * (j / 9.0 - 0.5)) }
            })
            .collect();
        let lanes = dd_lanes(&pts, 3000, 4.0);
        let mut capped = 0;
        for (k, (&pc, &(it, z, ok))) in pts.iter().zip(&lanes).enumerate() {
            let (it1, z1, ok1) = dd_pixel(pc, 3000, 4.0);
            assert_eq!((it, ok), (it1, ok1), "point {k}");
            assert_eq!(z.re.to_bits(), z1.re.to_bits(), "point {k}");
            assert_eq!(z.im.to_bits(), z1.im.to_bits(), "point {k}");
            capped += (it == 3000) as usize;
        }
        assert!(capped > 0 && capped < pts.len(), "grille sans mélange ({capped})");
    }

    /// Ventilation sur les scènes de l'axe `types` : signalés, faux, faux non
    /// signalés (f64), et ce que l'étage dd décide.
    #[test]
    #[ignore]
    fn diag_phoenix_ellipsoid() {
        for (view, w, h) in [(Some(EDGE), 64usize, 40usize), (None, 64, 40), (Some(EDGE), 128, 80)] {
            let p = scene(view, w, h, 5000);
            let pts = points(&p, w, h);
            let gmp = gmp_iters(&p, 256);
            let gmp512 = gmp_iters(&p, 512);
            let unconv = gmp.iter().zip(&gmp512).filter(|(a, b)| a != b).count();
            let (mut flagged, mut wrong, mut wrong_unflagged) = (0, 0, 0);
            let mut fl = Vec::new();
            for (k, &zp) in pts.iter().enumerate() {
                let r = iterate_point(&p, zp);
                let (it, z, bad) = phoenix_f64(zp, pixel_error(&p, zp), p.iteration_max, p.bailout);
                assert_eq!(it, r.iteration);
                assert_eq!(z.re.to_bits(), r.z.re.to_bits());
                if bad {
                    flagged += 1;
                    fl.push(k);
                }
                if it != gmp[k] {
                    wrong += 1;
                    if !bad {
                        wrong_unflagged += 1;
                        eprintln!("  non signalé : pixel {k} f64 {it} gmp {}", gmp[k]);
                    }
                }
            }
            let dpts: Vec<(Dd, Dd)> =
                fl.iter().map(|&k| (Dd::from_f64(pts[k].re), Dd::from_f64(pts[k].im))).collect();
            let res = dd_stage(&dpts, p.iteration_max, p.bailout);
            let ok = res.iter().filter(|r| r.2).count();
            let ok_wrong = fl.iter().zip(&res).filter(|(&k, r)| r.2 && r.0 != gmp[k]).count();
            eprintln!(
                "{view:?} {w}x{h} : juge 256≠512 {unconv} ; signalés {flagged} ({:.1} %), faux {wrong}, faux non signalés {wrong_unflagged} ; dd fiables {ok}/{flagged} (dont faux {ok_wrong})",
                100.0 * flagged as f64 / pts.len() as f64
            );
        }
    }

    /// Balayage de vues de bord Phoenix aléatoires contre GMP pur convergé
    /// (256 b ≡ 512 b, pixels non convergés exclus) : f64 tracé (faux non
    /// signalés) ET rendu de production (dd + GMP sous budget).
    /// `cargo test --release --lib sweep_phoenix_edges -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn sweep_phoenix_edges() {
        let (w, h) = (48usize, 30usize);
        // `FRACTALL_PHOENIX_SWEEP=deep` : graine et profondeurs ×1e4..1e11 (second balayage).
        let deep = std::env::var("FRACTALL_PHOENIX_SWEEP").as_deref() == Ok("deep");
        let mut rng = if deep { 0x2545f4914f6cdd1du64 } else { 0x9e3779b97f4a7c15u64 };
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut views, mut tries) = (0, 0);
        let (mut tot_px, mut tot_flag, mut tot_raw, mut tot_unflag, mut tot_prod, mut tot_unconv) =
            (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        let mut hardest = (0usize, 0.0f64, 0.0f64, 0.0f64, 0u32);
        while views < 50 && tries < 5000 {
            tries += 1;
            let cx = -1.6 + 3.2 * next();
            let cy = -1.2 + 2.4 * next();
            let zoom = if deep { 10f64.powf(4.0 + 7.0 * next()) } else { 10f64.powf(1.5 + 4.5 * next()) };
            let n = [2000u32, 5000, 20000][(next() * 3.0) as usize % 3];
            let p = scene(Some((cx, cy, zoom)), w, h, n);
            let pts = points(&p, w, h);
            let raw: Vec<(u32, bool)> = pts
                .iter()
                .map(|&zp| {
                    let (it, _, bad) = phoenix_f64(zp, pixel_error(&p, zp), n, p.bailout);
                    (it, bad)
                })
                .collect();
            let capped = raw.iter().filter(|r| r.0 >= n).count() as f64 / raw.len() as f64;
            let distinct = raw.iter().map(|r| r.0).collect::<std::collections::HashSet<_>>().len();
            if !(0.1..=0.9).contains(&capped) || distinct < 40 {
                continue;
            }
            views += 1;
            let g256 = gmp_iters(&p, 256);
            let g512 = gmp_iters(&p, 512);
            let prod = crate::render::render_escape_time(&p).iterations;
            let (mut flag, mut rw, mut unflag, mut pw, mut unconv) = (0, 0, 0, 0, 0);
            for k in 0..pts.len() {
                if g256[k] != g512[k] {
                    unconv += 1;
                    continue;
                }
                flag += raw[k].1 as usize;
                if raw[k].0 != g256[k] {
                    rw += 1;
                    unflag += (!raw[k].1) as usize;
                }
                pw += (prod[k] != g256[k]) as usize;
            }
            eprintln!(
                "vue {views:2} ({cx:.6}, {cy:.6}) ×{zoom:.0} n={n} : plafonnés {:.0} %, signalés {flag} ({:.1} %), f64 faux {rw} (non signalés {unflag}), production faux {pw}, juge non convergé {unconv}",
                100.0 * capped,
                100.0 * flag as f64 / pts.len() as f64
            );
            if rw > hardest.0 {
                hardest = (rw, cx, cy, zoom, n);
            }
            tot_px += pts.len();
            tot_flag += flag;
            tot_raw += rw;
            tot_unflag += unflag;
            tot_prod += pw;
            tot_unconv += unconv;
        }
        eprintln!(
            "TOTAL {views} vues : {tot_px} px, signalés {tot_flag} ({:.2} %), f64 faux {tot_raw}, non signalés {tot_unflag}, production faux {tot_prod}, non convergés {tot_unconv} ; vue la plus dure {hardest:?}",
            100.0 * tot_flag as f64 / tot_px as f64
        );
        assert_eq!(tot_unflag, 0);
        assert_eq!(tot_prod, 0);
    }

    /// Coût du traqueur seul, mono-fil, min de 9 passes alternées.
    #[test]
    #[ignore]
    fn bench_phoenix_tracker() {
        for view in [None, Some(EDGE)] {
            let (w, h) = (128usize, 80usize);
            let p = scene(view, w, h, 5000);
            let pts = points(&p, w, h);
            let (mut tp, mut tt) = (f64::MAX, f64::MAX);
            let mut sink = 0u64;
            let mut flagged = 0;
            for _ in 0..9 {
                let t = std::time::Instant::now();
                for &zp in &pts {
                    sink += iterate_point(&p, zp).iteration as u64;
                }
                tp = tp.min(t.elapsed().as_secs_f64());
                let t = std::time::Instant::now();
                flagged = 0;
                for &zp in &pts {
                    let (it, _, bad) = phoenix_f64(zp, pixel_error(&p, zp), p.iteration_max, p.bailout);
                    sink += it as u64;
                    flagged += bad as usize;
                }
                tt = tt.min(t.elapsed().as_secs_f64());
            }
            eprintln!(
                "{}: prod {tp:.4}s tracked {tt:.4}s (+{:.0} %), signalés {:.2} % [{sink}]",
                if view.is_some() { "edge" } else { "home" },
                100.0 * (tt / tp - 1.0),
                100.0 * flagged as f64 / pts.len() as f64
            );
        }
    }
}
