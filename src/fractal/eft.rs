//! Transformations sans erreur (EFT) partagées par l'arithmétique
//! double-double (`perturbation::dd`), quad-double (`perturbation::qd`) et
//! les étages directs en voies (`bytecode::direct_simd`).
//!
//! ⚠️ **`two_prod` et les binaires distribués.** Les binaires de release
//! x86_64 sont compilés en `target-cpu=x86-64` (baseline : ni FMA ni AVX).
//! Sans `fma` à la compilation, `f64::mul_add` n'est PAS une instruction :
//! c'est un appel à la libm — glibc aiguille vers l'instruction sur un CPU à
//! FMA (≈ un appel de 4 ns), mais le fma est LOGICIEL sur un CPU sans FMA
//! (et selon la libm de la cible, MinGW). `two_prod` choisit donc :
//! - FMA connu à la compilation (`target-cpu=native` récent, `apple-m1`,
//!   `cortex-a72` — `mul_add` y est toujours une instruction hors x86_64) →
//!   `a.mul_add(b, -p)` en ligne ;
//! - baseline x86-64 sur CPU à FMA (détecté UNE fois, `OnceLock`) → appel
//!   DIRECT d'une fonction compilée avec `fma` ([`fma_call`]). Mesuré : le
//!   découpage de Dekker scalaire y est PLUS LENT que l'appel (étage qd
//!   d'e227 : 0,42 → 0,60 s) — 17 opérations contre un `vfmadd` ;
//! - CPU sans FMA → découpage de Dekker/Veltkamp ([`dekker_err`]).
//!
//! Les boucles où le FMA en ligne paie vraiment (étages directs en voies,
//! noyau par lots, boucle rapide de perturbation) ont en plus une copie
//! compilée `avx2,fma`, choisie par [`avx2_fma_dispatch`].
//!
//! Tous rendent l'erreur EXACTE `a·b − fl(a·b)` — mêmes bits — dans le
//! domaine où Dekker est exact : pas de débordement du découpage
//! (`|a|, |b| ≤ 2⁹⁹⁵`) et pas de sous-dépassement des produits partiels
//! (`|fl(a·b)| ≥ 2⁻⁹⁶⁷`). Hors de ce domaine (jamais atteint par les boucles
//! dd/qd, qui travaillent dans un cadre mis à l'échelle ou sur des
//! orbites bornées par le rayon d'évasion) la voie Dekker retombe sur
//! `mul_add` : identité au bit près GARANTIE, quel que soit l'opérande. Les
//! opérandes nuls sont dans le domaine (Dekker et FMA rendent `+0`).
//! Verrou : `dekker_two_prod_matches_fma_bit_for_bit` (les deux
//! implémentations appelées EXPLICITEMENT, quel que soit le target).

/// Somme sans erreur de deux f64 (Knuth `two_sum`) : `s = fl(a + b)` et
/// `a + b = s + e` exactement, sans hypothèse sur les magnitudes.
#[inline(always)]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// Somme sans erreur rapide (Dekker `fast_two_sum`) : exige `|a| ≥ |b|`
/// (ou `a = 0`).
#[inline(always)]
pub fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

/// Produit sans erreur : `p = fl(a·b)`, `a·b = p + e` exactement.
/// - FMA connu à la compilation (ou hors x86_64, où `mul_add` est toujours
///   une instruction) : `mul_add` en ligne ;
/// - x86_64 baseline, CPU avec FMA (détecté UNE fois, [`fma_runtime`]) :
///   appel direct d'une fonction compilée avec `fma` ([`fma_hw`]) ;
/// - x86_64 sans FMA : Dekker gardé.
/// Mêmes bits dans les trois cas.
#[inline(always)]
pub fn two_prod(a: f64, b: f64) -> (f64, f64) {
    if fma_inline() {
        two_prod_fma(a, b)
    } else if fma_runtime() {
        let p = a * b;
        // SAFETY : FMA détecté (`fma_runtime`).
        (p, unsafe { fma_call(a, b, -p) })
    } else {
        two_prod_dekker(a, b)
    }
}

/// `mul_add` est-il une instruction en ligne dans CE binaire ?
#[inline(always)]
pub const fn fma_inline() -> bool {
    cfg!(target_feature = "fma") || !cfg!(target_arch = "x86_64")
}

/// Le CPU a-t-il le FMA ? Détecté une fois (`OnceLock`), jamais relu dans
/// le chemin par pixel autrement que par cette cellule.
#[inline(always)]
pub fn fma_runtime() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static FMA: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *FMA.get_or_init(|| std::arch::is_x86_feature_detected!("fma"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        true
    }
}

/// Aiguillage à l'exécution vers les variantes `#[target_feature(enable =
/// "avx2,fma")]` des noyaux chauds : seulement dans un binaire qui ne les a
/// PAS à la compilation (baseline x86-64 des releases) et sur un CPU qui les
/// a. Lu une fois (`OnceLock`, détection `cpuid` + interrupteur A/B
/// `FRACTALL_SIMD_DISPATCH=0`), jamais dans le chemin par pixel autrement
/// que par cette cellule.
#[inline(always)]
pub fn avx2_fma_dispatch() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if cfg!(all(target_feature = "avx2", target_feature = "fma")) {
            return false;
        }
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            let off = matches!(
                std::env::var("FRACTALL_SIMD_DISPATCH").ok().as_deref().map(str::trim),
                Some("0") | Some("off")
            );
            !off && std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `a·b + c` arrondi une fois, par l'instruction FMA.
///
/// # Safety
/// [`fma_runtime`] (ou [`fma_inline`]) doit être vrai : sinon l'instruction
/// FMA est exécutée sur un CPU qui ne l'a pas (`SIGILL`). `unsafe` parce que
/// la précondition n'est vérifiée qu'en debug — une fonction SÛRE et
/// publique ne doit pas pouvoir provoquer cela.
#[inline(always)]
pub unsafe fn fma_call(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if fma_inline() {
            return a.mul_add(b, c);
        }
        debug_assert!(fma_runtime());
        // SAFETY : `fma_hw` n'exige que la fonctionnalité CPU `fma`, que
        // l'appelant garantit (contrat de `fma_call`, détection `cpuid`).
        unsafe { fma_hw(a, b, c) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        a.mul_add(b, c)
    }
}

/// Compilée avec `fma` : `mul_add` y est l'instruction `vfmadd`. Non
/// inlinable dans un appelant sans `fma` : c'est un appel direct (plus de
/// passage par la libm, dont la version Windows/MinGW ou sans FMA est
/// logicielle).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
#[inline]
unsafe fn fma_hw(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

/// `two_prod` par FMA (`mul_add` : appel de bibliothèque sans `fma` à la
/// compilation — référence des verrous).
#[inline(always)]
pub fn two_prod_fma(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

/// `two_prod` par découpage de Dekker, repli `mul_add` hors du domaine
/// exact ([`dekker_exact`]) — identique au bit près à [`two_prod_fma`].
#[inline(always)]
pub fn two_prod_dekker(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    if dekker_exact(a, b, p) {
        (p, dekker_err(a, b, p))
    } else {
        (p, dekker_fallback(a, b, p))
    }
}

/// Hors domaine : rare, gardé hors ligne pour ne pas grossir les boucles.
#[cold]
#[inline(never)]
fn dekker_fallback(a: f64, b: f64, p: f64) -> f64 {
    a.mul_add(b, -p)
}

/// Seuil bas sur `|fl(a·b)|` : au-dessus, tous les produits partiels du
/// découpage (jusqu'à `al·bl`, ~2⁻¹⁰⁶·|a·b|) et l'erreur sont des normaux
/// exacts (marge de 2 sur l'exposant de `a·b`).
const DEKKER_P_MIN: f64 = f64::from_bits((1023 - 967) << 52); // 2⁻⁹⁶⁷
/// Seuil haut sur `|fl(a·b)|` : `ah·bh` reste fini.
const DEKKER_P_MAX: f64 = f64::from_bits((1023 + 1020) << 52); // 2¹⁰²⁰
/// Seuil sur les opérandes : `(2²⁷+1)·a` ne déborde pas.
const DEKKER_OP_MAX: f64 = f64::from_bits((1023 + 995) << 52); // 2⁹⁹⁵

/// `true` si le découpage de Dekker rend l'erreur EXACTE pour `(a, b, p)`.
/// Opérandes nuls compris (`p = ±0` avec un facteur nul : Dekker rend `+0`
/// comme le FMA). NaN / infinis : `false`.
#[inline(always)]
pub fn dekker_exact(a: f64, b: f64, p: f64) -> bool {
    let ap = p.abs();
    a.abs() <= DEKKER_OP_MAX
        && b.abs() <= DEKKER_OP_MAX
        && ap <= DEKKER_P_MAX
        && (ap >= DEKKER_P_MIN || a == 0.0 || b == 0.0)
}

/// Erreur exacte `a·b − p` par découpage de Dekker/Veltkamp (sans FMA),
/// SANS garde : n'appeler que dans le domaine de [`dekker_exact`].
#[inline(always)]
pub fn dekker_err(a: f64, b: f64, p: f64) -> f64 {
    const SPLIT: f64 = 134_217_729.0; // 2^27 + 1
    let t = SPLIT * a;
    let ah = t - (t - a);
    let al = a - ah;
    let t = SPLIT * b;
    let bh = t - (t - b);
    let bl = b - bh;
    ((ah * bh - p) + ah * bl + al * bh) + al * bl
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dekker_thresholds_are_the_documented_powers_of_two() {
        assert_eq!(DEKKER_P_MIN, 2f64.powi(-967));
        assert_eq!(DEKKER_P_MAX, 2f64.powi(1020));
        assert_eq!(DEKKER_OP_MAX, 2f64.powi(995));
    }

    fn xorshift(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    fn same(a: f64, b: f64) {
        let (pf, ef) = two_prod_fma(a, b);
        let (pd, ed) = two_prod_dekker(a, b);
        assert_eq!(pf.to_bits(), pd.to_bits(), "p {a:e}·{b:e}");
        // Le chemin de production (en ligne / appel `fma_hw` / Dekker).
        let (pp, ep) = two_prod(a, b);
        assert_eq!(pf.to_bits(), pp.to_bits(), "p {a:e}·{b:e}");
        assert!(ep.to_bits() == ef.to_bits() || (ep.is_nan() && ef.is_nan()), "{a:e}·{b:e}");
        if ef.is_nan() {
            assert!(ed.is_nan(), "{a:e}·{b:e}");
        } else {
            assert_eq!(ef.to_bits(), ed.to_bits(), "e {a:e}·{b:e} fma={ef:e} dekker={ed:e}");
        }
    }

    /// Dekker gardé == FMA au bit près : valeurs aléatoires sur TOUTE la
    /// plage d'exposants (sous-normaux et débordements compris), zéros
    /// signés, infinis, NaN, et les bords exacts du domaine.
    #[test]
    fn dekker_two_prod_matches_fma_bit_for_bit() {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        // Bits aléatoires : tous les exposants (y compris sous-normaux).
        for _ in 0..400_000 {
            let a = f64::from_bits(xorshift(&mut s));
            let b = f64::from_bits(xorshift(&mut s));
            same(a, b);
        }
        // Magnitudes usuelles des boucles dd/qd : produits ~2^-300..2^20.
        for _ in 0..400_000 {
            let r = xorshift(&mut s);
            let m = (r >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
            let a = m * 2f64.powi((r % 300) as i32 - 280);
            let r = xorshift(&mut s);
            let m = (r >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
            let b = m * 2f64.powi((r % 40) as i32 - 20);
            same(a, b);
        }
        // Bords : autour des seuils, zéros signés, spéciaux.
        let edges = [
            0.0,
            -0.0,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            5e-324,
            f64::MAX,
            -f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            1.0,
            -1.0,
            1.0 + f64::EPSILON,
            DEKKER_OP_MAX,
            DEKKER_OP_MAX * (1.0 + f64::EPSILON),
            2f64.powi(996),
            2f64.powi(-484),
            2f64.powi(-483) * (1.0 - f64::EPSILON),
            2f64.powi(510),
            2f64.powi(-537) * 1.6180339887,
            3.0f64.sqrt() * 2f64.powi(-500),
        ];
        for &a in &edges {
            for &b in &edges {
                same(a, b);
                same(a, -b);
                same(a * 1.3, b * 0.7);
            }
        }
    }
}
