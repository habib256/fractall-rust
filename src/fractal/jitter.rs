//! Anti-aliasing par sous-échantillonnage jitteré (« per-frame »).
//!
//! Port des primitives low-discrepancy de Fraktaler-3 (`hybrid.h`) :
//! `radical_inverse` (suite de Halton / van der Corput) + `triangle`
//! (distribution triangulaire = filtre tente). Pour l'AA multi-sample
//! per-frame, chaque sample `k` décale TOUTE la grille de pixels d'un offset
//! sous-pixel `sample_offset(k)` (constant sur le frame), et les rendus
//! colorés sont moyennés.
//!
//! F3 ajoute en plus une rotation Cranley-Patterson par pixel (`burtle_hash`,
//! `hybrid.h:62`) pour décorréler spatialement le motif d'échantillonnage ;
//! l'approche per-frame s'en passe (offset uniforme par frame) — la qualité
//! AA est équivalente dès N ≥ 4 samples car la moyenne reste un filtre tente.

/// Inverse radical (van der Corput) de `a` en base `base`, dans `[0, 1)`.
/// Port de `radical_inverse` (F3 `hybrid.h:27`).
#[allow(dead_code)]
pub fn radical_inverse(mut a: u64, base: u64) -> f64 {
    const ONE_MINUS_EPSILON: f64 = 0.999_999_999_999_999_89;
    let base1 = 1.0 / base as f64;
    let mut reversed: u64 = 0;
    let mut base1n = 1.0;
    while a != 0 {
        let next = a / base;
        let digit = a - base * next;
        reversed = reversed * base + digit;
        base1n *= base1;
        a = next;
    }
    (reversed as f64 * base1n).min(ONE_MINUS_EPSILON)
}

/// Transforme une uniforme `[0, 1]` en distribution triangulaire `[-1, 1]`
/// (filtre tente, pic en 0). Port de `triangle` (F3 `hybrid.h:53`).
///
/// **Écart volontaire avec F3** : à `v = 0.5` (`orig = 0`), F3 génère un NaN
/// (`0/sqrt(0)`) qu'il « nerfe » en `-1` puis retranche `sign(0)=+1` → `-2`
/// (hors plage). F3 ne tombe jamais sur ce point car son hash per-pixel décale
/// l'entrée ; en per-frame `radical_inverse(1, 2) = 0.5` EXACTEMENT, donc on
/// renvoie explicitement `0.0` (le centre correct de la tente).
#[allow(dead_code)]
pub fn triangle(v: f64) -> f64 {
    let orig = v * 2.0 - 1.0;
    if orig == 0.0 {
        return 0.0;
    }
    let t = (orig / orig.abs().sqrt()).max(-1.0);
    t - if orig >= 0.0 { 1.0 } else { -1.0 }
}

/// Offset sous-pixel du sample `k` (en unités de pixel), composantes dans
/// `[-1, 1]` avec distribution triangulaire. `k = 0` renvoie `(0, 0)` (centre
/// du pixel), les samples suivants se répartissent via Halton bases 2 et 3.
#[allow(dead_code)]
pub fn sample_offset(k: u64) -> (f64, f64) {
    (
        triangle(radical_inverse(k, 2)),
        triangle(radical_inverse(k, 3)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radical_inverse_basic() {
        assert_eq!(radical_inverse(0, 2), 0.0);
        assert_eq!(radical_inverse(1, 2), 0.5);
        assert_eq!(radical_inverse(2, 2), 0.25);
        assert_eq!(radical_inverse(3, 2), 0.75);
        assert!((radical_inverse(1, 3) - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn triangle_in_range_and_centered() {
        // Endpoints et centre.
        assert_eq!(triangle(0.0), 0.0);
        assert_eq!(triangle(1.0), 0.0);
        assert_eq!(triangle(0.5), 0.0); // pas de -2 (cf. doc)
        // Toujours dans [-1, 1] sur un échantillonnage dense.
        for i in 0..=1000 {
            let v = i as f64 / 1000.0;
            let t = triangle(v);
            assert!(t.is_finite() && (-1.0..=1.0).contains(&t), "triangle({v}) = {t}");
        }
    }

    #[test]
    fn sample_offset_zero_is_center() {
        assert_eq!(sample_offset(0), (0.0, 0.0));
        // Les samples > 0 restent dans la cellule [-1, 1]².
        for k in 1..64 {
            let (x, y) = sample_offset(k);
            assert!((-1.0..=1.0).contains(&x) && (-1.0..=1.0).contains(&y));
        }
    }

    #[test]
    fn sample_mean_approaches_center() {
        // La moyenne d'un grand nombre de samples doit tendre vers 0
        // (filtre tente symétrique).
        let n = 4096u64;
        let (mut sx, mut sy) = (0.0, 0.0);
        for k in 0..n {
            let (x, y) = sample_offset(k);
            sx += x;
            sy += y;
        }
        assert!((sx / n as f64).abs() < 0.05, "mean x = {}", sx / n as f64);
        assert!((sy / n as f64).abs() < 0.05, "mean y = {}", sy / n as f64);
    }
}
