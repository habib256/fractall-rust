//! Anti-aliasing par sous-échantillonnage jitteré (« per-frame »).
//!
//! Port des primitives low-discrepancy de Fraktaler-3 (`hybrid.h`) :
//! `radical_inverse` (suite de Halton / van der Corput) + `triangle`
//! (distribution triangulaire = filtre tente). Pour l'AA multi-sample
//! per-frame, chaque sample `k` décale TOUTE la grille de pixels d'un offset
//! sous-pixel `sample_offset(k)` (constant sur le frame), et les rendus
//! colorés sont moyennés.
//!
//! F3 ajoute en plus une **rotation Cranley-Patterson par pixel** (`burtle_hash`,
//! `hybrid.h:62`) pour décorréler spatialement le motif d'échantillonnage. À
//! **bas N** (2-4 samples) l'offset uniforme per-frame fait échantillonner
//! TOUS les pixels aux MÊMES sous-positions → aliasing corrélé (motif visible,
//! pas un vrai dithering). `pixel_offset` porte la variante F3 fidèle : chaque
//! pixel reçoit une rotation `h = burtle_hash(j·w+i)/2³²` ajoutée à la valeur
//! de Halton avant le filtre tente, si bien que les pixels voisins tirent des
//! sous-positions décorrélées (dithering) — et le point dégénéré `v=0.5` du
//! per-frame (cf. `triangle`) n'est jamais atteint (la rotation le décale).

/// Hash entier de Bob Jenkins (« burtle »), port bit-exact de F3
/// (`hybrid.h:16`, <http://www.burtleburtle.net/bob/hash/integer.html>).
/// Sert à dériver une rotation Cranley-Patterson par pixel pour décorréler
/// spatialement l'échantillonnage AA. Arithmétique `u32` en wrapping (comme le
/// C++ non signé).
#[allow(dead_code)]
pub fn burtle_hash(mut a: u32) -> u32 {
    a = a.wrapping_add(0x7ed5_5d16).wrapping_add(a << 12);
    a = (a ^ 0xc761_c23c) ^ (a >> 19);
    a = a.wrapping_add(0x1656_67b1).wrapping_add(a << 5);
    a = a.wrapping_add(0xd3a2_646c) ^ (a << 9);
    a = a.wrapping_add(0xfd70_46c5).wrapping_add(a << 3);
    a = (a ^ 0xb55a_4f09) ^ (a >> 16);
    a
}

/// Partie fractionnaire `v - floor(v)`, dans `[0, 1)`. Port de `wrap`
/// (F3 `hybrid.h:44`).
#[allow(dead_code)]
pub fn wrap(v: f64) -> f64 {
    v - v.floor()
}

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

/// Offset sous-pixel **par pixel** du sample `k`, en unités de pixel déjà
/// multipliées par `scale`. Port fidèle de F3 `jitter` (`hybrid.h:62`, frame 0):
/// une rotation Cranley-Patterson `h = burtle_hash(j·width + i)/2³²` décale la
/// séquence de Halton du pixel `(i, j)`, si bien que deux pixels voisins tirent
/// des sous-positions décorrélées (dithering spatial) au lieu de partager le
/// même motif — ce qui supprime l'aliasing corrélé du schéma per-frame à bas N.
///
/// À `k = 0` l'offset n'est PAS `(0, 0)` (contrairement à `sample_offset`) :
/// `triangle(wrap(0 + h)) = triangle(h)`, chaque pixel étant déjà jitteré —
/// exactement le comportement F3. La rotation garantit aussi que `v = 0.5`
/// (point NaN de `triangle`) n'est jamais atteint.
#[allow(dead_code)]
pub fn pixel_offset(width: usize, i: usize, j: usize, k: u64, scale: f64) -> (f64, f64) {
    // ix = (frame·height + j)·width + i, frame = 0 (F3 hybrid.h:64). Wrapping
    // vers u32 comme le C++ (le hash opère sur 32 bits).
    let ix = (j as u64).wrapping_mul(width as u64).wrapping_add(i as u64) as u32;
    let h = burtle_hash(ix) as f64 / 4_294_967_296.0; // / 2³²
    (
        triangle(wrap(radical_inverse(k, 2) + h)) * scale,
        triangle(wrap(radical_inverse(k, 3) + h)) * scale,
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
    fn burtle_hash_matches_f3_reference() {
        // Valeurs de référence calculées avec le C++ F3 (hybrid.h:16), u32
        // wrapping. Verrouille le port bit-exact.
        assert_eq!(burtle_hash(0), 0x6b4e_d927);
        assert_eq!(burtle_hash(1), 0xb486_81b6);
        // Déterminisme + dispersion : deux entrées voisines → hauts très
        // différents (avalanche).
        assert_ne!(burtle_hash(100), burtle_hash(101));
    }

    #[test]
    fn wrap_in_unit_interval() {
        assert_eq!(wrap(0.0), 0.0);
        assert!((wrap(1.25) - 0.25).abs() < 1e-15);
        assert!((wrap(-0.25) - 0.75).abs() < 1e-15);
        for i in -1000..=1000 {
            let v = i as f64 * 0.013;
            let w = wrap(v);
            assert!((0.0..1.0).contains(&w), "wrap({v}) = {w}");
        }
    }

    #[test]
    fn pixel_offset_in_cell_and_spatially_decorrelated() {
        let w = 64usize;
        // Reste dans la cellule sous-pixel [-scale, scale]² pour tout pixel/sample.
        let scale = 1.0;
        for k in 0..8u64 {
            for j in 0..8 {
                for i in 0..8 {
                    let (x, y) = pixel_offset(w, i, j, k, scale);
                    assert!(x.is_finite() && y.is_finite());
                    assert!((-scale..=scale).contains(&x) && (-scale..=scale).contains(&y));
                }
            }
        }
        // Décorrélation spatiale : au MÊME sample k, des pixels voisins tirent
        // des offsets DIFFÉRENTS (le per-frame les rendrait identiques). C'est
        // exactement ce qui casse l'aliasing corrélé à bas N.
        let a = pixel_offset(w, 10, 10, 0, 1.0);
        let b = pixel_offset(w, 11, 10, 0, 1.0);
        let c = pixel_offset(w, 10, 11, 0, 1.0);
        assert_ne!(a, b, "pixels voisins horizontaux non décorrélés");
        assert_ne!(a, c, "pixels voisins verticaux non décorrélés");
        // `scale` est un facteur linéaire pur.
        let (x1, y1) = pixel_offset(w, 3, 7, 2, 1.0);
        let (x2, y2) = pixel_offset(w, 3, 7, 2, 0.5);
        assert!((x2 - x1 * 0.5).abs() < 1e-15 && (y2 - y1 * 0.5).abs() < 1e-15);
    }

    #[test]
    fn pixel_offset_mean_is_tent_centered() {
        // Sur beaucoup de samples ET beaucoup de pixels, la moyenne des offsets
        // reste centrée (filtre tente symétrique décalé par des rotations
        // uniformes) → pas de biais directionnel introduit par le hash.
        let w = 128usize;
        let (mut sx, mut sy, mut n) = (0.0f64, 0.0f64, 0.0f64);
        for k in 0..16u64 {
            for j in (0..64).step_by(3) {
                for i in (0..64).step_by(3) {
                    let (x, y) = pixel_offset(w, i, j, k, 1.0);
                    sx += x;
                    sy += y;
                    n += 1.0;
                }
            }
        }
        assert!((sx / n).abs() < 0.02, "biais x = {}", sx / n);
        assert!((sy / n).abs() < 0.02, "biais y = {}", sy / n);
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
