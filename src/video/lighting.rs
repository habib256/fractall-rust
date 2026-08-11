//! Éclairage normal-map « spatial images » (G12 jalon 5, façon DeepDrill).
//!
//! Étage POST-colorisation, composable avec toutes les palettes : un champ de
//! hauteur est dérivé du compte d'itération lissé (smooth iteration, échelle
//! log), la normale vient des différences finies écran, et un Lambert
//! (ambient + diffus) module le RGB colorisé.
//!
//! **Écart assumé vs le plan initial** : DeepDrill stocke une normale
//! ANALYTIQUE (z/dz) calculée par son driller. Chez nous, exporter dz aurait
//! exigé de toucher toutes les boucles pixel du moteur (f64/exp/dd/GMP/
//! multi-phase) — exclu par l'invariant G12 « le moteur ne change pas ». La
//! normale écran (différences finies du champ lissé) donne le même effet
//! visuel pour un coût nul en amont ; c'est aussi la technique « slope » de
//! XaoS. Si un jour les boucles exportent dz, cet étage consommera le canal
//! sans changer d'API.
//!
//! Verrou jalon 5 : l'étage n'est PAS invoqué quand `lighting.enable = false`
//! (le chemin colorisation reste bit-identique à l'existant).

use num_complex::Complex64;

/// Part de lumière ambiante (0 = relief brut, 1 = pas d'effet).
const AMBIENT: f64 = 0.35;
/// Gain des pentes du champ de hauteur (échelle log → sans dimension).
const SLOPE_STRENGTH: f64 = 6.0;

/// Hauteur d'un pixel : smooth iteration en échelle log (stable à toute
/// profondeur de zoom, indépendante d'iteration_max en relatif).
fn height(iter: u32, z: Complex64, iter_max: u32) -> f64 {
    if iter >= iter_max {
        // Intérieur : plateau au-dessus de tout point échappé.
        return ((iter_max as f64) + 2.0).ln();
    }
    let norm = z.norm();
    let nu = if norm > 1.0 {
        // ν = n + 1 − ln(ln|z|)/ln 2 (même forme que le smooth coloring).
        let v = iter as f64 + 1.0 - (norm.ln().ln() / std::f64::consts::LN_2);
        if v.is_finite() { v.max(0.0) } else { iter as f64 }
    } else {
        iter as f64
    };
    (nu + 1.0).ln()
}

/// Applique l'éclairage Lambert au buffer RGB colorisé (in-place).
///
/// * `alpha_deg` — azimut de la lumière (degrés, 0 = est, CCW, repère visuel
///   y vers le haut) ;
/// * `beta_deg`  — inclinaison (90 = zénith → aucun relief visible, plus bas
///   = relief plus marqué).
pub fn shade_rgb(
    rgb: &mut [u8],
    iterations: &[u32],
    zs: &[Complex64],
    width: usize,
    height_px: usize,
    iter_max: u32,
    alpha_deg: f64,
    beta_deg: f64,
) {
    assert_eq!(rgb.len(), width * height_px * 3);
    assert_eq!(iterations.len(), width * height_px);
    assert_eq!(zs.len(), width * height_px);
    if width < 2 || height_px < 2 {
        return;
    }

    // Champ de hauteur.
    let h: Vec<f64> = iterations
        .iter()
        .zip(zs.iter())
        .map(|(&it, &z)| height(it, z, iter_max))
        .collect();

    let (sin_a, cos_a) = alpha_deg.to_radians().sin_cos();
    let (sin_b, cos_b) = beta_deg.to_radians().sin_cos();
    // Direction de la lumière (unitaire), repère visuel y vers le HAUT.
    let (lx, ly, lz) = (cos_a * cos_b, sin_a * cos_b, sin_b);

    use rayon::prelude::*;
    rgb.par_chunks_mut(width * 3)
        .enumerate()
        .for_each(|(j, row)| {
            for i in 0..width {
                // Différences centrées, bords clampés.
                let xl = h[j * width + i.saturating_sub(1)];
                let xr = h[j * width + (i + 1).min(width - 1)];
                let yu = h[j.saturating_sub(1) * width + i];
                let yd = h[(j + 1).min(height_px - 1) * width + i];
                let gx = (xr - xl) * 0.5 * SLOPE_STRENGTH;
                // Les lignes croissent vers le BAS de l'image : la pente
                // visuelle +y (vers le haut) est yu − yd.
                let gy = (yu - yd) * 0.5 * SLOPE_STRENGTH;
                // Normale de la surface z = h(x, y) : (−∂h/∂x, −∂h/∂y, 1).
                let inv_len = 1.0 / (gx * gx + gy * gy + 1.0).sqrt();
                let (nx, ny, nz) = (-gx * inv_len, -gy * inv_len, inv_len);

                let lambert = (nx * lx + ny * ly + nz * lz).max(0.0);
                let shade = AMBIENT + (1.0 - AMBIENT) * lambert;
                let o = i * 3;
                for c in 0..3 {
                    row[o + c] = (row[o + c] as f64 * shade).round().clamp(0.0, 255.0) as u8;
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Champ PLAT (itérations uniformes) → normale (0,0,1) partout → ombrage
    /// UNIFORME = ambient + (1−ambient)·sin β, identique sur chaque pixel.
    #[test]
    fn flat_field_shades_uniformly() {
        let (w, hh) = (8usize, 6usize);
        let iterations = vec![42u32; w * hh];
        let zs = vec![Complex64::new(3.0, 1.0); w * hh];
        let mut rgb = vec![200u8; w * hh * 3];
        shade_rgb(&mut rgb, &iterations, &zs, w, hh, 1000, 45.0, 30.0);

        let expected_shade = AMBIENT + (1.0 - AMBIENT) * 30f64.to_radians().sin();
        let expected = (200.0 * expected_shade).round() as u8;
        assert!(rgb.iter().all(|&v| v == expected), "ombrage uniforme attendu");
        assert!(expected < 200, "β=30° doit assombrir un champ plat");
    }

    /// β = 90° (zénith) : un champ plat n'est PAS modifié (lambert = 1,
    /// shade = 1). Le relief n'apparaît que sur les pentes.
    #[test]
    fn zenith_light_leaves_flat_field_untouched() {
        let (w, hh) = (6usize, 4usize);
        let iterations = vec![10u32; w * hh];
        let zs = vec![Complex64::new(5.0, 0.0); w * hh];
        let mut rgb: Vec<u8> = (0..w * hh * 3).map(|i| (i % 251) as u8).collect();
        let before = rgb.clone();
        shade_rgb(&mut rgb, &iterations, &zs, w, hh, 100, 0.0, 90.0);
        assert_eq!(rgb, before, "zénith + champ plat = identité");
    }

    /// Une pente éclairée de face est plus claire que la même pente éclairée
    /// à revers (le relief oriente bien la lumière).
    #[test]
    fn slope_orientation_changes_brightness() {
        let (w, hh) = (16usize, 8usize);
        // Gradient d'itérations croissant vers l'est → pente montante à l'est.
        let iterations: Vec<u32> = (0..w * hh).map(|idx| (idx % w) as u32 * 3).collect();
        let zs = vec![Complex64::new(4.0, 0.0); w * hh];
        let mut lit_east = vec![128u8; w * hh * 3];
        let mut lit_west = vec![128u8; w * hh * 3];
        shade_rgb(&mut lit_east, &iterations, &zs, w, hh, 1000, 180.0, 30.0);
        shade_rgb(&mut lit_west, &iterations, &zs, w, hh, 1000, 0.0, 30.0);
        // Au centre (loin des bords clampés) : lumière face à la pente
        // (azimut 180° = ouest → frappe la montée vers l'est) > à revers.
        let mid = (hh / 2) * w + w / 2;
        assert!(
            lit_east[mid * 3] > lit_west[mid * 3],
            "pente face à la lumière doit être plus claire ({} vs {})",
            lit_east[mid * 3],
            lit_west[mid * 3]
        );
    }
}
