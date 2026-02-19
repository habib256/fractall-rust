//! Infrastructure pour le rendu progressif multi-résolution.
//!
//! Ce module fournit les types et fonctions pour afficher la fractale
//! progressivement: d'abord une vue basse résolution, puis des passes
//! de plus en plus détaillées.

use std::sync::Arc;
use num_complex::Complex64;

use crate::fractal::perturbation::ReferenceOrbitCache;
use crate::fractal::orbit_traps::OrbitData;

/// Message envoyé du thread de rendu vers le GUI.
pub enum RenderMessage {
    /// Une passe de rendu est terminée.
    PassComplete {
        pass_index: u8,
        scale_divisor: u8,
        effective_mode: crate::fractal::AlgorithmMode,
        precision_label: String,
        iterations: Vec<u32>,
        zs: Vec<Complex64>,
        distances: Vec<f64>,
        /// Données d'orbite pour Orbit Traps / Wings (vide si non calculé)
        orbits: Vec<Option<OrbitData>>,
        width: u32,
        height: u32,
        /// Buffer RGBA pré-colorisé (évite de bloquer le thread UI)
        #[allow(dead_code)]
        colored_buffer: Vec<u8>,
    },
    /// Toutes les passes sont terminées.
    AllComplete {
        /// Updated orbit cache for reuse in subsequent renders.
        orbit_cache: Option<Arc<ReferenceOrbitCache>>,
    },
    /// Le rendu a été annulé.
    Cancelled,
}

/// Configuration du rendu progressif.
#[derive(Clone, Debug)]
pub struct ProgressiveConfig {
    /// Diviseurs de résolution pour chaque passe (ex: [8, 2, 1]).
    pub passes: Vec<u8>,
}

impl ProgressiveConfig {
    /// Configuration standard: 5 passes (1/16 → 1/8 → 1/4 → 1/2 → pleine résolution).
    /// Progression très progressive pour un feedback visuel fluide.
    pub fn standard() -> Self {
        Self { passes: vec![16, 8, 4, 2, 1] }
    }

    /// Configuration standard avec moins de passes (1/16, 1/4, pleine résolution).
    pub fn standard_basic() -> Self {
        Self { passes: vec![16, 4, 1] }
    }

    /// Configuration pour GMP (3 passes: 1/16, 1/8, pleine — déjà lent).
    pub fn gmp_mode() -> Self {
        Self { passes: vec![16, 8, 1] }
    }

    /// Configuration pour perturbation (3 passes: 1/16, 1/4, pleine).
    /// La perturbation a un overhead par passe (détection glitchs, références secondaires),
    /// donc moins de passes réduit le coût total tout en offrant un aperçu progressif.
    pub fn perturbation_mode() -> Self {
        Self { passes: vec![16, 4, 1] }
    }

    /// Configuration pour petites images (<256px): 4 passes progressives.
    pub fn fast() -> Self {
        Self { passes: vec![8, 4, 2, 1] }
    }

    /// Configuration rapide sans passe intermédiaire.
    pub fn fast_basic() -> Self {
        Self { passes: vec![8, 1] }
    }

    /// Configuration avec une seule passe (pas de progressif).
    pub fn single_pass() -> Self {
        Self { passes: vec![1] }
    }

    /// Choisit la configuration appropriée selon les paramètres.
    #[allow(dead_code)]
    pub fn for_params(width: u32, height: u32, use_gmp: bool) -> Self {
        Self::for_params_with_intermediate(width, height, use_gmp, true, false)
    }

    /// Choisit la configuration en optionnant la passe intermédiaire.
    pub fn for_params_with_intermediate(
        width: u32,
        height: u32,
        use_gmp: bool,
        allow_intermediate: bool,
        use_perturbation: bool,
    ) -> Self {
        if width < 64 || height < 64 {
            // Image trop petite pour le progressif
            Self::single_pass()
        } else if use_gmp {
            // GMP est déjà lent, moins de passes
            Self::gmp_mode()
        } else if use_perturbation && allow_intermediate {
            // Perturbation: overhead par passe (glitchs, refs secondaires), moins de passes
            Self::perturbation_mode()
        } else if width < 256 || height < 256 {
            // Petite image
            if allow_intermediate {
                Self::fast()
            } else {
                Self::fast_basic()
            }
        } else {
            // Configuration standard
            if allow_intermediate {
                Self::standard()
            } else {
                Self::standard_basic()
            }
        }
    }
}

/// Upscale les données d'itération et z en utilisant l'interpolation nearest-neighbor.
///
/// Cette méthode est rapide et donne un effet "pixelisé" qui indique
/// clairement que c'est une preview.
pub fn upscale_nearest(
    src_iterations: &[u32],
    src_zs: &[Complex64],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> (Vec<u32>, Vec<Complex64>) {
    let dst_size = (dst_width * dst_height) as usize;
    let mut dst_iterations = vec![0u32; dst_size];
    let mut dst_zs = vec![Complex64::new(0.0, 0.0); dst_size];

    // Ratios de scaling
    let x_ratio = src_width as f64 / dst_width as f64;
    let y_ratio = src_height as f64 / dst_height as f64;

    for dst_y in 0..dst_height {
        let src_y = ((dst_y as f64 * y_ratio) as u32).min(src_height.saturating_sub(1));
        for dst_x in 0..dst_width {
            let src_x = ((dst_x as f64 * x_ratio) as u32).min(src_width.saturating_sub(1));
            let src_idx = (src_y * src_width + src_x) as usize;
            let dst_idx = (dst_y * dst_width + dst_x) as usize;

            if src_idx < src_iterations.len() {
                dst_iterations[dst_idx] = src_iterations[src_idx];
                dst_zs[dst_idx] = src_zs[src_idx];
            }
        }
    }

    (dst_iterations, dst_zs)
}

/// Upscale un buffer RGB en utilisant l'interpolation nearest-neighbor.
#[allow(dead_code)]
pub fn upscale_rgb_nearest(
    src_buffer: &[u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Vec<u8> {
    let dst_size = (dst_width * dst_height * 3) as usize;
    let mut dst_buffer = vec![0u8; dst_size];

    let x_ratio = src_width as f64 / dst_width as f64;
    let y_ratio = src_height as f64 / dst_height as f64;

    for dst_y in 0..dst_height {
        let src_y = ((dst_y as f64 * y_ratio) as u32).min(src_height.saturating_sub(1));
        for dst_x in 0..dst_width {
            let src_x = ((dst_x as f64 * x_ratio) as u32).min(src_width.saturating_sub(1));
            let src_idx = ((src_y * src_width + src_x) * 3) as usize;
            let dst_idx = ((dst_y * dst_width + dst_x) * 3) as usize;

            if src_idx + 2 < src_buffer.len() && dst_idx + 2 < dst_buffer.len() {
                dst_buffer[dst_idx] = src_buffer[src_idx];
                dst_buffer[dst_idx + 1] = src_buffer[src_idx + 1];
                dst_buffer[dst_idx + 2] = src_buffer[src_idx + 2];
            }
        }
    }

    dst_buffer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upscale_nearest_2x() {
        // Source 2x2
        let src_iter = vec![1, 2, 3, 4];
        let src_zs = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];

        // Upscale to 4x4
        let (dst_iter, _dst_zs) = upscale_nearest(&src_iter, &src_zs, 2, 2, 4, 4);

        // Each source pixel should map to 2x2 destination pixels
        assert_eq!(dst_iter.len(), 16);
        // Top-left 2x2 should be 1
        assert_eq!(dst_iter[0], 1);
        assert_eq!(dst_iter[1], 1);
        assert_eq!(dst_iter[4], 1);
        assert_eq!(dst_iter[5], 1);
        // Top-right 2x2 should be 2
        assert_eq!(dst_iter[2], 2);
        assert_eq!(dst_iter[3], 2);
    }

    #[test]
    fn test_progressive_config_selection() {
        // Large image, no GMP -> standard (5 passes progressives)
        let config = ProgressiveConfig::for_params(1024, 768, false);
        assert_eq!(config.passes, vec![16, 8, 4, 2, 1]);

        // Large image, GMP -> gmp_mode (3 passes)
        let config = ProgressiveConfig::for_params(1024, 768, true);
        assert_eq!(config.passes, vec![16, 8, 1]);

        // Small image -> fast (4 passes)
        let config = ProgressiveConfig::for_params(200, 200, false);
        assert_eq!(config.passes, vec![8, 4, 2, 1]);

        // Very small image -> single pass
        let config = ProgressiveConfig::for_params(32, 32, false);
        assert_eq!(config.passes, vec![1]);
    }

    #[test]
    fn test_progressive_config_perturbation() {
        // Large image, perturbation -> perturbation_mode (3 passes)
        let config = ProgressiveConfig::for_params_with_intermediate(1024, 768, false, true, true);
        assert_eq!(config.passes, vec![16, 4, 1]);

        // Perturbation + GMP -> GMP prend priorité
        let config = ProgressiveConfig::for_params_with_intermediate(1024, 768, true, true, true);
        assert_eq!(config.passes, vec![16, 8, 1]);

        // Perturbation sans allow_intermediate -> standard_basic (pas de perturbation_mode)
        let config = ProgressiveConfig::for_params_with_intermediate(1024, 768, false, false, true);
        assert_eq!(config.passes, vec![16, 4, 1]);
    }
}
