//! Glitch detection legacy (Pauldelbrot tolerance + clustering + secondary
//! references).
//!
//! **Statut depuis P3.1 Session E** : voie historique conservée pour le
//! path GMP deep zoom et le path dual-numbers (distance estimation,
//! interior detection). Pour le path f64 standard, remplacée par le
//! pixel loop unifié bytecode (`fractal/bytecode/pixel_loop.rs`) qui
//! utilise rebasing F3 proactif au lieu de glitch detection corrective.
//!
//! Depuis P3.1 cleanup final : le champ `use_legacy_glitch_detection`
//! a été supprimé. Ces fonctions sont toujours appelées par le path
//! legacy (fallback bytecode) — à retirer complètement quand le path
//! bytecode supportera les dual numbers (distance/interior/orbit_traps).

use crate::fractal::FractalParams;

/// Représente un cluster de pixels glitchés.
/// Contient le centre du cluster et la liste des indices de pixels.
#[derive(Clone, Debug)]
pub struct GlitchCluster {
    /// Centre X du cluster dans l'espace complexe
    pub center_x: f64,
    /// Centre Y du cluster dans l'espace complexe
    pub center_y: f64,
    /// Indices des pixels dans le cluster (row-major order)
    pub pixel_indices: Vec<usize>,
}

impl GlitchCluster {
    pub fn new() -> Self {
        Self {
            center_x: 0.0,
            center_y: 0.0,
            pixel_indices: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.pixel_indices.len()
    }
}

/// Marque les pixels dont le compte d'itérations s'écarte fortement d'un de
/// leurs quatre voisins directs.
pub fn mark_neighbor_glitches(
    iterations: &[u32],
    width: u32,
    height: u32,
    threshold: u32,
) -> Vec<bool> {
    let size = (width * height) as usize;
    let mut mask = vec![false; size];
    if width < 3 || height < 3 || iterations.len() != size {
        return mask;
    }

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let idx = (y * width + x) as usize;
            let center = iterations[idx];
            let left = iterations[(y * width + (x - 1)) as usize];
            let right = iterations[(y * width + (x + 1)) as usize];
            let up = iterations[((y - 1) * width + x) as usize];
            let down = iterations[((y + 1) * width + x) as usize];
            let mut max_diff = center.abs_diff(left);
            max_diff = max_diff.max(center.abs_diff(right));
            max_diff = max_diff.max(center.abs_diff(up));
            max_diff = max_diff.max(center.abs_diff(down));
            if max_diff > threshold {
                mask[idx] = true;
            }
        }
    }
    mask
}

/// Détecte les clusters de glitchs par composantes connexes.
///
/// Utilise un algorithme de flood fill pour regrouper les pixels glitchés
/// adjacents en clusters. Seuls les clusters de taille >= min_cluster_size
/// sont retournés.
///
/// # Arguments
/// * `glitch_mask` - Masque booléen des pixels glitchés
/// * `width` - Largeur de l'image
/// * `height` - Hauteur de l'image
/// * `params` - Paramètres du fractal (pour calculer les coordonnées du centre)
/// * `min_cluster_size` - Taille minimale d'un cluster pour être retourné
///
/// # Returns
/// Liste de clusters triés par taille décroissante
pub fn detect_glitch_clusters(
    glitch_mask: &[bool],
    width: u32,
    height: u32,
    params: &FractalParams,
    min_cluster_size: usize,
) -> Vec<GlitchCluster> {
    if glitch_mask.len() != (width * height) as usize {
        return Vec::new();
    }

    let w = width as usize;
    let h = height as usize;
    let mut visited = vec![false; w * h];
    let mut clusters = Vec::new();

    // Parcourir tous les pixels
    for start_idx in 0..(w * h) {
        if !glitch_mask[start_idx] || visited[start_idx] {
            continue;
        }

        // Nouveau cluster trouvé - flood fill
        let mut cluster = GlitchCluster::new();
        let mut stack = vec![start_idx];

        while let Some(idx) = stack.pop() {
            if visited[idx] || !glitch_mask[idx] {
                continue;
            }
            visited[idx] = true;
            cluster.pixel_indices.push(idx);

            let x = idx % w;
            let y = idx / w;

            // Voisins 4-connexes
            if x > 0 {
                let left = idx - 1;
                if !visited[left] && glitch_mask[left] {
                    stack.push(left);
                }
            }
            if x + 1 < w {
                let right = idx + 1;
                if !visited[right] && glitch_mask[right] {
                    stack.push(right);
                }
            }
            if y > 0 {
                let up = idx - w;
                if !visited[up] && glitch_mask[up] {
                    stack.push(up);
                }
            }
            if y + 1 < h {
                let down = idx + w;
                if !visited[down] && glitch_mask[down] {
                    stack.push(down);
                }
            }
        }

        // Calculer le centre du cluster
        if cluster.len() >= min_cluster_size {
            let (sum_x, sum_y) =
                cluster
                    .pixel_indices
                    .iter()
                    .fold((0usize, 0usize), |(sx, sy), &idx| {
                        let x = idx % w;
                        let y = idx / w;
                        (sx + x, sy + y)
                    });

            let avg_x = sum_x as f64 / cluster.len() as f64;
            let avg_y = sum_y as f64 / cluster.len() as f64;

            // Convertir les coordonnées pixel en coordonnées complexes (pixel center = (idx+0.5)/size)
            cluster.center_x = params.center_x + ((avg_x + 0.5) / w as f64 - 0.5) * params.span_x;
            cluster.center_y = params.center_y + ((avg_y + 0.5) / h as f64 - 0.5) * params.span_y;

            clusters.push(cluster);
        }
    }

    // Trier par taille décroissante
    clusters.sort_by(|a, b| b.len().cmp(&a.len()));

    clusters
}

/// Sélectionne les N plus grands clusters pour les références secondaires.
///
/// # Arguments
/// * `clusters` - Liste de clusters (déjà triés par taille)
/// * `max_refs` - Nombre maximum de références secondaires
///
/// # Returns
/// Sous-ensemble des clusters sélectionnés
pub fn select_secondary_reference_points(
    clusters: &[GlitchCluster],
    max_refs: usize,
) -> Vec<&GlitchCluster> {
    clusters.iter().take(max_refs).collect()
}

/// Regroupe les pixels glitchés par profondeur d'itération et choisit, pour
/// chaque groupe, le pixel de plus petit |z| comme centre de référence.
pub fn segregate_glitches_by_iteration(
    glitch_mask: &[bool],
    iterations: &[u32],
    z_finals: &[num_complex::Complex64],
    width: u32,
    height: u32,
    params: &FractalParams,
    min_group_size: usize,
) -> Vec<GlitchCluster> {
    let w = width as usize;
    let total = w * height as usize;
    if glitch_mask.len() != total || iterations.len() != total || z_finals.len() != total {
        return Vec::new();
    }
    let max_iter = glitch_mask
        .iter()
        .enumerate()
        .filter(|(_, g)| **g)
        .map(|(idx, _)| iterations[idx])
        .max()
        .unwrap_or(0) as usize;
    let mut groups = vec![Vec::new(); max_iter + 1];
    for idx in 0..total {
        if glitch_mask[idx] {
            groups[iterations[idx] as usize].push(idx);
        }
    }
    let mut clusters: Vec<_> = groups
        .into_iter()
        .filter(|pixels| pixels.len() >= min_group_size)
        .map(|pixels| {
            let best_idx = pixels
                .iter()
                .copied()
                .min_by(|&a, &b| {
                    z_finals[a]
                        .norm_sqr()
                        .partial_cmp(&z_finals[b].norm_sqr())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(pixels[0]);
            let bx = best_idx % w;
            let by = best_idx / w;
            GlitchCluster {
                center_x: params.center_x + ((bx as f64 + 0.5) / w as f64 - 0.5) * params.span_x,
                center_y: params.center_y
                    + ((by as f64 + 0.5) / height as f64 - 0.5) * params.span_y,
                pixel_indices: pixels,
            }
        })
        .collect();
    clusters.sort_by(|a, b| b.len().cmp(&a.len()));
    clusters
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::definitions::default_params_for_type;
    use crate::fractal::{AlgorithmMode, FractalType};
    use num_complex::Complex64;

    fn test_params() -> FractalParams {
        let mut p = default_params_for_type(FractalType::Mandelbrot, 10, 10);
        p.span_x = 4.0;
        p.span_y = 4.0;
        p.iteration_max = 100;
        p.precision_bits = 192;
        p.algorithm_mode = AlgorithmMode::Perturbation;
        p.bla_threshold = 1e-6;
        p.glitch_neighbor_pass = false;
        p
    }

    #[test]
    fn test_detect_single_cluster() {
        let params = test_params();
        let mut mask = vec![false; 100];
        // Create a 3x3 cluster at center
        for y in 4..7 {
            for x in 4..7 {
                mask[y * 10 + x] = true;
            }
        }

        let clusters = detect_glitch_clusters(&mask, 10, 10, &params, 1);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 9);
    }

    #[test]
    fn test_detect_two_clusters() {
        let params = test_params();
        let mut mask = vec![false; 100];
        // Cluster 1: top-left 2x2
        mask[0] = true;
        mask[1] = true;
        mask[10] = true;
        mask[11] = true;
        // Cluster 2: bottom-right 2x2
        mask[88] = true;
        mask[89] = true;
        mask[98] = true;
        mask[99] = true;

        let clusters = detect_glitch_clusters(&mask, 10, 10, &params, 1);
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].len(), 4);
        assert_eq!(clusters[1].len(), 4);
    }

    #[test]
    fn test_min_cluster_size() {
        let params = test_params();
        let mut mask = vec![false; 100];
        // Small cluster (2 pixels)
        mask[0] = true;
        mask[1] = true;
        // Larger cluster (5 pixels)
        mask[50] = true;
        mask[51] = true;
        mask[52] = true;
        mask[60] = true;
        mask[61] = true;

        let clusters = detect_glitch_clusters(&mask, 10, 10, &params, 3);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 5);
    }

    #[test]
    fn test_select_secondary_refs() {
        let params = test_params();
        let mut mask = vec![false; 100];
        // Create 5 clusters of different sizes
        for i in 0..5 {
            for j in 0..(i + 1) {
                mask[i * 20 + j] = true;
            }
        }

        let clusters = detect_glitch_clusters(&mask, 10, 10, &params, 1);
        let selected = select_secondary_reference_points(&clusters, 3);
        assert_eq!(selected.len(), 3);
        // Should be sorted by size descending
        assert!(selected[0].len() >= selected[1].len());
        assert!(selected[1].len() >= selected[2].len());
    }

    #[test]
    fn segregates_by_iteration_and_chooses_smallest_norm_center() {
        let mut params = default_params_for_type(FractalType::Mandelbrot, 4, 2);
        params.center_x = 0.0;
        params.center_y = 0.0;
        params.span_x = 4.0;
        params.span_y = 2.0;

        let mut glitch_mask = vec![false; 8];
        glitch_mask[0] = true;
        glitch_mask[1] = true;
        glitch_mask[6] = true;
        let iterations = vec![7, 7, 0, 0, 0, 0, 9, 0];
        let mut z_finals = vec![Complex64::new(10.0, 0.0); 8];
        z_finals[0] = Complex64::new(3.0, 0.0);
        z_finals[1] = Complex64::new(1.0, 0.0);
        z_finals[6] = Complex64::new(0.5, 0.0);

        let clusters =
            segregate_glitches_by_iteration(&glitch_mask, &iterations, &z_finals, 4, 2, &params, 2);

        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].pixel_indices, vec![0, 1]);
        assert_eq!(clusters[0].center_x, -0.5);
        assert_eq!(clusters[0].center_y, -0.5);
    }

    #[test]
    fn segregate_by_iteration_rejects_misaligned_buffers() {
        let params = default_params_for_type(FractalType::Mandelbrot, 2, 2);

        assert!(segregate_glitches_by_iteration(
            &[true, false],
            &[1],
            &[Complex64::new(0.0, 0.0); 2],
            2,
            1,
            &params,
            1,
        )
        .is_empty());
    }
}
