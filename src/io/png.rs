use std::path::Path;

use image::{ImageError, RgbImage};
use num_complex::Complex64;
use rayon::prelude::*;

use crate::color::{color_for_pixel, color_for_nebulabrot_pixel, color_for_buddhabrot_pixel};
use crate::fractal::{FractalParams, FractalType};

/// Génére une image RGB colorisée à partir des matrices d'itérations et de z,
/// puis l'enregistre au format PNG.
///
/// La colorisation est également parallélisée pour améliorer les performances.
pub fn save_png(
    params: &FractalParams,
    iterations: &[u32],
    zs: &[Complex64],
    output: &Path,
) -> Result<(), ImageError> {
    let width = params.width;
    let height = params.height;
    let w = width as usize;
    let h = height as usize;

    assert_eq!(iterations.len(), w * h, "Taille de la matrice d'itérations invalide");
    assert_eq!(zs.len(), w * h, "Taille de la matrice des valeurs z invalide");

    let is_nebulabrot = params.fractal_type == FractalType::Nebulabrot;
    let is_buddhabrot = params.fractal_type == FractalType::Buddhabrot;

    // Parallélisation de la colorisation par lignes
    let buffer: Vec<u8> = (0..height as usize)
        .into_par_iter()
        .flat_map(|y| {
            (0..width)
                .flat_map(|x| {
                    let idx = y * w + x as usize;
                    let iter = iterations[idx];
                    let z = zs[idx];

                    let (r, g, b) = if is_nebulabrot {
                        color_for_nebulabrot_pixel(iter, z)
                    } else if is_buddhabrot {
                        color_for_buddhabrot_pixel(z, params.color_mode, params.color_repeat)
                    } else {
                        color_for_pixel(
                            iter,
                            z,
                            params.iteration_max,
                            params.color_mode,
                            params.color_repeat,
                            params.out_coloring_mode,
                            params.color_space,
                            None, // Orbit data not stored in PNG export yet
                        )
                    };

                    vec![r, g, b]
                })
                .collect::<Vec<u8>>()
        })
        .collect();

    // Créer l'image depuis le buffer
    let img = RgbImage::from_raw(width, height, buffer)
        .ok_or_else(|| ImageError::from(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Impossible de créer l'image depuis le buffer"
        )))?;

    // Avec image 0.25, save() détecte automatiquement le format depuis l'extension
    img.save(output)
}

