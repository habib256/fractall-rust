use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use num_complex::Complex64;
use png::{Decoder, Encoder};
use rayon::prelude::*;

use crate::color::{color_for_pixel, color_for_nebulabrot_pixel, color_for_buddhabrot_pixel};
use crate::fractal::{FractalParams, FractalType};

/// Clé du chunk tEXt pour les métadonnées fractales.
const METADATA_KEY: &str = "fractall-params";

/// Génère une image RGB colorisée avec métadonnées fractales intégrées dans un chunk tEXt.
///
/// Les métadonnées permettent de restaurer exactement l'état de la fractale
/// (coordonnées HP, type, paramètres) lors du chargement ultérieur de l'image.
pub fn save_png_with_metadata(
    params: &FractalParams,
    iterations: &[u32],
    zs: &[Complex64],
    output: &Path,
    center_x_hp: &str,
    center_y_hp: &str,
    span_x_hp: &str,
    span_y_hp: &str,
) -> Result<(), Box<dyn std::error::Error>> {
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
                            None,
                            None,
                            false,
                        )
                    };

                    vec![r, g, b]
                })
                .collect::<Vec<u8>>()
        })
        .collect();

    // Créer les params avec coordonnées HP complètes pour sérialisation
    let mut params_to_save = params.clone();
    params_to_save.center_x_hp = Some(center_x_hp.to_string());
    params_to_save.center_y_hp = Some(center_y_hp.to_string());
    params_to_save.span_x_hp = Some(span_x_hp.to_string());
    params_to_save.span_y_hp = Some(span_y_hp.to_string());

    // Sérialiser en JSON
    let metadata_json = serde_json::to_string(&params_to_save)?;

    // Écrire le PNG avec métadonnées via le crate png
    let file = File::create(output)?;
    let writer = BufWriter::new(file);

    let mut encoder = Encoder::new(writer, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);

    // Ajouter le chunk tEXt avec les métadonnées
    encoder.add_text_chunk(METADATA_KEY.to_string(), metadata_json)?;

    let mut png_writer = encoder.write_header()?;
    png_writer.write_image_data(&buffer)?;

    Ok(())
}

/// Charge les métadonnées fractales depuis un fichier PNG.
///
/// Retourne les FractalParams si le fichier contient les métadonnées fractall,
/// ou une erreur si le fichier n'est pas un PNG valide ou ne contient pas de métadonnées.
#[allow(dead_code)]
pub fn load_png_metadata(path: &Path) -> Result<FractalParams, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let decoder = Decoder::new(reader);
    let png_reader = decoder.read_info()?;
    let info = png_reader.info();

    // Chercher le chunk tEXt avec notre clé
    for text_chunk in &info.uncompressed_latin1_text {
        if text_chunk.keyword == METADATA_KEY {
            let params: FractalParams = serde_json::from_str(&text_chunk.text)?;
            return Ok(params);
        }
    }

    Err("Aucune métadonnée fractall trouvée dans le PNG".into())
}

