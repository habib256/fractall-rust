use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use num_complex::Complex64;
use png::{Decoder, Encoder};
use rayon::prelude::*;

use crate::color::{color_for_pixel_with_lut, color_for_nebulabrot_pixel, color_for_buddhabrot_pixel, PaletteLut};
use crate::fractal::{FractalParams, FractalType};

/// Clé du chunk tEXt pour les métadonnées fractales.
const METADATA_KEY: &str = "fractall-params";

/// Génère une image RGB colorisée avec métadonnées fractales intégrées dans un chunk tEXt.
///
/// Les métadonnées permettent de restaurer exactement l'état de la fractale
/// (coordonnées HP, type, paramètres) lors du chargement ultérieur de l'image.
/// Colorise les buffers bruts (itérations + z final) en RGB entrelacé
/// (3 octets/pixel, row-major). Factorisé pour être réutilisé par la
/// sauvegarde PNG ET par l'accumulation anti-aliasing multi-sample, qui
/// colorise chaque sample puis moyenne en espace RGB.
pub fn colorize_to_rgb(params: &FractalParams, iterations: &[u32], zs: &[Complex64]) -> Vec<u8> {
    let width = params.width;
    let w = width as usize;
    let h = params.height as usize;

    assert_eq!(iterations.len(), w * h, "Taille de la matrice d'itérations invalide");
    assert_eq!(zs.len(), w * h, "Taille de la matrice des valeurs z invalide");

    let is_nebulabrot = params.fractal_type == FractalType::Nebulabrot;
    let is_buddhabrot = params.fractal_type == FractalType::Buddhabrot;
    let lut = if !is_nebulabrot && !is_buddhabrot {
        Some(PaletteLut::new(params.color_mode, params.color_space))
    } else {
        None
    };

    // Parallélisation de la colorisation par lignes
    (0..h)
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
                        color_for_pixel_with_lut(
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
                            lut.as_ref(),
                        )
                    };

                    vec![r, g, b]
                })
                .collect::<Vec<u8>>()
        })
        .collect()
}

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
    let buffer = colorize_to_rgb(params, iterations, zs);
    save_png_rgb_with_metadata(
        params, &buffer, output, center_x_hp, center_y_hp, span_x_hp, span_y_hp,
    )
}

/// Sauvegarde un buffer RGB **déjà colorisé** (3 octets/pixel, row-major) avec
/// les métadonnées JSON. Utilisé par l'accumulation AA (moyenne RGB des samples)
/// où la colorisation a déjà eu lieu et où il n'y a plus de buffers bruts.
pub fn save_png_rgb_with_metadata(
    params: &FractalParams,
    buffer: &[u8],
    output: &Path,
    center_x_hp: &str,
    center_y_hp: &str,
    span_x_hp: &str,
    span_y_hp: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let width = params.width;
    let height = params.height;
    assert_eq!(
        buffer.len(),
        width as usize * height as usize * 3,
        "Taille du buffer RGB invalide"
    );

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
    // Encodage rapide (fdeflate ultra-fast) au lieu du défaut `Balanced`
    // (zlib niveau 6). La sauvegarde dominait le wall-clock des rendus rapides
    // (gt5 1024² : rendu 130 ms mais sauvegarde 280 ms) — F3 en batch Linux ne
    // sauve rien (EXR no-op), d'où un ratio speed trompeur. Fast rend le PNG
    // ~5× plus vite pour ~1.5× la taille : bon compromis pour un renderer qui
    // sauve chaque frame (PNG reste lossless → pixels identiques, goldens OK).
    encoder.set_compression(png::Compression::Fast);

    // Ajouter le chunk tEXt avec les métadonnées
    encoder.add_text_chunk(METADATA_KEY.to_string(), metadata_json)?;

    let mut png_writer = encoder.write_header()?;
    png_writer.write_image_data(buffer)?;

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Vérifie qu'on peut désérialiser un JSON legacy (sauvegardé avant
    /// l'ajout récent de champs comme jitter_scale, use_bytecode_engine).
    /// Les champs manquants doivent prendre leur default canonique sans
    /// casser.
    ///
    /// Régression historique : "Erreur chargement PNG: missing field
    /// `use_legacy_glitch_detection`" (rapporté par l'utilisateur avant
    /// la suppression de ce champ).
    #[test]
    fn deserialize_legacy_minimal_json() {
        // JSON minimum d'avant Session E : que les champs vraiment requis,
        // sans aucun des champs récents.
        let minimal = r#"{
            "width": 1920,
            "height": 1080,
            "center_x": -0.5,
            "center_y": 0.0,
            "span_x": 4.0,
            "span_y": 3.0,
            "seed": [0.0, 0.0],
            "fractal_type": "Mandelbrot",
            "iteration_max": 1000,
            "bailout": 4.0
        }"#;
        let params: FractalParams = serde_json::from_str(minimal)
            .expect("Minimal JSON should deserialize with defaults");
        // Champs requis fidèlement restaurés.
        assert_eq!(params.width, 1920);
        assert_eq!(params.iteration_max, 1000);
        assert_eq!(params.bailout, 4.0);
        // Champs récents : defaults canoniques.
        assert!(
            params.use_bytecode_engine,
            "use_bytecode_engine doit défauter à true sur PNG legacy"
        );
        assert_eq!(params.jitter_scale, 0.0);
        // Aligné F3 `engine.cc:283` : 1.0 / (1 << 24) ≈ 5.96e-8 (cf. P1.3
        // dans TODO.md). Anciennement 1e-8.
        assert_eq!(params.bla_threshold, 1.0 / (1u64 << 24) as f64);
        assert_eq!(params.glitch_tolerance, 1e-4);
        assert_eq!(params.multibrot_power, 2.5);
        assert_eq!(params.max_perturb_iterations, 1024);
        assert_eq!(params.max_bla_steps, 1024);
        assert_eq!(params.interior_threshold, 0.001);
        assert_eq!(params.max_secondary_refs, 3);
    }

    /// Vérifie qu'un JSON avec quelques-uns des champs récents présents
    /// préserve leur valeur (ne se fait pas écraser par le default).
    #[test]
    fn deserialize_respects_explicit_values() {
        let json = r#"{
            "width": 800,
            "height": 600,
            "center_x": 0.0,
            "center_y": 0.0,
            "span_x": 4.0,
            "span_y": 3.0,
            "seed": [0.0, 0.0],
            "fractal_type": "Julia",
            "iteration_max": 500,
            "bailout": 8.0,
            "use_bytecode_engine": false,
            "multibrot_power": 3.5
        }"#;
        let params: FractalParams = serde_json::from_str(json).expect("deserialize");
        assert!(!params.use_bytecode_engine);
        assert_eq!(params.multibrot_power, 3.5);
        assert_eq!(params.bailout, 8.0);
    }

    /// `transform_k` (P1.6.b-bis) : round-trip JSON. `None` ne s'écrit pas
    /// dans le JSON (skip_serializing_if). `Some([...])` est préservé tel quel.
    /// Charger un PNG legacy sans le champ → défaut `None` (fallback rotation).
    #[test]
    fn transform_k_round_trip_and_legacy_default() {
        use crate::fractal::definitions::default_params_for_type;
        use crate::fractal::FractalType;

        // Defaut : None, ne doit pas apparaître dans le JSON sérialisé.
        let p = default_params_for_type(FractalType::Mandelbrot, 100, 100);
        let json_none = serde_json::to_string(&p).expect("serialize None");
        assert!(
            !json_none.contains("transform_k"),
            "transform_k=None doit être skipped, JSON=\n{}",
            json_none
        );

        // Avec K : round-trip exact.
        let mut p2 = p.clone();
        p2.transform_k = Some([0.3, -0.4, 0.5, 0.6]);
        let json_some = serde_json::to_string(&p2).expect("serialize Some");
        let p2_back: FractalParams =
            serde_json::from_str(&json_some).expect("deserialize Some");
        assert_eq!(p2_back.transform_k, Some([0.3, -0.4, 0.5, 0.6]));

        // PNG legacy (champ absent) : défaut None.
        let legacy = r#"{
            "width": 100, "height": 100,
            "center_x": 0.0, "center_y": 0.0,
            "span_x": 4.0, "span_y": 3.0,
            "seed": [0.0, 0.0],
            "fractal_type": "Mandelbrot",
            "iteration_max": 1000,
            "bailout": 4.0
        }"#;
        let legacy_params: FractalParams =
            serde_json::from_str(legacy).expect("deserialize legacy");
        assert!(legacy_params.transform_k.is_none());
    }

    /// Régression : un PNG legacy avec `use_legacy_glitch_detection` (champ
    /// supprimé) doit charger sans erreur (le champ inconnu est ignoré par
    /// serde_json par défaut).
    /// Verrou de l'encodage rapide (`Compression::Fast`, fdeflate) : le PNG
    /// doit rester **lossless** (pixels décodés bit-identiques au buffer source)
    /// ET conserver les métadonnées (drag-and-drop). Un basculement accidentel
    /// vers un encodeur lossy/cassé, ou une régression du round-trip tEXt, casse
    /// ce test. Cf. io/png.rs `set_compression` (sauvegarde 5.6× plus rapide).
    #[test]
    fn save_rgb_fast_is_lossless_and_round_trips_metadata() {
        use crate::fractal::definitions::default_params_for_type;
        use crate::fractal::FractalType;
        // dims non alignées → exerce le filtrage par ligne
        let params = default_params_for_type(FractalType::Mandelbrot, 17, 13);
        // Buffer RGB déterministe et varié (évite un plat trivialement compressé).
        let n = (params.width * params.height) as usize;
        let src: Vec<u8> = (0..n * 3).map(|i| (i * 37 + 11) as u8).collect();

        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "fractall_png_fast_{}.png",
            std::process::id()
        ));
        save_png_rgb_with_metadata(
            &params, &src, &path, "-0.5", "0.0", "4.0", "3.0",
        )
        .expect("save doit réussir");

        // Métadonnées présentes et cohérentes.
        let loaded = load_png_metadata(&path).expect("metadata doit se charger");
        assert_eq!(loaded.width, 17);
        assert_eq!(loaded.height, 13);

        // Pixels décodés bit-identiques (lossless).
        let file = File::open(&path).unwrap();
        let mut reader = Decoder::new(BufReader::new(file)).read_info().unwrap();
        let mut out = vec![0u8; reader.output_buffer_size().unwrap()];
        let frame = reader.next_frame(&mut out).unwrap();
        assert_eq!(frame.color_type, png::ColorType::Rgb);
        assert_eq!(&out[..frame.buffer_size()], &src[..]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn deserialize_ignores_removed_legacy_field() {
        let json = r#"{
            "width": 800, "height": 600,
            "center_x": 0.0, "center_y": 0.0,
            "span_x": 4.0, "span_y": 3.0,
            "seed": [0.0, 0.0],
            "fractal_type": "Mandelbrot",
            "iteration_max": 500,
            "bailout": 4.0,
            "use_legacy_glitch_detection": false
        }"#;
        let params: FractalParams = serde_json::from_str(json)
            .expect("Removed champ doit être ignoré");
        assert_eq!(params.iteration_max, 500);
    }

    /// Test exhaustif sur les PNG du dossier `png/` du repo : tous doivent
    /// se charger sans erreur.
    #[test]
    fn deserialize_all_legacy_png_in_repo() {
        let png_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("png");
        if !png_dir.exists() {
            eprintln!("png/ dir not found, skipping");
            return;
        }
        let mut total = 0;
        let mut errors: Vec<String> = Vec::new();
        for entry in std::fs::read_dir(&png_dir).expect("read png/") {
            let path = entry.expect("entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("png") {
                continue;
            }
            total += 1;
            match load_png_metadata(&path) {
                Ok(_) => {}
                Err(e) => {
                    errors.push(format!(
                        "{}: {}",
                        path.file_name().unwrap().to_string_lossy(),
                        e
                    ));
                }
            }
        }
        eprintln!(
            "PNG legacy : {}/{} se chargent",
            total - errors.len(),
            total
        );
        if !errors.is_empty() {
            panic!(
                "Échecs de chargement legacy PNG :\n  - {}",
                errors.join("\n  - ")
            );
        }
    }
}
