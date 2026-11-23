use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use num_complex::Complex64;
use png::{Decoder, Encoder};
use rayon::prelude::*;

use crate::color::{color_for_pixel_with_lut, color_for_nebulabrot_pixel, color_for_buddhabrot_pixel, PaletteLut};
use crate::fractal::orbit_traps::OrbitData;
use crate::fractal::{FractalParams, FractalType};

/// Clé du chunk tEXt pour les métadonnées fractales.
const METADATA_KEY: &str = "fractall-params";
/// Chunk iTXt : commande `fractall-cli` CANONIQUE qui re-rend l'image, lisible
/// telle quelle (`exiftool`, `pngcheck -t`). Reconstruite depuis les paramètres
/// (même générateur que le bouton « 📋 CLI » de la GUI), JAMAIS depuis
/// `std::env::args()` : correcte aussi pour une sauvegarde GUI, itérations
/// résolues, sans chemin de la machine, identique pour deux images de la même
/// vue. Purement informatif — la source de vérité reste `fractall-params`.
pub const COMMAND_KEY: &str = "fractall-command";

/// Génère une image RGB colorisée avec métadonnées fractales intégrées dans un chunk tEXt.
///
/// Les métadonnées permettent de restaurer exactement l'état de la fractale
/// (coordonnées HP, type, paramètres) lors du chargement ultérieur de l'image.
/// Colorisation RGB entrelacée (3 octets/pixel, row-major) des buffers bruts
/// du dispatcher — **implémentation UNIQUE**, partagée par la sauvegarde PNG
/// (CLI), la recolorisation depuis une map `.fmap`, le pipeline vidéo et la
/// GUI (`gui::colorize_buffer` n'est plus qu'une enveloppe).
///
/// ⚠️ Ne pas dupliquer cette boucle : les deux copies historiques (CLI et GUI)
/// avaient divergé — la copie CLI ignorait `distances`/`orbits` (les modes
/// Distance*/OrbitTraps/Wings retombaient silencieusement sur Smooth) et ne
/// reconnaissait pas Anti-Buddhabrot comme un rendu de densité.
///
/// `distances` et `orbits` peuvent être vides (`&[]`) quand le path de rendu
/// ne les produit pas : les pixels manquants sont traités comme absents, ce
/// qui reproduit exactement l'ancien comportement pour les modes qui ne les
/// consomment pas. `width`/`height` sont explicites (la GUI colorise aussi des
/// TUILES, plus petites que `params.width × params.height`).
pub fn colorize_buffers(
    params: &FractalParams,
    iterations: &[u32],
    zs: &[Complex64],
    distances: &[f64],
    orbits: &[Option<OrbitData>],
    width: u32,
    height: u32,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let is_nebulabrot = params.fractal_type == FractalType::Nebulabrot;
    let is_buddhabrot = params.fractal_type == FractalType::Buddhabrot
        || params.fractal_type == FractalType::AntiBuddhabrot;
    let interior_flag_encoded = params.channels.enable_interior_detection;
    let lut = if !is_nebulabrot && !is_buddhabrot {
        Some(PaletteLut::cached(params.color.color_mode, params.color.color_space))
    } else {
        None
    };

    // Écriture DIRECTE dans le buffer de sortie (même patron que
    // `video::lighting::shade_rgb` / `interpolate_frame`).
    // ⚠️ Ne pas revenir à `flat_map(|x| vec![r, g, b]).collect()` : cette
    // forme allouait un `Vec` de 3 octets PAR PIXEL, puis un `Vec` par ligne,
    // puis concaténait le tout (cf. `examples/bench_colorize.rs`).
    let row_body = |y: usize, row: &mut [u8]| {
        for x in 0..w {
            let idx = y * w + x;
            let iter = iterations.get(idx).copied().unwrap_or(0);
            let z = zs.get(idx).copied().unwrap_or(Complex64::new(0.0, 0.0));
            let orbit = orbits.get(idx).and_then(|o| o.as_ref());
            let distance = distances.get(idx).copied().filter(|d| d.is_finite());

            let (r, g, b) = if is_nebulabrot {
                color_for_nebulabrot_pixel(iter, z)
            } else if is_buddhabrot {
                color_for_buddhabrot_pixel(z, params.color.color_mode, params.color.color_repeat)
            } else {
                color_for_pixel_with_lut(
                    iter,
                    z,
                    params.iteration_max,
                    params.color.color_mode,
                    params.color.color_repeat,
                    params.color.color_offset,
                    params.color.out_coloring_mode,
                    params.color.color_space,
                    orbit,
                    distance,
                    interior_flag_encoded,
                    lut.as_deref(),
                )
            };

            let o = x * 3;
            row[o] = r;
            row[o + 1] = g;
            row[o + 2] = b;
        }
    };

    // Parallélisation par lignes, y compris pour les petites TUILES du
    // streaming GUI : mesuré (`bench_colorize`, pool saturé) équivalent à une
    // boucle série sur une passe complète (25.7 vs 25.9 ms / 240 tuiles), mais
    // 3× plus rapide sur une tuile isolée (fin de passe, workers drainés).
    let mut out = vec![0u8; w * h * 3];
    out.par_chunks_mut(w * 3)
        .enumerate()
        .for_each(|(y, row)| row_body(y, row));
    out
}

/// Colorisation VÉRIFIÉE d'un `RenderOutput` complet (chantier G5).
///
/// Point d'entrée à privilégier dès qu'on colorise la sortie du dispatcher :
/// vérifie que les canaux requis par `params.out_coloring_mode` sont présents
/// (`RenderOutput::validate_channels`) et renvoie `Err` sinon — au lieu de
/// l'image plausible-mais-fausse (retombée silencieuse sur Smooth) qui a valu
/// 4 bugs à la chasse 2026-08-23 (copie CLI, boucle AA, vidéo, auto-GPU).
pub fn colorize_output(
    params: &FractalParams,
    out: &crate::render::output::RenderOutput,
) -> Result<Vec<u8>, String> {
    out.validate_channels(params)?;
    let n = params.width as usize * params.height as usize;
    if out.iterations.len() != n || out.zs.len() != n {
        return Err(format!(
            "incomplete render buffers: {} iterations / {} zs for {}×{} px",
            out.iterations.len(),
            out.zs.len(),
            params.width,
            params.height
        ));
    }
    Ok(colorize_buffers(
        params,
        &out.iterations,
        &out.zs,
        &out.distances,
        &out.orbits,
        params.width,
        params.height,
    ))
}

/// Colorise les buffers bruts (itérations + z final) en RGB entrelacé
/// (3 octets/pixel, row-major). Factorisé pour être réutilisé par la
/// sauvegarde PNG ET par l'accumulation anti-aliasing multi-sample, qui
/// colorise chaque sample puis moyenne en espace RGB.
///
/// Enveloppe sans canaux annexes : à réserver aux paths qui n'en produisent
/// pas (AA multi-sample). Les appelants qui DISPOSENT des distances/orbites
/// (CLI, `--from-map`, vidéo) doivent passer par
/// `colorize_to_rgb_with_extras`, sinon les modes Distance*/OrbitTraps/Wings
/// retombent silencieusement sur Smooth.
pub fn colorize_to_rgb(params: &FractalParams, iterations: &[u32], zs: &[Complex64]) -> Vec<u8> {
    colorize_to_rgb_with_extras(params, iterations, zs, &[], &[])
}

/// Comme `colorize_to_rgb`, mais en consommant les canaux annexes du
/// dispatcher : `distances` (modes Distance/DistanceAO/Distance3D) et `orbits`
/// (modes OrbitTraps/Wings). Passer `&[]` pour un canal absent.
pub fn colorize_to_rgb_with_extras(
    params: &FractalParams,
    iterations: &[u32],
    zs: &[Complex64],
    distances: &[f64],
    orbits: &[Option<OrbitData>],
) -> Vec<u8> {
    let n = params.width as usize * params.height as usize;
    assert_eq!(iterations.len(), n, "Invalid iteration buffer size");
    assert_eq!(zs.len(), n, "Invalid z-value buffer size");
    colorize_buffers(
        params,
        iterations,
        zs,
        distances,
        orbits,
        params.width,
        params.height,
    )
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
    save_png_rgb_with_metadata_aa(
        params, buffer, output, center_x_hp, center_y_hp, span_x_hp, span_y_hp, 1,
    )
}

/// Comme [`save_png_rgb_with_metadata`], pour un buffer issu de l'accumulation
/// AA : `aa_samples` n'est pas un champ de `FractalParams` (non persisté), il
/// n'est connu que de l'appelant et ne sert ici qu'à la commande embarquée
/// (`--aa-samples`, `--jitter-scale` lu dans `params.sampling.jitter_scale`).
#[allow(clippy::too_many_arguments)]
pub fn save_png_rgb_with_metadata_aa(
    params: &FractalParams,
    buffer: &[u8],
    output: &Path,
    center_x_hp: &str,
    center_y_hp: &str,
    span_x_hp: &str,
    span_y_hp: &str,
    aa_samples: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let width = params.width;
    let height = params.height;
    assert_eq!(
        buffer.len(),
        width as usize * height as usize * 3,
        "Invalid RGB buffer size"
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
    encoder.add_itxt_chunk(
        COMMAND_KEY.to_string(),
        canonical_command(&params_to_save, aa_samples),
    )?;

    let mut png_writer = encoder.write_header()?;
    png_writer.write_image_data(buffer)?;
    png_writer.finish()?;

    Ok(())
}

/// Commande canonique embarquée dans le chunk [`COMMAND_KEY`].
fn canonical_command(params: &FractalParams, aa_samples: u32) -> String {
    use crate::gui::cli_command::{build_cli_command, CliExtras};
    let extras = CliExtras {
        aa_samples: aa_samples.max(1),
        aa_jitter_scale: params.sampling.jitter_scale,
        // Centre EXACT : arrondi, il déplace la référence de perturbation et
        // la commande ne redonnerait plus l'image au pixel près.
        exact_center: true,
    };
    build_cli_command(params, extras).command
}

/// Commande `fractall-cli` embarquée dans un PNG fractall (`None` pour un PNG
/// antérieur à ce chunk ou étranger).
pub fn load_png_command(path: &Path) -> Option<String> {
    let decoder = Decoder::new(BufReader::new(File::open(path).ok()?));
    let reader = decoder.read_info().ok()?;
    reader
        .info()
        .utf8_text
        .iter()
        .find(|chunk| chunk.keyword == COMMAND_KEY)
        .and_then(|chunk| chunk.get_text().ok())
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

    Err("No fractall metadata found in the PNG".into())
}

/// Message d'erreur (anglais, destiné à l'utilisateur CLI) d'un PNG sans
/// chunk `fractall-params`.
pub const NOT_A_FRACTALL_PNG: &str = "not a fractall PNG: no embedded parameters — save it from \
fractall-gui with S or render it with fractall-cli";

/// `fractall-cli --from-png` : paramètres d'un PNG fractall prêts à être
/// re-rendus à la taille demandée — logique PURE (aucun rendu), testée ici.
///
/// Dimensions : `None` garde celle du PNG ; une seule fournie ⇒ l'autre est
/// dérivée du ratio du PNG (arrondi, ≥ 1).
///
/// La VUE est conservée : centre (chaînes HP intactes au caractère près) et
/// étendue verticale (`span_y` / `span_y_hp` intacts). Si le ratio change,
/// `span_x` / `span_x_hp` sont dérivés du nouveau ratio en haute précision
/// (`ViewHp::resize_keep_vertical_span`, jamais par un f64 : à 1e60 le
/// miroir f64 resterait juste mais une chaîne reconstruite depuis lui
/// perdrait les chiffres au-delà du 17e). Ratio inchangé (même taille,
/// upscale ×2…) : spans intacts au caractère près.
pub fn params_for_rerender(
    path: &Path,
    width: Option<u32>,
    height: Option<u32>,
) -> Result<FractalParams, String> {
    if !path.exists() {
        return Err(format!("{}: file not found", path.display()));
    }
    let file =
        File::open(path).map_err(|e| format!("{}: cannot open file: {e}", path.display()))?;
    let png_reader = Decoder::new(BufReader::new(file))
        .read_info()
        .map_err(|e| format!("{}: not a readable PNG image: {e}", path.display()))?;
    let text = png_reader
        .info()
        .uncompressed_latin1_text
        .iter()
        .find(|chunk| chunk.keyword == METADATA_KEY)
        .map(|chunk| chunk.text.clone())
        .ok_or_else(|| format!("{}: {NOT_A_FRACTALL_PNG}", path.display()))?;
    let mut params: FractalParams = serde_json::from_str(&text).map_err(|e| {
        format!("{}: corrupted fractall parameters in PNG: {e}", path.display())
    })?;

    let (old_w, old_h) = (params.width.max(1), params.height.max(1));
    let derive = |given: u32, num: u32, den: u32| -> u32 {
        ((given as f64 * num as f64 / den as f64).round() as u32).max(1)
    };
    let (new_w, new_h) = match (width, height) {
        (Some(w), Some(h)) => (w, h),
        (Some(w), None) => (w, derive(w, old_h, old_w)),
        (None, Some(h)) => (derive(h, old_w, old_h), h),
        (None, None) => (old_w, old_h),
    };
    if new_w == 0 || new_h == 0 {
        return Err(format!("--width/--height must be >= 1 (got {new_w}x{new_h})"));
    }

    let same_aspect = new_w as u64 * old_h as u64 == new_h as u64 * old_w as u64;
    if !same_aspect {
        let mut view = crate::fractal::ViewHp::from_params(&params);
        view.resize_keep_vertical_span(new_w, new_h);
        params.span_x = view.span_x().to_f64();
        params.span_x_hp = Some(view.span_x().to_string_radix(10, None));
    }
    params.width = new_w;
    params.height = new_h;
    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::definitions::default_params_for_type;
    use crate::fractal::OutColoringMode;

    /// Buffers d'un rendu de densité (Buddhabrot-like) : `z.re` porte la
    /// densité normalisée × 2, `iterations` la même densité × iteration_max.
    fn density_buffers(n: usize, iter_max: u32) -> (Vec<u32>, Vec<Complex64>) {
        let norm = |i: usize| (i as f64 + 1.0) / n as f64;
        (
            (0..n).map(|i| (norm(i) * iter_max as f64) as u32).collect(),
            (0..n).map(|i| Complex64::new(norm(i) * 2.0, 0.0)).collect(),
        )
    }

    /// Régression : Anti-Buddhabrot est un rendu de DENSITÉ, comme Buddhabrot.
    /// La colorisation CLI le traitait comme un escape-time (la GUI non) → le
    /// PNG sorti par `fractall-cli --type 33` ne ressemblait pas à ce que la
    /// GUI affichait, et le pixel de densité maximale (iter == iteration_max)
    /// tombait dans la branche « intérieur » → noir.
    #[test]
    fn antibuddhabrot_colorizes_as_density_like_buddhabrot() {
        let (w, h) = (8u32, 4u32);
        let n = (w * h) as usize;
        let mut anti = default_params_for_type(FractalType::AntiBuddhabrot, w, h);
        anti.iteration_max = 500;
        let (iterations, zs) = density_buffers(n, anti.iteration_max);

        let mut buddha = anti.clone();
        buddha.fractal_type = FractalType::Buddhabrot;
        buddha.color.color_mode = anti.color.color_mode;
        buddha.color.color_repeat = anti.color.color_repeat;

        let anti_rgb = colorize_to_rgb(&anti, &iterations, &zs);
        let buddha_rgb = colorize_to_rgb(&buddha, &iterations, &zs);
        assert_eq!(anti_rgb, buddha_rgb, "Anti-Buddhabrot doit suivre le path densité");

        // Le pixel de densité maximale n'est PAS noir (l'ancien path
        // escape-time le rendait noir via `iteration >= iteration_max`).
        let last = (n - 1) * 3;
        assert_ne!(
            &anti_rgb[last..last + 3],
            &[0u8, 0, 0],
            "densité maximale colorisée en noir = ancien bug escape-time"
        );
    }

    /// G5 `RenderOutput` : la colorisation VÉRIFIÉE refuse un canal requis
    /// absent (classe « Smooth silencieux ») et reste bit-identique à
    /// `colorize_buffers` quand les canaux sont là.
    #[test]
    fn colorize_output_refuses_dropped_required_channel() {
        let (w, h) = (8u32, 6u32);
        let n = (w * h) as usize;
        let mut params = default_params_for_type(FractalType::Mandelbrot, w, h);
        params.color.out_coloring_mode = OutColoringMode::Distance;

        let bare = crate::render::RenderOutput::without_extras(
            vec![1; n],
            vec![Complex64::new(5.0, 1.0); n],
        );
        assert!(
            colorize_output(&params, &bare).is_err(),
            "mode Distance sans canal distances doit être une ERREUR"
        );

        let mut with = bare.clone();
        with.distances = vec![1e-6; n];
        let rgb = colorize_output(&params, &with).expect("canal présent");
        assert_eq!(
            rgb,
            colorize_buffers(&params, &with.iterations, &with.zs, &with.distances, &with.orbits, w, h),
            "colorize_output == colorize_buffers (mêmes canaux)"
        );

        // OrbitTraps : même contrat sur le canal orbits.
        params.color.out_coloring_mode = OutColoringMode::Wings;
        assert!(colorize_output(&params, &bare).is_err());

        // Smooth : aucun canal requis → OK sans extras.
        params.color.out_coloring_mode = OutColoringMode::Smooth;
        assert!(colorize_output(&params, &bare).is_ok());

        // Buffers incomplets (iterations tronquées) refusés aussi.
        let short = crate::render::RenderOutput::without_extras(vec![1; n / 2], vec![Complex64::new(0.0, 0.0); n / 2]);
        assert!(colorize_output(&params, &short).is_err());
    }

    /// Régression : les modes Distance* consomment le canal `distances`. Sans
    /// lui la colorisation retombe SILENCIEUSEMENT sur Smooth — c'est ce que
    /// faisait le CLI (et `--from-map`, et le pipeline vidéo) alors que le
    /// dispatcher produisait bien les distances.
    #[test]
    fn distance_channel_changes_distance_mode_output() {
        let (w, h) = (16u32, 8u32);
        let n = (w * h) as usize;
        let mut params = default_params_for_type(FractalType::Mandelbrot, w, h);
        params.iteration_max = 100;
        params.channels.enable_distance_estimation = true;
        params.color.out_coloring_mode = OutColoringMode::Distance;

        // Pixels échappés (iter < iteration_max) sinon la couleur est noire
        // avant même de regarder le mode.
        let iterations: Vec<u32> = (0..n).map(|i| (i % 90) as u32 + 1).collect();
        let zs: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(5.0 + i as f64 * 0.01, 1.0))
            .collect();
        let distances: Vec<f64> = (0..n).map(|i| 1e-6 * (i as f64 + 1.0)).collect();

        let without = colorize_to_rgb(&params, &iterations, &zs);
        let with = colorize_to_rgb_with_extras(&params, &iterations, &zs, &distances, &[]);
        assert_ne!(with, without, "le canal distances doit être consommé");

        // Les modes qui n'utilisent PAS les distances restent bit-identiques
        // (garantie de non-régression des goldens).
        let mut smooth = params.clone();
        smooth.color.out_coloring_mode = OutColoringMode::Smooth;
        assert_eq!(
            colorize_to_rgb_with_extras(&smooth, &iterations, &zs, &distances, &[]),
            colorize_to_rgb(&smooth, &iterations, &zs),
            "Smooth ignore les distances : sortie inchangée"
        );
    }

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
            params.engine.use_bytecode_engine,
            "use_bytecode_engine doit défauter à true sur PNG legacy"
        );
        assert_eq!(params.sampling.jitter_scale, 0.0);
        // Aligné F3 `engine.cc:283` : 1.0 / (1 << 24) ≈ 5.96e-8 (cf. P1.3
        // dans TODO.md). Anciennement 1e-8.
        assert_eq!(params.perturbation.bla_threshold, 1.0 / (1u64 << 24) as f64);
        assert_eq!(params.perturbation.glitch_tolerance, 1e-4);
        assert_eq!(params.formula.multibrot_power, 2.5);
        assert_eq!(params.perturbation.max_perturb_iterations, 1024);
        assert_eq!(params.perturbation.max_bla_steps, 1024);
        assert_eq!(params.channels.interior_threshold, 0.001);
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
        assert!(!params.engine.use_bytecode_engine);
        assert_eq!(params.formula.multibrot_power, 3.5);
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

    /// Les réglages regroupés en sous-structures restent lisibles depuis les
    /// PNG écrits AVANT le regroupement : leurs clés sont toujours à plat dans
    /// le JSON, et doivent atterrir dans la bonne sous-structure.
    #[test]
    fn legacy_png_populates_grouped_params() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("png")
            .join("Deepfractal.png");
        if !path.exists() {
            eprintln!("{} absent, test ignoré", path.display());
            return;
        }
        let params = load_png_metadata(&path).expect("PNG legacy lisible");
        // Valeurs non triviales écrites par une version antérieure : si les
        // clés à plat n'alimentaient plus les sous-structures, on lirait les
        // défauts à la place.
        assert_eq!(params.color.color_repeat, 120);
        assert_eq!(params.color.color_mode, 0);
        assert_eq!(params.perturbation.max_bla_steps, 1024);
        assert!(params.perturbation.bla_threshold > 0.0);
        assert_eq!(
            params.engine.algorithm_mode,
            crate::fractal::AlgorithmMode::Perturbation
        );
        assert_eq!(params.engine.precision_bits, 256);
        // Écrit avant l'existence du champ : le défaut `true` du groupe doit
        // s'appliquer (et non le `false` de `Default` de Rust).
        assert!(params.engine.use_bytecode_engine);
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

    /// PNG fractall d'une vue profonde (1e60) à 160×90 pour les tests
    /// `--from-png` ; renvoie le chemin et les chaînes HP écrites.
    fn deep_png(tag: &str) -> (std::path::PathBuf, [String; 4]) {
        let mut params = default_params_for_type(FractalType::Mandelbrot, 160, 90);
        params.iteration_max = 7000;
        params.color.color_mode = 3;
        let hp = [
            "-1.861369427601275917203958599290880106793222303477204934785274017462633431321514"
                .to_string(),
            "0.00008728845098728640945449739639040797666621419255681987270008753766941151815473273"
                .to_string(),
            "1.2732395447351626861510701069801148962756771659236515899813387524711743810738122e-60"
                .to_string(),
            "7.161972439135290109599769351763146291550684058320540193645030482650355893540193e-61"
                .to_string(),
        ];
        params.center_x = hp[0].parse().unwrap();
        params.center_y = hp[1].parse().unwrap();
        params.span_x = hp[2].parse().unwrap();
        params.span_y = hp[3].parse().unwrap();
        let path = std::env::temp_dir().join(format!(
            "fractall_rerender_{}_{}.png",
            tag,
            std::process::id()
        ));
        let rgb = vec![0u8; 160 * 90 * 3];
        save_png_rgb_with_metadata(&params, &rgb, &path, &hp[0], &hp[1], &hp[2], &hp[3])
            .expect("save");
        (path, hp)
    }

    fn hp_float(s: &str) -> rug::Float {
        rug::Float::with_val(1024, rug::Float::parse(s).unwrap())
    }

    #[test]
    fn rerender_same_aspect_keeps_every_hp_string() {
        let (path, hp) = deep_png("same");
        for (w, h) in [(None, None), (Some(320), Some(180)), (Some(320), None), (None, Some(180))] {
            let p = params_for_rerender(&path, w, h).unwrap();
            let expect = if w.is_none() && h.is_none() { (160, 90) } else { (320, 180) };
            assert_eq!((p.width, p.height), expect);
            assert_eq!(p.center_x_hp.as_deref(), Some(hp[0].as_str()));
            assert_eq!(p.center_y_hp.as_deref(), Some(hp[1].as_str()));
            assert_eq!(p.span_x_hp.as_deref(), Some(hp[2].as_str()));
            assert_eq!(p.span_y_hp.as_deref(), Some(hp[3].as_str()));
            assert_eq!(p.iteration_max, 7000);
            assert_eq!(p.color.color_mode, 3);
        }
        let _ = std::fs::remove_file(&path);
    }

    /// Changement de ratio à 1e60 : centre et span_y intacts au caractère
    /// près, span_x = span_y·w/h exact en haute précision (une dérivation via
    /// f64 n'aurait que ~16 chiffres significatifs).
    #[test]
    fn rerender_aspect_change_keeps_center_and_span_y_in_hp() {
        let (path, hp) = deep_png("aspect");
        let p = params_for_rerender(&path, Some(100), Some(100)).unwrap();
        assert_eq!((p.width, p.height), (100, 100));
        assert_eq!(p.center_x_hp.as_deref(), Some(hp[0].as_str()));
        assert_eq!(p.center_y_hp.as_deref(), Some(hp[1].as_str()));
        assert_eq!(p.span_y_hp.as_deref(), Some(hp[3].as_str()));
        let span_y = hp_float(&hp[3]);
        let span_x = hp_float(p.span_x_hp.as_deref().unwrap());
        let diff = rug::Float::with_val(1024, &span_x - &span_y);
        let rel = rug::Float::with_val(1024, diff / &span_y).abs();
        assert!(rel < 1e-70, "span_x HP lost precision: rel {rel}");
        assert_eq!(p.span_x, span_y.to_f64());

        let wide = params_for_rerender(&path, Some(400), Some(100)).unwrap();
        let ratio = rug::Float::with_val(1024, hp_float(wide.span_x_hp.as_deref().unwrap()) / &span_y);
        assert!(rug::Float::with_val(1024, ratio - 4).abs() < 1e-70);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rerender_errors_are_explicit() {
        let missing = std::env::temp_dir().join("fractall_rerender_does_not_exist.png");
        let e = params_for_rerender(&missing, None, None).unwrap_err();
        assert!(e.contains("file not found"), "{e}");

        let plain = std::env::temp_dir()
            .join(format!("fractall_rerender_plain_{}.png", std::process::id()));
        image::RgbImage::new(4, 4).save(&plain).unwrap();
        let e = params_for_rerender(&plain, None, None).unwrap_err();
        assert!(e.contains(NOT_A_FRACTALL_PNG), "{e}");
        let _ = std::fs::remove_file(&plain);

        let (path, _) = deep_png("zero");
        assert!(params_for_rerender(&path, Some(0), None).is_err());
        let _ = std::fs::remove_file(&path);
    }

    /// Le PNG embarque la commande CANONIQUE (générateur du bouton « 📋 CLI »),
    /// pas la ligne de commande du processus : aucun chemin de la machine,
    /// `--aa-samples` présent seulement pour une accumulation AA, et le chunk
    /// `fractall-params` reste lisible à côté.
    #[test]
    fn saved_png_embeds_canonical_command() {
        use crate::gui::cli_command::{build_cli_command, CliExtras};
        let (w, h) = (8u32, 4u32);
        let mut params = default_params_for_type(FractalType::Mandelbrot, w, h);
        params.iteration_max = 777;
        let rgb = vec![0u8; (w * h * 3) as usize];
        let dir = std::env::temp_dir().join(format!("fractall-cmd-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let hp = ("-0.75", "0.1", "0.004", "0.002");

        let plain = dir.join("secret-user-dir-plain.png");
        save_png_rgb_with_metadata(&params, &rgb, &plain, hp.0, hp.1, hp.2, hp.3).unwrap();
        let cmd = load_png_command(&plain).expect("command chunk");
        let mut saved = params.clone();
        saved.center_x_hp = Some(hp.0.into());
        saved.center_y_hp = Some(hp.1.into());
        saved.span_x_hp = Some(hp.2.into());
        saved.span_y_hp = Some(hp.3.into());
        let exact = CliExtras { exact_center: true, ..CliExtras::default() };
        assert_eq!(cmd, build_cli_command(&saved, exact).command);
        assert!(cmd.starts_with("fractall-cli --type 3 "), "{cmd}");
        assert!(cmd.contains("--iterations 777"), "{cmd}");
        assert!(cmd.contains("--center-x-hp=-0.75"), "{cmd}");
        assert!(!cmd.contains("secret-user-dir") && !cmd.contains("aa-samples"), "{cmd}");
        assert_eq!(load_png_metadata(&plain).unwrap().iteration_max, 777);

        let aa = dir.join("aa.png");
        for jitter in [0.0, 0.5] {
            params.sampling.jitter_scale = jitter;
            save_png_rgb_with_metadata_aa(&params, &rgb, &aa, hp.0, hp.1, hp.2, hp.3, 4).unwrap();
            let cmd = load_png_command(&aa).unwrap();
            assert!(cmd.contains(&format!("--aa-samples 4 --jitter-scale {jitter}")), "{cmd}");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
