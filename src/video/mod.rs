//! Pipeline vidéo zoom (G12) — keyframes ×2 + interpolation.
//!
//! Architecture DeepDrill (deepmake/deepdrill/deepzoom) adaptée : un projet =
//! un dossier avec `manifest.toml` + une map `.fmap` par keyframe. Trois
//! étapes, chacune utile seule :
//!
//! 1. `plan`     — écrit le manifest (géométrie des keyframes dérivée du zoom
//!                 final, spans en progression ×2 calculés en GMP) ;
//! 2. `render`   — calcule les keyframes manquantes via le **dispatcher
//!                 unique** (`render_escape_time_cancellable_with_reuse`,
//!                 `cache/xaos/tiles = None`, sémantique single-shot) et les
//!                 persiste en `.fmap` ; reprise = skip des maps valides ;
//! 3. `assemble` — colorise les keyframes et interpole les frames
//!                 intermédiaires (cf. `assemble.rs`).
//!
//! ⚠️ Pas de réutilisation d'orbite entre keyframes : en régime deep la
//! troncature atom-domain d'une référence est baked à son span de construction
//! (cf. CLAUDE.md §Cache, `atom_regime_scale_mismatch`) — chaque keyframe est
//! un rendu single-shot.

pub mod assemble;
pub mod lighting;
pub mod spline;

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use rug::Float;
use serde::{Deserialize, Serialize};

use crate::fractal::{default_params_for_type, FractalParams, FractalType, OutColoringMode};
use crate::io::fmap::{load_fmap, save_fmap, FractalMap};
use crate::render::render_escape_time_cancellable_with_reuse;

/// Convention CLI : magnification 1 ⇔ span_x = 4 (cf. main.rs `--zoom`).
const SPAN_AT_ZOOM_1: f64 = 4.0;

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct LocationSection {
    /// Centre X haute précision (string GMP).
    pub real: String,
    /// Centre Y haute précision (string GMP).
    pub imag: String,
    /// Magnification finale (span_x_final = 4/zoom). Notation scientifique OK.
    pub zoom: String,
}

impl Default for LocationSection {
    fn default() -> Self {
        Self { real: "-0.5".into(), imag: "0.0".into(), zoom: "1e3".into() }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct ImageSection {
    pub width: u32,
    pub height: u32,
    /// Facteur de sur-échantillonnage des keyframes (AA vidéo : les maps sont
    /// rendues à `width×supersample`, le downscale de l'assembleur lisse).
    pub supersample: u32,
}

impl Default for ImageSection {
    fn default() -> Self {
        Self { width: 1280, height: 720, supersample: 1 }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct FractalSection {
    /// Id du type (3 = Mandelbrot, cf. `FractalType::from_id`).
    pub r#type: u8,
    /// Plafond d'itérations à la keyframe 0.
    pub iterations: u32,
    /// Itérations ajoutées par keyframe (linéaire — les zooms profonds en
    /// demandent plus).
    pub iterations_growth: f64,
    /// Estimation de distance (nécessaire aux modes Distance*).
    pub distance_estimation: bool,
}

impl Default for FractalSection {
    fn default() -> Self {
        Self { r#type: 3, iterations: 1000, iterations_growth: 0.0, distance_estimation: false }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct ColorSection {
    pub palette: u8,
    pub color_repeat: u32,
    pub outcoloring: String,
    /// Décalage cyclique de palette ∈ [0,1) (0 = neutre). Spline temporelle
    /// possible via `[dynamics] palette_offset` (jalon 4).
    pub palette_offset: f64,
}

impl Default for ColorSection {
    fn default() -> Self {
        Self { palette: 6, color_repeat: 40, outcoloring: "smooth".into(), palette_offset: 0.0 }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct VideoSection {
    pub fps: u32,
    /// Vitesse de zoom en keyframes (×2) par seconde. Valeur fixe (`"0.5"`)
    /// ou spline temporelle (`"0:0/0,0:2/1,…"`, cf. `spline.rs`, jalon 4).
    pub velocity: String,
    /// Nombre de segments keyframe (rempli par `plan` : ceil(log2(zoom))).
    /// Les maps rendues vont de 0 à `keyframes` INCLUS (bornes des segments).
    pub keyframes: u32,
}

impl Default for VideoSection {
    fn default() -> Self {
        Self { fps: 30, velocity: "1.0".into(), keyframes: 0 }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct LightingSection {
    /// Éclairage normal-map « spatial images » (jalon 5).
    pub enable: bool,
    /// Azimut de la lumière, degrés (0 = est, CCW).
    pub alpha: f64,
    /// Inclinaison de la lumière, degrés (90 = zénith ; plus bas = relief
    /// plus marqué).
    pub beta: f64,
}

impl Default for LightingSection {
    fn default() -> Self {
        Self { enable: false, alpha: 45.0, beta: 45.0 }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct DynamicsSection {
    /// Spline temporelle du décalage de palette (jalon 4). Prioritaire sur
    /// `[color] palette_offset` quand présente.
    pub palette_offset: Option<String>,
}

/// Manifest d'un projet vidéo (`manifest.toml`). Sert aussi de format de
/// config d'entrée à `plan` (qui remplit `video.keyframes`).
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct Manifest {
    pub location: LocationSection,
    pub image: ImageSection,
    pub fractal: FractalSection,
    pub color: ColorSection,
    pub video: VideoSection,
    pub lighting: LightingSection,
    pub dynamics: DynamicsSection,
}

impl Manifest {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&text)?)
    }

    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::write(path, toml::to_string_pretty(self)?)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Géométrie des keyframes
// ---------------------------------------------------------------------------

/// Nombre de segments keyframe pour atteindre `zoom` : `ceil(log2(zoom))`,
/// exact même pour les puissances de 2 (via mantisse·2^e GMP). Minimum 1.
pub fn keyframe_count(zoom: &str) -> Result<u32, String> {
    let f = Float::parse(zoom)
        .map(|p| Float::with_val(128, p))
        .map_err(|e| format!("zoom illisible '{zoom}': {e}"))?;
    if !f.is_finite() || f <= 0.0 {
        return Err(format!("zoom invalide '{zoom}' (doit être fini et > 0)"));
    }
    // f = m·2^e, m ∈ [0.5, 1) → log2(f) ∈ (e-1, e] ; ceil = e sauf si
    // f == 2^(e-1) exactement (m = 0.5) où c'est e-1.
    let e = f.get_exp().unwrap_or(0) as i64;
    let pow = Float::with_val(128, Float::i_exp(1, (e - 1).clamp(i32::MIN as i64, i32::MAX as i64) as i32));
    let n = if f == pow { e - 1 } else { e };
    Ok(n.max(1) as u32)
}

/// Précision GMP pour l'arithmétique des spans à la keyframe `k` :
/// `-log2(span_k) + 96` bits de marge, plancher 256 (même règle que la GUI,
/// `hp_arith_precision`).
fn span_precision(k: u32) -> u32 {
    (k + 96).max(256)
}

/// Paramètres COMPLETS de la keyframe `k` (0 = vue pleine, `k` = span/2^k).
/// Centre fixe (= la cible), spans en progression ×2 exacte : `span_x(k) =
/// 4/2^k` est une puissance de 2, donc **exacte en GMP à toute profondeur**.
pub fn keyframe_params(m: &Manifest, k: u32) -> Result<FractalParams, String> {
    let ftype = FractalType::from_id(m.fractal.r#type)
        .ok_or_else(|| format!("type de fractale invalide: {}", m.fractal.r#type))?;
    let ss = m.image.supersample.max(1);
    let (w, h) = (m.image.width * ss, m.image.height * ss);
    let mut p = default_params_for_type(ftype, w, h);

    let prec = span_precision(k);
    // Centre HP (strings du manifest, vérité absolue) + approximation f64.
    let cx = Float::parse(&m.location.real)
        .map(|v| Float::with_val(prec, v))
        .map_err(|e| format!("location.real illisible: {e}"))?;
    let cy = Float::parse(&m.location.imag)
        .map(|v| Float::with_val(prec, v))
        .map_err(|e| format!("location.imag illisible: {e}"))?;
    p.center_x_hp = Some(m.location.real.clone());
    p.center_y_hp = Some(m.location.imag.clone());
    p.center_x = cx.to_f64();
    p.center_y = cy.to_f64();

    // span_x = 4 / 2^k — division par une puissance de 2, EXACTE en binaire.
    let span_x = Float::with_val(prec, SPAN_AT_ZOOM_1) >> k;
    let aspect = h as f64 / w as f64; // même formule que main.rs (parité CLI)
    let span_y = Float::with_val(prec, &span_x * aspect);
    // Sérialisation décimale EXACTE : 2^(2−k) (et aspect·2^(2−k), aspect
    // dyadique ≤ 53 bits) ont une expansion décimale FINIE de ~0.7·k chiffres
    // (2^−n = 5^n·10^−n). `Display` n'imprime que ~prec·log10(2) chiffres →
    // un parse-back à précision supérieure divergerait au-delà (verrou
    // progression exacte). On imprime donc l'expansion complète.
    let digits = k as usize * 7 / 10 + 60;
    p.span_x_hp = Some(span_x.to_string_radix(10, Some(digits)));
    p.span_y_hp = Some(span_y.to_string_radix(10, Some(digits)));
    p.span_x = span_x.to_f64();
    p.span_y = span_y.to_f64();

    let iters = (m.fractal.iterations as f64 + m.fractal.iterations_growth * k as f64)
        .round()
        .max(1.0) as u32;
    p.iteration_max = iters;
    // Mirror la sémantique F3/loader TOML (main.rs) : caps = iterations,
    // sinon les pas directs sont tronqués à 1024 en deep (anneaux parasites).
    p.max_perturb_iterations = iters;
    p.max_bla_steps = iters;

    p.color_mode = m.color.palette;
    p.color_repeat = m.color.color_repeat.max(1);
    p.out_coloring_mode = OutColoringMode::from_cli_name(&m.color.outcoloring)
        .ok_or_else(|| format!("outcoloring invalide: '{}'", m.color.outcoloring))?;
    p.enable_distance_estimation = m.fractal.distance_estimation;
    Ok(p)
}

/// Chemin de la map de la keyframe `k` dans le projet.
pub fn keyframe_path(project: &Path, k: u32) -> PathBuf {
    project.join(format!("keyframe_{k:05}.fmap"))
}

/// Empreinte de VALIDITÉ d'une map de keyframe : les params SANS les champs
/// couleur. Les buffers d'une map (itérations, z, distances) ne dépendent pas
/// de la colorisation — changer la palette du manifest ne doit PAS invalider
/// des heures de calcul (c'est tout l'intérêt du format map).
pub fn map_fingerprint(params: &FractalParams) -> String {
    let mut p = params.clone();
    p.color_mode = 0;
    p.color_repeat = 1;
    p.out_coloring_mode = OutColoringMode::Smooth;
    serde_json::to_string(&p).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// plan / render
// ---------------------------------------------------------------------------

/// `plan` : lit une config (même format TOML que le manifest), calcule
/// `video.keyframes` depuis le zoom final et écrit `project/manifest.toml`.
/// Crée le dossier si besoin. Retourne le manifest écrit.
pub fn plan_project(config: &Path, project: &Path) -> Result<Manifest, Box<dyn std::error::Error>> {
    let mut m = Manifest::load(config)?;
    m.video.keyframes = keyframe_count(&m.location.zoom)?;
    // Valide la géométrie tôt (types/outcoloring invalides = erreur au plan,
    // pas au 30e keyframe du render).
    keyframe_params(&m, 0).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    std::fs::create_dir_all(project)?;
    m.save(&project.join("manifest.toml"))?;
    Ok(m)
}

/// `render` : calcule les keyframes 0..=keyframes manquantes. Une map
/// existante dont l'empreinte (params hors couleur) correspond est SKIPPÉE —
/// c'est la reprise après interruption. Retourne (rendues, skippées).
pub fn render_project(project: &Path) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let manifest = Manifest::load(&project.join("manifest.toml"))?;
    let n = manifest.video.keyframes;
    let cancel = Arc::new(AtomicBool::new(false));
    let (mut rendered, mut skipped) = (0usize, 0usize);

    for k in 0..=n {
        let params = keyframe_params(&manifest, k)
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        let path = keyframe_path(project, k);
        if path.exists() {
            if let Ok(existing) = load_fmap(&path) {
                if map_fingerprint(&existing.params) == map_fingerprint(&params) {
                    skipped += 1;
                    continue;
                }
            }
            println!("[{k}/{n}] keyframe_{k:05}.fmap invalide/obsolète → re-rendu");
        }
        let t0 = std::time::Instant::now();
        let mut orbit_cache = None; // single-shot : pas de réutilisation inter-échelle
        let Some((iterations, zs, _orbits, distances)) = render_escape_time_cancellable_with_reuse(
            &params, &cancel, None, &mut orbit_cache, None, None,
        ) else {
            return Err("rendu annulé".into());
        };
        let map = FractalMap {
            params,
            iterations,
            zs,
            distances: (!distances.is_empty()).then_some(distances),
        };
        save_fmap(&map, &path)?;
        rendered += 1;
        println!(
            "[{k}/{n}] keyframe_{k:05}.fmap rendue en {:.2}s",
            t0.elapsed().as_secs_f64()
        );
    }
    Ok((rendered, skipped))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::render_escape_time;

    fn tmp_project(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fractall_video_{}_{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Verrou jalon 2 : `keyframe_count` = ceil(log2(zoom)), exact aussi sur
    /// les puissances de 2.
    #[test]
    fn keyframe_count_is_ceil_log2() {
        assert_eq!(keyframe_count("2").unwrap(), 1);
        assert_eq!(keyframe_count("4").unwrap(), 2); // puissance de 2 exacte
        assert_eq!(keyframe_count("5").unwrap(), 3); // log2(5)=2.32 → 3
        assert_eq!(keyframe_count("1024").unwrap(), 10);
        assert_eq!(keyframe_count("1e3").unwrap(), 10); // log2(1000)=9.97
        assert_eq!(keyframe_count("1e20").unwrap(), 67); // log2(1e20)=66.44
        assert_eq!(keyframe_count("1").unwrap(), 1); // minimum
        assert!(keyframe_count("abc").is_err());
        assert!(keyframe_count("-3").is_err());
    }

    /// Verrou jalon 2 : progression des spans EXACTE en HP — `span_x(k)`
    /// rechargé depuis la string HP == 4/2^k calculé en GMP, à k profond
    /// (au-delà du f64 : k=1200 → span ~1e-360, underflow f64).
    #[test]
    fn keyframe_spans_exact_hp_progression() {
        let mut m = Manifest::default();
        m.location.zoom = "1e400".into(); // profond : force le régime HP
        for k in [0u32, 1, 7, 53, 1200] {
            let p = keyframe_params(&m, k).unwrap();
            let prec = (k + 128).max(256);
            let reloaded = Float::with_val(
                prec,
                Float::parse(p.span_x_hp.as_ref().unwrap()).expect("span_x_hp parsable"),
            );
            let expected = Float::with_val(prec, 4.0) >> k;
            assert_eq!(reloaded, expected, "span_x(k={k}) doit être exactement 4/2^{k}");
        }
        // Underflow f64 assumé en très deep : la vérité est la string HP.
        let deep = keyframe_params(&m, 1200).unwrap();
        assert_eq!(deep.span_x, 0.0, "span f64 underflow attendu à k=1200");
    }

    /// Verrou jalon 2 : une keyframe rendue depuis le manifest == le rendu
    /// direct des mêmes coordonnées construites indépendamment (pixel-exact).
    #[test]
    fn keyframe_render_matches_direct_render() {
        let mut m = Manifest::default();
        m.image.width = 48;
        m.image.height = 36;
        m.fractal.iterations = 300;
        m.location.real = "-0.7436".into();
        m.location.imag = "0.1318".into();
        m.location.zoom = "1e3".into();
        let k = 6; // span = 4/64 = 0.0625, path f64 standard

        let kp = keyframe_params(&m, k).unwrap();
        let (it_kf, zs_kf) = render_escape_time(&kp);

        // Construction indépendante « à la CLI » : center f64 + span f64
        // directs (4/2^6 est exact en f64).
        let mut direct = default_params_for_type(FractalType::Mandelbrot, 48, 36);
        direct.center_x = -0.7436;
        direct.center_y = 0.1318;
        direct.span_x = 4.0 / 64.0;
        direct.span_y = direct.span_x * (36.0 / 48.0);
        direct.iteration_max = 300;
        direct.max_perturb_iterations = 300;
        direct.max_bla_steps = 300;
        let (it_direct, zs_direct) = render_escape_time(&direct);

        assert_eq!(it_kf, it_direct, "itérations keyframe == rendu direct");
        assert_eq!(zs_kf, zs_direct, "zs keyframe == rendu direct");
    }

    /// Verrou jalon 2 : reprise — un 2e `render_project` ne re-rend RIEN
    /// (toutes les maps skippées), et un changement de PALETTE n'invalide pas
    /// les maps (l'empreinte ignore la couleur).
    #[test]
    fn render_project_resume_skips_valid_maps() {
        let dir = tmp_project("resume");
        let mut m = Manifest::default();
        m.image.width = 16;
        m.image.height = 12;
        m.fractal.iterations = 100;
        m.location.zoom = "8".into(); // 3 segments → 4 maps
        m.video.keyframes = keyframe_count(&m.location.zoom).unwrap();
        m.save(&dir.join("manifest.toml")).unwrap();

        let (r1, s1) = render_project(&dir).unwrap();
        assert_eq!((r1, s1), (4, 0), "premier passage : tout rendu");
        let (r2, s2) = render_project(&dir).unwrap();
        assert_eq!((r2, s2), (0, 4), "second passage : tout skippé");

        // Changement de palette → maps toujours valides (fingerprint couleur-blind).
        m.color.palette = 3;
        m.save(&dir.join("manifest.toml")).unwrap();
        let (r3, s3) = render_project(&dir).unwrap();
        assert_eq!((r3, s3), (0, 4), "palette ≠ ⇒ maps réutilisées");

        // Changement d'itérations → invalide, re-rendu.
        m.fractal.iterations = 200;
        m.save(&dir.join("manifest.toml")).unwrap();
        let (r4, _s4) = render_project(&dir).unwrap();
        assert_eq!(r4, 4, "iterations ≠ ⇒ re-rendu");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `plan_project` : écrit un manifest complet (keyframes remplies) dans un
    /// dossier neuf, relisible tel quel.
    #[test]
    fn plan_writes_complete_manifest() {
        let dir = tmp_project("plan");
        let config = dir.join("config.toml");
        std::fs::write(
            &config,
            r#"
[location]
real = "-0.75"
imag = "0.01"
zoom = "1e6"

[image]
width = 320
height = 200
"#,
        )
        .unwrap();
        let project = dir.join("proj");
        let m = plan_project(&config, &project).unwrap();
        assert_eq!(m.video.keyframes, 20); // log2(1e6) = 19.93 → 20
        let reloaded = Manifest::load(&project.join("manifest.toml")).unwrap();
        assert_eq!(reloaded.video.keyframes, 20);
        assert_eq!(reloaded.image.width, 320);
        assert_eq!(reloaded.location.real, "-0.75");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
