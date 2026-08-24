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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rug::Float;
use serde::{Deserialize, Serialize};

use crate::fractal::{
    default_params_for_type, ColorSpace, FractalParams, FractalType, OutColoringMode,
};
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
    /// Seed Julia adopté depuis la source. `None` conserve le défaut du type
    /// pour la compatibilité avec les anciens manifests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub julia_re: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub julia_im: Option<f64>,
    /// Exposant Multibrot adopté depuis la source. `None` conserve le défaut
    /// historique (2.5) des anciens manifests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multibrot_power: Option<f64>,
}

impl Default for FractalSection {
    fn default() -> Self {
        Self {
            r#type: 3,
            iterations: 1000,
            iterations_growth: 0.0,
            distance_estimation: false,
            julia_re: None,
            julia_im: None,
            multibrot_power: None,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct ColorSection {
    pub palette: u8,
    pub color_repeat: u32,
    pub color_space: ColorSpace,
    pub outcoloring: String,
    /// Décalage cyclique de palette ∈ [0,1) (0 = neutre). Spline temporelle
    /// possible via `[dynamics] palette_offset` (jalon 4).
    pub palette_offset: f64,
}

impl Default for ColorSection {
    fn default() -> Self {
        Self {
            palette: 6,
            color_repeat: 40,
            color_space: ColorSpace::Rgb,
            outcoloring: "smooth".into(),
            palette_offset: 0.0,
        }
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
        let text = toml::to_string_pretty(self)?;
        crate::io::atomic::write_atomic(path, |file| {
            use std::io::Write as _;
            file.write_all(text.as_bytes())?;
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// Géométrie des keyframes
// ---------------------------------------------------------------------------

/// Limite opérationnelle généreuse (~10^30103 de zoom). Au-delà, même les
/// noms/coordonnées des keyframes et leur planification deviennent une entrée
/// déraisonnable avant tout rendu utile.
const MAX_KEYFRAMES: u32 = 100_000;

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
    let n = n.max(1);
    if n > MAX_KEYFRAMES as i64 {
        return Err(format!(
            "zoom trop profond: {n} keyframes (maximum {MAX_KEYFRAMES})"
        ));
    }
    Ok(n as u32)
}

/// Vérifie qu'un manifest de PROJET (déjà planifié) n'a pas un compte stocké
/// désynchronisé de son zoom. Les configs d'entrée non planifiées gardent 0
/// et passent d'abord par `plan_from_manifest`.
pub(crate) fn validate_project_keyframes(m: &Manifest) -> Result<u32, String> {
    let expected = keyframe_count(&m.location.zoom)?;
    if m.video.keyframes != expected {
        return Err(format!(
            "video.keyframes={} ne correspond pas à location.zoom (attendu {expected}) — relancez `fractall-video plan`",
            m.video.keyframes
        ));
    }
    Ok(expected)
}

/// Précision GMP pour l'arithmétique des spans à la keyframe `k` :
/// `-log2(span_k) + 96` bits de marge, plancher 256 (même règle que la GUI,
/// `hp_arith_precision`).
fn span_precision(k: u32) -> u32 {
    (k + 96).max(256)
}

/// Dimensions réelles des maps après supersampling, validées sans wrap.
/// Tous les consommateurs du manifest doivent partager ce calcul.
pub(crate) fn render_dimensions(m: &Manifest) -> Result<(u32, u32), String> {
    if m.image.width == 0 || m.image.height == 0 {
        return Err("image.width/height doivent être > 0".into());
    }
    let ss = m.image.supersample.max(1);
    let w = m
        .image
        .width
        .checked_mul(ss)
        .ok_or_else(|| "image.width × supersample déborde u32".to_string())?;
    let h = m
        .image
        .height
        .checked_mul(ss)
        .ok_or_else(|| "image.height × supersample déborde u32".to_string())?;
    let pixels = w as u64 * h as u64;
    if pixels > usize::MAX as u64 / 16 {
        return Err("dimensions trop grandes pour les buffers de rendu".into());
    }
    Ok((w, h))
}

/// Paramètres COMPLETS de la keyframe `k` (0 = vue pleine, `k` = span/2^k).
/// Centre fixe (= la cible), spans en progression ×2 exacte : `span_x(k) =
/// 4/2^k` est une puissance de 2, donc **exacte en GMP à toute profondeur**.
pub fn keyframe_params(m: &Manifest, k: u32) -> Result<FractalParams, String> {
    let ftype = FractalType::from_id(m.fractal.r#type)
        .ok_or_else(|| format!("type de fractale invalide: {}", m.fractal.r#type))?;
    let (w, h) = render_dimensions(m)?;
    let mut p = default_params_for_type(ftype, w, h);
    match (m.fractal.julia_re, m.fractal.julia_im) {
        (Some(re), Some(im)) if re.is_finite() && im.is_finite() => {
            p.seed = num_complex::Complex64::new(re, im);
        }
        (None, None) => {}
        _ => return Err("fractal.julia_re/julia_im doivent être deux nombres finis".into()),
    }
    if let Some(power) = m.fractal.multibrot_power {
        if !power.is_finite() || power <= 0.0 {
            return Err(format!("fractal.multibrot_power invalide: {power}"));
        }
        p.multibrot_power = power;
    }

    let prec = span_precision(k);
    // Centre HP (strings du manifest, vérité absolue) + approximation f64.
    let cx = Float::parse(&m.location.real)
        .map(|v| Float::with_val(prec, v))
        .map_err(|e| format!("location.real illisible: {e}"))?;
    let cy = Float::parse(&m.location.imag)
        .map(|v| Float::with_val(prec, v))
        .map_err(|e| format!("location.imag illisible: {e}"))?;
    if !cx.is_finite() || !cy.is_finite() {
        return Err("location.real/imag doivent être finis".into());
    }
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
    p.color_space = m.color.color_space;
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
    p.color_space = ColorSpace::Rgb;
    p.color_offset = 0.0;
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
    plan_from_manifest(&Manifest::load(config)?, project)
}

/// Comme `plan_project`, mais depuis un `Manifest` EN MÉMOIRE : c'est le
/// chemin du studio GUI (G12 jalon 6), qui construit le manifest lui-même —
/// l'utilisateur n'édite aucun fichier. Remplit `video.keyframes`, valide la
/// géométrie tôt, crée le dossier et écrit `project/manifest.toml`.
pub fn plan_from_manifest(m: &Manifest, project: &Path) -> Result<Manifest, Box<dyn std::error::Error>> {
    let mut m = m.clone();
    m.video.keyframes = keyframe_count(&m.location.zoom)?;
    if !m.fractal.iterations_growth.is_finite() || m.fractal.iterations_growth < 0.0 {
        return Err("fractal.iterations_growth doit être fini et ≥ 0".into());
    }
    // Valide la géométrie tôt (types/outcoloring invalides = erreur au plan,
    // pas au 30e keyframe du render).
    keyframe_params(&m, 0).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    let oc_mode = assemble::video_outcoloring(&m.color.outcoloring)
        .map_err(|e| -> Box<dyn std::error::Error> { format!("color.outcoloring: {e}").into() })?;
    // G5 : un mode Distance* exige le canal `distances` dans les .fmap, donc
    // `[fractal] distance_estimation = true` — refusé au plan, pas au 30e
    // keyframe (ni, pire, à l'assemblage en Smooth silencieux).
    if oc_mode.requires_distance_channel() && !m.fractal.distance_estimation {
        return Err(format!(
            "color.outcoloring '{}' requiert [fractal] distance_estimation = true              (le canal distances doit être rendu dans les .fmap)",
            m.color.outcoloring
        )
        .into());
    }
    let velocity = spline::Dynamic::parse(&m.video.velocity)
        .map_err(|e| format!("video.velocity: {e}"))?;
    assemble::timeline_sample_count(m.video.keyframes, m.video.fps, &velocity)
        .map_err(|e| format!("video.velocity: {e}"))?;
    if let Some(offset) = &m.dynamics.palette_offset {
        spline::Dynamic::parse(offset)
            .map_err(|e| format!("dynamics.palette_offset: {e}"))?;
    }
    if !m.color.palette_offset.is_finite() {
        return Err("color.palette_offset doit être fini".into());
    }
    if !m.lighting.alpha.is_finite() || !m.lighting.beta.is_finite() {
        return Err("lighting.alpha/beta doivent être finis".into());
    }
    std::fs::create_dir_all(project)?;
    m.save(&project.join("manifest.toml"))?;
    Ok(m)
}

/// Événement de progression du rendu des keyframes (k ∈ 0..=n).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KeyframeEvent {
    /// Rendu de la keyframe `k` démarré.
    Started { k: u32, n: u32 },
    /// Map écrite (durée du rendu en secondes).
    Rendered { k: u32, n: u32, seconds: f64 },
    /// Reprise : map existante valide, réutilisée telle quelle.
    Skipped { k: u32, n: u32 },
    /// Map existante obsolète (empreinte ≠) → re-rendu (un `Started` suit).
    Invalidated { k: u32, n: u32 },
}

/// Issue d'un rendu annulable. L'annulation n'est PAS une erreur : les maps
/// déjà écrites restent valides (reprise par empreinte couleur-blind).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderOutcome {
    Complete { rendered: usize, skipped: usize },
    Cancelled { rendered: usize, skipped: usize },
}

/// `render` : calcule les keyframes 0..=keyframes manquantes. Une map
/// existante dont l'empreinte (params hors couleur) correspond est SKIPPÉE —
/// c'est la reprise après interruption. Retourne (rendues, skippées).
///
/// Enveloppe no-cancel de `render_project_with_progress` reproduisant la
/// sortie console historique (utilisée par le CLI `fractall-video render`).
pub fn render_project(project: &Path) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let outcome = render_project_with_progress(
        project,
        &Arc::new(AtomicBool::new(false)),
        &mut |ev| match ev {
            KeyframeEvent::Invalidated { k, n } => {
                println!("[{k}/{n}] keyframe_{k:05}.fmap invalide/obsolète → re-rendu");
            }
            KeyframeEvent::Rendered { k, n, seconds } => {
                println!("[{k}/{n}] keyframe_{k:05}.fmap rendue en {seconds:.2}s");
            }
            KeyframeEvent::Started { .. } | KeyframeEvent::Skipped { .. } => {}
        },
    )?;
    match outcome {
        RenderOutcome::Complete { rendered, skipped } => Ok((rendered, skipped)),
        // Inatteignable sans cancel externe — sémantique historique conservée.
        RenderOutcome::Cancelled { .. } => Err("rendu annulé".into()),
    }
}

/// Ordre de rendu des keyframes. Les keyframes étant INDÉPENDANTES (pas de
/// réutilisation d'orbite inter-échelle, cf. doc de module) et la reprise
/// par empreinte insensible à l'ordre, le choix est libre — il ne change ni
/// les maps produites ni la sémantique de reprise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderOrder {
    /// 0, 1, …, n (comportement CLI historique).
    Sequential,
    /// 0, n, puis milieux récursifs (largeur d'abord) : la timeline du studio
    /// se peuple à TOUTES les profondeurs dès les premières maps — une cible
    /// décentrée ou une palette ratée se voit en minutes, pas en heures.
    Bisection,
}

/// Ordre dichotomique : 0, n, puis les milieux de chaque intervalle en
/// largeur d'abord. Permutation exacte de 0..=n.
pub fn bisection_order(n: u32) -> Vec<u32> {
    let mut out = vec![0u32];
    if n > 0 {
        out.push(n);
    }
    let mut queue = std::collections::VecDeque::from([(0u32, n)]);
    while let Some((a, b)) = queue.pop_front() {
        if b - a >= 2 {
            let m = a + (b - a) / 2;
            out.push(m);
            queue.push_back((a, m));
            queue.push_back((m, b));
        }
    }
    out
}

/// Version annulable + progression (studio GUI, G12 jalon 6). Le `cancel` est
/// vérifié en tête de boucle ET passé au dispatcher (annulation mi-keyframe
/// réactive : la map en cours est perdue, les précédentes restent — la
/// reprise repartira exactement là).
pub fn render_project_with_progress(
    project: &Path,
    cancel: &Arc<AtomicBool>,
    progress: &mut dyn FnMut(KeyframeEvent),
) -> Result<RenderOutcome, Box<dyn std::error::Error>> {
    render_project_with_progress_ordered(project, RenderOrder::Sequential, cancel, progress)
}

/// Comme `render_project_with_progress`, avec choix de l'ordre de rendu
/// (G13 : le studio passe `Bisection` pour peupler sa timeline à toutes les
/// profondeurs au plus tôt).
pub fn render_project_with_progress_ordered(
    project: &Path,
    order: RenderOrder,
    cancel: &Arc<AtomicBool>,
    progress: &mut dyn FnMut(KeyframeEvent),
) -> Result<RenderOutcome, Box<dyn std::error::Error>> {
    let manifest = Manifest::load(&project.join("manifest.toml"))?;
    let n = validate_project_keyframes(&manifest)
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    let (mut rendered, mut skipped) = (0usize, 0usize);
    let ks: Vec<u32> = match order {
        RenderOrder::Sequential => (0..=n).collect(),
        RenderOrder::Bisection => bisection_order(n),
    };

    for k in ks {
        if cancel.load(Ordering::Relaxed) {
            return Ok(RenderOutcome::Cancelled { rendered, skipped });
        }
        let params = keyframe_params(&manifest, k)
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        let path = keyframe_path(project, k);
        if path.exists() {
            if let Ok(existing) = load_fmap(&path) {
                if map_fingerprint(&existing.params) == map_fingerprint(&params) {
                    skipped += 1;
                    progress(KeyframeEvent::Skipped { k, n });
                    continue;
                }
            }
            progress(KeyframeEvent::Invalidated { k, n });
        }
        progress(KeyframeEvent::Started { k, n });
        let t0 = std::time::Instant::now();
        let mut orbit_cache = None; // single-shot : pas de réutilisation inter-échelle
        let Some(out) = render_escape_time_cancellable_with_reuse(
            &params, cancel, None, &mut orbit_cache, None, None,
        ) else {
            return Ok(RenderOutcome::Cancelled { rendered, skipped });
        };
        let map = FractalMap {
            params,
            iterations: out.iterations,
            zs: out.zs,
            distances: (!out.distances.is_empty()).then_some(out.distances),
        };
        save_fmap(&map, &path)?;
        rendered += 1;
        progress(KeyframeEvent::Rendered { k, n, seconds: t0.elapsed().as_secs_f64() });
    }
    Ok(RenderOutcome::Complete { rendered, skipped })
}

/// Magnification manifest depuis un span_x HP : `zoom = 4/span_x` (convention
/// `SPAN_AT_ZOOM_1`). ⚠️ PAS le « Zoom » de la barre d'état de fractall-gui
/// (qui est `4·width/span`, par pixel). Calcul rug à précision
/// `-log2(span) + 96` bits (plancher 256), sérialisation `to_string_radix`.
pub fn zoom_from_span_x(span_x_hp: &str) -> Result<String, String> {
    let probe = Float::parse(span_x_hp)
        .map(|p| Float::with_val(128, p))
        .map_err(|e| format!("span illisible '{span_x_hp}': {e}"))?;
    if !probe.is_finite() || probe <= 0.0 {
        return Err(format!("span invalide '{span_x_hp}' (doit être fini et > 0)"));
    }
    let exp = probe.get_exp().unwrap_or(0) as i64;
    let prec = (((-exp).max(0) as u32).saturating_add(96)).max(256);
    let span = Float::with_val(prec, Float::parse(span_x_hp).expect("déjà parsé"));
    let zoom = Float::with_val(prec, SPAN_AT_ZOOM_1) / span;
    Ok(zoom.to_string_radix(10, None))
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

    /// Verrou bug 2026-08-23 : OrbitTraps/Wings (canal orbites non persisté
    /// dans les .fmap) sont REFUSÉS au plan au lieu de retomber en Smooth.
    #[test]
    fn plan_rejects_orbit_based_outcoloring() {
        for mode in ["orbittraps", "wings"] {
            assert!(assemble::video_outcoloring(mode).is_err(), "{mode}");
        }
        assert!(assemble::video_outcoloring("smooth").is_ok());
        assert!(assemble::video_outcoloring("distance").is_ok());
        assert!(assemble::video_outcoloring("nope").is_err());
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
        assert!(keyframe_count("1e40000").is_err(), "profondeur absurde bornée");
    }

    #[test]
    fn project_rejects_keyframe_count_desynchronized_from_zoom() {
        let dir = tmp_project("badcount");
        let mut m = Manifest::default();
        m.location.zoom = "8".into();
        m.video.keyframes = u32::MAX;
        m.save(&dir.join("manifest.toml")).unwrap();
        let err = render_project(&dir).unwrap_err().to_string();
        assert!(err.contains("attendu 3"), "erreur explicite: {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// G5 : un manifest `outcoloring = "distance"` sans
    /// `[fractal] distance_estimation = true` est refusé AU PLAN — les .fmap
    /// n'auraient pas le canal et l'assemblage retomberait sur Smooth.
    #[test]
    fn plan_rejects_distance_mode_without_distance_channel() {
        let dir = tmp_project("distgate");
        let mut m = Manifest::default();
        m.location.zoom = "8".into();
        m.color.outcoloring = "distance".into();
        m.fractal.distance_estimation = false;
        let err = plan_from_manifest(&m, &dir).unwrap_err().to_string();
        assert!(
            err.contains("distance_estimation"),
            "erreur explicite au plan: {err}"
        );
        m.fractal.distance_estimation = true;
        plan_from_manifest(&m, &dir).expect("avec le canal, le plan passe");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn render_dimensions_reject_zero_and_supersample_overflow() {
        let mut m = Manifest::default();
        m.image.width = 0;
        assert!(render_dimensions(&m).is_err());
        m.image.width = u32::MAX;
        m.image.height = 1;
        m.image.supersample = 2;
        assert!(render_dimensions(&m).is_err());
        m.image.width = 3840;
        m.image.height = 2160;
        m.image.supersample = 3;
        assert_eq!(render_dimensions(&m).unwrap(), (11_520, 6480));
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

    /// Les paramètres propres aux types exposés par le studio doivent
    /// traverser le manifest. Leur absence garde les défauts historiques des
    /// anciens projets.
    #[test]
    fn keyframe_preserves_julia_seed_and_multibrot_power() {
        let mut julia = Manifest::default();
        julia.fractal.r#type = 4;
        let historical = keyframe_params(&julia, 0).unwrap();
        assert_eq!(
            historical.seed,
            default_params_for_type(FractalType::Julia, 1, 1).seed,
            "ancien manifest sans seed"
        );
        julia.fractal.julia_re = Some(-0.745);
        julia.fractal.julia_im = Some(0.113);
        julia.color.color_space = ColorSpace::Lch;
        assert_eq!(
            keyframe_params(&julia, 0).unwrap().seed,
            num_complex::Complex64::new(-0.745, 0.113)
        );
        assert_eq!(keyframe_params(&julia, 0).unwrap().color_space, ColorSpace::Lch);

        let mut multi = Manifest::default();
        multi.fractal.r#type = 23;
        multi.fractal.multibrot_power = Some(3.75);
        assert_eq!(keyframe_params(&multi, 0).unwrap().multibrot_power, 3.75);
        multi.fractal.multibrot_power = Some(0.0);
        assert!(keyframe_params(&multi, 0).is_err());
        julia.fractal.julia_im = None;
        assert!(keyframe_params(&julia, 0).is_err(), "seed partiel refusé");
    }

    #[test]
    fn keyframe_rejects_non_finite_center() {
        for center in ["NaN", "inf", "-inf"] {
            let mut m = Manifest::default();
            m.location.real = center.into();
            assert!(keyframe_params(&m, 0).is_err(), "centre {center} refusé");
        }
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
        let kf = render_escape_time(&kp);
        let (it_kf, zs_kf) = (kf.iterations, kf.zs);

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
        let d = render_escape_time(&direct);
        let (it_direct, zs_direct) = (d.iterations, d.zs);

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

        // Changements de colorisation → maps toujours valides
        // (fingerprint couleur-blind).
        m.color.palette = 3;
        m.color.color_space = ColorSpace::Lch;
        m.save(&dir.join("manifest.toml")).unwrap();
        let (r3, s3) = render_project(&dir).unwrap();
        assert_eq!((r3, s3), (0, 4), "couleurs ≠ ⇒ maps réutilisées");

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

    fn progress_manifest(dir: &Path) -> Manifest {
        let mut m = Manifest::default();
        m.image.width = 16;
        m.image.height = 12;
        m.fractal.iterations = 80;
        m.location.zoom = "8".into(); // 3 segments → 4 maps
        m.video.keyframes = keyframe_count(&m.location.zoom).unwrap();
        m.save(&dir.join("manifest.toml")).unwrap();
        m
    }

    /// Verrou hooks (G12 jalon 6) : chaque keyframe rapporte Started puis
    /// Rendered dans l'ordre ; un second run rapporte n+1 Skipped.
    #[test]
    fn render_with_progress_reports_each_keyframe() {
        let dir = tmp_project("progress");
        progress_manifest(&dir);

        let mut events = Vec::new();
        let outcome = render_project_with_progress(
            &dir,
            &Arc::new(AtomicBool::new(false)),
            &mut |ev| events.push(ev),
        )
        .unwrap();
        assert_eq!(outcome, RenderOutcome::Complete { rendered: 4, skipped: 0 });
        let started: Vec<u32> = events
            .iter()
            .filter_map(|e| match e {
                KeyframeEvent::Started { k, .. } => Some(*k),
                _ => None,
            })
            .collect();
        let rendered: Vec<u32> = events
            .iter()
            .filter_map(|e| match e {
                KeyframeEvent::Rendered { k, .. } => Some(*k),
                _ => None,
            })
            .collect();
        assert_eq!(started, vec![0, 1, 2, 3]);
        assert_eq!(rendered, vec![0, 1, 2, 3]);

        let mut events2 = Vec::new();
        let outcome2 = render_project_with_progress(
            &dir,
            &Arc::new(AtomicBool::new(false)),
            &mut |ev| events2.push(ev),
        )
        .unwrap();
        assert_eq!(outcome2, RenderOutcome::Complete { rendered: 0, skipped: 4 });
        assert_eq!(events2.len(), 4);
        assert!(events2.iter().all(|e| matches!(e, KeyframeEvent::Skipped { .. })));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Verrou annulation + reprise : cancel après la 2e map → Cancelled{2,0},
    /// maps 0-1 présentes, 2-3 absentes ; relance sans cancel →
    /// Complete{rendered:2, skipped:2} — l'annulation n'a RIEN perdu.
    #[test]
    fn render_with_progress_cancel_then_resume() {
        let dir = tmp_project("cancel_resume");
        progress_manifest(&dir);

        let cancel = Arc::new(AtomicBool::new(false));
        let trigger = cancel.clone();
        let mut rendered_seen = 0u32;
        let outcome = render_project_with_progress(&dir, &cancel, &mut |ev| {
            if matches!(ev, KeyframeEvent::Rendered { .. }) {
                rendered_seen += 1;
                if rendered_seen == 2 {
                    trigger.store(true, Ordering::Relaxed);
                }
            }
        })
        .unwrap();
        assert_eq!(outcome, RenderOutcome::Cancelled { rendered: 2, skipped: 0 });
        assert!(keyframe_path(&dir, 0).exists());
        assert!(keyframe_path(&dir, 1).exists());
        assert!(!keyframe_path(&dir, 2).exists());

        let outcome2 =
            render_project_with_progress(&dir, &Arc::new(AtomicBool::new(false)), &mut |_| {})
                .unwrap();
        assert_eq!(outcome2, RenderOutcome::Complete { rendered: 2, skipped: 2 });
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Verrou `zoom_from_span_x` : inverse exact de la progression des spans,
    /// y compris en régime deep au-delà du f64 (k=1200 → span ~1e-360).
    #[test]
    fn zoom_from_span_x_roundtrip() {
        // span 4 → zoom 1 (comparaison en VALEUR GMP, pas en string).
        let z = zoom_from_span_x("4").unwrap();
        let zf = Float::with_val(128, Float::parse(&z).unwrap());
        assert_eq!(zf, 1);
        assert_eq!(keyframe_count(&z).unwrap(), 1); // clamp minimum

        let mut m = Manifest::default();
        m.location.zoom = "1e400".into();
        for k in [10u32, 100, 1200] {
            let p = keyframe_params(&m, k).unwrap();
            let zoom = zoom_from_span_x(p.span_x_hp.as_ref().unwrap()).unwrap();
            assert_eq!(keyframe_count(&zoom).unwrap(), k, "round-trip k={k}");
        }

        assert!(zoom_from_span_x("0").is_err());
        assert!(zoom_from_span_x("-1").is_err());
        assert!(zoom_from_span_x("abc").is_err());
    }

    /// `bisection_order` : permutation exacte de 0..=n, extrémités d'abord,
    /// milieu ensuite — l'ordre qui peuple la timeline à toutes les échelles.
    #[test]
    fn bisection_order_is_a_permutation_extremes_first() {
        for n in [0u32, 1, 2, 3, 7, 8, 33] {
            let order = bisection_order(n);
            let mut sorted = order.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..=n).collect::<Vec<_>>(), "permutation n={n}");
            assert_eq!(order[0], 0);
            if n > 0 {
                assert_eq!(order[1], n);
            }
            if n >= 2 {
                assert_eq!(order[2], n / 2, "le milieu vient en 3e (n={n})");
            }
        }
    }

    /// L'ordre Bisection produit LES MÊMES maps que Sequential (keyframes
    /// indépendantes) : mêmes fichiers, reprise complète au 2e passage.
    #[test]
    fn bisection_render_produces_same_maps_as_sequential() {
        let dir = tmp_project("bisect");
        progress_manifest(&dir);
        let mut started: Vec<u32> = Vec::new();
        let outcome = render_project_with_progress_ordered(
            &dir,
            RenderOrder::Bisection,
            &Arc::new(AtomicBool::new(false)),
            &mut |ev| {
                if let KeyframeEvent::Started { k, .. } = ev {
                    started.push(k);
                }
            },
        )
        .unwrap();
        assert_eq!(outcome, RenderOutcome::Complete { rendered: 4, skipped: 0 });
        assert_eq!(started, bisection_order(3), "ordre de rendu = dichotomie");
        // Un passage séquentiel derrière skippe tout : maps identiques valides.
        let outcome2 =
            render_project_with_progress(&dir, &Arc::new(AtomicBool::new(false)), &mut |_| {})
                .unwrap();
        assert_eq!(outcome2, RenderOutcome::Complete { rendered: 0, skipped: 4 });
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Verrou `plan_from_manifest` (chemin studio GUI, sans fichier config).
    #[test]
    fn plan_from_manifest_fills_keyframes() {
        let dir = tmp_project("planmem");
        let mut m = Manifest::default();
        m.location.zoom = "1e6".into();
        let planned = plan_from_manifest(&m, &dir).unwrap();
        assert_eq!(planned.video.keyframes, 20);
        let reloaded = Manifest::load(&dir.join("manifest.toml")).unwrap();
        assert_eq!(reloaded.video.keyframes, 20);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn plan_rejects_invalid_temporal_and_lighting_values_before_write() {
        for (tag, mutate) in [
            ("velocity", 0u8),
            ("palette", 1u8),
            ("lighting", 2u8),
            ("huge", 3u8),
            ("growth", 4u8),
        ] {
            let dir = tmp_project(tag);
            let mut m = Manifest::default();
            match mutate {
                0 => m.video.velocity = "0/1,inf/1".into(),
                1 => m.dynamics.palette_offset = Some("-1/0,2/1".into()),
                2 => m.lighting.beta = f64::INFINITY,
                3 => m.video.velocity = "0.000000001".into(),
                4 => m.fractal.iterations_growth = f64::NAN,
                _ => unreachable!(),
            }
            assert!(plan_from_manifest(&m, &dir).is_err(), "devait refuser {tag}");
            assert!(!dir.join("manifest.toml").exists(), "aucun manifest écrit pour {tag}");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
