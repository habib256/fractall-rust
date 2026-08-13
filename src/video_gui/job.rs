//! Worker du studio vidéo (G12 jalon 6) — zéro egui, testable.
//!
//! Construit le `Manifest` depuis l'état du studio (l'utilisateur n'édite
//! aucun fichier), lance plan → render → assemble dans UN thread de job, et
//! remonte la progression par messages mpsc (pattern `HqRenderMessage` du
//! générateur). Les erreurs traversent le canal en `String` (`Box<dyn Error>`
//! n'est pas `Send`) ; l'annulation passe par l'`Arc<AtomicBool>` partagé et
//! est ACQUITTÉE par `VideoJobMsg::Cancelled` (maps conservées — reprise).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::AtomicBool;
use std::sync::{mpsc, Arc};

use num_complex::Complex64;

use crate::fractal::FractalParams;
use crate::io::fmap::load_fmap;
use crate::video::assemble::{
    assemble_project_with_progress, colorize_keyframe, interpolate_frame, timeline,
    AssembleOptions, AssembleOutcome,
};
use crate::video::spline::Dynamic;
use crate::video::{
    self, bisection_order, keyframe_params, keyframe_path, map_fingerprint, plan_from_manifest,
    KeyframeEvent, Manifest, RenderOrder, RenderOutcome,
};

use super::timeline::{thumb_channels, SpeedCurve, THUMB_MAX_H};

/// Messages du job vidéo vers l'UI.
#[derive(Clone, Debug, PartialEq)]
pub enum VideoJobMsg {
    /// Rendu de la keyframe `k` démarré (surbrillance timeline).
    RenderStarted { k: u32 },
    /// Map complétée (rendue OU réutilisée) sur total = keyframes + 1.
    /// `seconds` = durée du rendu (None si map skippée), consommé par l'ETA.
    RenderProgress { done: u32, total: u32, k: u32, seconds: Option<f64> },
    /// Miniature de la keyframe `k` : canaux bruts sous-échantillonnés
    /// (l'UI colorise — recolorisation gratuite au changement de palette).
    Thumb {
        k: u32,
        w: u32,
        h: u32,
        iter_max: u32,
        iterations: Vec<u32>,
        zs: Vec<Complex64>,
        provisional: bool,
    },
    /// Frames assemblées (1-based) sur total.
    AssembleProgress { frame: usize, total: usize },
    /// Chemin du mp4 ou du dossier de frames.
    Done { output: String },
    /// Annulation acquittée (maps conservées, relancer reprendra).
    Cancelled,
    Error(String),
}

/// Cible du zoom, extraite de la preview par l'appelant.
#[derive(Clone, Debug)]
pub struct TargetView {
    pub real: String,
    pub imag: String,
    /// Magnification manifest (= `video::zoom_from_span_x(span de la vue)`).
    pub zoom: String,
    pub type_id: u8,
    pub palette: u8,
    pub color_repeat: u32,
    pub outcoloring: String,
}

/// Réglages du panneau (données pures, sans egui).
#[derive(Clone, Debug)]
pub struct StudioSettings {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    /// Keyframes (×2) par seconde, > 0 — vitesse de BASE, modulée par la
    /// courbe `speed_points` (G13).
    pub velocity: f64,
    /// Courbe de vitesse par zone : points (position keyframe, multiplicateur)
    /// édités sur la timeline, compilés en spline par `compiled_velocity`.
    /// Vide = vitesse constante (chemin exact historique).
    pub speed_points: Vec<(f64, f64)>,
    pub supersample: u32,
    pub iterations: u32,
    pub iterations_growth: f64,
    pub lighting: bool,
    pub lighting_beta: f64,
    pub palette_scroll: bool,
    /// Cycles de palette sur toute la durée de la vidéo.
    pub palette_cycles: f64,
}

impl Default for StudioSettings {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            fps: 30,
            velocity: 1.0,
            speed_points: Vec::new(),
            supersample: 1,
            iterations: 1000,
            iterations_growth: 50.0,
            lighting: false,
            lighting_beta: 45.0,
            palette_scroll: false,
            palette_cycles: 1.0,
        }
    }
}

/// Vitesse effective du manifest : courbe de zone compilée sur la vitesse de
/// base pour `n` segments keyframe. Courbe vide/plate → constante EXACTE
/// (même string que l'historique `format!("{velocity}")`).
pub fn compiled_velocity(s: &StudioSettings, n: u32) -> String {
    SpeedCurve { points: s.speed_points.clone() }.compile(s.velocity, n)
}

/// Dimensions PAIRES (l'encodage x264 yuv420p échoue sur une dimension
/// impaire) : arrondi pair inférieur, plancher 16.
pub fn even_dims(w: u32, h: u32) -> (u32, u32) {
    let f = |x: u32| x.max(16) & !1;
    (f(w), f(h))
}

/// Dossier de sortie par défaut : `fractall_video_<unix_ts>` dans le dossier
/// courant (même convention que les `fractal_<ts>.png` du générateur).
pub fn default_output_dir() -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("fractall_video_{ts}")
}

/// Rampe linéaire de défilement de palette sur la durée de la vidéo :
/// spline 2 nœuds `"0/0,<durée>/<cycles>"` (format `video/spline.rs`).
pub fn palette_scroll_spline(duration_s: f64, cycles: f64) -> String {
    format!("0/0,{duration_s}/{cycles}")
}

/// ffmpeg présent ? Sondé UNE fois par session studio (jamais dans update()).
pub fn detect_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Estimations affichées LIVE dans le panneau.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimates {
    pub keyframes: u32,
    /// Maps à rendre = keyframes + 1 (bornes des segments).
    pub maps: u32,
    pub duration_s: f64,
    pub frames: usize,
    /// Octets BRUTS des maps (avant compression zlib) : iterations u32 +
    /// zs 2×f64 par pixel supersamplé.
    pub map_bytes: u64,
}

/// Sources de vérité partagées : `keyframe_count` pour n, `timeline` pour le
/// nombre de frames (aucune formule dupliquée). `velocity` = la valeur
/// manifest (constante `"1.0"` OU spline compilée par `compiled_velocity`) —
/// les estimations reflètent donc la courbe de vitesse éditée.
pub fn estimates(
    zoom: &str,
    fps: u32,
    velocity: &str,
    w: u32,
    h: u32,
    ss: u32,
) -> Result<Estimates, String> {
    let n = video::keyframe_count(zoom)?;
    let velocity = Dynamic::parse(velocity)?;
    let fps = fps.max(1);
    let positions = timeline(n, fps, &velocity)?;
    let frames = positions.len();
    let ss = ss.max(1) as u64;
    let px = w as u64 * ss * h as u64 * ss;
    Ok(Estimates {
        keyframes: n,
        maps: n + 1,
        duration_s: frames as f64 / fps as f64,
        frames,
        map_bytes: (n as u64 + 1) * px * (4 + 16),
    })
}

/// Construit le Manifest complet depuis l'état du studio. `video.keyframes`
/// reste à 0 : rempli par `plan_from_manifest` au lancement du job.
pub fn build_manifest(s: &StudioSettings, t: &TargetView) -> Manifest {
    let (w, h) = even_dims(s.width, s.height);
    let n = video::keyframe_count(&t.zoom).unwrap_or(1);
    let velocity = compiled_velocity(s, n);
    let mut m = Manifest::default();
    m.location.real = t.real.clone();
    m.location.imag = t.imag.clone();
    m.location.zoom = t.zoom.clone();
    m.image.width = w;
    m.image.height = h;
    m.image.supersample = s.supersample.max(1);
    m.fractal.r#type = t.type_id;
    m.fractal.iterations = s.iterations.max(1);
    m.fractal.iterations_growth = s.iterations_growth.max(0.0);
    m.color.palette = t.palette;
    m.color.color_repeat = t.color_repeat.max(1);
    m.color.outcoloring = t.outcoloring.clone();
    m.video.fps = s.fps.clamp(1, 120);
    m.video.velocity = velocity.clone();
    m.lighting.enable = s.lighting;
    m.lighting.beta = s.lighting_beta;
    if s.palette_scroll && s.palette_cycles != 0.0 {
        if let Ok(est) = estimates(&t.zoom, m.video.fps, &velocity, w, h, 1) {
            m.dynamics.palette_offset =
                Some(palette_scroll_spline(est.duration_s, s.palette_cycles));
        }
    }
    m
}

fn run_assemble(
    m: &Manifest,
    project: &PathBuf,
    use_ffmpeg: bool,
    cancel: &Arc<AtomicBool>,
    tx: &mpsc::Sender<VideoJobMsg>,
) {
    let opts = AssembleOptions {
        output: use_ffmpeg.then(|| project.join("video.mp4")),
        frames_dir: (!use_ffmpeg).then(|| project.join("frames")),
        ffmpeg: "ffmpeg".into(),
    };
    let _ = m; // le manifest est relu depuis le disque par assemble (source unique)
    match assemble_project_with_progress(project, &opts, cancel, &mut |ev| {
        let _ = tx.send(VideoJobMsg::AssembleProgress { frame: ev.frame + 1, total: ev.total });
    }) {
        Err(e) => {
            let _ = tx.send(VideoJobMsg::Error(e.to_string()));
        }
        Ok(AssembleOutcome::Cancelled { .. }) => {
            let _ = tx.send(VideoJobMsg::Cancelled);
        }
        Ok(AssembleOutcome::Complete(_)) => {
            let output = if use_ffmpeg { project.join("video.mp4") } else { project.join("frames") };
            let _ = tx.send(VideoJobMsg::Done { output: output.display().to_string() });
        }
    }
}

/// Charge la map de la keyframe `k` et émet sa miniature (canaux
/// sous-échantillonnés). Silencieux si la map est illisible (mi-écriture,
/// annulation) : la miniature arrivera au prochain passage.
fn emit_thumb(project: &Path, k: u32, tx: &mpsc::Sender<VideoJobMsg>) {
    if let Ok(map) = load_fmap(&keyframe_path(project, k)) {
        let (w, h, iterations, zs) = thumb_channels(&map, THUMB_MAX_H);
        let _ = tx.send(VideoJobMsg::Thumb {
            k,
            w,
            h,
            iter_max: map.params.iteration_max,
            iterations,
            zs,
            provisional: false,
        });
    }
}

/// « Générer » : plan → render (progression par map) → assemble (progression
/// par frame) → Done. Reprise gratuite : les maps valides sont skippées.
/// Ordre DICHOTOMIQUE (G13) : la timeline se peuple à toutes les profondeurs
/// dès les premières maps (0, n, n/2, …).
pub fn spawn_generate(
    manifest: Manifest,
    project: PathBuf,
    use_ffmpeg: bool,
    cancel: Arc<AtomicBool>,
) -> mpsc::Receiver<VideoJobMsg> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let m = match plan_from_manifest(&manifest, &project) {
            Ok(m) => m,
            Err(e) => {
                let _ = tx.send(VideoJobMsg::Error(e.to_string()));
                return;
            }
        };
        let total = m.video.keyframes + 1;
        let mut done = 0u32;
        let render = video::render_project_with_progress_ordered(
            &project,
            RenderOrder::Bisection,
            &cancel,
            &mut |ev| match ev {
                KeyframeEvent::Started { k, .. } => {
                    let _ = tx.send(VideoJobMsg::RenderStarted { k });
                }
                KeyframeEvent::Rendered { k, seconds, .. } => {
                    done += 1;
                    let _ = tx.send(VideoJobMsg::RenderProgress {
                        done,
                        total,
                        k,
                        seconds: Some(seconds),
                    });
                    emit_thumb(&project, k, &tx);
                }
                KeyframeEvent::Skipped { k, .. } => {
                    done += 1;
                    let _ = tx.send(VideoJobMsg::RenderProgress { done, total, k, seconds: None });
                    emit_thumb(&project, k, &tx);
                }
                KeyframeEvent::Invalidated { .. } => {}
            },
        );
        match render {
            Err(e) => {
                let _ = tx.send(VideoJobMsg::Error(e.to_string()));
                return;
            }
            Ok(RenderOutcome::Cancelled { .. }) => {
                let _ = tx.send(VideoJobMsg::Cancelled);
                return;
            }
            Ok(RenderOutcome::Complete { .. }) => {}
        }
        run_assemble(&m, &project, use_ffmpeg, &cancel, &tx);
    });
    rx
}

/// Scan des maps EXISTANTES d'un projet (reprise / dossier adopté) : émet une
/// miniature par `.fmap` présent, en ordre dichotomique pour couvrir toutes
/// les profondeurs au plus tôt. Thread détaché, s'arrête de lui-même.
pub fn spawn_thumb_scan(project: PathBuf, n: u32) -> mpsc::Receiver<VideoJobMsg> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for k in bisection_order(n) {
            if keyframe_path(&project, k).exists() {
                emit_thumb(&project, k, &tx);
            }
        }
        // Boucle bornée : le thread se termine seul, même si le récepteur
        // est parti (les send échouent silencieusement).
    });
    rx
}

/// Rendu de la PREMIÈRE miniature (keyframe 0, vue pleine) avant tout calcul :
/// params fournis par l'appelant à la taille miniature, itérations plafonnées
/// (le détail deep-iter est invisible à 96 px). Émis `provisional: true` —
/// remplacé par la map réelle dès qu'elle est rendue.
pub fn spawn_first_thumb(params: FractalParams) -> mpsc::Receiver<VideoJobMsg> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let (iterations, zs) = crate::render::render_escape_time(&params);
        if iterations.len() == (params.width * params.height) as usize {
            let _ = tx.send(VideoJobMsg::Thumb {
                k: 0,
                w: params.width,
                h: params.height,
                iter_max: params.iteration_max,
                iterations,
                zs,
                provisional: true,
            });
        }
    });
    rx
}

// ---------------------------------------------------------------------------
// Scrubbing (G13) — frames interpolées par le MÊME code que l'assembleur
// ---------------------------------------------------------------------------

/// Demande de frame de scrub : position `p ∈ [0, n]` (keyframe fractionnaire),
/// taille de sortie, et version (les réponses périmées sont ignorées).
#[derive(Clone, Copy, Debug)]
pub struct ScrubRequest {
    pub p: f64,
    pub out_w: u32,
    pub out_h: u32,
    pub version: u64,
}

/// Réponse du worker de scrub.
pub enum ScrubReply {
    /// Frame interpolée (exactement ce que l'assembleur produira à cette
    /// position). `missing_next` : la keyframe k+1 n'est pas encore rendue —
    /// la frame est un zoom de k seule (pas de blend).
    Frame { rgb: Vec<u8>, w: u32, h: u32, version: u64, missing_next: bool },
    /// La keyframe porteuse n'existe pas encore.
    Missing { k: u32, version: u64 },
}

/// Worker de scrub : possède un petit cache de keyframes colorisées (LRU 4,
/// éviction de la plus éloignée) et répond à la DERNIÈRE demande reçue
/// (drain du canal = debounce naturel pendant un drag). Le thread s'arrête
/// quand l'émetteur est droppé (respawn au changement de palette/projet).
pub fn spawn_scrub_worker(
    project: PathBuf,
    manifest: Manifest,
) -> (mpsc::Sender<ScrubRequest>, mpsc::Receiver<ScrubReply>) {
    let (tx_req, rx_req) = mpsc::channel::<ScrubRequest>();
    let (tx_rep, rx_rep) = mpsc::channel();
    std::thread::spawn(move || {
        let ss = manifest.image.supersample.max(1);
        let (src_w, src_h) =
            ((manifest.image.width * ss) as usize, (manifest.image.height * ss) as usize);
        let n = manifest.video.keyframes;
        let offset = manifest.color.palette_offset;
        let mut cache: HashMap<u32, Arc<Vec<u8>>> = HashMap::new();

        // Colorise (ou ressort du cache) la keyframe k. None si la map est
        // absente/illisible ou de géométrie inattendue (projet obsolète).
        fn colorized(
            cache: &mut HashMap<u32, Arc<Vec<u8>>>,
            project: &Path,
            manifest: &Manifest,
            offset: f64,
            src_w: usize,
            src_h: usize,
            k: u32,
        ) -> Option<Arc<Vec<u8>>> {
            if let Some(c) = cache.get(&k) {
                return Some(c.clone());
            }
            let map = load_fmap(&keyframe_path(project, k)).ok()?;
            if map.params.width as usize != src_w || map.params.height as usize != src_h {
                return None;
            }
            let rgb = Arc::new(colorize_keyframe(&map, manifest, offset).ok()?);
            if cache.len() >= 4 {
                if let Some(&far) = cache.keys().max_by_key(|&&kk| kk.abs_diff(k)) {
                    cache.remove(&far);
                }
            }
            cache.insert(k, rgb.clone());
            Some(rgb)
        }

        while let Ok(first) = rx_req.recv() {
            let mut req = first;
            while let Ok(newer) = rx_req.try_recv() {
                req = newer;
            }
            let mut k = (req.p.floor() as u32).min(n);
            let mut frac = (req.p - k as f64).clamp(0.0, 1.0 - 1e-9);
            if k >= n {
                k = n;
                frac = 0.0;
            }
            let z = 2f64.powf(frac);
            let Some(curr) = colorized(&mut cache, &project, &manifest, offset, src_w, src_h, k)
            else {
                let _ = tx_rep.send(ScrubReply::Missing { k, version: req.version });
                continue;
            };
            let next = (z > 1.0 && k < n)
                .then(|| colorized(&mut cache, &project, &manifest, offset, src_w, src_h, k + 1))
                .flatten();
            let missing_next = z > 1.0 && k < n && next.is_none();
            let rgb = interpolate_frame(
                &curr,
                next.as_deref().map(|v| v.as_slice()),
                src_w,
                src_h,
                req.out_w as usize,
                req.out_h as usize,
                z,
            );
            let reply = ScrubReply::Frame {
                rgb,
                w: req.out_w,
                h: req.out_h,
                version: req.version,
                missing_next,
            };
            if tx_rep.send(reply).is_err() {
                return;
            }
        }
    });
    (tx_req, rx_rep)
}

/// « Ré-assembler seulement » : ré-écrit le manifest (couleurs/vitesse/
/// éclairage à jour) puis assemble SANS re-rendre. Garde-fou : la keyframe 0
/// existante doit porter la même empreinte hors-couleur que les réglages —
/// sinon (cible/dimensions/itérations changées) il faut « Générer ».
pub fn spawn_assemble_only(
    manifest: Manifest,
    project: PathBuf,
    use_ffmpeg: bool,
    cancel: Arc<AtomicBool>,
) -> mpsc::Receiver<VideoJobMsg> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let m = match plan_from_manifest(&manifest, &project) {
            Ok(m) => m,
            Err(e) => {
                let _ = tx.send(VideoJobMsg::Error(e.to_string()));
                return;
            }
        };
        let expected = match keyframe_params(&m, 0) {
            Ok(p) => p,
            Err(e) => {
                let _ = tx.send(VideoJobMsg::Error(e));
                return;
            }
        };
        match load_fmap(&keyframe_path(&project, 0)) {
            Ok(map0) if map_fingerprint(&map0.params) == map_fingerprint(&expected) => {}
            Ok(_) => {
                let _ = tx.send(VideoJobMsg::Error(
                    "les keyframes existantes ne correspondent plus aux réglages \
                     (cible, dimensions ou itérations modifiées) — utilisez Générer"
                        .into(),
                ));
                return;
            }
            Err(_) => {
                let _ = tx.send(VideoJobMsg::Error(
                    "aucune keyframe rendue dans ce dossier — utilisez Générer".into(),
                ));
                return;
            }
        }
        run_assemble(&m, &project, use_ffmpeg, &cancel, &tx);
    });
    rx
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn even_dims_rounds_down_with_floor() {
        assert_eq!(even_dims(1921, 1081), (1920, 1080));
        assert_eq!(even_dims(1920, 1080), (1920, 1080));
        assert_eq!(even_dims(17, 15), (16, 16)); // 15 → plancher 16
        assert_eq!(even_dims(0, 3), (16, 16));
    }

    /// Les estimations partagent les sources de vérité du pipeline : frames
    /// == longueur de `timeline`, keyframes == `keyframe_count`.
    #[test]
    fn estimates_match_pipeline_sources() {
        let e = estimates("8", 30, "1.0", 320, 180, 2).unwrap();
        assert_eq!(e.keyframes, 3);
        assert_eq!(e.maps, 4);
        let expected_frames = timeline(3, 30, &Dynamic::Constant(1.0)).unwrap().len();
        assert_eq!(e.frames, expected_frames); // 91
        assert!((e.duration_s - e.frames as f64 / 30.0).abs() < 1e-12);
        // 4 maps × (320·2)·(180·2) px × 20 octets
        assert_eq!(e.map_bytes, 4 * 640 * 360 * 20);
        assert!(estimates("8", 30, "0.0", 320, 180, 1).is_err());
        assert!(estimates("abc", 30, "1.0", 320, 180, 1).is_err());
        // Une spline de vitesse est comprise (la courbe éditée passe ici).
        let spline = estimates("8", 30, "0/1,6/0.5", 320, 180, 1).unwrap();
        assert!(spline.frames > e.frames, "un ralentissement allonge la vidéo");
    }

    /// `compiled_velocity` : sans courbe → constante EXACTE (chemin
    /// historique) ; avec une zone lente → spline parsable, durée allongée,
    /// et « Ré-assembler » suffit (la vitesse n'affecte pas les empreintes
    /// des maps).
    #[test]
    fn compiled_velocity_flat_is_exact_and_curve_extends() {
        let mut s = StudioSettings::default();
        assert_eq!(compiled_velocity(&s, 10), format!("{}", s.velocity));
        s.speed_points = vec![(0.0, 1.0), (4.0, 0.25), (8.0, 1.0)];
        let compiled = compiled_velocity(&s, 10);
        let d = Dynamic::parse(&compiled).expect("spline compilée parsable");
        assert!(d.end_time().unwrap() > 10.0, "zone ×0.25 ⇒ durée > 10 s");
        // La vitesse n'entre pas dans FractalParams : l'empreinte des maps
        // est inchangée ⇒ éditer la courbe n'invalide aucun calcul.
        let t = test_target();
        let m_flat = build_manifest(&StudioSettings::default(), &t);
        let m_curve = build_manifest(&s, &t);
        let mut mf = m_flat.clone();
        mf.video.keyframes = 3;
        let mut mc = m_curve.clone();
        mc.video.keyframes = 3;
        assert_eq!(
            map_fingerprint(&keyframe_params(&mf, 1).unwrap()),
            map_fingerprint(&keyframe_params(&mc, 1).unwrap()),
            "la courbe de vitesse ne doit pas invalider les maps"
        );
    }

    /// La rampe de défilement est une spline valide qui atteint exactement
    /// `cycles` à la fin de la vidéo.
    #[test]
    fn palette_scroll_spline_is_valid_and_exact() {
        let s = palette_scroll_spline(13.5, 2.0);
        let d = Dynamic::parse(&s).unwrap();
        assert_eq!(d.eval(0.0), 0.0);
        assert_eq!(d.eval(13.5), 2.0);
        assert_eq!(d.eval(99.0), 2.0, "clamp après la fin");
    }

    fn test_target() -> TargetView {
        TargetView {
            real: "-0.75".into(),
            imag: "0.01".into(),
            zoom: "8".into(),
            type_id: 3,
            palette: 4,
            color_repeat: 32,
            outcoloring: "smooth".into(),
        }
    }

    #[test]
    fn build_manifest_applies_even_dims_and_dynamics() {
        let mut s = StudioSettings { width: 1921, height: 1081, ..Default::default() };
        s.palette_scroll = true;
        s.palette_cycles = 3.0;
        let m = build_manifest(&s, &test_target());
        assert_eq!((m.image.width, m.image.height), (1920, 1080));
        assert_eq!(m.location.zoom, "8");
        assert_eq!(m.color.palette, 4);
        assert_eq!(m.video.keyframes, 0, "rempli par plan_from_manifest");
        let spline = m.dynamics.palette_offset.expect("défilement activé");
        let d = Dynamic::parse(&spline).unwrap();
        assert_eq!(d.eval(1e9), 3.0);

        let s2 = StudioSettings::default();
        assert!(build_manifest(&s2, &test_target()).dynamics.palette_offset.is_none());
    }

    /// Intégration : `spawn_generate` sur un projet minuscule → messages de
    /// progression des deux phases puis `Done` (frames PNG, sans ffmpeg).
    #[test]
    fn spawn_generate_end_to_end_frames() {
        let project = std::env::temp_dir()
            .join(format!("fractall_studio_gen_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&project);

        let settings = StudioSettings {
            width: 16,
            height: 12,
            fps: 5,
            iterations: 80,
            iterations_growth: 0.0,
            ..Default::default()
        };
        let target = TargetView { zoom: "4".into(), ..test_target() };
        let manifest = build_manifest(&settings, &target);
        let rx = spawn_generate(
            manifest,
            project.clone(),
            false,
            Arc::new(AtomicBool::new(false)),
        );

        let (mut render_msgs, mut assemble_msgs) = (0, 0);
        let mut started: Vec<u32> = Vec::new();
        let mut thumbs: Vec<u32> = Vec::new();
        let done = loop {
            match rx.recv_timeout(Duration::from_secs(120)).expect("job bloqué") {
                VideoJobMsg::RenderStarted { k } => started.push(k),
                VideoJobMsg::RenderProgress { .. } => render_msgs += 1,
                VideoJobMsg::Thumb { k, w, h, iterations, zs, provisional, .. } => {
                    assert!(!provisional, "les thumbs du job sont définitifs");
                    assert_eq!(iterations.len(), (w * h) as usize);
                    assert_eq!(zs.len(), (w * h) as usize);
                    thumbs.push(k);
                }
                VideoJobMsg::AssembleProgress { .. } => assemble_msgs += 1,
                VideoJobMsg::Done { output } => break output,
                other => panic!("message inattendu : {other:?}"),
            }
        };
        assert_eq!(render_msgs, 3, "zoom 4 → 2 segments → 3 maps");
        assert!(assemble_msgs > 0);
        // Ordre dichotomique + une miniature par map (G13).
        assert_eq!(started, crate::video::bisection_order(2), "ordre de rendu");
        thumbs.sort_unstable();
        assert_eq!(thumbs, vec![0, 1, 2], "une miniature par keyframe");
        let out = PathBuf::from(done);
        assert!(out.ends_with("frames") && out.join("frame_000000.png").exists());
        let _ = std::fs::remove_dir_all(&project);
    }

    /// Scan d'un projet existant : une miniature par map présente (reprise),
    /// rien pour les absentes — c'est ce qui repeuple la timeline à
    /// l'adoption d'un dossier.
    #[test]
    fn spawn_thumb_scan_emits_existing_maps_only() {
        let project = std::env::temp_dir()
            .join(format!("fractall_studio_scan_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&project);
        let settings = StudioSettings {
            width: 16,
            height: 12,
            fps: 5,
            iterations: 80,
            iterations_growth: 0.0,
            ..Default::default()
        };
        let target = TargetView { zoom: "8".into(), ..test_target() };
        let m = plan_from_manifest(&build_manifest(&settings, &target), &project).unwrap();
        video::render_project(&project).unwrap();
        // Supprime la map 2 : le scan ne doit émettre que 0, 1, 3.
        std::fs::remove_file(keyframe_path(&project, 2)).unwrap();

        let rx = spawn_thumb_scan(project.clone(), m.video.keyframes);
        let mut ks: Vec<u32> = Vec::new();
        while let Ok(msg) = rx.recv_timeout(Duration::from_secs(30)) {
            match msg {
                VideoJobMsg::Thumb { k, .. } => ks.push(k),
                other => panic!("message inattendu : {other:?}"),
            }
            if ks.len() == 3 {
                break;
            }
        }
        ks.sort_unstable();
        assert_eq!(ks, vec![0, 1, 3]);
        let _ = std::fs::remove_dir_all(&project);
    }

    /// Worker de scrub : à p entier la frame est la keyframe colorisée
    /// PIXEL-EXACTE (fast path z=1 de l'assembleur — le scrub prévisualise
    /// exactement la vidéo) ; une keyframe absente → Missing.
    #[test]
    fn scrub_worker_matches_assembler_at_integer_positions() {
        let project = std::env::temp_dir()
            .join(format!("fractall_studio_scrub_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&project);
        let settings = StudioSettings {
            width: 16,
            height: 12,
            fps: 5,
            iterations: 80,
            iterations_growth: 0.0,
            ..Default::default()
        };
        let target = TargetView { zoom: "4".into(), ..test_target() };
        let m = plan_from_manifest(&build_manifest(&settings, &target), &project).unwrap();
        video::render_project(&project).unwrap();

        let (tx, rx) = spawn_scrub_worker(project.clone(), m.clone());
        // p = 1 exact, sortie à la taille SOURCE (⚠️ even_dims plancher 16 :
        // le 16×12 demandé devient 16×16) → identité pixel-exacte.
        let (out_w, out_h) = (m.image.width, m.image.height);
        tx.send(ScrubRequest { p: 1.0, out_w, out_h, version: 1 }).unwrap();
        match rx.recv_timeout(Duration::from_secs(30)).expect("réponse scrub") {
            ScrubReply::Frame { rgb, missing_next, version, .. } => {
                assert_eq!(version, 1);
                assert!(!missing_next);
                let map1 = load_fmap(&keyframe_path(&project, 1)).unwrap();
                let expected =
                    colorize_keyframe(&map1, &m, m.color.palette_offset).unwrap();
                assert_eq!(rgb, expected, "scrub à p=1 == keyframe 1 colorisée");
            }
            ScrubReply::Missing { k, .. } => panic!("keyframe {k} manquante inattendue"),
        }
        // Keyframe supprimée → Missing explicite.
        std::fs::remove_file(keyframe_path(&project, 0)).unwrap();
        tx.send(ScrubRequest { p: 0.0, out_w, out_h, version: 2 }).unwrap();
        match rx.recv_timeout(Duration::from_secs(30)).expect("réponse scrub") {
            ScrubReply::Missing { k, version } => {
                assert_eq!((k, version), (0, 2));
            }
            ScrubReply::Frame { .. } => panic!("Missing attendu pour une map absente"),
        }
        let _ = std::fs::remove_dir_all(&project);
    }

    /// Garde-fou Ré-assembler : maps rendues pour une géométrie ≠ des
    /// réglages courants → Error explicite qui renvoie vers Générer.
    #[test]
    fn spawn_assemble_only_rejects_mismatched_maps() {
        let project = std::env::temp_dir()
            .join(format!("fractall_studio_mismatch_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&project);

        let settings = StudioSettings {
            width: 16,
            height: 12,
            fps: 5,
            iterations: 80,
            iterations_growth: 0.0,
            ..Default::default()
        };
        let target = TargetView { zoom: "4".into(), ..test_target() };
        // Rend les maps à 16×12…
        let m = plan_from_manifest(&build_manifest(&settings, &target), &project).unwrap();
        video::render_project(&project).unwrap();
        let _ = m;
        // …puis demande un ré-assemblage à 32×24 : refus explicite.
        let bigger = StudioSettings { width: 32, height: 24, ..settings };
        let rx = spawn_assemble_only(
            build_manifest(&bigger, &target),
            project.clone(),
            false,
            Arc::new(AtomicBool::new(false)),
        );
        match rx.recv_timeout(Duration::from_secs(60)).expect("réponse attendue") {
            VideoJobMsg::Error(e) => assert!(e.contains("Générer"), "message : {e}"),
            other => panic!("Error attendu, eu {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&project);
    }
}
