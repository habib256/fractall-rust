//! Worker du studio vidéo (G12 jalon 6) — zéro egui, testable.
//!
//! Construit le `Manifest` depuis l'état du studio (l'utilisateur n'édite
//! aucun fichier), lance plan → render → assemble dans UN thread de job, et
//! remonte la progression par messages mpsc (pattern `HqRenderMessage` du
//! générateur). Les erreurs traversent le canal en `String` (`Box<dyn Error>`
//! n'est pas `Send`) ; l'annulation passe par l'`Arc<AtomicBool>` partagé et
//! est ACQUITTÉE par `VideoJobMsg::Cancelled` (maps conservées — reprise).

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::AtomicBool;
use std::sync::{mpsc, Arc};

use crate::io::fmap::load_fmap;
use crate::video::assemble::{
    assemble_project_with_progress, timeline, AssembleOptions, AssembleOutcome,
};
use crate::video::spline::Dynamic;
use crate::video::{
    self, keyframe_params, keyframe_path, map_fingerprint, plan_from_manifest, KeyframeEvent,
    Manifest, RenderOutcome,
};

/// Messages du job vidéo vers l'UI.
#[derive(Clone, Debug, PartialEq)]
pub enum VideoJobMsg {
    /// Maps complétées (rendues OU réutilisées) sur total = keyframes + 1.
    RenderProgress { done: u32, total: u32 },
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
    /// Keyframes (×2) par seconde, > 0.
    pub velocity: f64,
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
/// nombre de frames (aucune formule dupliquée).
pub fn estimates(
    zoom: &str,
    fps: u32,
    velocity: f64,
    w: u32,
    h: u32,
    ss: u32,
) -> Result<Estimates, String> {
    let n = video::keyframe_count(zoom)?;
    if !(velocity > 0.0 && velocity.is_finite()) {
        return Err(format!("vitesse invalide: {velocity}"));
    }
    let fps = fps.max(1);
    let positions = timeline(n, fps, &Dynamic::Constant(velocity))?;
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
    m.video.velocity = format!("{}", s.velocity);
    m.lighting.enable = s.lighting;
    m.lighting.beta = s.lighting_beta;
    if s.palette_scroll && s.palette_cycles != 0.0 {
        if let Ok(est) = estimates(&t.zoom, m.video.fps, s.velocity, w, h, 1) {
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

/// « Générer » : plan → render (progression par map) → assemble (progression
/// par frame) → Done. Reprise gratuite : les maps valides sont skippées.
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
        let render = video::render_project_with_progress(&project, &cancel, &mut |ev| match ev {
            KeyframeEvent::Rendered { .. } | KeyframeEvent::Skipped { .. } => {
                done += 1;
                let _ = tx.send(VideoJobMsg::RenderProgress { done, total });
            }
            KeyframeEvent::Started { .. } | KeyframeEvent::Invalidated { .. } => {}
        });
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
        let e = estimates("8", 30, 1.0, 320, 180, 2).unwrap();
        assert_eq!(e.keyframes, 3);
        assert_eq!(e.maps, 4);
        let expected_frames = timeline(3, 30, &Dynamic::Constant(1.0)).unwrap().len();
        assert_eq!(e.frames, expected_frames); // 91
        assert!((e.duration_s - e.frames as f64 / 30.0).abs() < 1e-12);
        // 4 maps × (320·2)·(180·2) px × 20 octets
        assert_eq!(e.map_bytes, 4 * 640 * 360 * 20);
        assert!(estimates("8", 30, 0.0, 320, 180, 1).is_err());
        assert!(estimates("abc", 30, 1.0, 320, 180, 1).is_err());
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
        let done = loop {
            match rx.recv_timeout(Duration::from_secs(120)).expect("job bloqué") {
                VideoJobMsg::RenderProgress { .. } => render_msgs += 1,
                VideoJobMsg::AssembleProgress { .. } => assemble_msgs += 1,
                VideoJobMsg::Done { output } => break output,
                other => panic!("message inattendu : {other:?}"),
            }
        };
        assert_eq!(render_msgs, 3, "zoom 4 → 2 segments → 3 maps");
        assert!(assemble_msgs > 0);
        let out = PathBuf::from(done);
        assert!(out.ends_with("frames") && out.join("frame_000000.png").exists());
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
