//! Fractall Video Studio (G12 jalon 6) — application eframe DÉDIÉE à la
//! production de vidéos de zoom, séparée du générateur `fractall-gui`
//! (exigence : le générateur n'est pas modifié).
//!
//! Trois zones : panneau gauche (réglages vidéo + estimations live +
//! Générer / Ré-assembler / Annuler), zone centrale (preview NAVIGABLE :
//! molette = zoom ancré au curseur, glisser = pan, drag-and-drop d'un PNG
//! fractall ou d'une map .fmap = adoption de la cible), barre basse (statut).
//!
//! Aucun fichier à éditer : le studio construit le `Manifest` en mémoire
//! (`job::build_manifest`) et le pipeline G12 (`video::plan_from_manifest` →
//! `render_project_with_progress` → `assemble_project_with_progress`) fait le
//! reste dans un thread de job annulable.
//!
//! La preview passe par le dispatcher UNIQUE (`render_escape_time_
//! cancellable_with_reuse`, invariant CLAUDE.md) en 2 passes progressives
//! (¼ puis pleine résolution), avec cache d'orbite référence inter-passes.

pub mod job;
pub mod nav;
pub mod timeline;
pub mod timeline_state;

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

use num_complex::Complex64;
use rug::Float;

use crate::fractal::perturbation::ReferenceOrbitCache;
use crate::fractal::{
    default_params_for_type, ColorSpace, FractalParams, FractalType, OutColoringMode,
};
use crate::io::fmap::load_fmap;
use crate::io::png::{colorize_buffers, load_png_metadata};
use crate::render::{render_request, RenderRequest};
use crate::video::spline::Dynamic;
use crate::video::zoom_from_span_x;
use job::{ScrubReply, ScrubRequest, StudioSettings, TargetView, VideoJobMsg};
use nav::HpView;
use timeline::ThumbSlot;

/// Types proposés dans le studio : famille escape-time bytecode+perturbation
/// (id CLI, type, label menu). OrbitTraps/Wings et les types spéciaux sont
/// hors périmètre (données par-pixel non persistées en .fmap).
const STUDIO_TYPES: &[(u8, FractalType, &str)] = &[
    (3, FractalType::Mandelbrot, "Mandelbrot"),
    (4, FractalType::Julia, "Julia"),
    (13, FractalType::BurningShip, "Burning Ship"),
    (14, FractalType::Tricorn, "Tricorn"),
    (19, FractalType::Celtic, "Celtic"),
    (8, FractalType::Buffalo, "Buffalo"),
    (18, FractalType::PerpendicularBurningShip, "Perp. Burning Ship"),
    (23, FractalType::Multibrot, "Multibrot"),
];

/// Modes de colorisation sûrs pour la vidéo (colorisables depuis
/// iterations+zs seuls) : (nom CLI/manifest, label).
const STUDIO_OUTCOLORINGS: &[(&str, &str)] = &[
    ("smooth", "Smooth"),
    ("iter", "Iter"),
    ("potential", "Potential"),
    ("binary", "Binary"),
    ("color-decomp", "Color decomp"),
];

fn studio_outcoloring_index(mode: OutColoringMode) -> Option<usize> {
    let name = mode.cli_name();
    STUDIO_OUTCOLORINGS.iter().position(|(cli, _)| *cli == name)
}

/// Frame assemblée dont la position de zoom est la plus proche de `p`.
/// La timeline peut reculer (spline à vitesse négative), donc elle n'est pas
/// nécessairement triée et ne peut pas être cherchée par partition binaire.
fn nearest_position_index(positions: &[f64], p: f64) -> usize {
    positions
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (*a - p).abs().total_cmp(&(*b - p).abs()))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Résolutions presets (label, w, h) + entrée custom.
const RESOLUTIONS: &[(&str, u32, u32)] = &[
    ("640×360", 640, 360),
    ("1280×720", 1280, 720),
    ("1920×1080", 1920, 1080),
    ("2560×1440", 2560, 1440),
    ("3840×2160 (4K)", 3840, 2160),
];

// --- Géométrie du panneau timeline (G13) ---
const TL_THUMB_H: f32 = 54.0;
const TL_CURVE_H: f32 = 46.0;
const TL_RULER_H: f32 = 14.0;
const TL_LABEL_H: f32 = 14.0;
const TL_CELL_PAD: f32 = 3.0;
/// Plage log2 de la courbe de vitesse : ×1/8 … ×8 (0 = ×1 au centre).
const TL_CURVE_LOG_RANGE: f32 = 3.0;
/// Textures créées par frame au maximum (lisse le burst d'un scan complet).
const TL_MAX_TEXTURES_PER_FRAME: usize = 32;

/// Durée lisible : « 45 s », « 3 min 20 s », « 1 h 02 min ».
fn fmt_duration(seconds: f64) -> String {
    let s = seconds.max(0.0).round() as u64;
    match s {
        0..=59 => format!("{s} s"),
        60..=3599 => format!("{} min {:02} s", s / 60, s % 60),
        _ => format!("{} h {:02} min", s / 3600, (s % 3600) / 60),
    }
}

/// Message du thread de preview : une passe colorisée.
struct PreviewPass {
    rgb: Vec<u8>,
    w: u32,
    h: u32,
    is_final: bool,
    cache: Option<Arc<ReferenceOrbitCache>>,
    version: u64,
}

pub struct VideoStudioApp {
    // --- Vue preview (cible du zoom) ---
    view: HpView,
    type_idx: usize,
    outcoloring_idx: usize,
    palette: u8,
    color_repeat: u32,
    color_space: ColorSpace,
    seed: Complex64,
    multibrot_power: f64,
    preview_iters: u32,

    // --- Rendu preview asynchrone ---
    preview_version: crate::gui::async_version::AsyncVersion,
    preview_cancel: Arc<AtomicBool>,
    preview_rx: Option<mpsc::Receiver<PreviewPass>>,
    preview_rendering: bool,
    preview_dirty: bool,
    preview_texture: Option<egui::TextureHandle>,
    orbit_cache: Option<Arc<ReferenceOrbitCache>>,
    panel_size: (u32, u32),

    // --- Réglages vidéo ---
    settings: StudioSettings,
    resolution_idx: usize,
    output_dir: String,
    ffmpeg_available: bool,

    // --- Job vidéo (pattern render_cancel du générateur) ---
    job_rx: Option<mpsc::Receiver<VideoJobMsg>>,
    job_cancel: Arc<AtomicBool>,
    job_running: bool,
    job_phase: String,
    job_progress: f32,
    job_started: Option<Instant>,
    cancel_requested: bool,
    job_result: Option<Result<String, String>>,

    // --- Timeline (G13) : miniatures + courbe de vitesse + scrub ---
    tl_slots: Vec<ThumbSlot>,
    tl_textures: Vec<Option<egui::TextureHandle>>,
    /// Empreinte couleurs des textures (palette, repeat, espace, outcoloring,
    /// type) :
    /// un changement invalide les textures, PAS les canaux (recolorisation
    /// in-memory, aucun accès disque).
    tl_stamp: (u8, u32, ColorSpace, usize, usize, bool, u64),
    /// Keyframe en cours de rendu (surbrillance + auto-scroll).
    tl_current: Option<u32>,
    tl_autoscrolled: Option<u32>,
    /// Durées de rendu mesurées (k, secondes) — ETA par régression.
    tl_measured: Vec<(u32, f64)>,
    tl_done: HashSet<u32>,
    job_total: Option<u32>,
    /// Scan des maps existantes (reprise / dossier adopté).
    scan_rx: Option<mpsc::Receiver<VideoJobMsg>>,
    scanned_dir: String,
    /// Manifest du projet sur disque (sondé au changement de dossier).
    project_manifest: Option<crate::video::Manifest>,
    probed_dir: Option<String>,
    /// Miniature provisoire keyframe 0 (vue pleine), cachée par empreinte
    /// type + centre + géométrie + itérations.
    first_thumb: Option<(String, ThumbSlot)>,
    /// (empreinte au moment du spawn, canal) : une réponse terminant après un
    /// changement de cible ne doit jamais être attribuée à la nouvelle vue.
    /// (clé, receiver, cancel) — le cancel permet d'arrêter un mini-rendu
    /// périmé (dropper le receiver ne stoppait pas le calcul).
    first_thumb_rx: Option<(String, mpsc::Receiver<VideoJobMsg>, Arc<AtomicBool>)>,
    /// Miniature provisoire keyframe finale (copie réduite de la preview).
    preview_thumb: Option<(Vec<u8>, u32, u32)>,
    /// Index du point de la courbe de vitesse en cours de drag.
    curve_drag: Option<usize>,

    // --- Scrub (préview vidéo interpolée, G13) ---
    scrub_tx: Option<mpsc::Sender<ScrubRequest>>,
    scrub_rx: Option<mpsc::Receiver<ScrubReply>>,
    scrub_fingerprint: String,
    scrub_texture: Option<egui::TextureHandle>,
    scrub_label: String,
    scrub: timeline_state::ScrubState,
    /// Positions p(frame) de l'assembleur, cachées par (n, fps, vitesse).
    scrub_positions: Option<(String, Vec<f64>)>,
    /// Compteur bumpé à chaque job terminé : invalide le cache du worker de
    /// scrub (les maps sur disque ont pu changer).
    generation: u64,

    status: String,
}

impl VideoStudioApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let p = default_params_for_type(FractalType::Mandelbrot, 800, 600);
        Self {
            view: HpView::new(p.center_x, p.center_y, p.span_x),
            type_idx: 0,
            outcoloring_idx: 0,
            palette: p.color_mode,
            color_repeat: p.color_repeat,
            color_space: p.color_space,
            seed: p.seed,
            multibrot_power: p.multibrot_power,
            preview_iters: 1000,
            preview_version: Default::default(),
            preview_cancel: Arc::new(AtomicBool::new(false)),
            preview_rx: None,
            preview_rendering: false,
            preview_dirty: true,
            preview_texture: None,
            orbit_cache: None,
            panel_size: (0, 0),
            settings: StudioSettings::default(),
            resolution_idx: 1, // 1280×720
            output_dir: job::default_output_dir(),
            ffmpeg_available: job::detect_ffmpeg(),
            job_rx: None,
            job_cancel: Arc::new(AtomicBool::new(false)),
            job_running: false,
            job_phase: String::new(),
            job_progress: 0.0,
            job_started: None,
            cancel_requested: false,
            job_result: None,
            tl_slots: Vec::new(),
            tl_textures: Vec::new(),
            tl_stamp: (255, 0, ColorSpace::Rgb, usize::MAX, usize::MAX, false, 0),
            tl_current: None,
            tl_autoscrolled: None,
            tl_measured: Vec::new(),
            tl_done: HashSet::new(),
            job_total: None,
            scan_rx: None,
            scanned_dir: String::new(),
            project_manifest: None,
            probed_dir: None,
            first_thumb: None,
            first_thumb_rx: None,
            preview_thumb: None,
            curve_drag: None,
            scrub_tx: None,
            scrub_rx: None,
            scrub_fingerprint: String::new(),
            scrub_texture: None,
            scrub_label: String::new(),
            scrub: Default::default(),
            scrub_positions: None,
            generation: 0,
            status: "Naviguez vers la cible (molette = zoom, glisser = déplacer) ou déposez un PNG fractall".into(),
        }
    }

    fn current_type(&self) -> (u8, FractalType) {
        let (id, t, _) = STUDIO_TYPES[self.type_idx];
        (id, t)
    }

    fn current_outcoloring(&self) -> &'static str {
        STUDIO_OUTCOLORINGS[self.outcoloring_idx].0
    }

    /// Params de la preview à la résolution du panneau : la vue HP est la
    /// vérité, span_y dérivé de l'aspect en GMP.
    fn preview_params(&self, w: u32, h: u32) -> Option<FractalParams> {
        let (_, ftype) = self.current_type();
        let mut p = default_params_for_type(ftype, w, h);
        p.seed = self.seed;
        p.multibrot_power = self.multibrot_power;
        p.color_space = self.color_space;
        let prec = nav::view_precision(&self.view.sx);
        let cx = Float::parse(&self.view.cx).ok().map(|v| Float::with_val(prec, v))?;
        let cy = Float::parse(&self.view.cy).ok().map(|v| Float::with_val(prec, v))?;
        let sx = Float::parse(&self.view.sx).ok().map(|v| Float::with_val(prec, v))?;
        if !sx.is_finite() || sx <= 0.0 {
            return None;
        }
        let aspect = h as f64 / w as f64;
        let sy = Float::with_val(prec, &sx * aspect);
        p.center_x = cx.to_f64();
        p.center_y = cy.to_f64();
        p.span_x = sx.to_f64();
        p.span_y = sy.to_f64();
        p.center_x_hp = Some(self.view.cx.clone());
        p.center_y_hp = Some(self.view.cy.clone());
        p.span_x_hp = Some(self.view.sx.clone());
        p.span_y_hp = Some(sy.to_string_radix(10, None));
        let iters = self.preview_iters.max(50);
        p.iteration_max = iters;
        p.max_perturb_iterations = iters;
        p.max_bla_steps = iters;
        p.color_mode = self.palette;
        p.color_repeat = self.color_repeat.max(1);
        p.out_coloring_mode =
            OutColoringMode::from_cli_name(self.current_outcoloring()).unwrap_or(OutColoringMode::Smooth);
        Some(p)
    }

    /// (Re)lance le rendu de preview : annule l'ancien (nouvel Arc, pattern
    /// du générateur), 2 passes ¼ → pleine résolution dans un thread.
    fn start_preview_render(&mut self) {
        let (w, h) = self.panel_size;
        if w < 64 || h < 48 {
            return;
        }
        let Some(params) = self.preview_params(w, h) else {
            self.status = "Vue invalide — coordonnées illisibles".into();
            return;
        };
        self.preview_cancel.store(true, Ordering::Relaxed);
        self.preview_cancel = Arc::new(AtomicBool::new(false));
        let version = self.preview_version.issue();
        let cancel = self.preview_cancel.clone();
        let (tx, rx) = mpsc::channel();
        self.preview_rx = Some(rx);
        self.preview_rendering = true;
        self.preview_dirty = false;
        // La vue change : la copie réduite de l'ANCIENNE preview n'est plus la
        // keyframe finale (bug 2026-08-23 : sans projet sur disque,
        // `view_is_project_target()` est toujours vrai et l'ancienne cible
        // restait collée sur la nouvelle timeline). On retire aussi le
        // provisoire RGB déjà appliqué sur le dernier slot.
        self.preview_thumb = None;
        if let Some(last) = self.tl_slots.len().checked_sub(1) {
            if matches!(self.tl_slots[last], ThumbSlot::Rgb { .. }) {
                self.tl_slots[last] = ThumbSlot::Empty;
                self.tl_textures[last] = None;
            }
        }

        let cache0 = self.orbit_cache.clone();
        std::thread::spawn(move || {
            let mut cache = cache0;
            for (div, is_final) in [(4u32, false), (1u32, true)] {
                if cancel.load(Ordering::Relaxed) {
                    return;
                }
                let mut p = params.clone();
                p.width = (params.width / div).max(64);
                p.height = (params.height / div).max(48);
                let mut oc = cache.clone();
                let Some(out) = render_request(RenderRequest::new(&p, &cancel), &mut oc) else {
                    return;
                };
                cache = oc;
                // Canaux annexes transmis : sans eux la preview des modes
                // Distance*/OrbitTraps retombait silencieusement sur Smooth (G5).
                let rgb = crate::io::png::colorize_to_rgb_with_extras(
                    &p, &out.iterations, &out.zs, &out.distances, &out.orbits,
                );
                let pass = PreviewPass { rgb, w: p.width, h: p.height, is_final, cache: cache.clone(), version };
                if tx.send(pass).is_err() {
                    return;
                }
            }
        });
    }

    /// Adopte la cible d'un PNG fractall (métadonnées) ou d'une map .fmap.
    fn adopt_file(&mut self, path: &Path) {
        let params: Result<FractalParams, String> = match path.extension().and_then(|e| e.to_str()) {
            Some("png") => load_png_metadata(path).map_err(|e| e.to_string()),
            Some("fmap") => load_fmap(path).map(|m| m.params).map_err(|e| e.to_string()),
            _ => Err("format non reconnu (PNG fractall ou .fmap)".into()),
        };
        let params = match params {
            Ok(p) => p,
            Err(e) => {
                self.status = format!("Échec du chargement de {} : {e}", path.display());
                return;
            }
        };
        // Type : adopté s'il fait partie de la famille supportée.
        if let Some(idx) = STUDIO_TYPES.iter().position(|(_, t, _)| *t == params.fractal_type) {
            self.type_idx = idx;
        } else {
            self.status = format!(
                "Type {:?} hors famille vidéo — cible adoptée avec le type courant",
                params.fractal_type
            );
        }
        self.view = HpView {
            cx: params.center_x_hp.clone().unwrap_or_else(|| params.center_x.to_string()),
            cy: params.center_y_hp.clone().unwrap_or_else(|| params.center_y.to_string()),
            sx: params.span_x_hp.clone().unwrap_or_else(|| params.span_x.to_string()),
        };
        self.palette = params.color_mode;
        self.color_repeat = params.color_repeat;
        self.color_space = params.color_space;
        self.outcoloring_idx = studio_outcoloring_index(params.out_coloring_mode).unwrap_or(0);
        self.seed = params.seed;
        self.multibrot_power = params.multibrot_power;
        self.preview_iters = params.iteration_max;
        self.settings.iterations = params.iteration_max;
        self.orbit_cache = None;
        self.preview_dirty = true;
        self.status = format!("Cible adoptée depuis {}", path.display());
    }

    fn target_view(&self) -> Result<TargetView, String> {
        let zoom = zoom_from_span_x(&self.view.sx)?;
        let (type_id, _) = self.current_type();
        Ok(TargetView {
            real: self.view.cx.clone(),
            imag: self.view.cy.clone(),
            zoom,
            type_id,
            seed: self.seed,
            multibrot_power: self.multibrot_power,
            color_space: self.color_space,
            palette: self.palette,
            color_repeat: self.color_repeat,
            outcoloring: self.current_outcoloring().to_string(),
        })
    }

    fn start_job(&mut self, assemble_only: bool) {
        let mut target = match self.target_view() {
            Ok(t) => t,
            Err(e) => {
                self.status = e;
                return;
            }
        };
        // Ré-assembler = les maps du PROJET : la cible (centre/zoom) vient du
        // manifest sur disque, pas de la vue courante — inspecter une keyframe
        // par clic sur une miniature ne doit pas re-planifier la vidéo.
        if assemble_only {
            if let Some(m) = &self.project_manifest {
                target.real = m.location.real.clone();
                target.imag = m.location.imag.clone();
                target.zoom = m.location.zoom.clone();
            }
        }
        let manifest = job::build_manifest(&self.settings, &target);
        let project = PathBuf::from(self.output_dir.trim());
        self.job_cancel = Arc::new(AtomicBool::new(false));
        let cancel = self.job_cancel.clone();
        self.job_rx = Some(if assemble_only {
            job::spawn_assemble_only(manifest, project, self.ffmpeg_available, cancel)
        } else {
            job::spawn_generate(manifest, project, self.ffmpeg_available, cancel)
        });
        self.job_running = true;
        self.job_phase = if assemble_only { "Assemblage…" } else { "Préparation…" }.into();
        self.job_progress = 0.0;
        self.job_started = Some(Instant::now());
        self.cancel_requested = false;
        self.job_result = None;
        // Un nouveau rendu peut viser une autre cible tout en gardant le même
        // nombre de keyframes. Les anciennes miniatures définitives ne
        // doivent pas survivre jusqu'à leur remplacement progressif.
        if !assemble_only {
            self.clear_timeline_slots();
        }
        // Timeline/ETA : nouveau job = nouvelles mesures ; le manifest vient
        // d'être (ré)écrit par plan → re-sonder le dossier.
        self.tl_measured.clear();
        self.tl_done.clear();
        self.job_total = None;
        self.tl_current = None;
        self.probed_dir = None;
        self.scrub.close();
    }

    fn drain_job_messages(&mut self) {
        let Some(rx) = self.job_rx.take() else { return };
        let mut finished = false;
        loop {
            match rx.try_recv() {
                Ok(VideoJobMsg::RenderStarted { k }) => {
                    self.tl_current = Some(k);
                    // Le plan vient d'écrire le manifest : re-sonder pour que
                    // timeline/scrub/banner voient le projet dès le job.
                    self.probed_dir = None;
                }
                Ok(VideoJobMsg::RenderProgress { done, total, k, seconds }) => {
                    self.job_phase = format!("Keyframes {done}/{total}");
                    // Deux phases : rendu 0→50 %, assemblage 50→100 %.
                    self.job_progress = 0.5 * done as f32 / total.max(1) as f32;
                    self.job_total = Some(total);
                    self.tl_done.insert(k);
                    if let Some(s) = seconds {
                        self.tl_measured.push((k, s));
                    }
                }
                Ok(VideoJobMsg::Thumb {
                    k,
                    w,
                    h,
                    fractal_type,
                    color_space,
                    iter_max,
                    iterations,
                    zs,
                    provisional,
                }) => {
                    self.apply_thumb(
                        k as usize,
                        ThumbSlot::Channels {
                            w,
                            h,
                            fractal_type,
                            color_space,
                            iter_max,
                            iterations,
                            zs,
                            provisional,
                        },
                    );
                }
                Ok(VideoJobMsg::AssembleProgress { frame, total }) => {
                    self.job_phase = format!("Assemblage {frame}/{total}");
                    self.job_progress = 0.5 + 0.5 * frame as f32 / total.max(1) as f32;
                    self.tl_current = None;
                }
                Ok(VideoJobMsg::Done { output }) => {
                    self.job_result = Some(Ok(output));
                    finished = true;
                }
                Ok(VideoJobMsg::Cancelled) => {
                    self.job_result = Some(Err(
                        "Annulé — keyframes conservées : relancer Générer reprendra ici".into(),
                    ));
                    finished = true;
                }
                Ok(VideoJobMsg::Error(e)) => {
                    self.job_result = Some(Err(e));
                    finished = true;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    if self.job_running && !finished {
                        self.job_result = Some(Err("thread vidéo interrompu".into()));
                        finished = true;
                    }
                    break;
                }
            }
        }
        if finished {
            self.job_running = false;
            self.job_total = None;
            self.tl_current = None;
            // Les maps ont changé sur disque : re-sonder le manifest et
            // invalider le cache du worker de scrub.
            self.probed_dir = None;
            self.generation += 1;
        } else {
            self.job_rx = Some(rx);
        }
    }

    /// Applique une miniature reçue (job, scan ou provisoire). Un provisoire
    /// n'écrase jamais une miniature définitive.
    fn apply_thumb(&mut self, k: usize, slot: ThumbSlot) {
        if k >= self.tl_slots.len() {
            return;
        }
        let incoming_provisional = !slot.is_final();
        if incoming_provisional && self.tl_slots[k].is_final() {
            return;
        }
        self.tl_slots[k] = slot;
        self.tl_textures[k] = None;
    }

    fn clear_timeline_slots(&mut self) {
        self.tl_slots.fill(ThumbSlot::Empty);
        self.tl_textures.fill(None);
        self.tl_current = None;
        self.tl_autoscrolled = None;
    }

    /// Sonde le dossier projet quand il change (ou après un job) : manifest
    /// sur disque + scan des maps existantes pour repeupler la timeline.
    fn probe_project(&mut self) {
        let dir = self.output_dir.trim().to_string();
        if self.probed_dir.as_deref() == Some(dir.as_str()) {
            return;
        }
        self.probed_dir = Some(dir.clone());
        let dir_changed = self.scanned_dir != dir;
        if dir_changed {
            // Deux projets de même longueur partageraient sinon les mêmes
            // slots : les maps absentes du nouveau dossier resteraient
            // illustrées par les miniatures définitives de l'ancien.
            self.scan_rx = None;
            self.clear_timeline_slots();
            self.scanned_dir = dir.clone();
        }
        self.project_manifest = (!dir.is_empty())
            .then(|| crate::video::Manifest::load(&Path::new(&dir).join("manifest.toml")).ok())
            .flatten()
            .filter(|m| crate::video::validate_project_keyframes(m).is_ok());
        if let Some(m) = &self.project_manifest {
            if dir_changed {
                self.scan_rx = Some(job::spawn_thumb_scan(
                    PathBuf::from(&dir),
                    m.video.keyframes,
                    m.color.color_space,
                ));
            }
        }
    }

    fn drain_scan_messages(&mut self) {
        let Some(rx) = self.scan_rx.take() else { return };
        let mut disconnected = false;
        loop {
            match rx.try_recv() {
                Ok(VideoJobMsg::Thumb {
                    k,
                    w,
                    h,
                    fractal_type,
                    color_space,
                    iter_max,
                    iterations,
                    zs,
                    provisional,
                }) => {
                    self.apply_thumb(
                        k as usize,
                        ThumbSlot::Channels {
                            w,
                            h,
                            fractal_type,
                            color_space,
                            iter_max,
                            iterations,
                            zs,
                            provisional,
                        },
                    );
                }
                Ok(_) => {}
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        if !disconnected {
            self.scan_rx = Some(rx);
        }
    }

    fn drain_first_thumb(&mut self) {
        let Some((spawn_key, rx, cancel)) = self.first_thumb_rx.take() else { return };
        match rx.try_recv() {
            Ok(VideoJobMsg::Thumb {
                k: 0,
                w,
                h,
                fractal_type,
                color_space,
                iter_max,
                iterations,
                zs,
                ..
            }) => {
                let slot = ThumbSlot::Channels {
                    w,
                    h,
                    fractal_type,
                    color_space,
                    iter_max,
                    iterations,
                    zs,
                    provisional: true,
                };
                self.first_thumb = Some((spawn_key, slot));
            }
            Ok(_) | Err(mpsc::TryRecvError::Empty) => {
                self.first_thumb_rx = Some((spawn_key, rx, cancel));
            }
            Err(mpsc::TryRecvError::Disconnected) => {}
        }
    }

    /// Le centre de la vue correspond-il au projet sur disque ? (Le zoom n'est
    /// PAS comparé : inspecter une keyframe intermédiaire via la timeline est
    /// une navigation légitime, pas une divergence.)
    fn view_center_matches_project(&self) -> bool {
        match &self.project_manifest {
            Some(m) => m.location.real == self.view.cx && m.location.imag == self.view.cy,
            None => true,
        }
    }

    /// La vue courante EST la cible du projet (centre + profondeur) : seul
    /// cas où la preview peut servir de provisoire « keyframe finale ».
    fn view_is_project_target(&self) -> bool {
        match &self.project_manifest {
            Some(m) => {
                self.view_center_matches_project()
                    && timeline::span_matches_keyframe(&self.view.sx, m.video.keyframes)
            }
            None => true,
        }
    }

    fn drain_preview_messages(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.preview_rx.take() else { return };
        let mut latest: Option<PreviewPass> = None;
        let mut disconnected = false;
        loop {
            match rx.try_recv() {
                Ok(pass) if self.preview_version.accepts(pass.version) => latest = Some(pass),
                Ok(_) => {} // passe périmée (navigation depuis) : ignorée
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        if let Some(pass) = latest {
            let img = egui::ColorImage::from_rgb([pass.w as usize, pass.h as usize], &pass.rgb);
            match &mut self.preview_texture {
                Some(tex) => tex.set(img, egui::TextureOptions::LINEAR),
                None => {
                    self.preview_texture =
                        Some(ctx.load_texture("studio-preview", img, egui::TextureOptions::LINEAR))
                }
            }
            if pass.cache.is_some() {
                self.orbit_cache = pass.cache;
            }
            if pass.is_final {
                self.preview_rendering = false;
                // Provisoire « keyframe finale » de la timeline : la cible est
                // exactement ce que la preview affiche (copie réduite) —
                // seulement si la vue EST la cible du projet (sinon on
                // collerait une image sans rapport sur la timeline du projet).
                if self.view_is_project_target() && !self.settings.lighting {
                    let (thumb, tw, th) = timeline::downscale_rgb_nearest(
                        &pass.rgb,
                        pass.w,
                        pass.h,
                        timeline::THUMB_MAX_H,
                    );
                    self.preview_thumb = Some((thumb, tw, th));
                    if let Some(last) = self.tl_slots.len().checked_sub(1) {
                        if matches!(self.tl_slots[last], ThumbSlot::Empty | ThumbSlot::Rgb { .. }) {
                            let (rgb, w, h) = self.preview_thumb.clone().unwrap();
                            self.apply_thumb(last, ThumbSlot::Rgb { rgb, w, h });
                        }
                    }
                }
            }
        }
        if disconnected && !self.preview_rendering {
            self.preview_rx = None;
        } else {
            self.preview_rx = Some(rx);
        }
    }

    // -----------------------------------------------------------------------
    // Timeline (G13)
    // -----------------------------------------------------------------------

    /// Nombre de maps de la timeline (n+1) : total du job en cours, sinon le
    /// projet sur disque, sinon la cible courante (estimation).
    fn timeline_len(&self) -> Option<u32> {
        if let Some(t) = self.job_total {
            return Some(t);
        }
        if let Some(m) = &self.project_manifest {
            return Some(m.video.keyframes + 1);
        }
        zoom_from_span_x(&self.view.sx)
            .ok()
            .and_then(|z| crate::video::keyframe_count(&z).ok())
            .map(|n| n + 1)
    }

    /// Dimensionne la timeline et pose les provisoires : keyframe 0 (mini
    /// rendu vue pleine, caché par type) et keyframe finale (copie réduite de
    /// la preview — la cible est déjà à l'écran).
    fn ensure_timeline(&mut self, len: usize) {
        if self.tl_slots.len() != len {
            self.tl_slots = (0..len).map(|_| ThumbSlot::Empty).collect();
            self.tl_textures = (0..len).map(|_| None).collect();
        }
        // Empreinte couleurs : recoloriser (textures seulement, canaux gardés).
        let stamp = (
            self.palette,
            self.color_repeat,
            self.color_space,
            self.outcoloring_idx,
            self.type_idx,
            self.settings.lighting,
            self.settings.lighting_beta.to_bits(),
        );
        if stamp != self.tl_stamp {
            self.tl_stamp = stamp;
            for t in &mut self.tl_textures {
                *t = None;
            }
            // La miniature RGB finale vient de la preview non éclairée et ne
            // possède plus les canaux nécessaires au relief. Ne pas la faire
            // passer pour une approximation de la sortie éclairée.
            if self.settings.lighting {
                if let Some(last) = self.tl_slots.last_mut() {
                    if matches!(last, ThumbSlot::Rgb { .. }) {
                        *last = ThumbSlot::Empty;
                    }
                }
            }
        }
        // Provisoire keyframe 0 : même centre/type/aspect que le projet (ou la
        // cible courante sans projet), avec les itérations plafonnées.
        let (ftype, type_id, seed, power, color_space, cx, cy, source_w, source_h, iters) =
            match &self.project_manifest {
                Some(m) => {
                    let Some(ftype) = FractalType::from_id(m.fractal.r#type) else {
                        return;
                    };
                    let defaults = default_params_for_type(ftype, 1, 1);
                    let seed = match (m.fractal.julia_re, m.fractal.julia_im) {
                        (Some(re), Some(im)) => Complex64::new(re, im),
                        (None, None) => defaults.seed,
                        _ => return,
                    };
                    (
                        ftype,
                        m.fractal.r#type,
                        seed,
                        m.fractal.multibrot_power.unwrap_or(defaults.multibrot_power),
                        m.color.color_space,
                        m.location.real.as_str(),
                        m.location.imag.as_str(),
                        m.image.width,
                        m.image.height,
                        m.fractal.iterations,
                    )
                }
                None => {
                    let (type_id, ftype) = self.current_type();
                    (
                        ftype,
                        type_id,
                        self.seed,
                        self.multibrot_power,
                        self.color_space,
                        self.view.cx.as_str(),
                        self.view.cy.as_str(),
                        self.settings.width,
                        self.settings.height,
                        self.settings.iterations,
                    )
                }
            };
        let first_key = format!(
            "{type_id}|{:x},{:x}|{:x}|{color_space:?}|{cx}|{cy}|{source_w}x{source_h}|{}",
            seed.re.to_bits(),
            seed.im.to_bits(),
            power.to_bits(),
            iters.clamp(50, 5000)
        );
        let cached_matches = self.first_thumb.as_ref().is_some_and(|(key, _)| *key == first_key);
        let pending_matches = self
            .first_thumb_rx
            .as_ref()
            .is_some_and(|(key, _, _)| *key == first_key);
        if !cached_matches && !pending_matches {
            // Annuler le mini-rendu périmé (en drag, la clé change à chaque
            // frame : sans cancel, des dizaines de rendus fantômes saturaient
            // le pool rayon — bug 2026-08-23) puis dropper son receiver.
            self.first_thumb = None;
            if let Some((_, _, cancel)) = self.first_thumb_rx.take() {
                cancel.store(true, Ordering::Relaxed);
            }
            if self.tl_slots.first().is_some_and(|slot| !slot.is_final()) {
                self.tl_slots[0] = ThumbSlot::Empty;
                self.tl_textures[0] = None;
            }
            if let Ok(p) =
                job::first_thumb_params(
                    ftype,
                    seed,
                    power,
                    color_space,
                    cx,
                    cy,
                    source_w,
                    source_h,
                    iters,
                )
            {
                let cancel = Arc::new(AtomicBool::new(false));
                self.first_thumb_rx =
                    Some((first_key.clone(), job::spawn_first_thumb(p, cancel.clone()), cancel));
            }
        }
        if let Some((key, slot)) = &self.first_thumb {
            if *key == first_key && matches!(self.tl_slots.first(), Some(ThumbSlot::Empty)) {
                let slot = slot.clone();
                self.apply_thumb(0, slot);
            }
        }
        // Provisoire keyframe finale depuis la preview.
        if let Some((rgb, w, h)) = &self.preview_thumb {
            if self.view_is_project_target()
                && !self.settings.lighting
                && len > 0
                && matches!(self.tl_slots[len - 1], ThumbSlot::Empty)
            {
                let slot = ThumbSlot::Rgb { rgb: rgb.clone(), w: *w, h: *h };
                self.apply_thumb(len - 1, slot);
            }
        }
    }

    /// Colorise une miniature avec les couleurs COURANTES du panneau
    /// (recolorisation in-memory, mêmes fonctions que tous les paths de
    /// sortie — invariant colorisation unique).
    fn thumb_rgb(&self, slot: &ThumbSlot) -> Option<(Vec<u8>, u32, u32)> {
        match slot {
            ThumbSlot::Empty => None,
            ThumbSlot::Rgb { rgb, w, h } => Some((rgb.clone(), *w, *h)),
            ThumbSlot::Channels {
                w,
                h,
                fractal_type,
                color_space: _,
                iter_max,
                iterations,
                zs,
                ..
            } => {
                let mut p = default_params_for_type(*fractal_type, *w, *h);
                p.color_space = self.color_space;
                p.iteration_max = *iter_max;
                p.color_mode = self.palette;
                p.color_repeat = self.color_repeat.max(1);
                p.out_coloring_mode = OutColoringMode::from_cli_name(self.current_outcoloring())
                    .unwrap_or(OutColoringMode::Smooth);
                let mut rgb = colorize_buffers(&p, iterations, zs, &[], &[], *w, *h);
                if self.settings.lighting {
                    crate::video::lighting::shade_rgb(
                        &mut rgb,
                        iterations,
                        zs,
                        *w as usize,
                        *h as usize,
                        *iter_max,
                        45.0,
                        self.settings.lighting_beta,
                    );
                }
                Some((rgb, *w, *h))
            }
        }
    }

    /// Clic sur une miniature : la preview saute à cette profondeur (centre du
    /// projet si présent — les miniatures lui appartiennent — sinon la vue).
    fn jump_to_keyframe(&mut self, k: u32) {
        if let Some(m) = &self.project_manifest {
            self.view.cx = m.location.real.clone();
            self.view.cy = m.location.imag.clone();
        }
        self.view.sx = timeline::span_at_keyframe(k);
        let iters = (self.settings.iterations as f64
            + self.settings.iterations_growth * k as f64)
            .round()
            .max(50.0) as u32;
        self.preview_iters = iters;
        self.orbit_cache = None;
        self.preview_dirty = true;
        self.scrub.close();
        self.status = format!("Aperçu keyframe {k} — zoom e{:.1}", k as f64 * 2f64.log10());
    }

    /// La courbe de vitesse a changé : recalage des positions de scrub et
    /// rappel que « Ré-assembler » suffit (la vitesse n'est consommée qu'à
    /// l'assemblage — aucune keyframe à recalculer).
    fn on_curve_edited(&mut self) {
        self.scrub_positions = None;
        if self.project_manifest.is_some() && !self.job_running {
            self.status =
                "Vitesse modifiée — « Ré-assembler » suffit (les keyframes restent valides)"
                    .into();
        }
    }

    // -----------------------------------------------------------------------
    // Scrub (G13)
    // -----------------------------------------------------------------------

    /// (Re)démarre le worker de scrub si le projet ou les couleurs ont changé.
    /// Le manifest passé au worker = géométrie du PROJET (les maps sur
    /// disque) + couleurs/éclairage COURANTS (ce qu'un Ré-assembler produira).
    fn ensure_scrub_worker(&mut self) -> bool {
        let Some(pm) = &self.project_manifest else {
            return false;
        };
        let dir = self.probed_dir.clone().unwrap_or_default();
        let fp = format!(
            "{}|{}|{}|{:?}|{}|{}|{}|{}",
            dir,
            self.palette,
            self.color_repeat,
            self.color_space,
            self.current_outcoloring(),
            self.settings.lighting,
            self.settings.lighting_beta,
            self.generation,
        );
        if self.scrub_tx.is_none() || fp != self.scrub_fingerprint {
            let mut m = pm.clone();
            m.color.palette = self.palette;
            m.color.color_repeat = self.color_repeat.max(1);
            m.color.color_space = self.color_space;
            m.color.outcoloring = self.current_outcoloring().to_string();
            m.lighting.enable = self.settings.lighting;
            m.lighting.beta = self.settings.lighting_beta;
            let (tx, rx) = job::spawn_scrub_worker(PathBuf::from(dir), m);
            self.scrub_tx = Some(tx); // l'ancien Sender droppe → worker exit
            self.scrub_rx = Some(rx);
            self.scrub_fingerprint = fp;
        }
        true
    }

    /// Positions p(frame) de l'assembleur pour le projet courant, avec la
    /// vitesse ÉDITÉE (fps/courbe du panneau) : c'est la timeline temporelle
    /// réelle de la vidéo qu'un Ré-assembler produirait.
    fn refresh_scrub_positions(&mut self) {
        let Some(pm) = &self.project_manifest else {
            self.scrub_positions = None;
            return;
        };
        let n = pm.video.keyframes;
        let fps = self.settings.fps.clamp(1, 120);
        let vel = job::compiled_velocity(&self.settings, n);
        let key = format!("{n}|{fps}|{vel}");
        if self.scrub_positions.as_ref().is_some_and(|(k, _)| *k == key) {
            return;
        }
        let positions = Dynamic::parse(&vel)
            .ok()
            .and_then(|d| crate::video::assemble::timeline(n, fps, &d).ok());
        self.scrub_positions = positions.map(|p| (key, p));
    }

    /// Demande la frame de scrub à la position `p` (keyframe fractionnaire).
    fn start_scrub(&mut self, p: f64) {
        let version = self.scrub.start(p);
        // Libellé temporel : frame la plus proche de p dans la timeline réelle.
        let fps = self.settings.fps.clamp(1, 120) as f64;
        let mut frame_idx = 0usize;
        let mut last_frame = 0usize;
        if let Some((_, positions)) = &self.scrub_positions {
            last_frame = positions.len().saturating_sub(1);
            frame_idx = nearest_position_index(positions, p);
            self.scrub_label = format!(
                "Frame {idx}/{} · t = {} · p = {p:.2}",
                positions.len().saturating_sub(1),
                fmt_duration(frame_idx as f64 / fps),
                idx = frame_idx,
            );
        } else {
            self.scrub_label = format!("p = {p:.2}");
        }
        if self.ensure_scrub_worker() {
            let (w, h) = self.panel_size;
            if let Some(tx) = &self.scrub_tx {
                let _ = tx.send(ScrubRequest {
                    p,
                    palette_offset: if self.settings.palette_scroll {
                        if last_frame == 0 {
                            0.0
                        } else {
                            self.settings.palette_cycles * frame_idx as f64 / last_frame as f64
                        }
                    } else {
                        0.0
                    },
                    out_w: w.max(64),
                    out_h: h.max(48),
                    version,
                });
            }
        } else {
            self.scrub_label = "Aucun projet rendu dans ce dossier — lancez Générer".into();
        }
    }

    fn drain_scrub_replies(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.scrub_rx else { return };
        let mut latest = None;
        while let Ok(reply) = rx.try_recv() {
            latest = Some(reply);
        }
        match latest {
            Some(ScrubReply::Frame { rgb, w, h, version, missing_next }) => {
                if !self.scrub.accepts(version) {
                    return; // réponse périmée (drag depuis)
                }
                let img = egui::ColorImage::from_rgb([w as usize, h as usize], &rgb);
                match &mut self.scrub_texture {
                    Some(tex) => tex.set(img, egui::TextureOptions::LINEAR),
                    None => {
                        self.scrub_texture = Some(ctx.load_texture(
                            "studio-scrub",
                            img,
                            egui::TextureOptions::LINEAR,
                        ))
                    }
                }
                if missing_next {
                    self.scrub_label.push_str("  (keyframe suivante pas encore rendue)");
                }
            }
            Some(ScrubReply::Missing { k, version }) => {
                if self.scrub.accepts(version) {
                    self.scrub_label = format!("Keyframe {k} pas encore rendue — Générer la produira");
                }
            }
            None => {}
        }
    }

    /// Zoom courant, formaté court : décimal simple en shallow, scientifique
    /// ensuite, mantisse tronquée de la string HP au-delà du f64 (deep).
    fn zoom_label(&self) -> String {
        match zoom_from_span_x(&self.view.sx) {
            Ok(z) => {
                let approx = Float::parse(&z)
                    .ok()
                    .map(|p| Float::with_val(128, p).to_f64());
                match approx {
                    Some(v) if v.is_finite() && v < 1e4 => format!("{v:.2}"),
                    Some(v) if v.is_finite() => format!("{v:.3e}"),
                    _ => match z.split_once('e') {
                        Some((mant, exp)) => format!("{}e{}", &mant[..mant.len().min(5)], exp),
                        None => z.chars().take(12).collect(),
                    },
                }
            }
            Err(_) => "?".into(),
        }
    }

    fn settings_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("🎬 Fractall Video Studio");
        ui.add_space(4.0);

        // --- Cible / preview ---
        egui::ComboBox::from_label("Type")
            .selected_text(STUDIO_TYPES[self.type_idx].2)
            .show_ui(ui, |ui| {
                for (i, (_, _, label)) in STUDIO_TYPES.iter().enumerate() {
                    if ui.selectable_value(&mut self.type_idx, i, *label).changed() {
                        let p = default_params_for_type(STUDIO_TYPES[i].1, 800, 600);
                        self.view = HpView::new(p.center_x, p.center_y, p.span_x);
                        self.color_space = p.color_space;
                        self.seed = p.seed;
                        self.multibrot_power = p.multibrot_power;
                        self.orbit_cache = None;
                        self.preview_dirty = true;
                    }
                }
            });
        ui.horizontal(|ui| {
            ui.label("Palette");
            if ui.add(egui::DragValue::new(&mut self.palette).range(0..=26)).changed() {
                self.preview_dirty = true;
            }
            ui.label("Repeat");
            if ui.add(egui::DragValue::new(&mut self.color_repeat).range(1..=120)).changed() {
                self.preview_dirty = true;
            }
        });
        ui.horizontal(|ui| {
            egui::ComboBox::from_label("Coloring")
                .selected_text(STUDIO_OUTCOLORINGS[self.outcoloring_idx].1)
                .show_ui(ui, |ui| {
                    for (i, (_, label)) in STUDIO_OUTCOLORINGS.iter().enumerate() {
                        if ui.selectable_value(&mut self.outcoloring_idx, i, *label).changed() {
                            self.preview_dirty = true;
                        }
                    }
                });
        });
        ui.horizontal(|ui| {
            ui.label("Iter. preview");
            if ui
                .add(egui::DragValue::new(&mut self.preview_iters).range(50..=1_000_000).speed(50))
                .changed()
            {
                self.preview_dirty = true;
            }
        });
        if ui.button("⟲ Vue pleine").clicked() {
            let p = default_params_for_type(self.current_type().1, 800, 600);
            self.view = HpView::new(p.center_x, p.center_y, p.span_x);
            self.orbit_cache = None;
            self.preview_dirty = true;
        }

        ui.separator();

        // --- Réglages vidéo ---
        egui::ComboBox::from_label("Résolution")
            .selected_text(if self.resolution_idx < RESOLUTIONS.len() {
                RESOLUTIONS[self.resolution_idx].0.to_string()
            } else {
                format!("{}×{} (custom)", self.settings.width, self.settings.height)
            })
            .show_ui(ui, |ui| {
                for (i, (label, w, h)) in RESOLUTIONS.iter().enumerate() {
                    if ui.selectable_value(&mut self.resolution_idx, i, *label).changed() {
                        self.settings.width = *w;
                        self.settings.height = *h;
                    }
                }
                ui.selectable_value(&mut self.resolution_idx, RESOLUTIONS.len(), "Custom…");
            });
        if self.resolution_idx >= RESOLUTIONS.len() {
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut self.settings.width).range(16..=7680));
                ui.label("×");
                ui.add(egui::DragValue::new(&mut self.settings.height).range(16..=4320));
                ui.label("(arrondi pair)");
            });
        }
        ui.horizontal(|ui| {
            ui.label("FPS");
            ui.add(egui::DragValue::new(&mut self.settings.fps).range(1..=120));
            ui.label("AA");
            egui::ComboBox::from_id_salt("ss")
                .selected_text(match self.settings.supersample {
                    1 => "Off",
                    2 => "2×",
                    _ => "3×",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.settings.supersample, 1, "Off");
                    ui.selectable_value(&mut self.settings.supersample, 2, "2×");
                    ui.selectable_value(&mut self.settings.supersample, 3, "3×");
                });
        });
        ui.add(
            egui::Slider::new(&mut self.settings.velocity, 0.05..=4.0)
                .logarithmic(true)
                .text("vitesse (×2/s)"),
        );
        if !self.settings.speed_points.is_empty() {
            ui.horizontal(|ui| {
                ui.label(format!(
                    "Courbe vitesse : {} point(s)",
                    self.settings.speed_points.len()
                ));
                if ui.small_button("✕ réinitialiser").clicked() {
                    self.settings.speed_points.clear();
                    self.on_curve_edited();
                }
            });
        }
        ui.horizontal(|ui| {
            ui.label("Iter. base");
            ui.add(egui::DragValue::new(&mut self.settings.iterations).range(50..=10_000_000).speed(50));
            ui.label("+/kf");
            ui.add(egui::DragValue::new(&mut self.settings.iterations_growth).range(0.0..=100_000.0));
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.settings.lighting, "Éclairage");
            ui.add_enabled(
                self.settings.lighting,
                egui::Slider::new(&mut self.settings.lighting_beta, 5.0..=90.0).text("β°"),
            );
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.settings.palette_scroll, "Palette animée");
            ui.add_enabled(
                self.settings.palette_scroll,
                egui::DragValue::new(&mut self.settings.palette_cycles)
                    .range(-20.0..=20.0)
                    .speed(0.1),
            );
            ui.label("cycles");
        });
        ui.horizontal(|ui| {
            ui.label("Dossier");
            ui.add(egui::TextEdit::singleline(&mut self.output_dir).desired_width(180.0));
        });
        // Glyphes limités aux polices egui par défaut (✓/→/🔁 = carrés).
        if self.ffmpeg_available {
            ui.label("ffmpeg détecté : sortie video.mp4");
        } else {
            ui.colored_label(egui::Color32::YELLOW, "ffmpeg introuvable : sortie frames PNG");
        }

        ui.separator();

        // --- Estimations live (avec la courbe de vitesse compilée) ---
        if let Ok(t) = self.target_view() {
            let n = crate::video::keyframe_count(&t.zoom).unwrap_or(1);
            let velocity = job::compiled_velocity(&self.settings, n);
            match job::estimates(
                &t.zoom,
                self.settings.fps,
                &velocity,
                self.settings.width,
                self.settings.height,
                self.settings.supersample,
            ) {
                Ok(e) => {
                    ui.label(format!("Zoom cible : {}", self.zoom_label()));
                    ui.label(format!(
                        "{} keyframes ({} maps, ~{} Mo bruts)",
                        e.keyframes,
                        e.maps,
                        e.map_bytes / 1_000_000
                    ));
                    ui.label(format!(
                        "{} frames — durée {}:{:04.1}",
                        e.frames,
                        (e.duration_s / 60.0) as u32,
                        e.duration_s % 60.0
                    ));
                }
                Err(e) => {
                    ui.colored_label(egui::Color32::YELLOW, e);
                }
            }
        }

        ui.add_space(6.0);

        // --- Job ---
        if self.job_running {
            ui.label(&self.job_phase);
            ui.add(egui::ProgressBar::new(self.job_progress).show_percentage().animate(true));
            if let Some(t0) = self.job_started {
                ui.label(format!("{:.0} s écoulées", t0.elapsed().as_secs_f64()));
            }
            // ETA pondérée (G13) : les keyframes profondes coûtent plus cher —
            // régression sur les durées mesurées, pas une moyenne naïve.
            if let Some(total) = self.job_total {
                let remaining: Vec<u32> =
                    (0..total).filter(|k| !self.tl_done.contains(k)).collect();
                if let Some(eta) = timeline::eta_seconds(&self.tl_measured, &remaining) {
                    if eta > 0.5 {
                        ui.label(format!("≈ restant : {}", fmt_duration(eta)));
                    }
                }
            }
            if self.cancel_requested {
                ui.label("Annulation…");
            } else if ui.button("■ Annuler").clicked() {
                self.job_cancel.store(true, Ordering::Relaxed);
                self.cancel_requested = true;
            }
        } else {
            let ready = !self.output_dir.trim().is_empty();
            ui.horizontal(|ui| {
                if ui.add_enabled(ready, egui::Button::new("🎬 Générer")).clicked() {
                    self.start_job(false);
                }
                let has_manifest =
                    Path::new(self.output_dir.trim()).join("manifest.toml").exists();
                if ui
                    .add_enabled(ready && has_manifest, egui::Button::new("⟲ Ré-assembler"))
                    .on_hover_text(
                        "Refait la vidéo depuis les keyframes existantes \
                         (palette / éclairage / vitesse) sans recalculer",
                    )
                    .clicked()
                {
                    self.start_job(true);
                }
            });
            if let Some(result) = &self.job_result {
                match result {
                    Ok(output) => {
                        ui.colored_label(egui::Color32::LIGHT_GREEN, format!("Terminé : {output}"));
                    }
                    Err(e) => {
                        ui.colored_label(egui::Color32::LIGHT_RED, format!("Échec : {e}"));
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Panneau timeline (G13)
    // -----------------------------------------------------------------------

    /// Timeline en bas : courbe de vitesse éditable, règle de scrub,
    /// miniatures des keyframes (placeholder → provisoire → définitive).
    fn timeline_panel(&mut self, ui: &mut egui::Ui) {
        let Some(total) = self.timeline_len() else {
            ui.weak("Timeline : naviguez vers une cible pour planifier la vidéo");
            return;
        };
        let n = total.saturating_sub(1);
        self.ensure_timeline(total as usize);
        self.refresh_scrub_positions();

        // Bandeau de désynchronisation : les miniatures appartiennent au
        // PROJET sur disque ; si le CENTRE de la vue a divergé, le dire.
        // (Le zoom n'est pas comparé : inspecter une keyframe intermédiaire
        // via un clic sur la timeline est une navigation légitime.)
        if self.project_manifest.is_some() && !self.view_center_matches_project() {
            ui.colored_label(
                egui::Color32::YELLOW,
                "⚠ La vue courante diffère du projet affiché — « Générer » re-planifiera sur la nouvelle cible",
            );
        }

        let aspect = self.settings.width.max(1) as f32 / self.settings.height.max(1) as f32;
        let cell_w = (TL_THUMB_H * aspect).clamp(40.0, 160.0).round() + TL_CELL_PAD;

        egui::ScrollArea::horizontal().id_salt("tl-scroll").show(ui, |ui| {
            let content_w = cell_w * total as f32;
            let content_h = TL_CURVE_H + TL_RULER_H + TL_THUMB_H + TL_LABEL_H;
            let (full_rect, _) = ui
                .allocate_exact_size(egui::vec2(content_w, content_h), egui::Sense::hover());
            let curve_rect = egui::Rect::from_min_size(
                full_rect.min,
                egui::vec2(content_w, TL_CURVE_H),
            );
            let ruler_rect = egui::Rect::from_min_size(
                egui::pos2(full_rect.min.x, curve_rect.max.y),
                egui::vec2(content_w, TL_RULER_H),
            );
            let thumbs_top = ruler_rect.max.y;

            self.draw_speed_curve(ui, curve_rect, n, cell_w);
            self.draw_scrub_ruler(ui, ruler_rect, full_rect, n, cell_w);
            self.draw_thumbs(ui, full_rect.min.x, thumbs_top, n, cell_w);

            // Curseur de scrub sur toute la hauteur.
            if self.scrub.active() {
                let x = full_rect.min.x + ((self.scrub.position() + 0.5) * cell_w as f64) as f32;
                ui.painter().vline(
                    x,
                    full_rect.y_range(),
                    egui::Stroke::new(2.0_f32, ui.visuals().warn_fg_color),
                );
            }
        });
    }

    /// Courbe de vitesse : ligne ×1 au centre, points éditables (double-clic =
    /// ajouter, glisser = déplacer, clic droit = supprimer), échelle log2
    /// ×1/8…×8. La courbe module la vitesse de base par POSITION de keyframe.
    fn draw_speed_curve(&mut self, ui: &mut egui::Ui, rect: egui::Rect, n: u32, cell_w: f32) {
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color.linear_multiply(0.5));
        let half = rect.height() * 0.5 - 4.0;
        let x_of = |p: f64| rect.min.x + ((p + 0.5) * cell_w as f64) as f32;
        let p_of =
            |x: f32| ((((x - rect.min.x) / cell_w) as f64) - 0.5).clamp(0.0, n.max(1) as f64);
        let y_of = |m: f64| {
            rect.center().y - (m.max(0.01).log2() as f32 / TL_CURVE_LOG_RANGE).clamp(-1.0, 1.0) * half
        };
        let m_of = |y: f32| {
            2f64.powf((((rect.center().y - y) / half) * TL_CURVE_LOG_RANGE) as f64)
                .clamp(0.125, 8.0)
        };

        // Repères ×1 (plein) et ×½ / ×2 (pointillés discrets).
        let weak = ui.visuals().weak_text_color();
        painter.hline(rect.x_range(), y_of(1.0), egui::Stroke::new(1.0_f32, weak));
        for m in [0.5, 2.0] {
            painter.hline(
                rect.x_range(),
                y_of(m),
                egui::Stroke::new(0.5_f32, weak.linear_multiply(0.4)),
            );
        }

        let curve = timeline::SpeedCurve { points: self.settings.speed_points.clone() };
        let accent = ui.visuals().selection.bg_fill;
        let steps = (rect.width() / 4.0).max(8.0) as usize;
        let pts: Vec<egui::Pos2> = (0..=steps)
            .map(|i| {
                let x = rect.min.x + rect.width() * i as f32 / steps as f32;
                egui::pos2(x, y_of(curve.multiplier_at(p_of(x))))
            })
            .collect();
        painter.add(egui::Shape::line(pts, egui::Stroke::new(1.5_f32, accent)));

        // Interactions.
        let resp = ui.interact(rect, ui.id().with("speed-curve"), egui::Sense::click_and_drag());
        let pointer = resp.interact_pointer_pos().or_else(|| resp.hover_pos());
        let near = pointer.and_then(|pp| {
            self.settings
                .speed_points
                .iter()
                .enumerate()
                .map(|(i, &(p, m))| (i, egui::pos2(x_of(p), y_of(m)).distance(pp)))
                .filter(|&(_, d)| d < 12.0)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(i, _)| i)
        });
        if resp.drag_started() {
            self.curve_drag = near;
        }
        if let (Some(i), Some(pp)) = (self.curve_drag, resp.interact_pointer_pos()) {
            if resp.dragged() && i < self.settings.speed_points.len() {
                // Position clampée entre les voisins : l'ordre reste trié.
                // Bornes = voisins (marge 0.05 si la place le permet, sinon
                // les voisins eux-mêmes) : `hi.max(lo)` pouvait pousser le
                // point AU-DELÀ du suivant quand deux voisins étaient à
                // < 0.1 → liste désordonnée (bug 2026-08-23).
                let prev = if i > 0 { self.settings.speed_points[i - 1].0 } else { 0.0 };
                let next = if i + 1 < self.settings.speed_points.len() {
                    self.settings.speed_points[i + 1].0
                } else {
                    n.max(1) as f64
                };
                let (lo, hi) = if next - prev > 0.1 { (prev + 0.05, next - 0.05) } else { (prev, next) };
                self.settings.speed_points[i] = (p_of(pp.x).clamp(lo, hi.max(lo)), m_of(pp.y));
                self.on_curve_edited();
            }
        }
        if resp.drag_stopped() {
            self.curve_drag = None;
        }
        if resp.double_clicked() {
            if let Some(pp) = resp.interact_pointer_pos() {
                let pt = (p_of(pp.x), m_of(pp.y));
                let at = self.settings.speed_points.partition_point(|&(p, _)| p < pt.0);
                self.settings.speed_points.insert(at, pt);
                self.on_curve_edited();
            }
        }
        if resp.secondary_clicked() {
            if let Some(i) = near {
                self.settings.speed_points.remove(i);
                self.curve_drag = None;
                self.on_curve_edited();
            }
        }

        // Points par-dessus la courbe.
        for &(p, m) in &self.settings.speed_points {
            let c = egui::pos2(x_of(p), y_of(m));
            painter.circle_filled(c, 4.0, accent);
            painter.circle_stroke(c, 4.0, egui::Stroke::new(1.0_f32, ui.visuals().strong_text_color()));
        }
        if self.settings.speed_points.is_empty() {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Vitesse : double-clic pour ajouter un point (ralenti/accéléré par zone) — Ré-assembler suffit",
                egui::FontId::proportional(11.0),
                weak,
            );
        } else if let Some(pp) = resp.hover_pos() {
            let m = curve.multiplier_at(p_of(pp.x));
            resp.on_hover_text(format!(
                "×{m:.2} à la keyframe {:.1}\nglisser = déplacer · clic droit = supprimer",
                p_of(pp.x)
            ));
        }
    }

    /// Règle de scrub : cliquer/glisser prévisualise la frame vidéo à cette
    /// position (interpolée par le MÊME code que l'assembleur).
    fn draw_scrub_ruler(
        &mut self,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        _full: egui::Rect,
        n: u32,
        cell_w: f32,
    ) {
        let painter = ui.painter_at(rect);
        let weak = ui.visuals().weak_text_color();
        painter.rect_filled(rect, 0.0, ui.visuals().faint_bg_color);
        for k in 0..=n {
            let x = rect.min.x + ((k as f64 + 0.5) * cell_w as f64) as f32;
            painter.vline(
                x,
                egui::Rangef::new(rect.max.y - 5.0, rect.max.y),
                egui::Stroke::new(1.0_f32, weak),
            );
        }
        let resp = ui.interact(rect, ui.id().with("tl-ruler"), egui::Sense::click_and_drag());
        if (resp.clicked() || resp.dragged()) && self.panel_size.0 > 0 {
            if let Some(pp) = resp.interact_pointer_pos() {
                let p = ((((pp.x - rect.min.x) / cell_w) as f64) - 0.5).clamp(0.0, n as f64);
                self.start_scrub(p);
            }
        }
        resp.on_hover_text("Scrub : prévisualise la frame vidéo à cette position");
    }

    /// Rangée de miniatures : image si disponible, cadre numéroté sinon,
    /// surbrillance de la keyframe en cours, libellé de profondeur, clic =
    /// aperçu à cette échelle.
    fn draw_thumbs(&mut self, ui: &mut egui::Ui, left: f32, top: f32, n: u32, cell_w: f32) {
        let mut created = 0usize;
        let mut jump: Option<u32> = None;
        let visuals = ui.visuals().clone();
        for k in 0..=n as usize {
            let cell = egui::Rect::from_min_size(
                egui::pos2(left + k as f32 * cell_w, top),
                egui::vec2(cell_w - TL_CELL_PAD, TL_THUMB_H),
            );
            // Ne créer les textures que pour les cellules visibles, par lots.
            if ui.is_rect_visible(cell)
                && self.tl_textures[k].is_none()
                && created < TL_MAX_TEXTURES_PER_FRAME
            {
                if let Some((rgb, w, h)) = self.thumb_rgb(&self.tl_slots[k]) {
                    let img = egui::ColorImage::from_rgb([w as usize, h as usize], &rgb);
                    self.tl_textures[k] = Some(ui.ctx().load_texture(
                        format!("tl-thumb-{k}"),
                        img,
                        egui::TextureOptions::LINEAR,
                    ));
                    created += 1;
                }
            }
            let painter = ui.painter();
            match &self.tl_textures[k] {
                Some(tex) => {
                    painter.image(
                        tex.id(),
                        cell,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                    // Provisoire : voile + « ≈ » (remplacé par la vraie map).
                    if !self.tl_slots[k].is_final() {
                        painter.rect_filled(
                            cell,
                            0.0,
                            egui::Color32::from_black_alpha(60),
                        );
                        painter.text(
                            cell.right_top() + egui::vec2(-4.0, 2.0),
                            egui::Align2::RIGHT_TOP,
                            "≈",
                            egui::FontId::proportional(12.0),
                            egui::Color32::WHITE,
                        );
                    }
                }
                None => {
                    painter.rect_stroke(
                        cell,
                        2.0,
                        egui::Stroke::new(1.0_f32, visuals.weak_text_color()),
                        egui::StrokeKind::Inside,
                    );
                    painter.text(
                        cell.center(),
                        egui::Align2::CENTER_CENTER,
                        format!("{k}"),
                        egui::FontId::proportional(11.0),
                        visuals.weak_text_color(),
                    );
                }
            }
            if self.tl_current == Some(k as u32) {
                painter.rect_stroke(
                    cell,
                    2.0,
                    egui::Stroke::new(2.0_f32, visuals.selection.bg_fill),
                    egui::StrokeKind::Outside,
                );
            }
            // Profondeur : zoom à la keyframe k = 2^k = 10^(k·log10 2).
            painter.text(
                egui::pos2(cell.center().x, cell.max.y + 2.0),
                egui::Align2::CENTER_TOP,
                format!("e{:.0}", k as f64 * 2f64.log10()),
                egui::FontId::proportional(9.0),
                visuals.weak_text_color(),
            );

            let resp = ui.interact(cell, ui.id().with(("tl-thumb", k)), egui::Sense::click());
            if resp.clicked() {
                jump = Some(k as u32);
            }
            resp.on_hover_text(format!(
                "Keyframe {k} — zoom e{:.1}\nClic : aperçu à cette profondeur",
                k as f64 * 2f64.log10()
            ));

            // Auto-scroll vers la keyframe en cours de rendu.
            if self.tl_current == Some(k as u32) && self.tl_autoscrolled != self.tl_current {
                ui.scroll_to_rect(cell, Some(egui::Align::Center));
                self.tl_autoscrolled = self.tl_current;
            }
        }
        if let Some(k) = jump {
            self.jump_to_keyframe(k);
        }
    }

    fn preview_panel(&mut self, ui: &mut egui::Ui) {
        let avail = ui.available_size();
        let (w, h) = ((avail.x as u32).clamp(64, 1600), (avail.y as u32).clamp(48, 1200));
        // Re-rendu seulement sur variation notable (le resize continu d'une
        // fenêtre relancerait un rendu par frame).
        let (pw, ph) = self.panel_size;
        if pw.abs_diff(w) > 24 || ph.abs_diff(h) > 24 {
            self.panel_size = (w, h);
            self.preview_dirty = true;
        } else if self.panel_size == (0, 0) {
            self.panel_size = (w, h);
            self.preview_dirty = true;
        }

        let rect = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());

        // Mode scrub (G13) : la zone centrale montre la FRAME VIDÉO interpolée
        // (même code que l'assembleur). Toute interaction de navigation
        // (clic, glisser, molette) rend la main à la preview live.
        if self.scrub.active() {
            if let Some(tex) = &self.scrub_texture {
                ui.scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| {
                    ui.add(egui::Image::new(tex).fit_to_exact_size(rect.size()));
                });
            } else {
                ui.put(rect, egui::Spinner::new());
            }
            let painter = ui.painter_at(rect);
            let label = format!("🎞 {}  —  clic : retour à la navigation", self.scrub_label);
            painter.rect_filled(
                egui::Rect::from_min_size(rect.min, egui::vec2(rect.width(), 22.0)),
                0.0,
                egui::Color32::from_black_alpha(140),
            );
            painter.text(
                rect.min + egui::vec2(8.0, 4.0),
                egui::Align2::LEFT_TOP,
                label,
                egui::FontId::proportional(13.0),
                egui::Color32::WHITE,
            );
            let wheel = ui.ctx().input(|i| i.smooth_scroll_delta.y) != 0.0;
            if response.clicked() || response.dragged() || (response.hovered() && wheel) {
                self.scrub.close();
            }
            return;
        }

        if let Some(tex) = &self.preview_texture {
            ui.scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| {
                ui.add(egui::Image::new(tex).fit_to_exact_size(rect.size()));
            });
        } else {
            ui.put(rect, egui::Spinner::new());
        }

        let aspect = h as f64 / w as f64;
        // Molette : zoom ancré au curseur.
        if response.hovered() {
            let scroll = ui.ctx().input(|i| i.smooth_scroll_delta.y) as f64;
            if scroll != 0.0 {
                if let Some(pos) = response.hover_pos() {
                    let cursor = (
                        ((pos.x - rect.min.x) / rect.width()) as f64,
                        ((pos.y - rect.min.y) / rect.height()) as f64,
                    );
                    let factor = 1.2f64.powf((scroll / 50.0).clamp(-3.0, 3.0));
                    if let Some(v) = nav::zoom_anchored(&self.view, cursor, aspect, factor) {
                        self.view = v;
                        self.preview_dirty = true;
                    }
                }
            }
        }
        // Glisser : pan (l'image suit la souris).
        if response.dragged() {
            let d = response.drag_delta();
            if d != egui::Vec2::ZERO {
                let drag = ((d.x / rect.width()) as f64, (d.y / rect.height()) as f64);
                if let Some(v) = nav::pan(&self.view, drag, aspect) {
                    self.view = v;
                    self.preview_dirty = true;
                }
            }
        }
    }
}

impl eframe::App for VideoStudioApp {
    // eframe 0.34+ exige `ui` sur le trait App ; comme le générateur, toute
    // la logique par panneaux reste dans `update` (stub `ui` vide).
    fn ui(&mut self, _ui: &mut egui::Ui, _frame: &mut eframe::Frame) {}

    #[allow(deprecated)]
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.probe_project();
        self.drain_job_messages();
        self.drain_scan_messages();
        self.drain_first_thumb();
        self.drain_scrub_replies(ctx);
        self.drain_preview_messages(ctx);

        // Drag-and-drop : PNG fractall ou .fmap → adoption de la cible.
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect()
        });
        if let Some(path) = dropped.first() {
            self.adopt_file(path);
        }

        egui::SidePanel::left("settings")
            .resizable(false)
            .default_width(320.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.settings_panel(ui));
            });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("Zoom : {}", self.zoom_label()));
                if self.preview_rendering {
                    ui.spinner();
                    ui.label("preview…");
                }
                ui.separator();
                ui.label(&self.status);
            });
        });

        // Timeline (G13) : courbe de vitesse + scrub + miniatures, au-dessus
        // de la barre de statut.
        egui::TopBottomPanel::bottom("timeline")
            .resizable(false)
            .show(ctx, |ui| self.timeline_panel(ui));

        egui::CentralPanel::default().show(ctx, |ui| self.preview_panel(ui));

        // Relance de la preview si la vue/les réglages ont changé. Pendant un
        // drag, chaque relance annule la précédente — la passe ¼ étant quasi
        // instantanée, la navigation reste fluide.
        if self.preview_dirty {
            self.start_preview_render();
        }

        if self.preview_rendering
            || self.job_running
            || self.scan_rx.is_some()
            || self.first_thumb_rx.is_some()
        {
            ctx.request_repaint();
        }
    }
}

// Ré-export pour le binaire-enveloppe.
pub use VideoStudioApp as App;

#[cfg(test)]
mod tests {
    use super::*;

    /// La table des types du studio est cohérente : chaque id CLI redonne le
    /// même FractalType via from_id (pas de dérive de la table).
    #[test]
    fn studio_types_ids_match_from_id() {
        for (id, t, label) in STUDIO_TYPES {
            assert_eq!(
                FractalType::from_id(*id),
                Some(*t),
                "id {id} ({label}) ne correspond pas"
            );
        }
    }

    /// Tous les outcolorings proposés sont des noms CLI valides.
    #[test]
    fn studio_outcolorings_are_valid_cli_names() {
        for (name, _) in STUDIO_OUTCOLORINGS {
            let mode = OutColoringMode::from_cli_name(name)
                .unwrap_or_else(|| panic!("outcoloring invalide: {name}"));
            assert_eq!(studio_outcoloring_index(mode).map(|i| STUDIO_OUTCOLORINGS[i].0), Some(*name));
        }
        assert_eq!(studio_outcoloring_index(OutColoringMode::Distance), None);
    }

    #[test]
    fn scrub_nearest_frame_handles_reversing_timeline() {
        let positions = [0.0, 0.8, 1.6, 1.1, 0.7, 1.4];
        assert_eq!(nearest_position_index(&positions, 1.05), 3);
        assert_eq!(nearest_position_index(&positions, 1.45), 5);
        assert_eq!(nearest_position_index(&[], 1.0), 0);
    }

    /// Les résolutions presets sont toutes paires (contrainte x264 yuv420p).
    #[test]
    fn preset_resolutions_are_even() {
        for (label, w, h) in RESOLUTIONS {
            assert_eq!((w % 2, h % 2), (0, 0), "résolution impaire: {label}");
            assert_eq!(job::even_dims(*w, *h), (*w, *h));
        }
    }
}
