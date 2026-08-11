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

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

use rug::Float;

use crate::fractal::perturbation::ReferenceOrbitCache;
use crate::fractal::{default_params_for_type, FractalParams, FractalType, OutColoringMode};
use crate::io::fmap::load_fmap;
use crate::io::png::{colorize_to_rgb, load_png_metadata};
use crate::render::render_escape_time_cancellable_with_reuse;
use crate::video::zoom_from_span_x;
use job::{StudioSettings, TargetView, VideoJobMsg};
use nav::HpView;

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

/// Résolutions presets (label, w, h) + entrée custom.
const RESOLUTIONS: &[(&str, u32, u32)] = &[
    ("640×360", 640, 360),
    ("1280×720", 1280, 720),
    ("1920×1080", 1920, 1080),
    ("2560×1440", 2560, 1440),
    ("3840×2160 (4K)", 3840, 2160),
];

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
    preview_iters: u32,

    // --- Rendu preview asynchrone ---
    preview_version: u64,
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
            preview_iters: 1000,
            preview_version: 0,
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
        self.preview_version += 1;
        let version = self.preview_version;
        let cancel = self.preview_cancel.clone();
        let (tx, rx) = mpsc::channel();
        self.preview_rx = Some(rx);
        self.preview_rendering = true;
        self.preview_dirty = false;

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
                let Some((it, zs, _orbits, _dist)) = render_escape_time_cancellable_with_reuse(
                    &p, &cancel, None, &mut oc, None, None,
                ) else {
                    return;
                };
                cache = oc;
                let rgb = colorize_to_rgb(&p, &it, &zs);
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
            palette: self.palette,
            color_repeat: self.color_repeat,
            outcoloring: self.current_outcoloring().to_string(),
        })
    }

    fn start_job(&mut self, assemble_only: bool) {
        let target = match self.target_view() {
            Ok(t) => t,
            Err(e) => {
                self.status = e;
                return;
            }
        };
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
    }

    fn drain_job_messages(&mut self) {
        let Some(rx) = self.job_rx.take() else { return };
        let mut finished = false;
        loop {
            match rx.try_recv() {
                Ok(VideoJobMsg::RenderProgress { done, total }) => {
                    self.job_phase = format!("Keyframes {done}/{total}");
                    // Deux phases : rendu 0→50 %, assemblage 50→100 %.
                    self.job_progress = 0.5 * done as f32 / total.max(1) as f32;
                }
                Ok(VideoJobMsg::AssembleProgress { frame, total }) => {
                    self.job_phase = format!("Assemblage {frame}/{total}");
                    self.job_progress = 0.5 + 0.5 * frame as f32 / total.max(1) as f32;
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
        } else {
            self.job_rx = Some(rx);
        }
    }

    fn drain_preview_messages(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.preview_rx.take() else { return };
        let mut latest: Option<PreviewPass> = None;
        let mut disconnected = false;
        loop {
            match rx.try_recv() {
                Ok(pass) if pass.version == self.preview_version => latest = Some(pass),
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
            }
        }
        if disconnected && !self.preview_rendering {
            self.preview_rx = None;
        } else {
            self.preview_rx = Some(rx);
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

        // --- Estimations live ---
        if let Ok(t) = self.target_view() {
            match job::estimates(
                &t.zoom,
                self.settings.fps,
                self.settings.velocity,
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
        self.drain_job_messages();
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

        egui::CentralPanel::default().show(ctx, |ui| self.preview_panel(ui));

        // Relance de la preview si la vue/les réglages ont changé. Pendant un
        // drag, chaque relance annule la précédente — la passe ¼ étant quasi
        // instantanée, la navigation reste fluide.
        if self.preview_dirty {
            self.start_preview_render();
        }

        if self.preview_rendering || self.job_running {
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
            assert!(
                OutColoringMode::from_cli_name(name).is_some(),
                "outcoloring invalide: {name}"
            );
        }
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
