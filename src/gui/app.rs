use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use egui::{Context, TextureHandle, TextureOptions};
use image::RgbImage;
use num_complex::Complex64;
use rug::Float;

use crate::color::generate_palette_preview;
use crate::fractal::perturbation::ReferenceOrbitCache;
use crate::fractal::xaos::{self, XaosSourceFrame};
use crate::fractal::{
    apply_lyapunov_preset, default_params_for_type, AlgorithmMode, FractalParams, FractalType,
    LyapunovPreset, OutColoringMode, PlaneTransform, ViewHp,
};
use crate::gpu::GpuRenderer;
use crate::gui::hq_render_state::{HqRenderEvent, HqRenderResult, HqRenderState};
use crate::gui::nav;
use crate::gui::progressive::{upscale_nearest, ProgressiveConfig, RenderMessage};
use crate::gui::sage::{self, IterAction};
use crate::gui::texture::rgb_image_to_color_image;
use crate::gui::view_history::{self, ViewHistory};
use crate::render::{render_planned, render_request, PlannedRenderRequest, RenderRequest};

/// Précision par défaut pour les calculs de coordonnées haute précision (en bits).
const HP_PRECISION: u32 = 256;

/// Snapshot d'une vue (centre + span en HP strings) + axialité, pour le warp GPU
/// de la dernière frame (G10.1). Les strings préservent la précision au deep zoom
/// (le warp calcule `(cx_tex - cx_live)/span_live`, un ratio O(1), en HP puis f64).
#[derive(Clone, PartialEq)]
struct ViewSnapshot {
    cx: String,
    cy: String,
    sx: String,
    sy: String,
    /// Vrai si rotation=0 ET transform_k=None : la transformée frame→frame est
    /// alors séparable (warp valide). Sinon on ne warpe pas (fallback affichage brut).
    axis_aligned: bool,
}

/// Cœur (pur, testable) du warp GPU G10.1. À partir des vues HP texture/live,
/// renvoie `(n_cx, v_cy, rx, ry)` : centre normalisé (u,v) où placer la texture
/// dans la vue live + échelle (span_tex/span_live) en x/y. `None` si parse HP
/// impossible, span live nul, non-fini, ou vues identiques (pas de warp utile).
///
/// Convention : le renderer mappe `j → c_im = centre + ((j+0.5)/h − 0.5)·span`
/// (NON inversé : ligne 0 = haut d'écran = imag le plus PETIT, l'imag croît vers
/// le bas — même convention que l'écho XaoS). Donc x ET y sont symétriques :
/// `u = 0.5 + (c_re − centre_re)/span_re` (droite) et
/// `v = 0.5 + (c_im − centre_im)/span_im` (bas). Un `-dy` briserait cette
/// symétrie et mal-placerait le warp verticalement sur un zoom ancré hors-centre.
fn compute_warp_norm(tex: &ViewSnapshot, live: &ViewSnapshot) -> Option<(f64, f64, f64, f64)> {
    let tex = ViewHp::from_decimal_parts(&tex.cx, &tex.cy, &tex.sx, &tex.sy, 1, 1, HP_PRECISION)?;
    let live =
        ViewHp::from_decimal_parts(&live.cx, &live.cy, &live.sx, &live.sy, 1, 1, HP_PRECISION)?;
    // Placement de la texture (cible) dans la vue live (source du repère).
    let transform = live.transform_to(&tex);
    let (dx, dy, rx, ry) = (
        transform.offset_x,
        transform.offset_y,
        transform.scale_x,
        transform.scale_y,
    );
    if ![dx, dy, rx, ry].iter().all(|v| v.is_finite()) {
        return None;
    }
    // Vues identiques → pas de warp (rendu terminé ou pas encore bougé).
    if dx.abs() < 1e-9 && dy.abs() < 1e-9 && (rx - 1.0).abs() < 1e-9 && (ry - 1.0).abs() < 1e-9 {
        return None;
    }
    Some((0.5 + dx, 0.5 + dy, rx, ry))
}

/// Colorise un buffer d'itérations en RGB.
/// Cette fonction est appelée dans le thread de rendu ou dans l'UI pour re-coloriser avec la palette actuelle.
///
/// ⚠️ Enveloppe de `io::png::colorize_buffers` (implémentation UNIQUE partagée
/// avec le CLI / `--from-map` / le pipeline vidéo). Ne PAS réintroduire une
/// boucle de colorisation ici : les deux copies historiques avaient divergé
/// (canaux distances/orbites et Anti-Buddhabrot perdus côté CLI).
fn colorize_buffer(
    iterations: &[u32],
    zs: &[Complex64],
    distances: &[f64],
    orbits: &[Option<crate::fractal::orbit_traps::OrbitData>],
    params: &FractalParams,
    width: u32,
    height: u32,
) -> Vec<u8> {
    crate::io::png::colorize_buffers(params, iterations, zs, distances, orbits, width, height)
}

/// Mode de rendu CPU effectif (pour le label de statut uniquement). Réplique la
/// décision du dispatcher unifié `render_request` :
/// en Auto, perturbation pour les types escape-time supportés (plan Mu), sinon
/// GMP reference au-delà de ~1e16, sinon f64 standard.
fn effective_cpu_mode(plan: crate::render::CpuRenderPlan) -> AlgorithmMode {
    use crate::fractal::wisdom;
    match plan.wisdom().algorithm {
        wisdom::Algorithm::Perturbation => AlgorithmMode::Perturbation,
        wisdom::Algorithm::ReferenceGmp => AlgorithmMode::ReferenceGmp,
        wisdom::Algorithm::StandardF64 => AlgorithmMode::StandardF64,
    }
}

/// Étiquette de précision/algorithme CPU pour la barre de statut.
fn cpu_precision_label(params: &FractalParams, mode: AlgorithmMode) -> String {
    match mode {
        AlgorithmMode::ReferenceGmp => format!(
            "CPU GMP {}b",
            crate::fractal::perturbation::compute_perturbation_precision_bits(params)
        ),
        _ => "CPU f64".to_string(),
    }
}

/// Application principale egui pour fractall.
pub struct FractallApp {
    // État fractale
    params: FractalParams,
    // G10.3 : buffers bruts partagés via Arc → la recolorisation (thread) clone
    // un Arc (quasi-gratuit) au lieu de ~74 Mo sur le thread UI par tick de palette.
    iterations: Arc<Vec<u32>>,
    zs: Arc<Vec<Complex64>>,
    distances: Arc<Vec<f64>>,
    orbits: Arc<Vec<Option<crate::fractal::orbit_traps::OrbitData>>>,

    // Texture egui pour l'affichage
    texture: Option<TextureHandle>,
    displayed_frame: Option<DisplayedFrame>,
    rendering_params: FractalParams,
    raw_preview: bool,
    /// Overlay partiel de navigation (RGBA, transparent là où le rendu en cours
    /// n'a pas encore calculé) + la vue à laquelle il correspond. Composé
    /// PAR-DESSUS la texture principale warpée : port de la « dynamic
    /// resolution » XaoS, cf. `RenderMessage::NavProgress`.
    nav_overlay_texture: Option<TextureHandle>,
    nav_overlay_view: Option<ViewSnapshot>,
    /// Vue (centre/span HP) que représente la texture actuellement affichée.
    /// Sert au warp GPU (G10.1) : pendant un rendu, la texture est encore l'ANCIENNE
    /// vue ; on la transforme par (view_texture → view_live) pour un aperçu fluide.
    texture_view: Option<ViewSnapshot>,
    /// Vue du rendu en cours (posée à `start_render`), copiée dans `texture_view`
    /// au chargement d'une texture de rendu (pas de recolorisation).
    rendering_view: Option<ViewSnapshot>,

    // Cache des textures de prévisualisation des palettes
    palette_preview_textures: [Option<TextureHandle>; 27],

    // État UI
    selected_type: FractalType,
    palette_index: u8,
    color_repeat: u32,
    out_coloring_mode: OutColoringMode,
    selected_lyapunov_preset: LyapunovPreset,
    /// G4 jalon 5c : buffer d'édition de la séquence de phases hybride
    /// (menu Type → « Hybride »). Appliqué → `params.hybrid_phases`.
    hybrid_seq: Vec<FractalType>,
    gpu_renderer: Option<Arc<GpuRenderer>>,
    /// N'a pas encore tenté de créer le GPU (init différée après ouverture de la fenêtre)
    gpu_init_attempted: bool,
    use_gpu: bool,

    /// Source de vérité géométrique possédée par le contrôleur GUI.
    view: ViewHp,

    // Sélection rectangulaire pour zoom
    selection: crate::gui::selection::SelectionState,

    // Rendu progressif
    rendering: bool,
    render_thread: Option<thread::JoinHandle<()>>,
    render_cancel: Arc<AtomicBool>,
    render_receiver: Option<mpsc::Receiver<RenderMessage>>,
    /// Canal pour recevoir les textures pré-colorisées (calcul hors thread principal).
    texture_ready_sender: Option<mpsc::Sender<TextureReadyMessage>>,
    texture_ready_receiver: Option<mpsc::Receiver<TextureReadyMessage>>,
    render_progress: crate::gui::render_state::RenderProgress,
    is_preview: bool,
    /// Nombre d'échantillons anti-aliasing (per-frame jitter). 1 = désactivé.
    /// Après les passes progressives, N-1 samples plein cadre supplémentaires
    /// sont rendus avec offset sous-pixel et moyennés en RGB (CPU uniquement).
    aa_samples: u32,
    /// Amplitude du filtre tente sous-pixel, dans l'intervalle 0..=1 pixel.
    aa_jitter_scale: f64,
    /// Retour visuel bref après une copie presse-papiers (barre d'état).
    clipboard_feedback: Option<(&'static str, Instant)>,
    /// Vrai pendant que la texture principale montre une accumulation AA de ce rendu.
    showing_render_aa: bool,

    // Métriques
    last_render_time: Option<f64>, // en secondes
    render_start_time: Option<Instant>,
    last_render_device_label: Option<String>,
    last_render_method_label: Option<String>,

    // Cache for orbit/BLA to accelerate deep zoom re-renders
    orbit_cache: Option<Arc<ReferenceOrbitCache>>,

    // G10.4 : réutilisation pixels inter-frame XaoS.
    /// Frame source (buffers bruts Arc de la dernière passe CPU complétée +
    /// vue HP + positions vraies par axe). Mise à jour à chaque PassComplete
    /// CPU ; les passes GPU ne la remplacent pas (précision f32).
    xaos: crate::gui::xaos_state::XaosLifecycle,
    /// La dernière passe finale a copié des pixels (approximés ≤ 0.5 px) →
    /// un rendu de raffinement exact est programmé à l'idle (~400 ms).
    /// Fin du dernier rendu (référence du délai idle de raffinement).
    last_render_finished: Option<Instant>,
    /// G10.5 : dernière position du curseur au-dessus de l'image, normalisée
    /// [0,1]² — point de priorité de la file de tuiles (centre si None).
    hover_norm: Option<(f32, f32)>,

    // Fenêtre de rendu haute résolution
    show_render_dialog: bool,
    render_resolution_preset: RenderResolutionPreset,
    hq_render_state: HqRenderState,
    hq_render_receiver: Option<mpsc::Receiver<HqRenderEvent>>,

    // Canal dédié pour les recolorisations asynchrones (changement palette/color_repeat)
    // Persiste toute la vie de l'application, contrairement à texture_ready_* qui n'existe que pendant le rendu
    recolor_sender: mpsc::Sender<RecolorReadyMessage>,
    recolor_receiver: mpsc::Receiver<RecolorReadyMessage>,
    /// Compteur de version pour ignorer les recolorisations obsolètes (si l'utilisateur change le slider rapidement)
    recolor_version: crate::gui::async_version::AsyncVersion,

    // Zone de texte pour le nombre d'itérations
    iteration_input: String,
    /// True si le champ Itérations avait le focus la frame précédente (pour ne pas intercepter "0" en raccourci reset)
    iteration_input_has_focus: bool,

    // Mini-Julia preview (shown when hovering over Mandelbrot)
    julia_preview_enabled: bool,
    julia_preview_texture: Option<TextureHandle>,
    julia_preview_sender: mpsc::Sender<JuliaPreviewMessage>,
    julia_preview_receiver: mpsc::Receiver<JuliaPreviewMessage>,
    julia_preview_version: crate::gui::async_version::AsyncVersion,
    julia_preview_last_seed: Option<Complex64>,
    julia_preview_last_request_time: Option<Instant>,
    julia_preview_cancel: Arc<AtomicBool>,
    julia_preview_rendering: bool,

    /// Navigation « à la XaoS » (case de la barre d'outils, off par défaut) :
    /// bouton maintenu = zoom CONTINU ancré au curseur (gauche = in, droit =
    /// out) au lieu du clic ×2 / de la sélection rectangle.
    xaos_nav_mode: bool,
    /// Itérations automatiques (`wisdom_iter`, couche 1 par image) : état
    /// GUI, jamais sérialisé (seule la valeur résolue va dans
    /// `params.iteration_max`). Coupé dès que l'utilisateur tape un nombre.
    iter_auto: bool,
    /// Anti-boucle des re-rendus automatiques (croissances consécutives).
    auto_grow_guard: crate::gui::sage::AutoGrowGuard,
    /// Plafond abaissé par le sage, appliqué au PROCHAIN rendu (pas de
    /// re-rendu pour ça) : tant qu'il attend, `params.iteration_max` reste
    /// celui de l'image affichée (barre d'état et « Copy CLI » exacts).
    pending_iter_shrink: Option<u32>,
    /// Passe finale du rendu EN COURS (posée au PassComplete final, traitée
    /// à AllComplete) et celle du dernier rendu terminé (réévaluation quand
    /// l'utilisateur recoche Auto).
    pending_final_pass: Option<(crate::gui::sage::FinalPassInfo, Arc<Vec<u32>>)>,
    last_final_pass: Option<(crate::gui::sage::FinalPassInfo, Arc<Vec<u32>>)>,
    /// Avis du sage affiché quelques secondes dans la barre d'état.
    sage_notice: crate::gui::sage::SageNotice,
    /// Instant du dernier tick de zoom continu — cadence indépendante du
    /// framerate (`facteur = RATE^(dir·dt)`). `None` = bouton relâché.
    xaos_nav_last_tick: Option<Instant>,
    /// Bouton maintenu en cours (drag de navigation actif) : sélectionne la
    /// config de rendu « navigation » (passe unique) dans `start_render_with`.
    xaos_nav_dragging: bool,
    /// Pan au bouton du MILIEU maintenu (commencé sur l'image) : la vue suit
    /// la souris sans zoomer, rendus de navigation enchaînés, rendu pleine
    /// résolution au relâchement.
    middle_pan_active: bool,
    /// Hauteur MAXIMALE vue de chaque barre (haut, état, progression) pour la largeur de
    /// fenêtre courante : `(largeur, hauteur)`. Une barre qui passe à la
    /// ligne puis revient (texte « Computing… » pendant le rendu) changerait
    /// la taille de l'image, donc relancerait le rendu, à l'infini.
    bar_heights: [(f32, f32); 3],
    /// Machine à états de la navigation continue (vitesse, ancrage, cadence).
    /// Logique PURE et testée dans `gui::nav` — cf. son en-tête pour le pourquoi.
    nav: nav::NavState,
    /// Rendu de STABILISATION (bouton relâché, vue figée) : navigation encore,
    /// donc passe unique + streaming partiel, mais résolution PLEINE — la
    /// résolution dynamique ne s'applique qu'au mouvement.
    xaos_nav_settle: bool,
    /// Diviseur de résolution du rendu en cours / du dernier terminé — sert à
    /// extrapoler le coût plein cadre depuis `last_render_time` (coût ∝ d²).
    rendering_divisor: u8,
    last_render_divisor: u8,

    /// FRACTALL_UI_TRACE=1 : trace des à-coups du thread UI (lu une fois).
    ui_trace: bool,
    /// Début de la frame UI précédente (mesure de l'intervalle réel).
    ui_frame_last: Option<Instant>,
    /// Événements notables de la frame en cours (rendu lancé, texture chargée…).
    ui_frame_events: Vec<String>,

    /// Historique de navigation (retour / avance) — logique pure dans
    /// `gui::view_history`, nourrie à chaque frame (`observe`) et à chaque
    /// changement de scène (`on_new_scene` → `commit`).
    history: ViewHistory,
    /// Taille intérieure de la fenêtre (points), relevée à chaque frame et
    /// sauvegardée avec la session.
    window_size: Option<[f32; 2]>,
    /// Fichier de session écrit à la fermeture (`None` : pas de HOME).
    session_path: Option<std::path::PathBuf>,
}

/// Seuil de signalement d'un trou de frame UI (ms) sous `FRACTALL_UI_TRACE`.
const UI_TRACE_GAP_MS: f64 = 40.0;

/// Cadence de publication des frames partielles de navigation (~25 FPS, milieu
/// de la fenêtre XaoS 15-35 FPS en animation, `ui_helper.cpp`).
const NAV_STREAM_INTERVAL: Duration = Duration::from_millis(40);
/// Durée du retour « … copied » dans la barre d'état.
const CLIPBOARD_FEEDBACK_DURATION: Duration = Duration::from_millis(1500);

/// Presets de résolution pour le rendu haute qualité.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RenderResolutionPreset {
    Window,
    Res4K,
    Res8K,
}

impl RenderResolutionPreset {
    fn resolution(self, window_w: u32, window_h: u32) -> (u32, u32) {
        match self {
            RenderResolutionPreset::Window => (window_w, window_h),
            RenderResolutionPreset::Res4K => (3840, 2160),
            RenderResolutionPreset::Res8K => (7680, 4320),
        }
    }
}

fn same_colors(a: &FractalParams, b: &FractalParams) -> bool {
    let a = &a.color;
    let b = &b.color;
    a.color_mode == b.color_mode
        && a.color_repeat == b.color_repeat
        && a.out_coloring_mode == b.out_coloring_mode
        && a.color_space == b.color_space
        && a.color_offset == b.color_offset
}

/// Pixels de la texture principale et paramètres qui les ont produits.
struct DisplayedFrame {
    image: RgbImage,
    params: FractalParams,
    aa_samples: u32,
}

impl DisplayedFrame {
    fn save(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let (cx, cy, sx, sy) = ViewHp::from_params(&self.params).decimal_parts();
        crate::io::png::save_png_rgb_with_metadata_aa(
            &self.params, self.image.as_raw(), path,
            self.params.center_x_hp.as_deref().unwrap_or(&cx),
            self.params.center_y_hp.as_deref().unwrap_or(&cy),
            self.params.span_x_hp.as_deref().unwrap_or(&sx),
            self.params.span_y_hp.as_deref().unwrap_or(&sy),
            self.aa_samples,
        )
    }
}

/// Message envoyé par le thread de colorisation vers l'UI (évite de bloquer le thread principal).
struct TextureReadyMessage {
    params: FractalParams,
    pass_index: u8,
    display_buffer: Vec<u8>,
    // Arc (G10.3/G10.4) : assignés tels quels à l'état de l'app (pas de memcpy).
    iterations: Arc<Vec<u32>>,
    zs: Arc<Vec<Complex64>>,
    distances: Vec<f64>,
    orbits: Vec<Option<crate::fractal::orbit_traps::OrbitData>>,
    is_preview: bool,
    effective_mode: AlgorithmMode,
    precision_label: String,
}

/// Message pour les recolorisations asynchrones (changement palette/color_repeat).
/// Plus simple que TextureReadyMessage car on ne change pas les données d'itération.
struct RecolorReadyMessage {
    params: FractalParams,
    display_buffer: Vec<u8>,
    version: u64,
    preview: bool,
}

/// Message pour le rendu asynchrone de la preview Julia.
struct JuliaPreviewMessage {
    display_buffer: Vec<u8>,
    width: u32,
    height: u32,
    seed: Complex64,
    version: u64,
}


/// Groupe insécable d'une barre `horizontal_wrapped` : passe à la ligne
/// AVANT d'être dessiné s'il ne tient plus. egui ne connaît la largeur d'un
/// `ui.horizontal` qu'après l'avoir dessiné (une seule passe), et un groupe
/// trop large déborde au lieu de passer à la ligne : on mémorise donc sa
/// largeur de la frame précédente.
fn wrap_group<R>(
    ui: &mut egui::Ui,
    id_salt: &str,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    let id = ui.scope_id().with(id_salt);
    let last_width: Option<f32> = ui.ctx().data(|d| d.get_temp(id));
    let at_row_start = ui.cursor().min.x <= ui.max_rect().min.x + 1.0;
    if let Some(width) = last_width {
        if !at_row_start && width > ui.available_size_before_wrap().x {
            ui.end_row();
        }
    }
    let response = ui.horizontal(add_contents);
    ui.ctx()
        .data_mut(|d| d.insert_temp(id, response.response.rect.width()));
    response.inner
}


/// Hauteur minimale d'une barre : la plus grande déjà vue pour cette largeur
/// de fenêtre (remise à zéro quand la largeur change).
fn sticky_bar_min(slot: &mut (f32, f32), window_width: f32) -> f32 {
    if (slot.0 - window_width).abs() > 0.5 {
        *slot = (window_width, 0.0);
    }
    slot.1
}

/// Retient la hauteur rendue d'une barre (maximum pour la largeur courante).
fn sticky_bar_record(slot: &mut (f32, f32), height: f32) {
    slot.1 = slot.1.max(height);
}

impl FractallApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Self::with_session(cc, None)
    }

    /// Démarre sur la vue courante d'une session sauvegardée (reprise), ou
    /// sur la vue par défaut si `session` est `None`.
    pub fn with_session(
        _cc: &eframe::CreationContext<'_>,
        session: Option<view_history::Session>,
    ) -> Self {
        let default_type = FractalType::Mandelbrot;
        let width = 800;
        let height = 600;
        let params = match &session {
            Some(s) => {
                let mut p = s.current().clone();
                p.sampling.aa_subpixel_offset = [0.0, 0.0];
                p.sampling.aa_jitter = None;
                p
            }
            None => default_params_for_type(default_type, width, height),
        };
        let history = match &session {
            Some(s) => ViewHistory::from_session(s),
            None => ViewHistory::new(&params),
        };
        let iter_auto = session.as_ref().is_none_or(|s| s.iter_auto);

        // Ne pas créer le GPU ici : init différée au premier update() pour éviter
        // "Parent device is lost" sur NVIDIA (eframe doit d'abord créer sa fenêtre/device).
        let gpu_renderer = None;
        let gpu_init_attempted = false;
        // Par défaut, rester en mode CPU même si le GPU est disponible.
        let use_gpu_default = false;

        // Canal pour les recolorisations asynchrones (persiste toute la vie de l'app)
        let (recolor_tx, recolor_rx) = mpsc::channel();

        // Canal pour le rendu asynchrone de la preview Julia
        let (julia_preview_tx, julia_preview_rx) = mpsc::channel();

        let mut app = Self {
            params: params.clone(),
            iterations: Arc::new(Vec::new()),
            zs: Arc::new(Vec::new()),
            distances: Arc::new(Vec::new()),
            orbits: Arc::new(Vec::new()),
            texture: None,
            displayed_frame: None,
            rendering_params: params.clone(),
            raw_preview: false,
            nav_overlay_texture: None,
            nav_overlay_view: None,
            texture_view: None,
            rendering_view: None,
            palette_preview_textures: [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None, None, None,
            ],
            selected_type: default_type,
            palette_index: 6, // SmoothPlasma par défaut
            color_repeat: 40,
            out_coloring_mode: OutColoringMode::Smooth,
            selected_lyapunov_preset: LyapunovPreset::default(),
            hybrid_seq: Vec::new(),
            gpu_renderer,
            gpu_init_attempted,
            use_gpu: use_gpu_default,
            view: ViewHp::from_params(&params),
            selection: Default::default(),
            rendering: false,
            render_thread: None,
            render_cancel: Arc::new(AtomicBool::new(false)),
            render_receiver: None,
            texture_ready_sender: None,
            texture_ready_receiver: None,
            render_progress: Default::default(),
            is_preview: false,
            aa_samples: 1,
            aa_jitter_scale: 1.0,
            clipboard_feedback: None,
            showing_render_aa: false,
            last_render_time: None,
            render_start_time: None,
            last_render_device_label: None,
            last_render_method_label: None,
            orbit_cache: None,
            xaos: Default::default(),
            hover_norm: None,
            last_render_finished: None,
            show_render_dialog: false,
            render_resolution_preset: RenderResolutionPreset::Res4K,
            hq_render_state: Default::default(),
            hq_render_receiver: None,
            recolor_sender: recolor_tx,
            recolor_receiver: recolor_rx,
            recolor_version: Default::default(),

            iteration_input: params.iteration_max.to_string(),
            iteration_input_has_focus: false,

            // Mini-Julia preview (désactivé par défaut pour Mandelbrot)
            julia_preview_enabled: false,
            julia_preview_texture: None,
            julia_preview_sender: julia_preview_tx,
            julia_preview_receiver: julia_preview_rx,
            julia_preview_version: Default::default(),
            julia_preview_last_seed: None,
            julia_preview_last_request_time: None,
            julia_preview_cancel: Arc::new(AtomicBool::new(false)),
            julia_preview_rendering: false,

            // Mode de navigation décidé par le sage (préférence Auto) ;
            // itérations automatiques actives par défaut.
            xaos_nav_mode: false,
            iter_auto,
            auto_grow_guard: Default::default(),
            pending_iter_shrink: None,
            pending_final_pass: None,
            last_final_pass: None,
            sage_notice: Default::default(),
            xaos_nav_last_tick: None,
            xaos_nav_dragging: false,
            middle_pan_active: false,
            bar_heights: [(0.0, 0.0); 3],
            nav: nav::NavState::new(),
            xaos_nav_settle: false,
            rendering_divisor: 1,
            last_render_divisor: 1,

            ui_trace: std::env::var("FRACTALL_UI_TRACE").is_ok_and(|v| v != "0"),
            ui_frame_last: None,
            ui_frame_events: Vec::new(),

            history,
            window_size: session.as_ref().and_then(|s| s.window_size),
            session_path: view_history::session_file_path(),
        };
        if session.is_some() {
            app.sync_ui_from_params();
        }
        app
    }

    /// Aligne les widgets (type, palette, Lyapunov, séquence hybride, champ
    /// itérations) sur `self.params` après un remplacement en bloc (reprise
    /// de session, retour d'historique).
    fn sync_ui_from_params(&mut self) {
        self.selected_type = self.params.fractal_type;
        self.palette_index = self.params.color.color_mode;
        self.color_repeat = self.params.color.color_repeat;
        self.out_coloring_mode = self.params.color.out_coloring_mode;
        if self.params.fractal_type == FractalType::Lyapunov {
            self.selected_lyapunov_preset = self.params.formula.lyapunov_preset;
        }
        if let Some(seq) = &self.params.formula.hybrid_phases {
            self.hybrid_seq = seq.clone();
        }
        self.iteration_input = self.params.iteration_max.to_string();
    }

    /// Retour (`forward = false`) ou avance dans l'historique : applique la
    /// vue restaurée et relance le rendu par le chemin normal. Le cache
    /// d'orbite est gardé pour une même scène (le moteur juge sa validité,
    /// comme pour un zoom) et vidé sinon ; XaoS décide seul par empreinte.
    fn navigate_history(&mut self, forward: bool) {
        let restored = if forward {
            self.history.forward(&self.params)
        } else {
            self.history.back(&self.params)
        };
        let Some(restored) = restored else {
            return;
        };
        let scene_changed = restored.fractal_type != self.params.fractal_type
            || serde_json::to_value(&restored.formula).ok()
                != serde_json::to_value(&self.params.formula).ok()
            || restored.plane_transform != self.params.plane_transform;
        self.params = restored;
        // Vue canonique reconstruite SANS réécrire les chaînes HP (restaurées
        // telles quelles, dimensions déjà adaptées par `restore_params`).
        self.view = ViewHp::from_params(&self.params);
        self.history.refresh_current(&self.params);
        self.sync_ui_from_params();
        // Geste en cours abandonné : la vue vient d'être remplacée.
        self.nav.reset();
        self.selection.cancel();
        self.xaos_nav_last_tick = None;
        self.xaos_nav_dragging = false;
        if scene_changed {
            self.use_gpu = false;
            self.orbit_cache = None;
            self.julia_preview_texture = None;
            self.julia_preview_last_seed = None;
            self.julia_preview_cancel.store(true, Ordering::Relaxed);
            self.on_new_scene();
        } else {
            // Nouvelle vue pour le sage des itérations.
            self.on_new_view();
        }
        self.start_render();
    }

    /// Navigation continue (zoom au bouton maintenu) : case « XaoS ».
    fn continuous_nav(&self) -> bool {
        self.xaos_nav_mode
    }

    /// Nouvelle scène (type, formule, preset, fichier déposé, reset de vue) :
    /// l'historique des itérations automatiques ne vaut plus.
    fn on_new_scene(&mut self) {
        self.on_new_view();
        // Historique : la vue en attente est figée, la nouvelle scène empilée.
        self.history.commit(&self.params);
    }

    /// Nouvelle vue (retour d'historique dans la même scène) : le garde-fou
    /// et l'évaluation des itérations automatiques repartent de zéro.
    fn on_new_view(&mut self) {
        self.auto_grow_guard.reset();
        self.last_final_pass = None;
        self.pending_iter_shrink = None;
    }

    /// Itérations automatiques effectivement actives pour les params courants.
    fn iter_auto_active(&self) -> bool {
        self.iter_auto && crate::fractal::wisdom::auto_iterations_applicable(&self.params)
    }

    /// Clé de vue du garde-fou anti-boucle (vue HP + dims + type).
    fn auto_iter_view_key(&self) -> String {
        format!(
            "{:?}|{}x{}|{:?}",
            xaos::view_strings(&self.params),
            self.params.width,
            self.params.height,
            self.params.fractal_type
        )
    }

    /// Couche 1 du sage des itérations sur une passe finale terminée.
    /// `Grow` : relève le plafond et re-rend (borné par `AutoGrowGuard`) ;
    /// `Shrink` : abaisse le plafond pour le rendu SUIVANT, sans re-rendu.
    fn assess_final_pass(&mut self, info: &sage::FinalPassInfo, iterations: &[u32]) {
        let view_key = self.auto_iter_view_key();
        let ctx = sage::IterContext {
            // Champ en cours d'édition : ne pas écraser la saisie.
            auto_on: self.iter_auto && !self.iteration_input_has_focus,
            applicable: crate::fractal::wisdom::auto_iterations_applicable(&self.params),
            current_cap: self.params.iteration_max,
            gesture_active: self.nav.is_active() || self.xaos_nav_dragging,
            view_key: &view_key,
        };
        match sage::decide_iterations(info, iterations, ctx, &mut self.auto_grow_guard) {
            IterAction::Keep => {}
            IterAction::GrowAndRender(n) => {
                self.pending_iter_shrink = None;
                self.params.iteration_max = n;
                self.iteration_input = n.to_string();
                self.orbit_cache = None;
                self.sage_notice
                    .post(format!("Iterations raised to {n} (auto)"), Instant::now());
                self.start_render();
            }
            IterAction::ShrinkNext(n) => self.pending_iter_shrink = Some(n),
        }
    }

    /// Fin d'un rendu (AllComplete) : nourrit le sage des itérations avec le
    /// buffer final.
    fn sage_after_render(&mut self) {
        let Some((info, iterations)) = self.pending_final_pass.take() else {
            return;
        };
        self.last_final_pass = Some((info, Arc::clone(&iterations)));
        self.assess_final_pass(&info, &iterations);
    }

    /// Met à jour les coordonnées HP depuis les params f64.
    /// Appelé quand on change de type de fractale ou reset.
    /// IMPORTANT: Utiliser to_string_radix avec précision maximale pour préserver la précision
    /// aux zooms profonds (>e16). format!("{:.20e}") limite à 20 chiffres significatifs.
    fn sync_params_to_hp(&mut self) {
        self.view = ViewHp::from_params(&self.params);
        self.view.write_to_params(&mut self.params);
    }

    /// Modifie la vue canonique puis republie atomiquement ses valeurs HP et
    /// leurs miroirs f64.
    fn edit_view_hp(&mut self, edit: impl FnOnce(&mut ViewHp)) {
        edit(&mut self.view);
        self.view.write_to_params(&mut self.params);
    }

    /// Snapshot de la vue live courante (centre/span HP + axialité).
    fn current_view_snapshot(&self) -> ViewSnapshot {
        let (cx, cy, sx, sy) = self.view.decimal_parts();
        ViewSnapshot {
            cx,
            cy,
            sx,
            sy,
            axis_aligned: self.params.rotation == 0.0 && self.params.transform_k.is_none(),
        }
    }

    /// Warp GPU (G10.1) : rectangle écran où dessiner la texture (ANCIENNE vue) pour
    /// qu'elle s'aligne sur la vue live courante pendant qu'un rendu calcule.
    /// `None` = dessiner tel quel (vues identiques, non-axiales, ou parse HP impossible).
    ///
    /// Séparable (rotation=0) : une coordonnée `c` à la position normalisée
    /// `u = 0.5 + (c - centre)/span` (v inversé pour l'imag). On place les bords de
    /// la texture (centre_tex ± span_tex/2) dans la vue live. Ratios O(1) calculés
    /// en HP puis f64 → exact à toute profondeur.
    fn compute_warp_rect(&self, image_rect: egui::Rect) -> Option<egui::Rect> {
        self.warp_rect_for(self.texture_view.as_ref()?, image_rect)
    }

    /// Idem pour une vue de texture arbitraire (texture principale ou overlay
    /// partiel de navigation, qui peut dater d'une vue légèrement antérieure).
    fn warp_rect_for(&self, tex: &ViewSnapshot, image_rect: egui::Rect) -> Option<egui::Rect> {
        let live = self.current_view_snapshot();
        if !tex.axis_aligned || !live.axis_aligned {
            return None;
        }
        let (n_cx, v_cy, rx, ry) = compute_warp_norm(tex, &live)?;
        let w = image_rect.width() as f64;
        let h = image_rect.height() as f64;
        let x0 = image_rect.min.x as f64;
        let y0 = image_rect.min.y as f64;
        Some(egui::Rect::from_min_max(
            egui::pos2(
                (x0 + (n_cx - rx * 0.5) * w) as f32,
                (y0 + (v_cy - ry * 0.5) * h) as f32,
            ),
            egui::pos2(
                (x0 + (n_cx + rx * 0.5) * w) as f32,
                (y0 + (v_cy + ry * 0.5) * h) as f32,
            ),
        ))
    }

    /// Effectue un zoom en haute précision.
    /// `ratio_x`, `ratio_y` : position relative dans l'image (0.0-1.0)
    /// `zoom_factor` : facteur de zoom (>1 = zoom in, <1 = zoom out)
    fn zoom_hp(&mut self, ratio_x: f64, ratio_y: f64, zoom_factor: f64) {
        self.edit_view_hp(|view| view.focus_at(ratio_x, ratio_y, zoom_factor));
    }

    /// Zoom ANCRÉ en haute précision : le point à la position relative
    /// (`ratio_x`, `ratio_y`) reste FIXE à l'écran (zoom molette), au lieu
    /// d'être re-centré comme `zoom_hp`. `zoom_factor` > 1 = zoom in.
    /// Géométrie : C' = C + (r − 0.5)·S·(1 − 1/f), S' = S/f.
    fn zoom_anchored_hp(&mut self, ratio_x: f64, ratio_y: f64, zoom_factor: f64) {
        self.edit_view_hp(|view| view.zoom_at(ratio_x, ratio_y, zoom_factor));
    }

    /// Effectue un zoom rectangulaire en haute précision.
    fn zoom_rect_hp(&mut self, xr1: f64, yr1: f64, xr2: f64, yr2: f64) {
        self.edit_view_hp(|view| view.select_rect(xr1, yr1, xr2, yr2));
    }

    /// Dézoom en haute précision.
    fn zoom_out_hp(&mut self, factor: f64) {
        self.edit_view_hp(|view| view.zoom_at(0.5, 0.5, 1.0 / factor));
    }

    /// Charge l'état de la fractale depuis un fichier PNG contenant des métadonnées.
    fn load_from_png(&mut self, path: &std::path::Path) {
        match crate::io::png::load_png_metadata(path) {
            Ok(params) => {
                // Restaurer le type de fractale
                self.selected_type = params.fractal_type;

                // Restaurer les paramètres de couleur
                self.palette_index = params.color.color_mode;
                self.color_repeat = params.color.color_repeat;
                self.out_coloring_mode = params.color.out_coloring_mode;

                // Restaurer les params (mais garder les dimensions actuelles de la fenêtre)
                let current_width = self.params.width;
                let current_height = self.params.height;
                self.params = params;
                self.params.width = current_width;
                self.params.height = current_height;

                // Fractales densité: color_repeat limité à 1..=8
                let is_density = matches!(
                    self.params.fractal_type,
                    FractalType::Buddhabrot | FractalType::Nebulabrot | FractalType::AntiBuddhabrot
                );
                if is_density && self.params.color.color_repeat > 8 {
                    self.params.color.color_repeat = 8;
                    self.color_repeat = 8;
                }

                // Construire la vue canonique depuis les métadonnées chargées.
                self.sync_params_to_hp();
                self.iteration_input = self.params.iteration_max.to_string();
                // Un PNG porte un nombre d'itérations DÉLIBÉRÉ : le garder,
                // itérations automatiques coupées.
                self.iter_auto = false;
                self.on_new_scene();

                // Invalider le cache et relancer le rendu
                self.orbit_cache = None;
                self.start_render();
            }
            Err(e) => {
                eprintln!("Error loading PNG: {}", e);
            }
        }
    }

    /// Charge l'état de la fractale depuis un fichier TOML (format rust-fractal-core).
    fn load_from_toml(&mut self, path: &std::path::Path) {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Error reading TOML: {}", e);
                return;
            }
        };
        let table: toml::Table = match content.parse() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Error parsing TOML: {}", e);
                return;
            }
        };

        let real_str = match table.get("real").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => {
                eprintln!("TOML: missing 'real' field");
                return;
            }
        };
        let imag_str = match table.get("imag").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => {
                eprintln!("TOML: missing 'imag' field");
                return;
            }
        };
        let zoom_str = match table.get("zoom").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => {
                eprintln!("TOML: missing 'zoom' field");
                return;
            }
        };
        let explicit_iterations = table
            .get("iterations")
            .and_then(|v| v.as_integer())
            .map(|i| i as u32);
        let iterations = explicit_iterations.unwrap_or(self.params.iteration_max);

        // Parse zoom with GMP and compute span = 4/zoom
        let prec = 1024u32;
        let zoom_gmp = match Float::parse(&zoom_str) {
            Ok(parsed) => Float::with_val(prec, parsed),
            Err(_) => {
                eprintln!("TOML: cannot parse zoom: '{}'", zoom_str);
                return;
            }
        };
        let four = Float::with_val(prec, 4.0);
        let span_x_gmp = four / &zoom_gmp;
        let aspect = self.params.height as f64 / self.params.width as f64;
        let span_y_gmp = Float::with_val(prec, &span_x_gmp * aspect);

        let span_x_hp = span_x_gmp.to_string();
        let span_y_hp = span_y_gmp.to_string();

        // Set f64 approximations
        self.params.center_x = Float::with_val(prec, Float::parse(&real_str).unwrap()).to_f64();
        self.params.center_y = Float::with_val(prec, Float::parse(&imag_str).unwrap()).to_f64();
        self.params.span_x = span_x_gmp.to_f64();
        self.params.span_y = span_y_gmp.to_f64();

        // HP strings in params
        self.params.center_x_hp = Some(real_str);
        self.params.center_y_hp = Some(imag_str);
        self.params.span_x_hp = Some(span_x_hp);
        self.params.span_y_hp = Some(span_y_hp);

        // Iterations
        self.params.iteration_max = iterations;
        self.iteration_input = iterations.to_string();

        // Default to Mandelbrot
        self.selected_type = FractalType::Mandelbrot;
        self.params.fractal_type = FractalType::Mandelbrot;

        // Valeur explicite du fichier : respectée (Auto coupé).
        if explicit_iterations.is_some() {
            self.iter_auto = false;
        }
        // Sync and render (vue synchronisée AVANT `on_new_scene` : c'est
        // elle que l'historique enregistre).
        self.sync_params_to_hp();
        self.on_new_scene();
        self.orbit_cache = None;
        self.start_render();

        eprintln!("TOML loaded: zoom={}, iterations={}", zoom_str, iterations);
    }

    /// Lance un rendu haute résolution asynchrone et sauvegarde le résultat.
    fn render_high_quality(&mut self) {
        let (window_width, window_height) = self.view.dimensions();
        let (render_width, render_height) = self
            .render_resolution_preset
            .resolution(window_width, window_height);

        // Params du rendu haute résolution : dimensions, aspect et invariant
        // mode → canaux, dans une fonction PURE (donc verrouillée par tests).
        let render_params = crate::gui::hq_render_state::hq_render_params(
            &self.params,
            render_width,
            render_height,
        );

        // Copier les coordonnées HP pour le thread
        let (center_x_hp, center_y_hp, span_x_hp, span_y_hp) = self.view.decimal_parts();

        // Canal de communication
        let (tx, rx) = mpsc::channel();
        self.hq_render_receiver = Some(rx);
        self.hq_render_state.begin();

        println!("Rendering at {}x{}...", render_width, render_height);

        // Configuration progressive (même logique que le rendu fenêtre)
        let allow_intermediate = !matches!(
            render_params.fractal_type,
            FractalType::VonKoch
                | FractalType::Dragon
                | FractalType::Buddhabrot
                | FractalType::Nebulabrot
                | FractalType::AntiBuddhabrot
                | FractalType::Lyapunov
        );
        let render_plan = crate::fractal::wisdom::plan(&render_params);
        let will_use_perturbation = allow_intermediate
            && render_plan.algorithm == crate::fractal::wisdom::Algorithm::Perturbation;
        let config = ProgressiveConfig::for_params_with_intermediate(
            render_width,
            render_height,
            render_params.engine.use_gmp,
            allow_intermediate,
            will_use_perturbation,
        );

        // Lancer le rendu progressif dans un thread séparé (multi-passes pour barre de progression)
        thread::spawn(move || {
            let cancel = Arc::new(AtomicBool::new(false));
            let total_passes = config.passes.len() as f32;
            let mut previous_pass: Option<(Vec<u32>, Vec<Complex64>, u32, u32)> = None;
            /// Résultat final du rendu HQ : sortie typée du dispatcher (G5).
            type HqFinal = crate::render::RenderOutput;
            let mut final_result: Option<HqFinal> = None;

            for (pass_index, &scale_divisor) in config.passes.iter().enumerate() {
                if cancel.load(Ordering::Relaxed) {
                    let _ = tx.send(HqRenderEvent::Error("Render cancelled".to_string()));
                    return;
                }
                let pass_width = (render_width / scale_divisor as u32).max(1);
                let pass_height = (render_height / scale_divisor as u32).max(1);
                let mut pass_params = render_params.clone();
                pass_params.width = pass_width;
                pass_params.height = pass_height;

                // ⚠️ PAS de réutilisation grossière inter-passes : elle copie le
                // pixel (i/r, j/r) de la passe précédente dont le centre est
                // décalé de (r−1)/2 px (1,5 px à r=4) → treillis 1/16 de pixels
                // FAUX dans l'image finale (mesuré : 559/65536 px à 256², vue
                // par défaut). Gain ≤ 6 % pour une sortie inexacte — supprimé
                // (bug 2026-08-23). Le dispatcher garde le mécanisme pour
                // d'autres appelants/tests.
                let _ = &previous_pass;
                let result = render_request(RenderRequest::new(&pass_params, &cancel), &mut None);

                match result {
                    Some(out) => {
                        // Progression : 5% au début, puis 5–90% répartis sur les passes, 90–95% sauvegarde
                        let pass_progress = (pass_index + 1) as f32 / total_passes * 0.85f32; // 0.85 * (1/5..5/5)
                        let progress = 0.05f32 + pass_progress; // 5% -> 90%
                        let _ = tx.send(HqRenderEvent::Progress(progress));

                        if pass_index + 1 == config.passes.len() {
                            // Canaux annexes conservés : sans eux le PNG HQ des
                            // modes Distance*/OrbitTraps/Wings retombe sur Smooth.
                            final_result = Some(out);
                            break;
                        }
                        previous_pass = Some((out.iterations, out.zs, pass_width, pass_height));
                    }
                    None => {
                        let _ = tx.send(HqRenderEvent::Error("Render cancelled".to_string()));
                        return;
                    }
                }
            }

            if let Some(out) = final_result {
                let _ = tx.send(HqRenderEvent::Progress(0.92));

                use std::path::Path;
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let filename = format!(
                    "fractal_{}x{}_{}.png",
                    render_width, render_height, timestamp
                );

                // Colorisation VÉRIFIÉE (G5) : un canal requis absent = erreur
                // affichée, pas un PNG HQ silencieusement retombé sur Smooth.
                let buffer = match crate::io::png::colorize_output(&render_params, &out) {
                    Ok(rgb) => rgb,
                    Err(e) => {
                        let _ = tx.send(HqRenderEvent::Error(format!(
                            "Colorization error: {e}"
                        )));
                        return;
                    }
                };
                if let Err(e) = crate::io::png::save_png_rgb_with_metadata(
                    &render_params,
                    &buffer,
                    Path::new(&filename),
                    &center_x_hp,
                    &center_y_hp,
                    &span_x_hp,
                    &span_y_hp,
                ) {
                    let _ = tx.send(HqRenderEvent::Error(format!("Error saving PNG: {}", e)));
                } else {
                    let _ = tx.send(HqRenderEvent::Progress(1.0));
                    println!("High quality render saved: {}", filename);
                    let _ = tx.send(HqRenderEvent::Done(filename));
                }
            }
        });
    }

    /// Lance le rendu progressif de la fractale dans un thread séparé.
    fn start_render(&mut self) {
        self.start_render_with(false);
    }

    /// G10.4 : rendu de raffinement idle — passe unique pleine résolution,
    /// réutilisation XaoS désactivée (recalcul exact des pixels approximés),
    /// remplacement silencieux de l'image (une seule PassComplete, à la fin).
    fn start_render_refine(&mut self) {
        self.start_render_with(true);
    }

    fn start_render_with(&mut self, refine: bool) {
        // Annuler tout rendu en cours
        if self.rendering {
            self.render_cancel.store(true, Ordering::Relaxed);

            // Vider tous les messages en attente du receiver pour éviter de traiter des messages obsolètes
            if let Some(receiver) = &self.render_receiver {
                while receiver.try_recv().is_ok() {
                    // Vider tous les messages
                }
            }

            // Nettoyer l'ancien thread et receiver
            if let Some(handle) = self.render_thread.take() {
                // On ne peut pas attendre le thread ici (bloquerait l'UI), mais on le marque pour nettoyage
                // Le thread se terminera de lui-même quand il verra le flag cancel
                drop(handle);
            }
            self.render_receiver = None;
            self.texture_ready_sender = None;
            self.texture_ready_receiver = None;
            self.rendering = false;
        }

        // Créer un nouveau flag d'annulation
        self.render_cancel = Arc::new(AtomicBool::new(false));
        // La passe finale d'un rendu annulé ne doit jamais nourrir le sage.
        self.pending_final_pass = None;
        // Plafond abaissé par le sage : appliqué maintenant (jamais à un
        // raffinement idle, qui doit garder l'empreinte de sa frame source).
        if !refine {
            let auto_active = self.iter_auto_active() && !self.iteration_input_has_focus;
            if let Some(n) = sage::take_pending_shrink(
                &mut self.pending_iter_shrink,
                auto_active,
                self.params.iteration_max,
            ) {
                self.params.iteration_max = n;
                self.iteration_input = n.to_string();
                self.orbit_cache = None;
            }
        }
        self.last_render_device_label = None;
        self.last_render_method_label = None;

        // Configuration progressive selon les paramètres
        let allow_intermediate = !matches!(
            self.params.fractal_type,
            FractalType::VonKoch
                | FractalType::Dragon
                | FractalType::Buddhabrot
                | FractalType::Nebulabrot
                | FractalType::AntiBuddhabrot
                | FractalType::Lyapunov
        );
        // Device (G9.5) : en mode « 🔄 Auto » (algorithm_mode == Auto), le wisdom
        // arbitre CPU/GPU (benchmark machine + garde-fou correction — même
        // décision que le CLI). Une sélection EXPLICITE du menu CPU/GPU
        // (algorithm_mode ≠ Auto) est un override manuel via `self.use_gpu`.
        let auto_plan = crate::render::RenderPlan::auto(&self.params, self.gpu_renderer.is_some());
        let device_want_gpu = if self.params.engine.algorithm_mode == AlgorithmMode::Auto {
            auto_plan.wisdom().device == crate::fractal::wisdom::Device::Gpu
        } else {
            self.use_gpu
        };
        let use_gpu = device_want_gpu
            && self.gpu_renderer.is_some()
            && matches!(
                self.params.fractal_type,
                FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip
            );
        let render_device = if use_gpu {
            crate::fractal::wisdom::Device::Gpu
        } else {
            crate::fractal::wisdom::Device::Cpu
        };
        let render_plan = if self.params.engine.algorithm_mode == AlgorithmMode::Auto
            && render_device == auto_plan.wisdom().device
        {
            auto_plan
        } else {
            crate::render::RenderPlan::for_device(&self.params, render_device)
        };
        let will_use_perturbation = allow_intermediate
            && render_plan.wisdom().algorithm == crate::fractal::wisdom::Algorithm::Perturbation;
        // Navigation XaoS (bouton maintenu) : une passe UNIQUE pleine
        // résolution. Les passes basse résolution repeignent l'image flou→net
        // à chaque cycle de rendu — un pulsing perçu comme une saccade — et
        // n'apportent aucun feedback ici : le warp (G10.1) affiche déjà la
        // vue live en continu, en plus net que la passe 1/16. Vaut AUSSI pour
        // les scènes lentes (c'est là que le flash de la passe 1/16 est le
        // plus visible), et pour le rendu final au relâchement.
        // Résolution DYNAMIQUE de navigation (port XaoS « dynamic resolution ») :
        // en mouvement, on calcule ce qui tient dans NAV_FRAME_TARGET_SECS plutôt
        // que d'imposer la pleine résolution coûte que coûte. Le coût varie en d²
        // (côté/d) ; on extrapole le coût plein cadre depuis la dernière durée
        // mesurée et son propre diviseur, et on prend le plus petit d qui tient.
        // Le rendu de stabilisation (bouton relâché) reste en pleine résolution,
        // et le raffinement idle rend l'image exacte ensuite.
        let nav_divisor = if self.xaos_nav_dragging && !self.xaos_nav_settle {
            nav::nav_divisor(self.last_render_time, self.last_render_divisor)
        } else {
            1
        };

        let config = if refine || self.xaos_nav_dragging {
            ProgressiveConfig {
                passes: vec![nav_divisor],
            }
        } else {
            ProgressiveConfig::for_params_with_intermediate(
                self.params.width,
                self.params.height,
                self.params.engine.use_gmp,
                allow_intermediate,
                will_use_perturbation,
            )
        };

        // G10.4 : un nouveau rendu remplace tout raffinement programmé (le
        // flag sera re-posé à la passe finale si elle copie des pixels).
        self.xaos.begin_render();

        if self.ui_trace {
            self.ui_frame_events.push(format!(
                "start_render(passes={},refine={},nav={})",
                config.passes.len(),
                refine,
                self.xaos_nav_dragging
            ));
        }

        self.rendering = true;
        self.rendering_divisor = *config.passes.last().unwrap_or(&1);
        self.is_preview = true;
        // G10.1 : mémoriser la vue de ce rendu ; elle deviendra `texture_view` au
        // chargement de la 1re texture. Jusque-là, la texture affichée reste
        // l'ancienne vue → le warp l'aligne sur la vue live.
        self.rendering_view = Some(self.current_view_snapshot());
        self.render_start_time = Some(Instant::now());

        // Créer le channel de communication
        let (sender, receiver) = mpsc::channel();
        self.render_receiver = Some(receiver);

        // Canal pour les textures pré-colorisées (évite de bloquer l'UI)
        let (tex_tx, tex_rx) = mpsc::channel();
        self.texture_ready_sender = Some(tex_tx);
        self.texture_ready_receiver = Some(tex_rx);

        // Paramètres pour le thread : règle UNIQUE mode → canaux, partagée
        // avec le CLI et l'export HQ (`render::ensure_required_channels`).
        let mut params = self.params.clone();
        crate::render::ensure_required_channels(&mut params);
        let cancel = Arc::clone(&self.render_cancel);
        let full_width = self.params.width;
        let full_height = self.params.height;
        let gpu_renderer = self.gpu_renderer.clone();
        let use_gpu = use_gpu;
        let orbit_cache = self.orbit_cache.clone();
        // Anti-aliasing multi-sample : jitter F3 par pixel sur CPU et GPU.
        let aa_samples = self.aa_samples.max(1);
        let aa_jitter_scale =
            crate::gui::render_state::normalized_jitter_scale(self.aa_jitter_scale);
        self.rendering_params = params.clone();
        self.rendering_params.sampling.jitter_scale = aa_jitter_scale;
        self.showing_render_aa = false;
        self.recolor_version.issue();
        self.render_progress.begin(config.passes.len(), aa_samples);

        // G10.4 : frame source pour la réutilisation pixels inter-frame.
        // Désactivée en mode AA multi-sample (les échantillons doivent être
        // exacts). Le clone est un bump d'Arc (buffers partagés avec l'état de
        // l'app). Le raffinement UTILISE la frame mais à tolérance EXACTE :
        // il conserve les pixels dont la position est déjà vraie (calculés
        // frais ou copiés alignés) et ne recalcule que les approximations —
        // au lieu de refaire 100 % du travail de la passe finale.
        let xaos_source = self.xaos.source_for_render(aa_samples);

        // G10.5 : point de priorité de la file de tuiles (dernier survol
        // curseur, sinon centre) — le zoom molette ancre le curseur, donc la
        // zone d'intérêt est calculée en premier à chaque cran.
        let tile_priority: (f64, f64) = self
            .hover_norm
            .map(|(x, y)| (x as f64, y as f64))
            .unwrap_or((0.5, 0.5));

        // Navigation XaoS en cours → publier des frames PARTIELLES (overlay
        // alpha) au fil des tuiles, au lieu d'attendre l'image complète.
        let nav_stream = self.xaos_nav_dragging;

        // Spawner le thread de rendu progressif
        let handle = thread::spawn(move || {
            let mut previous_pass: Option<(Arc<Vec<u32>>, Arc<Vec<Complex64>>, u32, u32)> = None;
            // G10.5 : dernière passe colorisée (base du streaming tuiles).
            let mut last_colored: Option<(Vec<u8>, u32, u32)> = None;
            let mut current_orbit_cache = orbit_cache;
            // G10.4 : vue + fingerprint de CE rendu (identiques pour toutes
            // les passes — seules les dims changent, neutralisées).
            let (view_cx, view_cy, view_sx, view_sy) = xaos::view_strings(&params);
            let render_fingerprint = xaos::params_fingerprint(&params);
            let mut centered_curve_aa = None;

            for (pass_index, &scale_divisor) in config.passes.iter().enumerate() {
                // Vérifier l'annulation
                if cancel.load(Ordering::Relaxed) {
                    let _ = sender.send(RenderMessage::Cancelled);
                    return;
                }

                // Calculer les dimensions pour cette passe
                let pass_width = (full_width / scale_divisor as u32).max(1);
                let pass_height = (full_height / scale_divisor as u32).max(1);

                // Créer les params pour cette passe
                let mut pass_params = params.clone();
                pass_params.width = pass_width;
                pass_params.height = pass_height;

                // G10.4 : mapping colonnes/lignes frame précédente → cette
                // passe (gates + fingerprint + transformée HP validés dans
                // build_map ; None si rien de réutilisable). Le raffinement
                // utilise le map UNION : il conserve tout pixel déjà exact
                // (axe frais ou frame exacte) et ne recalcule que les
                // approximations de la passe écho.
                let xaos_map = xaos_source.as_ref().and_then(|src| {
                    if refine {
                        xaos::build_refine_map(src, &pass_params)
                    } else {
                        xaos::build_map(src, &pass_params)
                    }
                });

                // G10.4b : une passe intermédiaire écho-pur (100 % copiée)
                // n'apporte RIEN de plus que le warp G10.1 déjà affiché (même
                // contenu, en moins net) — la sauter supprime le « pompage »
                // flou preview→full pendant la navigation et fait démarrer le
                // travail frais plus tôt. La passe FINALE tourne toujours
                // (matérialise les buffers, programme le raffinement).
                let is_final_pass = pass_index + 1 == config.passes.len();
                if !is_final_pass {
                    if let Some(m) = &xaos_map {
                        if m.is_pure_copy(pass_width as usize, pass_height as usize) {
                            continue;
                        }
                    }
                }

                // G10.5 : streaming intra-passe de la file de tuiles — base =
                // dernière passe colorisée (upscalée aux dims de cette passe),
                // recouverte des tuiles au fil de leur complétion (le point
                // d'intérêt arrive en premier), envoi throttlé ~100 ms. Gates :
                // raffinement silencieux exclu ; OrbitTraps/Wings exclus (la
                // colorisation par tuile n'a pas les orbites) ; pas de base
                // (1re passe) → pas de streaming (le warp G10.1 affiche déjà
                // l'ancienne frame, mieux qu'un fond noir).
                let sink_state = if refine
                    || matches!(
                        params.color.out_coloring_mode,
                        crate::fractal::OutColoringMode::OrbitTraps
                            | crate::fractal::OutColoringMode::Wings
                    ) {
                    None
                } else {
                    last_colored.as_ref().map(|(rgb, w, h)| {
                        let base = if (*w, *h) == (pass_width, pass_height) {
                            rgb.clone()
                        } else {
                            crate::gui::progressive::upscale_rgb_nearest(
                                rgb,
                                *w,
                                *h,
                                pass_width,
                                pass_height,
                            )
                        };
                        std::sync::Mutex::new((base, Instant::now(), sender.clone()))
                    })
                };
                // Navigation XaoS : overlay PARTIEL alpha (cf. RenderMessage::
                // NavProgress). S'applique là où le streaming classique renonce —
                // une passe SANS base (la seule passe du rendu de navigation) —
                // parce que la base ne manque qu'en apparence : la GUI compose
                // sur la texture précédente warpée. Publication plus rapprochée
                // que le streaming classique (budget type XaoS : 15-35 FPS en
                // animation, `ui_helper.cpp` FRAMERATE).
                let nav_overlay = nav_stream
                    && sink_state.is_none()
                    && !refine
                    && !matches!(
                        params.color.out_coloring_mode,
                        crate::fractal::OutColoringMode::OrbitTraps
                            | crate::fractal::OutColoringMode::Wings
                    );
                let nav_state = nav_overlay.then(|| {
                    std::sync::Mutex::new((
                        vec![0u8; (pass_width as usize) * (pass_height as usize) * 4],
                        Instant::now(),
                        sender.clone(),
                    ))
                });

                let sink_params = pass_params.clone();
                let sink_closure = (sink_state.is_some() || nav_state.is_some()).then(|| {
                    let sink_state = &sink_state;
                    let nav_state = &nav_state;
                    move |u: crate::render::tiles::TileUpdate| {
                        // Colorisation par tuile (per-pixel, mêmes fonctions que
                        // les passes) hors verrou, puis blit + envoi sous verrou.
                        let tile_rgb = colorize_buffer(
                            &u.iterations,
                            &u.zs,
                            &u.distances,
                            &[],
                            &sink_params,
                            u.w as u32,
                            u.h as u32,
                        );
                        if let Some(state) = sink_state {
                            let mut guard = state.lock().unwrap();
                            let (base, last_sent, tx) = &mut *guard;
                            for row in 0..u.h {
                                let dst = ((u.y0 + row) * pass_width as usize + u.x0) * 3;
                                let src = row * u.w * 3;
                                base[dst..dst + u.w * 3]
                                    .copy_from_slice(&tile_rgb[src..src + u.w * 3]);
                            }
                            if last_sent.elapsed() >= Duration::from_millis(100) {
                                *last_sent = Instant::now();
                                let _ = tx.send(RenderMessage::TileProgress {
                                    display_buffer: base.clone(),
                                    width: pass_width,
                                    height: pass_height,
                                });
                            }
                        }
                        if let Some(state) = nav_state {
                            let mut guard = state.lock().unwrap();
                            let (rgba, last_sent, tx) = &mut *guard;
                            for row in 0..u.h {
                                for col in 0..u.w {
                                    let dst = ((u.y0 + row) * pass_width as usize + u.x0 + col) * 4;
                                    let src = (row * u.w + col) * 3;
                                    rgba[dst] = tile_rgb[src];
                                    rgba[dst + 1] = tile_rgb[src + 1];
                                    rgba[dst + 2] = tile_rgb[src + 2];
                                    rgba[dst + 3] = 255; // calculé ⇒ opaque
                                }
                            }
                            if last_sent.elapsed() >= NAV_STREAM_INTERVAL {
                                *last_sent = Instant::now();
                                let _ = tx.send(RenderMessage::NavProgress {
                                    rgba: rgba.clone(),
                                    width: pass_width,
                                    height: pass_height,
                                });
                            }
                        }
                    }
                });
                let tile_opts = Some(crate::render::tiles::TileOpts {
                    priority: tile_priority,
                    sink: sink_closure
                        .as_ref()
                        .map(|c| c as &(dyn Fn(crate::render::tiles::TileUpdate) + Sync)),
                });

                // Rendre cette passe. ⚠️ PAS de réutilisation grossière de la
                // passe précédente : le pixel (i/r, j/r) copié a son centre
                // décalé de (r−1)/2 px → treillis 1/16 de pixels FAUX dans la
                // passe finale (559/65536 px mesurés à 256²), que le stockage
                // XaoS marquait ensuite `col_exact` (map None) → jamais
                // raffinés, propagés par l'écho de zoom en zoom (bug 2026-08-23).
                let reuse: Option<(&[u32], &[Complex64], u32, u32)> = None;
                let _ = &previous_pass;
                // Le plan global choisit le backend ; sa spécialisation CPU
                // reste disponible pour le libellé en cas de choix/fallback CPU.
                let cpu_plan = crate::render::CpuRenderPlan::for_params(&pass_params);
                let device = if use_gpu {
                    crate::fractal::wisdom::Device::Gpu
                } else {
                    crate::fractal::wisdom::Device::Cpu
                };
                let plan = crate::render::RenderPlan::for_device(&pass_params, device);
                let mut request = RenderRequest::new(&pass_params, &cancel);
                if let Some(previous) = reuse {
                    request = request.with_progressive_reuse(previous.into());
                }
                if let Some(map) = xaos_map.as_ref() {
                    request = request.with_xaos(map);
                }
                if let Some(tiles) = tile_opts.as_ref() {
                    request = request.with_tiles(tiles);
                }
                let mut cache = current_orbit_cache.take();
                let result = crate::render::render_planned(
                    crate::render::PlannedRenderRequest::new(plan, request),
                    gpu_renderer.as_deref(),
                    &mut cache,
                )
                .map(|rendered| {
                    let from_gpu = rendered.device == crate::fractal::wisdom::Device::Gpu;
                    let effective_mode = if from_gpu {
                        if rendered.used_perturbation {
                            AlgorithmMode::Perturbation
                        } else {
                            AlgorithmMode::StandardF64
                        }
                    } else {
                        effective_cpu_mode(cpu_plan)
                    };
                    let precision_label = if from_gpu {
                        format!(
                            "GPU {}",
                            gpu_renderer
                                .as_ref()
                                .expect("a GPU render requires a renderer")
                                .precision_label()
                        )
                    } else {
                        cpu_precision_label(&pass_params, effective_mode)
                    };
                    (rendered.output, effective_mode, precision_label, from_gpu)
                });
                current_orbit_cache = cache;

                match result {
                    Some((out, effective_mode, precision_label, from_gpu)) => {
                        // La passe CPU pleine résolution d’une courbe est le premier
                        // échantillon AA (décalage global nul pour k = 0).
                        if aa_samples > 1 && is_final_pass && !from_gpu && !use_gpu
                            && (pass_width, pass_height) == (full_width, full_height)
                            && xaos_map.is_none()
                            && matches!(params.fractal_type, FractalType::VonKoch | FractalType::Dragon)
                            && params.sampling.aa_subpixel_offset == [0.0, 0.0]
                            && params.sampling.aa_jitter.is_none()
                        {
                            let mut accum = crate::io::png::AaAccumulator::new(&pass_params, true);
                            if let Err(e) = accum.add(&pass_params, &out) {
                                eprintln!("AA sample 1: {e}");
                                let _ = sender.send(RenderMessage::Cancelled);
                                return;
                            }
                            centered_curve_aa = Some(accum);
                        }
                        // Sortie typée (G5) : les canaux annexes suivent
                        // iterations/zs jusqu'à la colorisation et l'état app.
                        let crate::render::RenderOutput {
                            iterations,
                            zs,
                            orbits,
                            distances,
                        } = out;
                        // Buffers bruts en Arc : partagés (sans memcpy) entre la
                        // passe suivante, la frame source XaoS et l'état de l'app.
                        let iterations = Arc::new(iterations);
                        let zs = Arc::new(zs);
                        if pass_index + 1 < config.passes.len() {
                            previous_pass = Some((
                                Arc::clone(&iterations),
                                Arc::clone(&zs),
                                pass_width,
                                pass_height,
                            ));
                        } else {
                            previous_pass = None;
                        }

                        // G10.4 : frame source XaoS pour les rendus suivants
                        // (positions vraies héritées du mapping si des pixels
                        // ont été copiés, zéro sinon). Passes GPU exclues (le
                        // map n'y est PAS consommé : ni label ni refine).
                        // `approx` = des pixels copiés dévient réellement de
                        // la grille → label ≈ + raffinement idle ; les copies
                        // exactes (pan entier, refine ε) n'en déclenchent pas.
                        let xaos_approx = match (&xaos_map, from_gpu) {
                            (Some(m), false) => {
                                m.any_reuse() && m.max_abs_err() > xaos::XAOS_EXACT_TOLERANCE_PX
                            }
                            _ => false,
                        };
                        // Une passe écho-pur (100 % copiée) n'apporte aucune
                        // information : garder la frame source existante au
                        // lieu de la dégrader en copie de copie (zooms/pans
                        // enchaînés qui annulent les rendus en cours).
                        let pure_copy = xaos_map.as_ref().is_some_and(|m| {
                            m.is_pure_copy(pass_width as usize, pass_height as usize)
                        });
                        let xaos_frame = if from_gpu || pure_copy {
                            None
                        } else {
                            let view = (
                                view_cx.clone(),
                                view_cy.clone(),
                                view_sx.clone(),
                                view_sy.clone(),
                            );
                            // Erreur de position PAR PIXEL héritée du mapping
                            // (vraie position des copies, 0 pour les pixels
                            // calculés) ; frame sans map = exacte partout.
                            Some(match &xaos_map {
                                Some(m) => XaosSourceFrame::from_map(
                                    Arc::clone(&iterations),
                                    Arc::clone(&zs),
                                    pass_width,
                                    pass_height,
                                    view,
                                    render_fingerprint.clone(),
                                    m,
                                ),
                                None => XaosSourceFrame::exact(
                                    Arc::clone(&iterations),
                                    Arc::clone(&zs),
                                    pass_width,
                                    pass_height,
                                    view,
                                    render_fingerprint.clone(),
                                ),
                            })
                        };

                        // Sage : description de la passe FINALE (coût de
                        // navigation + éligibilité au jugement des itérations).
                        // Le map n'est consommé que sur CPU.
                        let final_pass = is_final_pass.then(|| {
                            let cpu_map = xaos_map.as_ref().filter(|_| !from_gpu);
                            crate::gui::sage::FinalPassInfo {
                                divisor: scale_divisor,
                                from_gpu,
                                fresh_fraction: cpu_map.map_or(1.0, |m| {
                                    1.0 - m.reused_fraction(
                                        pass_width as usize,
                                        pass_height as usize,
                                    )
                                }),
                                approx_copies: xaos_approx,
                                iteration_cap: pass_params.iteration_max,
                            }
                        });

                        // Coloriser dans le thread de rendu pour éviter de bloquer l'UI
                        let colored_buffer = colorize_buffer(
                            &iterations,
                            &zs,
                            &distances,
                            &orbits,
                            &pass_params,
                            pass_width,
                            pass_height,
                        );
                        // G10.5 : base du streaming tuiles de la passe suivante.
                        last_colored = Some((colored_buffer.clone(), pass_width, pass_height));

                        let _ = sender.send(RenderMessage::PassComplete {
                            pass_index: pass_index as u8,
                            scale_divisor,
                            effective_mode,
                            precision_label,
                            iterations,
                            zs,
                            distances,
                            orbits,
                            width: pass_width,
                            height: pass_height,
                            colored_buffer,
                            xaos_frame,
                            xaos_approx,
                            final_pass,
                        });

                        // Délai pour laisser l'UI afficher cette passe avant de continuer
                        // Sans ce délai, les passes rapides s'empilent et l'affichage progressif
                        // ne fonctionne pas (ex: passe 3 et 4 quasi simultanées)
                        if pass_index + 1 < config.passes.len() {
                            thread::sleep(Duration::from_millis(16)); // ~1 frame à 60fps
                        }
                    }
                    None => {
                        let _ = sender.send(RenderMessage::Cancelled);
                        return;
                    }
                }
            }

            // Chaque moyenne intermédiaire est publiée pour l’affichage progressif.
            if aa_samples > 1 && !cancel.load(Ordering::Relaxed) {
                let reused_center = centered_curve_aa.is_some();
                let mut accum = centered_curve_aa.unwrap_or_else(|| {
                    crate::io::png::AaAccumulator::new(&params, !use_gpu)
                });
                let mut p = params.clone();
                for k in 0..aa_samples {
                    if cancel.load(Ordering::Relaxed) {
                        let _ = sender.send(RenderMessage::Cancelled);
                        return;
                    }
                    if k > 0 || !reused_center {
                        p.sampling.aa_jitter = Some((u64::from(k), aa_jitter_scale));
                        let rendered = render_planned(
                            PlannedRenderRequest::new(render_plan, RenderRequest::new(&p, &cancel)),
                            gpu_renderer.as_deref(),
                            &mut current_orbit_cache,
                        );
                        let Some(sample) = rendered else {
                            let _ = sender.send(RenderMessage::Cancelled);
                            return;
                        };
                        if let Err(e) = accum.add(&p, &sample.output) {
                            eprintln!("AA sample {}: {e}", k + 1);
                            let _ = sender.send(RenderMessage::Cancelled);
                            return;
                        }
                    }
                    let _ = sender.send(RenderMessage::AaProgress {
                        display_buffer: accum.average(),
                        width: full_width,
                        height: full_height,
                        sample: k + 1,
                        total: aa_samples,
                    });
                }
            }

            let _ = sender.send(RenderMessage::AllComplete {
                orbit_cache: current_orbit_cache,
            });
        });

        self.render_thread = Some(handle);
    }

    /// Vérifie si des passes de rendu sont terminées et met à jour l'affichage.
    /// Ne traite qu'un seul message PassComplete par frame pour permettre l'affichage progressif.
    /// La colorisation (upscale + colorize) est faite dans un thread dédié pour ne pas bloquer l'UI.
    fn check_render_complete(&mut self, ctx: &Context) {
        // Traiter les recolorisations asynchrones (changement palette/color_repeat)
        // Ignorer les messages obsolètes (version != version actuelle)
        while let Ok(msg) = self.recolor_receiver.try_recv() {
            if self.recolor_version.accepts(msg.version) {
                self.load_texture_from_buffer(ctx, msg.display_buffer, msg.params, (!msg.preview).then_some(1));
                ctx.request_repaint();
                break;
            }
            // Message obsolète, on continue à vider le channel
        }

        // Traiter d'abord une texture prête (calcul faite hors thread principal)
        if let Some(ref rx) = self.texture_ready_receiver {
            if let Ok(tex) = rx.try_recv() {
                if !self.render_progress.accepts_pass(tex.pass_index) {
                    ctx.request_repaint();
                    return;
                }
                self.render_progress.pass_ready(tex.pass_index);
                self.last_render_device_label = Some(tex.precision_label.clone());
                self.last_render_method_label = Some(match tex.effective_mode {
                    AlgorithmMode::ReferenceGmp => String::new(),
                    AlgorithmMode::Perturbation => "Perturbation".to_string(),
                    _ => "Standard".to_string(),
                });
                self.iterations = tex.iterations;
                self.zs = tex.zs;
                self.raw_preview = tex.is_preview;
                self.distances = Arc::new(tex.distances);
                self.orbits = Arc::new(tex.orbits);
                // Le brut reste disponible pour recoloriser, même quand la
                // moyenne AA de cette vue est déjà affichée.
                if self.showing_render_aa && same_colors(&self.params, &self.rendering_params) {
                    ctx.request_repaint();
                    return;
                }
                if !same_colors(&tex.params, &self.params) {
                    self.update_texture(ctx);
                } else {
                    self.recolor_version.issue();
                    self.is_preview = tex.is_preview;
                    self.load_texture_from_buffer(ctx, tex.display_buffer, tex.params, (!tex.is_preview).then_some(1));
                }
                ctx.request_repaint();
                return;
            }
        }

        if !self.rendering {
            return;
        }

        // Toujours demander un repaint pour animer le spinner pendant le rendu
        ctx.request_repaint();

        // Récupérer un seul message pour permettre l'affichage de chaque passe
        let msg = {
            let receiver = match &self.render_receiver {
                Some(r) => r,
                None => return, // Pas de receiver, mais on a déjà demandé le repaint
            };
            receiver.try_recv().ok()
        };

        let Some(msg) = msg else {
            return; // Pas de message, mais on a déjà demandé le repaint
        };

        match msg {
            RenderMessage::PassComplete {
                pass_index,
                scale_divisor,
                effective_mode,
                precision_label,
                iterations,
                zs,
                distances,
                orbits,
                width,
                height,
                colored_buffer: _, // ignoré : on re-colorise avec la palette actuelle (params)
                xaos_frame,
                xaos_approx,
                final_pass,
            } => {
                if let Some(info) = final_pass {
                    self.pending_final_pass = Some((info, Arc::clone(&iterations)));
                }
                // G10.4 : mémoriser la frame source pour la réutilisation
                // pixels inter-frame (Arc partagés, pas de memcpy). None =
                // passe GPU ou écho pur → garder la source existante. Si la
                // passe a copié des pixels APPROXIMÉS (erreur réelle > ε),
                // programmer le raffinement exact à l'idle — les copies
                // exactes (pan entier, refine) n'en redéclenchent pas.
                // Ne jamais REMPLACER une frame source plus résolue par une
                // passe intermédiaire (zoom-out / grand facteur : la passe
                // 1/16 n'est pas un écho pur et arrivait en premier ; deux
                // crans rapides → le rendu suivant partait d'une frame 1/16 →
                // XaoS dégénéré en rendu complet, bug 2026-08-23). Une frame
                // de résolution ≥ (passe finale, ou même dims) remplace.
                self.xaos.accept_pass(xaos_frame, xaos_approx);
                // Déléguer upscale + colorize à un thread pour ne pas bloquer l'UI (évite "ne répond pas")
                let tx = match &self.texture_ready_sender {
                    Some(t) => t.clone(),
                    None => return,
                };
                let mut params = self.rendering_params.clone();
                params.color = self.params.color.clone();
                let repaint_ctx = ctx.clone();
                let (total_width, total_height) = (params.width, params.height);
                thread::spawn(move || {
                    let (iterations, zs, distances, orbits, disp_w, disp_h, is_preview) =
                        if scale_divisor > 1 {
                            let (upscaled_iter, upscaled_zs) = upscale_nearest(
                                &iterations,
                                &zs,
                                width,
                                height,
                                total_width,
                                total_height,
                            );
                            (
                                Arc::new(upscaled_iter),
                                Arc::new(upscaled_zs),
                                Vec::new(),
                                Vec::new(),
                                total_width,
                                total_height,
                                true,
                            )
                        } else {
                            (iterations, zs, distances, orbits, width, height, false)
                        };
                    let display_buffer = colorize_buffer(
                        &iterations,
                        &zs,
                        &distances,
                        &orbits,
                        &params,
                        disp_w,
                        disp_h,
                    );
                    let _ = tx.send(TextureReadyMessage {
                        params,
                        pass_index,
                        display_buffer,
                        iterations,
                        zs,
                        distances,
                        orbits,
                        is_preview,
                        effective_mode,
                        precision_label,
                    });
                    repaint_ctx.request_repaint();
                });
                ctx.request_repaint();
            }

            RenderMessage::TileProgress {
                display_buffer,
                width,
                height,
            } => {
                // Buffer à la taille de la PASSE : upscaler vers la vue pleine
                // (comme PassComplete) — l'affichage dimensionne sur la texture
                // (scale ≤ 1), une passe ¼ s'affichait rétrécie (bug 2026-08-23).
                let (fw, fh) = (self.rendering_params.width, self.rendering_params.height);
                if (width, height) != (fw, fh) && width > 0 && height > 0 {
                    let up = crate::gui::progressive::upscale_rgb_nearest(
                        &display_buffer,
                        width,
                        height,
                        fw,
                        fh,
                    );
                    self.load_texture_from_buffer(ctx, up, self.rendering_params.clone(), None);
                } else {
                    self.load_texture_from_buffer(ctx, display_buffer, self.rendering_params.clone(), None);
                }
                ctx.request_repaint();
            }

            RenderMessage::NavProgress {
                rgba,
                width,
                height,
            } => {
                // Overlay partiel : composé sur la texture warpée, il ne
                // REMPLACE pas l'image courante (les zones non calculées sont
                // transparentes). Sa vue de référence est celle du rendu en
                // cours, pour rester correctement placé si la navigation a
                // continué depuis.
                if (rgba.len() as u32) == width * height * 4 {
                    let img = egui::ColorImage::from_rgba_unmultiplied(
                        [width as usize, height as usize],
                        &rgba,
                    );
                    match self.nav_overlay_texture.as_mut() {
                        Some(t) if t.size() == [width as usize, height as usize] => {
                            t.set(img, TextureOptions::LINEAR);
                        }
                        _ => {
                            self.nav_overlay_texture = Some(ctx.load_texture(
                                "fractal_nav_overlay",
                                img,
                                TextureOptions::LINEAR,
                            ));
                        }
                    }
                    self.nav_overlay_view = self.rendering_view.clone();
                    if self.ui_trace {
                        self.ui_frame_events.push("nav_overlay".to_string());
                    }
                    ctx.request_repaint();
                }
            }

            RenderMessage::AaProgress {
                display_buffer,
                width: _,
                height: _,
                sample,
                total,
            } => {
                // Moyenne RGB déjà colorisée : on l'affiche directement (pas de
                // re-colorisation depuis le brut, cf. doc du message). La passe
                // courante est la dernière → image nette, plus en preview.
                self.is_preview = false;
                self.render_progress.aa_ready(sample, total);
                if same_colors(&self.params, &self.rendering_params) {
                    self.recolor_version.issue();
                    self.load_texture_from_buffer(ctx, display_buffer, self.rendering_params.clone(), (sample > 1).then_some(sample));
                    self.showing_render_aa = true;
                } // sinon : couleurs changées depuis le lancement → garder la recolorisation
                ctx.request_repaint();
            }

            RenderMessage::AllComplete { orbit_cache } => {
                // Les colorisations lancées peuvent encore arriver dans le désordre.
                // Chaque worker réveille l’UI et passe par le même filtre de progression.
                self.texture_ready_sender = None;

                self.rendering = false;
                self.is_preview = false;
                self.render_thread = None;
                self.render_receiver = None;

                // Store the updated orbit cache for future renders
                if orbit_cache.is_some() {
                    self.orbit_cache = orbit_cache;
                }

                if let Some(start) = self.render_start_time.take() {
                    self.last_render_time = Some(start.elapsed().as_secs_f64());
                    // Le diviseur qui a produit cette durée : sans lui, la
                    // résolution dynamique lirait un coût déjà réduit comme un
                    // coût plein cadre et ne redescendrait jamais.
                    self.last_render_divisor = self.rendering_divisor;
                }
                // G10.4 : point de départ du délai idle avant raffinement.
                self.last_render_finished = Some(Instant::now());
                // Sage : coût de navigation + itérations automatiques (peut
                // relancer un rendu si le plafond doit grandir).
                self.sage_after_render();

            }

            RenderMessage::Cancelled => {
                self.rendering = false;
                self.render_thread = None;
                self.render_receiver = None;
                self.texture_ready_sender = None;
                self.texture_ready_receiver = None;
                // Garder la dernière texture valide affichée
            }
        }
    }

    /// Planifie ou applique un redimensionnement de la surface de rendu.
    /// Si un rendu est en cours, on l'annule et on relance immédiatement à la nouvelle taille
    /// pour éviter d'afficher une image à l'ancienne résolution étirée dans la nouvelle fenêtre.
    fn queue_resize(&mut self, new_width: u32, new_height: u32) {
        if new_width == 0 || new_height == 0 {
            return;
        }
        if (new_width, new_height) == self.view.dimensions() {
            return;
        }
        if self.rendering {
            // Annuler le rendu en cours et relancer tout de suite à la nouvelle taille
            self.render_cancel.store(true, Ordering::Relaxed);
            if let Some(ref receiver) = &self.render_receiver {
                while receiver.try_recv().is_ok() {}
            }
            if let Some(ref receiver) = &self.texture_ready_receiver {
                while receiver.try_recv().is_ok() {}
            }
            if let Some(handle) = self.render_thread.take() {
                drop(handle);
            }
            self.render_receiver = None;
            self.texture_ready_sender = None;
            self.texture_ready_receiver = None;
            self.rendering = false;
        }
        self.apply_resize(new_width, new_height);
    }

    /// Applique le redimensionnement et relance un rendu.
    fn apply_resize(&mut self, new_width: u32, new_height: u32) {
        // ⚠️ Spans recalculés en HP, centre INTOUCHÉ. L'ancienne version
        // travaillait en f64 puis `sync_params_to_hp()` réécrivait les 4 chaînes
        // HP (centre inclus) depuis leurs approximations 53 bits : à zoom
        // profond, tout redimensionnement de fenêtre arrondissait le centre →
        // vue fausse / image uniforme (bug 2026-08-23).
        self.view.resize(new_width, new_height);
        self.view.write_to_params(&mut self.params);
        // Redimensionner n'est pas naviguer : l'entrée courante suit la
        // fenêtre (une vue en attente sera empilée à la bonne taille).
        if !self.history.has_pending() {
            self.history.refresh_current(&self.params);
        }
        self.iterations = Arc::new(Vec::new());
        self.zs = Arc::new(Vec::new());
        self.orbits = Arc::new(Vec::new());
        self.texture = None;
        self.displayed_frame = None;
        self.start_render();
    }

    /// Charge une texture depuis un buffer RGB pré-colorisé.
    /// Utilisé pour les passes de rendu (évite de bloquer l'UI).
    fn load_texture_from_buffer(
        &mut self,
        ctx: &Context,
        rgb_buffer: Vec<u8>,
        params: FractalParams,
        aa_samples: Option<u32>,
    ) {
        let (width, height) = (params.width, params.height);
        if rgb_buffer.len() != width as usize * height as usize * 3 {
            return;
        }
        let Some(img) = RgbImage::from_raw(width, height, rgb_buffer) else {
            return;
        };
        if self.ui_trace {
            self.ui_frame_events.push(format!("tex:{width}x{height}"));
        }
        let color_image = rgb_image_to_color_image(&img);
        self.showing_render_aa = false;
        self.displayed_frame = aa_samples.map(|aa_samples| DisplayedFrame { image: img, params, aa_samples });
        self.texture_view = self.rendering_view.clone();
        self.nav_overlay_texture = None;
        self.nav_overlay_view = None;
        // Mise à jour EN PLACE quand les dimensions sont inchangées : `load_texture`
        // ALLOUE une nouvelle texture GPU à chaque appel (le nom n'est qu'un label)
        // et libère l'ancienne. En navigation continue (dizaines d'images/s) ce
        // churn d'allocations GPU provoque des à-coups périodiques quand le driver
        // récupère la mémoire. `TextureHandle::set` réécrit la texture existante.
        match self.texture.as_mut() {
            Some(t) if t.size() == [width as usize, height as usize] => {
                t.set(color_image, TextureOptions::LINEAR);
            }
            _ => {
                self.texture =
                    Some(ctx.load_texture("fractal", color_image, TextureOptions::LINEAR));
            }
        }
    }

    /// Met à jour la texture egui à partir des données de fractale.
    /// Utilisé pour les changements de palette/couleur (re-colorisation nécessaire).
    /// La colorisation est faite de manière asynchrone pour ne pas bloquer l'UI.
    fn update_texture(&mut self, ctx: &Context) {
        if self.render_progress.passes().0 == 0 {
            return;
        }
        let mut params = self.rendering_params.clone();
        let (width, height) = (params.width, params.height);
        if self.iterations.len() != width as usize * height as usize || self.iterations.len() != self.zs.len() {
            return;
        }

        // S'assurer que orbits a la même taille que iterations/zs.
        // `make_mut` : copy-on-write si un thread de recolor précédent tient encore
        // l'Arc (rare) ; sinon mutation en place.
        let target_len = self.iterations.len();
        if self.orbits.len() != target_len {
            Arc::make_mut(&mut self.orbits).resize(target_len, None);
        }

        // Cloner les Arc (bump de refcount, ~gratuit) pour le thread de colorisation.
        let iterations = Arc::clone(&self.iterations);
        let zs = Arc::clone(&self.zs);
        let distances = Arc::clone(&self.distances);
        let orbits = Arc::clone(&self.orbits);
        params.color = self.params.color.clone();
        // Mettre à jour les params avec les valeurs actuelles de l'UI
        params.color.color_mode = self.palette_index;
        params.color.color_repeat = self.color_repeat;
        params.color.out_coloring_mode = self.out_coloring_mode;

        let tx = self.recolor_sender.clone();
        let preview = self.raw_preview;

        // Incrémenter la version pour ignorer les résultats obsolètes
        let version = self.recolor_version.issue();

        // Spawner un thread pour la colorisation (ne bloque pas l'UI)
        let repaint_ctx = ctx.clone();
        thread::spawn(move || {
            let display_buffer = colorize_buffer(
                &iterations,
                &zs,
                &distances,
                &orbits,
                &params,
                width,
                height,
            );
            let _ = tx.send(RecolorReadyMessage {
                params,
                display_buffer,
                version,
                preview,
            });
            repaint_ctx.request_repaint();
        });
    }

    /// Demande un rendu asynchrone de la preview Julia pour le seed donné.
    /// Throttle les requêtes pour éviter de surcharger le CPU.
    fn request_julia_preview(&mut self, seed: Complex64) {
        // Throttling: ignorer si moins de 50ms depuis la dernière requête
        if let Some(last_time) = self.julia_preview_last_request_time {
            if last_time.elapsed() < Duration::from_millis(50) {
                return;
            }
        }

        // Ignorer si le seed n'a pas changé significativement
        if let Some(last_seed) = self.julia_preview_last_seed {
            let dist = (seed - last_seed).norm();
            if dist < 1e-10 {
                return;
            }
        }

        // Annuler le rendu précédent
        self.julia_preview_cancel.store(true, Ordering::Relaxed);

        // Mettre à jour l'état
        self.julia_preview_last_seed = Some(seed);
        self.julia_preview_last_request_time = Some(Instant::now());
        let version = self.julia_preview_version.issue();

        // Nouveau flag d'annulation
        self.julia_preview_cancel = Arc::new(AtomicBool::new(false));
        let cancel = Arc::clone(&self.julia_preview_cancel);

        // Créer les params pour la preview Julia (type correspondant au Mandelbrot actuel)
        let julia_type = match self.selected_type.julia_variant() {
            Some(jt) => jt,
            None => return, // Pas de variante Julia pour ce type
        };
        let preview_width = 160u32;
        let preview_height = 120u32;
        let mut params = default_params_for_type(julia_type, preview_width, preview_height);
        params.seed = seed;
        params.iteration_max = 256;
        params.engine.algorithm_mode = AlgorithmMode::StandardF64;
        params.color.color_mode = self.palette_index;
        params.color.color_repeat = self.color_repeat;
        params.color.out_coloring_mode = self.out_coloring_mode;
        crate::render::ensure_required_channels(&mut params);

        let tx = self.julia_preview_sender.clone();
        self.julia_preview_rendering = true;

        // Spawner le thread de rendu
        thread::spawn(move || {
            if cancel.load(Ordering::Relaxed) {
                return;
            }

            let result = render_request(RenderRequest::new(&params, &cancel), &mut None);

            if cancel.load(Ordering::Relaxed) {
                return;
            }

            if let Some(out) = result {
                let display_buffer = colorize_buffer(
                    &out.iterations,
                    &out.zs,
                    &out.distances,
                    &out.orbits,
                    &params,
                    preview_width,
                    preview_height,
                );
                let _ = tx.send(JuliaPreviewMessage {
                    display_buffer,
                    width: preview_width,
                    height: preview_height,
                    seed,
                    version,
                });
            }
        });
    }

    /// Vérifie si un rendu de preview Julia est terminé et met à jour la texture.
    fn check_julia_preview_complete(&mut self, ctx: &Context) {
        while let Ok(msg) = self.julia_preview_receiver.try_recv() {
            // Ignorer les résultats obsolètes
            if !self.julia_preview_version.accepts(msg.version) {
                continue;
            }

            // Créer la texture depuis le buffer
            let expected_size = (msg.width as usize) * (msg.height as usize) * 3;
            if msg.display_buffer.len() == expected_size {
                if let Some(img) = RgbImage::from_raw(msg.width, msg.height, msg.display_buffer) {
                    let color_image = rgb_image_to_color_image(&img);
                    self.julia_preview_texture = Some(ctx.load_texture(
                        "julia_preview",
                        color_image,
                        TextureOptions::LINEAR,
                    ));
                    self.julia_preview_last_seed = Some(msg.seed);
                    self.julia_preview_rendering = false;
                    ctx.request_repaint();
                }
            }
            break;
        }
    }

    /// Ratio ([0,1]²) d'un pixel dans l'image de rendu — entrée des zooms HP.
    fn pixel_to_ratio(&self, pixel_x: f32, pixel_y: f32) -> (f64, f64) {
        (
            pixel_x as f64 / self.params.width.max(1) as f64,
            pixel_y as f64 / self.params.height.max(1) as f64,
        )
    }

    /// Calcule les coordonnées complexes (f64) depuis les coordonnées pixel.
    /// Réservé aux usages f64 (seed Julia) — PAS au zoom (cf. `zoom_at_ratio`).
    fn pixel_to_complex(
        &self,
        pixel_x: f32,
        pixel_y: f32,
        viewport_width: f32,
        viewport_height: f32,
    ) -> Complex64 {
        let x_ratio = pixel_x as f64 / viewport_width as f64;
        let y_ratio = pixel_y as f64 / viewport_height as f64;

        // Utiliser center+span directement pour éviter les problèmes de précision
        // x = center_x + (ratio - 0.5) * span_x
        let x = self.params.center_x + (x_ratio - 0.5) * self.params.span_x;
        // y = center_y + (ratio - 0.5) * span_y (même convention que le rendu)
        let y = self.params.center_y + (y_ratio - 0.5) * self.params.span_y;

        Complex64::new(x, y)
    }

    /// Zoom au point spécifié par son RATIO dans l'image ([0,1]², 0.5 = centre)
    /// avec un facteur donné.
    ///
    /// ⚠️ Ne pas passer par un `Complex64` f64 : `center + (r−0.5)·span` puis
    /// `(point − center)/span + 0.5` absorbe le décalage dès que
    /// `span < ~1e-16·|center|` (zoom > ~1e15) → ratio 0.5 quel que soit le
    /// clic (NaN si span f64 = 0). Les ratios pixel vont directement au zoom HP,
    /// comme la molette et le rectangle.
    fn zoom_at_ratio(&mut self, ratio_x: f64, ratio_y: f64, factor: f64) {
        // Utiliser le zoom haute précision
        self.zoom_hp(ratio_x, ratio_y, factor);

        // G10.2 : NE PAS invalider le cache d'orbite sur un zoom. Le moteur décide
        // (`can_subset_reuse`) : si la nouvelle vue est contenue dans l'empreinte de
        // la référence, il la réutilise off-center (rendu quasi-gratuit) ; sinon il
        // rebuild. Correct par construction (sous-ensemble de pixels déjà rendus).
        // GPU : passe `allow_subset_reuse=false` → rebuild exact (pas de régression).

        self.start_render();
    }

    /// Zoom sur une zone rectangulaire sélectionnée.
    /// Les coordonnées sont en pixels dans l'image affichée.
    /// L'image affichée représente déjà une zone zoomée définie par center+span.
    fn zoom_to_rectangle(
        &mut self,
        rect_min: egui::Pos2,
        rect_max: egui::Pos2,
        image_rect: egui::Rect,
    ) {
        // Calculer les ratios relatifs dans l'image affichée (0.0 à 1.0)
        let x_ratio1 = ((rect_min.x - image_rect.min.x) / image_rect.width()) as f64;
        let y_ratio1 = ((rect_min.y - image_rect.min.y) / image_rect.height()) as f64;
        let x_ratio2 = ((rect_max.x - image_rect.min.x) / image_rect.width()) as f64;
        let y_ratio2 = ((rect_max.y - image_rect.min.y) / image_rect.height()) as f64;

        // Clamper les ratios entre 0.0 et 1.0
        let x_ratio1 = x_ratio1.clamp(0.0, 1.0);
        let y_ratio1 = y_ratio1.clamp(0.0, 1.0);
        let x_ratio2 = x_ratio2.clamp(0.0, 1.0);
        let y_ratio2 = y_ratio2.clamp(0.0, 1.0);

        // S'assurer que r1 < r2
        let (xr1, xr2) = if x_ratio1 < x_ratio2 {
            (x_ratio1, x_ratio2)
        } else {
            (x_ratio2, x_ratio1)
        };
        let (yr1, yr2) = if y_ratio1 < y_ratio2 {
            (y_ratio1, y_ratio2)
        } else {
            (y_ratio2, y_ratio1)
        };

        // Vérifier que le rectangle a une taille minimale
        if (xr2 - xr1) < 0.01 || (yr2 - yr1) < 0.01 {
            return; // Rectangle trop petit (moins de 1% de l'image)
        }

        // Utiliser le zoom rectangulaire haute précision
        self.zoom_rect_hp(xr1, yr1, xr2, yr2);

        // G10.2 : cache d'orbite conservé (réutilisation subset off-center si la
        // vue est contenue ; rebuild sinon — décidé par le moteur). Cf. zoom_at_ratio.

        self.start_render();
    }

    /// Dézoom avec un facteur donné.
    fn zoom_out(&mut self, factor: f64) {
        self.zoom_out_hp(factor);
        self.start_render();
    }

    /// Dézoom au point spécifié : symétrique de `zoom_at_ratio` (re-centre sur le
    /// point sous le curseur puis élargit le span). Avant, `_point` était ignoré
    /// → le dézoom se faisait toujours au centre (asymétrie avec le clic gauche).
    fn zoom_out_at_ratio(&mut self, ratio_x: f64, ratio_y: f64, factor: f64) {
        // zoom_factor < 1 ⇒ new_span = span / zoom_factor = span × factor (dézoom),
        // tout en re-centrant sur le point (même chemin HP que le zoom in).
        self.zoom_hp(ratio_x, ratio_y, 1.0 / factor);
        self.orbit_cache = None;
        self.start_render();
    }

    /// Change le preset Lyapunov.
    fn change_lyapunov_preset(&mut self, preset: LyapunovPreset) {
        if self.selected_lyapunov_preset == preset && self.selected_type == FractalType::Lyapunov {
            return;
        }

        self.selected_lyapunov_preset = preset;

        // Si on n'est pas déjà sur Lyapunov, changer le type
        if self.selected_type != FractalType::Lyapunov {
            self.selected_type = FractalType::Lyapunov;
            let width = self.params.width;
            let height = self.params.height;
            self.params = default_params_for_type(FractalType::Lyapunov, width, height);
        }

        // Appliquer le preset
        apply_lyapunov_preset(&mut self.params, preset);
        // Synchroniser les coordonnées HP
        self.sync_params_to_hp();
        self.on_new_scene();
        self.start_render();
    }

    /// Change le type de fractale.
    /// Réinitialise toujours la position au domaine par défaut (zoom, centre, etc.).
    /// Si on reselectionne le même type, recharge quand même les defaults.
    /// Nom court d'une phase pour l'affichage de la séquence hybride (menu +
    /// label). Mirror des alias CLI `FractalType::from_hybrid_name`.
    fn hybrid_short_name(t: FractalType) -> &'static str {
        match t {
            FractalType::Mandelbrot => "M",
            FractalType::Julia => "J",
            FractalType::BurningShip => "BS",
            FractalType::Tricorn => "Tri",
            FractalType::Celtic => "Cel",
            FractalType::Buffalo => "Buf",
            FractalType::PerpendicularBurningShip => "PBS",
            FractalType::Multibrot => "Mu",
            _ => "?",
        }
    }

    fn hybrid_seq_label(seq: &[FractalType]) -> String {
        seq.iter()
            .map(|t| Self::hybrid_short_name(*t))
            .collect::<Vec<_>>()
            .join("⊕")
    }

    /// G4 jalon 5c : applique la séquence hybride éditée (menu Type →
    /// « Hybride »). Même reset que `change_fractal_type` (les hybrides
    /// portent la convention Mandelbrot : type 3, δ₀=0, dc=pixel) + pose
    /// `params.hybrid_phases`. Le rendu suit le chemin unique CLI↔GUI
    /// (`formula_for_params` → f64-std ou perturbation multi-phase).
    fn apply_hybrid_sequence(&mut self) {
        if self.hybrid_seq.len() < 2 {
            return;
        }
        self.selected_type = FractalType::Mandelbrot;
        let width = self.params.width;
        let height = self.params.height;
        let mut new_params = crate::fractal::params_for_type_keeping_preferences(
            &self.params,
            FractalType::Mandelbrot,
            width,
            height,
        );
        new_params.formula.hybrid_phases = Some(self.hybrid_seq.clone());
        self.params = new_params;
        self.color_repeat = self.params.color.color_repeat;
        self.iteration_input = self.params.iteration_max.to_string();
        self.sync_params_to_hp();
        self.use_gpu = false; // GPU ne cycle pas les phases (render_dispatch → None)
        self.orbit_cache = None;
        self.julia_preview_texture = None;
        self.julia_preview_last_seed = None;
        self.julia_preview_cancel.store(true, Ordering::Relaxed);
        self.on_new_scene();
        self.start_render();
    }

    fn change_fractal_type(&mut self, new_type: FractalType) {
        self.selected_type = new_type;
        let width = self.params.width;
        let height = self.params.height;

        // Défauts du nouveau type + préférences de l'utilisateur (couleur,
        // échantillonnage, moteur, perturbation, canaux) : la frontière est
        // écrite groupe par groupe dans `params_for_type_keeping_preferences`,
        // plus par une liste blanche de champs qui perdait le reste.
        // Géométrie : toujours le domaine par défaut, pour bien centrer.
        let new_params =
            crate::fractal::params_for_type_keeping_preferences(&self.params, new_type, width, height);
        self.params = new_params;
        self.color_repeat = self.params.color.color_repeat;
        self.iteration_input = self.params.iteration_max.to_string();
        // Synchroniser les coordonnées HP depuis les nouvelles params
        self.sync_params_to_hp();
        self.use_gpu = false;
        // Invalidate orbit cache when changing fractal type
        self.orbit_cache = None;

        // Clear Julia preview (only relevant for Mandelbrot)
        self.julia_preview_texture = None;
        self.julia_preview_last_seed = None;
        self.julia_preview_cancel.store(true, Ordering::Relaxed);

        self.on_new_scene();
        self.start_render();
    }

    /// Params de la vue telle qu'affichée, pour la copie de coordonnées et la
    /// commande CLI : couleurs courantes de l'UI (comme la recolorisation),
    /// chaînes HP de la vue, dimensions de la fenêtre.
    fn export_params(&self) -> FractalParams {
        let mut p = self.params.clone();
        p.color.color_mode = self.palette_index;
        p.color.color_repeat = self.color_repeat;
        p.color.out_coloring_mode = self.out_coloring_mode;
        let (cx, cy, sx, sy) = self.view.decimal_parts();
        p.center_x_hp = Some(cx);
        p.center_y_hp = Some(cy);
        p.span_x_hp = Some(sx);
        p.span_y_hp = Some(sy);
        let (w, h) = self.view.dimensions();
        p.width = w;
        p.height = h;
        p
    }

    fn effective_algorithm_mode(&self) -> AlgorithmMode {
        match self.params.engine.algorithm_mode {
            AlgorithmMode::Auto => {
                if !matches!(
                    self.selected_type,
                    FractalType::Mandelbrot
                        | FractalType::Julia
                        | FractalType::BurningShip
                        | FractalType::Tricorn
                ) || self.params.width == 0
                {
                    return AlgorithmMode::StandardF64;
                }
                // Use appropriate threshold based on GPU/CPU mode
                let gpu_mode = self.use_gpu && self.gpu_renderer.is_some();
                if crate::render::escape_time::should_use_perturbation(&self.params, gpu_mode) {
                    AlgorithmMode::Perturbation
                } else {
                    AlgorithmMode::StandardF64
                }
            }
            other => other,
        }
    }
}

impl eframe::App for FractallApp {
    // eframe 0.37 : `ui` est l'unique point d'entrée (l'ancien `update` a été
    // retiré). Les panneaux s'affichent dans le `Ui` racine ; fenêtres et
    // entrées passent par le `Context` qu'il porte.
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = &ui.ctx().clone();
        // Trace des à-coups d'affichage (FRACTALL_UI_TRACE=1) : signale toute
        // frame UI dont l'intervalle dépasse le seuil, avec les événements de la
        // frame précédente. Sert à distinguer un « pop » de contenu (nouvelle
        // image nette qui remplace le warp) d'un vrai gel du thread UI.
        if self.ui_trace {
            let now = Instant::now();
            if let Some(prev) = self.ui_frame_last {
                let ms = now.duration_since(prev).as_secs_f64() * 1000.0;
                if ms > UI_TRACE_GAP_MS {
                    println!(
                        "[UI] gap={:.1}ms rendering={} events=[{}]",
                        ms,
                        self.rendering,
                        self.ui_frame_events.join(" ")
                    );
                }
            }
            self.ui_frame_last = Some(now);
            self.ui_frame_events.clear();
        }

        // Init différée du GPU : une fois la fenêtre eframe prête, éviter "Parent device is lost" sur NVIDIA
        if !self.gpu_init_attempted {
            self.gpu_init_attempted = true;
            if self.gpu_renderer.is_none() {
                self.gpu_renderer = GpuRenderer::new().map(Arc::new);
                if self.gpu_renderer.is_none() {
                    eprintln!("⚠️  GPU not available - GPU rendering disabled (CPU only)");
                }
            }
        }

        // Vérifier si le rendu est terminé
        self.check_render_complete(ctx);
        // Mode souris : appliquer la décision du sage hors geste.

        // G10.4 : raffinement idle — la dernière frame contient des pixels
        // copiés (approximés ≤ 0.5 px) ; après ~400 ms sans nouveau rendu,
        // relancer un rendu exact (passe unique, XaoS off) qui remplace
        // silencieusement l'image. Toute interaction (pan/zoom) le supplante.
        if self.xaos.refine_pending()
            && !self.rendering
            && !self.hq_render_state.is_running()
        {
            const XAOS_REFINE_IDLE: Duration = Duration::from_millis(400);
            let idle_for = self
                .last_render_finished
                .map(|t| t.elapsed())
                .unwrap_or(Duration::ZERO);
            if idle_for >= XAOS_REFINE_IDLE {
                if self.xaos.take_refine() {
                    self.start_render_refine();
                }
            } else {
                // S'assurer qu'update() retourne après le délai même sans input.
                ctx.request_repaint_after(XAOS_REFINE_IDLE - idle_for);
            }
        }

        // Vérifier si la preview Julia est terminée
        self.check_julia_preview_complete(ctx);

        // Détection des fichiers déposés (drag-and-drop)
        ctx.input(|i| {
            for file in &i.raw.dropped_files {
                {
                    let path = file.path();
                    match path.extension().and_then(|e| e.to_str()) {
                        Some("png") => self.load_from_png(path),
                        Some("toml") => self.load_from_toml(path),
                        _ => {}
                    }
                }
            }
        });

        // Gestion des raccourcis clavier — inhibés quand un champ texte a le
        // focus (itérations, coordonnées…) : taper `-`, `+`, `c`… dans le champ
        // zoomait / changeait la palette (bug 2026-08-23 ; seul `0` était gaté).
        let text_field_focused = ctx.egui_wants_keyboard_input();
        // Historique : Backspace / Alt+← = retour, Alt+→ = avance (jamais
        // quand un champ texte a le focus : Backspace y efface).
        if !text_field_focused && !self.iteration_input_has_focus {
            let (back, forward) = ctx.input(|i| {
                (
                    i.key_pressed(egui::Key::Backspace)
                        || (i.modifiers.alt && i.key_pressed(egui::Key::ArrowLeft)),
                    i.modifiers.alt && i.key_pressed(egui::Key::ArrowRight),
                )
            });
            if back {
                self.navigate_history(false);
            } else if forward {
                self.navigate_history(true);
            }
        }
        ctx.input(|i| {
            if text_field_focused {
                return;
            }
            // F1-F12 pour changer le type
            for (key, fractal_id) in [
                (egui::Key::F1, 3),
                (egui::Key::F2, 4),
                (egui::Key::F3, 5),
                (egui::Key::F4, 6),
                (egui::Key::F5, 7),
                (egui::Key::F6, 8),
                (egui::Key::F7, 9),
                (egui::Key::F8, 10),
                (egui::Key::F9, 11),
                (egui::Key::F10, 12),
                (egui::Key::F11, 13),
                (egui::Key::F12, 14),
            ] {
                if i.key_pressed(key) {
                    if let Some(fractal_type) = FractalType::from_id(fractal_id) {
                        self.change_fractal_type(fractal_type);
                    }
                }
            }

            // C pour cycle palette
            if i.key_pressed(egui::Key::C) {
                self.palette_index = (self.palette_index + 1) % 27;
                // Synchroniser params (comme R et les boutons < / >) : sinon la
                // passe suivante / un screenshot reviennent à l'ancienne palette.
                self.params.color.color_mode = self.palette_index;
                if !self.iterations.is_empty() {
                    self.update_texture(ctx);
                }
            }

            // R pour color_repeat (1-120 ou 1-8 pour densité)
            if i.key_pressed(egui::Key::R) {
                let max_repeat = if matches!(
                    self.selected_type,
                    FractalType::Buddhabrot | FractalType::Nebulabrot | FractalType::AntiBuddhabrot
                ) {
                    8
                } else {
                    120
                };
                self.color_repeat = if self.color_repeat >= max_repeat {
                    1
                } else {
                    self.color_repeat + 1
                };
                self.params.color.color_repeat = self.color_repeat;
                if !self.iterations.is_empty() {
                    self.update_texture(ctx);
                }
            }

            // J : basculer vers Julia si seed dispo, sinon activer le mode preview Julia
            if i.key_pressed(egui::Key::J) {
                if self.selected_type.has_julia_variant()
                    && self.julia_preview_enabled
                    && self.julia_preview_last_seed.is_some()
                {
                    let seed = self.julia_preview_last_seed.unwrap();
                    let julia_type = self.selected_type.julia_variant().unwrap();
                    // Désactiver le mode preview
                    self.julia_preview_enabled = false;
                    self.julia_preview_texture = None;
                    self.julia_preview_cancel.store(true, Ordering::Relaxed);
                    // Basculer vers Julia avec ce seed
                    let width = self.params.width;
                    let height = self.params.height;
                    let mut new_params = default_params_for_type(julia_type, width, height);
                    new_params.seed = seed;
                    new_params.engine.use_gmp = self.params.engine.use_gmp;
                    new_params.engine.precision_bits = self.params.engine.precision_bits;
                    new_params.color.color_mode = self.params.color.color_mode;
                    new_params.color.color_repeat = self.params.color.color_repeat;
                    new_params.engine.algorithm_mode = AlgorithmMode::Auto;
                    self.params = new_params;
                    self.selected_type = julia_type;
                    self.iteration_input = self.params.iteration_max.to_string();
                    self.sync_params_to_hp();
                    self.orbit_cache = None;
                    self.on_new_scene();
                    self.start_render();
                } else if self.selected_type.has_julia_variant() && !self.julia_preview_enabled {
                    // Pas de sélection Julia : activer le mode preview et cocher la case
                    self.julia_preview_enabled = true;
                }
            }

            // S pour screenshot avec métadonnées
            if i.key_pressed(egui::Key::S) {
                if let Some(frame) = self.displayed_frame.as_ref().filter(|_| {
                    self.nav_overlay_texture.is_none()
                        && self.texture_view.as_ref() == Some(&self.current_view_snapshot())
                }) {
                    let timestamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let filename = format!("fractal_{timestamp}.png");
                    match frame.save(std::path::Path::new(&filename)) {
                        Ok(()) => println!("Screenshot saved with metadata: {filename}"),
                        Err(e) => eprintln!("PNG export error: {e}"),
                    }
                } else {
                    self.clipboard_feedback = Some(("Screenshot needs a complete frame of this view", Instant::now()));
                }
            }

            // Désactiver les raccourcis de zoom en mode Julia preview
            let julia_mode = self.selected_type.has_julia_variant() && self.julia_preview_enabled;

            // + ou = pour zoom au centre (désactivé en mode Julia)
            if !julia_mode {
                let zoom_in_pressed = i
                    .events
                    .iter()
                    .any(|e| matches!(e, egui::Event::Text(text) if text == "+" || text == "="));
                if zoom_in_pressed {
                    self.zoom_at_ratio(0.5, 0.5, 1.5);
                }
            }

            // - pour dézoom au centre (désactivé en mode Julia)
            if !julia_mode && i.key_pressed(egui::Key::Minus) {
                self.zoom_out_at_ratio(0.5, 0.5, 1.5);
            }

            // 0 pour reset zoom (désactivé en mode Julia) — ne pas déclencher si le focus est sur le champ Itérations (pour pouvoir taper "50", "500", etc.)
            if !julia_mode && !self.iteration_input_has_focus && i.key_pressed(egui::Key::Num0) {
                use crate::fractal::default_params_for_type;
                let width = self.params.width;
                let height = self.params.height;
                let mut new_params = default_params_for_type(self.selected_type, width, height);
                // Conserver certains paramètres
                new_params.engine.use_gmp = self.params.engine.use_gmp;
                new_params.engine.precision_bits = self.params.engine.precision_bits;
                new_params.color.color_mode = self.params.color.color_mode;
                new_params.color.color_repeat = self.params.color.color_repeat;
                self.params = new_params;
                self.iteration_input = self.params.iteration_max.to_string();
                // Synchroniser les coordonnées HP
                self.sync_params_to_hp();
                self.orbit_cache = None;
                self.on_new_scene();
                self.start_render();
            }
        });

        // Panneau de contrôle en haut
        let window_width = ctx.viewport_rect().width();
        let top_min = sticky_bar_min(&mut self.bar_heights[0], window_width);
        let top_response = egui::Panel::top("controls").min_size(top_min).show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {

                    // Menu Type (toutes catégories)
                    wrap_group(ui, "bar_group_2827", |ui| {
                        ui.label("Type:");

                        // Menu Type : Mandelbrots à la racine, Julias dans un dossier
                        let density_types = [(16, "Buddhabrot"), (24, "Nebulabrot"), (33, "Anti-Buddhabrot")];

                        let current_category = self.selected_type.menu_family();
                        let current_label = match self.selected_type {
                            FractalType::Lyapunov => self.selected_lyapunov_preset.name(),
                            _ => self.selected_type.name(),
                        };

                        let type_menu_label = if let Some(seq) =
                            self.params.formula.hybrid_phases.as_ref().filter(|s| !s.is_empty())
                        {
                            format!("⏷ Hybrid: {}", Self::hybrid_seq_label(seq))
                        } else if current_category == current_label {
                            format!("⏷ {}", current_label)
                        } else {
                            format!("⏷ {}: {}", current_category, current_label)
                        };
                        ui.menu_button(&type_menu_label, |ui| {
                            // Nova tout en haut
                            if let Some(fractal_type) = FractalType::from_id(22) {
                                if ui.selectable_label(self.selected_type == fractal_type, "Nova").clicked() {
                                    self.change_fractal_type(fractal_type);
                                    ui.close();
                                }
                            }

                            // Lyapunov et Densité
                            ui.menu_button("Lyapunov", |ui| {
                                for preset in LyapunovPreset::all() {
                                    let is_selected = self.selected_type == FractalType::Lyapunov
                                        && self.selected_lyapunov_preset == *preset;
                                    if ui.selectable_label(is_selected, preset.name()).clicked() {
                                        self.change_lyapunov_preset(*preset);
                                        ui.close();
                                    }
                                }
                            });

                            ui.menu_button("Density", |ui| {
                                for (id, label) in density_types.iter() {
                                    if let Some(fractal_type) = FractalType::from_id(*id) {
                                        if ui.selectable_label(self.selected_type == fractal_type, *label).clicked() {
                                            self.change_fractal_type(fractal_type);
                                            ui.close();
                                        }
                                    }
                                }
                            });

                            ui.separator();

                            // Mandelbrots à la racine (pas de dossiers)
                            let mandelbrot_types = [
                                (3, "Mandelbrot"),
                                (10, "Barnsley Mandelbrot"),
                                (12, "Magnet Mandelbrot"),
                                (13, "Burning Ship"),
                                (18, "Perp. Burning Ship"),
                                (14, "Tricorn"),
                                (19, "Celtic"),
                                (8, "Buffalo"),
                                (23, "Multibrot"),
                                (20, "Alpha Mandelbrot"),
                                (32, "Mandelbrot Sin"),
                            ];
                            for (id, label) in mandelbrot_types.iter() {
                                if let Some(fractal_type) = FractalType::from_id(*id) {
                                    if ui.selectable_label(self.selected_type == fractal_type, *label).clicked() {
                                        self.change_fractal_type(fractal_type);
                                        ui.close();
                                    }
                                }
                            }

                            // G4 jalon 5c : éditeur de séquence hybride multi-phase
                            // (phases itérées cycliquement, CLI `--phases`).
                            ui.menu_button("Hybrid (sequence)", |ui| {
                                let seq_label = if self.hybrid_seq.is_empty() {
                                    "—".to_string()
                                } else {
                                    Self::hybrid_seq_label(&self.hybrid_seq)
                                };
                                ui.label(format!("Sequence: {seq_label}"));
                                ui.separator();
                                let eligible: [(FractalType, &str); 7] = [
                                    (FractalType::Mandelbrot, "Mandelbrot"),
                                    (FractalType::BurningShip, "Burning Ship"),
                                    (FractalType::Tricorn, "Tricorn"),
                                    (FractalType::Celtic, "Celtic"),
                                    (FractalType::Buffalo, "Buffalo"),
                                    (FractalType::PerpendicularBurningShip, "Perp. Burning Ship"),
                                    (FractalType::Multibrot, "Multibrot"),
                                ];
                                for (t, name) in eligible.iter() {
                                    if ui.button(format!("➕ {name}")).clicked() {
                                        self.hybrid_seq.push(*t);
                                    }
                                }
                                ui.separator();
                                ui.horizontal(|ui| {
                                    if ui.button("⌫ Remove last").clicked() {
                                        self.hybrid_seq.pop();
                                    }
                                    if ui.button("🗑 Clear").clicked() {
                                        self.hybrid_seq.clear();
                                    }
                                });
                                ui.separator();
                                let can_apply = self.hybrid_seq.len() >= 2;
                                if ui
                                    .add_enabled(can_apply, egui::Button::new("▶ Apply (render)"))
                                    .clicked()
                                {
                                    self.apply_hybrid_sequence();
                                    ui.close();
                                }
                                if !can_apply {
                                    ui.small("≥ 2 phases required");
                                }
                                if self.params.formula.hybrid_phases.is_some()
                                    && ui.button("✖ Disable (Mandelbrot)").clicked()
                                {
                                    self.hybrid_seq.clear();
                                    self.change_fractal_type(FractalType::Mandelbrot);
                                    ui.close();
                                }
                            });

                            // Dossier Julia all juste après Alpha Mandelbrot
                            ui.menu_button("Julia all", |ui| {
                                let julia_types = [
                                    (4, "Julia"),
                                    (9, "Barnsley Julia"),
                                    (11, "Magnet Julia"),
                                    (25, "Burning Ship Julia"),
                                    (30, "Perpendicular Burning Ship Julia"),
                                    (26, "Tricorn Julia"),
                                    (27, "Celtic Julia"),
                                    (28, "Buffalo Julia"),
                                    (29, "Multibrot Julia"),
                                    (31, "Alpha Mandelbrot Julia"),
                                    (5, "Julia Sin"),
                                ];
                                for (id, label) in julia_types.iter() {
                                    if let Some(fractal_type) = FractalType::from_id(*id) {
                                        if ui.selectable_label(self.selected_type == fractal_type, *label).clicked() {
                                            self.change_fractal_type(fractal_type);
                                            ui.close();
                                        }
                                    }
                                }
                            });

                            ui.separator();

                            // Mandelbulb à la racine
                            if let Some(fractal_type) = FractalType::from_id(15) {
                                if ui.selectable_label(self.selected_type == fractal_type, "Mandelbulb").clicked() {
                                    self.change_fractal_type(fractal_type);
                                    ui.close();
                                }
                            }

                            let autres_types = [
                                (6, "Newton"),
                                (7, "Phoenix"),
                                (21, "Pickover Stalks"),
                            ];
                            for (id, label) in autres_types.iter() {
                                if let Some(fractal_type) = FractalType::from_id(*id) {
                                    if ui.selectable_label(self.selected_type == fractal_type, *label).clicked() {
                                        self.change_fractal_type(fractal_type);
                                        ui.close();
                                    }
                                }
                            }

                        });
                    });

                    ui.separator();

                    // Plane (XaoS-style) uniquement pour fractales escape-time
                    let is_escape_time = !matches!(
                        self.selected_type,
                        FractalType::VonKoch | FractalType::Dragon | FractalType::Buddhabrot | FractalType::Lyapunov | FractalType::Nebulabrot | FractalType::AntiBuddhabrot
                    );
                    
                    if is_escape_time {
                        wrap_group(ui, "bar_group_3020", |ui| {
                            ui.label("Plane:");
                            let old_plane = self.params.plane_transform;
                            egui::ComboBox::from_id_salt("plane_transform")
                                .selected_text(self.params.plane_transform.name())
                                .show_ui(ui, |ui| {
                                    for plane in PlaneTransform::all() {
                                        ui.selectable_value(&mut self.params.plane_transform, *plane, plane.name());
                                    }
                                });
                            if old_plane != self.params.plane_transform {
                                self.orbit_cache = None;
                                self.on_new_scene();
                                if self.params.plane_transform != PlaneTransform::Mu
                                    && self.params.engine.algorithm_mode == AlgorithmMode::Perturbation
                                {
                                    self.params.engine.algorithm_mode = AlgorithmMode::Auto;
                                }
                                self.start_render();
                            }
                        });
                    }

                    ui.separator();

                    // Checkbox Julia mode (pour les types Mandelbrot-like avec variante Julia)
                    if self.selected_type.has_julia_variant() {
                        let old_enabled = self.julia_preview_enabled;
                        ui.checkbox(&mut self.julia_preview_enabled, "Julia");
                        if old_enabled && !self.julia_preview_enabled {
                            // Nettoyer quand désactivé
                            self.julia_preview_texture = None;
                            self.julia_preview_last_seed = None;
                            self.julia_preview_cancel.store(true, Ordering::Relaxed);
                        }
                    }

                    // Checkbox navigation XaoS : remplace le clic ×2 / la
                    // sélection rectangle par un zoom continu au bouton maintenu.
                    let old_xaos_nav = self.xaos_nav_mode;
                    ui.checkbox(&mut self.xaos_nav_mode, "XaoS").on_hover_text(
                        "XaoS-style navigation: hold the left button = continuous zoom in \
                         anchored at the cursor, right button = zoom out. Replaces the ×2 click and \
                         the rectangle selection.",
                    );
                    if old_xaos_nav != self.xaos_nav_mode {
                        // Sortie/entrée du mode : abandonner tout état transitoire.
                        self.xaos_nav_last_tick = None;
                        self.xaos_nav_dragging = false;
                        self.nav.reset();
                        self.selection.cancel();
                    }

                    ui.separator();

                    wrap_group(ui, "bar_group_3075", |ui| {
                        ui.label("Iter:");
                        let response = ui.add(
                            egui::TextEdit::singleline(&mut self.iteration_input)
                                .desired_width(60.0)
                        );
                        self.iteration_input_has_focus = response.has_focus();
                        // Itérations automatiques (sage) : grisées pour les types
                        // sans plafond « escape-time » (densité, Lyapunov, Newton…).
                        let auto_applicable =
                            crate::fractal::wisdom::auto_iterations_applicable(&self.params);
                        let mut auto_shown = self.iter_auto && auto_applicable;
                        let auto_response = ui
                            .add_enabled(auto_applicable, egui::Checkbox::new(&mut auto_shown, "Auto"))
                            .on_hover_text(
                                "fractall adjusts the iteration limit to the view; type a number \
                                 to set it yourself.",
                            )
                            .on_disabled_hover_text(
                                "Automatic iterations do not apply to this fractal type: \
                                 the limit is set by hand.",
                            );
                        if auto_response.changed() {
                            self.iter_auto = auto_shown;
                            self.pending_iter_shrink = None;
                            if self.iter_auto {
                                // Réactivation : réévaluer la dernière image terminée
                                // (un rendu en cours sera évalué à sa fin).
                                self.auto_grow_guard.reset();
                                if !self.rendering {
                                    if let Some((info, iterations)) = self.last_final_pass.clone() {
                                        self.assess_final_pass(&info, &iterations);
                                    }
                                }
                            }
                        }
                        // Appliquer la valeur à la perte de focus ou sur Entrée
                        if response.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                            if let Ok(val) = self.iteration_input.trim().parse::<u32>() {
                                let val = val.max(1);
                                if val != self.params.iteration_max {
                                    // Valeur choisie par l'utilisateur : respectée.
                                    self.iter_auto = false;
                                    self.pending_iter_shrink = None;
                                    self.params.iteration_max = val;
                                    self.iteration_input = val.to_string();
                                    self.orbit_cache = None;
                                    self.start_render();
                                }
                            } else {
                                self.iteration_input = self.params.iteration_max.to_string();
                            }
                        }
                    });

                    ui.separator();

                    // Section Tech (entre Iter et Render)
                    let supports_advanced_modes = matches!(
                        self.selected_type,
                        FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip
                    );
                    let is_lyapunov = self.selected_type == FractalType::Lyapunov;

                    if supports_advanced_modes || is_lyapunov {
                        let old_mode = self.params.engine.algorithm_mode;
                        let old_use_gpu = self.use_gpu;
                        wrap_group(ui, "bar_group_3142", |ui| {
                            ui.label("Tech:");
                            let gpu_available = self.gpu_renderer.is_some();

                            let render_text = if is_lyapunov {
                                match self.params.engine.algorithm_mode {
                                    AlgorithmMode::Auto => "🔄 Auto".to_string(),
                                    AlgorithmMode::StandardF64 => "💻 CPU Standard f64".to_string(),
                                    _ => "🔄 Auto".to_string(),
                                }
                            } else {
                                match (self.use_gpu && gpu_available, self.params.engine.algorithm_mode) {
                                    (_, AlgorithmMode::Auto) => "🔄 Auto".to_string(),
                                    (false, AlgorithmMode::StandardF64) => "💻 CPU Standard f64".to_string(),
                                    (false, AlgorithmMode::Perturbation) => "💻 CPU Perturbation f64".to_string(),
                                    (false, AlgorithmMode::ReferenceGmp) => "💻 CPU GMP Reference".to_string(),
                                    (true, AlgorithmMode::StandardF64) => "🎮 GPU Standard f32".to_string(),
                                    (true, AlgorithmMode::Perturbation) => "🎮 GPU Perturbation f32".to_string(),
                                    (true, AlgorithmMode::ReferenceGmp) => "🔄 Auto".to_string(),
                                }
                            };

                            ui.menu_button(&render_text, |ui| {
                                if ui.selectable_label(
                                    self.params.engine.algorithm_mode == AlgorithmMode::Auto,
                                    "🔄 Auto"
                                ).clicked() {
                                    self.params.engine.algorithm_mode = AlgorithmMode::Auto;
                                    if is_lyapunov {
                                        self.use_gpu = false;
                                    }
                                    ui.close();
                                }

                                if is_lyapunov {
                                    if ui.selectable_label(
                                        !self.use_gpu && self.params.engine.algorithm_mode == AlgorithmMode::StandardF64,
                                        "💻 CPU Standard f64"
                                    ).clicked() {
                                        self.use_gpu = false;
                                        self.params.engine.algorithm_mode = AlgorithmMode::StandardF64;
                                        ui.close();
                                    }
                                } else {
                                    ui.separator();

                                    ui.menu_button("💻 CPU", |ui| {
                                        if ui.selectable_label(
                                            !self.use_gpu && self.params.engine.algorithm_mode == AlgorithmMode::StandardF64,
                                            "📊 Standard f64"
                                        ).clicked() {
                                            self.use_gpu = false;
                                            self.params.engine.algorithm_mode = AlgorithmMode::StandardF64;
                                            ui.close();
                                        }

                                        if ui.selectable_label(
                                            !self.use_gpu && self.params.engine.algorithm_mode == AlgorithmMode::ReferenceGmp,
                                            "🔢 GMP Reference"
                                        ).clicked() {
                                            self.use_gpu = false;
                                            self.params.engine.algorithm_mode = AlgorithmMode::ReferenceGmp;
                                            ui.close();
                                        }

                                        let plane_ok = self.params.plane_transform == PlaneTransform::Mu;
                                        if ui
                                            .add_enabled(
                                                plane_ok,
                                                egui::Button::selectable(
                                                    !self.use_gpu && self.params.engine.algorithm_mode == AlgorithmMode::Perturbation,
                                                    "🔬 Perturbation f64",
                                                ),
                                            )
                                            .clicked()
                                        {
                                            self.use_gpu = false;
                                            self.params.engine.algorithm_mode = AlgorithmMode::Perturbation;
                                            ui.close();
                                        }
                                    });

                                    if gpu_available {
                                        ui.menu_button("🎮 GPU", |ui| {
                                            if ui.selectable_label(
                                                self.use_gpu && self.params.engine.algorithm_mode == AlgorithmMode::StandardF64,
                                                "⚡ Standard f32"
                                            ).clicked() {
                                                self.use_gpu = true;
                                                self.params.engine.algorithm_mode = AlgorithmMode::StandardF64;
                                                ui.close();
                                            }

                                            let plane_ok = self.params.plane_transform == PlaneTransform::Mu;
                                            if ui
                                                .add_enabled(
                                                    plane_ok,
                                                    egui::Button::selectable(
                                                        self.use_gpu && self.params.engine.algorithm_mode == AlgorithmMode::Perturbation,
                                                        "🚀 Perturbation f32",
                                                    ),
                                                )
                                                .clicked()
                                            {
                                                self.use_gpu = true;
                                                self.params.engine.algorithm_mode = AlgorithmMode::Perturbation;
                                                ui.close();
                                            }
                                        });
                                    } else {
                                        ui.add_enabled(false, egui::Label::new("🎮 GPU (unavailable)"));
                                    }
                                }
                            });
                        });

                        if supports_advanced_modes {
                            if old_use_gpu != self.use_gpu {
                                if self.use_gpu && self.params.engine.algorithm_mode == AlgorithmMode::ReferenceGmp {
                                    self.params.engine.algorithm_mode = AlgorithmMode::Auto;
                                }
                                self.orbit_cache = None;
                                self.start_render();
                            }

                            if self.params.plane_transform != PlaneTransform::Mu
                                && self.params.engine.algorithm_mode == AlgorithmMode::Perturbation
                            {
                                self.params.engine.algorithm_mode = AlgorithmMode::Auto;
                            }

                            if old_mode != self.params.engine.algorithm_mode {
                                self.orbit_cache = None;
                                self.start_render();
                            }
                        } else if is_lyapunov && old_mode != self.params.engine.algorithm_mode {
                            self.start_render();
                        }
                    }

                    ui.separator();
                    if ui.button("🎬 Render").clicked() {
                        self.show_render_dialog = true;
                    }
                    ui.separator();
                    // Historique de navigation (◀ retour, ▶ avance).
                    wrap_group(ui, "bar_group_2808", |ui| {
                        if ui
                            .add_enabled(self.history.can_back(), egui::Button::new("◀"))
                            .on_hover_text("Back to the previous view (Backspace or Alt+←)")
                            .clicked()
                        {
                            self.navigate_history(false);
                        }
                        if ui
                            .add_enabled(self.history.can_forward(), egui::Button::new("▶"))
                            .on_hover_text("Forward to the next view (Alt+→)")
                            .clicked()
                        {
                            self.navigate_history(true);
                        }
                    });
                });
                
                ui.separator();
                
                ui.horizontal_wrapped(|ui| {
                    wrap_group(ui, "bar_group_3291", |ui| {
                        ui.label("Palette:");
                        if ui.button("<").clicked() {
                            self.palette_index = if self.palette_index == 0 { 26 } else { self.palette_index - 1 };
                            self.params.color.color_mode = self.palette_index;
                            if !self.iterations.is_empty() {
                                self.update_texture(ctx);
                            }
                        }
                    
                        // Afficher la prévisualisation de la palette
                        let palette_idx = self.palette_index as usize;
                        if self.palette_preview_textures[palette_idx].is_none() {
                            let preview_image = generate_palette_preview(self.palette_index, 100, 12, self.params.color.color_space);
                            self.palette_preview_textures[palette_idx] = Some(ctx.load_texture(
                                format!("palette_preview_{}", self.palette_index),
                                preview_image,
                                egui::TextureOptions::LINEAR
                            ));
                        }
                    
                        if let Some(ref palette_texture) = self.palette_preview_textures[palette_idx] {
                            ui.add(egui::Image::new(palette_texture)
                                .fit_to_exact_size(egui::Vec2::new(100.0, 12.0)));
                        }
                    
                        if ui.button(">").clicked() {
                            self.palette_index = (self.palette_index + 1) % 27; // 13 palettes maintenant (0-12)
                            self.params.color.color_mode = self.palette_index;
                            if !self.iterations.is_empty() {
                                self.update_texture(ctx);
                            }
                        }
                    });
                    
                    ui.separator();

                    let old_out_mode = self.out_coloring_mode;
                    wrap_group(ui, "bar_group_3329", |ui| {
                        ui.label("Outcoloring:");
                        egui::ComboBox::from_id_salt("outcoloring_mode")
                            .selected_text(self.out_coloring_mode.name())
                            .show_ui(ui, |ui| {
                                for mode in OutColoringMode::menu_modes() {
                                    ui.selectable_value(&mut self.out_coloring_mode, *mode, mode.name());
                                }
                            });
                    });
                    if old_out_mode != self.out_coloring_mode {
                        self.params.color.out_coloring_mode = self.out_coloring_mode;
                        // Distance/OrbitTraps/Wings modes require data computed during render
                        let needs_rerender = matches!(self.out_coloring_mode,
                            OutColoringMode::Distance | OutColoringMode::DistanceAO | OutColoringMode::Distance3D
                        ) && self.distances.is_empty()
                        || matches!(self.out_coloring_mode,
                            OutColoringMode::OrbitTraps | OutColoringMode::Wings
                        ) && self.orbits.iter().all(|o| o.is_none());
                        if needs_rerender {
                            self.start_render();
                        } else if !self.iterations.is_empty() {
                            self.update_texture(ctx);
                        }
                    }

                    ui.separator();

                    wrap_group(ui, "bar_group_3357", |ui| {
                        ui.label("Color Repeat:");
                        let is_density_type = matches!(
                            self.selected_type,
                            FractalType::Buddhabrot | FractalType::Nebulabrot | FractalType::AntiBuddhabrot
                        );
                        let (min_repeat, max_repeat) = if is_density_type { (1, 8) } else { (1, 120) };
                        if is_density_type && self.color_repeat > max_repeat {
                            self.color_repeat = max_repeat;
                            self.params.color.color_repeat = self.color_repeat;
                        }
                        let old_repeat = self.color_repeat;
                        ui.add(egui::Slider::new(&mut self.color_repeat, min_repeat..=max_repeat));
                        if old_repeat != self.color_repeat {
                            self.params.color.color_repeat = self.color_repeat;
                            if !self.iterations.is_empty() {
                                self.update_texture(ctx);
                            }
                        }
                    });

                    ui.separator();

                    // Anti-aliasing : nombre d'échantillons sous-pixel jitterés
                    // moyennés après les passes progressives (CPU uniquement).
                    wrap_group(ui, "bar_group_3382", |ui| {
                        ui.label("AA:");
                        let prev_aa = self.aa_samples;
                        egui::ComboBox::from_id_salt("aa_samples")
                            .width(46.0)
                            .selected_text(if self.aa_samples <= 1 {
                                "Off".to_string()
                            } else {
                                format!("{}×", self.aa_samples)
                            })
                            .show_ui(ui, |ui| {
                                for &n in &[1u32, 2, 4, 8, 16, 32] {
                                    let label = if n == 1 { "Off".to_string() } else { format!("{n}×") };
                                    ui.selectable_value(&mut self.aa_samples, n, label);
                                }
                            })
                            .response
                            .on_hover_text(
                                "Anti-aliasing: averaged jittered sub-pixel samples (CPU). Off = disabled.",
                            );
                        if self.aa_samples != prev_aa {
                            self.start_render();
                        }

                        let prev_jitter_scale = self.aa_jitter_scale;
                        ui.add_enabled(
                            self.aa_samples > 1,
                            egui::Slider::new(&mut self.aa_jitter_scale, 0.0..=1.0)
                                .text("filter")
                                .fixed_decimals(2),
                        )
                        .on_hover_text(
                            "AA jitter amplitude in pixels: 0 = pixel center, 1 = full tent filter.",
                        );
                        if self.aa_jitter_scale != prev_jitter_scale {
                            self.start_render();
                        }
                    });


                });
        });
        sticky_bar_record(&mut self.bar_heights[0], top_response.response.rect.height());

        // Traiter les messages du rendu HQ en cours
        if let Some(ref receiver) = self.hq_render_receiver {
            while let Ok(msg) = receiver.try_recv() {
                self.hq_render_state.apply(msg);
            }
        }

        // Fenêtre de configuration du rendu haute résolution
        if self.show_render_dialog {
            let (window_width, window_height) = self.view.dimensions();
            egui::Window::new("🎬 High Quality Render")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    if self.hq_render_state.is_running() {
                        // Afficher la barre de progression pendant le rendu
                        let resolution_label = match self.render_resolution_preset {
                            RenderResolutionPreset::Window => {
                                format!("{}×{}", window_width, window_height)
                            }
                            RenderResolutionPreset::Res4K => "4K".to_string(),
                            RenderResolutionPreset::Res8K => "8K".to_string(),
                        };
                        ui.heading(format!("Rendering with {} precision...", resolution_label));
                        ui.add_space(10.0);

                        let progress_bar = egui::ProgressBar::new(self.hq_render_state.progress())
                            .show_percentage()
                            .animate(true);
                        ui.add(progress_bar);

                        ui.add_space(10.0);
                        ui.label("Please wait...");

                        // Demander un repaint pour mettre à jour la barre
                        ctx.request_repaint();
                    } else if let Some(result) = self.hq_render_state.result() {
                        // Afficher le résultat
                        ui.heading("Complete");
                        ui.add_space(10.0);
                        match result {
                            HqRenderResult::Saved(filename) => {
                                ui.label(format!("✓ Saved: {filename}"));
                            }
                            HqRenderResult::Error(error) => {
                                ui.label(format!("✗ {error}"));
                            }
                        }
                        ui.add_space(20.0);

                        if ui.button("Close").clicked() {
                            self.show_render_dialog = false;
                            self.hq_render_state.clear();
                            self.hq_render_receiver = None;
                        }
                    } else {
                        // Afficher les options de résolution
                        ui.heading("Resolution");
                        ui.add_space(10.0);

                        // Presets de résolution
                        ui.horizontal(|ui| {
                            ui.selectable_value(
                                &mut self.render_resolution_preset,
                                RenderResolutionPreset::Window,
                                format!("Window ({}×{})", window_width, window_height),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.selectable_value(
                                &mut self.render_resolution_preset,
                                RenderResolutionPreset::Res4K,
                                "4K (3840×2160)",
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.selectable_value(
                                &mut self.render_resolution_preset,
                                RenderResolutionPreset::Res8K,
                                "8K (7680×4320)",
                            );
                        });

                        ui.add_space(20.0);
                        ui.separator();
                        ui.add_space(10.0);

                        // Boutons d'action
                        ui.horizontal(|ui| {
                            if ui.button("Cancel").clicked() {
                                self.show_render_dialog = false;
                            }

                            ui.add_space(20.0);

                            if ui.button("✓ Render & Save").clicked() {
                                self.render_high_quality();
                            }
                        });
                    }
                });
        }

        // Barre d'état en bas — AVANT le CentralPanel : depuis egui 0.37 les
        // panneaux s'affichent dans le Ui racine et un panneau ajouté après le
        // CentralPanel ne reçoit plus d'espace.
        // Barre de PROGRESSION, tout en bas, sur une seule ligne : temps à
        // gauche, puis l'état du rendu et la barre qui prend le reste. Elle
        // garde la même hauteur que le rendu tourne ou non : la barre d'état
        // au-dessus ne change donc plus de forme quand la progression
        // apparaît ou disparaît.
        //
        // En navigation XaoS les rendus s'enchaînent à ~30 Hz : on affiche un
        // état STABLE pendant tout le geste (sinon la barre animée et le temps
        // clignoteraient à cette cadence). Le rendu de stabilisation retrouve
        // la barre normale.
        let progress_min = sticky_bar_min(&mut self.bar_heights[2], window_width);
        let progress_response = egui::Panel::bottom("progress").min_size(progress_min).show(ui, |ui| {
            ui.horizontal(|ui| {
                let format_secs = |secs: f64| {
                    if secs < 1.0 {
                        format!("{:.0} ms", secs * 1000.0)
                    } else if secs < 60.0 {
                        format!("{:.2} s", secs)
                    } else {
                        format!("{}m {:.1}s", (secs / 60.0).floor() as u32, secs % 60.0)
                    }
                };
                let nav_active = self.continuous_nav() && self.nav.is_active();
                if self.rendering && !nav_active {
                    let elapsed = self
                        .render_start_time
                        .map(|t| t.elapsed().as_secs_f64())
                        .unwrap_or(0.0);
                    ui.label(format!("⏱ {}…", format_secs(elapsed)));
                } else if let Some(render_time) = self.last_render_time {
                    ui.label(format!("⏱ {}", format_secs(render_time)));
                }
                ui.separator();
                if nav_active {
                    ui.label("Navigation XaoS");
                } else if self.rendering {
                    let (current_pass, total_passes) = self.render_progress.passes();
                    let aa_suffix = match self.render_progress.aa() {
                        Some((s, t)) if t > 1 && s > 0 => format!(" - AA {s}/{t}"),
                        _ => String::new(),
                    };
                    ui.label(format!(
                        "Computing… pass {current_pass}/{total_passes}{aa_suffix}"
                    ));
                    ui.add(
                        egui::ProgressBar::new(self.render_progress.fraction())
                            .desired_width(ui.available_width())
                            .show_percentage()
                            .animate(true),
                    );
                } else if self.is_preview {
                    ui.label("Preview");
                } else {
                    ui.label("Ready");
                }
            });
        });
        sticky_bar_record(&mut self.bar_heights[2], progress_response.response.rect.height());

        let bottom_min = sticky_bar_min(&mut self.bar_heights[1], window_width);
        let bottom_response = egui::Panel::bottom("status").min_size(bottom_min).show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                // Vue exportée (CLI + coordonnées) ; bouton CLI tout à gauche.
                let export_params = self.export_params();
                let cli = crate::gui::cli_command::build_cli_command(
                    &export_params,
                    crate::gui::cli_command::CliExtras {
                        aa_samples: self.aa_samples.max(1),
                        aa_jitter_scale: crate::gui::render_state::normalized_jitter_scale(
                            self.aa_jitter_scale,
                        ),
                        exact_center: false,
                    },
                );
                let mut cli_hover = String::from(
                    "Copy a fractall-cli command that renders this view.\n\
                     Width/height are the window size: edit --width/--height \
                     (e.g. 3840 and 2160) for a larger render.\n\
                     --zoom fixes the horizontal extent; another aspect ratio \
                     shows more or less vertically.",
                );
                if !cli.unexpressed.is_empty() {
                    cli_hover.push_str("\n\nNot expressible on the command line: ");
                    cli_hover.push_str(&cli.unexpressed.join("; "));
                }
                if ui.small_button("📋 CLI").on_hover_text(cli_hover).clicked() {
                    ctx.copy_text(cli.command);
                    self.clipboard_feedback = Some(("CLI command copied", Instant::now()));
                }
                ui.separator();
                // Avis du sage (bascule de navigation, itérations relevées…) :
                // bien visible à gauche, quelques secondes.
                if let Some((text, left)) = self.sage_notice.visible(Instant::now()) {
                    ui.label(
                        egui::RichText::new(format!(" {text} "))
                            .strong()
                            .color(egui::Color32::BLACK)
                            .background_color(egui::Color32::from_rgb(255, 205, 80)),
                    );
                    ctx.request_repaint_after(left);
                    ui.separator();
                }
                // Centre (compact ; clic = copie pleine précision), itérations, zoom
                wrap_group(ui, "bar_group_3547", |ui| {
                    let (full_x, full_y) = crate::gui::coords::center_strings(&export_params);
                    let center_response = ui
                        .add(
                            egui::Label::new(crate::gui::coords::format_center_status(
                                &export_params,
                            ))
                            .sense(egui::Sense::click()),
                        )
                        .on_hover_text(format!(
                            "Click to copy full-precision coordinates (as a TOML view \
                             for fractall-cli --toml).\nre = {}\nim = {}",
                            crate::gui::coords::exact_coordinate(&full_x),
                            crate::gui::coords::exact_coordinate(&full_y),
                        ));
                    if center_response.clicked() {
                        ctx.copy_text(crate::gui::coords::full_precision_clipboard_text(
                            &export_params,
                        ));
                        self.clipboard_feedback = Some(("Coordinates copied", Instant::now()));
                    }
                    if let Some((message, at)) = self.clipboard_feedback {
                        let shown_for = at.elapsed();
                        if shown_for < CLIPBOARD_FEEDBACK_DURATION {
                            ui.label(egui::RichText::new(format!("✔ {message}")).weak());
                            ctx.request_repaint_after(CLIPBOARD_FEEDBACK_DURATION - shown_for);
                        } else {
                            self.clipboard_feedback = None;
                        }
                    }
                });
                ui.separator();
                if self.iter_auto_active() {
                    ui.label(format!("Iter.: {} (auto)", self.params.iteration_max));
                } else {
                    ui.label(format!("Iter.: {}", self.params.iteration_max));
                }
                ui.separator();
                let zoom_text = crate::gui::coords::zoom_label(&self.params);
                ui.label(zoom_text);
                ui.separator();

                // ═══════════════════════════════════════════════════════════════
                // Affichage du mode de calcul effectif
                // Format: [Device] [Precision] [Algorithme]
                // Ex: "GPU f32 Perturbation" ou "CPU f64 Standard"
                // ═══════════════════════════════════════════════════════════════

                let effective_mode = self.effective_algorithm_mode();
                let gpu_active = self.use_gpu && self.gpu_renderer.is_some();

                let mode_display = if let (Some(device_label), Some(method_label)) = (
                    &self.last_render_device_label,
                    &self.last_render_method_label,
                ) {
                    if method_label.is_empty() {
                        device_label.clone()
                    } else {
                        format!("{} {}", device_label, method_label)
                    }
                } else {
                    let device = if gpu_active { "GPU" } else { "CPU" };

                    let (precision, algo) = match (gpu_active, effective_mode) {
                        // GPU modes
                        (true, AlgorithmMode::Perturbation) => ("f32", "Perturbation"),
                        (true, AlgorithmMode::StandardF64) => ("f32", "Standard"),
                        (true, _) => ("f32", "Standard"), // Fallback GPU

                        // CPU modes
                        (false, AlgorithmMode::ReferenceGmp) => {
                            // GMP avec bits affichés séparément plus bas
                            ("GMP", "")
                        }
                        (false, AlgorithmMode::Perturbation) => ("f64", "Perturbation"),
                        (false, AlgorithmMode::StandardF64) => ("f64", "Standard"),
                        (false, AlgorithmMode::Auto) => ("f64", "Standard"),
                    };

                    // Cas spécial pour GMP: afficher les bits (précision effective selon le zoom)
                    if effective_mode == AlgorithmMode::ReferenceGmp && !gpu_active {
                        let effective_prec =
                            crate::fractal::perturbation::compute_perturbation_precision_bits(
                                &self.params,
                            );
                        format!("{} GMP {}b", device, effective_prec)
                    } else {
                        format!("{} {} {}", device, precision, algo)
                    }
                };

                ui.label(mode_display);

                // Afficher la précision GMP effective pour le mode perturbation
                if effective_mode == AlgorithmMode::Perturbation {
                    let effective_prec =
                        crate::fractal::perturbation::compute_perturbation_precision_bits(
                            &self.params,
                        );
                    ui.separator();
                    ui.label(format!("GMP: {}b", effective_prec));
                }
            });
        });
        sticky_bar_record(&mut self.bar_heights[1], bottom_response.response.rect.height());

        // Interface principale - zone d'affichage de la fractale
        // Désactiver la sélection de texte pour éviter que le curseur devienne un curseur de texte
        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(egui::Color32::TRANSPARENT))
            .show(ui, |ui| {
                // Désactiver la sélection de texte dans cette zone
                ui.style_mut().interaction.selectable_labels = false;
                ui.vertical_centered(|ui| {
                    let available_size = ui.available_size();
                    let target_width = available_size.x.max(1.0).floor() as u32;
                    let target_height = available_size.y.max(1.0).floor() as u32;
                    self.queue_resize(target_width, target_height);

                    // Afficher la texture si elle existe (même pendant le rendu progressif)
                    if let Some(texture) = &self.texture {
                        // Utiliser la taille réelle de la texture pour préserver le ratio et éviter toute déformation
                        let image_size = texture.size_vec2();

                        // Ajuster pour tenir dans l'espace disponible, sans jamais agrandir au-delà de la résolution (évite le pixelisé)
                        let scale = (available_size.x / image_size.x)
                            .min(available_size.y / image_size.y)
                            .min(1.0);
                        let display_size = image_size * scale;

                        // Gestion des interactions sur l'image
                        // Désactiver la sélection de texte pour forcer le curseur à rester une flèche.
                        // On alloue le rect d'interaction et on peint la texture à la main (au lieu du
                        // widget Image) pour pouvoir la WARPER (G10.1) pendant un rendu : la texture
                        // affichée est encore l'ancienne vue, on la transforme vers la vue live.
                        let (image_rect, response) =
                            ui.allocate_exact_size(display_size, egui::Sense::click_and_drag());
                        let uv =
                            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                        // G10.1 : la texture affichée représente `texture_view` ; dès
                        // qu'elle diffère de la vue live, on la WARPE — que le rendu
                        // soit en cours ou non. Gater sur `self.rendering` repeignait
                        // l'image NON transformée sur les frames sans rendu actif
                        // (juste après la fin d'un rendu, avant le suivant) : en
                        // navigation continue la vue live a déjà avancé → saut en
                        // arrière d'un rendu complet, puis saut en avant à la frame
                        // suivante = saccade à la cadence de rendu. `compute_warp_rect`
                        // renvoie déjà `None` quand les vues sont identiques (statique
                        // inchangé, affichage bit-identique).
                        let warp = self.compute_warp_rect(image_rect);
                        match warp {
                            // Warp actif : le zoom-in fait déborder l'ancienne image → clip au rect.
                            Some(r) => ui.painter().with_clip_rect(image_rect).image(
                                texture.id(),
                                r,
                                uv,
                                egui::Color32::WHITE,
                            ),
                            None => ui.painter().image(
                                texture.id(),
                                image_rect,
                                uv,
                                egui::Color32::WHITE,
                            ),
                        };

                        // Overlay partiel de navigation PAR-DESSUS : ses pixels déjà
                        // calculés (alpha 255) recouvrent le warp, les autres le
                        // laissent transparaître — c'est le remplissage, équivalent
                        // de la duplication de lignes de XaoS (« reducing
                        // resolution »). Warpé lui aussi : la vue a pu avancer
                        // depuis sa publication.
                        if let Some(overlay) = &self.nav_overlay_texture {
                            let rect = self
                                .nav_overlay_view
                                .as_ref()
                                .and_then(|v| self.warp_rect_for(v, image_rect))
                                .unwrap_or(image_rect);
                            ui.painter().with_clip_rect(image_rect).image(
                                overlay.id(),
                                rect,
                                uv,
                                egui::Color32::WHITE,
                            );
                        }

                        // Forcer le curseur à être une flèche quand on survole l'image
                        // IMPORTANT: Toujours afficher une flèche (Default) et non un curseur de texte ou autre
                        if response.hovered() {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::Default);

                            // G10.5 : mémoriser la position normalisée du curseur —
                            // point de priorité de la file de tuiles au prochain rendu.
                            if let Some(pos) = ui.ctx().pointer_hover_pos() {
                                if image_rect.contains(pos) {
                                    let local = pos - image_rect.min;
                                    self.hover_norm = Some((
                                        local.x / image_rect.width(),
                                        local.y / image_rect.height(),
                                    ));
                                }
                            }

                            // Track mouse position for Julia preview (on Mandelbrot-like types)
                            if self.selected_type.has_julia_variant()
                                && self.julia_preview_enabled
                                && !self.selection.is_active()
                            {
                                if let Some(pos) = ui.ctx().pointer_hover_pos() {
                                    if image_rect.contains(pos) {
                                        let local_pos = pos - image_rect.min;
                                        let pixel_x = (local_pos.x / image_rect.width())
                                            * self.params.width as f32;
                                        let pixel_y = (local_pos.y / image_rect.height())
                                            * self.params.height as f32;
                                        let seed = self.pixel_to_complex(
                                            pixel_x,
                                            pixel_y,
                                            self.params.width as f32,
                                            self.params.height as f32,
                                        );
                                        self.request_julia_preview(seed);
                                    }
                                }
                            }
                        }

                        // Détecter le début d'une sélection rectangulaire (drag avec bouton gauche)
                        // Désactivé en mode Julia preview
                        let julia_mode =
                            self.selected_type.has_julia_variant() && self.julia_preview_enabled;

                        // Molette : zoom continu ANCRÉ au curseur (le point sous le
                        // curseur reste fixe). Les petits incréments maximisent la
                        // réutilisation XaoS inter-frame (écho immédiat, raffinement
                        // exact à l'idle) — c'est le chemin de zoom fluide.
                        // Bouton du milieu maintenu = pan, jamais zoom : appuyer sur
                        // une molette la fait souvent tourner d'un cran.
                        let middle_held = ctx.input(|i| i.pointer.middle_down());
                        if !julia_mode
                            && response.hovered()
                            && !self.selection.is_active()
                            && !middle_held
                            && !self.middle_pan_active
                        {
                            let scroll_y = ctx.input(|i| i.smooth_scroll_delta.y);
                            if scroll_y.abs() > 0.5 {
                                if let Some(pos) = ctx.pointer_hover_pos() {
                                    if image_rect.contains(pos) {
                                        let local = pos - image_rect.min;
                                        let rx = (local.x / image_rect.width()) as f64;
                                        let ry = (local.y / image_rect.height()) as f64;
                                        // ≈ ×1.2 par cran (50 px), borné (trackpads rapides).
                                        let factor =
                                            1.2f64.powf((scroll_y as f64 / 50.0).clamp(-3.0, 3.0));
                                        self.zoom_anchored_hp(rx, ry, factor);
                                        self.start_render();
                                    }
                                }
                            }
                        }

                        // Navigation XaoS : bouton MAINTENU = zoom continu ancré au
                        // curseur (gauche = in, droit = out), cadence indépendante du
                        // framerate. Le clic ×2 et la sélection rectangle sont
                        // désactivés dans ce mode.
                        //
                        // Découplage géométrie / rendu (clé de la fluidité) : la vue
                        // avance à CHAQUE frame (60 Hz), le rendu ne redémarre que
                        // quand le précédent est fini ET qu'au moins
                        // XAOS_NAV_MIN_RENDER_INTERVAL s'est écoulé. Entre deux, le
                        // warp G10.1 transporte la texture vers la vue live → mouvement
                        // continu ; la réutilisation pixels G10.4 rend chaque rendu
                        // quasi gratuit (petit incrément).
                        if self.continuous_nav() && !julia_mode {
                            let (dir, pointer_pos) = ctx.input(|i| {
                                let d = if i.pointer.primary_down() {
                                    1.0
                                } else if i.pointer.secondary_down() {
                                    -1.0
                                } else {
                                    0.0
                                };
                                // Le geste ne compte que s'il a COMMENCÉ sur l'image :
                                // `primary_down()` est global, donc sans ce garde un
                                // appui dans la barre de menus (ou tout autre panneau)
                                // pilotait le zoom via le dernier ancrage mémorisé.
                                // `press_origin` = position de l'appui en cours, tous
                                // boutons confondus ; `None` au relâchement, ce qui
                                // laisse la décélération se poursuivre normalement.
                                let started_on_image = i
                                    .pointer
                                    .press_origin()
                                    .is_some_and(|p| image_rect.contains(p));
                                let d = if started_on_image { d } else { 0.0 };
                                (d, i.pointer.interact_pos())
                            });
                            // Ancrage : position courante si le curseur est sur
                            // l'image ; `None` conserve le dernier ancrage connu
                            // (le zoom reste ancré pendant la décélération, même
                            // curseur sorti de l'image).
                            let anchor = pointer_pos.filter(|p| image_rect.contains(*p)).map(|p| {
                                let local = p - image_rect.min;
                                (local.x / image_rect.width(), local.y / image_rect.height())
                            });

                            let now = Instant::now();
                            let dt = self
                                .xaos_nav_last_tick
                                .map(|t| now.duration_since(t).as_secs_f64())
                                .unwrap_or(0.0);

                            // Toute la décision (rampe de vitesse, ancrage, cadence
                            // de rendu, stabilisation) vit dans `gui::nav`, testée
                            // sans fenêtre. Ici on ne fait qu'EXÉCUTER la sortie.
                            let outcome = self.nav.tick(nav::NavInput {
                                dir,
                                dt,
                                anchor,
                                cruise: nav::cruise_rate(self.last_render_time),
                                rendering: self.rendering,
                            });
                            self.xaos_nav_last_tick = self.nav.is_active().then_some(now);

                            if let Some((rx, ry, factor)) = outcome.zoom {
                                self.zoom_anchored_hp(rx as f64, ry as f64, factor);
                            }
                            match outcome.render {
                                nav::NavRender::Moving => {
                                    self.xaos_nav_dragging = true;
                                    self.start_render();
                                }
                                nav::NavRender::Settle => {
                                    // `dragging` reste vrai pour la config de rendu
                                    // (passe unique + streaming partiel, pas de
                                    // retour au flou d'une passe 1/16) ; `settle`
                                    // force la PLEINE résolution, la résolution
                                    // dynamique ne servant que pendant le mouvement.
                                    self.xaos_nav_dragging = true;
                                    self.xaos_nav_settle = true;
                                    self.start_render();
                                    self.xaos_nav_settle = false;
                                    self.xaos_nav_dragging = false;
                                }
                                nav::NavRender::None => {}
                            }
                            if outcome.repaint {
                                ctx.request_repaint();
                            }
                        }

                        // Bouton du MILIEU maintenu : déplacement latéral (pan) sans
                        // zoom, dans les deux modes de navigation. La vue suit la
                        // souris à chaque frame (le warp G10.1 aligne l'ancienne
                        // texture) ; un rendu de navigation repart dès que le
                        // précédent est fini, puis rendu pleine résolution au
                        // relâchement.
                        if !julia_mode {
                            let (middle_down, delta, started_on_image) = ctx.input(|i| {
                                (
                                    i.pointer.middle_down(),
                                    i.pointer.delta(),
                                    i.pointer
                                        .press_origin()
                                        .is_some_and(|p| image_rect.contains(p)),
                                )
                            });
                            if middle_down && (self.middle_pan_active || started_on_image) {
                                self.middle_pan_active = true;
                                if delta != egui::Vec2::ZERO
                                    && image_rect.width() > 0.0
                                    && image_rect.height() > 0.0
                                {
                                    let dx = -(delta.x / image_rect.width()) as f64;
                                    let dy = -(delta.y / image_rect.height()) as f64;
                                    self.edit_view_hp(|view| view.pan_by(dx, dy));
                                }
                                if !self.rendering {
                                    self.xaos_nav_dragging = true;
                                    self.start_render();
                                    self.xaos_nav_dragging = false;
                                }
                                ctx.request_repaint();
                            } else if self.middle_pan_active {
                                self.middle_pan_active = false;
                                self.xaos_nav_dragging = true;
                                self.xaos_nav_settle = true;
                                self.start_render();
                                self.xaos_nav_settle = false;
                                self.xaos_nav_dragging = false;
                            }
                        }

                        if !julia_mode && !self.continuous_nav() {
                            ctx.input(|i| {
                                // Vérifier si le bouton gauche est pressé et si on est dans la zone de l'image
                                if i.pointer.primary_down() {
                                    if let Some(pointer_pos) = i.pointer.interact_pos() {
                                        if image_rect.contains(pointer_pos) {
                                            if !self.selection.is_active() {
                                                // Commencer une nouvelle sélection
                                                self.selection.begin(pointer_pos);
                                            } else {
                                                // Mettre à jour la position actuelle
                                                let clamped_pos = egui::Pos2::new(
                                                    pointer_pos
                                                        .x
                                                        .max(image_rect.min.x)
                                                        .min(image_rect.max.x),
                                                    pointer_pos
                                                        .y
                                                        .max(image_rect.min.y)
                                                        .min(image_rect.max.y),
                                                );
                                                self.selection.update(clamped_pos);
                                            }
                                        }
                                    }
                                } else if self.selection.is_active() && i.pointer.primary_released()
                                {
                                    // Le bouton a été relâché, terminer la sélection
                                    if let Some(rect) = self.selection.finish() {
                                        // Vérifier que le rectangle est dans l'image et a une taille minimale
                                        if image_rect.contains(rect.min)
                                            && image_rect.contains(rect.max)
                                        {
                                            if rect.width() > 5.0 && rect.height() > 5.0 {
                                                self.zoom_to_rectangle(
                                                    rect.min, rect.max, image_rect,
                                                );
                                            }
                                        }
                                    }
                                }
                            });
                        }

                        // Dessiner le rectangle de sélection par-dessus l'image
                        if let Some(rect) = self.selection.rect() {
                            // Fond semi-transparent pour mieux voir la sélection
                            ui.painter().rect_filled(
                                rect,
                                0.0,
                                egui::Color32::from_rgba_unmultiplied(255, 255, 0, 30), // Jaune très transparent
                            );

                            // Rectangle extérieur jaune épais
                            ui.painter().rect_stroke(
                                rect,
                                0.0,
                                egui::Stroke::new(3.0_f32, egui::Color32::from_rgb(255, 255, 0)),
                                egui::StrokeKind::Middle,
                            );
                            // Rectangle intérieur pour meilleure visibilité
                            ui.painter().rect_stroke(
                                rect.expand(-1.0),
                                0.0,
                                egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(0, 0, 0)), // Bordure noire intérieure
                                egui::StrokeKind::Middle,
                            );

                            // Demander un re-rendu pour mettre à jour le rectangle en temps réel
                            ctx.request_repaint();
                        }

                        // Clic simple (sans drag) : zoom au point
                        // Seulement si on n'a pas fait de sélection et pas en mode Julia
                        if response.clicked()
                            && !self.selection.is_active()
                            && !julia_mode
                            && !self.continuous_nav()
                        {
                            if let Some(pos) = response.interact_pointer_pos() {
                                let local_pos = pos - image_rect.min;
                                let pixel_x =
                                    (local_pos.x / image_rect.width()) * self.params.width as f32;
                                let pixel_y =
                                    (local_pos.y / image_rect.height()) * self.params.height as f32;
                                let (rx, ry) = self.pixel_to_ratio(pixel_x, pixel_y);
                                self.zoom_at_ratio(rx, ry, 2.0);
                            }
                        }

                        // Clic droit : dézoom centré sur la position de la souris
                        // (désactivé en mode Julia et en navigation XaoS — là le
                        // bouton droit maintenu fait le dézoom continu)
                        if !julia_mode && !self.continuous_nav() {
                            if response.secondary_clicked() {
                                if let Some(pos) = response.interact_pointer_pos() {
                                    let local_pos = pos - image_rect.min;
                                    let pixel_x = (local_pos.x / image_rect.width())
                                        * self.params.width as f32;
                                    let pixel_y = (local_pos.y / image_rect.height())
                                        * self.params.height as f32;
                                    let (rx, ry) = self.pixel_to_ratio(pixel_x, pixel_y);
                                    self.zoom_out_at_ratio(rx, ry, 2.0);
                                } else {
                                    // Fallback si pas de position : dézoom au centre
                                    self.zoom_out(2.0);
                                }
                            } else {
                                ctx.input(|i| {
                                    if i.pointer.secondary_clicked() {
                                        if let Some(pos) = i.pointer.interact_pos() {
                                            if image_rect.contains(pos) {
                                                let local_pos = pos - image_rect.min;
                                                let pixel_x = (local_pos.x / image_rect.width())
                                                    * self.params.width as f32;
                                                let pixel_y = (local_pos.y / image_rect.height())
                                                    * self.params.height as f32;
                                                let (rx, ry) =
                                                    self.pixel_to_ratio(pixel_x, pixel_y);
                                                self.zoom_out_at_ratio(rx, ry, 2.0);
                                            }
                                        }
                                    }
                                });
                            }
                        }

                        // Overlay Julia preview (en mode Julia sur Mandelbrot)
                        if julia_mode {
                            // Taille et position de l'overlay (coin supérieur droit)
                            let preview_width = 200.0;
                            let preview_height = 150.0;
                            let margin = 10.0;
                            let overlay_pos = egui::Pos2::new(
                                image_rect.max.x - preview_width - margin,
                                image_rect.min.y + margin,
                            );
                            let overlay_rect = egui::Rect::from_min_size(
                                overlay_pos,
                                egui::Vec2::new(preview_width, preview_height),
                            );

                            // Fond semi-transparent
                            ui.painter().rect_filled(
                                overlay_rect,
                                4.0,
                                egui::Color32::from_rgba_unmultiplied(0, 0, 0, 180),
                            );

                            // Bordure
                            ui.painter().rect_stroke(
                                overlay_rect,
                                4.0,
                                egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(100, 100, 100)),
                                egui::StrokeKind::Middle,
                            );

                            // Afficher la preview Julia ou un spinner
                            if let Some(ref texture) = self.julia_preview_texture {
                                let tex_rect = egui::Rect::from_min_size(
                                    egui::Pos2::new(overlay_pos.x + 20.0, overlay_pos.y + 5.0),
                                    egui::Vec2::new(160.0, 120.0),
                                );
                                ui.painter().image(
                                    texture.id(),
                                    tex_rect,
                                    egui::Rect::from_min_max(
                                        egui::pos2(0.0, 0.0),
                                        egui::pos2(1.0, 1.0),
                                    ),
                                    egui::Color32::WHITE,
                                );
                            } else if self.julia_preview_rendering {
                                // Indicateur de chargement (simple texte)
                                ui.painter().text(
                                    overlay_rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    "Loading...",
                                    egui::FontId::default(),
                                    egui::Color32::WHITE,
                                );
                            } else {
                                ui.painter().text(
                                    overlay_rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    "Move mouse",
                                    egui::FontId::default(),
                                    egui::Color32::GRAY,
                                );
                            }

                            // Afficher les coordonnées du seed et l'aide
                            if let Some(seed) = self.julia_preview_last_seed {
                                let text = format!("c = {:.4} + {:.4}i", seed.re, seed.im);
                                ui.painter().text(
                                    egui::Pos2::new(
                                        overlay_pos.x + preview_width / 2.0,
                                        overlay_pos.y + preview_height - 20.0,
                                    ),
                                    egui::Align2::CENTER_CENTER,
                                    text,
                                    egui::FontId::proportional(11.0),
                                    egui::Color32::WHITE,
                                );
                            }
                            ui.painter().text(
                                egui::Pos2::new(
                                    overlay_pos.x + preview_width / 2.0,
                                    overlay_pos.y + preview_height - 8.0,
                                ),
                                egui::Align2::CENTER_CENTER,
                                "Press J to switch",
                                egui::FontId::proportional(10.0),
                                egui::Color32::GRAY,
                            );
                        }
                    } else if self.rendering {
                        // Pas encore de texture, afficher le spinner
                        ui.spinner();
                        ui.label("Rendering...");
                    } else {
                        // Premier rendu
                        self.start_render();
                    }
                });
            });


        // Historique : une vue n'est empilée qu'une fois le geste terminé
        // et la vue stable (`view_history::SETTLE`) ; jamais pendant le mode
        // preview Julia.
        let julia_mode = self.selected_type.has_julia_variant() && self.julia_preview_enabled;
        if !julia_mode {
            let gesture = ctx.input(|i| i.pointer.any_down())
                || self.selection.is_active()
                || self.nav.is_active()
                || self.xaos_nav_dragging;
            let now = Instant::now();
            self.history.observe(&self.params, now, gesture);
            if let Some(wait) = self.history.time_to_settle(now) {
                ctx.request_repaint_after(wait + Duration::from_millis(10));
            }
        }
        // Taille de la zone egui en points logiques (même unité que
        // `with_inner_size`). ⚠️ PAS `viewport().inner_rect` : sous Wayland
        // winit ne connaît pas la position de la fenêtre et le laisse à None
        // — la taille n'était jamais sauvegardée.
        let rect = ctx.viewport_rect();
        if rect.width() > 0.0 && rect.height() > 0.0 {
            self.window_size = Some([rect.width(), rect.height()]);
        }

        // Demander un re-rendu seulement si un rendu est en cours
        if self.rendering || self.hq_render_state.is_running() {
            ctx.request_repaint();
        }
    }

    /// Fermeture : la vue courante et l'historique sont sauvegardés pour la
    /// reprise au prochain démarrage (`fractall-gui --fresh` les ignore).
    fn on_exit(&mut self) {
        let Some(path) = self.session_path.clone() else {
            return;
        };
        let session = self.history.to_session(self.iter_auto, self.window_size);
        if let Err(e) = view_history::save_session(&path, &session) {
            eprintln!("Could not save the GUI session to {}: {e}", path.display());
        }
    }
}

#[cfg(test)]
mod screenshot_tests {
    use super::*;

    #[test]
    fn curve_aa_progress_matches_independent_full_resolution_samples() {
        let ctx = Context::default();
        let mut app = FractallApp::new(&eframe::CreationContext::_new_kittest(ctx));
        app.aa_samples = 3;
        app.aa_jitter_scale = 0.75;
        for (kind, offset) in [
            (FractalType::VonKoch, [0.0, 0.0]),
            (FractalType::Dragon, [0.375, -0.25]),
        ] {
            let mut p = default_params_for_type(kind, 59, 37);
            p.iteration_max = 6;
            p.sampling.aa_subpixel_offset = offset;
            app.params = p.clone();
            app.start_render();
            let rx = app.render_receiver.take().unwrap();
            let mut reference = crate::io::png::AaAccumulator::new(&p, false);
            let mut received = 0;
            loop {
                match rx.recv_timeout(Duration::from_secs(5)).unwrap() {
                    RenderMessage::AaProgress { display_buffer, width, height, sample, total } => {
                        received += 1;
                        assert_eq!((width, height, sample, total), (59, 37, received, 3));
                        p.sampling.aa_jitter = Some((u64::from(sample - 1), 0.75));
                        let (it, z) = if kind == FractalType::VonKoch {
                            crate::fractal::render_von_koch(&p)
                        } else {
                            crate::fractal::render_dragon(&p)
                        };
                        reference.add(&p, &crate::render::RenderOutput::without_extras(it, z)).unwrap();
                        assert_eq!(display_buffer, reference.average());
                    }
                    RenderMessage::AllComplete { .. } => break,
                    RenderMessage::Cancelled => panic!("unexpected cancellation"),
                    _ => {}
                }
            }
            assert_eq!(received, 3);
        }
    }

    #[test]
    fn aa_pixels_and_geometry_survive_late_passes_after_completion() {
        let ctx = Context::default();
        let mut app = FractallApp::new(&eframe::CreationContext::_new_kittest(ctx.clone()));
        let mut params = default_params_for_type(FractalType::Mandelbrot, 3, 2);
        params.center_x_hp = Some("-0.12345678901234567890123456789".into());
        params.sampling.jitter_scale = 0.25;
        app.params = params.clone();
        app.rendering_params = params.clone();
        app.rendering_view = Some(app.current_view_snapshot());
        app.rendering = true;
        app.render_progress.begin(2, 4);
        let (tx, rx) = mpsc::channel();
        let (tex_tx, tex_rx) = mpsc::channel();
        app.render_receiver = Some(rx);
        app.texture_ready_receiver = Some(tex_rx);
        tx.send(RenderMessage::TileProgress {
            display_buffer: vec![1; 18], width: 3, height: 2,
        }).unwrap();
        app.check_render_complete(&ctx);
        assert!(app.displayed_frame.is_none());
        let rgb: Vec<u8> = (0..18).collect();
        tx.send(RenderMessage::AaProgress {
            display_buffer: rgb.clone(), width: 3, height: 2, sample: 4, total: 4,
        }).unwrap();
        tx.send(RenderMessage::AllComplete { orbit_cache: None }).unwrap();
        app.check_render_complete(&ctx);
        app.check_render_complete(&ctx);
        for pass_index in [1, 0] {
            tex_tx.send(TextureReadyMessage {
                params: params.clone(), pass_index, display_buffer: vec![99; 18],
                iterations: Arc::new(vec![pass_index as u32; 6]), zs: Arc::new(vec![Complex64::default(); 6]),
                distances: Vec::new(), orbits: Vec::new(), is_preview: pass_index == 0,
                effective_mode: AlgorithmMode::Auto, precision_label: String::new(),
            }).unwrap();
            app.check_render_complete(&ctx);
        }
        assert_eq!(app.iterations.as_slice(), &[1; 6]);
        let path = std::env::temp_dir().join(format!("fractall_screenshot_{}.png", std::process::id()));
        let frame = app.displayed_frame.as_ref().unwrap();
        frame.save(&path).unwrap();
        assert_eq!(image::open(&path).unwrap().to_rgb8().into_raw(), rgb);
        let saved = crate::io::png::load_png_metadata(&path).unwrap();
        assert_eq!(saved.center_x_hp, params.center_x_hp);
        assert_eq!((saved.width, saved.height), (3, 2));
        assert!(crate::io::png::load_png_command(&path).unwrap().contains("--aa-samples 4 --jitter-scale 0.25"));
        std::fs::remove_file(path).unwrap();

        // Changer la vue live avant la recolorisation conserve la géométrie du brut.
        app.params.center_x_hp = Some("1".into());
        app.params.width = 200;
        app.update_texture(&ctx);
        let recolor = app.recolor_receiver.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(recolor.params.width, 3);
        assert_eq!(recolor.params.center_x_hp, params.center_x_hp);
    }
}

#[cfg(test)]
mod warp_tests {
    use super::{compute_warp_norm, ViewSnapshot};

    fn v(cx: &str, cy: &str, sx: &str, sy: &str) -> ViewSnapshot {
        ViewSnapshot {
            cx: cx.to_string(),
            cy: cy.to_string(),
            sx: sx.to_string(),
            sy: sy.to_string(),
            axis_aligned: true,
        }
    }

    #[test]
    fn identity_view_no_warp() {
        let a = v("0", "0", "4", "3");
        assert!(compute_warp_norm(&a, &a).is_none());
    }

    #[test]
    fn zoom_in_2x_centered() {
        // Texture span 4, live span 2 (zoom ×2), même centre → vieille image
        // magnifiée ×2, centrée.
        let tex = v("0", "0", "4", "4");
        let live = v("0", "0", "2", "2");
        let (n_cx, v_cy, rx, ry) = compute_warp_norm(&tex, &live).unwrap();
        assert!((n_cx - 0.5).abs() < 1e-9);
        assert!((v_cy - 0.5).abs() < 1e-9);
        assert!((rx - 2.0).abs() < 1e-9);
        assert!((ry - 2.0).abs() < 1e-9);
    }

    #[test]
    fn pan_right_shifts_texture_left() {
        // Vue déplacée vers +réel d'un demi-span → l'ancien contenu va à GAUCHE.
        let tex = v("0", "0", "4", "4");
        let live = v("2", "0", "4", "4"); // cx_live = cx_tex + sx/2
        let (n_cx, _v_cy, rx, _ry) = compute_warp_norm(&tex, &live).unwrap();
        assert!((n_cx - 0.0).abs() < 1e-9, "n_cx={n_cx}");
        assert!((rx - 1.0).abs() < 1e-9);
    }

    #[test]
    fn pan_toward_higher_imag_shifts_texture_up() {
        // Symétrique de `pan_right_shifts_texture_left` : la vue se déplace vers
        // +imag (écran vers le BAS, imag croissant non-inversé) → l'ancien
        // contenu (imag plus petit) remonte vers le HAUT (v_cy < 0.5, ici 0.0).
        // Un `v_cy = 0.5 - dy` donnerait 1.0 (contenu en bas) = bug d'ancrage
        // vertical du warp sur zoom molette hors-centre.
        let tex = v("0", "0", "4", "4");
        let live = v("0", "2", "4", "4"); // cy_live = cy_tex + sy/2
        let (_n_cx, v_cy, _rx, ry) = compute_warp_norm(&tex, &live).unwrap();
        assert!((v_cy - 0.0).abs() < 1e-9, "v_cy={v_cy}");
        assert!((ry - 1.0).abs() < 1e-9);
    }

    #[test]
    fn deep_zoom_precision_needs_hp() {
        // cx_tex=1.0, cx_live=1+1e-40, span=2e-40 → dx=-0.5, n_cx=0.0.
        // En f64 pur, 1.0 et 1+1e-40 sont ÉGAUX (dx=0, faux). Le HP donne -0.5.
        let tex = v("1.0", "0", "2e-40", "2e-40");
        let live = v(
            "1.0000000000000000000000000000000000000001",
            "0",
            "2e-40",
            "2e-40",
        );
        let (n_cx, _v_cy, rx, _ry) = compute_warp_norm(&tex, &live).unwrap();
        assert!(
            (n_cx - 0.0).abs() < 1e-6,
            "n_cx={n_cx} (HP requis, pas f64)"
        );
        assert!((rx - 1.0).abs() < 1e-9);
    }

    #[test]
    fn zero_span_live_is_none() {
        let tex = v("0", "0", "4", "4");
        let live = v("0", "0", "0", "4");
        assert!(compute_warp_norm(&tex, &live).is_none());
    }
}
