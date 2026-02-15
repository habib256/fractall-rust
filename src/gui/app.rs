use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use egui::{Context, TextureHandle, TextureOptions};
use image::RgbImage;
use num_complex::Complex64;
use rug::Float;

use crate::color::{color_for_pixel, color_for_nebulabrot_pixel, color_for_buddhabrot_pixel, generate_palette_preview};
use crate::fractal::{AlgorithmMode, apply_lyapunov_preset, default_params_for_type, FractalParams, FractalType, LyapunovPreset, OutColoringMode, PlaneTransform};
use crate::fractal::perturbation::ReferenceOrbitCache;
use crate::render::render_escape_time_cancellable_with_reuse;
use crate::gui::texture::rgb_image_to_color_image;
use crate::gui::progressive::{ProgressiveConfig, RenderMessage, upscale_nearest};
use crate::gpu::GpuRenderer;

/// Précision par défaut pour les calculs de coordonnées haute précision (en bits).
const HP_PRECISION: u32 = 256;

/// Colorise un buffer d'itérations en RGB.
/// Cette fonction est appelée dans le thread de rendu ou dans l'UI pour re-coloriser avec la palette actuelle.
fn colorize_buffer(
    iterations: &[u32],
    zs: &[Complex64],
    distances: &[f64],
    orbits: &[Option<crate::fractal::orbit_traps::OrbitData>],
    params: &FractalParams,
    width: u32,
    height: u32,
) -> Vec<u8> {
    use rayon::prelude::*;

    let w = width as usize;
    let is_nebulabrot = params.fractal_type == FractalType::Nebulabrot;
    let is_buddhabrot = params.fractal_type == FractalType::Buddhabrot
        || params.fractal_type == FractalType::AntiBuddhabrot;
    let iter_max = params.iteration_max;
    let palette_idx = params.color_mode as u8;
    let color_rep = params.color_repeat;
    let out_mode = params.out_coloring_mode;
    let color_space = params.color_space;
    let interior_flag_encoded = params.enable_interior_detection;

    (0..height as usize)
        .into_par_iter()
        .flat_map(|y| {
            (0..width)
                .flat_map(|x| {
                    let idx = y * w + x as usize;
                    let iter = iterations.get(idx).copied().unwrap_or(0);
                    let z = zs.get(idx).copied().unwrap_or(Complex64::new(0.0, 0.0));
                    let orbit = orbits.get(idx).and_then(|o| o.as_ref());
                    let distance = distances.get(idx).copied().filter(|d| d.is_finite());

                    let (r, g, b) = if is_nebulabrot {
                        color_for_nebulabrot_pixel(iter, z)
                    } else if is_buddhabrot {
                        color_for_buddhabrot_pixel(z, palette_idx, color_rep)
                    } else {
                        color_for_pixel(
                            iter,
                            z,
                            iter_max,
                            palette_idx,
                            color_rep,
                            out_mode,
                            color_space,
                            orbit,
                            distance,
                            interior_flag_encoded,
                        )
                    };

                    vec![r, g, b]
                })
                .collect::<Vec<u8>>()
        })
        .collect()
}

/// Application principale egui pour fractall.
pub struct FractallApp {
    // État fractale
    params: FractalParams,
    iterations: Vec<u32>,
    zs: Vec<Complex64>,
    distances: Vec<f64>,
    orbits: Vec<Option<crate::fractal::orbit_traps::OrbitData>>,
    
    // Texture egui pour l'affichage
    texture: Option<TextureHandle>,
    
    // Cache des textures de prévisualisation des palettes
    palette_preview_textures: [Option<TextureHandle>; 13],
    
    // État UI
    selected_type: FractalType,
    palette_index: u8,
    color_repeat: u32,
    out_coloring_mode: OutColoringMode,
    selected_lyapunov_preset: LyapunovPreset,
    gpu_renderer: Option<Arc<GpuRenderer>>,
    /// N'a pas encore tenté de créer le GPU (init différée après ouverture de la fenêtre)
    gpu_init_attempted: bool,
    use_gpu: bool,
    
    // Coordonnées haute précision (représentation décimale exacte)
    // Permettent des zooms au-delà de la limite f64 (~1e-15)
    center_x_hp: String,
    center_y_hp: String,
    span_x_hp: String,
    span_y_hp: String,
    
    // Sélection rectangulaire pour zoom
    selecting: bool,
    select_start: Option<egui::Pos2>,
    select_current: Option<egui::Pos2>,
    
    // Rendu progressif
    rendering: bool,
    render_thread: Option<thread::JoinHandle<()>>,
    render_cancel: Arc<AtomicBool>,
    render_receiver: Option<mpsc::Receiver<RenderMessage>>,
    /// Canal pour recevoir les textures pré-colorisées (calcul hors thread principal).
    texture_ready_sender: Option<mpsc::Sender<TextureReadyMessage>>,
    texture_ready_receiver: Option<mpsc::Receiver<TextureReadyMessage>>,
    current_pass: u8,
    total_passes: u8,
    is_preview: bool,

    // Métriques
    last_render_time: Option<f64>, // en secondes
    render_start_time: Option<Instant>,
    last_render_device_label: Option<String>,
    last_render_method_label: Option<String>,

    // Dimensions de la fenêtre (pour calculer le viewport)
    window_width: u32,
    window_height: u32,
    pending_resize: Option<(u32, u32)>,

    // Cache for orbit/BLA to accelerate deep zoom re-renders
    orbit_cache: Option<Arc<ReferenceOrbitCache>>,

    // Fenêtre de rendu haute résolution
    show_render_dialog: bool,
    render_resolution_preset: RenderResolutionPreset,
    hq_rendering: bool,
    hq_render_progress: f32,
    hq_render_receiver: Option<mpsc::Receiver<HqRenderMessage>>,
    hq_render_result: Option<String>,

    // Canal dédié pour les recolorisations asynchrones (changement palette/color_repeat)
    // Persiste toute la vie de l'application, contrairement à texture_ready_* qui n'existe que pendant le rendu
    recolor_sender: mpsc::Sender<RecolorReadyMessage>,
    recolor_receiver: mpsc::Receiver<RecolorReadyMessage>,
    /// Compteur de version pour ignorer les recolorisations obsolètes (si l'utilisateur change le slider rapidement)
    recolor_version: u64,

    // Zone de texte pour le nombre d'itérations
    iteration_input: String,
    /// True si le champ Itérations avait le focus la frame précédente (pour ne pas intercepter "0" en raccourci reset)
    iteration_input_has_focus: bool,

    // Mini-Julia preview (shown when hovering over Mandelbrot)
    julia_preview_enabled: bool,
    julia_preview_texture: Option<TextureHandle>,
    julia_preview_sender: mpsc::Sender<JuliaPreviewMessage>,
    julia_preview_receiver: mpsc::Receiver<JuliaPreviewMessage>,
    julia_preview_version: u64,
    julia_preview_last_seed: Option<Complex64>,
    julia_preview_last_request_time: Option<Instant>,
    julia_preview_cancel: Arc<AtomicBool>,
    julia_preview_rendering: bool,
}

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

/// Message pour le rendu haute qualité asynchrone.
enum HqRenderMessage {
    Progress(f32),      // 0.0 à 1.0
    Done(String),       // Chemin du fichier sauvegardé
    Error(String),      // Message d'erreur
}

/// Message envoyé par le thread de colorisation vers l'UI (évite de bloquer le thread principal).
struct TextureReadyMessage {
    pass_index: u8,
    display_buffer: Vec<u8>,
    width: u32,
    height: u32,
    iterations: Vec<u32>,
    zs: Vec<Complex64>,
    distances: Vec<f64>,
    orbits: Vec<Option<crate::fractal::orbit_traps::OrbitData>>,
    is_preview: bool,
    effective_mode: AlgorithmMode,
    precision_label: String,
}

/// Message pour les recolorisations asynchrones (changement palette/color_repeat).
/// Plus simple que TextureReadyMessage car on ne change pas les données d'itération.
struct RecolorReadyMessage {
    display_buffer: Vec<u8>,
    width: u32,
    height: u32,
    version: u64,
}

/// Message pour le rendu asynchrone de la preview Julia.
struct JuliaPreviewMessage {
    display_buffer: Vec<u8>,
    width: u32,
    height: u32,
    seed: Complex64,
    version: u64,
}

impl FractallApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let default_type = FractalType::Mandelbrot;
        let width = 800;
        let height = 600;
        let params = default_params_for_type(default_type, width, height);

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

        Self {
            params: params.clone(),
            iterations: Vec::new(),
            zs: Vec::new(),
            distances: Vec::new(),
            orbits: Vec::new(),
            texture: None,
            palette_preview_textures: [
                None, None, None, None, None,
                None, None, None, None, None,
                None, None, None,
            ],
            selected_type: default_type,
            palette_index: 6, // SmoothPlasma par défaut
            color_repeat: 40,
            out_coloring_mode: OutColoringMode::Smooth,
            selected_lyapunov_preset: LyapunovPreset::default(),
            gpu_renderer,
            gpu_init_attempted,
            use_gpu: use_gpu_default,
            // Initialiser les coordonnées haute précision depuis les params f64
            center_x_hp: params.center_x.to_string(),
            center_y_hp: params.center_y.to_string(),
            span_x_hp: params.span_x.to_string(),
            span_y_hp: params.span_y.to_string(),
            selecting: false,
            select_start: None,
            select_current: None,
            rendering: false,
            render_thread: None,
            render_cancel: Arc::new(AtomicBool::new(false)),
            render_receiver: None,
            texture_ready_sender: None,
            texture_ready_receiver: None,
            current_pass: 0,
            total_passes: 0,
            is_preview: false,
            last_render_time: None,
            render_start_time: None,
            last_render_device_label: None,
            last_render_method_label: None,
            window_width: width,
            window_height: height,
            pending_resize: None,
            orbit_cache: None,
            show_render_dialog: false,
            render_resolution_preset: RenderResolutionPreset::Res4K,
            hq_rendering: false,
            hq_render_progress: 0.0,
            hq_render_receiver: None,
            hq_render_result: None,
            recolor_sender: recolor_tx,
            recolor_receiver: recolor_rx,
            recolor_version: 0,

            iteration_input: params.iteration_max.to_string(),
            iteration_input_has_focus: false,

            // Mini-Julia preview (désactivé par défaut pour Mandelbrot)
            julia_preview_enabled: false,
            julia_preview_texture: None,
            julia_preview_sender: julia_preview_tx,
            julia_preview_receiver: julia_preview_rx,
            julia_preview_version: 0,
            julia_preview_last_seed: None,
            julia_preview_last_request_time: None,
            julia_preview_cancel: Arc::new(AtomicBool::new(false)),
            julia_preview_rendering: false,
        }
    }
    
    /// Synchronise les coordonnées haute précision vers les params f64 et String.
    /// Appelé après chaque modification des coordonnées HP.
    /// Stocke les String dans FractalParams pour préserver la précision pour les calculs GMP.
    fn sync_hp_to_params(&mut self) {
        // Stocker les String directement dans FractalParams pour préserver la précision
        self.params.center_x_hp = Some(self.center_x_hp.clone());
        self.params.center_y_hp = Some(self.center_y_hp.clone());
        self.params.span_x_hp = Some(self.span_x_hp.clone());
        self.params.span_y_hp = Some(self.span_y_hp.clone());
        
        // Parser les strings HP vers rug::Float puis convertir en f64 pour compatibilité GPU/CPU standard
        let prec = HP_PRECISION;
        
        if let Ok(cx) = Float::parse(&self.center_x_hp) {
            self.params.center_x = Float::with_val(prec, cx).to_f64();
        }
        if let Ok(cy) = Float::parse(&self.center_y_hp) {
            self.params.center_y = Float::with_val(prec, cy).to_f64();
        }
        if let Ok(sx) = Float::parse(&self.span_x_hp) {
            self.params.span_x = Float::with_val(prec, sx).to_f64();
        }
        if let Ok(sy) = Float::parse(&self.span_y_hp) {
            self.params.span_y = Float::with_val(prec, sy).to_f64();
        }
    }
    
    /// Met à jour les coordonnées HP depuis les params f64.
    /// Appelé quand on change de type de fractale ou reset.
    /// IMPORTANT: Utiliser to_string_radix avec précision maximale pour préserver la précision
    /// aux zooms profonds (>e16). format!("{:.20e}") limite à 20 chiffres significatifs.
    fn sync_params_to_hp(&mut self) {
        let prec = HP_PRECISION;
        // Convertir f64 → GMP Float → String avec précision maximale
        // Cela préserve beaucoup plus de précision que format!("{:.20e}")
        let cx_gmp = Float::with_val(prec, self.params.center_x);
        let cy_gmp = Float::with_val(prec, self.params.center_y);
        let sx_gmp = Float::with_val(prec, self.params.span_x);
        let sy_gmp = Float::with_val(prec, self.params.span_y);
        
        // Utiliser to_string_radix(10, None) pour obtenir tous les chiffres significatifs
        self.center_x_hp = cx_gmp.to_string_radix(10, None);
        self.center_y_hp = cy_gmp.to_string_radix(10, None);
        self.span_x_hp = sx_gmp.to_string_radix(10, None);
        self.span_y_hp = sy_gmp.to_string_radix(10, None);
    }
    
    /// Effectue un zoom en haute précision.
    /// `ratio_x`, `ratio_y` : position relative dans l'image (0.0-1.0)
    /// `zoom_factor` : facteur de zoom (>1 = zoom in, <1 = zoom out)
    fn zoom_hp(&mut self, ratio_x: f64, ratio_y: f64, zoom_factor: f64) {
        let prec = HP_PRECISION;
        
        // Parser les coordonnées actuelles
        let center_x = Float::parse(&self.center_x_hp)
            .map(|p| Float::with_val(prec, p))
            .unwrap_or_else(|_| Float::with_val(prec, self.params.center_x));
        let center_y = Float::parse(&self.center_y_hp)
            .map(|p| Float::with_val(prec, p))
            .unwrap_or_else(|_| Float::with_val(prec, self.params.center_y));
        let span_x = Float::parse(&self.span_x_hp)
            .map(|p| Float::with_val(prec, p))
            .unwrap_or_else(|_| Float::with_val(prec, self.params.span_x));
        let span_y = Float::parse(&self.span_y_hp)
            .map(|p| Float::with_val(prec, p))
            .unwrap_or_else(|_| Float::with_val(prec, self.params.span_y));
        
        // Calculer le nouveau centre: center + (ratio - 0.5) * span
        let offset_x = Float::with_val(prec, ratio_x - 0.5) * &span_x;
        let offset_y = Float::with_val(prec, ratio_y - 0.5) * &span_y;
        let new_center_x = center_x + offset_x;
        let new_center_y = center_y + offset_y;
        
        // Nouveaux spans (divisés par le facteur de zoom)
        let target_aspect = self.params.width as f64 / self.params.height as f64;
        let new_span_y = span_y / Float::with_val(prec, zoom_factor);
        let new_span_x = Float::with_val(prec, target_aspect) * &new_span_y;
        
        // Sauvegarder en strings avec précision maximale
        // IMPORTANT: Utiliser to_string_radix(10, None) pour préserver toute la précision GMP
        // aux zooms profonds (>e16). to_string() peut limiter la précision dans certains cas.
        self.center_x_hp = new_center_x.to_string_radix(10, None);
        self.center_y_hp = new_center_y.to_string_radix(10, None);
        self.span_x_hp = new_span_x.to_string_radix(10, None);
        self.span_y_hp = new_span_y.to_string_radix(10, None);
        
        // Synchroniser vers f64 pour le rendu
        self.sync_hp_to_params();
    }
    
    /// Effectue un zoom rectangulaire en haute précision.
    fn zoom_rect_hp(&mut self, xr1: f64, yr1: f64, xr2: f64, yr2: f64) {
        let prec = HP_PRECISION;
        
        // Parser les coordonnées actuelles
        let center_x = Float::parse(&self.center_x_hp)
            .map(|p| Float::with_val(prec, p))
            .unwrap_or_else(|_| Float::with_val(prec, self.params.center_x));
        let center_y = Float::parse(&self.center_y_hp)
            .map(|p| Float::with_val(prec, p))
            .unwrap_or_else(|_| Float::with_val(prec, self.params.center_y));
        let span_x = Float::parse(&self.span_x_hp)
            .map(|p| Float::with_val(prec, p))
            .unwrap_or_else(|_| Float::with_val(prec, self.params.span_x));
        let span_y = Float::parse(&self.span_y_hp)
            .map(|p| Float::with_val(prec, p))
            .unwrap_or_else(|_| Float::with_val(prec, self.params.span_y));
        
        // Nouveau centre: center + ((r1+r2)/2 - 0.5) * span
        let mid_ratio_x = (xr1 + xr2) * 0.5 - 0.5;
        let mid_ratio_y = (yr1 + yr2) * 0.5 - 0.5;
        let new_center_x = &center_x + Float::with_val(prec, mid_ratio_x) * &span_x;
        let new_center_y = &center_y + Float::with_val(prec, mid_ratio_y) * &span_y;
        
        // Nouveaux spans: (r2 - r1) * span
        let selection_span_x = Float::with_val(prec, xr2 - xr1) * &span_x;
        let selection_span_y = Float::with_val(prec, yr2 - yr1) * &span_y;
        
        // Ajuster pour le ratio d'aspect
        let target_aspect = self.params.width as f64 / self.params.height as f64;
        let selection_aspect = selection_span_x.to_f64() / selection_span_y.to_f64();
        
        let (new_span_x, new_span_y) = if selection_aspect > target_aspect {
            // Sélection plus large : élargir span_y
            let sy = &selection_span_x / Float::with_val(prec, target_aspect);
            (selection_span_x, sy)
        } else {
            // Sélection plus haute : élargir span_x
            let sx = &selection_span_y * Float::with_val(prec, target_aspect);
            (sx, selection_span_y)
        };
        
        // Sauvegarder en strings avec précision maximale
        // IMPORTANT: Utiliser to_string_radix(10, None) pour préserver toute la précision GMP
        self.center_x_hp = new_center_x.to_string_radix(10, None);
        self.center_y_hp = new_center_y.to_string_radix(10, None);
        self.span_x_hp = new_span_x.to_string_radix(10, None);
        self.span_y_hp = new_span_y.to_string_radix(10, None);
        
        // Synchroniser vers f64
        self.sync_hp_to_params();
    }
    
    /// Dézoom en haute précision.
    fn zoom_out_hp(&mut self, factor: f64) {
        let prec = HP_PRECISION;
        
        let span_y = Float::parse(&self.span_y_hp)
            .map(|p| Float::with_val(prec, p))
            .unwrap_or_else(|_| Float::with_val(prec, self.params.span_y));
        
        let target_aspect = self.params.width as f64 / self.params.height as f64;
        let new_span_y = span_y * Float::with_val(prec, factor);
        let new_span_x = Float::with_val(prec, target_aspect) * &new_span_y;
        
        // IMPORTANT: Utiliser to_string_radix(10, None) pour préserver toute la précision GMP
        self.span_x_hp = new_span_x.to_string_radix(10, None);
        self.span_y_hp = new_span_y.to_string_radix(10, None);

        self.sync_hp_to_params();
    }

    /// Charge l'état de la fractale depuis un fichier PNG contenant des métadonnées.
    fn load_from_png(&mut self, path: &std::path::Path) {
        match crate::io::png::load_png_metadata(path) {
            Ok(params) => {
                // Restaurer le type de fractale
                self.selected_type = params.fractal_type;

                // Restaurer les paramètres de couleur
                self.palette_index = params.color_mode;
                self.color_repeat = params.color_repeat;
                self.out_coloring_mode = params.out_coloring_mode;

                // Restaurer les coordonnées HP depuis les params
                self.center_x_hp = params.center_x_hp.clone().unwrap_or_else(|| params.center_x.to_string());
                self.center_y_hp = params.center_y_hp.clone().unwrap_or_else(|| params.center_y.to_string());
                self.span_x_hp = params.span_x_hp.clone().unwrap_or_else(|| params.span_x.to_string());
                self.span_y_hp = params.span_y_hp.clone().unwrap_or_else(|| params.span_y.to_string());

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
                if is_density && self.params.color_repeat > 8 {
                    self.params.color_repeat = 8;
                    self.color_repeat = 8;
                }

                // Synchroniser HP vers params
                self.sync_hp_to_params();
                self.iteration_input = self.params.iteration_max.to_string();

                // Invalider le cache et relancer le rendu
                self.orbit_cache = None;
                self.start_render();

                println!("État restauré depuis: {}", path.display());
            }
            Err(e) => {
                eprintln!("Erreur chargement PNG: {}", e);
            }
        }
    }

    /// Lance un rendu haute résolution asynchrone et sauvegarde le résultat.
    fn render_high_quality(&mut self) {
        let (render_width, render_height) = self.render_resolution_preset.resolution(
            self.window_width,
            self.window_height,
        );

        // Créer les params pour le rendu haute résolution
        let mut render_params = self.params.clone();
        render_params.width = render_width;
        render_params.height = render_height;

        // Ajuster span pour conserver le ratio d'aspect
        let current_aspect = self.params.span_x / self.params.span_y;
        let target_aspect = render_width as f64 / render_height as f64;

        if current_aspect > target_aspect {
            render_params.span_y = render_params.span_x / target_aspect;
        } else {
            render_params.span_x = render_params.span_y * target_aspect;
        }

        // Copier les coordonnées HP pour le thread
        let center_x_hp = self.center_x_hp.clone();
        let center_y_hp = self.center_y_hp.clone();
        let span_x_hp = self.span_x_hp.clone();
        let span_y_hp = self.span_y_hp.clone();

        // Canal de communication
        let (tx, rx) = mpsc::channel();
        self.hq_render_receiver = Some(rx);
        self.hq_rendering = true;
        self.hq_render_progress = 0.0;
        self.hq_render_result = None;

        println!("Rendering at {}x{}...", render_width, render_height);

        // Configuration progressive (même logique que le rendu fenêtre)
        let allow_intermediate = !matches!(
            render_params.fractal_type,
            FractalType::VonKoch | FractalType::Dragon | FractalType::Buddhabrot
                | FractalType::Nebulabrot | FractalType::AntiBuddhabrot | FractalType::Lyapunov
        );
        let config = ProgressiveConfig::for_params_with_intermediate(
            render_width,
            render_height,
            render_params.use_gmp,
            allow_intermediate,
        );

        // Lancer le rendu progressif dans un thread séparé (multi-passes pour barre de progression)
        thread::spawn(move || {
            let cancel = Arc::new(AtomicBool::new(false));
            let total_passes = config.passes.len() as f32;
            let mut previous_pass: Option<(Vec<u32>, Vec<Complex64>, u32, u32)> = None;
            let mut final_result: Option<(Vec<u32>, Vec<Complex64>)> = None;

            for (pass_index, &scale_divisor) in config.passes.iter().enumerate() {
                if cancel.load(Ordering::Relaxed) {
                    let _ = tx.send(HqRenderMessage::Error("Render cancelled".to_string()));
                    return;
                }
                let pass_width = (render_width / scale_divisor as u32).max(1);
                let pass_height = (render_height / scale_divisor as u32).max(1);
                let mut pass_params = render_params.clone();
                pass_params.width = pass_width;
                pass_params.height = pass_height;

                let reuse = previous_pass.as_ref().map(|(i, z, w, h)| (i.as_slice(), z.as_slice(), *w, *h));
                let result = crate::render::escape_time::render_escape_time_cancellable_with_reuse(
                    &pass_params,
                    &cancel,
                    reuse,
                );

                match result {
                    Some((iterations, zs, _orbits, _distances)) => {
                        // Progression : 5% au début, puis 5–90% répartis sur les passes, 90–95% sauvegarde
                        let pass_progress = (pass_index + 1) as f32 / total_passes * 0.85f32; // 0.85 * (1/5..5/5)
                        let progress = 0.05f32 + pass_progress; // 5% -> 90%
                        let _ = tx.send(HqRenderMessage::Progress(progress));

                        if pass_index + 1 == config.passes.len() {
                            final_result = Some((iterations, zs));
                            break;
                        }
                        previous_pass = Some((iterations, zs, pass_width, pass_height));
                    }
                    None => {
                        let _ = tx.send(HqRenderMessage::Error("Render cancelled".to_string()));
                        return;
                    }
                }
            }

            if let Some((iterations, zs)) = final_result {
                let _ = tx.send(HqRenderMessage::Progress(0.92));

                use std::path::Path;
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let filename = format!("fractal_{}x{}_{}.png", render_width, render_height, timestamp);

                if let Err(e) = crate::io::png::save_png_with_metadata(
                    &render_params,
                    &iterations,
                    &zs,
                    Path::new(&filename),
                    &center_x_hp,
                    &center_y_hp,
                    &span_x_hp,
                    &span_y_hp,
                ) {
                    let _ = tx.send(HqRenderMessage::Error(format!("Error saving PNG: {}", e)));
                } else {
                    let _ = tx.send(HqRenderMessage::Progress(1.0));
                    println!("High quality render saved: {}", filename);
                    let _ = tx.send(HqRenderMessage::Done(filename));
                }
            }
        });
    }

    /// Lance le rendu progressif de la fractale dans un thread séparé.
    fn start_render(&mut self) {
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
        let use_gpu = self.use_gpu
            && self.gpu_renderer.is_some()
            && matches!(
                self.params.fractal_type,
                FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip
            );
        let config = ProgressiveConfig::for_params_with_intermediate(
            self.params.width,
            self.params.height,
            self.params.use_gmp,
            allow_intermediate,
        );

        self.total_passes = config.passes.len() as u8;
        self.current_pass = 0;
        self.rendering = true;
        self.is_preview = true;
        self.render_start_time = Some(Instant::now());

        // Créer le channel de communication
        let (sender, receiver) = mpsc::channel();
        self.render_receiver = Some(receiver);

        // Canal pour les textures pré-colorisées (évite de bloquer l'UI)
        let (tex_tx, tex_rx) = mpsc::channel();
        self.texture_ready_sender = Some(tex_tx);
        self.texture_ready_receiver = Some(tex_rx);

        // Paramètres pour le thread (activer orbit traps / distance selon outcoloring)
        let mut params = self.params.clone();
        match params.out_coloring_mode {
            crate::fractal::OutColoringMode::OrbitTraps | crate::fractal::OutColoringMode::Wings => {
                params.enable_orbit_traps = true;
            }
            _ => {}
        }
        if matches!(
            params.out_coloring_mode,
            crate::fractal::OutColoringMode::Distance
                | crate::fractal::OutColoringMode::DistanceAO
                | crate::fractal::OutColoringMode::Distance3D
        ) {
            params.enable_distance_estimation = true;
        }
        let cancel = Arc::clone(&self.render_cancel);
        let full_width = self.params.width;
        let full_height = self.params.height;
        let gpu_renderer = self.gpu_renderer.clone();
        let use_gpu = use_gpu;
        let orbit_cache = self.orbit_cache.clone();

        // Spawner le thread de rendu progressif
        let handle = thread::spawn(move || {
            let mut previous_pass: Option<(Vec<u32>, Vec<Complex64>, u32, u32)> = None;
            let mut current_orbit_cache = orbit_cache;

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

                // Rendre cette passe (réutiliser la passe précédente si possible)
                let reuse = previous_pass.as_ref().map(|(iter, zs, w, h)| {
                    (iter.as_slice(), zs.as_slice(), *w, *h)
                });
                let result = if use_gpu {
                    let gpu = gpu_renderer.as_ref().unwrap();
                    let use_perturbation = match pass_params.algorithm_mode {
                        AlgorithmMode::Auto => crate::render::escape_time::should_use_perturbation(&pass_params, true),
                        AlgorithmMode::Perturbation => true,
                        _ => false,
                    };
                    let use_perturbation =
                        use_perturbation && pass_params.plane_transform == PlaneTransform::Mu;
                    
                    let gpu_result = match pass_params.fractal_type {
                        FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip
                            if use_perturbation =>
                        {
                            gpu.render_perturbation_with_cache(&pass_params, &cancel, reuse, current_orbit_cache.as_ref())
                                .map(|((iterations, zs), cache)| {
                                    current_orbit_cache = Some(cache);
                                    let n = iterations.len();
                                    (iterations, zs, Vec::new(), vec![None; n])
                                })
                        }
                        FractalType::Mandelbrot => gpu.render_mandelbrot(&pass_params, &cancel)
                            .map(|(i, z)| { let n = i.len(); (i, z, Vec::new(), vec![None; n]) }),
                        FractalType::Julia => gpu.render_julia(&pass_params, &cancel)
                            .map(|(i, z)| { let n = i.len(); (i, z, Vec::new(), vec![None; n]) }),
                        FractalType::BurningShip => gpu.render_burning_ship(&pass_params, &cancel)
                            .map(|(i, z)| { let n = i.len(); (i, z, Vec::new(), vec![None; n]) }),
                        _ => None,
                    };
                    if let Some((iterations, zs, distances, orbits)) = gpu_result {
                        let base_precision = gpu.precision_label();
                        let precision = base_precision.to_string();
                        let effective_mode = if use_perturbation {
                            AlgorithmMode::Perturbation
                        } else {
                            AlgorithmMode::StandardF64
                        };
                        Some((
                            (iterations, zs, distances, orbits),
                            effective_mode,
                            format!("GPU {}", precision),
                        ))
                    } else {
                        // GPU fallback to CPU: check if we should use perturbation with cache
                        let fallback_use_perturbation = match pass_params.algorithm_mode {
                            AlgorithmMode::Auto => crate::render::escape_time::should_use_perturbation(&pass_params, false),
                            AlgorithmMode::Perturbation => true,
                            _ => false,
                        };
                        let fallback_use_perturbation =
                            fallback_use_perturbation && pass_params.plane_transform == PlaneTransform::Mu;

                        if fallback_use_perturbation && matches!(pass_params.fractal_type,
                            FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip)
                        {
                            // Use cache-aware CPU perturbation rendering for GPU fallback
                            use crate::fractal::perturbation::render_perturbation_with_cache;
                            render_perturbation_with_cache(&pass_params, &cancel, reuse, current_orbit_cache.as_ref())
                                .map(|((iterations, zs, distances), cache)| {
                                    current_orbit_cache = Some(cache);
                                    let n = iterations.len();
                                    ((iterations, zs, distances, vec![None; n]), AlgorithmMode::Perturbation, "CPU f64".to_string())
                                })
                        } else {
                            let cpu_result =
                                render_escape_time_cancellable_with_reuse(&pass_params, &cancel, reuse);
                            cpu_result.map(|(iterations, zs, orbits, distances)| {
                                let effective_mode = match pass_params.algorithm_mode {
                                    AlgorithmMode::ReferenceGmp => AlgorithmMode::ReferenceGmp,
                                    AlgorithmMode::Perturbation => AlgorithmMode::Perturbation,
                                    AlgorithmMode::StandardF64 => AlgorithmMode::StandardF64,
                                    AlgorithmMode::Auto => {
                                        if crate::render::escape_time::should_use_perturbation(
                                            &pass_params,
                                            false,
                                        ) {
                                            AlgorithmMode::Perturbation
                                        } else {
                                            AlgorithmMode::StandardF64
                                        }
                                    }
                                };
                                let precision_label = match effective_mode {
                                    AlgorithmMode::ReferenceGmp => {
                                        let effective_prec = crate::fractal::perturbation::compute_perturbation_precision_bits(&pass_params);
                                        format!("CPU GMP {}b", effective_prec)
                                    }
                                    AlgorithmMode::Perturbation => "CPU f64".to_string(),
                                    _ => "CPU f64".to_string(),
                                };
                                ((iterations, zs, distances, orbits), effective_mode, precision_label)
                            })
                        }
                    }
                } else {
                    // CPU rendering - La perturbation f64 est désactivée en mode Auto car trop lente.
                    // On utilise CPU f64 standard jusqu'à zoom ~10^16, puis GMP reference au-delà.
                    let use_perturbation = match pass_params.algorithm_mode {
                        AlgorithmMode::Auto => false, // Désactivé: trop lent comparé à CPU f64 standard et GMP reference
                        AlgorithmMode::Perturbation => true,
                        _ => false,
                    };
                    let use_perturbation =
                        use_perturbation && pass_params.plane_transform == PlaneTransform::Mu;

                    if use_perturbation && matches!(pass_params.fractal_type, FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip) {
                        // Use cache-aware CPU perturbation rendering
                        use crate::fractal::perturbation::render_perturbation_with_cache;
                        render_perturbation_with_cache(&pass_params, &cancel, reuse, current_orbit_cache.as_ref())
                            .map(|((iterations, zs, distances), cache)| {
                                current_orbit_cache = Some(cache);
                                let n = iterations.len();
                                ((iterations, zs, distances, vec![None; n]), AlgorithmMode::Perturbation, "CPU f64".to_string())
                            })
                    } else {
                        render_escape_time_cancellable_with_reuse(&pass_params, &cancel, reuse).map(|(iterations, zs, orbits, distances)| {
                            let effective_mode = match pass_params.algorithm_mode {
                                AlgorithmMode::ReferenceGmp => AlgorithmMode::ReferenceGmp,
                                AlgorithmMode::Perturbation => AlgorithmMode::Perturbation,
                                AlgorithmMode::StandardF64 => AlgorithmMode::StandardF64,
                                AlgorithmMode::Auto => {
                                    // En mode Auto: CPU f64 standard jusqu'à zoom ~10^16, puis GMP reference
                                    if crate::render::escape_time::should_use_gmp_reference(
                                        &pass_params,
                                    ) {
                                        AlgorithmMode::ReferenceGmp
                                    } else {
                                        AlgorithmMode::StandardF64
                                    }
                                }
                            };
                            let precision_label = match effective_mode {
                                AlgorithmMode::ReferenceGmp => {
                                    let effective_prec = crate::fractal::perturbation::compute_perturbation_precision_bits(&pass_params);
                                    format!("CPU GMP {}b", effective_prec)
                                }
                                AlgorithmMode::Perturbation => "CPU f64".to_string(),
                                _ => "CPU f64".to_string(),
                            };
                            ((iterations, zs, distances, orbits), effective_mode, precision_label)
                        })
                    }
                };

                match result {
                    Some(((iterations, zs, distances, orbits), effective_mode, precision_label)) => {
                        // Garder une copie pour la passe suivante afin d'éviter le recalcul
                        if pass_index + 1 < config.passes.len() {
                            previous_pass = Some((iterations.clone(), zs.clone(), pass_width, pass_height));
                        } else {
                            previous_pass = None;
                        }

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

            let _ = sender.send(RenderMessage::AllComplete { orbit_cache: current_orbit_cache });
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
            if msg.version == self.recolor_version {
                self.load_texture_from_buffer(ctx, &msg.display_buffer, msg.width, msg.height);
                ctx.request_repaint();
                break;
            }
            // Message obsolète, on continue à vider le channel
        }

        // Traiter d'abord une texture prête (calcul faite hors thread principal)
        if let Some(ref rx) = self.texture_ready_receiver {
            if let Ok(tex) = rx.try_recv() {
                self.current_pass = tex.pass_index + 1;
                self.last_render_device_label = Some(tex.precision_label.clone());
                self.last_render_method_label = Some(match tex.effective_mode {
                    AlgorithmMode::ReferenceGmp => String::new(),
                    AlgorithmMode::Perturbation => "Perturbation".to_string(),
                    _ => "Standard".to_string(),
                });
                self.iterations = tex.iterations;
                self.zs = tex.zs;
                self.distances = tex.distances;
                self.orbits = tex.orbits;
                self.is_preview = tex.is_preview;
                self.load_texture_from_buffer(ctx, &tex.display_buffer, tex.width, tex.height);
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
            } => {
                // Déléguer upscale + colorize à un thread pour ne pas bloquer l'UI (évite "ne répond pas")
                let tx = match &self.texture_ready_sender {
                    Some(t) => t.clone(),
                    None => return,
                };
                let params = self.params.clone();
                let total_width = self.params.width;
                let total_height = self.params.height;
                thread::spawn(move || {
                    let (iterations, zs, distances, orbits, disp_w, disp_h, is_preview) = if scale_divisor > 1 {
                        let (upscaled_iter, upscaled_zs) = upscale_nearest(
                            &iterations,
                            &zs,
                            width,
                            height,
                            total_width,
                            total_height,
                        );
                        (
                            upscaled_iter,
                            upscaled_zs,
                            Vec::new(),
                            Vec::new(),
                            total_width,
                            total_height,
                            true,
                        )
                    } else {
                        (
                            iterations,
                            zs,
                            distances,
                            orbits,
                            width,
                            height,
                            false,
                        )
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
                        pass_index,
                        display_buffer,
                        width: disp_w,
                        height: disp_h,
                        iterations,
                        zs,
                        distances,
                        orbits,
                        is_preview,
                        effective_mode,
                        precision_label,
                    });
                });
                ctx.request_repaint();
            }

            RenderMessage::AllComplete { orbit_cache } => {
                // Avant de fermer les canaux : récupérer toute texture déjà envoyée par le worker
                // (ex. dernière passe pleine résolution), sinon attendre brièvement pour éviter
                // que l'image reste pixellisée quand AllComplete arrive avant TextureReadyMessage.
                if let Some(rx) = self.texture_ready_receiver.take() {
                    let mut last_tex = None;
                    while let Ok(tex) = rx.try_recv() {
                        last_tex = Some(tex);
                    }
                    if last_tex.is_none() {
                        if let Ok(tex) = rx.recv_timeout(Duration::from_millis(100)) {
                            last_tex = Some(tex);
                        }
                    }
                    if let Some(tex) = last_tex {
                        self.current_pass = tex.pass_index + 1;
                        self.last_render_device_label = Some(tex.precision_label.clone());
                        self.last_render_method_label = Some(match tex.effective_mode {
                            AlgorithmMode::ReferenceGmp => String::new(),
                            AlgorithmMode::Perturbation => "Perturbation".to_string(),
                            _ => "Standard".to_string(),
                        });
                        self.iterations = tex.iterations;
                        self.zs = tex.zs;
                        self.distances = tex.distances;
                        self.orbits = tex.orbits;
                        self.is_preview = tex.is_preview;
                        self.load_texture_from_buffer(ctx, &tex.display_buffer, tex.width, tex.height);
                        ctx.request_repaint();
                    }
                }
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
                }

                // Gérer le redimensionnement en attente
                if let Some((w, h)) = self.pending_resize.take() {
                    self.apply_resize(w, h);
                }
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
        if new_width == self.window_width && new_height == self.window_height {
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
        let current_aspect = self.params.span_x / self.params.span_y;
        let target_aspect = new_width as f64 / new_height as f64;

        let (new_span_x, new_span_y) = if current_aspect > target_aspect {
            // Élargir la hauteur pour éviter toute déformation.
            (self.params.span_x, self.params.span_x / target_aspect)
        } else {
            // Élargir la largeur pour éviter toute déformation.
            (self.params.span_y * target_aspect, self.params.span_y)
        };

        self.params.span_x = new_span_x;
        self.params.span_y = new_span_y;
        self.window_width = new_width;
        self.window_height = new_height;
        self.params.width = new_width;
        self.params.height = new_height;
        // Synchroniser les chaînes HP pour éviter un saut au prochain zoom profond
        self.sync_params_to_hp();
        self.iterations.clear();
        self.zs.clear();
        self.orbits.clear();
        self.texture = None;
        self.start_render();
    }

    /// Charge une texture depuis un buffer RGB pré-colorisé.
    /// Utilisé pour les passes de rendu (évite de bloquer l'UI).
    fn load_texture_from_buffer(&mut self, ctx: &Context, rgb_buffer: &[u8], width: u32, height: u32) {
        let expected_size = (width as usize) * (height as usize) * 3;
        if rgb_buffer.len() != expected_size {
            eprintln!("Warning: RGB buffer size mismatch: got {}, expected {}", rgb_buffer.len(), expected_size);
            return;
        }
        let Some(img) = RgbImage::from_raw(width, height, rgb_buffer.to_vec()) else {
            eprintln!("Warning: Failed to create RgbImage from buffer");
            return;
        };
        let color_image = rgb_image_to_color_image(&img);
        self.texture = Some(ctx.load_texture("fractal", color_image, TextureOptions::LINEAR));
    }

    /// Met à jour la texture egui à partir des données de fractale.
    /// Utilisé pour les changements de palette/couleur (re-colorisation nécessaire).
    /// La colorisation est faite de manière asynchrone pour ne pas bloquer l'UI.
    fn update_texture(&mut self, ctx: &Context) {
        let width = self.params.width;
        let height = self.params.height;

        if self.iterations.is_empty() || self.zs.is_empty() {
            return;
        }

        // S'assurer que orbits a la même taille que iterations/zs
        let target_len = self.iterations.len();
        if self.orbits.len() < target_len {
            self.orbits.resize(target_len, None);
        } else if self.orbits.len() > target_len {
            self.orbits.truncate(target_len);
        }

        // Cloner les données pour le thread de colorisation
        let iterations = self.iterations.clone();
        let zs = self.zs.clone();
        let distances = self.distances.clone();
        let orbits = self.orbits.clone();
        let mut params = self.params.clone();
        // Mettre à jour les params avec les valeurs actuelles de l'UI
        params.color_mode = self.palette_index;
        params.color_repeat = self.color_repeat;
        params.out_coloring_mode = self.out_coloring_mode;

        let tx = self.recolor_sender.clone();

        // Incrémenter la version pour ignorer les résultats obsolètes
        self.recolor_version = self.recolor_version.wrapping_add(1);
        let version = self.recolor_version;

        // Spawner un thread pour la colorisation (ne bloque pas l'UI)
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
                display_buffer,
                width,
                height,
                version,
            });
        });

        // Demander un repaint pour recevoir le résultat
        ctx.request_repaint();
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
        self.julia_preview_version = self.julia_preview_version.wrapping_add(1);
        let version = self.julia_preview_version;

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
        params.algorithm_mode = AlgorithmMode::StandardF64;
        params.color_mode = self.palette_index;
        params.color_repeat = self.color_repeat;
        params.out_coloring_mode = self.out_coloring_mode;

        let tx = self.julia_preview_sender.clone();
        self.julia_preview_rendering = true;

        // Spawner le thread de rendu
        thread::spawn(move || {
            if cancel.load(Ordering::Relaxed) {
                return;
            }

            let result = render_escape_time_cancellable_with_reuse(&params, &cancel, None);

            if cancel.load(Ordering::Relaxed) {
                return;
            }

            if let Some((iterations, zs, orbits, distances)) = result {
                let display_buffer = colorize_buffer(
                    &iterations,
                    &zs,
                    &distances,
                    &orbits,
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
            if msg.version != self.julia_preview_version {
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

    /// Calcule les coordonnées complexes depuis les coordonnées pixel.
    fn pixel_to_complex(&self, pixel_x: f32, pixel_y: f32, viewport_width: f32, viewport_height: f32) -> Complex64 {
        let x_ratio = pixel_x as f64 / viewport_width as f64;
        let y_ratio = pixel_y as f64 / viewport_height as f64;
        
        // Utiliser center+span directement pour éviter les problèmes de précision
        // x = center_x + (ratio - 0.5) * span_x
        let x = self.params.center_x + (x_ratio - 0.5) * self.params.span_x;
        // y = center_y + (ratio - 0.5) * span_y (même convention que le rendu)
        let y = self.params.center_y + (y_ratio - 0.5) * self.params.span_y;
        
        Complex64::new(x, y)
    }
    
    /// Zoom au point spécifié avec un facteur donné.
    fn zoom_at_point(&mut self, point: Complex64, factor: f64) {
        // Calculer le ratio du point dans l'image
        // point = center + (ratio - 0.5) * span
        // => ratio = (point - center) / span + 0.5
        let ratio_x = (point.re - self.params.center_x) / self.params.span_x + 0.5;
        let ratio_y = (point.im - self.params.center_y) / self.params.span_y + 0.5;
        
        // Utiliser le zoom haute précision
        self.zoom_hp(ratio_x, ratio_y, factor);
        
        // Invalider le cache d'orbite car le centre a changé
        self.orbit_cache = None;
        
        self.start_render();
    }
    
    /// Zoom sur une zone rectangulaire sélectionnée.
    /// Les coordonnées sont en pixels dans l'image affichée.
    /// L'image affichée représente déjà une zone zoomée définie par center+span.
    fn zoom_to_rectangle(&mut self, rect_min: egui::Pos2, rect_max: egui::Pos2, image_rect: egui::Rect) {
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
        let (xr1, xr2) = if x_ratio1 < x_ratio2 { (x_ratio1, x_ratio2) } else { (x_ratio2, x_ratio1) };
        let (yr1, yr2) = if y_ratio1 < y_ratio2 { (y_ratio1, y_ratio2) } else { (y_ratio2, y_ratio1) };
        
        // Vérifier que le rectangle a une taille minimale
        if (xr2 - xr1) < 0.01 || (yr2 - yr1) < 0.01 {
            return; // Rectangle trop petit (moins de 1% de l'image)
        }
        
        // Utiliser le zoom rectangulaire haute précision
        self.zoom_rect_hp(xr1, yr1, xr2, yr2);
        
        // Invalider le cache d'orbite car le centre a changé
        self.orbit_cache = None;
        
        self.start_render();
    }
    
    /// Dézoom avec un facteur donné.
    fn zoom_out(&mut self, factor: f64) {
        self.zoom_out_hp(factor);
        self.start_render();
    }
    
    /// Dézoom au point spécifié avec un facteur donné.
    fn zoom_out_at_point(&mut self, _point: Complex64, factor: f64) {
        // Le centre reste le même, on élargit juste le span
        self.zoom_out_hp(factor);
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
        self.start_render();
    }

    /// Change le type de fractale.
    /// Réinitialise toujours la position au domaine par défaut (zoom, centre, etc.).
    /// Si on reselectionne le même type, recharge quand même les defaults.
    fn change_fractal_type(&mut self, new_type: FractalType) {
        self.selected_type = new_type;
        let width = self.params.width;
        let height = self.params.height;

        // Obtenir les paramètres par défaut pour le nouveau type
        let mut new_params = default_params_for_type(new_type, width, height);

        // Conserver les paramètres de rendu (GMP, palette, etc.)
        let is_density_type = matches!(
            new_type,
            FractalType::Buddhabrot | FractalType::Nebulabrot | FractalType::AntiBuddhabrot
        );
        new_params.use_gmp = self.params.use_gmp;
        new_params.precision_bits = self.params.precision_bits;
        new_params.color_mode = self.params.color_mode;
        // Fractales densité : toujours 1 par défaut à la sélection (ne pas conserver l’ancienne valeur)
        new_params.color_repeat = if is_density_type {
            1
        } else {
            self.params.color_repeat
        };
        new_params.algorithm_mode = AlgorithmMode::Auto;
        new_params.bla_threshold = self.params.bla_threshold;
        new_params.glitch_tolerance = self.params.glitch_tolerance;

        // Toujours utiliser le domaine par défaut pour bien centrer la fractale
        self.params = new_params;
        self.color_repeat = self.params.color_repeat;
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

        self.start_render();
    }

    fn effective_algorithm_mode(&self) -> AlgorithmMode {
        match self.params.algorithm_mode {
            AlgorithmMode::Auto => {
                if !matches!(
                    self.selected_type,
                    FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip | FractalType::Tricorn
                ) || self.params.width == 0 {
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
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        // Init différée du GPU : une fois la fenêtre eframe prête, éviter "Parent device is lost" sur NVIDIA
        if !self.gpu_init_attempted {
            self.gpu_init_attempted = true;
            if self.gpu_renderer.is_none() {
                self.gpu_renderer = GpuRenderer::new().map(Arc::new);
                if self.gpu_renderer.is_none() {
                    eprintln!("⚠️  GPU non disponible - le rendu GPU sera désactivé (mode CPU uniquement)");
                }
            }
        }

        // Vérifier si le rendu est terminé
        self.check_render_complete(ctx);

        // Vérifier si la preview Julia est terminée
        self.check_julia_preview_complete(ctx);

        // Détection des fichiers déposés (drag-and-drop)
        ctx.input(|i| {
            for file in &i.raw.dropped_files {
                if let Some(path) = &file.path {
                    if path.extension().map(|e| e == "png").unwrap_or(false) {
                        self.load_from_png(path);
                    }
                }
            }
        });

        // Gestion des raccourcis clavier
        ctx.input(|i| {
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
                self.palette_index = (self.palette_index + 1) % 13;
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
                self.color_repeat = if self.color_repeat >= max_repeat { 1 } else { self.color_repeat + 1 };
                self.params.color_repeat = self.color_repeat;
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
                    new_params.use_gmp = self.params.use_gmp;
                    new_params.precision_bits = self.params.precision_bits;
                    new_params.color_mode = self.params.color_mode;
                    new_params.color_repeat = self.params.color_repeat;
                    new_params.algorithm_mode = AlgorithmMode::Auto;
                    self.params = new_params;
                    self.selected_type = julia_type;
                    self.iteration_input = self.params.iteration_max.to_string();
                    self.sync_params_to_hp();
                    self.orbit_cache = None;
                    self.start_render();
                } else if self.selected_type.has_julia_variant() && !self.julia_preview_enabled {
                    // Pas de sélection Julia : activer le mode preview et cocher la case
                    self.julia_preview_enabled = true;
                }
            }

            // S pour screenshot avec métadonnées
            if i.key_pressed(egui::Key::S) {
                use crate::io::png::save_png_with_metadata;
                use std::path::Path;
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let filename = format!("fractal_{}.png", timestamp);
                if let Err(e) = save_png_with_metadata(
                    &self.params,
                    &self.iterations,
                    &self.zs,
                    Path::new(&filename),
                    &self.center_x_hp,
                    &self.center_y_hp,
                    &self.span_x_hp,
                    &self.span_y_hp,
                ) {
                    eprintln!("Erreur export PNG: {}", e);
                } else {
                    println!("Screenshot sauvegardé avec métadonnées: {}", filename);
                }
            }
            
            // Désactiver les raccourcis de zoom en mode Julia preview
            let julia_mode = self.selected_type.has_julia_variant() && self.julia_preview_enabled;

            // + ou = pour zoom au centre (désactivé en mode Julia)
            if !julia_mode {
                let zoom_in_pressed = i.events.iter().any(|e| {
                    matches!(e, egui::Event::Text(text) if text == "+" || text == "=")
                });
                if zoom_in_pressed {
                    let center = Complex64::new(self.params.center_x, self.params.center_y);
                    self.zoom_at_point(center, 1.5);
                }
            }

            // - pour dézoom au centre (désactivé en mode Julia)
            if !julia_mode && i.key_pressed(egui::Key::Minus) {
                let center = Complex64::new(self.params.center_x, self.params.center_y);
                self.zoom_out_at_point(center, 1.5);
            }

            // 0 pour reset zoom (désactivé en mode Julia) — ne pas déclencher si le focus est sur le champ Itérations (pour pouvoir taper "50", "500", etc.)
            if !julia_mode && !self.iteration_input_has_focus && i.key_pressed(egui::Key::Num0) {
                use crate::fractal::default_params_for_type;
                let width = self.params.width;
                let height = self.params.height;
                let mut new_params = default_params_for_type(self.selected_type, width, height);
                // Conserver certains paramètres
                new_params.use_gmp = self.params.use_gmp;
                new_params.precision_bits = self.params.precision_bits;
                new_params.color_mode = self.params.color_mode;
                new_params.color_repeat = self.params.color_repeat;
                self.params = new_params;
                self.iteration_input = self.params.iteration_max.to_string();
                // Synchroniser les coordonnées HP
                self.sync_params_to_hp();
                self.orbit_cache = None;
                self.start_render();
            }
        });
        
        // Panneau de contrôle en haut
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    // Menu Type (toutes catégories)
                    ui.label("Type:");

                    // Menu Type : Mandelbrots à la racine, Julias dans un dossier
                    let density_types = [(16, "Buddhabrot"), (24, "Nebulabrot"), (33, "Anti-Buddhabrot")];

                    let current_category = self.selected_type.menu_family();
                    let current_label = match self.selected_type {
                        FractalType::Lyapunov => self.selected_lyapunov_preset.name(),
                        _ => self.selected_type.name(),
                    };

                    let type_menu_label = if current_category == current_label {
                        format!("▼ {}", current_label)
                    } else {
                        format!("▼ {}: {}", current_category, current_label)
                    };
                    ui.menu_button(&type_menu_label, |ui| {
                        // Nova tout en haut
                        if let Some(fractal_type) = FractalType::from_id(22) {
                            if ui.selectable_label(self.selected_type == fractal_type, "Nova").clicked() {
                                self.change_fractal_type(fractal_type);
                                ui.close_menu();
                            }
                        }

                        // Lyapunov et Densité
                        ui.menu_button("Lyapunov", |ui| {
                            for preset in LyapunovPreset::all() {
                                let is_selected = self.selected_type == FractalType::Lyapunov
                                    && self.selected_lyapunov_preset == *preset;
                                if ui.selectable_label(is_selected, preset.name()).clicked() {
                                    self.change_lyapunov_preset(*preset);
                                    ui.close_menu();
                                }
                            }
                        });

                        ui.menu_button("Densité", |ui| {
                            for (id, label) in density_types.iter() {
                                if let Some(fractal_type) = FractalType::from_id(*id) {
                                    if ui.selectable_label(self.selected_type == fractal_type, *label).clicked() {
                                        self.change_fractal_type(fractal_type);
                                        ui.close_menu();
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
                                    ui.close_menu();
                                }
                            }
                        }

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
                                        ui.close_menu();
                                    }
                                }
                            }
                        });

                        ui.separator();

                        // Mandelbulb à la racine
                        if let Some(fractal_type) = FractalType::from_id(15) {
                            if ui.selectable_label(self.selected_type == fractal_type, "Mandelbulb").clicked() {
                                self.change_fractal_type(fractal_type);
                                ui.close_menu();
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
                                    ui.close_menu();
                                }
                            }
                        }

                    });

                    ui.separator();

                    // Plane (XaoS-style) uniquement pour fractales escape-time
                    let is_escape_time = !matches!(
                        self.selected_type,
                        FractalType::VonKoch | FractalType::Dragon | FractalType::Buddhabrot | FractalType::Lyapunov | FractalType::Nebulabrot | FractalType::AntiBuddhabrot
                    );
                    
                    if is_escape_time {
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
                            if self.params.plane_transform != PlaneTransform::Mu
                                && self.params.algorithm_mode == AlgorithmMode::Perturbation
                            {
                                self.params.algorithm_mode = AlgorithmMode::Auto;
                            }
                            self.start_render();
                        }
                    }

                    ui.separator();

                    // Checkbox Julia mode (pour les types Mandelbrot-like avec variante Julia)
                    if self.selected_type.has_julia_variant() {
                        ui.separator();
                        let old_enabled = self.julia_preview_enabled;
                        ui.checkbox(&mut self.julia_preview_enabled, "Julia");
                        if old_enabled && !self.julia_preview_enabled {
                            // Nettoyer quand désactivé
                            self.julia_preview_texture = None;
                            self.julia_preview_last_seed = None;
                            self.julia_preview_cancel.store(true, Ordering::Relaxed);
                        }
                    }

                    ui.separator();

                    ui.label("Iter:");
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut self.iteration_input)
                            .desired_width(60.0)
                    );
                    self.iteration_input_has_focus = response.has_focus();
                    // Appliquer la valeur à la perte de focus ou sur Entrée
                    if response.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        if let Ok(val) = self.iteration_input.trim().parse::<u32>() {
                            let val = val.max(1);
                            if val != self.params.iteration_max {
                                self.params.iteration_max = val;
                                self.iteration_input = val.to_string();
                                self.orbit_cache = None;
                                self.start_render();
                            }
                        } else {
                            self.iteration_input = self.params.iteration_max.to_string();
                        }
                    }

                    ui.separator();

                    // Section Tech (entre Iter et Render)
                    let supports_advanced_modes = matches!(
                        self.selected_type,
                        FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip
                    );
                    let is_lyapunov = self.selected_type == FractalType::Lyapunov;

                    if supports_advanced_modes || is_lyapunov {
                        ui.label("Tech:");
                        let gpu_available = self.gpu_renderer.is_some();
                        let old_use_gpu = self.use_gpu;
                        let old_mode = self.params.algorithm_mode;

                        let render_text = if is_lyapunov {
                            match self.params.algorithm_mode {
                                AlgorithmMode::Auto => "🔄 Auto".to_string(),
                                AlgorithmMode::StandardF64 => "💻 CPU Standard f64".to_string(),
                                _ => "🔄 Auto".to_string(),
                            }
                        } else {
                            match (self.use_gpu && gpu_available, self.params.algorithm_mode) {
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
                                self.params.algorithm_mode == AlgorithmMode::Auto,
                                "🔄 Auto"
                            ).clicked() {
                                self.params.algorithm_mode = AlgorithmMode::Auto;
                                if is_lyapunov {
                                    self.use_gpu = false;
                                }
                                ui.close_menu();
                            }

                            if is_lyapunov {
                                if ui.selectable_label(
                                    !self.use_gpu && self.params.algorithm_mode == AlgorithmMode::StandardF64,
                                    "💻 CPU Standard f64"
                                ).clicked() {
                                    self.use_gpu = false;
                                    self.params.algorithm_mode = AlgorithmMode::StandardF64;
                                    ui.close_menu();
                                }
                            } else {
                                ui.separator();

                                ui.menu_button("💻 CPU", |ui| {
                                    if ui.selectable_label(
                                        !self.use_gpu && self.params.algorithm_mode == AlgorithmMode::StandardF64,
                                        "📊 Standard f64"
                                    ).clicked() {
                                        self.use_gpu = false;
                                        self.params.algorithm_mode = AlgorithmMode::StandardF64;
                                        ui.close_menu();
                                    }

                                    if ui.selectable_label(
                                        !self.use_gpu && self.params.algorithm_mode == AlgorithmMode::ReferenceGmp,
                                        "🔢 GMP Reference"
                                    ).clicked() {
                                        self.use_gpu = false;
                                        self.params.algorithm_mode = AlgorithmMode::ReferenceGmp;
                                        ui.close_menu();
                                    }

                                    let plane_ok = self.params.plane_transform == PlaneTransform::Mu;
                                    if ui
                                        .add_enabled(
                                            plane_ok,
                                            egui::SelectableLabel::new(
                                                !self.use_gpu && self.params.algorithm_mode == AlgorithmMode::Perturbation,
                                                "🔬 Perturbation f64",
                                            ),
                                        )
                                        .clicked()
                                    {
                                        self.use_gpu = false;
                                        self.params.algorithm_mode = AlgorithmMode::Perturbation;
                                        ui.close_menu();
                                    }
                                });

                                if gpu_available {
                                    ui.menu_button("🎮 GPU", |ui| {
                                        if ui.selectable_label(
                                            self.use_gpu && self.params.algorithm_mode == AlgorithmMode::StandardF64,
                                            "⚡ Standard f32"
                                        ).clicked() {
                                            self.use_gpu = true;
                                            self.params.algorithm_mode = AlgorithmMode::StandardF64;
                                            ui.close_menu();
                                        }

                                        let plane_ok = self.params.plane_transform == PlaneTransform::Mu;
                                        if ui
                                            .add_enabled(
                                                plane_ok,
                                                egui::SelectableLabel::new(
                                                    self.use_gpu && self.params.algorithm_mode == AlgorithmMode::Perturbation,
                                                    "🚀 Perturbation f32",
                                                ),
                                            )
                                            .clicked()
                                        {
                                            self.use_gpu = true;
                                            self.params.algorithm_mode = AlgorithmMode::Perturbation;
                                            ui.close_menu();
                                        }
                                    });
                                } else {
                                    ui.add_enabled(false, egui::Label::new("🎮 GPU (Non disponible)"));
                                }
                            }
                        });

                        if supports_advanced_modes {
                            if old_use_gpu != self.use_gpu {
                                if self.use_gpu && self.params.algorithm_mode == AlgorithmMode::ReferenceGmp {
                                    self.params.algorithm_mode = AlgorithmMode::Auto;
                                }
                                self.orbit_cache = None;
                                self.start_render();
                            }

                            if self.params.plane_transform != PlaneTransform::Mu
                                && self.params.algorithm_mode == AlgorithmMode::Perturbation
                            {
                                self.params.algorithm_mode = AlgorithmMode::Auto;
                            }

                            if old_mode != self.params.algorithm_mode {
                                self.orbit_cache = None;
                                self.start_render();
                            }
                        } else if is_lyapunov && old_mode != self.params.algorithm_mode {
                            self.start_render();
                        }
                    }

                    ui.separator();
                    if ui.button("🎬 Render").clicked() {
                        self.show_render_dialog = true;
                    }
                });
                
                ui.separator();
                
                ui.horizontal(|ui| {
                    ui.label("Palette:");
                    if ui.button("<").clicked() {
                        self.palette_index = if self.palette_index == 0 { 12 } else { self.palette_index - 1 };
                        self.params.color_mode = self.palette_index;
                        if !self.iterations.is_empty() {
                            self.update_texture(ctx);
                        }
                    }
                    
                    // Afficher la prévisualisation de la palette
                    let palette_idx = self.palette_index as usize;
                    if self.palette_preview_textures[palette_idx].is_none() {
                        let preview_image = generate_palette_preview(self.palette_index, 100, 12, self.params.color_space);
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
                        self.palette_index = (self.palette_index + 1) % 13; // 13 palettes maintenant (0-12)
                        self.params.color_mode = self.palette_index;
                        if !self.iterations.is_empty() {
                            self.update_texture(ctx);
                        }
                    }
                    
                    ui.separator();

                    ui.label("Outcoloring:");
                    let old_out_mode = self.out_coloring_mode;
                    egui::ComboBox::from_id_salt("outcoloring_mode")
                        .selected_text(self.out_coloring_mode.name())
                        .show_ui(ui, |ui| {
                            for mode in OutColoringMode::menu_modes() {
                                ui.selectable_value(&mut self.out_coloring_mode, *mode, mode.name());
                            }
                        });
                    if old_out_mode != self.out_coloring_mode {
                        self.params.out_coloring_mode = self.out_coloring_mode;
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

                    ui.label("Color Repeat:");
                    let is_density_type = matches!(
                        self.selected_type,
                        FractalType::Buddhabrot | FractalType::Nebulabrot | FractalType::AntiBuddhabrot
                    );
                    let (min_repeat, max_repeat) = if is_density_type { (1, 8) } else { (1, 120) };
                    if is_density_type && self.color_repeat > max_repeat {
                        self.color_repeat = max_repeat;
                        self.params.color_repeat = self.color_repeat;
                    }
                    let old_repeat = self.color_repeat;
                    ui.add(egui::Slider::new(&mut self.color_repeat, min_repeat..=max_repeat));
                    if old_repeat != self.color_repeat {
                        self.params.color_repeat = self.color_repeat;
                        if !self.iterations.is_empty() {
                            self.update_texture(ctx);
                        }
                    }

                    ui.separator();

                });
        });

        // Traiter les messages du rendu HQ en cours
        if let Some(ref receiver) = self.hq_render_receiver {
            while let Ok(msg) = receiver.try_recv() {
                match msg {
                    HqRenderMessage::Progress(p) => {
                        self.hq_render_progress = p;
                    }
                    HqRenderMessage::Done(filename) => {
                        self.hq_rendering = false;
                        self.hq_render_progress = 1.0;
                        self.hq_render_result = Some(format!("✓ Saved: {}", filename));
                    }
                    HqRenderMessage::Error(err) => {
                        self.hq_rendering = false;
                        self.hq_render_result = Some(format!("✗ {}", err));
                    }
                }
            }
        }

        // Fenêtre de configuration du rendu haute résolution
        if self.show_render_dialog {
            egui::Window::new("🎬 High Quality Render")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    if self.hq_rendering {
                        // Afficher la barre de progression pendant le rendu
                        let resolution_label = match self.render_resolution_preset {
                            RenderResolutionPreset::Window => format!("{}×{}", self.window_width, self.window_height),
                            RenderResolutionPreset::Res4K => "4K".to_string(),
                            RenderResolutionPreset::Res8K => "8K".to_string(),
                        };
                        ui.heading(format!("Rendering with {} precision...", resolution_label));
                        ui.add_space(10.0);

                        let progress_bar = egui::ProgressBar::new(self.hq_render_progress)
                            .show_percentage()
                            .animate(true);
                        ui.add(progress_bar);

                        ui.add_space(10.0);
                        ui.label("Please wait...");

                        // Demander un repaint pour mettre à jour la barre
                        ctx.request_repaint();
                    } else if let Some(ref result) = self.hq_render_result {
                        // Afficher le résultat
                        ui.heading("Complete");
                        ui.add_space(10.0);
                        ui.label(result.as_str());
                        ui.add_space(20.0);

                        if ui.button("Close").clicked() {
                            self.show_render_dialog = false;
                            self.hq_render_result = None;
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
                                format!("Window ({}×{})", self.window_width, self.window_height)
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.selectable_value(&mut self.render_resolution_preset, RenderResolutionPreset::Res4K, "4K (3840×2160)");
                        });
                        ui.horizontal(|ui| {
                            ui.selectable_value(&mut self.render_resolution_preset, RenderResolutionPreset::Res8K, "8K (7680×4320)");
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

        // Interface principale - zone d'affichage de la fractale
        // Désactiver la sélection de texte pour éviter que le curseur devienne un curseur de texte
        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(egui::Color32::TRANSPARENT))
            .show(ctx, |ui| {
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
                    // Désactiver la sélection de texte pour forcer le curseur à rester une flèche
                    let response = ui.add(
                        egui::Image::new(texture)
                            .fit_to_exact_size(display_size)
                            .sense(egui::Sense::click_and_drag())
                    );
                    let image_rect = response.rect;
                    
                    // Forcer le curseur à être une flèche quand on survole l'image
                    // IMPORTANT: Toujours afficher une flèche (Default) et non un curseur de texte ou autre
                    if response.hovered() {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::Default);

                        // Track mouse position for Julia preview (on Mandelbrot-like types)
                        if self.selected_type.has_julia_variant()
                            && self.julia_preview_enabled
                            && !self.selecting
                        {
                            if let Some(pos) = ui.ctx().pointer_hover_pos() {
                                if image_rect.contains(pos) {
                                    let local_pos = pos - image_rect.min;
                                    let pixel_x = (local_pos.x / image_rect.width()) * self.params.width as f32;
                                    let pixel_y = (local_pos.y / image_rect.height()) * self.params.height as f32;
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
                    let julia_mode = self.selected_type.has_julia_variant() && self.julia_preview_enabled;
                    if !julia_mode {
                        ctx.input(|i| {
                            // Vérifier si le bouton gauche est pressé et si on est dans la zone de l'image
                            if i.pointer.primary_down() {
                                if let Some(pointer_pos) = i.pointer.interact_pos() {
                                    if image_rect.contains(pointer_pos) {
                                        if !self.selecting {
                                            // Commencer une nouvelle sélection
                                            self.selecting = true;
                                            self.select_start = Some(pointer_pos);
                                            self.select_current = Some(pointer_pos);
                                        } else {
                                            // Mettre à jour la position actuelle
                                            let clamped_pos = egui::Pos2::new(
                                                pointer_pos.x.max(image_rect.min.x).min(image_rect.max.x),
                                                pointer_pos.y.max(image_rect.min.y).min(image_rect.max.y),
                                            );
                                            self.select_current = Some(clamped_pos);
                                        }
                                    }
                                }
                            } else if self.selecting && i.pointer.primary_released() {
                                // Le bouton a été relâché, terminer la sélection
                                if let (Some(start), Some(current)) = (self.select_start, self.select_current) {
                                    let rect_min = egui::Pos2::new(
                                        start.x.min(current.x),
                                        start.y.min(current.y),
                                    );
                                    let rect_max = egui::Pos2::new(
                                        start.x.max(current.x),
                                        start.y.max(current.y),
                                    );

                                    // Vérifier que le rectangle est dans l'image et a une taille minimale
                                    if image_rect.contains(rect_min) && image_rect.contains(rect_max) {
                                        let width = rect_max.x - rect_min.x;
                                        let height = rect_max.y - rect_min.y;

                                        if width > 5.0 && height > 5.0 {
                                            self.zoom_to_rectangle(rect_min, rect_max, image_rect);
                                        }
                                    }
                                }

                                self.selecting = false;
                                self.select_start = None;
                                self.select_current = None;
                            }
                        });
                    }
                    
                    // Dessiner le rectangle de sélection par-dessus l'image
                    if self.selecting {
                        if let (Some(start), Some(current)) = (self.select_start, self.select_current) {
                            let rect = egui::Rect::from_two_pos(start, current);
                            
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
                                egui::Stroke::new(3.0, egui::Color32::from_rgb(255, 255, 0)),
                            );
                            // Rectangle intérieur pour meilleure visibilité
                            ui.painter().rect_stroke(
                                rect.expand(-1.0),
                                0.0,
                                egui::Stroke::new(1.0, egui::Color32::from_rgb(0, 0, 0)), // Bordure noire intérieure
                            );
                            
                            // Demander un re-rendu pour mettre à jour le rectangle en temps réel
                            ctx.request_repaint();
                        }
                    }
                    
                    // Clic simple (sans drag) : zoom au point
                    // Seulement si on n'a pas fait de sélection et pas en mode Julia
                    if response.clicked() && !self.selecting && !julia_mode {
                        if let Some(pos) = response.interact_pointer_pos() {
                            let local_pos = pos - image_rect.min;
                            let pixel_x = (local_pos.x / image_rect.width()) * self.params.width as f32;
                            let pixel_y = (local_pos.y / image_rect.height()) * self.params.height as f32;
                            let point = self.pixel_to_complex(pixel_x, pixel_y, self.params.width as f32, self.params.height as f32);
                            self.zoom_at_point(point, 2.0);
                        }
                    }

                    // Clic droit : dézoom centré sur la position de la souris (désactivé en mode Julia)
                    if !julia_mode {
                        if response.secondary_clicked() {
                            if let Some(pos) = response.interact_pointer_pos() {
                                let local_pos = pos - image_rect.min;
                                let pixel_x = (local_pos.x / image_rect.width()) * self.params.width as f32;
                                let pixel_y = (local_pos.y / image_rect.height()) * self.params.height as f32;
                                let point = self.pixel_to_complex(pixel_x, pixel_y, self.params.width as f32, self.params.height as f32);
                                self.zoom_out_at_point(point, 2.0);
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
                                            let pixel_x = (local_pos.x / image_rect.width()) * self.params.width as f32;
                                            let pixel_y = (local_pos.y / image_rect.height()) * self.params.height as f32;
                                            let point = self.pixel_to_complex(pixel_x, pixel_y, self.params.width as f32, self.params.height as f32);
                                            self.zoom_out_at_point(point, 2.0);
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
                        let overlay_rect = egui::Rect::from_min_size(overlay_pos, egui::Vec2::new(preview_width, preview_height));

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
                            egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 100, 100)),
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
                                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
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
                                egui::Pos2::new(overlay_pos.x + preview_width / 2.0, overlay_pos.y + preview_height - 20.0),
                                egui::Align2::CENTER_CENTER,
                                text,
                                egui::FontId::proportional(11.0),
                                egui::Color32::WHITE,
                            );
                        }
                        ui.painter().text(
                            egui::Pos2::new(overlay_pos.x + preview_width / 2.0, overlay_pos.y + preview_height - 8.0),
                            egui::Align2::CENTER_CENTER,
                            "Press J to switch",
                            egui::FontId::proportional(10.0),
                            egui::Color32::GRAY,
                        );
                    }
                } else if self.rendering {
                    // Pas encore de texture, afficher le spinner
                    ui.spinner();
                    ui.label("Rendu en cours...");
                } else {
                    // Premier rendu
                    self.start_render();
                }
            });
        });
        
        // Barre d'état en bas
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    // Centre, itérations, zoom
                    let center_text = if self.params.center_x.abs() < 1e-6 && self.params.center_y.abs() < 1e-6 {
                        format!("Centre: ({:.6}, {:.6})", self.params.center_x, self.params.center_y)
                    } else if self.params.center_x.abs() > 1e3 || self.params.center_y.abs() > 1e3 {
                        format!("Centre: ({:.2e}, {:.2e})", self.params.center_x, self.params.center_y)
                    } else {
                        format!("Centre: ({:.6}, {:.6})", self.params.center_x, self.params.center_y)
                    };
                    ui.label(center_text);
                    ui.separator();
                    ui.label(format!("Iter.: {}", self.params.iteration_max));
                    ui.separator();
                    let base_range = 4.0;
                    let pixel_size = if self.params.width > 0 && self.params.height > 0 {
                        self.params.span_x.abs().max(self.params.span_y.abs()) / self.params.width as f64
                    } else {
                        0.0
                    };
                    let zoom = if pixel_size > 0.0 && pixel_size.is_finite() {
                        base_range / pixel_size
                    } else {
                        1.0
                    };
                    let zoom_text = if zoom >= 1e6 {
                        format!("Zoom: {:.2e}", zoom)
                    } else {
                        format!("Zoom: {:.2}", zoom)
                    };
                    ui.label(zoom_text);
                    ui.separator();

                    // ═══════════════════════════════════════════════════════════════
                    // Affichage du mode de calcul effectif
                    // Format: [Device] [Precision] [Algorithme]
                    // Ex: "GPU f32 Perturbation" ou "CPU f64 Standard"
                    // ═══════════════════════════════════════════════════════════════
                    
                    let effective_mode = self.effective_algorithm_mode();
                    let gpu_active = self.use_gpu && self.gpu_renderer.is_some();
                    
                    let mode_display = if let (Some(device_label), Some(method_label)) =
                        (&self.last_render_device_label, &self.last_render_method_label)
                    {
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
                            let effective_prec = crate::fractal::perturbation::compute_perturbation_precision_bits(&self.params);
                            format!("{} GMP {}b", device, effective_prec)
                        } else {
                            format!("{} {} {}", device, precision, algo)
                        }
                    };
                    
                    ui.label(mode_display);

                    // Afficher la précision GMP effective pour le mode perturbation
                    if effective_mode == AlgorithmMode::Perturbation {
                        let effective_prec = crate::fractal::perturbation::compute_perturbation_precision_bits(&self.params);
                        ui.separator();
                        ui.label(format!("GMP: {}b", effective_prec));
                    }

                    // Afficher le statut du rendu progressif + barre de progression
                    if self.rendering {
                        ui.separator();
                        let total = self.total_passes.max(1) as f32;
                        let progress = (self.current_pass as f32 / total).min(1.0);
                        let progress_bar = egui::ProgressBar::new(progress)
                            .show_percentage()
                            .animate(true);
                        ui.add(progress_bar);
                        if let Some(start) = self.render_start_time {
                            let elapsed = start.elapsed().as_secs_f64();
                            let time_str = if elapsed < 60.0 {
                                format!("{:.1}s", elapsed)
                            } else {
                                let mins = (elapsed / 60.0).floor() as u32;
                                let secs = elapsed % 60.0;
                                format!("{}m {:.0}s", mins, secs)
                            };
                            ui.label(format!("Calcul en cours... ({}) - Passe {}/{}", time_str, self.current_pass, self.total_passes));
                        } else {
                            ui.label(format!("Calcul en cours... - Passe {}/{}", self.current_pass, self.total_passes));
                        }
                    } else if self.is_preview {
                        ui.separator();
                        ui.label("Preview");
                    }

                    // ═══════════════════════════════════════════════════════════════
                    // Temps de rendu aligné à droite
                    // ═══════════════════════════════════════════════════════════════
                    
                    // Espace flexible pour pousser le temps à droite
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if let Some(render_time) = self.last_render_time {
                            let time_str = if render_time < 1.0 {
                                format!("{:.0} ms", render_time * 1000.0)
                            } else if render_time < 60.0 {
                                format!("{:.2} s", render_time)
                            } else {
                                let mins = (render_time / 60.0).floor() as u32;
                                let secs = render_time % 60.0;
                                format!("{}m {:.1}s", mins, secs)
                            };
                            ui.label(format!("⏱ {}", time_str));
                        } else if self.rendering {
                            if let Some(start) = self.render_start_time {
                                let elapsed = start.elapsed().as_secs_f64();
                                let time_str = if elapsed < 1.0 {
                                    format!("{:.0} ms", elapsed * 1000.0)
                                } else {
                                    format!("{:.1} s", elapsed)
                                };
                                ui.label(format!("⏱ {}...", time_str));
                            }
                        }
                    });
                });
        });
        
        // Demander un re-rendu seulement si un rendu est en cours
        if self.rendering || self.hq_rendering {
            ctx.request_repaint();
        }
    }
}
