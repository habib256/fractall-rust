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

/// Application principale egui pour fractall.
pub struct FractallApp {
    // État fractale
    params: FractalParams,
    iterations: Vec<u32>,
    zs: Vec<Complex64>,
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
    render_preset: RenderPreset,
    gpu_renderer: Option<Arc<GpuRenderer>>,
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
}

impl FractallApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let default_type = FractalType::Mandelbrot;
        let width = 800;
        let height = 600;
        let mut params = default_params_for_type(default_type, width, height);
        apply_default_preset(&mut params, RenderPreset::Standard);
        
        // Initialiser le GPU renderer (peut échouer silencieusement si GPU indisponible)
        let gpu_renderer = GpuRenderer::new().map(Arc::new);
        // Par défaut, rester en mode CPU même si le GPU est disponible.
        let use_gpu_default = false;
        if gpu_renderer.is_none() {
            eprintln!("⚠️  GPU non disponible - le rendu GPU sera désactivé");
            eprintln!("   L'application fonctionnera en mode CPU uniquement");
        }
        
        Self {
            params: params.clone(),
            iterations: Vec::new(),
            zs: Vec::new(),
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
            render_preset: RenderPreset::Standard,
            gpu_renderer,
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

        // Paramètres pour le thread
        let params = self.params.clone();
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
                            // Use cache-aware GPU rendering for perturbation
                            gpu.render_perturbation_with_cache(&pass_params, &cancel, reuse, current_orbit_cache.as_ref())
                                .map(|(result, cache)| {
                                    current_orbit_cache = Some(cache);
                                    result
                                })
                        }
                        FractalType::Mandelbrot => gpu.render_mandelbrot(&pass_params, &cancel),
                        FractalType::Julia => gpu.render_julia(&pass_params, &cancel),
                        FractalType::BurningShip => gpu.render_burning_ship(&pass_params, &cancel),
                        _ => None,
                    };
                    if let Some(result) = gpu_result {
                        let base_precision = gpu.precision_label();
                        let precision = base_precision.to_string();
                        let effective_mode = if use_perturbation {
                            AlgorithmMode::Perturbation
                        } else {
                            AlgorithmMode::StandardF64
                        };
                        Some((
                            result,
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
                                .map(|(result, cache)| {
                                    current_orbit_cache = Some(cache);
                                    (result, AlgorithmMode::Perturbation, "CPU f64".to_string())
                                })
                        } else {
                            let cpu_result =
                                render_escape_time_cancellable_with_reuse(&pass_params, &cancel, reuse);
                            cpu_result.map(|r| {
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
                                (r, effective_mode, precision_label)
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
                            .map(|(result, cache)| {
                                current_orbit_cache = Some(cache);
                                (result, AlgorithmMode::Perturbation, "CPU f64".to_string())
                            })
                    } else {
                        render_escape_time_cancellable_with_reuse(&pass_params, &cancel, reuse).map(|r| {
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
                            (r, effective_mode, precision_label)
                        })
                    }
                };

                match result {
                    Some(((iterations, zs), effective_mode, precision_label)) => {
                        // Garder une copie pour la passe suivante afin d'éviter le recalcul
                        if pass_index + 1 < config.passes.len() {
                            previous_pass = Some((iterations.clone(), zs.clone(), pass_width, pass_height));
                        } else {
                            previous_pass = None;
                        }

                        let _ = sender.send(RenderMessage::PassComplete {
                            pass_index: pass_index as u8,
                            scale_divisor,
                            effective_mode,
                            precision_label,
                            iterations,
                            zs,
                            width: pass_width,
                            height: pass_height,
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
    fn check_render_complete(&mut self, ctx: &Context) {
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
                width,
                height,
            } => {
                self.current_pass = pass_index + 1;
                self.last_render_device_label = Some(precision_label);
                // Note: pour GMP, le label est déjà dans precision_label, pas besoin de répéter
                self.last_render_method_label = Some(match effective_mode {
                    AlgorithmMode::ReferenceGmp => String::new(), // Évite "CPU GMP 160b GMP"
                    AlgorithmMode::Perturbation => "Perturbation".to_string(),
                    _ => "Standard".to_string(),
                });

                // Upscale si pas à pleine résolution
                if scale_divisor > 1 {
                    let (upscaled_iter, upscaled_zs) = upscale_nearest(
                        &iterations,
                        &zs,
                        width,
                        height,
                        self.params.width,
                        self.params.height,
                    );
                    self.iterations = upscaled_iter;
                    self.zs = upscaled_zs;
                    self.is_preview = true;
                } else {
                    self.iterations = iterations;
                    self.zs = zs;
                    self.is_preview = false;
                }

                self.update_texture(ctx);
                ctx.request_repaint();
            }

            RenderMessage::AllComplete { orbit_cache } => {
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
                // Garder la dernière texture valide affichée
            }
        }
    }

    /// Planifie ou applique un redimensionnement de la surface de rendu.
    fn queue_resize(&mut self, new_width: u32, new_height: u32) {
        if new_width == 0 || new_height == 0 {
            return;
        }
        if new_width == self.window_width && new_height == self.window_height {
            return;
        }
        if self.rendering {
            self.pending_resize = Some((new_width, new_height));
        } else {
            self.apply_resize(new_width, new_height);
        }
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
        self.iterations.clear();
        self.zs.clear();
        self.orbits.clear();
        self.texture = None;
        self.start_render();
    }
    
    /// Met à jour la texture egui à partir des données de fractale.
    fn update_texture(&mut self, ctx: &Context) {
        let width = self.params.width;
        let height = self.params.height;
        let w = width as usize;

        if self.iterations.is_empty() || self.zs.is_empty() {
            return;
        }
        
        // S'assurer que orbits a la même taille que iterations/zs
        while self.orbits.len() < self.iterations.len() {
            self.orbits.push(None);
        }

        let is_nebulabrot = self.params.fractal_type == FractalType::Nebulabrot;
        let is_buddhabrot = self.params.fractal_type == FractalType::Buddhabrot;

        // Créer l'image RGB avec colorisation parallélisée
        use rayon::prelude::*;
        let iterations = &self.iterations;
        let zs = &self.zs;
        let iter_max = self.params.iteration_max;
        let palette_idx = self.palette_index;
        let color_rep = self.color_repeat;
        let out_mode = self.out_coloring_mode;

        let buffer: Vec<u8> = (0..height as usize)
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
                            color_for_buddhabrot_pixel(z, palette_idx, color_rep)
                        } else {
                            let orbit = self.orbits.get(idx).and_then(|o| o.as_ref());
                            color_for_pixel(
                                iter,
                                z,
                                iter_max,
                                palette_idx,
                                color_rep,
                                out_mode,
                                self.params.color_space,
                                orbit,
                            )
                        };

                        vec![r, g, b]
                    })
                    .collect::<Vec<u8>>()
            })
            .collect();
        
        let img = RgbImage::from_raw(width, height, buffer).unwrap();
        
        // Convertir en texture egui
        let color_image = rgb_image_to_color_image(&img);
        self.texture = Some(ctx.load_texture(
            "fractal",
            color_image,
            TextureOptions::LINEAR
        ));
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
    /// Réinitialise toujours la position au domaine par défaut de la nouvelle fractale.
    fn change_fractal_type(&mut self, new_type: FractalType) {
        if new_type == self.selected_type {
            return;
        }

        self.selected_type = new_type;
        let width = self.params.width;
        let height = self.params.height;

        // Obtenir les paramètres par défaut pour le nouveau type
        let mut new_params = default_params_for_type(new_type, width, height);

        // Conserver les paramètres de rendu (GMP, palette, etc.)
        new_params.use_gmp = self.params.use_gmp;
        new_params.precision_bits = self.params.precision_bits;
        new_params.color_mode = self.params.color_mode;
        new_params.color_repeat = self.params.color_repeat;
        new_params.algorithm_mode = AlgorithmMode::Auto;
        new_params.bla_threshold = self.params.bla_threshold;
        new_params.glitch_tolerance = self.params.glitch_tolerance;

        // Toujours utiliser le domaine par défaut pour bien centrer la fractale
        self.params = new_params;
        if self.selected_type == FractalType::Mandelbrot {
            self.apply_render_preset(self.render_preset);
        }
        // Synchroniser les coordonnées HP depuis les nouvelles params
        self.sync_params_to_hp();
        self.use_gpu = false;
        // Invalidate orbit cache when changing fractal type
        self.orbit_cache = None;
        self.start_render();
    }

    fn apply_render_preset(&mut self, preset: RenderPreset) {
        self.render_preset = preset;
        match preset {
            RenderPreset::Standard => {
                self.params.algorithm_mode = AlgorithmMode::Auto;
                self.params.bla_threshold = 1e-6;
                self.params.glitch_tolerance = 1e-4;
                self.params.precision_bits = 256;
            self.params.series_order = 2;
            self.params.series_threshold = 1e-6;
            self.params.series_error_tolerance = 1e-9;
            self.params.glitch_neighbor_pass = true;
            }
            RenderPreset::Fast => {
                self.params.algorithm_mode = AlgorithmMode::Auto;
                self.params.bla_threshold = 5e-6;
                self.params.glitch_tolerance = 5e-4;
                self.params.precision_bits = 192;
            self.params.series_order = 2;
            self.params.series_threshold = 3e-6;
            self.params.series_error_tolerance = 5e-9;
            self.params.glitch_neighbor_pass = true;
            }
            RenderPreset::Ultra => {
                self.params.algorithm_mode = AlgorithmMode::Auto;
                self.params.bla_threshold = 1e-5;
                self.params.glitch_tolerance = 1e-3;
                self.params.precision_bits = 160;
            self.params.series_order = 2;
            self.params.series_threshold = 1e-5;
            self.params.series_error_tolerance = 1e-8;
            self.params.glitch_neighbor_pass = true;
            }
        }
    }

    fn effective_algorithm_mode(&self) -> AlgorithmMode {
        match self.params.algorithm_mode {
            AlgorithmMode::Auto => {
                if !matches!(
                    self.selected_type,
                    FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RenderPreset {
    Standard,
    Fast,
    Ultra,
}

impl RenderPreset {
    fn name(self) -> &'static str {
        match self {
            RenderPreset::Standard => "Standard (Légère)",
            RenderPreset::Fast => "Fast (Modérée)",
            RenderPreset::Ultra => "Ultra (Agressive)",
        }
    }
}

fn apply_default_preset(params: &mut FractalParams, preset: RenderPreset) {
    match preset {
        RenderPreset::Standard => {
            params.algorithm_mode = AlgorithmMode::Auto;
            params.bla_threshold = 1e-6;
            params.glitch_tolerance = 1e-4;
            params.precision_bits = 256;
            params.series_order = 2;
            params.series_threshold = 1e-6;
            params.series_error_tolerance = 1e-9;
            params.glitch_neighbor_pass = true;
        }
        RenderPreset::Fast => {
            params.algorithm_mode = AlgorithmMode::Auto;
            params.bla_threshold = 5e-6;
            params.glitch_tolerance = 5e-4;
            params.precision_bits = 192;
            params.series_order = 2;
            params.series_threshold = 3e-6;
            params.series_error_tolerance = 5e-9;
            params.glitch_neighbor_pass = true;
        }
        RenderPreset::Ultra => {
            params.algorithm_mode = AlgorithmMode::Auto;
            params.bla_threshold = 1e-5;
            params.glitch_tolerance = 1e-3;
            params.precision_bits = 160;
            params.series_order = 2;
            params.series_threshold = 1e-5;
            params.series_error_tolerance = 1e-8;
            params.glitch_neighbor_pass = true;
        }
    }
}


impl eframe::App for FractallApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        // Vérifier si le rendu est terminé
        self.check_render_complete(ctx);
        
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
            
            // R pour color_repeat (1-60, par pas de 1)
            if i.key_pressed(egui::Key::R) {
                self.color_repeat = if self.color_repeat >= 60 { 1 } else { self.color_repeat + 1 };
                self.params.color_repeat = self.color_repeat;
                if !self.iterations.is_empty() {
                    self.update_texture(ctx);
                }
            }
            
            // S pour screenshot
            if i.key_pressed(egui::Key::S) {
                use crate::io::png::save_png;
                use std::path::Path;
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                let filename = format!("fractal_{}.png", timestamp);
                if let Err(e) = save_png(&self.params, &self.iterations, &self.zs, Path::new(&filename)) {
                    eprintln!("Erreur export PNG: {}", e);
                } else {
                    println!("Screenshot sauvegardé: {}", filename);
                }
            }
            
            // + ou = pour zoom au centre
            let zoom_in_pressed = i.events.iter().any(|e| {
                matches!(e, egui::Event::Text(text) if text == "+" || text == "=")
            });
            if zoom_in_pressed {
                let center = Complex64::new(self.params.center_x, self.params.center_y);
                self.zoom_at_point(center, 1.5);
            }
            
            // - pour dézoom au centre
            if i.key_pressed(egui::Key::Minus) {
                let center = Complex64::new(self.params.center_x, self.params.center_y);
                self.zoom_out_at_point(center, 1.0 / 1.5);
            }
            
            // 0 pour reset zoom
            if i.key_pressed(egui::Key::Num0) {
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
                // Synchroniser les coordonnées HP
                self.sync_params_to_hp();
                self.orbit_cache = None;
                self.start_render();
            }
        });
        
        // Panneau de contrôle en haut
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    // Section technique (CPU / GPU en premier)
                    let supports_advanced_modes = matches!(
                        self.selected_type,
                        FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip
                    );
                    if supports_advanced_modes {
                        ui.label("Tech:");
                        let gpu_available = self.gpu_renderer.is_some();
                        let old_use_gpu = self.use_gpu;
                        let old_mode = self.params.algorithm_mode;

                        let render_text = match (self.use_gpu && gpu_available, self.params.algorithm_mode) {
                            (_, AlgorithmMode::Auto) => "🔄 Auto".to_string(),
                            (false, AlgorithmMode::StandardF64) => "💻 CPU Standard f64".to_string(),
                            (false, AlgorithmMode::Perturbation) => "💻 CPU Perturbation f64".to_string(),
                            (false, AlgorithmMode::ReferenceGmp) => "💻 CPU GMP Reference".to_string(),
                            (true, AlgorithmMode::StandardF64) => "🎮 GPU Standard f32".to_string(),
                            (true, AlgorithmMode::Perturbation) => "🎮 GPU Perturbation f32".to_string(),
                            (true, AlgorithmMode::ReferenceGmp) => "🔄 Auto".to_string(),
                        };

                        ui.menu_button(&render_text, |ui| {
                            if ui.selectable_label(
                                self.params.algorithm_mode == AlgorithmMode::Auto,
                                "🔄 Auto"
                            ).clicked() {
                                self.params.algorithm_mode = AlgorithmMode::Auto;
                                ui.close_menu();
                            }

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

                                if ui.selectable_label(
                                    !self.use_gpu && self.params.algorithm_mode == AlgorithmMode::ReferenceGmp,
                                    "🔢 GMP Reference"
                                ).clicked() {
                                    self.use_gpu = false;
                                    self.params.algorithm_mode = AlgorithmMode::ReferenceGmp;
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
                        });

                        // Si on passe de GPU à CPU ou vice-versa, ajuster l'algo si nécessaire
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

                        ui.separator();
                    }

                    // Puis le menu Type (toutes catégories)
                    ui.label("Type:");

                    // Catégories de fractales dans un seul menu
                    let vector_types = [(1, "Von Koch"), (2, "Dragon")];
                    let density_types = [(16, "Buddhabrot"), (24, "Nebulabrot")];

                    let current_category = match self.selected_type {
                        FractalType::VonKoch | FractalType::Dragon => "Vector",
                        FractalType::Buddhabrot | FractalType::Nebulabrot => "Densité",
                        FractalType::Lyapunov => "Lyapunov",
                        _ => "Escape-Time",
                    };

                    let current_label = match self.selected_type {
                        FractalType::Lyapunov => self.selected_lyapunov_preset.name(),
                        _ => self.selected_type.name(),
                    };

                    let type_menu_label = format!("▼ {}: {}", current_category, current_label);
                    ui.menu_button(&type_menu_label, |ui| {
                        ui.menu_button("Vector", |ui| {
                            for (id, label) in vector_types.iter() {
                                if let Some(fractal_type) = FractalType::from_id(*id) {
                                    if ui.selectable_label(self.selected_type == fractal_type, *label).clicked() {
                                        self.change_fractal_type(fractal_type);
                                        ui.close_menu();
                                    }
                                }
                            }
                        });

                        ui.menu_button("Escape-Time", |ui| {
                            // Sous-menu Mandelbrot
                            ui.menu_button("Mandelbrot", |ui| {
                                for (id, label) in [(3, "Mandelbrot"), (4, "Julia")] {
                                    if let Some(fractal_type) = FractalType::from_id(id) {
                                        if ui.selectable_label(self.selected_type == fractal_type, label).clicked() {
                                            self.change_fractal_type(fractal_type);
                                            ui.close_menu();
                                        }
                                    }
                                }
                            });

                            // Sous-menu Barnsley
                            ui.menu_button("Barnsley", |ui| {
                                for (id, label) in [(10, "Mandelbrot"), (9, "Julia")] {
                                    if let Some(fractal_type) = FractalType::from_id(id) {
                                        if ui.selectable_label(self.selected_type == fractal_type, label).clicked() {
                                            self.change_fractal_type(fractal_type);
                                            ui.close_menu();
                                        }
                                    }
                                }
                            });

                            // Sous-menu Magnet
                            ui.menu_button("Magnet", |ui| {
                                for (id, label) in [(12, "Mandelbrot"), (11, "Julia")] {
                                    if let Some(fractal_type) = FractalType::from_id(id) {
                                        if ui.selectable_label(self.selected_type == fractal_type, label).clicked() {
                                            self.change_fractal_type(fractal_type);
                                            ui.close_menu();
                                        }
                                    }
                                }
                            });

                            // Sous-menu Burning Ship
                            ui.menu_button("Burning Ship", |ui| {
                                for (id, label) in [(13, "Standard"), (18, "Perpendicular")] {
                                    if let Some(fractal_type) = FractalType::from_id(id) {
                                        if ui.selectable_label(self.selected_type == fractal_type, label).clicked() {
                                            self.change_fractal_type(fractal_type);
                                            ui.close_menu();
                                        }
                                    }
                                }
                            });

                            ui.separator();

                            // Sous-menu Variantes Mandelbrot
                            ui.menu_button("Variantes M", |ui| {
                                for (id, label) in [
                                    (14, "Tricorn"),
                                    (15, "Mandelbulb"),
                                    (19, "Celtic"),
                                    (20, "Alpha"),
                                    (23, "Multibrot"),
                                ] {
                                    if let Some(fractal_type) = FractalType::from_id(id) {
                                        if ui.selectable_label(self.selected_type == fractal_type, label).clicked() {
                                            self.change_fractal_type(fractal_type);
                                            ui.close_menu();
                                        }
                                    }
                                }
                            });

                            // Sous-menu Autres
                            ui.menu_button("Autres", |ui| {
                                for (id, label) in [
                                    (5, "Julia Sin"),
                                    (6, "Newton"),
                                    (7, "Phoenix"),
                                    (8, "Buffalo"),
                                    (21, "Pickover Stalks"),
                                    (22, "Nova"),
                                ] {
                                    if let Some(fractal_type) = FractalType::from_id(id) {
                                        if ui.selectable_label(self.selected_type == fractal_type, label).clicked() {
                                            self.change_fractal_type(fractal_type);
                                            ui.close_menu();
                                        }
                                    }
                                }
                            });
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
                    });

                    ui.separator();

                    // Plane (XaoS-style) uniquement pour fractales escape-time
                    let is_escape_time = !matches!(
                        self.selected_type,
                        FractalType::VonKoch | FractalType::Dragon | FractalType::Buddhabrot | FractalType::Lyapunov | FractalType::Nebulabrot
                    );
                    
                    if is_escape_time {
                        ui.label("Plane:");
                        let old_plane = self.params.plane_transform;
                        egui::ComboBox::from_id_source("plane_transform")
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

                    // Qualité (uniquement pour fractales supportées)
                    let supports_advanced_modes = matches!(
                        self.selected_type,
                        FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip
                    );
                    let show_tolerance = matches!(
                        self.params.algorithm_mode, 
                        AlgorithmMode::Auto | AlgorithmMode::Perturbation
                    );
                    
                    if supports_advanced_modes && show_tolerance {
                        ui.separator();
                        let old_preset = self.render_preset;
                        ui.label("Qualité:");
                        egui::ComboBox::from_id_source("render_preset")
                            .selected_text(self.render_preset.name())
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.render_preset, RenderPreset::Ultra, "Ultra (rapide)");
                                ui.selectable_value(&mut self.render_preset, RenderPreset::Fast, "Fast (équilibré)");
                                ui.selectable_value(&mut self.render_preset, RenderPreset::Standard, "Standard (précis)");
                            });
                        if old_preset != self.render_preset {
                            self.apply_render_preset(self.render_preset);
                            self.start_render();
                        }
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
                        let preview_image = generate_palette_preview(self.palette_index, 100, 12);
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
                    
                    ui.label("Color Repeat:");
                    let old_repeat = self.color_repeat;
                    ui.add(egui::Slider::new(&mut self.color_repeat, 1..=60));
                    if old_repeat != self.color_repeat {
                        self.params.color_repeat = self.color_repeat;
                        if !self.iterations.is_empty() {
                            self.update_texture(ctx);
                        }
                    }

                    ui.separator();

                    ui.label("Outcoloring:");
                    let old_out_mode = self.out_coloring_mode;
                    egui::ComboBox::from_id_source("outcoloring_mode")
                        .selected_text(self.out_coloring_mode.name())
                        .show_ui(ui, |ui| {
                            for mode in OutColoringMode::all() {
                                ui.selectable_value(&mut self.out_coloring_mode, *mode, mode.name());
                            }
                        });
                    if old_out_mode != self.out_coloring_mode {
                        self.params.out_coloring_mode = self.out_coloring_mode;
                        if !self.iterations.is_empty() {
                            self.update_texture(ctx);
                        }
                    }

                    ui.separator();

                });
                
                // Afficher les informations (centre, itérations, zoom) sous la barre de menu
                ui.separator();
                ui.horizontal(|ui| {
                    // Coordonnées du centre
                    let center_text = if self.params.center_x.abs() < 1e-6 && self.params.center_y.abs() < 1e-6 {
                        format!("Centre: ({:.6}, {:.6})", self.params.center_x, self.params.center_y)
                    } else if self.params.center_x.abs() > 1e3 || self.params.center_y.abs() > 1e3 {
                        format!("Centre: ({:.2e}, {:.2e})", self.params.center_x, self.params.center_y)
                    } else {
                        format!("Centre: ({:.6}, {:.6})", self.params.center_x, self.params.center_y)
                    };
                    ui.label(center_text);
                    
                    ui.separator();
                    
                    // Nombre d'itérations
                    ui.label(format!("Iter.: {}", self.params.iteration_max));
                    
                    ui.separator();
                    
                    // Calcul du zoom
                    let base_range = 4.0; // Plage de base pour Mandelbrot
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
                });
        });
        
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
                    let image_size = egui::Vec2::new(
                        self.params.width as f32,
                        self.params.height as f32,
                    );
                    
                    // Ajuster la taille pour tenir dans l'espace disponible
                    let scale = (available_size.x / image_size.x).min(available_size.y / image_size.y).min(1.0);
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
                    }
                    
                    // Détecter le début d'une sélection rectangulaire (drag avec bouton gauche)
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
                    // Seulement si on n'a pas fait de sélection
                    if response.clicked() && !self.selecting {
                        if let Some(pos) = response.interact_pointer_pos() {
                            let local_pos = pos - image_rect.min;
                            let pixel_x = (local_pos.x / image_rect.width()) * self.params.width as f32;
                            let pixel_y = (local_pos.y / image_rect.height()) * self.params.height as f32;
                            let point = self.pixel_to_complex(pixel_x, pixel_y, self.params.width as f32, self.params.height as f32);
                            self.zoom_at_point(point, 2.0);
                        }
                    }
                    
                    // Clic droit : dézoom centré sur la position de la souris
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

                    // Afficher le statut du rendu progressif
                    if self.rendering {
                        ui.separator();
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
        
        // Demander un re-rendu si nécessaire
        ctx.request_repaint();
    }
}
