use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Instant;

use egui::{Context, TextureHandle, TextureOptions};
use image::RgbImage;
use num_complex::Complex64;

use crate::color::{color_for_pixel, color_for_nebulabrot_pixel, color_for_buddhabrot_pixel};
use crate::fractal::{default_params_for_type, apply_lyapunov_preset, FractalParams, FractalType, LyapunovPreset};
use crate::render::render_escape_time_cancellable_with_reuse;
use crate::gui::texture::rgb_image_to_color_image;
use crate::gui::progressive::{ProgressiveConfig, RenderMessage, upscale_nearest};

/// Application principale egui pour fractall.
pub struct FractallApp {
    // État fractale
    params: FractalParams,
    iterations: Vec<u32>,
    zs: Vec<Complex64>,
    
    // Texture egui pour l'affichage
    texture: Option<TextureHandle>,
    
    // État UI
    selected_type: FractalType,
    palette_index: u8,
    color_repeat: u32,
    selected_lyapunov_preset: LyapunovPreset,
    
    // Zoom/interaction (conservés pour usage futur)
    #[allow(dead_code)]
    center_x: f64,
    #[allow(dead_code)]
    center_y: f64,
    
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

    // Dimensions de la fenêtre (pour calculer le viewport)
    window_width: u32,
    window_height: u32,
    pending_resize: Option<(u32, u32)>,
}

impl FractallApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let default_type = FractalType::Mandelbrot;
        let width = 1024;
        let height = 768;
        let params = default_params_for_type(default_type, width, height);
        
        Self {
            params: params.clone(),
            iterations: Vec::new(),
            zs: Vec::new(),
            texture: None,
            selected_type: default_type,
            palette_index: 6, // SmoothPlasma par défaut
            color_repeat: 40,
            selected_lyapunov_preset: LyapunovPreset::default(),
            center_x: 0.0,
            center_y: 0.0,
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
            window_width: width,
            window_height: height,
            pending_resize: None,
        }
    }
    
    /// Lance le rendu progressif de la fractale dans un thread séparé.
    fn start_render(&mut self) {
        // Annuler tout rendu en cours
        self.render_cancel.store(true, Ordering::Relaxed);

        // Créer un nouveau flag d'annulation
        self.render_cancel = Arc::new(AtomicBool::new(false));

        // Configuration progressive selon les paramètres
        let allow_intermediate = !matches!(
            self.params.fractal_type,
            FractalType::VonKoch
                | FractalType::Dragon
                | FractalType::Buddhabrot
                | FractalType::Nebulabrot
                | FractalType::Lyapunov
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

        // Spawner le thread de rendu progressif
        let handle = thread::spawn(move || {
            let mut previous_pass: Option<(Vec<u32>, Vec<Complex64>, u32, u32)> = None;
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
                let result = render_escape_time_cancellable_with_reuse(&pass_params, &cancel, reuse);

                match result {
                    Some((iterations, zs)) => {
                        // Garder une copie pour la passe suivante afin d'éviter le recalcul
                        if pass_index + 1 < config.passes.len() {
                            previous_pass = Some((iterations.clone(), zs.clone(), pass_width, pass_height));
                        } else {
                            previous_pass = None;
                        }

                        let _ = sender.send(RenderMessage::PassComplete {
                            pass_index: pass_index as u8,
                            scale_divisor,
                            iterations,
                            zs,
                            width: pass_width,
                            height: pass_height,
                        });
                    }
                    None => {
                        let _ = sender.send(RenderMessage::Cancelled);
                        return;
                    }
                }
            }

            let _ = sender.send(RenderMessage::AllComplete);
        });

        self.render_thread = Some(handle);
    }

    /// Vérifie si des passes de rendu sont terminées et met à jour l'affichage.
    /// Ne traite qu'un seul message PassComplete par frame pour permettre l'affichage progressif.
    fn check_render_complete(&mut self, ctx: &Context) {
        if !self.rendering {
            return;
        }

        // Récupérer un seul message pour permettre l'affichage de chaque passe
        let msg = {
            let receiver = match &self.render_receiver {
                Some(r) => r,
                None => return,
            };
            receiver.try_recv().ok()
        };

        let Some(msg) = msg else {
            return;
        };

        match msg {
            RenderMessage::PassComplete {
                pass_index,
                scale_divisor,
                iterations,
                zs,
                width,
                height,
            } => {
                self.current_pass = pass_index + 1;

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

            RenderMessage::AllComplete => {
                self.rendering = false;
                self.is_preview = false;
                self.render_thread = None;
                self.render_receiver = None;

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
        let center_x = (self.params.xmin + self.params.xmax) / 2.0;
        let center_y = (self.params.ymin + self.params.ymax) / 2.0;
        let span_x = self.params.xmax - self.params.xmin;
        let span_y = self.params.ymax - self.params.ymin;
        let current_aspect = span_x / span_y;
        let target_aspect = new_width as f64 / new_height as f64;

        let (new_span_x, new_span_y) = if current_aspect > target_aspect {
            // Élargir la hauteur pour éviter toute déformation.
            (span_x, span_x / target_aspect)
        } else {
            // Élargir la largeur pour éviter toute déformation.
            (span_y * target_aspect, span_y)
        };

        self.params.xmin = center_x - new_span_x / 2.0;
        self.params.xmax = center_x + new_span_x / 2.0;
        self.params.ymin = center_y - new_span_y / 2.0;
        self.params.ymax = center_y + new_span_y / 2.0;
        self.window_width = new_width;
        self.window_height = new_height;
        self.params.width = new_width;
        self.params.height = new_height;
        self.iterations.clear();
        self.zs.clear();
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

        let is_nebulabrot = self.params.fractal_type == FractalType::Nebulabrot;
        let is_buddhabrot = self.params.fractal_type == FractalType::Buddhabrot;

        // Créer l'image RGB avec colorisation parallélisée
        use rayon::prelude::*;
        let iterations = &self.iterations;
        let zs = &self.zs;
        let iter_max = self.params.iteration_max;
        let palette_idx = self.palette_index;
        let color_rep = self.color_repeat;

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
                            color_for_pixel(
                                iter,
                                z,
                                iter_max,
                                palette_idx,
                                color_rep,
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
        
        let x = self.params.xmin + x_ratio * (self.params.xmax - self.params.xmin);
        let y = self.params.ymax - y_ratio * (self.params.ymax - self.params.ymin); // Inverser Y
        
        Complex64::new(x, y)
    }
    
    /// Zoom au point spécifié avec un facteur donné.
    fn zoom_at_point(&mut self, point: Complex64, factor: f64) {
        let span_x = self.params.xmax - self.params.xmin;
        let span_y = self.params.ymax - self.params.ymin;
        
        let new_span_x = span_x / factor;
        let new_span_y = span_y / factor;
        
        self.params.xmin = point.re - new_span_x / 2.0;
        self.params.xmax = point.re + new_span_x / 2.0;
        self.params.ymin = point.im - new_span_y / 2.0;
        self.params.ymax = point.im + new_span_y / 2.0;
        
        self.start_render();
    }
    
    /// Zoom sur une zone rectangulaire sélectionnée.
    /// Les coordonnées sont en pixels dans l'image affichée.
    fn zoom_to_rectangle(&mut self, rect_min: egui::Pos2, rect_max: egui::Pos2, image_rect: egui::Rect) {
        // Convertir les coordonnées du rectangle de sélection en coordonnées pixels de l'image
        let image_width = self.params.width as f32;
        let image_height = self.params.height as f32;
        
        // Coordonnées relatives dans l'image (0.0 à 1.0)
        let rel_x1 = (rect_min.x - image_rect.min.x) / image_rect.width();
        let rel_y1 = (rect_min.y - image_rect.min.y) / image_rect.height();
        let rel_x2 = (rect_max.x - image_rect.min.x) / image_rect.width();
        let rel_y2 = (rect_max.y - image_rect.min.y) / image_rect.height();
        
        // Coordonnées pixels dans l'image
        let pixel_x1 = (rel_x1 * image_width).max(0.0).min(image_width - 1.0);
        let pixel_y1 = (rel_y1 * image_height).max(0.0).min(image_height - 1.0);
        let pixel_x2 = (rel_x2 * image_width).max(0.0).min(image_width - 1.0);
        let pixel_y2 = (rel_y2 * image_height).max(0.0).min(image_height - 1.0);
        
        // S'assurer que x1 < x2 et y1 < y2
        let (x1, x2) = if pixel_x1 < pixel_x2 {
            (pixel_x1, pixel_x2)
        } else {
            (pixel_x2, pixel_x1)
        };
        let (y1, y2) = if pixel_y1 < pixel_y2 {
            (pixel_y1, pixel_y2)
        } else {
            (pixel_y2, pixel_y1)
        };
        
        // Vérifier que le rectangle a une taille minimale
        let width = x2 - x1;
        let height = y2 - y1;
        if width < 5.0 || height < 5.0 {
            return; // Rectangle trop petit
        }
        
        // Convertir en coordonnées complexes
        let xmin_old = self.params.xmin;
        let xmax_old = self.params.xmax;
        let ymin_old = self.params.ymin;
        let ymax_old = self.params.ymax;
        
        let range_x_old = xmax_old - xmin_old;
        let range_y_old = ymax_old - ymin_old;
        let complex_aspect_ratio = range_x_old / range_y_old;
        
        // Conversion pixel -> complexe
        let pixel_to_x = range_x_old / image_width as f64;
        let pixel_to_y = range_y_old / image_height as f64;
        
        // Centre du rectangle sélectionné en coordonnées complexes
        let center_x_complex = xmin_old + ((x1 + x2) / 2.0) as f64 * pixel_to_x;
        let center_y_complex = ymin_old + ((y1 + y2) / 2.0) as f64 * pixel_to_y;
        
        // Taille du rectangle sélectionné dans l'espace complexe
        let selected_range_x = width as f64 * pixel_to_x;
        let selected_range_y = height as f64 * pixel_to_y;
        let selected_aspect_ratio = selected_range_x / selected_range_y;
        
        // Ajuster pour préserver le rapport d'aspect dans l'espace complexe
        let (new_range_x, new_range_y) = if selected_aspect_ratio > complex_aspect_ratio {
            // Rectangle sélectionné trop large, ajuster la hauteur
            (selected_range_x, selected_range_x / complex_aspect_ratio)
        } else {
            // Rectangle sélectionné trop haut, ajuster la largeur
            (selected_range_y * complex_aspect_ratio, selected_range_y)
        };
        
        // Calculer les nouvelles bornes centrées sur le centre du rectangle sélectionné
        self.params.xmin = center_x_complex - new_range_x / 2.0;
        self.params.xmax = center_x_complex + new_range_x / 2.0;
        self.params.ymin = center_y_complex - new_range_y / 2.0;
        self.params.ymax = center_y_complex + new_range_y / 2.0;
        
        self.start_render();
    }
    
    /// Dézoom avec un facteur donné.
    fn zoom_out(&mut self, factor: f64) {
        let center_x = (self.params.xmin + self.params.xmax) / 2.0;
        let center_y = (self.params.ymin + self.params.ymax) / 2.0;
        let span_x = self.params.xmax - self.params.xmin;
        let span_y = self.params.ymax - self.params.ymin;
        
        self.params.xmin = center_x - span_x * factor / 2.0;
        self.params.xmax = center_x + span_x * factor / 2.0;
        self.params.ymin = center_y - span_y * factor / 2.0;
        self.params.ymax = center_y + span_y * factor / 2.0;
        
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

        // Toujours utiliser le domaine par défaut pour bien centrer la fractale
        self.params = new_params;
        self.start_render();
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
                self.palette_index = (self.palette_index + 1) % 9;
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
        });
        
        // Panneau de contrôle en haut
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Type:");

                    // Catégories de fractales avec menus déroulants
                    let vector_types = [(1, "Von Koch"), (2, "Dragon")];


                    let density_types = [(16, "Buddhabrot"), (24, "Nebulabrot")];

                    // Helper pour déterminer la catégorie actuelle
                    let current_category = match self.selected_type {
                        FractalType::VonKoch | FractalType::Dragon => "Vector",
                        FractalType::Buddhabrot | FractalType::Nebulabrot => "Densité",
                        FractalType::Lyapunov => "Lyapunov",
                        _ => "Escape-Time",
                    };

                    // Menu Vector
                    let vector_label = if current_category == "Vector" {
                        format!("▼ Vector: {}", self.selected_type.name())
                    } else {
                        "Vector".to_string()
                    };
                    ui.menu_button(&vector_label, |ui| {
                        for (id, label) in vector_types.iter() {
                            if let Some(fractal_type) = FractalType::from_id(*id) {
                                if ui.selectable_label(self.selected_type == fractal_type, *label).clicked() {
                                    self.change_fractal_type(fractal_type);
                                    ui.close_menu();
                                }
                            }
                        }
                    });

                    // Menu Escape-Time avec sous-menus
                    let escape_label = if current_category == "Escape-Time" {
                        format!("▼ Escape-Time: {}", self.selected_type.name())
                    } else {
                        "Escape-Time".to_string()
                    };
                    ui.menu_button(&escape_label, |ui| {
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

                    // Menu Densité
                    let density_label = if current_category == "Densité" {
                        format!("▼ Densité: {}", self.selected_type.name())
                    } else {
                        "Densité".to_string()
                    };
                    ui.menu_button(&density_label, |ui| {
                        for (id, label) in density_types.iter() {
                            if let Some(fractal_type) = FractalType::from_id(*id) {
                                if ui.selectable_label(self.selected_type == fractal_type, *label).clicked() {
                                    self.change_fractal_type(fractal_type);
                                    ui.close_menu();
                                }
                            }
                        }
                    });

                    // Menu Lyapunov avec presets
                    let lyapunov_label = if current_category == "Lyapunov" {
                        format!("▼ Lyapunov: {}", self.selected_lyapunov_preset.name())
                    } else {
                        "Lyapunov".to_string()
                    };
                    ui.menu_button(&lyapunov_label, |ui| {
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
                
                ui.horizontal(|ui| {
                    ui.label("Palette:");
                    for i in 0..9u8 {
                        let is_selected = self.palette_index == i;
                        let label = format!("{}", i);
                        let button = if is_selected {
                            ui.selectable_label(true, label)
                        } else {
                            ui.selectable_label(false, label)
                        };
                        
                        if button.clicked() {
                            self.palette_index = i;
                            self.params.color_mode = i;
                            if !self.iterations.is_empty() {
                                self.update_texture(ctx);
                            }
                        }
                    }
                    
                    ui.separator();
                    
                    ui.label("Itérations:");
                    let old_iter = self.params.iteration_max;
                    ui.add(egui::Slider::new(&mut self.params.iteration_max, 100..=500000).logarithmic(true));
                    if old_iter != self.params.iteration_max && !self.iterations.is_empty() {
                        self.start_render();
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

                    let old_use_gmp = self.params.use_gmp;
                    let old_prec = self.params.precision_bits;
                    ui.checkbox(&mut self.params.use_gmp, "GMP");
                    if self.params.use_gmp {
                        ui.label("Precision bits:");
                        ui.add(egui::Slider::new(&mut self.params.precision_bits, 64..=2048));
                    }
                    if old_use_gmp != self.params.use_gmp || old_prec != self.params.precision_bits {
                        if !self.rendering {
                            self.start_render();
                        }
                    }
                });
        });
        
        // Interface principale - zone d'affichage de la fractale
        egui::CentralPanel::default().show(ctx, |ui| {
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
                    let response = ui.add(egui::Image::new(texture).fit_to_exact_size(display_size));
                    let image_rect = response.rect;
                    
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
                    
                    // Clic droit : dézoom (seulement si pas de sélection)
                    if response.secondary_clicked() && !self.selecting {
                        self.zoom_out(2.0);
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
                    ui.label(format!("Type: {}", self.selected_type.name()));
                    ui.separator();
                    ui.label(format!("Palette: {}", self.palette_index));
                    ui.separator();
                    ui.label(format!("Iterations: {}", self.params.iteration_max));
                    ui.separator();
                    ui.label(format!("Color Repeat: {}", self.color_repeat));
                    ui.separator();
                    
                    let span_x = self.params.xmax - self.params.xmin;
                    let base_range = 4.0;
                    let zoom_factor = base_range / span_x;
                    ui.label(format!("Zoom: {:.2}x", zoom_factor));
                    
                    ui.separator();
                    let center_x = (self.params.xmin + self.params.xmax) / 2.0;
                    let center_y = (self.params.ymin + self.params.ymax) / 2.0;
                    ui.label(format!("Centre: ({:.6}, {:.6})", center_x, center_y));
                    
                    if let Some(time) = self.last_render_time {
                        ui.separator();
                        ui.label(format!("Temps: {:.2}s", time));
                    }

                    // Afficher le statut du rendu progressif
                    if self.rendering {
                        ui.separator();
                        ui.label(format!("Rendu pass {}/{}", self.current_pass, self.total_passes));
                    } else if self.is_preview {
                        ui.separator();
                        ui.label("Preview");
                    }
                });
        });
        
        // Demander un re-rendu si nécessaire
        ctx.request_repaint();
    }
}
