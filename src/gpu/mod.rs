use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::mpsc::RecvTimeoutError;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use num_complex::Complex64;
use rayon::prelude::*;
use wgpu::util::DeviceExt;

use crate::fractal::{FractalParams, FractalType};
use crate::fractal::gmp::{complex_from_xy, complex_to_complex64, iterate_point_mpc, MpcParams};
use crate::fractal::perturbation::{compute_perturbation_precision_bits, compute_dc_gmp, mark_neighbor_glitches};
use crate::fractal::perturbation::orbit::{compute_reference_orbit_cached, ReferenceOrbitCache};

const WORKGROUP_SIZE: u32 = 16;
const MAX_LEVELS: usize = 17;

fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => false,
    }
}

/// Sélectionne les backends wgpu appropriés selon l'OS détecté.
/// 
/// - macOS : Metal (requis pour macOS)
/// - Linux : Vulkan et OpenGL (Vulkan prioritaire pour NVIDIA)
/// - Windows : DirectX12 et Vulkan
/// - Autres : Tous les backends disponibles
fn select_backends_for_platform() -> wgpu::Backends {
    match std::env::consts::OS {
        "macos" => wgpu::Backends::METAL,
        "linux" => wgpu::Backends::VULKAN | wgpu::Backends::GL,
        "windows" => wgpu::Backends::DX12 | wgpu::Backends::VULKAN,
        _ => wgpu::Backends::all(), // Fallback pour autres OS
    }
}

/// Cache pour les buffers GPU de perturbation.
/// Permet d'éviter de re-uploader les données d'orbite quand elles n'ont pas changé.
struct PerturbationBufferCache {
    /// Buffer contenant l'orbite de référence z_ref
    zref_buffer: wgpu::Buffer,
    /// Buffer contenant les noeuds BLA aplatis
    bla_buffer: wgpu::Buffer,
    /// Buffer contenant les métadonnées BLA
    meta_buffer: wgpu::Buffer,
    /// Identifiant unique de l'orbite (center_x_gmp + center_y_gmp + iteration_max)
    orbit_id: String,
}

impl PerturbationBufferCache {
    /// Génère un identifiant unique pour une orbite basé sur le cache.
    fn generate_orbit_id(cache: &ReferenceOrbitCache) -> String {
        format!(
            "{}_{}_{}_{:?}",
            cache.center_x_gmp,
            cache.center_y_gmp,
            cache.iteration_max,
            cache.fractal_type
        )
    }
}

pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_f32: wgpu::ComputePipeline,
    #[allow(dead_code)]
    pipeline_f64: Option<PipelinesF64>,
    pipeline_julia_f32: wgpu::ComputePipeline,
    pipeline_burning_ship_f32: wgpu::ComputePipeline,
    pipeline_perturbation: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    bind_group_layout_f64: Option<wgpu::BindGroupLayout>,
    bind_group_layout_perturb: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    supports_f64: bool,
    /// Cache des buffers GPU pour la perturbation (zref, bla, meta)
    /// Utilise Mutex pour permettre l'accès mutable depuis &self (interior mutability)
    perturbation_cache: Mutex<Option<PerturbationBufferCache>>,
}

impl GpuRenderer {
    pub fn new() -> Option<Self> {
        // Capturer les panics de wgpu lors de l'initialisation EGL
        // pour permettre à l'application de démarrer sans GPU
        std::panic::catch_unwind(|| {
            pollster::block_on(async {
            // Sélectionner les backends appropriés selon l'OS
            let backends = select_backends_for_platform();
            let os_name = std::env::consts::OS;
            eprintln!("OS détecté: {}, Backends wgpu sélectionnés: {:?}", os_name, backends);
            
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends,
                ..Default::default()
            });
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await?;

            // Afficher les infos de l'adaptateur GPU
            let info = adapter.get_info();
            // Sur Apple Silicon, info.name contient le CPU/GPU unifié (ex: "Apple M1")
            // et info.backend contient le backend (Metal)
            if !info.driver.is_empty() {
                eprintln!("GPU détecté: {} (Backend: {:?}), Driver: {}", info.name, info.backend, info.driver);
            } else {
                eprintln!("CPU/GPU détecté: {} (Backend: {:?})", info.name, info.backend);
            }

            // Ne plus utiliser f64 en mode GPU, toujours utiliser f32
            let supports_f64 = false;
            let required_features = wgpu::Features::empty();

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("gpu-device"),
                        required_features,
                        required_limits: wgpu::Limits::default(),
                    },
                    None,
                )
                .await
                .ok()?;

            let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mandelbrot-bind-group-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(std::mem::size_of::<ParamsF32>() as u64),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mandelbrot-pipeline-layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            // Ne plus créer les layouts f64 car on utilise uniquement f32
            let bind_group_layout_f64: Option<wgpu::BindGroupLayout> = None;
            let _pipeline_layout_f64: Option<wgpu::PipelineLayout> = None;

            let shader_f32 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mandelbrot-f32"),
                source: wgpu::ShaderSource::Wgsl(include_str!("mandelbrot_f32.wgsl").into()),
            });

            let pipeline_f32 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("mandelbrot-pipeline-f32"),
                layout: Some(&pipeline_layout),
                module: &shader_f32,
                entry_point: "main",
            });

            let shader_julia_f32 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("julia-f32"),
                source: wgpu::ShaderSource::Wgsl(include_str!("julia_f32.wgsl").into()),
            });

            let pipeline_julia_f32 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("julia-pipeline-f32"),
                layout: Some(&pipeline_layout),
                module: &shader_julia_f32,
                entry_point: "main",
            });

            let shader_burning_ship_f32 =
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("burning-ship-f32"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("burning_ship_f32.wgsl").into(),
                    ),
                });

            let pipeline_burning_ship_f32 =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("burning-ship-pipeline-f32"),
                    layout: Some(&pipeline_layout),
                    module: &shader_burning_ship_f32,
                    entry_point: "main",
                });

            // Ne plus créer les pipelines f64 car on utilise uniquement f32
            let pipeline_f64: Option<PipelinesF64> = None;

            let bind_group_layout_perturb =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("perturbation-bind-group-layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: NonZeroU64::new(
                                    std::mem::size_of::<PerturbParams>() as u64,
                                ),
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

            let pipeline_layout_perturb =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("perturbation-pipeline-layout"),
                    bind_group_layouts: &[&bind_group_layout_perturb],
                    push_constant_ranges: &[],
                });

            let shader_perturb = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("perturbation-shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("perturbation.wgsl").into()),
            });

            let pipeline_perturbation =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("perturbation-pipeline"),
                    layout: Some(&pipeline_layout_perturb),
                    module: &shader_perturb,
                    entry_point: "main",
                });

            Some(Self {
                device,
                queue,
                pipeline_f32,
                pipeline_f64,
                pipeline_julia_f32,
                pipeline_burning_ship_f32,
                pipeline_perturbation,
                bind_group_layout,
                bind_group_layout_f64,
                bind_group_layout_perturb,
                supports_f64,
                perturbation_cache: Mutex::new(None),
            })
        })
        }).ok().flatten()
    }

    pub fn render_mandelbrot(
        &self,
        params: &FractalParams,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Option<(Vec<u32>, Vec<Complex64>)> {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        // Toujours utiliser f32 en mode GPU
        self.render_mandelbrot_f32(params, cancel)
    }


    pub fn render_julia(
        &self,
        params: &FractalParams,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Option<(Vec<u32>, Vec<Complex64>)> {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        // Toujours utiliser f32 en mode GPU
        self.render_julia_f32(params, cancel)
    }

    pub fn render_burning_ship(
        &self,
        params: &FractalParams,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Option<(Vec<u32>, Vec<Complex64>)> {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        // Toujours utiliser f32 en mode GPU
        self.render_burning_ship_f32(params, cancel)
    }

    pub fn precision_label(&self) -> &'static str {
        // Toujours utiliser f32 en mode GPU
        "f32"
    }

    /// Render perturbation with optional orbit cache support.
    /// Returns the result and the updated cache for reuse in subsequent frames.
    /// Uses GPU buffer caching to avoid re-uploading orbit data when unchanged.
    pub fn render_perturbation_with_cache(
        &self,
        params: &FractalParams,
        cancel: &std::sync::atomic::AtomicBool,
        reuse: Option<(&[u32], &[Complex64], u32, u32)>,
        orbit_cache: Option<&Arc<ReferenceOrbitCache>>,
    ) -> Option<((Vec<u32>, Vec<Complex64>), Arc<ReferenceOrbitCache>)> {
        let stats = env_flag("FRACTALL_GPU_PERTURB_STATS") || env_flag("FRACTALL_PERTURB_STATS");
        let t_all = Instant::now();
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        let supports = matches!(
            params.fractal_type,
            FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip
        );
        if !supports {
            return None;
        }

        let mut orbit_params = params.clone();
        orbit_params.precision_bits = compute_perturbation_precision_bits(params);

        // Use cached orbit/BLA or compute fresh
        let t_ref = Instant::now();
        let cache = compute_reference_orbit_cached(&orbit_params, Some(cancel), orbit_cache)?;
        let dt_ref = t_ref.elapsed();
        let ref_orbit = &cache.orbit;
        let bla_table = &cache.bla_table;
        let supports_bla = matches!(params.fractal_type, FractalType::Mandelbrot | FractalType::Julia);
        let bla_levels = if supports_bla {
            bla_table.levels.len().min(MAX_LEVELS)
        } else {
            0
        };

        // Check if we can reuse GPU buffers from the cache
        let current_orbit_id = PerturbationBufferCache::generate_orbit_id(&cache);
        
        // Prepare buffer data outside the lock
        let mut level_offsets = [0u32; MAX_LEVELS];
        let mut level_lengths = [0u32; MAX_LEVELS];
        let mut flattened = Vec::new();
        let mut offset = 0u32;
        for (idx, level) in bla_table.levels.iter().enumerate().take(MAX_LEVELS) {
            level_offsets[idx] = offset;
            level_lengths[idx] = level.len() as u32;
            offset += level.len() as u32;
            flattened.extend(level.iter().map(|node| BlaNode {
                a_re: node.a.re as f32,
                a_im: node.a.im as f32,
                b_re: node.b.re as f32,
                b_im: node.b.im as f32,
                c_re: node.c.re as f32,
                c_im: node.c.im as f32,
                validity: node.validity_radius as f32,
                _pad: 0.0,
            }));
        }

        let z_ref_data: Vec<ZRef> = ref_orbit
            .z_ref_f64
            .iter()
            .map(|z| ZRef {
                re: z.re as f32,
                im: z.im as f32,
            })
            .collect();

        // Access the cache with interior mutability via Mutex.
        // Recover from poison if another thread panicked (e.g. device lost) to avoid double panic.
        let mut cache_guard = self.perturbation_cache.lock().unwrap_or_else(|e| e.into_inner());
        
        // Check if we can reuse existing buffers
        let can_reuse = cache_guard.as_ref()
            .map(|c| c.orbit_id == current_orbit_id)
            .unwrap_or(false);
        
        if !can_reuse {
            // Need to create new buffers
            let zref_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("perturb-zref-cached"),
                contents: bytemuck::cast_slice(&z_ref_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

            // wgpu n'accepte pas les buffers de taille zéro dans un bind group
            let bla_contents: Vec<BlaNode> = if flattened.is_empty() {
                vec![BlaNode {
                    a_re: 0.0, a_im: 0.0, b_re: 0.0, b_im: 0.0, c_re: 0.0, c_im: 0.0,
                    validity: 0.0, _pad: 0.0,
                }]
            } else {
                flattened
            };
            let bla_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("perturb-bla-nodes-cached"),
                contents: bytemuck::cast_slice(&bla_contents),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let meta_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("perturb-bla-meta-cached"),
                contents: bytemuck::bytes_of(&BlaMeta {
                    level_offsets,
                    level_lengths,
                    _pad: [0; 2],
                }),
                usage: wgpu::BufferUsages::STORAGE,
            });

            // Store in cache for next frame
            *cache_guard = Some(PerturbationBufferCache {
                zref_buffer,
                bla_buffer,
                meta_buffer,
                orbit_id: current_orbit_id,
            });
        }
        
        // Get references to cached buffers
        let cached = cache_guard.as_ref().unwrap();
        let zref_buffer = &cached.zref_buffer;
        let bla_buffer = &cached.bla_buffer;
        let meta_buffer = &cached.meta_buffer;

        let iter_max = params
            .iteration_max
            .min(ref_orbit.z_ref_f64.len().saturating_sub(1) as u32);

        let width = params.width as usize;
        let height = params.height as usize;
        if width == 0 || height == 0 {
            return Some(((Vec::new(), Vec::new()), cache));
        }

        let output_count = width * height;
        let output_size = (output_count * std::mem::size_of::<PixelOut>()) as u64;

        // IMPORTANT: La perturbation ne supporte pas la réutilisation des pixels entre passes (dc change).
        // Le CPU désactive déjà cette réutilisation pour éviter des artefacts. On fait pareil côté GPU.
        // Pour debug, on peut forcer l'activation avec FRACTALL_GPU_PERTURB_ENABLE_REUSE=1.
        let enable_reuse = env_flag("FRACTALL_GPU_PERTURB_ENABLE_REUSE");
        let reuse = if enable_reuse { build_reuse(params, reuse) } else { None };
        let mut mask: Vec<u32> = vec![1u32; output_count];
        let mut initial_output = vec![
            PixelOut {
                iter: 0,
                z_re: 0.0,
                z_im: 0.0,
                flags: 0,
            };
            output_count
        ];
        if let Some(reuse) = reuse.as_ref() {
            let ratio = reuse.ratio as usize;
            for y in 0..height {
                if y % ratio != 0 {
                    continue;
                }
                let src_y = y / ratio;
                for x in 0..width {
                    if x % ratio != 0 {
                        continue;
                    }
                    let src_x = x / ratio;
                    let src_idx = src_y * reuse.width as usize + src_x;
                    if src_idx < reuse.iterations.len() {
                        let dst_idx = y * width + x;
                        mask[dst_idx] = 0;
                        initial_output[dst_idx] = PixelOut {
                            iter: reuse.iterations[src_idx],
                            z_re: reuse.zs[src_idx].re as f32,
                            z_im: reuse.zs[src_idx].im as f32,
                            flags: 0,
                        };
                    }
                }
            }
        }
        if stats && enable_reuse {
            let mut zero_count = 0usize;
            let mut min_x = width;
            let mut min_y = height;
            let mut max_x = 0usize;
            let mut max_y = 0usize;
            for (idx, &m) in mask.iter().enumerate() {
                if m == 0 {
                    zero_count += 1;
                    let x = idx % width;
                    let y = idx / width;
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                }
            }
            eprintln!(
                "[GPU PERTURB] reuse_enabled=1 mask_zero={} bbox=({},{})-({},{})",
                zero_count, min_x, min_y, max_x, max_y
            );
        }

        let output_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("perturb-output"),
            contents: bytemuck::cast_slice(&initial_output),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("perturb-readback"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let span_x = params.span_x;
        let span_y = params.span_y;
        let center_x = params.center_x;
        let center_y = params.center_y;

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("perturb-params"),
            contents: bytemuck::bytes_of(&PerturbParams {
                center_x: center_x as f32,
                center_y: center_y as f32,
                span_x: span_x as f32,
                span_y: span_y as f32,
                cref_x: ref_orbit.cref.re as f32,
                cref_y: ref_orbit.cref.im as f32,
                width: params.width,
                height: params.height,
                iter_max,
                bailout: params.bailout as f32,
                bla_levels: bla_levels as u32,
                fractal_kind: match params.fractal_type {
                    FractalType::Mandelbrot => 0,
                    FractalType::Julia => 1,
                    FractalType::BurningShip => 2,
                    _ => 0,
                },
                glitch_tolerance: params.glitch_tolerance as f32,
                series_order: params.series_order as u32,
                series_threshold: params.series_threshold as f32,
                _pad_align: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let mask_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("perturb-reuse-mask"),
            contents: bytemuck::cast_slice(&mask),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("perturb-bind-group"),
            layout: &self.bind_group_layout_perturb,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bla_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: zref_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: mask_buffer.as_entire_binding(),
                },
            ],
        });

        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }

        let t_gpu = Instant::now();
        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("perturb-encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("perturb-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_perturbation);
            pass.set_bind_group(0, &bind_group, &[]);
            let dispatch_x = (params.width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            let dispatch_y = (params.height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_size);
        self.queue.submit(Some(encoder.finish()));

        // Improved GPU readback with channel notification instead of busy-wait
        let buffer_slice = readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.device.poll(wgpu::Maintain::Poll);

        loop {
            match receiver.recv_timeout(Duration::from_millis(10)) {
                Ok(result) => {
                    result.ok()?;
                    break;
                }
                Err(RecvTimeoutError::Timeout) => {
                    if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                        readback_buffer.unmap();
                        return None;
                    }
                    self.device.poll(wgpu::Maintain::Poll);
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return None;
                }
            }
        }

        let data = buffer_slice.get_mapped_range();
        let pixels: &[PixelOut] = bytemuck::cast_slice(&data);
        let mut iterations = Vec::with_capacity(output_count);
        let mut zs = Vec::with_capacity(output_count);
        let mut glitch_mask = vec![false; output_count];

        // Stats de diagnostic: détecter les zones non calculées (iter=0, flags=0, z=0)
        let mut count_iter0 = 0usize;
        let mut count_flags = 0usize;
        let mut count_z0 = 0usize;
        let mut count_unwritten = 0usize;
        let mut min_x = width;
        let mut min_y = height;
        let mut max_x = 0usize;
        let mut max_y = 0usize;

        // Échantillonner quelques pixels pour vérifier la variation
        let sample_indices = if stats && output_count > 100 {
            vec![0, output_count / 4, output_count / 2, 3 * output_count / 4, output_count - 1]
        } else {
            vec![]
        };
        
        for (idx, p) in pixels.iter().enumerate() {
            if p.flags != 0 {
                glitch_mask[idx] = true;
            }
            iterations.push(p.iter);
            zs.push(Complex64::new(p.z_re as f64, p.z_im as f64));
            
            if stats && sample_indices.contains(&idx) {
                let x = idx % width;
                let y = idx / width;
                eprintln!(
                    "[GPU PERTURB] sample pixel({},{}) idx={} iter={} z=({:.6e},{:.6e})",
                    x, y, idx, p.iter, p.z_re as f64, p.z_im as f64
                );
            }

            if stats {
                if p.iter == 0 {
                    count_iter0 += 1;
                }
                if p.flags != 0 {
                    count_flags += 1;
                }
                if p.z_re == 0.0 && p.z_im == 0.0 {
                    count_z0 += 1;
                }
                if p.iter == 0 && p.flags == 0 && p.z_re == 0.0 && p.z_im == 0.0 {
                    count_unwritten += 1;
                    let x = idx % width;
                    let y = idx / width;
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                }
            }
        }
        drop(data);
        readback_buffer.unmap();

        let dt_gpu = t_gpu.elapsed();
        if stats {
            // Analyser la distribution des itérations pour détecter des zones uniformes
            let mut iter_counts = std::collections::HashMap::new();
            for &iter_val in &iterations {
                *iter_counts.entry(iter_val).or_insert(0) += 1;
            }
            let max_iter_count = iter_counts.values().max().copied().unwrap_or(0);
            let most_common_iter = iter_counts.iter()
                .max_by_key(|(_, &count)| count)
                .map(|(iter, _)| *iter)
                .unwrap_or(0);
            
            // Échantillonner quelques pixels au centre pour voir leurs valeurs
            let center_x = width / 2;
            let center_y = height / 2;
            let center_idx = center_y * width + center_x;
            let center_iter = if center_idx < iterations.len() { iterations[center_idx] } else { 0 };
            let center_z_re = if center_idx < zs.len() { zs[center_idx].re } else { 0.0 };
            let center_z_im = if center_idx < zs.len() { zs[center_idx].im } else { 0.0 };
            
            eprintln!(
                "[GPU PERTURB] {}x{} enable_reuse={} ref={:.3}s gpu+readback={:.3}s iter0={} flags!=0={} z0={} unwritten={} bbox=({},{})-({},{}) total={:.3}s",
                params.width,
                params.height,
                enable_reuse as u8,
                dt_ref.as_secs_f64(),
                dt_gpu.as_secs_f64(),
                count_iter0,
                count_flags,
                count_z0,
                count_unwritten,
                min_x,
                min_y,
                max_x,
                max_y,
                t_all.elapsed().as_secs_f64(),
            );
            eprintln!(
                "[GPU PERTURB] stats: max_iter_count={} (iter={}), center({},{}) iter={} z=({:.6e},{:.6e})",
                max_iter_count, most_common_iter, center_x, center_y, center_iter, center_z_re, center_z_im
            );
        }

        // Fast-path petites images: éviter le post-traitement voisinage (coût fixe non négligeable)
        // Comme côté CPU, désactiver neighbor_pass pour petites images
        let small_image = params.width.max(params.height) <= 512;
        if !small_image && params.glitch_neighbor_pass {
            let neighbor_threshold = (params.iteration_max / 50).max(8);
            let neighbor_mask =
                mark_neighbor_glitches(&iterations, params.width, params.height, neighbor_threshold);
            for (idx, flagged) in neighbor_mask.into_iter().enumerate() {
                if flagged {
                    glitch_mask[idx] = true;
                }
            }
        }

        let glitched_indices: Vec<u32> = glitch_mask
            .iter()
            .enumerate()
            .filter_map(|(idx, flagged)| if *flagged { Some(idx as u32) } else { None })
            .collect();

        // Fallback complet vers GMP si trop de glitches (>30% des pixels)
        // Augmenté de 10% à 30% pour éviter de recalculer toute l'image trop souvent.
        // La correction individuelle avec perturbation GMP est maintenant plus efficace.
        let total_pixels = output_count as f64;
        let glitch_ratio = glitched_indices.len() as f64 / total_pixels;
        const GLITCH_FALLBACK_THRESHOLD: f64 = 0.30; // 30%

        if glitch_ratio > GLITCH_FALLBACK_THRESHOLD {
            // Trop de glitches: recalculer tous les pixels en GMP
            let gmp_params = MpcParams::from_params(&orbit_params);
            let prec = compute_perturbation_precision_bits(params);
            let width_u32 = params.width;
            
            // Utiliser compute_dc_gmp pour calculer directement en GMP (comme fallback CPU)
            let center_x_gmp = if let Some(ref cx_hp) = params.center_x_hp {
                match rug::Float::parse(cx_hp) {
                    Ok(parse_result) => rug::Float::with_val(prec, parse_result),
                    Err(_) => rug::Float::with_val(prec, params.center_x),
                }
            } else {
                rug::Float::with_val(prec, params.center_x)
            };
            let center_y_gmp = if let Some(ref cy_hp) = params.center_y_hp {
                match rug::Float::parse(cy_hp) {
                    Ok(parse_result) => rug::Float::with_val(prec, parse_result),
                    Err(_) => rug::Float::with_val(prec, params.center_y),
                }
            } else {
                rug::Float::with_val(prec, params.center_y)
            };
            
            // Recalculer tous les pixels en GMP (comme fallback CPU)
            let width_usize = width_u32 as usize;
            let all_corrections: Vec<_> = (0..output_count)
                .par_bridge()
                .map(|idx| {
                    let i = idx % width_usize;
                    let j = idx / width_usize;
                    
                    // Calculer dc en GMP directement avec compute_dc_gmp
                    let dc_gmp = compute_dc_gmp(i, j, params, &center_x_gmp, &center_y_gmp, prec);
                    
                    // Calculer le point pixel = center + dc en GMP
                    let mut z_pixel_re = center_x_gmp.clone();
                    z_pixel_re += dc_gmp.real();
                    let mut z_pixel_im = center_y_gmp.clone();
                    z_pixel_im += dc_gmp.imag();
                    let z_pixel = complex_from_xy(prec, z_pixel_re, z_pixel_im);
                    
                    let (iter_val, z_final) = iterate_point_mpc(&gmp_params, &z_pixel);
                    (idx, iter_val, complex_to_complex64(&z_final))
                })
                .collect();
            
            // Remplacer tous les résultats
            for (idx, iter_val, z_final) in all_corrections {
                iterations[idx] = iter_val;
                zs[idx] = z_final;
            }
            
            if stats {
                eprintln!(
                    "[GPU PERTURB] fallback_gmp: glitch_ratio={:.3} > {:.3}, recalculé {} pixels",
                    glitch_ratio, GLITCH_FALLBACK_THRESHOLD, output_count
                );
            }
            
            return Some(((iterations, zs), cache));
        }

        // Parallel glitch correction using Rayon
        if !glitched_indices.is_empty() {
            let gmp_params = MpcParams::from_params(&orbit_params);
            let prec = gmp_params.prec;
            let width = params.width;

            // Hoist center GMP parsing outside the parallel closure
            let center_x_gmp = if let Some(ref cx_hp) = params.center_x_hp {
                match rug::Float::parse(cx_hp) {
                    Ok(parse_result) => rug::Float::with_val(prec, parse_result),
                    Err(_) => rug::Float::with_val(prec, params.center_x),
                }
            } else {
                rug::Float::with_val(prec, params.center_x)
            };
            let center_y_gmp = if let Some(ref cy_hp) = params.center_y_hp {
                match rug::Float::parse(cy_hp) {
                    Ok(parse_result) => rug::Float::with_val(prec, parse_result),
                    Err(_) => rug::Float::with_val(prec, params.center_y),
                }
            } else {
                rug::Float::with_val(prec, params.center_y)
            };

            // Use compute_dc_gmp for correct +0.5 pixel centering (matching CPU path)
            let corrections: Vec<_> = glitched_indices
                .par_iter()
                .map(|&idx| {
                    let i = (idx % width) as usize;
                    let j = (idx / width) as usize;

                    let dc_gmp = compute_dc_gmp(i, j, params, &center_x_gmp, &center_y_gmp, prec);

                    let mut z_pixel_re = center_x_gmp.clone();
                    z_pixel_re += dc_gmp.real();
                    let mut z_pixel_im = center_y_gmp.clone();
                    z_pixel_im += dc_gmp.imag();
                    let z_pixel = complex_from_xy(prec, z_pixel_re, z_pixel_im);
                    let (iter_val, z_final) = iterate_point_mpc(&gmp_params, &z_pixel);
                    (idx as usize, iter_val, complex_to_complex64(&z_final))
                })
                .collect();

            // Apply corrections
            for (idx, iter_val, z_final) in corrections {
                iterations[idx] = iter_val;
                zs[idx] = z_final;
            }
        }

        Some(((iterations, zs), cache))
    }

    fn render_mandelbrot_f32(
        &self,
        params: &FractalParams,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Option<(Vec<u32>, Vec<Complex64>)> {
        self.render_escape_f32(
            params,
            cancel,
            &self.pipeline_f32,
            ParamsF32::from_params(params, None),
            "mandelbrot",
        )
    }

    #[allow(dead_code)]
    fn render_mandelbrot_f64(
        &self,
        params: &FractalParams,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Option<(Vec<u32>, Vec<Complex64>)> {
        let pipelines = self.pipeline_f64.as_ref()?;
        self.render_escape_f64(
            params,
            cancel,
            &pipelines.mandelbrot,
            ParamsF64::from_params(params, None),
            "mandelbrot",
        )
    }

    fn render_julia_f32(
        &self,
        params: &FractalParams,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Option<(Vec<u32>, Vec<Complex64>)> {
        self.render_escape_f32(
            params,
            cancel,
            &self.pipeline_julia_f32,
            ParamsF32::from_params(params, Some(params.seed)),
            "julia",
        )
    }

    fn render_burning_ship_f32(
        &self,
        params: &FractalParams,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Option<(Vec<u32>, Vec<Complex64>)> {
        self.render_escape_f32(
            params,
            cancel,
            &self.pipeline_burning_ship_f32,
            ParamsF32::from_params(params, None),
            "burning-ship",
        )
    }

    #[allow(dead_code)]
    fn render_julia_f64(
        &self,
        params: &FractalParams,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Option<(Vec<u32>, Vec<Complex64>)> {
        let pipelines = self.pipeline_f64.as_ref()?;
        self.render_escape_f64(
            params,
            cancel,
            &pipelines.julia,
            ParamsF64::from_params(params, Some(params.seed)),
            "julia",
        )
    }

    #[allow(dead_code)]
    fn render_burning_ship_f64(
        &self,
        params: &FractalParams,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Option<(Vec<u32>, Vec<Complex64>)> {
        let pipelines = self.pipeline_f64.as_ref()?;
        self.render_escape_f64(
            params,
            cancel,
            &pipelines.burning_ship,
            ParamsF64::from_params(params, None),
            "burning-ship",
        )
    }

    fn render_escape_f32(
        &self,
        params: &FractalParams,
        cancel: &std::sync::atomic::AtomicBool,
        pipeline: &wgpu::ComputePipeline,
        params_value: ParamsF32,
        label: &str,
    ) -> Option<(Vec<u32>, Vec<Complex64>)> {
        let width = params.width as usize;
        let height = params.height as usize;
        if width == 0 || height == 0 {
            return Some((Vec::new(), Vec::new()));
        }
        let output_count = width * height;
        let output_size = (output_count * std::mem::size_of::<PixelOut>()) as u64;

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}-output-f32")),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}-readback-f32")),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label}-params-f32")),
            contents: bytemuck::bytes_of(&params_value),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}-bind-group-f32")),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("{label}-encoder-f32")),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{label}-pass-f32")),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let dispatch_x = (params.width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            let dispatch_y = (params.height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_size);
        self.queue.submit(Some(encoder.finish()));

        // Improved GPU readback with channel notification
        let buffer_slice = readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.device.poll(wgpu::Maintain::Poll);

        loop {
            match receiver.recv_timeout(Duration::from_millis(10)) {
                Ok(result) => {
                    result.ok()?;
                    break;
                }
                Err(RecvTimeoutError::Timeout) => {
                    if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                        readback_buffer.unmap();
                        return None;
                    }
                    self.device.poll(wgpu::Maintain::Poll);
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return None;
                }
            }
        }

        let data = buffer_slice.get_mapped_range();
        let pixels: &[PixelOut] = bytemuck::cast_slice(&data);
        let mut iterations = Vec::with_capacity(output_count);
        let mut zs = Vec::with_capacity(output_count);
        for p in pixels {
            iterations.push(p.iter);
            zs.push(Complex64::new(p.z_re as f64, p.z_im as f64));
        }
        drop(data);
        readback_buffer.unmap();

        Some((iterations, zs))
    }

    #[allow(dead_code)]
    fn render_escape_f64(
        &self,
        params: &FractalParams,
        cancel: &std::sync::atomic::AtomicBool,
        pipeline: &wgpu::ComputePipeline,
        params_value: ParamsF64,
        label: &str,
    ) -> Option<(Vec<u32>, Vec<Complex64>)> {
        let width = params.width as usize;
        let height = params.height as usize;
        if width == 0 || height == 0 {
            return Some((Vec::new(), Vec::new()));
        }
        let output_count = width * height;
        let output_size = (output_count * std::mem::size_of::<PixelOut>()) as u64;

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}-output-f64")),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}-readback-f64")),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label}-params-f64")),
            contents: bytemuck::bytes_of(&params_value),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group_layout_f64 = self.bind_group_layout_f64.as_ref()?;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}-bind-group-f64")),
            layout: bind_group_layout_f64,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("{label}-encoder-f64")),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{label}-pass-f64")),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let dispatch_x = (params.width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            let dispatch_y = (params.height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_size);
        self.queue.submit(Some(encoder.finish()));

        // Improved GPU readback with channel notification
        let buffer_slice = readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.device.poll(wgpu::Maintain::Poll);

        loop {
            match receiver.recv_timeout(Duration::from_millis(10)) {
                Ok(result) => {
                    result.ok()?;
                    break;
                }
                Err(RecvTimeoutError::Timeout) => {
                    if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                        readback_buffer.unmap();
                        return None;
                    }
                    self.device.poll(wgpu::Maintain::Poll);
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return None;
                }
            }
        }

        let data = buffer_slice.get_mapped_range();
        let pixels: &[PixelOut] = bytemuck::cast_slice(&data);
        let mut iterations = Vec::with_capacity(output_count);
        let mut zs = Vec::with_capacity(output_count);
        for p in pixels {
            iterations.push(p.iter);
            zs.push(Complex64::new(p.z_re as f64, p.z_im as f64));
        }
        drop(data);
        readback_buffer.unmap();

        Some((iterations, zs))
    }
}

#[allow(dead_code)]
struct PipelinesF64 {
    mandelbrot: wgpu::ComputePipeline,
    julia: wgpu::ComputePipeline,
    burning_ship: wgpu::ComputePipeline,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParamsF32 {
    center_x: f32,
    center_y: f32,
    span_x: f32,
    span_y: f32,
    seed_re: f32,
    seed_im: f32,
    width: u32,
    height: u32,
    iter_max: u32,
    plane_transform: u32,
    bailout: f32,
    _pad2: [f32; 3],
    _pad3: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParamsF64 {
    center_x: f64,
    center_y: f64,
    span_x: f64,
    span_y: f64,
    seed_re: f64,
    seed_im: f64,
    width: u32,
    height: u32,
    iter_max: u32,
    _pad: u32,
    bailout: f64,
    _pad2: f64,
}

impl ParamsF32 {
    fn from_params(params: &FractalParams, seed: Option<Complex64>) -> Self {
        let seed = seed.unwrap_or(Complex64::new(0.0, 0.0));
        Self {
            center_x: params.center_x as f32,
            center_y: params.center_y as f32,
            span_x: params.span_x as f32,
            span_y: params.span_y as f32,
            seed_re: seed.re as f32,
            seed_im: seed.im as f32,
            width: params.width,
            height: params.height,
            iter_max: params.iteration_max,
            plane_transform: params.plane_transform.id() as u32,
            bailout: params.bailout as f32,
            _pad2: [0.0; 3],
            _pad3: [0; 4],
        }
    }
}

impl ParamsF64 {
    #[allow(dead_code)]
    fn from_params(params: &FractalParams, seed: Option<Complex64>) -> Self {
        let seed = seed.unwrap_or(Complex64::new(0.0, 0.0));
        Self {
            center_x: params.center_x,
            center_y: params.center_y,
            span_x: params.span_x,
            span_y: params.span_y,
            seed_re: seed.re,
            seed_im: seed.im,
            width: params.width,
            height: params.height,
            iter_max: params.iteration_max,
            _pad: 0,
            bailout: params.bailout,
            _pad2: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PixelOut {
    iter: u32,
    z_re: f32,
    z_im: f32,
    flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ZRef {
    re: f32,
    im: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlaNode {
    a_re: f32,
    a_im: f32,
    b_re: f32,
    b_im: f32,
    c_re: f32,
    c_im: f32,
    validity: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlaMeta {
    level_offsets: [u32; MAX_LEVELS],
    level_lengths: [u32; MAX_LEVELS],
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PerturbParams {
    center_x: f32,
    center_y: f32,
    span_x: f32,
    span_y: f32,
    cref_x: f32,
    cref_y: f32,
    width: u32,
    height: u32,
    iter_max: u32,
    bailout: f32,
    bla_levels: u32,
    fractal_kind: u32,
    glitch_tolerance: f32,
    series_order: u32,
    series_threshold: f32,
    _pad_align: u32,
}

struct ReuseData<'a> {
    iterations: &'a [u32],
    zs: &'a [Complex64],
    width: u32,
    ratio: u32,
}

fn build_reuse<'a>(
    params: &FractalParams,
    reuse: Option<(&'a [u32], &'a [Complex64], u32, u32)>,
) -> Option<ReuseData<'a>> {
    let (iterations, zs, width, height) = reuse?;
    if width == 0 || height == 0 {
        return None;
    }
    let expected_len = (width * height) as usize;
    if iterations.len() != expected_len || zs.len() != expected_len {
        return None;
    }
    if params.width % width != 0 || params.height % height != 0 {
        return None;
    }
    let ratio_x = params.width / width;
    let ratio_y = params.height / height;
    if ratio_x < 2 || ratio_y < 2 || ratio_x != ratio_y {
        return None;
    }
    Some(ReuseData {
        iterations,
        zs,
        width,
        ratio: ratio_x,
    })
}
