use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::mpsc::RecvTimeoutError;
use std::time::Duration;

use bytemuck::{Pod, Zeroable};
use num_complex::Complex64;
use rayon::prelude::*;
use wgpu::util::DeviceExt;

use crate::fractal::{FractalParams, FractalType};
use crate::fractal::gmp::{complex_from_xy, complex_to_complex64, iterate_point_mpc, MpcParams};
use crate::fractal::perturbation::orbit::{compute_reference_orbit_cached, ReferenceOrbitCache};

const WORKGROUP_SIZE: u32 = 16;
const MAX_LEVELS: usize = 17;

pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_f32: wgpu::ComputePipeline,
    pipeline_f64: Option<PipelinesF64>,
    pipeline_julia_f32: wgpu::ComputePipeline,
    pipeline_burning_ship_f32: wgpu::ComputePipeline,
    pipeline_perturbation: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group_layout_perturb: wgpu::BindGroupLayout,
    supports_f64: bool,
}

impl GpuRenderer {
    pub fn new() -> Option<Self> {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await?;

            let adapter_features = adapter.features();
            let supports_f64 = adapter_features.contains(wgpu::Features::SHADER_F64);
            let required_features = if supports_f64 {
                wgpu::Features::SHADER_F64
            } else {
                wgpu::Features::empty()
            };

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

            let pipeline_f64 = if supports_f64 {
                let shader_f64 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("mandelbrot-f64"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("mandelbrot_f64.wgsl").into()),
                });

                let pipeline_mandelbrot_f64 =
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("mandelbrot-pipeline-f64"),
                        layout: Some(&pipeline_layout),
                        module: &shader_f64,
                        entry_point: "main",
                    });

                let shader_julia_f64 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("julia-f64"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("julia_f64.wgsl").into()),
                });

                let pipeline_julia_f64 =
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("julia-pipeline-f64"),
                        layout: Some(&pipeline_layout),
                        module: &shader_julia_f64,
                        entry_point: "main",
                    });

                let shader_burning_ship_f64 =
                    device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("burning-ship-f64"),
                        source: wgpu::ShaderSource::Wgsl(
                            include_str!("burning_ship_f64.wgsl").into(),
                        ),
                    });

                let pipeline_burning_ship_f64 =
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("burning-ship-pipeline-f64"),
                        layout: Some(&pipeline_layout),
                        module: &shader_burning_ship_f64,
                        entry_point: "main",
                    });

                Some(PipelinesF64 {
                    mandelbrot: pipeline_mandelbrot_f64,
                    julia: pipeline_julia_f64,
                    burning_ship: pipeline_burning_ship_f64,
                })
            } else {
                None
            };

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
                bind_group_layout_perturb,
                supports_f64,
            })
        })
    }

    pub fn render_mandelbrot(
        &self,
        params: &FractalParams,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Option<(Vec<u32>, Vec<Complex64>)> {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        if self.supports_f64 {
            self.render_mandelbrot_f64(params, cancel)
        } else {
            self.render_mandelbrot_f32(params, cancel)
        }
    }

    pub fn render_julia(
        &self,
        params: &FractalParams,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Option<(Vec<u32>, Vec<Complex64>)> {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        if self.supports_f64 {
            self.render_julia_f64(params, cancel)
        } else {
            self.render_julia_f32(params, cancel)
        }
    }

    pub fn render_burning_ship(
        &self,
        params: &FractalParams,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Option<(Vec<u32>, Vec<Complex64>)> {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        if self.supports_f64 {
            self.render_burning_ship_f64(params, cancel)
        } else {
            self.render_burning_ship_f32(params, cancel)
        }
    }

    pub fn precision_label(&self) -> &'static str {
        if self.supports_f64 {
            "f64"
        } else {
            "f32"
        }
    }

    /// Render perturbation with optional orbit cache support.
    /// Returns the result and the updated cache for reuse in subsequent frames.
    pub fn render_perturbation_with_cache(
        &self,
        params: &FractalParams,
        cancel: &std::sync::atomic::AtomicBool,
        reuse: Option<(&[u32], &[Complex64], u32, u32)>,
        orbit_cache: Option<&Arc<ReferenceOrbitCache>>,
    ) -> Option<((Vec<u32>, Vec<Complex64>), Arc<ReferenceOrbitCache>)> {
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

        // Use cached orbit/BLA or compute fresh
        let cache = compute_reference_orbit_cached(params, Some(cancel), orbit_cache)?;
        let ref_orbit = &cache.orbit;
        let bla_table = &cache.bla_table;
        let supports_bla = matches!(params.fractal_type, FractalType::Mandelbrot | FractalType::Julia);
        let bla_levels = if supports_bla {
            bla_table.levels.len().min(MAX_LEVELS)
        } else {
            0
        };
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
                validity: node.validity_radius as f32,
                _pad: 0.0,
            }));
        }

        let z_ref: Vec<ZRef> = ref_orbit
            .z_ref
            .iter()
            .map(|z| ZRef {
                re: z.re as f32,
                im: z.im as f32,
            })
            .collect();
        let iter_max = params
            .iteration_max
            .min(ref_orbit.z_ref.len().saturating_sub(1) as u32);

        let width = params.width as usize;
        let height = params.height as usize;
        if width == 0 || height == 0 {
            return Some(((Vec::new(), Vec::new()), cache));
        }

        let output_count = width * height;
        let output_size = (output_count * std::mem::size_of::<PixelOut>()) as u64;

        let reuse = build_reuse(params, reuse);
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

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("perturb-params"),
            contents: bytemuck::bytes_of(&PerturbParams {
                xmin: params.xmin as f32,
                xmax: params.xmax as f32,
                ymin: params.ymin as f32,
                ymax: params.ymax as f32,
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
                _pad_align: [0; 3],
                _pad0: [0; 4],
                _pad: [0; 4],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let meta_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("perturb-bla-meta"),
            contents: bytemuck::bytes_of(&BlaMeta {
                level_offsets,
                level_lengths,
                _pad: [0; 2],
            }),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bla_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("perturb-bla-nodes"),
            contents: bytemuck::cast_slice(&flattened),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let zref_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("perturb-zref"),
            contents: bytemuck::cast_slice(&z_ref),
            usage: wgpu::BufferUsages::STORAGE,
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
        let mut glitched_indices = Vec::new();
        for (idx, p) in pixels.iter().enumerate() {
            if p.flags != 0 {
                glitched_indices.push(idx as u32);
            }
            iterations.push(p.iter);
            zs.push(Complex64::new(p.z_re as f64, p.z_im as f64));
        }
        drop(data);
        readback_buffer.unmap();

        // Parallel glitch correction using Rayon
        if !glitched_indices.is_empty() {
            let gmp_params = MpcParams::from_params(params);
            let prec = gmp_params.prec;
            let x_range = params.xmax - params.xmin;
            let y_range = params.ymax - params.ymin;
            let x_step = x_range / params.width as f64;
            let y_step = y_range / params.height as f64;
            let width = params.width;
            let xmin = params.xmin;
            let ymin = params.ymin;

            // Collect results in parallel
            let corrections: Vec<_> = glitched_indices
                .par_iter()
                .map(|&idx| {
                    let x = (idx % width) as f64;
                    let y = (idx / width) as f64;
                    let xg = x_step * x + xmin;
                    let yg = y_step * y + ymin;
                    let z_pixel = complex_from_xy(
                        prec,
                        rug::Float::with_val(prec, xg),
                        rug::Float::with_val(prec, yg),
                    );
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

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}-bind-group-f64")),
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

struct PipelinesF64 {
    mandelbrot: wgpu::ComputePipeline,
    julia: wgpu::ComputePipeline,
    burning_ship: wgpu::ComputePipeline,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParamsF32 {
    xmin: f32,
    xmax: f32,
    ymin: f32,
    ymax: f32,
    seed_re: f32,
    seed_im: f32,
    width: u32,
    height: u32,
    iter_max: u32,
    _pad: u32,
    bailout: f32,
    _pad2: [f32; 3],
    _pad3: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParamsF64 {
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
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
            xmin: params.xmin as f32,
            xmax: params.xmax as f32,
            ymin: params.ymin as f32,
            ymax: params.ymax as f32,
            seed_re: seed.re as f32,
            seed_im: seed.im as f32,
            width: params.width,
            height: params.height,
            iter_max: params.iteration_max,
            _pad: 0,
            bailout: params.bailout as f32,
            _pad2: [0.0; 3],
            _pad3: [0; 4],
        }
    }
}

impl ParamsF64 {
    fn from_params(params: &FractalParams, seed: Option<Complex64>) -> Self {
        let seed = seed.unwrap_or(Complex64::new(0.0, 0.0));
        Self {
            xmin: params.xmin,
            xmax: params.xmax,
            ymin: params.ymin,
            ymax: params.ymax,
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
    xmin: f32,
    xmax: f32,
    ymin: f32,
    ymax: f32,
    cref_x: f32,
    cref_y: f32,
    width: u32,
    height: u32,
    iter_max: u32,
    bailout: f32,
    bla_levels: u32,
    fractal_kind: u32,
    glitch_tolerance: f32,
    _pad_align: [u32; 3],
    _pad0: [u32; 4],
    _pad: [u32; 4],
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
