use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use num_complex::Complex64;
use wgpu::util::DeviceExt;

use crate::fractal::FractalParams;
use crate::fractal::perturbation::bla::build_bla_table;
use crate::fractal::perturbation::orbit::compute_reference_orbit;

const WORKGROUP_SIZE: u32 = 16;
const MAX_LEVELS: usize = 17;

pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_f32: wgpu::ComputePipeline,
    pipeline_f64: Option<wgpu::ComputePipeline>,
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

            let pipeline_f64 = if supports_f64 {
                let shader_f64 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("mandelbrot-f64"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("mandelbrot_f64.wgsl").into()),
                });
                Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("mandelbrot-pipeline-f64"),
                    layout: Some(&pipeline_layout),
                    module: &shader_f64,
                    entry_point: "main",
                }))
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

    pub fn precision_label(&self) -> &'static str {
        if self.supports_f64 {
            "f64"
        } else {
            "f32"
        }
    }

    pub fn render_perturbation(
        &self,
        params: &FractalParams,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Option<(Vec<u32>, Vec<Complex64>)> {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        let ref_orbit = compute_reference_orbit(params, Some(cancel))?;
        let bla_table = build_bla_table(&ref_orbit.z_ref, params);
        let bla_levels = bla_table.levels.len().min(MAX_LEVELS);
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

        let width = params.width as usize;
        let height = params.height as usize;
        if width == 0 || height == 0 {
            return Some((Vec::new(), Vec::new()));
        }

        let output_count = width * height;
        let output_size = (output_count * std::mem::size_of::<PixelOut>()) as u64;

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("perturb-output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
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
                iter_max: params.iteration_max,
                bailout: params.bailout as f32,
                bla_levels: bla_levels as u32,
                _pad: 0,
                _pad2: [0.0; 3],
                _pad3: 0,
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

        let buffer_slice = readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        loop {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                readback_buffer.unmap();
                return None;
            }
            if let Ok(result) = receiver.try_recv() {
                result.ok()?;
                break;
            }
            self.device.poll(wgpu::Maintain::Poll);
            std::thread::sleep(std::time::Duration::from_millis(1));
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

    fn render_mandelbrot_f32(
        &self,
        params: &FractalParams,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Option<(Vec<u32>, Vec<Complex64>)> {
        let width = params.width as usize;
        let height = params.height as usize;
        if width == 0 || height == 0 {
            return Some((Vec::new(), Vec::new()));
        }
        let output_count = width * height;
        let output_size = (output_count * std::mem::size_of::<PixelOut>()) as u64;

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mandelbrot-output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mandelbrot-readback"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mandelbrot-params-f32"),
            contents: bytemuck::bytes_of(&ParamsF32 {
                xmin: params.xmin as f32,
                xmax: params.xmax as f32,
                ymin: params.ymin as f32,
                ymax: params.ymax as f32,
                width: params.width,
                height: params.height,
                iter_max: params.iteration_max,
                _pad: 0,
                bailout: params.bailout as f32,
                _pad2: [0.0; 3],
                _pad3: [0; 4],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mandelbrot-bind-group-f32"),
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
                    label: Some("mandelbrot-encoder-f32"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mandelbrot-pass-f32"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_f32);
            pass.set_bind_group(0, &bind_group, &[]);
            let dispatch_x = (params.width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            let dispatch_y = (params.height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        loop {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                readback_buffer.unmap();
                return None;
            }
            if let Ok(result) = receiver.try_recv() {
                result.ok()?;
                break;
            }
            self.device.poll(wgpu::Maintain::Poll);
            std::thread::sleep(std::time::Duration::from_millis(1));
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

    fn render_mandelbrot_f64(
        &self,
        params: &FractalParams,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Option<(Vec<u32>, Vec<Complex64>)> {
        let pipeline = self.pipeline_f64.as_ref()?;
        let width = params.width as usize;
        let height = params.height as usize;
        if width == 0 || height == 0 {
            return Some((Vec::new(), Vec::new()));
        }
        let output_count = width * height;
        let output_size = (output_count * std::mem::size_of::<PixelOut>()) as u64;

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mandelbrot-output-f64"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mandelbrot-readback-f64"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mandelbrot-params-f64"),
            contents: bytemuck::bytes_of(&ParamsF64 {
                xmin: params.xmin,
                xmax: params.xmax,
                ymin: params.ymin,
                ymax: params.ymax,
                width: params.width,
                height: params.height,
                iter_max: params.iteration_max,
                _pad: 0,
                bailout: params.bailout,
                _pad2: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mandelbrot-bind-group-f64"),
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
                    label: Some("mandelbrot-encoder-f64"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mandelbrot-pass-f64"),
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

        let buffer_slice = readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        loop {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                readback_buffer.unmap();
                return None;
            }
            if let Ok(result) = receiver.try_recv() {
                result.ok()?;
                break;
            }
            self.device.poll(wgpu::Maintain::Poll);
            std::thread::sleep(std::time::Duration::from_millis(1));
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParamsF32 {
    xmin: f32,
    xmax: f32,
    ymin: f32,
    ymax: f32,
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
    width: u32,
    height: u32,
    iter_max: u32,
    _pad: u32,
    bailout: f64,
    _pad2: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PixelOut {
    iter: u32,
    z_re: f32,
    z_im: f32,
    _pad: u32,
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
    _pad: u32,
    _pad2: [f32; 3],
    _pad3: u32,
}
