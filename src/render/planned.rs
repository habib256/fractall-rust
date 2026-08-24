//! Routage commun d'un plan global vers le backend réellement exécutable.

use std::sync::Arc;

use crate::fractal::perturbation::ReferenceOrbitCache;
use crate::fractal::wisdom::{Algorithm, Device};
use crate::gpu::GpuRenderer;

use super::{CpuRenderPlan, RenderOutput, RenderPlan, RenderRequest};

pub struct PlannedRenderRequest<'a> {
    pub plan: RenderPlan,
    pub cpu: RenderRequest<'a>,
}

impl<'a> PlannedRenderRequest<'a> {
    pub fn new(plan: RenderPlan, cpu: RenderRequest<'a>) -> Self {
        Self { plan, cpu }
    }
}

pub struct PlannedRenderOutput {
    pub output: RenderOutput,
    pub device: Device,
    pub used_perturbation: bool,
    pub fell_back_to_cpu: bool,
}

/// Exécute le device choisi par le plan et retombe sur le dispatcher CPU si le
/// GPU est absent ou refuse la frame. Le fallback est centralisé ici afin que
/// CLI et GUI ne réimplémentent plus cette politique.
pub fn render_planned(
    mut request: PlannedRenderRequest<'_>,
    gpu: Option<&GpuRenderer>,
    orbit_cache: &mut Option<Arc<ReferenceOrbitCache>>,
) -> Option<PlannedRenderOutput> {
    let selected_device = request.plan.wisdom().device;
    if let Some(gpu_plan) = request.plan.into_gpu() {
        if let Some(gpu) = gpu {
            if let Some(result) = gpu.render_dispatch(
                gpu_plan,
                request.cpu.params,
                request.cpu.cancel,
                request
                    .cpu
                    .progressive_reuse
                    .map(|reuse| (reuse.iterations, reuse.zs, reuse.width, reuse.height)),
                orbit_cache.as_ref(),
            ) {
                if let Some(cache) = result.orbit_cache {
                    *orbit_cache = Some(cache);
                }
                return Some(PlannedRenderOutput {
                    output: RenderOutput::without_extras(result.iterations, result.zs),
                    device: Device::Gpu,
                    used_perturbation: result.used_perturbation,
                    fell_back_to_cpu: false,
                });
            }
        }
    }

    // Si le plan global avait choisi CPU, préserver EXACTEMENT sa décision.
    // Un fallback depuis GPU exige en revanche un nouveau plan CPU puisque
    // algorithme/tier/capacités dépendent du device.
    let cpu_plan = request
        .plan
        .into_cpu()
        .unwrap_or_else(|| CpuRenderPlan::for_params(request.cpu.params));
    let used_perturbation = cpu_plan.wisdom().algorithm == Algorithm::Perturbation;
    request.cpu.plan = Some(cpu_plan);
    super::render_request(request.cpu, orbit_cache).map(|output| PlannedRenderOutput {
        output,
        device: Device::Cpu,
        used_perturbation,
        fell_back_to_cpu: selected_device == Device::Gpu,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use super::*;
    use crate::fractal::{default_params_for_type, FractalType};

    #[test]
    fn unavailable_selected_gpu_falls_back_to_cpu() {
        let params = default_params_for_type(FractalType::Mandelbrot, 12, 8);
        let cancel = Arc::new(AtomicBool::new(false));
        let plan = RenderPlan::for_device(&params, Device::Gpu);
        let request = PlannedRenderRequest::new(plan, RenderRequest::new(&params, &cancel));
        let mut cache = None;
        let result = render_planned(request, None, &mut cache).unwrap();
        assert_eq!(result.device, Device::Cpu);
        assert!(result.fell_back_to_cpu);
        assert_eq!(result.output.iterations.len(), 12 * 8);
    }

    #[test]
    fn selected_cpu_matches_direct_dispatch() {
        let params = default_params_for_type(FractalType::Mandelbrot, 12, 8);
        let cancel = Arc::new(AtomicBool::new(false));
        let plan = RenderPlan::for_device(&params, Device::Cpu);
        let request = PlannedRenderRequest::new(plan, RenderRequest::new(&params, &cancel));
        let mut planned_cache = None;
        let planned = render_planned(request, None, &mut planned_cache).unwrap();

        let mut direct_cache = None;
        let direct =
            super::super::render_request(RenderRequest::new(&params, &cancel), &mut direct_cache)
                .unwrap();
        assert_eq!(planned.device, Device::Cpu);
        assert!(!planned.fell_back_to_cpu);
        assert_eq!(
            planned.used_perturbation,
            plan.wisdom().algorithm == Algorithm::Perturbation
        );
        assert_eq!(planned.output.iterations, direct.iterations);
        assert_eq!(planned.output.zs, direct.zs);
    }
}
