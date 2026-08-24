//! Contrat nommé d'une invocation du dispatcher de rendu.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use num_complex::Complex64;

use crate::fractal::wisdom::{self, WisdomPlan};
use crate::fractal::xaos::XaosMap;
use crate::fractal::FractalParams;
use crate::render::tiles::TileOpts;

/// Plan global résolu, device compris. Il est spécialisé juste avant l'appel
/// du backend afin que les frontières CPU/GPU restent impossibles à confondre.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderPlan(WisdomPlan);

impl RenderPlan {
    pub fn auto(params: &FractalParams, gpu_available: bool) -> Self {
        Self(wisdom::auto_plan(params, gpu_available))
    }

    pub fn for_device(params: &FractalParams, device: wisdom::Device) -> Self {
        Self(wisdom::plan_for(params, device))
    }

    pub fn wisdom(self) -> WisdomPlan {
        self.0
    }

    pub fn into_cpu(self) -> Option<CpuRenderPlan> {
        (self.0.device == wisdom::Device::Cpu).then_some(CpuRenderPlan(self.0))
    }

    pub fn into_gpu(self) -> Option<GpuRenderPlan> {
        (self.0.device == wisdom::Device::Gpu).then_some(GpuRenderPlan(self.0))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpuRenderPlan(WisdomPlan);

impl CpuRenderPlan {
    pub fn for_params(params: &FractalParams) -> Self {
        RenderPlan::for_device(params, wisdom::Device::Cpu)
            .into_cpu()
            .expect("un plan demandé pour CPU doit rester CPU")
    }

    pub fn wisdom(self) -> WisdomPlan {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuRenderPlan(WisdomPlan);

impl GpuRenderPlan {
    pub fn for_params(params: &FractalParams) -> Self {
        RenderPlan::for_device(params, wisdom::Device::Gpu)
            .into_gpu()
            .expect("un plan demandé pour GPU doit rester GPU")
    }

    pub fn wisdom(self) -> WisdomPlan {
        self.0
    }
}

/// Buffers d'une passe progressive antérieure, avec ses dimensions.
#[derive(Clone, Copy)]
pub struct ProgressiveReuse<'a> {
    pub iterations: &'a [u32],
    pub zs: &'a [Complex64],
    pub width: u32,
    pub height: u32,
}

impl<'a> From<(&'a [u32], &'a [Complex64], u32, u32)> for ProgressiveReuse<'a> {
    fn from(value: (&'a [u32], &'a [Complex64], u32, u32)) -> Self {
        Self {
            iterations: value.0,
            zs: value.1,
            width: value.2,
            height: value.3,
        }
    }
}

/// Tout ce qui décrit UNE demande de rendu. Le cache d'orbite reste un état
/// in/out séparé : il appartient au moteur entre plusieurs requêtes.
pub struct RenderRequest<'a> {
    pub params: &'a FractalParams,
    pub cancel: &'a Arc<AtomicBool>,
    pub progressive_reuse: Option<ProgressiveReuse<'a>>,
    pub xaos: Option<&'a XaosMap>,
    pub tiles: Option<&'a TileOpts<'a>>,
    /// Plan CPU déjà résolu par l'orchestrateur. `None` laisse le dispatcher
    /// le calculer une fois au dernier moment.
    pub plan: Option<CpuRenderPlan>,
}

impl<'a> RenderRequest<'a> {
    pub fn new(params: &'a FractalParams, cancel: &'a Arc<AtomicBool>) -> Self {
        Self {
            params,
            cancel,
            progressive_reuse: None,
            xaos: None,
            tiles: None,
            plan: None,
        }
    }

    pub fn with_progressive_reuse(mut self, reuse: ProgressiveReuse<'a>) -> Self {
        self.progressive_reuse = Some(reuse);
        self
    }

    pub fn with_xaos(mut self, xaos: &'a XaosMap) -> Self {
        self.xaos = Some(xaos);
        self
    }

    pub fn with_tiles(mut self, tiles: &'a TileOpts<'a>) -> Self {
        self.tiles = Some(tiles);
        self
    }

    pub fn with_plan(mut self, plan: CpuRenderPlan) -> Self {
        self.plan = Some(plan);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::{default_params_for_type, wisdom, FractalType};

    #[test]
    fn carries_a_resolved_cpu_plan() {
        let params = default_params_for_type(FractalType::Mandelbrot, 8, 6);
        let cancel = Arc::new(AtomicBool::new(false));
        let plan = CpuRenderPlan::for_params(&params);
        let request = RenderRequest::new(&params, &cancel).with_plan(plan);
        assert_eq!(request.plan, Some(plan));
        assert_eq!(plan.wisdom().device, wisdom::Device::Cpu);
    }

    #[test]
    fn global_plan_specializes_only_to_its_selected_device() {
        let params = default_params_for_type(FractalType::Mandelbrot, 8, 6);
        let cpu = RenderPlan::for_device(&params, wisdom::Device::Cpu);
        assert!(cpu.into_cpu().is_some());
        assert!(cpu.into_gpu().is_none());

        let gpu = RenderPlan::for_device(&params, wisdom::Device::Gpu);
        assert!(gpu.into_gpu().is_some());
        assert!(gpu.into_cpu().is_none());
    }
}
