pub(crate) mod dd_trig;
pub(crate) mod scalar_qd;
pub(crate) mod scalar_lanes;
pub mod escape_time;
pub mod output;
pub mod planned;
pub mod request;
pub mod scalar_reliability;
pub mod std_reliability;
#[cfg(test)]
mod stateful_tests;
pub mod tiles;

#[allow(unused_imports)]
pub use output::{ensure_required_channels, required_channels, ChannelRequirements, RenderOutput};
pub use planned::{render_planned, PlannedRenderOutput, PlannedRenderRequest};
pub use request::{CpuRenderPlan, GpuRenderPlan, ProgressiveReuse, RenderPlan, RenderRequest};

#[allow(unused_imports)]
pub use escape_time::render_escape_time;
pub use escape_time::render_request;
