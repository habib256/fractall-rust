pub mod types;
pub mod definitions;
pub mod iterations;
pub mod jitter;
pub mod bytecode;
pub mod gmp;
pub mod lyapunov;
pub mod vectorial;
pub mod buddhabrot;
pub mod perturbation;
pub mod wisdom;
pub mod orbit_traps;

pub use types::{AlgorithmMode, FractalParams, FractalResult, FractalType, OutColoringMode, PlaneTransform, ColorSpace};
// `apply_lyapunov_preset` et `LyapunovPreset` ne sont pas consommés par
// `fractall-quality` (qui ne touche pas Lyapunov) ; les autres binaires les
// utilisent. Silenciation locale du warning unused_imports pour ce binaire.
#[allow(unused_imports)]
pub use definitions::{default_params_for_type, apply_lyapunov_preset};
#[allow(unused_imports)]
pub use lyapunov::{render_lyapunov, LyapunovPreset};
pub use vectorial::{render_von_koch, render_dragon};
#[allow(unused_imports)]
pub use buddhabrot::{render_buddhabrot, render_nebulabrot, render_antibuddhabrot};

