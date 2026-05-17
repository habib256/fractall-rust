//! Bytecode hybride à la Fraktaler-3 pour les formules escape-time.
//!
//! Une formule = liste de **phases** itérées cycliquement. Chaque phase = liste
//! d'**opcodes** parmi un jeu de 8 (cf. F3 `hybrid.h`, `param.cc::compile_formula`).
//! Cette représentation unifie Mandelbrot, Burning Ship, Tricorn, Celtic, Buffalo,
//! Perpendicular Burning Ship, Multibrot et leurs variantes Julia derrière un
//! unique interpréteur. Les hybrides (ex. "5× Mandelbrot puis 3× Burning Ship")
//! deviennent un simple `Vec<Phase>` à plusieurs entrées.
//!
//! L'opcode `Add` clôt logiquement une phase (z := z + c, +1 itération). Il est
//! attendu en dernière position de chaque phase.

use crate::fractal::FractalType;

pub mod bla_dual;
pub mod compile;
pub mod delta_form;
pub mod interp;
pub mod interp_gmp;
pub mod pixel_loop;

#[cfg(test)]
mod tests;

pub use bla_dual::build_bla_table_for_formula;
pub use compile::compile_formula;
pub use interp::iterate_bytecode_f64;
pub use interp_gmp::GmpInterpState;

/// Jeu d'opcodes Fraktaler-3 (`hybrid.h`).
///
/// - `Sqr`   : z := z * z
/// - `Mul`   : z := z * stored (utilise le registre `stored` posé par `Store`)
/// - `Store` : stored := z
/// - `AbsX`  : z.re := |z.re|
/// - `AbsY`  : z.im := |z.im|
/// - `NegX`  : z.re := -z.re
/// - `NegY`  : z.im := -z.im (= conjugaison)
/// - `Add`   : z := z + c (fin de phase, increment itération)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    Sqr,
    Mul,
    Store,
    AbsX,
    AbsY,
    /// Pas utilisé par les types actuels — réservé pour les hybrides
    /// (Mandelbar variants où on négocie la partie réelle).
    #[allow(dead_code)]
    NegX,
    NegY,
    Add,
}

/// Une phase = séquence d'opcodes appliquée par itération.
///
/// Doit se terminer par `Op::Add`. Les puissances Multibrot (`z^N + c`) sont
/// décomposées en chaîne `Sqr` + `Mul` (avec `Store` initial) via décomposition
/// binaire de l'exposant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Phase {
    pub ops: Vec<Op>,
}

impl Phase {
    pub fn new(ops: Vec<Op>) -> Self {
        debug_assert!(
            matches!(ops.last(), Some(Op::Add)),
            "Une phase doit se terminer par Op::Add"
        );
        Self { ops }
    }
}

/// Une formule = liste de phases itérées cycliquement.
///
/// `phase = (phase + 1) % phases.len()` après chaque itération complète (cf.
/// F3 `hybrid_render`). Pour une formule mono-phase (Mandelbrot pur), une seule
/// entrée.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Formula {
    pub phases: Vec<Phase>,
}

impl Formula {
    pub fn single(phase: Phase) -> Self {
        Self { phases: vec![phase] }
    }

    /// Indique comment placer pixel et seed dans (z₀, c).
    ///
    /// - `Mandelbrot` : z₀ = seed (souvent 0), c = pixel
    /// - `Julia`      : z₀ = pixel,            c = seed
    ///
    /// Cette distinction est *appelante* et non intrinsèque au bytecode : la
    /// formule de Burning Ship Julia est strictement identique à Burning Ship,
    /// seule la convention d'appel change.
    pub fn is_julia_for(ft: FractalType) -> bool {
        use FractalType::*;
        matches!(
            ft,
            Julia
                | JuliaSin
                | BarnsleyJulia
                | MagnetJulia
                | BurningShipJulia
                | TricornJulia
                | CelticJulia
                | BuffaloJulia
                | MultibrotJulia
                | PerpendicularBurningShipJulia
                | AlphaMandelbrotJulia
        )
    }
}
