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
pub mod pixel_loop_exp;

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
/// - `Rot{cos,sin}` : z := z * (cos + sin·i) (rotation complexe, F3 op_rot)
#[derive(Clone, Copy, Debug, PartialEq)]
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
    /// Rotation complexe : z := z * (cos + sin·i).
    /// Aligné F3 `op_rot` (`hybrid.h:85,116`, `types.h:115`). Permet
    /// d'encoder l'orientation d'un minibrot dans le bytecode plutôt qu'au
    /// niveau du mapping pixel→c, et débloque la parité avec les TOML F3
    /// qui contiennent `[[formula]]` rotation. La transformation est
    /// linéaire en δ : `δ' = δ · (cos + sin·i)`, donc BLA-compatible
    /// (matrice constante de rotation, det = 1).
    ///
    /// Pas `Eq`/`Hash` à cause des f64 — on accepte cette restriction
    /// puisque les comparaisons d'opcodes restent structurelles via
    /// `PartialEq`.
    #[allow(dead_code)]
    Rot { cos_theta: f64, sin_theta: f64 },
}

impl Op {
    /// Tag entier de l'opcode (mêmes valeurs que l'ordre de déclaration,
    /// utilisées par le shader WGSL `bytecode_kernel.wgsl`).
    ///
    /// `Op::Rot` ne peut pas être encodé dans le buffer u32 GPU actuel
    /// (payload (f64, f64)) — le caller GPU doit refuser ces formules
    /// avant l'upload. CPU et GMP gèrent `Rot` directement via le match.
    #[allow(dead_code)]
    pub fn opcode_tag(self) -> u32 {
        match self {
            Op::Sqr => 0,
            Op::Mul => 1,
            Op::Store => 2,
            Op::AbsX => 3,
            Op::AbsY => 4,
            Op::NegX => 5,
            Op::NegY => 6,
            Op::Add => 7,
            Op::Rot { .. } => 8,
        }
    }
}

/// Une phase = séquence d'opcodes appliquée par itération.
///
/// Doit se terminer par `Op::Add`. Les puissances Multibrot (`z^N + c`) sont
/// décomposées en chaîne `Sqr` + `Mul` (avec `Store` initial) via décomposition
/// binaire de l'exposant.
#[derive(Clone, Debug, PartialEq)]
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
#[derive(Clone, Debug, PartialEq)]
pub struct Formula {
    pub phases: Vec<Phase>,
}

impl Formula {
    pub fn single(phase: Phase) -> Self {
        Self { phases: vec![phase] }
    }

    /// Construit une formule hybride à partir d'une liste de phases.
    ///
    /// Les phases sont itérées cycliquement : itération `n` applique
    /// `phases[n % phases.len()]`. Permet "5× Mandelbrot puis 3× Burning Ship"
    /// en passant `vec![mb_phase; 5].extend(vec![bs_phase; 3])`.
    ///
    /// Cf. `docs/fraktaler-3-analysis.md` §2 (chaînage de phases = hybrides).
    #[allow(dead_code)]
    pub fn hybrid(phases: Vec<Phase>) -> Self {
        assert!(!phases.is_empty(), "Formula::hybrid : phases ne peut pas être vide");
        Self { phases }
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
