//! Compilation `FractalType` → `Formula` bytecode.
//!
//! Mirrors `param.cc::compile_formula` de Fraktaler-3. Pour chaque type escape-time
//! supporté, produit la liste d'opcodes correspondante. Renvoie `None` pour les
//! types qui ne sont pas (encore) representables en bytecode 8-opcodes — ces
//! types restent en codepath dédié dans `iterations.rs`.

use super::{Formula, Op, Phase};
use crate::fractal::FractalType;

/// Compile un type de fractale en bytecode hybride.
///
/// Pour Multibrot/MultibrotJulia, `multibrot_power` doit être un entier `≥ 2`
/// (la décomposition en `Sqr`+`Mul` ne supporte que les puissances entières).
/// Pour les puissances non-entières, retourne `None` (le path `powf` reste actif).
pub fn compile_formula(ft: FractalType, multibrot_power: f64) -> Option<Formula> {
    phase_ops_for_type(ft, multibrot_power).map(|ops| Formula::single(Phase::new(ops)))
}

/// Compile une formule **HYBRIDE multi-phase** (G4) : une phase par entrée de
/// `phase_types`, itérées CYCLIQUEMENT (`phases[n % len]`, cf. `Formula::hybrid`
/// et `GmpInterpState::step`). Chaque phase réutilise EXACTEMENT le bytecode du
/// type escape-time correspondant ([`phase_ops_for_type`]) → composition pure,
/// aucune nouvelle sémantique d'opcode. Ex. `[Mandelbrot, BurningShip]` alterne
/// `z²+c` et `(|x|+i|y|)²+c` à chaque itération (le « Mandel-Ship » de F3) ;
/// répéter un type (`[M, M, BS]`) donne « 2× Mandelbrot puis 1× Burning Ship ».
///
/// `None` si `phase_types` est vide ou si un type n'est pas représentable en
/// bytecode (Newton, Magnet, …). `multibrot_power` s'applique aux phases
/// Multibrot. Réf : `docs/fraktaler-3-analysis.md` §2 (chaînage de phases).
// G4 jalon 1 : brique de compilation (testée) ; le câblage render
// (`params.hybrid_phases` → reference orbit + pixel-loop, déjà multi-phase-ready)
// est le jalon SUIVANT — d'où `allow(dead_code)` en attendant le consommateur.
#[allow(dead_code)]
pub fn compile_hybrid_formula(
    phase_types: &[FractalType],
    multibrot_power: f64,
) -> Option<Formula> {
    if phase_types.is_empty() {
        return None;
    }
    let phases: Option<Vec<Phase>> = phase_types
        .iter()
        .map(|&ft| phase_ops_for_type(ft, multibrot_power).map(Phase::new))
        .collect();
    phases.map(Formula::hybrid)
}

/// Bytecode d'UNE phase pour un type escape-time (partie réutilisable de
/// [`compile_formula`] et [`compile_hybrid_formula`]). `None` = type non
/// représentable en bytecode 8-opcodes. **Behavior-preserving** : les mêmes
/// séquences d'ops qu'avant (verrouillées par les goldens par-type).
fn phase_ops_for_type(ft: FractalType, multibrot_power: f64) -> Option<Vec<Op>> {
    use FractalType::*;
    match ft {
        // Mandelbrot et Julia : même bytecode, seule la convention d'appel diffère.
        Mandelbrot | Julia => Some(vec![Op::Sqr, Op::Add]),

        // Burning Ship : (|x| + i|y|)² + c
        BurningShip | BurningShipJulia => Some(vec![Op::AbsX, Op::AbsY, Op::Sqr, Op::Add]),

        // Tricorn (Mandelbar) : conj(z)² + c = (negy then sqr)
        Tricorn | TricornJulia => Some(vec![Op::NegY, Op::Sqr, Op::Add]),

        // Celtic : |Re(z²)| + i·Im(z²) + c (l'abs vient APRÈS le carré, sur la partie réelle)
        Celtic | CelticJulia => Some(vec![Op::Sqr, Op::AbsX, Op::Add]),

        // Buffalo : |Re(z²)| + i·|Im(z²)| + c
        Buffalo | BuffaloJulia => Some(vec![Op::Sqr, Op::AbsX, Op::AbsY, Op::Add]),

        // Perpendicular Burning Ship : (x - i|y|)² + c
        // Ordre d'opérations : on rend y positif (AbsY), puis on le négocie (NegY),
        // puis on élève au carré, puis on ajoute c.
        PerpendicularBurningShip | PerpendicularBurningShipJulia => {
            Some(vec![Op::AbsY, Op::NegY, Op::Sqr, Op::Add])
        }

        // Multibrot : z^N + c pour N entier ≥ 2 via décomposition binaire.
        Multibrot | MultibrotJulia => {
            let n_round = multibrot_power.round();
            let is_integer = (multibrot_power - n_round).abs() < 1e-10;
            if !is_integer || n_round < 2.0 {
                return None;
            }
            Some(compile_power(n_round as u32))
        }

        // Hors scope bytecode 8-opcodes : Alpha Mandelbrot (besoin de z² + (z²+c)²),
        // Newton, Phoenix, Magnet, Barnsley, JuliaSin/MandelbrotSin, Mandelbulb,
        // Lyapunov, Buddhabrot/Nebulabrot/Anti, VonKoch/Dragon, Pickover, Nova.
        _ => None,
    }
}

/// Compile `z^n + c` en bytecode pour `n ≥ 2` entier.
///
/// Stratégie binaire : on consomme les bits de `n` du MSB-1 vers le LSB.
/// Pour chaque bit on émet `Sqr` ; si le bit est à 1, on émet `Mul` (multiplication
/// par le `Store` initial = z). Termine par `Add` (z := z + c).
///
/// Exemples :
/// - n=2 (`10`) : pas de bit après le MSB → `Sqr, Add`. Pas besoin de Store.
/// - n=3 (`11`) : MSB seul, bit suivant=1 → `Store, Sqr, Mul, Add`.
/// - n=4 (`100`): deux zéros après MSB → `Sqr, Sqr, Add`.
/// - n=5 (`101`): zéro puis un → `Store, Sqr, Sqr, Mul, Add`.
/// - n=8 (`1000`): trois zéros après MSB → `Sqr, Sqr, Sqr, Add`.
fn compile_power(n: u32) -> Vec<Op> {
    debug_assert!(n >= 2);
    let highest_bit = 31 - n.leading_zeros(); // position du MSB
    let needs_store = n != (1u32 << highest_bit); // un bit additionnel hors MSB
    let mut ops = Vec::with_capacity((highest_bit as usize) * 2 + 2);
    if needs_store {
        ops.push(Op::Store);
    }
    for bit in (0..highest_bit).rev() {
        ops.push(Op::Sqr);
        if (n >> bit) & 1 == 1 {
            ops.push(Op::Mul);
        }
    }
    ops.push(Op::Add);
    ops
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_2_is_sqr_add() {
        assert_eq!(compile_power(2), vec![Op::Sqr, Op::Add]);
    }

    #[test]
    fn power_3_uses_store() {
        assert_eq!(compile_power(3), vec![Op::Store, Op::Sqr, Op::Mul, Op::Add]);
    }

    #[test]
    fn power_4_is_double_sqr() {
        assert_eq!(compile_power(4), vec![Op::Sqr, Op::Sqr, Op::Add]);
    }

    #[test]
    fn power_5_store_sqr_sqr_mul() {
        assert_eq!(
            compile_power(5),
            vec![Op::Store, Op::Sqr, Op::Sqr, Op::Mul, Op::Add]
        );
    }

    #[test]
    fn power_8_is_triple_sqr() {
        assert_eq!(compile_power(8), vec![Op::Sqr, Op::Sqr, Op::Sqr, Op::Add]);
    }

    #[test]
    fn power_7_111_binary() {
        // 7 = 0b111 → MSB puis deux bits à 1 → Store, Sqr, Mul, Sqr, Mul, Add
        assert_eq!(
            compile_power(7),
            vec![Op::Store, Op::Sqr, Op::Mul, Op::Sqr, Op::Mul, Op::Add]
        );
    }

    #[test]
    fn mandelbrot_compiles() {
        let f = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        assert_eq!(f.phases.len(), 1);
        assert_eq!(f.phases[0].ops, vec![Op::Sqr, Op::Add]);
    }

    // ── G4 : compilation hybride multi-phase ───────────────────────────────
    #[test]
    fn hybrid_two_phases_mandel_ship() {
        // [Mandelbrot, BurningShip] = 2 phases, chacune le bytecode de son type.
        let f =
            compile_hybrid_formula(&[FractalType::Mandelbrot, FractalType::BurningShip], 2.0)
                .unwrap();
        assert_eq!(f.phases.len(), 2);
        assert_eq!(f.phases[0].ops, vec![Op::Sqr, Op::Add]);
        assert_eq!(f.phases[1].ops, vec![Op::AbsX, Op::AbsY, Op::Sqr, Op::Add]);
    }

    #[test]
    fn hybrid_single_type_equals_compile_formula() {
        // [Mandelbrot] hybride ≡ compile_formula(Mandelbrot) (une seule phase).
        let h = compile_hybrid_formula(&[FractalType::Mandelbrot], 2.0).unwrap();
        let s = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        assert_eq!(h.phases.len(), 1);
        assert_eq!(h.phases[0].ops, s.phases[0].ops);
    }

    #[test]
    fn hybrid_repeated_type_repeats_phase() {
        // [M, M, BS] = « 2× Mandelbrot puis 1× Burning Ship » (répétition = comptage).
        let f = compile_hybrid_formula(
            &[
                FractalType::Mandelbrot,
                FractalType::Mandelbrot,
                FractalType::BurningShip,
            ],
            2.0,
        )
        .unwrap();
        assert_eq!(f.phases.len(), 3);
        assert_eq!(f.phases[0].ops, f.phases[1].ops); // deux Mandelbrot identiques
        assert_ne!(f.phases[1].ops, f.phases[2].ops); // ≠ Burning Ship
    }

    #[test]
    fn hybrid_multibrot_phase_honours_power() {
        let f = compile_hybrid_formula(&[FractalType::Mandelbrot, FractalType::Multibrot], 3.0)
            .unwrap();
        assert_eq!(f.phases[1].ops, compile_power(3));
    }

    #[test]
    fn hybrid_empty_or_unrepresentable_is_none() {
        assert!(compile_hybrid_formula(&[], 2.0).is_none());
        // Newton n'est pas représentable en bytecode → toute la formule None.
        assert!(compile_hybrid_formula(
            &[FractalType::Mandelbrot, FractalType::Newton],
            2.0
        )
        .is_none());
        // Multibrot puissance non-entière dans une phase → None.
        assert!(
            compile_hybrid_formula(&[FractalType::Multibrot], 2.5).is_none()
        );
    }

    #[test]
    fn burning_ship_compiles() {
        let f = compile_formula(FractalType::BurningShip, 2.0).unwrap();
        assert_eq!(
            f.phases[0].ops,
            vec![Op::AbsX, Op::AbsY, Op::Sqr, Op::Add]
        );
    }

    #[test]
    fn multibrot_non_integer_returns_none() {
        assert!(compile_formula(FractalType::Multibrot, 2.5).is_none());
    }

    #[test]
    fn newton_unsupported() {
        assert!(compile_formula(FractalType::Newton, 2.0).is_none());
    }
}
