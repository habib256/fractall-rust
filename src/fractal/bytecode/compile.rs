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
    use FractalType::*;
    match ft {
        // Mandelbrot et Julia : même bytecode, seule la convention d'appel diffère.
        Mandelbrot | Julia => Some(Formula::single(Phase::new(vec![Op::Sqr, Op::Add]))),

        // Burning Ship : (|x| + i|y|)² + c
        BurningShip | BurningShipJulia => Some(Formula::single(Phase::new(vec![
            Op::AbsX,
            Op::AbsY,
            Op::Sqr,
            Op::Add,
        ]))),

        // Tricorn (Mandelbar) : conj(z)² + c = (negy then sqr)
        Tricorn | TricornJulia => Some(Formula::single(Phase::new(vec![
            Op::NegY,
            Op::Sqr,
            Op::Add,
        ]))),

        // Celtic : |Re(z²)| + i·Im(z²) + c (l'abs vient APRÈS le carré, sur la partie réelle)
        Celtic | CelticJulia => Some(Formula::single(Phase::new(vec![
            Op::Sqr,
            Op::AbsX,
            Op::Add,
        ]))),

        // Buffalo : |Re(z²)| + i·|Im(z²)| + c
        Buffalo | BuffaloJulia => Some(Formula::single(Phase::new(vec![
            Op::Sqr,
            Op::AbsX,
            Op::AbsY,
            Op::Add,
        ]))),

        // Perpendicular Burning Ship : (x - i|y|)² + c
        // Ordre d'opérations : on rend y positif (AbsY), puis on le négocie (NegY),
        // puis on élève au carré, puis on ajoute c.
        PerpendicularBurningShip | PerpendicularBurningShipJulia => Some(Formula::single(
            Phase::new(vec![Op::AbsY, Op::NegY, Op::Sqr, Op::Add]),
        )),

        // Multibrot : z^N + c pour N entier ≥ 2 via décomposition binaire.
        Multibrot | MultibrotJulia => {
            let n_round = multibrot_power.round();
            let is_integer = (multibrot_power - n_round).abs() < 1e-10;
            if !is_integer || n_round < 2.0 {
                return None;
            }
            let n = n_round as u32;
            Some(Formula::single(Phase::new(compile_power(n))))
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
