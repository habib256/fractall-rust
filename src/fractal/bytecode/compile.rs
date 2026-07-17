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

/// Formule bytecode EFFECTIVE d'un rendu (G4 jalon 2) : si
/// `params.hybrid_phases` est `Some` (non vide) → hybride multi-phase
/// ([`compile_hybrid_formula`]) ; sinon la mono-formule du type
/// ([`compile_formula`]). Source UNIQUE consommée par tous les callsites du
/// render (orbite référence, pixel loop, f64 standard, gates `.is_some()`) →
/// une frame hybride itère la même formule partout.
pub fn formula_for_params(params: &crate::fractal::FractalParams) -> Option<Formula> {
    // Formule opcodes F3 (G4 Op::Rot) : PRIORITAIRE — seule voie qui émet
    // Op::Rot. `hybrid_opcodes` vide/blanc = absent (mirror is_hybrid_formula).
    if let Some(text) = params
        .hybrid_opcodes
        .as_ref()
        .filter(|s| !s.trim().is_empty())
    {
        return parse_opcodes_formula(text);
    }
    match &params.hybrid_phases {
        Some(phases) if !phases.is_empty() => {
            compile_hybrid_formula(phases, params.multibrot_power)
        }
        _ => compile_formula(params.fractal_type, params.multibrot_power),
    }
}

/// Parse une chaîne d'opcodes au format Fraktaler-3 (`[[formula]]
/// opcodes = "…"`, cf. F3 `param.cc::parse_opcodess`) en [`Formula`].
///
/// Mots reconnus : `add sqr mul store absx absy negx negy rot{DEG}` —
/// chaque `add` TERMINE une phase (z := z + c), les phases s'itèrent
/// cycliquement. `rot{θ}` (degrés) émet [`Op::Rot`] avec cos/sin calculés
/// **en f32** (parité F3 : `opcode.u.rot.{x,y}` sont des `float`,
/// `param.cc:1060-1066`).
///
/// Mirror du `need_store` de F3 (`param.cc::compile_formula`) : si une phase
/// contient `mul` avant tout `store` explicite, un `store` est préfixé
/// (sémantique z^n : le Store capture le z d'entrée de phase).
///
/// `None` si : mot inconnu, degrés non parsables, mots orphelins après le
/// dernier `add` (phase non terminée), ou aucune phase.
pub fn parse_opcodes_formula(text: &str) -> Option<Formula> {
    let mut phases: Vec<Phase> = Vec::new();
    let mut current: Vec<Op> = Vec::new();
    for word in text.split_whitespace() {
        let op = if let Some(deg_str) =
            word.strip_prefix("rot{").and_then(|w| w.strip_suffix('}'))
        {
            let degrees: f32 = deg_str.parse().ok()?;
            let radians = degrees * std::f32::consts::PI / 180.0;
            Op::Rot {
                cos_theta: radians.cos() as f64,
                sin_theta: radians.sin() as f64,
            }
        } else {
            match word {
                "add" => Op::Add,
                "sqr" => Op::Sqr,
                "mul" => Op::Mul,
                "store" => Op::Store,
                "absx" => Op::AbsX,
                "absy" => Op::AbsY,
                "negx" => Op::NegX,
                "negy" => Op::NegY,
                _ => return None,
            }
        };
        let ends_phase = matches!(op, Op::Add);
        current.push(op);
        if ends_phase {
            let ops = ensure_store_prefix(std::mem::take(&mut current));
            phases.push(Phase::new(ops));
        }
    }
    if !current.is_empty() || phases.is_empty() {
        return None;
    }
    Some(if phases.len() == 1 {
        Formula::single(phases.pop().expect("len == 1"))
    } else {
        Formula::hybrid(phases)
    })
}

/// `need_store` F3 (`param.cc::compile_formula`) : un `mul` rencontré avant
/// tout `store` explicite multiplie par le Store initial (= z d'entrée de
/// phase) → préfixer `store`. Sans lui, notre `Op::Mul` lirait un
/// `stored_z`/`stored_delta` obsolètes (état du constructeur DeltaState).
fn ensure_store_prefix(ops: Vec<Op>) -> Vec<Op> {
    let mut need_store = false;
    for op in &ops {
        match op {
            Op::Mul => {
                need_store = true;
                break;
            }
            Op::Store => break,
            _ => {}
        }
    }
    if need_store {
        let mut with_store = Vec::with_capacity(ops.len() + 1);
        with_store.push(Op::Store);
        with_store.extend(ops);
        with_store
    } else {
        ops
    }
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

    // — parse_opcodes_formula (G4 Op::Rot) —

    #[test]
    fn parse_opcodes_mandelbrot_matches_compile() {
        let parsed = parse_opcodes_formula("sqr add").unwrap();
        let compiled = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        assert_eq!(parsed.phases.len(), 1);
        assert_eq!(parsed.phases[0].ops, compiled.phases[0].ops);
    }

    #[test]
    fn parse_opcodes_hybrid_mbs_matches_compile() {
        let parsed = parse_opcodes_formula("sqr add absx absy sqr add").unwrap();
        let compiled = compile_hybrid_formula(
            &[FractalType::Mandelbrot, FractalType::BurningShip],
            2.0,
        )
        .unwrap();
        assert_eq!(parsed.phases.len(), 2);
        assert_eq!(parsed.phases[0].ops, compiled.phases[0].ops);
        assert_eq!(parsed.phases[1].ops, compiled.phases[1].ops);
    }

    #[test]
    fn parse_opcodes_need_store_prefix_matches_power3() {
        // F3 need_store : `mul` avant tout `store` explicite → store préfixé.
        let parsed = parse_opcodes_formula("sqr mul add").unwrap();
        assert_eq!(parsed.phases[0].ops, compile_power(3)); // Store Sqr Mul Add
        // Store explicite : pas de double préfixe.
        let explicit = parse_opcodes_formula("store sqr mul add").unwrap();
        assert_eq!(explicit.phases[0].ops, compile_power(3));
    }

    #[test]
    fn parse_opcodes_rot_f32_coefficients() {
        let f = parse_opcodes_formula("sqr rot{30} add").unwrap();
        assert_eq!(f.phases.len(), 1);
        let Op::Rot { cos_theta, sin_theta } = f.phases[0].ops[1] else {
            panic!("attendu Op::Rot, ops = {:?}", f.phases[0].ops);
        };
        // Parité F3 : cos/sin calculés en f32 (opcode.u.rot.{x,y} float).
        let rad = 30.0f32 * std::f32::consts::PI / 180.0;
        assert_eq!(cos_theta, rad.cos() as f64);
        assert_eq!(sin_theta, rad.sin() as f64);
        // rot{0} = identité exacte.
        let f0 = parse_opcodes_formula("sqr rot{0} add").unwrap();
        let Op::Rot { cos_theta: c0, sin_theta: s0 } = f0.phases[0].ops[1] else {
            panic!("attendu Op::Rot");
        };
        assert_eq!((c0, s0), (1.0, 0.0));
    }

    #[test]
    fn parse_opcodes_rejects_invalid() {
        assert!(parse_opcodes_formula("").is_none()); // aucune phase
        assert!(parse_opcodes_formula("sqr").is_none()); // pas de add final
        assert!(parse_opcodes_formula("sqr add sqr").is_none()); // orphelins
        assert!(parse_opcodes_formula("foo add").is_none()); // mot inconnu
        assert!(parse_opcodes_formula("rot{abc} add").is_none()); // degrés KO
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
    fn hybrid_MM_iterates_identically_to_single_M() {
        // Invariant : [Mandelbrot, Mandelbrot] (2 phases identiques z²+c) doit
        // itérer EXACTEMENT comme le Mandelbrot mono-phase (même trajectoire,
        // même compte d'itérations, même z final).
        use super::super::iterate_bytecode_f64;
        use num_complex::Complex64;
        let m = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let mm =
            compile_hybrid_formula(&[FractalType::Mandelbrot, FractalType::Mandelbrot], 2.0)
                .unwrap();
        for &(re, im) in &[(-0.5, 0.5), (0.3, 0.0), (-1.0, 0.1), (0.28, 0.53), (-0.75, 0.0)] {
            let c = Complex64::new(re, im);
            let z0 = Complex64::new(0.0, 0.0);
            let rm = iterate_bytecode_f64(&m, z0, c, 500, 25.0);
            let rmm = iterate_bytecode_f64(&mm, z0, c, 500, 25.0);
            assert_eq!(rm.iteration, rmm.iteration, "iter mismatch c={c:?}");
            assert!((rm.z - rmm.z).norm() < 1e-12, "z mismatch c={c:?}");
        }
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
