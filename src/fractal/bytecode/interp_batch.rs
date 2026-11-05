//! Interpréteur bytecode f64 **par lots**, à voies entrelacées.
//!
//! [`super::interp::iterate_bytecode_f64`] itère UN point : sa boucle est une
//! seule chaîne de dépendance `z → z² + c → …` (~10 cycles de latence par
//! itération) pendant que le cœur pourrait en mener plusieurs de front, et
//! l'interprétation (`match op` par opcode) est payée à chaque itération de
//! chaque point.
//!
//! Ici [`LANES`] points avancent en **lockstep** : chaque opcode est appliqué
//! à toutes les voies dans une petite boucle à borne fixe (vectorisable), le
//! `match` est amorti sur les voies, et les chaînes de dépendance des voies
//! se recouvrent. Dès qu'une voie termine (évasion, non-fini ou plafond
//! d'itérations) son résultat est écrit et la voie est **rechargée** avec le
//! point suivant du lot : les voies restent pleines tant qu'il reste du travail.
//!
//! **Bit-identique à l'interpréteur scalaire, par construction** : chaque voie
//! exécute exactement la même suite d'opérations flottantes, dans le même
//! ordre (pas de FMA, pas de réassociation), et la sortie est détectée à
//! l'itération même où le scalaire sort. Verrou :
//! `batch_matches_scalar_interpreter_bit_for_bit`.
//!
//! Limite : formule **mono-phase** (un hybride fait avancer chaque point dans
//! une phase qui dépend de SON compteur d'itérations — les voies ne
//! partageraient plus le même opcode).

use num_complex::Complex64;

use super::{Op, Phase};

/// Nombre de points menés de front. 8 = quatre vecteurs NEON/SSE2 de deux
/// f64 (deux AVX2), assez pour recouvrir la latence de la chaîne `z² + c` sur
/// un cœur à quatre unités flottantes.
pub const LANES: usize = 8;

/// Voie sans point (lot épuisé) : elle continue d'itérer des zéros inoffensifs.
const NO_SLOT: usize = usize::MAX;

/// Itère `points` avec la phase `phase`, écrit le compte d'itérations et le z
/// final de chaque point dans `out_iter` / `out_z` (mêmes indices).
///
/// Convention (identique au dispatch scalaire `iterate_via_bytecode`) :
/// - `is_julia == false` (Mandelbrot-like) : `z₀ = seed`, `c = point` ;
/// - `is_julia == true` : `z₀ = point`, `c = seed`.
///
/// # Panics
/// Si `out_iter` ou `out_z` n'ont pas la longueur de `points`.
pub fn iterate_bytecode_f64_batch(
    phase: &Phase,
    points: &[Complex64],
    seed: Complex64,
    is_julia: bool,
    iteration_max: u32,
    bailout: f64,
    out_iter: &mut [u32],
    out_z: &mut [Complex64],
) {
    let n = points.len();
    assert_eq!(out_iter.len(), n);
    assert_eq!(out_z.len(), n);
    let bailout_sqr = bailout * bailout;
    // Nombre de `Add` de la phase (= incrément du compteur par passe ; 1 en
    // pratique, `Add` terminant la phase).
    let adds = phase.ops.iter().filter(|op| matches!(op, Op::Add)).count() as u32;

    let mut re = [0.0f64; LANES];
    let mut im = [0.0f64; LANES];
    let mut stored_re = [0.0f64; LANES];
    let mut stored_im = [0.0f64; LANES];
    let mut c_re = [0.0f64; LANES];
    let mut c_im = [0.0f64; LANES];
    let mut iter = [0u32; LANES];
    let mut slot = [NO_SLOT; LANES];
    let mut done = [false; LANES];

    let mut next = 0usize;
    let mut live = 0usize;

    // Charge le prochain point non trivial dans la voie `l`. Les points
    // résolus sans itérer (déjà échappés à l'entrée, plafond nul) sont écrits
    // directement — mêmes sorties anticipées que l'interpréteur scalaire.
    macro_rules! load {
        ($l:expr) => {{
            let l = $l;
            slot[l] = NO_SLOT;
            re[l] = 0.0;
            im[l] = 0.0;
            stored_re[l] = 0.0;
            stored_im[l] = 0.0;
            c_re[l] = 0.0;
            c_im[l] = 0.0;
            iter[l] = 0;
            while next < n {
                let i = next;
                next += 1;
                let p = points[i];
                let (z0, c) = if is_julia { (p, seed) } else { (seed, p) };
                // `>=` exact du scalaire (un z₀ NaN n'est PAS une sortie ici :
                // il itère une fois puis sort « non fini » à l'itération 1).
                if z0.norm_sqr() >= bailout_sqr || iteration_max == 0 {
                    out_iter[i] = 0;
                    out_z[i] = z0;
                    continue;
                }
                re[l] = z0.re;
                im[l] = z0.im;
                stored_re[l] = z0.re;
                stored_im[l] = z0.im;
                c_re[l] = c.re;
                c_im[l] = c.im;
                slot[l] = i;
                live += 1;
                break;
            }
        }};
    }

    for l in 0..LANES {
        load!(l);
    }

    while live > 0 {
        // Passes de phase jusqu'à ce qu'au moins une voie termine.
        loop {
            for op in &phase.ops {
                match *op {
                    Op::Sqr => {
                        for l in 0..LANES {
                            let (a, b) = (re[l], im[l]);
                            re[l] = a * a - b * b;
                            im[l] = a * b + b * a;
                        }
                    }
                    Op::Mul => {
                        for l in 0..LANES {
                            let (a, b) = (re[l], im[l]);
                            let (sa, sb) = (stored_re[l], stored_im[l]);
                            re[l] = a * sa - b * sb;
                            im[l] = a * sb + b * sa;
                        }
                    }
                    Op::Store => {
                        stored_re = re;
                        stored_im = im;
                    }
                    Op::AbsX => {
                        for v in &mut re {
                            *v = v.abs();
                        }
                    }
                    Op::AbsY => {
                        for v in &mut im {
                            *v = v.abs();
                        }
                    }
                    Op::NegX => {
                        for v in &mut re {
                            *v = -*v;
                        }
                    }
                    Op::NegY => {
                        for v in &mut im {
                            *v = -*v;
                        }
                    }
                    Op::Add => {
                        for l in 0..LANES {
                            re[l] += c_re[l];
                            im[l] += c_im[l];
                        }
                    }
                    Op::Rot {
                        cos_theta,
                        sin_theta,
                    } => {
                        for l in 0..LANES {
                            let (a, b) = (re[l], im[l]);
                            re[l] = a * cos_theta - b * sin_theta;
                            im[l] = a * sin_theta + b * cos_theta;
                        }
                    }
                }
            }
            // Fin de passe. Le scalaire sort sur « non fini » PUIS sur
            // `|z|² ≥ ER²` PUIS sur le plafond : `!(|z|² < ER²)` couvre les
            // deux premiers (une composante non finie donne une norme
            // infinie ou NaN), avec le même z et le même compteur.
            let mut any = false;
            for l in 0..LANES {
                iter[l] += adds;
                let norm_sqr = re[l] * re[l] + im[l] * im[l];
                let d = !(norm_sqr < bailout_sqr) | (iter[l] >= iteration_max);
                done[l] = d;
                any |= d;
            }
            if any {
                break;
            }
        }
        for l in 0..LANES {
            if done[l] {
                if slot[l] != NO_SLOT {
                    out_iter[slot[l]] = iter[l];
                    out_z[slot[l]] = Complex64::new(re[l], im[l]);
                    live -= 1;
                }
                load!(l);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::bytecode::{
        compile_formula, iterate_bytecode_f64, parse_opcodes_formula, Formula,
    };
    use crate::fractal::FractalType;

    /// Générateur déterministe (xorshift) — pas de dépendance de test.
    fn rng(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Points couvrant l'ensemble, son bord, l'extérieur et les valeurs
    /// pathologiques (NaN, infinis, zéro signé, déjà échappé à l'entrée).
    fn probe_points(count: usize, seed: u64) -> Vec<Complex64> {
        let mut r = rng(seed);
        let mut pts: Vec<Complex64> = (0..count)
            .map(|_| Complex64::new(r() * 4.0 - 2.5, r() * 3.0 - 1.5))
            .collect();
        // Bord fin : beaucoup d'itérations, sorties à des instants variés.
        pts.extend((0..count / 4).map(|k| {
            let t = k as f64 / (count / 4) as f64;
            Complex64::new(-0.75 + 1e-3 * (t - 0.5), 0.1 + 1e-4 * t)
        }));
        pts.extend([
            Complex64::new(0.0, 0.0),
            Complex64::new(-0.0, 0.0),
            Complex64::new(f64::NAN, 0.3),
            Complex64::new(0.2, f64::NAN),
            Complex64::new(f64::INFINITY, 0.0),
            Complex64::new(0.0, f64::NEG_INFINITY),
            Complex64::new(1e200, 1e200),
            Complex64::new(30.0, 0.0),
            Complex64::new(25.0, 0.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(f64::MIN_POSITIVE, -f64::MIN_POSITIVE),
        ]);
        pts
    }

    fn same(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits()
    }

    fn assert_batch_matches_scalar(
        label: &str,
        formula: &Formula,
        points: &[Complex64],
        seed: Complex64,
        is_julia: bool,
        iteration_max: u32,
        bailout: f64,
    ) {
        assert_eq!(formula.phases.len(), 1, "{label} : mono-phase attendu");
        let mut it = vec![u32::MAX; points.len()];
        let mut zs = vec![Complex64::new(f64::NAN, f64::NAN); points.len()];
        iterate_bytecode_f64_batch(
            &formula.phases[0],
            points,
            seed,
            is_julia,
            iteration_max,
            bailout,
            &mut it,
            &mut zs,
        );
        for (i, &p) in points.iter().enumerate() {
            let (z0, c) = if is_julia { (p, seed) } else { (seed, p) };
            let r = iterate_bytecode_f64(formula, z0, c, iteration_max, bailout);
            assert!(
                it[i] == r.iteration && same(zs[i].re, r.z.re) && same(zs[i].im, r.z.im),
                "{label} point #{i} {p:?} (julia={is_julia}, max={iteration_max}) : \
                 lot ({}, {:?}) ≠ scalaire ({}, {:?})",
                it[i],
                zs[i],
                r.iteration,
                r.z
            );
        }
    }

    /// Verrou central : le lot rend, pour chaque point, EXACTEMENT ce que rend
    /// l'interpréteur scalaire — compte d'itérations et z final au bit près —
    /// sur toutes les familles compilables, en convention Mandelbrot et Julia,
    /// y compris les entrées pathologiques et les tailles de lot qui ne
    /// remplissent pas les voies.
    #[test]
    fn batch_matches_scalar_interpreter_bit_for_bit() {
        let types = [
            FractalType::Mandelbrot,
            FractalType::BurningShip,
            FractalType::Tricorn,
            FractalType::Celtic,
            FractalType::Buffalo,
            FractalType::PerpendicularBurningShip,
        ];
        let points = probe_points(600, 0x9E37_79B9_7F4A_7C15);
        let julia_seed = Complex64::new(-0.8, 0.156);
        for ty in types {
            let formula = compile_formula(ty, 2.0).expect("formule compilable");
            for max in [0u32, 1, 2, 7, 300] {
                assert_batch_matches_scalar(
                    &format!("{ty:?}"),
                    &formula,
                    &points,
                    Complex64::new(0.0, 0.0),
                    false,
                    max,
                    25.0,
                );
                assert_batch_matches_scalar(
                    &format!("{ty:?}"),
                    &formula,
                    &points,
                    julia_seed,
                    true,
                    max,
                    25.0,
                );
            }
        }
        // Puissances entières (chaînes Store/Sqr/Mul) et seed Mandelbrot non nul.
        for power in [3.0, 4.0, 5.0, 7.0] {
            let formula = compile_formula(FractalType::Multibrot, power).expect("multibrot entier");
            assert_batch_matches_scalar(
                &format!("Multibrot^{power}"),
                &formula,
                &points,
                Complex64::new(0.05, -0.02),
                false,
                120,
                25.0,
            );
        }
        // Opcodes F3 avec rotation et négations, bailout historique 2.
        let formula = parse_opcodes_formula("negx sqr rot{30} absy add").expect("opcodes");
        assert_batch_matches_scalar(
            "opcodes rot30",
            &formula,
            &points,
            Complex64::new(0.0, 0.0),
            false,
            200,
            2.0,
        );
    }

    /// Lots plus petits que le nombre de voies, vides, et multiples exacts.
    #[test]
    fn batch_handles_every_batch_size() {
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).expect("mandelbrot");
        let points = probe_points(40, 42);
        for len in [0usize, 1, 2, LANES - 1, LANES, LANES + 1, 2 * LANES, 37] {
            assert_batch_matches_scalar(
                &format!("len={len}"),
                &formula,
                &points[..len],
                Complex64::new(0.0, 0.0),
                false,
                500,
                25.0,
            );
        }
    }
}
