//! Interpréteur CPU f64 du bytecode hybride.
//!
//! Boucle pixel équivalente à `hybrid_render` (CPU, sans perturbation) : itère
//! les phases cycliquement jusqu'à `iteration_max` ou bailout `|z|² ≥ bailout²`.

use num_complex::Complex64;

use super::{Formula, Op};

/// Résultat d'une itération bytecode.
pub struct BytecodeResult {
    /// Nombre d'itérations effectuées (incrémenté à chaque `Op::Add`).
    pub iteration: u32,
    /// Valeur finale de z (utile pour le smooth coloring).
    pub z: Complex64,
}

/// Itère le bytecode pour un pixel.
///
/// - `z0` : valeur initiale de z (typiquement `seed` pour Mandelbrot, `pixel` pour Julia).
/// - `c`  : constante ajoutée par `Op::Add` (`pixel` pour Mandelbrot, `seed` pour Julia).
/// - `bailout` : rayon d'évasion (compare `|z|² ≥ bailout²`).
///
/// La boucle effectue `Op::Add` puis vérifie le bailout : conforme à
/// `hybrid_render` qui termine quand `z² ≥ ER²` après application de la phase.
#[inline]
pub fn iterate_bytecode_f64(
    formula: &Formula,
    z0: Complex64,
    c: Complex64,
    iteration_max: u32,
    bailout: f64,
) -> BytecodeResult {
    let bailout_sqr = bailout * bailout;
    let mut z = z0;
    let mut stored = z; // registre `Store`, valeur initiale arbitraire
    let mut iter = 0u32;
    let mut phase_idx = 0usize;
    let n_phases = formula.phases.len();

    // Bailout initial (cohérent avec les fonctions actuelles qui testent en début
    // de boucle while).
    if z.norm_sqr() >= bailout_sqr {
        return BytecodeResult { iteration: 0, z };
    }

    while iter < iteration_max {
        let phase = &formula.phases[phase_idx];
        for op in &phase.ops {
            match op {
                Op::Sqr => {
                    z = z * z;
                }
                Op::Mul => {
                    z = z * stored;
                }
                Op::Store => {
                    stored = z;
                }
                Op::AbsX => {
                    z.re = z.re.abs();
                }
                Op::AbsY => {
                    z.im = z.im.abs();
                }
                Op::NegX => {
                    z.re = -z.re;
                }
                Op::NegY => {
                    z.im = -z.im;
                }
                Op::Add => {
                    z += c;
                    iter += 1;
                }
                Op::Rot { cos_theta, sin_theta } => {
                    // z := z * (cos + sin·i)
                    let r = Complex64::new(*cos_theta, *sin_theta);
                    z = z * r;
                }
            }
        }

        // Cycle de phase (cf. F3 `phase = (phase + 1) % opss.size()`).
        if n_phases > 1 {
            phase_idx = (phase_idx + 1) % n_phases;
        }

        // Protection NaN/Inf : sortie propre, comme les fonctions dédiées.
        if !z.re.is_finite() || !z.im.is_finite() {
            break;
        }
        if z.norm_sqr() >= bailout_sqr {
            break;
        }
    }

    BytecodeResult { iteration: iter, z }
}
