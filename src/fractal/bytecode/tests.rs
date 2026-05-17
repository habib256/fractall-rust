//! Iso-tests : l'interpréteur bytecode doit produire le MÊME nombre d'itérations
//! que les fonctions dédiées de `iterations.rs` sur une grille de pixels.
//!
//! Les fonctions dédiées contiennent parfois des features additionnelles
//! (orbit traps, distance estimation, périodicité Mandelbrot). On désactive
//! tout ça dans les tests pour comparer le coeur de l'itération escape-time.
//!
//! La périodicité de Mandelbrot fait sauter l'itération à `iteration_max` dès
//! qu'un cycle est détecté ; on désactive en mettant `enable_orbit_traps` (ce
//! qui désactive `use_periodicity` dans le code actuel) — mais on doit aussi
//! comparer hors-périodicité. Pour Mandelbrot on filtre les pixels qui
//! atteignent `iteration_max` pour éviter ce biais.

use num_complex::Complex64;

use super::{compile_formula, iterate_bytecode_f64, Formula};
use crate::fractal::iterations::iterate_point;
use crate::fractal::{default_params_for_type, FractalParams, FractalType};

/// Construit la convention (z₀, c) selon le type.
fn z0_and_c(ft: FractalType, params: &FractalParams, pixel: Complex64) -> (Complex64, Complex64) {
    if Formula::is_julia_for(ft) {
        (pixel, params.seed)
    } else {
        (params.seed, pixel)
    }
}

/// Test générique : pour chaque pixel d'une grille, vérifie
/// `iterate_bytecode_f64` ≡ `iterate_point` sur le compte d'itérations.
fn iso_test(ft: FractalType, grid_extent: f64, grid_n: usize) {
    let mut params = default_params_for_type(ft, 100, 100);
    // Désactive features qui divergent du coeur escape-time.
    params.enable_orbit_traps = true; // désactive le path périodicité Mandelbrot
    params.enable_distance_estimation = false;
    params.iteration_max = 256;

    let formula = compile_formula(ft, params.multibrot_power)
        .unwrap_or_else(|| panic!("compile_formula({:?}) returned None", ft));

    let mut divergences = 0usize;
    let mut total = 0usize;
    let step = 2.0 * grid_extent / (grid_n as f64);
    for i in 0..grid_n {
        for j in 0..grid_n {
            let x = -grid_extent + step * (i as f64 + 0.5);
            let y = -grid_extent + step * (j as f64 + 0.5);
            let pixel = Complex64::new(x, y);
            let (z0, c) = z0_and_c(ft, &params, pixel);

            let bc = iterate_bytecode_f64(&formula, z0, c, params.iteration_max, params.bailout);
            let dedicated = iterate_point(&params, pixel);
            total += 1;

            // Pour Mandelbrot non-Julia, le path dédié a la détection
            // cardioïde/bulb-2 qui retourne iteration_max immédiatement.
            // On skip les pixels qui sont à iteration_max côté dédié.
            if matches!(ft, FractalType::Mandelbrot) && dedicated.iteration == params.iteration_max
            {
                continue;
            }

            if bc.iteration != dedicated.iteration {
                divergences += 1;
                if divergences <= 5 {
                    eprintln!(
                        "[{:?}] divergence à ({:.4}, {:.4}): bc={} dedicated={}",
                        ft, x, y, bc.iteration, dedicated.iteration
                    );
                }
            }
        }
    }
    assert_eq!(
        divergences, 0,
        "{:?}: {} divergences sur {} pixels",
        ft, divergences, total
    );
}

#[test]
fn mandelbrot_iso() {
    iso_test(FractalType::Mandelbrot, 2.0, 60);
}

#[test]
fn julia_iso() {
    iso_test(FractalType::Julia, 2.0, 60);
}

#[test]
fn burning_ship_iso() {
    iso_test(FractalType::BurningShip, 2.0, 60);
}

#[test]
fn burning_ship_julia_iso() {
    iso_test(FractalType::BurningShipJulia, 2.0, 60);
}

#[test]
fn tricorn_iso() {
    iso_test(FractalType::Tricorn, 2.0, 60);
}

#[test]
fn tricorn_julia_iso() {
    iso_test(FractalType::TricornJulia, 2.0, 60);
}

#[test]
fn celtic_iso() {
    iso_test(FractalType::Celtic, 2.0, 60);
}

#[test]
fn celtic_julia_iso() {
    iso_test(FractalType::CelticJulia, 2.0, 60);
}

#[test]
fn buffalo_iso() {
    iso_test(FractalType::Buffalo, 2.0, 60);
}

#[test]
fn buffalo_julia_iso() {
    iso_test(FractalType::BuffaloJulia, 2.0, 60);
}

#[test]
fn perpendicular_burning_ship_iso() {
    iso_test(FractalType::PerpendicularBurningShip, 2.0, 60);
}

#[test]
fn perpendicular_burning_ship_julia_iso() {
    iso_test(FractalType::PerpendicularBurningShipJulia, 2.0, 60);
}

#[test]
fn multibrot_integer_iso() {
    // multibrot_power par défaut = 2.5 (non entier) → on doit le forcer entier.
    for power in [2.0_f64, 3.0, 4.0, 5.0, 7.0] {
        let mut params = default_params_for_type(FractalType::Multibrot, 100, 100);
        params.multibrot_power = power;
        params.enable_orbit_traps = true;
        params.iteration_max = 256;

        let formula =
            compile_formula(FractalType::Multibrot, power).expect("integer power supported");

        let mut divergences = 0usize;
        for i in 0..40 {
            for j in 0..40 {
                let x = -2.0 + 0.1 * (i as f64 + 0.5);
                let y = -2.0 + 0.1 * (j as f64 + 0.5);
                let pixel = Complex64::new(x, y);
                let bc = iterate_bytecode_f64(
                    &formula,
                    params.seed,
                    pixel,
                    params.iteration_max,
                    params.bailout,
                );
                let dedicated = iterate_point(&params, pixel);
                if bc.iteration != dedicated.iteration {
                    divergences += 1;
                    if divergences <= 3 {
                        eprintln!(
                            "[Multibrot pow={}] ({:.3},{:.3}) bc={} ded={}",
                            power, x, y, bc.iteration, dedicated.iteration
                        );
                    }
                }
            }
        }
        assert_eq!(divergences, 0, "Multibrot power={}: {} divergences", power, divergences);
    }
}

#[test]
fn multibrot_non_integer_returns_none() {
    assert!(compile_formula(FractalType::Multibrot, 2.5).is_none());
    assert!(compile_formula(FractalType::Multibrot, 3.7).is_none());
}
