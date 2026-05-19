use num_complex::Complex64;

use crate::fractal::bytecode::{compile_formula, iterate_bytecode_f64, Formula};
use crate::fractal::{FractalParams, FractalResult, FractalType};
use crate::fractal::orbit_traps::OrbitData;

/// Vrai si l'interpréteur bytecode peut servir ce pixel.
///
/// Conditions :
/// - `use_bytecode_engine` activé ;
/// - le type compile en bytecode.
///
/// `enable_orbit_traps`, `enable_distance_estimation`, `enable_interior_detection`
/// sont tous supportés par le bytecode f64 standard via tracking de dz dans
/// iterate_via_bytecode. Pour la perturbation, ces features tombent sur le
/// path legacy (cf. delta.rs::try_bytecode_unified_path).
#[inline]
fn can_use_bytecode(params: &FractalParams) -> bool {
    params.use_bytecode_engine
        && compile_formula(params.fractal_type, params.multibrot_power).is_some()
}

/// Dispatch interpréteur : (z₀, c) selon convention Mandelbrot/Julia.
/// Construit aussi un OrbitData si orbit_traps est demandé.
#[inline]
fn iterate_via_bytecode(params: &FractalParams, z_pixel: Complex64) -> FractalResult {
    let formula = compile_formula(params.fractal_type, params.multibrot_power)
        .expect("can_use_bytecode garantit que compile_formula renvoie Some");
    let (z0, c) = if Formula::is_julia_for(params.fractal_type) {
        (z_pixel, params.seed)
    } else {
        (params.seed, z_pixel)
    };

    let needs_orbit = params.enable_orbit_traps;
    let needs_distance = params.enable_distance_estimation;
    let needs_interior = params.enable_interior_detection;

    if needs_orbit || needs_distance || needs_interior {
        // Path avec tracking : itère manuellement pour hooker orbit_data,
        // propager dz (dérivée pour distance + interior detection).
        //
        // Pour la distance estimation :
        // - Mandelbrot-like : dz/dc, init=0 (z₀=seed ne dépend pas de c).
        //   Sur Op::Add, dz += 1. Formule : d = 2·|z|·ln(|z|) / |dz|
        // - Julia-like : dz/dz₀, init=1. Add inchangé. d = |z|·ln(|z|)/|dz|
        //
        // Pour interior detection :
        // - On utilise le MÊME dz tracker (dz/dz_critical où z_critical = 0
        //   pour Mandelbrot, pixel pour Julia). |dz| < threshold = interior.
        let is_julia = Formula::is_julia_for(params.fractal_type);
        let mut z = z0;
        let mut stored_z = z;
        let mut dz = if is_julia {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        };
        let mut stored_dz = dz;
        let mut iter = 0u32;
        let bailout_sqr = params.bailout * params.bailout;
        let phase = &formula.phases[0]; // mono-phase
        let mut orbit_data = if needs_orbit {
            let mut od = OrbitData::new(params.orbit_trap_type);
            od.add_point(z, 0);
            Some(od)
        } else {
            None
        };

        while iter < params.iteration_max {
            if z.norm_sqr() >= bailout_sqr {
                break;
            }
            for op in &phase.ops {
                use crate::fractal::bytecode::Op;
                match op {
                    Op::Sqr => {
                        // dz' = 2·z·dz, puis z' = z²
                        dz = z * dz * 2.0;
                        z = z * z;
                    }
                    Op::Mul => {
                        // dz' = stored·dz + z·stored_dz, puis z' = z·stored
                        dz = stored_z * dz + z * stored_dz;
                        z = z * stored_z;
                    }
                    Op::Store => {
                        stored_z = z;
                        stored_dz = dz;
                    }
                    Op::AbsX => {
                        // d|x|/dx = sign(x). Si z.re < 0, flip dz.re.
                        if z.re < 0.0 {
                            dz = Complex64::new(-dz.re, dz.im);
                        }
                        z = Complex64::new(z.re.abs(), z.im);
                    }
                    Op::AbsY => {
                        if z.im < 0.0 {
                            dz = Complex64::new(dz.re, -dz.im);
                        }
                        z = Complex64::new(z.re, z.im.abs());
                    }
                    Op::NegX => {
                        z = Complex64::new(-z.re, z.im);
                        dz = Complex64::new(-dz.re, dz.im);
                    }
                    Op::NegY => {
                        z = Complex64::new(z.re, -z.im);
                        dz = Complex64::new(dz.re, -dz.im);
                    }
                    Op::Add => {
                        z += c;
                        // dz += 1 si c dépend du pixel (Mandelbrot-like),
                        // sinon dz inchangé (Julia-like, c = seed constant).
                        if !is_julia {
                            dz += Complex64::new(1.0, 0.0);
                        }
                        iter += 1;
                    }
                    Op::Rot { cos_theta, sin_theta } => {
                        let r = Complex64::new(*cos_theta, *sin_theta);
                        z = z * r;
                        dz = dz * r;
                    }
                }
            }
            if !z.re.is_finite() || !z.im.is_finite() {
                break;
            }
            if let Some(ref mut od) = orbit_data {
                od.add_point(z, iter);
            }
        }

        // Distance estimation : formule selon Mandelbrot ou Julia.
        let distance = if needs_distance && iter > 0 && iter < params.iteration_max {
            let z_norm = z.norm().max(2.0);
            let dz_norm = dz.norm();
            if dz_norm > 1e-300 && z_norm > 1.0 {
                let factor = if is_julia { 1.0 } else { 2.0 };
                Some(factor * z_norm * z_norm.ln() / dz_norm)
            } else {
                None
            }
        } else {
            None
        };

        // Interior detection : |dz| < threshold indique un cycle attracteur.
        // Encoder via le signe de z.im (convention legacy : z.im négatif = interior).
        let mut z_out = z;
        if needs_interior && iter >= params.iteration_max {
            let dz_norm = dz.norm();
            if dz_norm.is_finite() && dz_norm > 0.0 && dz_norm < params.interior_threshold {
                z_out = Complex64::new(z.re, -z.im.abs());
            }
        }

        return FractalResult {
            iteration: iter,
            z: z_out,
            orbit: orbit_data,
            distance,
        };
    }

    let r = iterate_bytecode_f64(&formula, z0, c, params.iteration_max, params.bailout);
    FractalResult {
        iteration: r.iteration,
        z: r.z,
        orbit: None,
        distance: None,
    }
}

/// Calcule l'itération pour un point donné, en suivant `FormulaSelector` côté C.
#[inline]
pub fn iterate_point(params: &FractalParams, z_pixel: Complex64) -> FractalResult {
    if can_use_bytecode(params) {
        return iterate_via_bytecode(params, z_pixel);
    }
    match params.fractal_type {
        FractalType::VonKoch | FractalType::Dragon => {
            panic!("Les fractales vectorielles doivent être rendues via render_von_koch/render_dragon")
        }
        FractalType::Mandelbrot => mandelbrot(params, z_pixel),
        FractalType::Julia => julia(params, z_pixel),
        FractalType::JuliaSin => julia_sin(params, z_pixel),
        FractalType::Newton => newton(params, z_pixel),
        FractalType::Phoenix => phoenix(params, z_pixel),
        FractalType::Buffalo => buffalo(params, z_pixel),
        FractalType::BarnsleyJulia => barnsley_julia(params, z_pixel),
        FractalType::BarnsleyMandelbrot => barnsley_mandelbrot(params, z_pixel),
        FractalType::MagnetJulia => magnet_julia(params, z_pixel),
        FractalType::MagnetMandelbrot => magnet_mandelbrot(params, z_pixel),
        FractalType::BurningShip => burning_ship(params, z_pixel),
        FractalType::Tricorn => tricorn(params, z_pixel),
        FractalType::Mandelbulb => mandelbulb(params, z_pixel),
        FractalType::PerpendicularBurningShip => perpendicular_burning_ship(params, z_pixel),
        FractalType::Celtic => celtic(params, z_pixel),
        FractalType::AlphaMandelbrot => alpha_mandelbrot(params, z_pixel),
        FractalType::PickoverStalks => pickover_stalks(params, z_pixel),
        FractalType::Nova => nova(params, z_pixel),
        FractalType::Multibrot => multibrot(params, z_pixel),
        FractalType::BurningShipJulia => burning_ship_julia(params, z_pixel),
        FractalType::TricornJulia => tricorn_julia(params, z_pixel),
        FractalType::CelticJulia => celtic_julia(params, z_pixel),
        FractalType::BuffaloJulia => buffalo_julia(params, z_pixel),
        FractalType::MultibrotJulia => multibrot_julia(params, z_pixel),
        FractalType::PerpendicularBurningShipJulia => perpendicular_burning_ship_julia(params, z_pixel),
        FractalType::AlphaMandelbrotJulia => alpha_mandelbrot_julia(params, z_pixel),
        FractalType::MandelbrotSin => mandelbrot_sin(params, z_pixel),
        FractalType::Buddhabrot => {
            panic!("Buddhabrot doit être rendu via render_buddhabrot(), pas iterate_point()")
        }
        FractalType::Lyapunov => {
            panic!("Lyapunov doit être rendu via render_lyapunov(), pas iterate_point()")
        }
        FractalType::Nebulabrot => {
            panic!("Nebulabrot doit être rendu via render_nebulabrot(), pas iterate_point()")
        }
        FractalType::AntiBuddhabrot => {
            panic!("Anti-Buddhabrot doit être rendu via render_antibuddhabrot(), pas iterate_point()")
        }
    }
}

#[inline]
fn mandelbrot(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Mandelbrot_Iteration: z_{n+1} = z_n^2 + c, dz/dc: z'_{n+1} = 2*z_n*z'_n + 1
    let mut z = p.seed;
    let mut dz = Complex64::new(0.0, 0.0); // dz_0/dc = 0 (seed is constant w.r.t. c)
    let mut i = 0u32;
    let mut z_old = z;
    let mut period = 0u32;
    let mut check_period = 1u32;
    let use_periodicity = !p.enable_orbit_traps && !p.enable_distance_estimation;
    let bailout_sqr = p.bailout * p.bailout;

    // Cardioid and period-2 bulb detection (skip if orbit traps or distance estimation needed)
    if !p.enable_orbit_traps && !p.enable_distance_estimation {
        let re = z_pixel.re;
        let im_sq = z_pixel.im * z_pixel.im;
        // Cardioid: q*(q + re - 0.25) <= 0.25*im^2
        let q = (re - 0.25) * (re - 0.25) + im_sq;
        if q * (q + re - 0.25) <= 0.25 * im_sq {
            return FractalResult { iteration: p.iteration_max, z: Complex64::new(0.0, 0.0), orbit: None, distance: None };
        }
        // Period-2 bulb: (re+1)^2 + im^2 <= 1/16
        if (re + 1.0) * (re + 1.0) + im_sq <= 0.0625 {
            return FractalResult { iteration: p.iteration_max, z: Complex64::new(0.0, 0.0), orbit: None, distance: None };
        }
    }

    // Initialiser orbit data si orbit traps activés
    let mut orbit_data = if p.enable_orbit_traps {
        Some(OrbitData::new(p.orbit_trap_type))
    } else {
        None
    };

    // Stocker le point initial si orbit traps activés
    if let Some(ref mut orbit) = orbit_data {
        orbit.add_point(z, 0);
    }

    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        // z'_{n+1} = 2*z_n*z'_n + 1 (dérivée par rapport à c)
        dz = z * dz * 2.0 + Complex64::new(1.0, 0.0);
        z = z * z + z_pixel;
        // Vérifier que z reste fini pour éviter NaN/infini qui causent des artefacts
        if !z.re.is_finite() || !z.im.is_finite() {
            // Si z devient invalide, utiliser une valeur de repli basée sur la dernière valeur valide
            // ou une valeur par défaut si c'est la première itération
            if i == 0 {
                z = Complex64::new(z_pixel.re * 10.0, z_pixel.im * 10.0);
            }
            break;
        }
        i += 1;
        if use_periodicity {
            period += 1;
            if (z - z_old).norm_sqr() < 1e-30 {
                i = p.iteration_max;
                break;
            }
            if period >= check_period {
                z_old = z;
                period = 0;
                check_period = check_period.min(4096) << 1;
            }
        }

        // Stocker le point dans l'orbite si orbit traps activés
        if let Some(ref mut orbit) = orbit_data {
            orbit.add_point(z, i);
        }
    }
    // S'assurer que z est valide avant de retourner
    if !z.re.is_finite() || !z.im.is_finite() {
        // Valeur de repli pour éviter les artefacts
        z = Complex64::new(z_pixel.re * 10.0, z_pixel.im * 10.0);
    }
    // Estimation de distance Mandelbrot: d = 2 * |z| * ln(|z|) / |dz/dc|
    let distance = if p.enable_distance_estimation && i < p.iteration_max && i > 0 {
        let z_norm = z.norm().max(2.0);
        let dz_norm = dz.norm();
        if dz_norm > 1e-300 && z_norm > 1.0 {
            Some(2.0 * z_norm * z_norm.ln() / dz_norm)
        } else {
            None
        }
    } else {
        None
    };
    FractalResult { iteration: i, z, orbit: orbit_data, distance }
}

#[inline]
fn julia(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Julia_Iteration: z_{n+1} = z_n^2 + c, dérivée par rapport à z_0: z'_{n+1} = 2*z_n*z'_n
    let mut z = z_pixel;
    let mut dz = Complex64::new(1.0, 0.0); // dz/dz_0
    let mut i = 0u32;
    let mut z_old = z;
    let mut period = 0u32;
    let mut check_period = 1u32;
    let use_periodicity = !p.enable_orbit_traps && !p.enable_distance_estimation;
    let bailout_sqr = p.bailout * p.bailout;

    let mut orbit_data = if p.enable_orbit_traps {
        Some(OrbitData::new(p.orbit_trap_type))
    } else {
        None
    };

    if let Some(ref mut orbit) = orbit_data {
        orbit.add_point(z, 0);
    }

    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        dz = z * dz * 2.0;
        z = z * z + p.seed;
        if !z.re.is_finite() || !z.im.is_finite() {
            if i == 0 {
                z = Complex64::new(p.seed.re * 10.0, p.seed.im * 10.0);
            }
            break;
        }
        i += 1;
        if use_periodicity {
            period += 1;
            if (z - z_old).norm_sqr() < 1e-30 {
                i = p.iteration_max;
                break;
            }
            if period >= check_period {
                z_old = z;
                period = 0;
                check_period = check_period.min(4096) << 1;
            }
        }
        if let Some(ref mut orbit) = orbit_data {
            orbit.add_point(z, i);
        }
    }
    if !z.re.is_finite() || !z.im.is_finite() {
        z = Complex64::new(p.seed.re * 10.0, p.seed.im * 10.0);
    }
    // Estimation de distance: d = |z| * ln(|z|) / |dz/dz_0|
    let distance = if p.enable_distance_estimation && i < p.iteration_max && i > 0 {
        let z_norm = z.norm().max(2.0);
        let dz_norm = dz.norm();
        if dz_norm > 1e-300 && z_norm > 1.0 {
            Some(z_norm * z_norm.ln() / dz_norm)
        } else {
            None
        }
    } else {
        None
    };
    FractalResult { iteration: i, z, orbit: orbit_data, distance }
}

#[inline]
fn julia_sin(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // JuliaSin_Iteration: z_{n+1} = c * sin(z_n), z_0 = pixel, c = seed
    let mut z = z_pixel;
    let mut i = 0u32;
    let bailout_sqr = p.bailout * p.bailout;
    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        z = p.seed * z.sin();
        i += 1;
    }
    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn mandelbrot_sin(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // MandelbrotSin_Iteration: z_{n+1} = c * sin(z_n), z_0 = seed, c = pixel (contrepartie de Julia Sin)
    let mut z = p.seed;
    let mut i = 0u32;
    let bailout_sqr = p.bailout * p.bailout;
    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        z = z_pixel * z.sin();
        i += 1;
    }
    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn newton(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Newton_Iteration
    let mut z = z_pixel;
    let mut i = 0u32;
    let bailout_sqr = p.bailout * p.bailout;
    // Degree polynomial (pris sur la partie réelle du seed)
    let degree = p.seed.re.round() as i32;
    let degree = if degree <= 0 { 3 } else { degree }; // garde-fou
    let one = Complex64::new(1.0, 0.0);

    if degree == 2 {
        // Fast-path: z^2 - 1, derivative = 2*z
        while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            let z_sq = z * z;
            let numerator = z_sq - one;
            let denominator = Complex64::new(2.0, 0.0) * z;
            if denominator.norm_sqr() < 1e-24 { break; }
            z = z - numerator / denominator;
            i += 1;
        }
    } else if degree == 3 {
        // Fast-path: z^3 - 1, derivative = 3*z^2
        while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            let z_sq = z * z;
            let z_cube = z_sq * z;
            let numerator = z_cube - one;
            let denominator = Complex64::new(3.0, 0.0) * z_sq;
            if denominator.norm_sqr() < 1e-24 { break; }
            z = z - numerator / denominator;
            i += 1;
        }
    } else {
        // General path with powc
        while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            let p_c = degree as f64;
            let z_pow = z.powc(Complex64::new(p_c, 0.0));
            let z_pow_deriv = z.powc(Complex64::new(p_c - 1.0, 0.0));

            let numerator = z_pow - one;
            let denominator = Complex64::new(p_c, 0.0) * z_pow_deriv;
            if denominator.norm_sqr() < 1e-24 { break; }
            z = z - numerator / denominator;
            i += 1;
        }
    }
    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn phoenix(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Phoenix_Iteration (degree 0, paramètres constants)
    let mut z = z_pixel;
    let mut y = Complex64::new(0.0, 0.0);
    let mut i = 0u32;
    let p1 = 0.56667;
    let p2 = -0.5;
    let bailout_sqr = p.bailout * p.bailout;

    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        let z_sq = z * z;
        let mut z_temp = Complex64::new(z_sq.re + p1, z_sq.im);
        let zp_temp = y * p2;
        z_temp += zp_temp;
        y = z;
        z = z_temp;
        i += 1;
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn barnsley_julia(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Barnsleyj1_Iteration
    let mut z = z_pixel;
    let mut i = 0u32;
    let bailout_sqr = p.bailout * p.bailout;

    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        let re = z.re;
        if re >= 0.0 {
            z = (z - Complex64::new(1.0, 0.0)) * p.seed;
        } else {
            z = (z + Complex64::new(1.0, 0.0)) * p.seed;
        }
        i += 1;
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn barnsley_mandelbrot(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Barnsleym1_Iteration
    let mut z = z_pixel;
    let c = z_pixel;
    let mut i = 0u32;
    let bailout_sqr = p.bailout * p.bailout;

    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        let re = z.re;
        if re >= 0.0 {
            z = (z - Complex64::new(1.0, 0.0)) * c;
        } else {
            z = (z + Complex64::new(1.0, 0.0)) * c;
        }
        i += 1;
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn magnet_julia(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Magnet1j_Iteration: z_{n+1} = ((z_n² + c - 1) / (2*z_n + c - 2))²
    let mut z = z_pixel;
    let mut i = 0u32;
    let bailout_sqr = p.bailout * p.bailout;

    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        let seed_minus_one = Complex64::new(p.seed.re - 1.0, p.seed.im);
        let seed_minus_two = Complex64::new(p.seed.re - 2.0, p.seed.im);
        let n = z * z + seed_minus_one;
        let q = Complex64::new(2.0, 0.0) * z + seed_minus_two;
        // Eviter division par zero
        if q.norm_sqr() < 1e-24 {
            break;
        }
        let ratio = n / q;
        z = ratio * ratio;
        i += 1;
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn magnet_mandelbrot(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Magnet1m_Iteration: z_{n+1} = ((z_n² + c - 1) / (2*z_n + c - 2))²
    let c = z_pixel;
    let mut z = Complex64::new(0.0, 0.0);
    let mut i = 0u32;
    let bailout_sqr = p.bailout * p.bailout;

    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        let c_minus_one = Complex64::new(c.re - 1.0, c.im);
        let c_minus_two = Complex64::new(c.re - 2.0, c.im);
        let n = z * z + c_minus_one;
        let q = Complex64::new(2.0, 0.0) * z + c_minus_two;
        // Eviter division par zero
        if q.norm_sqr() < 1e-24 {
            break;
        }
        let ratio = n / q;
        z = ratio * ratio;
        i += 1;
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn burning_ship(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // BurningShip_Iteration
    let mut z = p.seed;
    let mut i = 0u32;
    let bailout_sqr = p.bailout * p.bailout;

    let mut orbit_data = if p.enable_orbit_traps {
        Some(OrbitData::new(p.orbit_trap_type))
    } else {
        None
    };

    if let Some(ref mut orbit) = orbit_data {
        orbit.add_point(z, 0);
    }

    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        let re = z.re.abs();
        let im = z.im.abs();
        let mut z_temp = Complex64::new(re, im);
        z_temp = z_temp * z_temp;
        z = z_temp + z_pixel;
        if !z.re.is_finite() || !z.im.is_finite() {
            if i == 0 {
                z = Complex64::new(z_pixel.re * 10.0, z_pixel.im * 10.0);
            }
            break;
        }
        i += 1;
        if let Some(ref mut orbit) = orbit_data {
            orbit.add_point(z, i);
        }
    }
    if !z.re.is_finite() || !z.im.is_finite() {
        z = Complex64::new(z_pixel.re * 10.0, z_pixel.im * 10.0);
    }

    FractalResult { iteration: i, z, orbit: orbit_data, distance: None }
}

#[inline]
fn buffalo(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Buffalo_Iteration
    let mut z = p.seed;
    let mut i = 0u32;
    let bailout_sqr = p.bailout * p.bailout;

    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        let z_sq = z * z;
        let re_sq = z_sq.re.abs();
        let im_sq = z_sq.im.abs();
        z = Complex64::new(re_sq + z_pixel.re, im_sq + z_pixel.im);
        i += 1;
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn tricorn(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Tricorn_Iteration
    let mut z = p.seed;
    let mut i = 0u32;
    let bailout_sqr = p.bailout * p.bailout;

    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        let z_conj = Complex64::new(z.re, -z.im);
        let z_temp = z_conj * z_conj;
        z = z_temp + z_pixel;
        i += 1;
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn mandelbulb(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Mandelbulb_Iteration (2D, puissance 8)
    let mut z = p.seed;
    let mut i = 0u32;
    let bailout_sqr = p.bailout * p.bailout;

    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        // z^8 via multiplications successives
        let mut z_temp = z * z;      // z^2
        z_temp = z_temp * z_temp;    // z^4
        z_temp = z_temp * z_temp;    // z^8
        z = z_temp + z_pixel;
        i += 1;
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn perpendicular_burning_ship(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // PerpendicularBurningShip_Iteration
    let mut z = p.seed;
    let mut i = 0u32;
    let bailout_sqr = p.bailout * p.bailout;

    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        let x = z.re;
        let y = z.im;
        let y_abs = y.abs();

        // (x - i*|y|)^2 = x^2 - y_abs^2 - i*2*x*y_abs
        let x2 = x * x;
        let y2 = y_abs * y_abs;
        z = Complex64::new(
            x2 - y2 + z_pixel.re,
            -2.0 * x * y_abs + z_pixel.im,
        );
        i += 1;
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn celtic(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Celtic_Iteration
    let mut z = p.seed;
    let mut i = 0u32;
    let bailout_sqr = p.bailout * p.bailout;

    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        let x = z.re;
        let y = z.im;
        let u = x * x - y * y;
        let v = 2.0 * x * y;
        z = Complex64::new(u.abs() + z_pixel.re, v + z_pixel.im);
        i += 1;
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn alpha_mandelbrot(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // AlphaMandelbrot_Iteration
    let mut z = p.seed;
    let mut i = 0u32;
    let bailout_sqr = p.bailout * p.bailout;

    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        let z_sq = z * z;
        let m = z_sq + z_pixel;
        let m_sq = m * m;
        z = z_sq + m_sq + z_pixel;
        i += 1;
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn pickover_stalks(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // PickoverStalks_Iteration (trap based)
    let mut z = p.seed;
    let mut i = 0u32;
    let mut trap_min = 1e10f64;
    let trap_divisor = 0.03f64;
    let bailout_sqr = p.bailout * p.bailout;

    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        // Mandelbrot like iteration
        z = z * z + z_pixel;

        let re_abs = z.re.abs();
        let im_abs = z.im.abs();
        let trap_distance = re_abs.min(im_abs);
        if trap_distance < trap_min {
            trap_min = trap_distance;
        }

        i += 1;
    }

    let mut iter_value: u32;
    if trap_min > 1e-10 {
        let log_trap = -(trap_min / trap_divisor).ln();
        iter_value = (log_trap * 100.0) as u32;
        if iter_value >= p.iteration_max {
            iter_value = p.iteration_max.saturating_sub(1);
        }
    } else {
        iter_value = p.iteration_max.saturating_sub(1);
    }

    FractalResult {
        iteration: iter_value,
        z,
        orbit: None,
        distance: None,
    }
}

#[inline]
fn nova(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Nova_Iteration - fast-path for p=3 (hardcoded default)
    let mut z = Complex64::new(1.0, 0.0); // z0 = 1
    let mut z_prev;
    let conv_epsilon = 1e-7;
    let conv_epsilon_sq = conv_epsilon * conv_epsilon;
    let bailout_sqr = p.bailout * p.bailout;
    let one = Complex64::new(1.0, 0.0);
    let three = Complex64::new(3.0, 0.0);

    let mut i = 0u32;
    while i < p.iteration_max {
        // z^3 and z^2 via multiplication (eliminates 2 powc calls per iteration)
        let z_sq = z * z;
        let z_cube = z_sq * z;

        // p(z) = z^3 - 1
        let numerator = z_cube - one;
        // p'(z) = 3 * z^2
        let denominator = three * z_sq;

        if denominator.norm_sqr() < 1e-20 {
            break;
        }

        let newton_step = numerator / denominator;

        z_prev = z;
        z = z - newton_step + z_pixel;

        let diff_sq = (z - z_prev).norm_sqr();
        let z_sq_norm = z.norm_sqr();
        let denom = if z_sq_norm < 1.0 { 1.0 } else { z_sq_norm };

        if diff_sq / denom < conv_epsilon_sq {
            break;
        }

        if z_sq_norm > bailout_sqr {
            break;
        }

        i += 1;
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn multibrot(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Multibrot_Iteration (puissance configurable, défaut 2.5)
    let mut z = p.seed;
    let mut i = 0u32;
    let d = p.multibrot_power;
    let bailout_sqr = p.bailout * p.bailout;

    // Check once if power is integer for fast-path
    let d_round = d.round();
    let is_integer = (d - d_round).abs() < 1e-10;
    let int_power = if is_integer { d_round as i32 } else { 0 };

    match int_power {
        2 => while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            z = z * z + z_pixel;
            i += 1;
        },
        3 => while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            let z_sq = z * z;
            z = z_sq * z + z_pixel;
            i += 1;
        },
        4 => while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            let z2 = z * z;
            z = z2 * z2 + z_pixel;
            i += 1;
        },
        n if is_integer && n > 4 => while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            // Chain multiply for larger integer powers
            let mut z_pow = z;
            for _ in 1..n {
                z_pow = z_pow * z;
            }
            if !z_pow.re.is_finite() || !z_pow.im.is_finite() { break; }
            z = z_pow + z_pixel;
            i += 1;
        },
        _ => while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            let z_pow = z.powf(d);
            if !z_pow.re.is_finite() || !z_pow.im.is_finite() { break; }
            z = z_pow + z_pixel;
            i += 1;
        },
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

// Julia variants: z_0 = z_pixel, c = p.seed (same formula as Mandelbrot-like counterpart)

#[inline]
fn burning_ship_julia(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    let mut z = z_pixel;
    let mut i = 0u32;
    let bailout_sqr = p.bailout * p.bailout;
    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        let re = z.re.abs();
        let im = z.im.abs();
        let mut z_temp = Complex64::new(re, im);
        z_temp = z_temp * z_temp;
        z = z_temp + p.seed;
        if !z.re.is_finite() || !z.im.is_finite() {
            break;
        }
        i += 1;
    }
    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn tricorn_julia(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    let mut z = z_pixel;
    let mut i = 0u32;
    let bailout_sqr = p.bailout * p.bailout;
    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        let z_conj = Complex64::new(z.re, -z.im);
        let z_temp = z_conj * z_conj;
        z = z_temp + p.seed;
        i += 1;
    }
    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn celtic_julia(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    let mut z = z_pixel;
    let mut i = 0u32;
    let bailout_sqr = p.bailout * p.bailout;
    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        let x = z.re;
        let y = z.im;
        let u = x * x - y * y;
        let v = 2.0 * x * y;
        z = Complex64::new(u.abs() + p.seed.re, v + p.seed.im);
        i += 1;
    }
    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn buffalo_julia(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    let mut z = z_pixel;
    let mut i = 0u32;
    let bailout_sqr = p.bailout * p.bailout;
    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        let z_sq = z * z;
        let re_sq = z_sq.re.abs();
        let im_sq = z_sq.im.abs();
        z = Complex64::new(re_sq + p.seed.re, im_sq + p.seed.im);
        i += 1;
    }
    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn multibrot_julia(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    let mut z = z_pixel;
    let mut i = 0u32;
    let d = p.multibrot_power;
    let bailout_sqr = p.bailout * p.bailout;

    // Check once if power is integer for fast-path
    let d_round = d.round();
    let is_integer = (d - d_round).abs() < 1e-10;
    let int_power = if is_integer { d_round as i32 } else { 0 };

    match int_power {
        2 => while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            z = z * z + p.seed;
            i += 1;
        },
        3 => while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            let z_sq = z * z;
            z = z_sq * z + p.seed;
            i += 1;
        },
        4 => while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            let z2 = z * z;
            z = z2 * z2 + p.seed;
            i += 1;
        },
        n if is_integer && n > 4 => while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            let mut z_pow = z;
            for _ in 1..n {
                z_pow = z_pow * z;
            }
            if !z_pow.re.is_finite() || !z_pow.im.is_finite() { break; }
            z = z_pow + p.seed;
            i += 1;
        },
        _ => while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            let z_pow = z.powf(d);
            if !z_pow.re.is_finite() || !z_pow.im.is_finite() { break; }
            z = z_pow + p.seed;
            i += 1;
        },
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn perpendicular_burning_ship_julia(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    let mut z = z_pixel;
    let mut i = 0u32;
    let bailout_sqr = p.bailout * p.bailout;
    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        let x = z.re;
        let y = z.im;
        let y_abs = y.abs();
        let x2 = x * x;
        let y2 = y_abs * y_abs;
        z = Complex64::new(x2 - y2 + p.seed.re, -2.0 * x * y_abs + p.seed.im);
        i += 1;
    }
    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn alpha_mandelbrot_julia(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    let mut z = z_pixel;
    let mut i = 0u32;
    let bailout_sqr = p.bailout * p.bailout;
    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        let z_sq = z * z;
        let m = z_sq + p.seed;
        let m_sq = m * m;
        z = z_sq + m_sq + p.seed;
        i += 1;
    }
    FractalResult { iteration: i, z, orbit: None, distance: None }
}
