use num_complex::Complex64;

use crate::fractal::{FractalParams, FractalResult, FractalType};
use crate::fractal::orbit_traps::OrbitData;

/// Calcule l'itération pour un point donné, en suivant `FormulaSelector` côté C.
pub fn iterate_point(params: &FractalParams, z_pixel: Complex64) -> FractalResult {
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
    }
}

fn mandelbrot(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Mendelbrot_Iteration: z_{n+1} = z_n^2 + c, dz/dc: z'_{n+1} = 2*z_n*z'_n + 1
    let mut z = p.seed;
    let mut dz = Complex64::new(1.0, 0.0); // dz/dc
    let mut i = 0u32;
    
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
    
    while i < p.iteration_max && z.norm() < p.bailout {
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
    // Estimation de distance: d = |z| * ln(|z|) / |dz/dc| (en unités plan complexe)
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

fn julia(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Julia_Iteration: z_{n+1} = z_n^2 + c, dérivée par rapport à z_0: z'_{n+1} = 2*z_n*z'_n
    let mut z = z_pixel;
    let mut dz = Complex64::new(1.0, 0.0); // dz/dz_0
    let mut i = 0u32;
    
    let mut orbit_data = if p.enable_orbit_traps {
        Some(OrbitData::new(p.orbit_trap_type))
    } else {
        None
    };
    
    if let Some(ref mut orbit) = orbit_data {
        orbit.add_point(z, 0);
    }
    
    while i < p.iteration_max && z.norm() < p.bailout {
        dz = z * dz * 2.0;
        z = z * z + p.seed;
        if !z.re.is_finite() || !z.im.is_finite() {
            if i == 0 {
                z = Complex64::new(p.seed.re * 10.0, p.seed.im * 10.0);
            }
            break;
        }
        i += 1;
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

fn julia_sin(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // JuliaSin_Iteration: z_{n+1} = c * sin(z_n), z_0 = pixel, c = seed
    let mut z = z_pixel;
    let mut i = 0u32;
    while i < p.iteration_max && z.norm() < p.bailout {
        z = p.seed * z.sin();
        i += 1;
    }
    FractalResult { iteration: i, z, orbit: None, distance: None }
}

fn mandelbrot_sin(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // MandelbrotSin_Iteration: z_{n+1} = c * sin(z_n), z_0 = seed, c = pixel (contrepartie de Julia Sin)
    let mut z = p.seed;
    let mut i = 0u32;
    while i < p.iteration_max && z.norm() < p.bailout {
        z = z_pixel * z.sin();
        i += 1;
    }
    FractalResult { iteration: i, z, orbit: None, distance: None }
}

fn newton(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Newton_Iteration
    let mut z = z_pixel;
    let mut i = 0u32;
    // Degree polynomial (pris sur la partie réelle du seed)
    let degree = p.seed.re.round() as i32;
    let degree = if degree <= 0 { 3 } else { degree }; // garde-fou
    while i < p.iteration_max && z.norm() < p.bailout {
        let p_c = degree as f64;
        let z_pow = z.powc(Complex64::new(p_c, 0.0));
        let z_pow_deriv = z.powc(Complex64::new(p_c - 1.0, 0.0));

        let numerator = z_pow - Complex64::new(1.0, 0.0);
        let denominator = Complex64::new(p_c, 0.0) * z_pow_deriv;
        if denominator.norm() < 1e-12 {
            break;
        }
        let z_quot = numerator / denominator;
        z = z - z_quot;
        i += 1;
    }
    FractalResult { iteration: i, z, orbit: None, distance: None }
}

fn phoenix(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Phoenix_Iteration (degree 0, paramètres constants)
    let mut z = z_pixel;
    let mut y = Complex64::new(0.0, 0.0);
    let mut i = 0u32;
    let p1 = 0.56667;
    let p2 = -0.5;

    while i < p.iteration_max && z.norm() < p.bailout {
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

fn barnsley_julia(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Barnsleyj1_Iteration
    let mut z = z_pixel;
    let mut i = 0u32;

    while i < p.iteration_max && z.norm() < p.bailout {
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

fn barnsley_mandelbrot(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Barnsleym1_Iteration
    let mut z = z_pixel;
    let c = z_pixel;
    let mut i = 0u32;

    while i < p.iteration_max && z.norm() < p.bailout {
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

fn magnet_julia(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Magnet1j_Iteration
    let mut z = z_pixel;
    let mut i = 0u32;

    while i < p.iteration_max && z.norm() < p.bailout {
        let seed_minus_one = Complex64::new(p.seed.re - 1.0, p.seed.im);
        let seed_minus_two = Complex64::new(p.seed.re - 2.0, p.seed.im);
        let mut n = z * z + seed_minus_one;
        n = n * n;
        let q = Complex64::new(2.0, 0.0) * z + seed_minus_two;
        // Eviter division par zero
        if q.norm() < 1e-12 {
            break;
        }
        z = n / q;
        i += 1;
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

fn magnet_mandelbrot(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Magnet1m_Iteration
    let c = z_pixel;
    let mut z = Complex64::new(0.0, 0.0);
    let mut i = 0u32;

    while i < p.iteration_max && z.norm() < p.bailout {
        let c_minus_one = Complex64::new(c.re - 1.0, c.im);
        let c_minus_two = Complex64::new(c.re - 2.0, c.im);
        let mut n = z * z + c_minus_one;
        n = n * n;
        let q = Complex64::new(2.0, 0.0) * z + c_minus_two;
        // Eviter division par zero
        if q.norm() < 1e-12 {
            break;
        }
        z = n / q;
        i += 1;
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

fn burning_ship(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // BurningShip_Iteration
    let mut z = p.seed;
    let mut i = 0u32;

    let mut orbit_data = if p.enable_orbit_traps {
        Some(OrbitData::new(p.orbit_trap_type))
    } else {
        None
    };
    
    if let Some(ref mut orbit) = orbit_data {
        orbit.add_point(z, 0);
    }

    while i < p.iteration_max && z.norm() < p.bailout {
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

fn buffalo(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Buffalo_Iteration
    let mut z = p.seed;
    let mut i = 0u32;

    while i < p.iteration_max && z.norm() < p.bailout {
        let z_sq = z * z;
        let re_sq = z_sq.re.abs();
        let im_sq = z_sq.im.abs();
        z = Complex64::new(re_sq + z_pixel.re, im_sq + z_pixel.im);
        i += 1;
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

fn tricorn(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Tricorn_Iteration
    let mut z = p.seed;
    let mut i = 0u32;

    while i < p.iteration_max && z.norm() < p.bailout {
        let z_conj = Complex64::new(z.re, -z.im);
        let z_temp = z_conj * z_conj;
        z = z_temp + z_pixel;
        i += 1;
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

fn mandelbulb(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Mandelbulb_Iteration (2D, puissance 8)
    let mut z = p.seed;
    let mut i = 0u32;

    while i < p.iteration_max && z.norm() < p.bailout {
        // z^8 via multiplications successives
        let mut z_temp = z * z;      // z^2
        z_temp = z_temp * z_temp;    // z^4
        z_temp = z_temp * z_temp;    // z^8
        z = z_temp + z_pixel;
        i += 1;
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

fn perpendicular_burning_ship(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // PerpendicularBurningShip_Iteration
    let mut z = p.seed;
    let mut i = 0u32;

    while i < p.iteration_max && z.norm() < p.bailout {
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

fn celtic(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Celtic_Iteration
    let mut z = p.seed;
    let mut i = 0u32;

    while i < p.iteration_max && z.norm() < p.bailout {
        let x = z.re;
        let y = z.im;
        let u = x * x - y * y;
        let v = 2.0 * x * y;
        z = Complex64::new(u.abs() + z_pixel.re, v + z_pixel.im);
        i += 1;
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

fn alpha_mandelbrot(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // AlphaMandelbrot_Iteration
    let mut z = p.seed;
    let mut i = 0u32;

    while i < p.iteration_max && z.norm() < p.bailout {
        let z_sq = z * z;
        let m = z_sq + z_pixel;
        let m_sq = m * m;
        z = z_sq + m_sq + z_pixel;
        i += 1;
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

fn pickover_stalks(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // PickoverStalks_Iteration (trap based)
    let mut z = p.seed;
    let mut i = 0u32;
    let mut trap_min = 1e10f64;
    let trap_divisor = 0.03f64;

    while i < p.iteration_max && z.norm() < p.bailout {
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

fn nova(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Nova_Iteration
    let mut z = Complex64::new(1.0, 0.0); // z0 = 1
    let mut z_prev;
    let a_relax = Complex64::new(1.0, 0.0);
    let p_poly = 3.0;
    let conv_epsilon = 1e-7;
    let conv_epsilon_sq = conv_epsilon * conv_epsilon;

    let mut i = 0u32;
    while i < p.iteration_max {
        let z_pow = z.powc(Complex64::new(p_poly, 0.0));
        let z_pow_deriv = z.powc(Complex64::new(p_poly - 1.0, 0.0));

        // p(z) = z^p - 1
        let numerator = z_pow - Complex64::new(1.0, 0.0);
        // p'(z) = p * z^(p-1)
        let denominator = Complex64::new(p_poly, 0.0) * z_pow_deriv;

        if denominator.norm() < 1e-10 {
            break;
        }

        let mut newton_step = numerator / denominator;
        newton_step *= a_relax;

        z_prev = z;
        z = z - newton_step + z_pixel;

        let diff_sq = (z - z_prev).norm_sqr();
        let z_sq = z.norm_sqr();
        let denom = if z_sq < 1.0 { 1.0 } else { z_sq };

        if diff_sq / denom < conv_epsilon_sq {
            break;
        }

        if z.norm() > p.bailout {
            break;
        }

        i += 1;
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

fn multibrot(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Multibrot_Iteration (puissance configurable, défaut 2.5)
    let mut z = p.seed;
    let mut i = 0u32;
    let d = p.multibrot_power;

    while i < p.iteration_max && z.norm() < p.bailout {
        let z_pow = z.powf(d);
        if !z_pow.re.is_finite() || !z_pow.im.is_finite() {
            break;
        }
        z = z_pow + z_pixel;
        i += 1;
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

// Julia variants: z_0 = z_pixel, c = p.seed (same formula as Mandelbrot-like counterpart)

fn burning_ship_julia(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    let mut z = z_pixel;
    let mut i = 0u32;
    while i < p.iteration_max && z.norm() < p.bailout {
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

fn tricorn_julia(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    let mut z = z_pixel;
    let mut i = 0u32;
    while i < p.iteration_max && z.norm() < p.bailout {
        let z_conj = Complex64::new(z.re, -z.im);
        let z_temp = z_conj * z_conj;
        z = z_temp + p.seed;
        i += 1;
    }
    FractalResult { iteration: i, z, orbit: None, distance: None }
}

fn celtic_julia(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    let mut z = z_pixel;
    let mut i = 0u32;
    while i < p.iteration_max && z.norm() < p.bailout {
        let x = z.re;
        let y = z.im;
        let u = x * x - y * y;
        let v = 2.0 * x * y;
        z = Complex64::new(u.abs() + p.seed.re, v + p.seed.im);
        i += 1;
    }
    FractalResult { iteration: i, z, orbit: None, distance: None }
}

fn buffalo_julia(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    let mut z = z_pixel;
    let mut i = 0u32;
    while i < p.iteration_max && z.norm() < p.bailout {
        let z_sq = z * z;
        let re_sq = z_sq.re.abs();
        let im_sq = z_sq.im.abs();
        z = Complex64::new(re_sq + p.seed.re, im_sq + p.seed.im);
        i += 1;
    }
    FractalResult { iteration: i, z, orbit: None, distance: None }
}

fn multibrot_julia(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    let mut z = z_pixel;
    let mut i = 0u32;
    let d = p.multibrot_power;
    while i < p.iteration_max && z.norm() < p.bailout {
        let z_pow = z.powf(d);
        if !z_pow.re.is_finite() || !z_pow.im.is_finite() {
            break;
        }
        z = z_pow + p.seed;
        i += 1;
    }
    FractalResult { iteration: i, z, orbit: None, distance: None }
}

fn perpendicular_burning_ship_julia(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    let mut z = z_pixel;
    let mut i = 0u32;
    while i < p.iteration_max && z.norm() < p.bailout {
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

fn alpha_mandelbrot_julia(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    let mut z = z_pixel;
    let mut i = 0u32;
    while i < p.iteration_max && z.norm() < p.bailout {
        let z_sq = z * z;
        let m = z_sq + p.seed;
        let m_sq = m * m;
        z = z_sq + m_sq + p.seed;
        i += 1;
    }
    FractalResult { iteration: i, z, orbit: None, distance: None }
}

