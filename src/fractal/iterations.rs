use num_complex::Complex64;

use crate::fractal::bytecode::{formula_for_params, Formula};
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
/// Formule bytecode du path f64 standard pour ces paramètres (`None` =
/// fonctions dédiées par type). À compiler UNE fois par rendu puis passer à
/// [`iterate_point_with_formula`] : `iterate_point` la recompilait deux fois
/// par pixel (quatre allocations).
#[inline]
pub fn standard_formula(params: &FractalParams) -> Option<Formula> {
    if params.engine.use_bytecode_engine {
        formula_for_params(params)
    } else {
        None
    }
}

/// Dispatch interpréteur : (z₀, c) selon convention Mandelbrot/Julia.
/// Construit aussi un OrbitData si orbit_traps est demandé.
#[inline]
fn iterate_via_bytecode(
    params: &FractalParams,
    formula: &Formula,
    z_pixel: Complex64,
) -> FractalResult {
    let (z0, c) = if Formula::is_julia_for(params.fractal_type) {
        (z_pixel, params.seed)
    } else {
        (params.seed, z_pixel)
    };

    let needs_orbit = params.channels.enable_orbit_traps;
    let needs_distance = params.channels.enable_distance_estimation;
    let needs_interior = params.channels.enable_interior_detection;

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
        // G4 : phase cyclée par itération (`iter` = nb de phases complétées, Add
        // dernier op de chaque phase). n_phases=1 ⇒ toujours phases[0] (identique
        // à l'ancien comportement mono-phase, goldens inchangés).
        let n_phases = formula.phases.len();
        let mut orbit_data = if needs_orbit {
            let mut od = OrbitData::new(params.channels.orbit_trap_type);
            od.add_point(z, 0);
            Some(od)
        } else {
            None
        };

        while iter < params.iteration_max {
            if z.norm_sqr() >= bailout_sqr {
                break;
            }
            let phase = &formula.phases[iter as usize % n_phases];
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
            if dz_norm.is_finite() && dz_norm > 0.0 && dz_norm < params.channels.interior_threshold {
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

    let r = crate::fractal::bytecode::interp::iterate_bytecode_f64_cut(
        formula,
        z0,
        c,
        params.iteration_max,
        params.bailout,
    );
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
    let formula = standard_formula(params);
    iterate_point_with_formula(params, formula.as_ref(), z_pixel)
}

/// [`iterate_point`] avec la formule déjà compilée par l'appelant —
/// `formula` DOIT valoir `standard_formula(params)`.
#[inline]
pub fn iterate_point_with_formula(
    params: &FractalParams,
    formula: Option<&Formula>,
    z_pixel: Complex64,
) -> FractalResult {
    if let Some(formula) = formula {
        return iterate_via_bytecode(params, formula, z_pixel);
    }
    match params.fractal_type {
        FractalType::VonKoch | FractalType::Dragon => {
            panic!("vector fractals must be rendered via render_von_koch/render_dragon")
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
        special @ (FractalType::Buddhabrot
        | FractalType::Lyapunov
        | FractalType::Nebulabrot
        | FractalType::AntiBuddhabrot) => panic!(
            "{special:?} must be rendered via the cancellable dispatcher, not iterate_point()"
        ),
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
    let use_periodicity = !p.channels.enable_orbit_traps && !p.channels.enable_distance_estimation;
    let bailout_sqr = p.bailout * p.bailout;

    // Cardioid and period-2 bulb detection (skip if orbit traps or distance estimation needed)
    if !p.channels.enable_orbit_traps && !p.channels.enable_distance_estimation {
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
    let mut orbit_data = if p.channels.enable_orbit_traps {
        Some(OrbitData::new(p.channels.orbit_trap_type))
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
    let distance = if p.channels.enable_distance_estimation && i < p.iteration_max && i > 0 {
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
    let use_periodicity = !p.channels.enable_orbit_traps && !p.channels.enable_distance_estimation;
    let bailout_sqr = p.bailout * p.bailout;

    let mut orbit_data = if p.channels.enable_orbit_traps {
        Some(OrbitData::new(p.channels.orbit_trap_type))
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
    let distance = if p.channels.enable_distance_estimation && i < p.iteration_max && i > 0 {
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

/// `sin(z)` complexe en UN `sincos` et UN `expm1`, au lieu des quatre appels
/// libm de `Complex::sin` (`sin`/`cos` fusionnés, `sinh` et `cosh` qui
/// calculent chacun leur exponentielle). Avec `t = e^|y| − 1` :
/// `sinh|y| = (t + t/(t+1))/2` et `cosh y = 1 + (t/2)·t/(t+1)` — sommes de
/// termes positifs, sans cancellation (précision ~1 ulp, comme la libm), mais
/// pas les mêmes bits que `Complex::sin`. Domaine : |Im z| < ~709 (les boucles
/// Sin l'appellent sous le rayon d'évasion).
#[inline(always)]
fn csin(z: Complex64) -> Complex64 {
    let (s, c, sh, ch) = csin_parts(z);
    Complex64::new(s * ch, c * sh)
}

// Garder les deux résultats dans le même appel : après inlining, LLVM peut
// émettre sin et cos séparément dans une boucle et sincos dans l'autre.
// Sur macOS ARM64, leurs arrondis divergent et changent l'orbite.
#[inline(never)]
fn orbit_sin_cos(x: f64) -> (f64, f64) {
    x.sin_cos()
}

/// (sin x, cos x, sinh y, cosh y), communs au rendu et à sa borne d'erreur.
#[inline(always)]
pub(crate) fn csin_parts(z: Complex64) -> (f64, f64, f64, f64) {
    const TINY_HYPERBOLIC: f64 = 1.0 / (1u64 << 28) as f64;
    let (s, c) = orbit_sin_cos(z.re);
    let a = z.im.abs();
    if a < TINY_HYPERBOLIC {
        // À cette échelle, sinh(y) et cosh(y) s’arrondissent à y et 1.
        return (s, c, z.im, 1.0);
    }
    let t = a.exp_m1();
    let q = t / (t + 1.0);
    let sh = 0.5 * (t + q);
    let ch = 1.0 + 0.5 * t * q;
    (s, c, sh.copysign(z.im), ch)
}

/// Racine carrée PRINCIPALE (même branche que `Complex::sqrt` et
/// `Complex::powf` : argument dans (−π, π], `Im z = −0` du côté −π), sous forme
/// algébrique — deux racines réelles et une division au lieu de
/// `hypot`/`atan2`/`sincos`.
#[inline(always)]
fn csqrt_principal(z: Complex64) -> Complex64 {
    let (x, y) = (z.re, z.im);
    if x == 0.0 && y == 0.0 {
        return Complex64::new(0.0, y);
    }
    let n = x * x + y * y;
    let r = if n > 1e-290 && n < 1e290 { n.sqrt() } else { x.hypot(y) };
    let t = (0.5 * (r + x.abs())).sqrt();
    if x >= 0.0 {
        Complex64::new(t, y / (2.0 * t))
    } else {
        Complex64::new(y.abs() / (2.0 * t), t.copysign(y))
    }
}

/// `z^d` pour une puissance DEMI-entière `d = k + ½` (k ≥ 0) :
/// `z^k · √z`, égal à `exp(d·Log z)` sur la branche principale (l'argument
/// de `z^k` n'est jamais réduit modulo 2π, celui de `√z` vaut `θ/2`).
/// Multibrot par défaut (d = 2,5) : boucle ×2,1-2,6 (la racine coûte ~9 ns,
/// `hypot` + `atan2` + `pow` + `sincos` ~110 ns), plus précis (aucune
/// exponentielle d'un logarithme arrondi). Pas les mêmes bits que `powf`.
#[inline(always)]
fn cpow_half_int(z: Complex64, k: u32) -> Complex64 {
    // `z^k` ne dépend pas de la racine : calculée en parallèle de la chaîne
    // racine → racine → division (la boucle est bornée par la latence).
    let s = csqrt_principal(z);
    match k {
        0 => s,
        1 => s * z,
        2 => s * (z * z),
        _ => {
            let mut zk = z;
            for _ in 1..k {
                zk = zk * z;
            }
            s * zk
        }
    }
}

/// `Some(k)` si `d = k + ½` exactement, k ≥ 0 entier (k ≤ 64).
#[inline]
pub(crate) fn half_integer_power(d: f64) -> Option<u32> {
    let k = d - 0.5;
    (k >= 0.0 && k <= 64.0 && k == k.floor()).then_some(k as u32)
}

#[inline]
fn julia_sin(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // JuliaSin_Iteration: z_{n+1} = c * sin(z_n), z_0 = pixel, c = seed
    let mut z = z_pixel;
    let mut i = 0u32;
    let mut cyc = super::cycle::CycleCut::<2>::new();
    let bailout_sqr = p.bailout * p.bailout;
    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        z = p.seed * csin(z);
        i += 1;
        cyc.observe_each(&mut i, p.iteration_max, [z.re, z.im]);
    }
    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn mandelbrot_sin(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // MandelbrotSin_Iteration: z_{n+1} = c * sin(z_n), z_0 = seed, c = pixel (contrepartie de Julia Sin)
    let mut z = p.seed;
    let mut i = 0u32;
    let mut cyc = super::cycle::CycleCut::<2>::new();
    let bailout_sqr = p.bailout * p.bailout;
    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        z = z_pixel * csin(z);
        i += 1;
        cyc.observe_each(&mut i, p.iteration_max, [z.re, z.im]);
    }
    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn newton(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    let degree = p.seed.re.round() as i32;
    let degree = if degree <= 0 { 3 } else { degree as u32 };
    // Si |z| < 16 et d ≤ 64 tous les produits de la division tiennent dans f64.
    if degree > 64 || p.bailout.abs() > 16.0 {
        return newton_orbit::<true>(p, z_pixel, degree, |z| z.powu(degree - 1));
    }
    // Un exposant constant évite de le décomposer à chaque itération et
    // conserve l'ordre des multiplications de powu.
    match degree {
        1 => newton_orbit::<false>(p, z_pixel, degree, |_| Complex64::new(1.0, 0.0)),
        2 => newton_orbit::<false>(p, z_pixel, degree, |z| z),
        3 => newton_orbit::<false>(p, z_pixel, degree, |z| z * z),
        4 => newton_orbit::<false>(p, z_pixel, degree, |z| z.powu(3)),
        5 => newton_orbit::<false>(p, z_pixel, degree, |z| z.powu(4)),
        6 => newton_orbit::<false>(p, z_pixel, degree, |z| z.powu(5)),
        7 => newton_orbit::<false>(p, z_pixel, degree, |z| z.powu(6)),
        8 => newton_orbit::<false>(p, z_pixel, degree, |z| z.powu(7)),
        d => newton_orbit::<false>(p, z_pixel, d, |z| z.powu(d - 1)),
    }
}

#[inline]
fn newton_orbit<const CHECK_RANGE: bool>(
    p: &FractalParams,
    mut z: Complex64,
    degree: u32,
    derivative_power: impl Fn(Complex64) -> Complex64,
) -> FractalResult {
    let mut i = 0u32;
    let mut cyc = super::cycle::CycleCut::<2>::new();
    let detect_short_cycles = super::cycle::shortcut_enabled();
    let mut previous = z;
    let bailout_sqr = p.bailout * p.bailout;
    let one = Complex64::new(1.0, 0.0);
    let degree_c = Complex64::new(f64::from(degree), 0.0);
    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        let power = derivative_power(z);
        let numerator = power * z - one;
        let denominator = degree_c * power;
        let norm = denominator.norm_sqr();
        if norm < 1e-24 { break; }
        let current = z;
        z -= if !CHECK_RANGE || norm <= 1e150 {
            // |z|² est fini à l'entrée. Dans cette plage, les produits croisés
            // sont bornés par |z|·norm/d + √norm et tiennent dans f64.
            numerator / denominator
        } else {
            newton_large_step(z, degree, numerator, denominator, norm)
        };
        i += 1;
        // Aux racines, l'arrondi fige souvent z ou le fait alterner entre
        // deux valeurs. À égalité exacte avec z deux pas plus tôt, le
        // dernier état dépend seulement de la parité des pas restants.
        if detect_short_cycles
            && z.re.to_bits() == previous.re.to_bits()
            && z.im.to_bits() == previous.im.to_bits()
        {
            if (p.iteration_max - i) % 2 == 1 {
                z = current;
            }
            i = p.iteration_max;
            break;
        }
        previous = current;
        cyc.observe_each(&mut i, p.iteration_max, [z.re, z.im]);
    }
    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[cold]
fn newton_large_step(
    z: Complex64,
    degree: u32,
    numerator: Complex64,
    denominator: Complex64,
    norm: f64,
) -> Complex64 {
    if norm.is_finite() {
        let step = numerator / denominator;
        if step.is_finite() {
            return step;
        }
    }
    // (z^d − 1)/(d·z^(d−1)) = (z − (1/z)^(d−1))/d.
    // Si le quotient déborde, |z| > 1 et la puissance de l'inverse reste bornée.
    let inverse = z.conj() / z.norm_sqr();
    (z - inverse.powu(degree - 1)) / f64::from(degree)
}

#[inline]
fn phoenix(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Phoenix_Iteration (degree 0, paramètres constants)
    let mut z = z_pixel;
    let mut y = Complex64::new(0.0, 0.0);
    let mut i = 0u32;
    let mut cyc = super::cycle::CycleCut::<4>::new();
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
        cyc.observe(&mut i, p.iteration_max, [z.re, z.im, y.re, y.im]);
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn barnsley_julia(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Barnsleyj1_Iteration
    let mut z = z_pixel;
    let mut i = 0u32;
    let mut cyc = super::cycle::CycleCut::<2>::new();
    let bailout_sqr = p.bailout * p.bailout;

    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        let re = z.re;
        if re >= 0.0 {
            z = (z - Complex64::new(1.0, 0.0)) * p.seed;
        } else {
            z = (z + Complex64::new(1.0, 0.0)) * p.seed;
        }
        i += 1;
        cyc.observe(&mut i, p.iteration_max, [z.re, z.im]);
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn barnsley_mandelbrot(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Barnsleym1_Iteration
    let mut z = z_pixel;
    let c = z_pixel;
    let mut i = 0u32;
    let mut cyc = super::cycle::CycleCut::<2>::new();
    let bailout_sqr = p.bailout * p.bailout;

    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        let re = z.re;
        if re >= 0.0 {
            z = (z - Complex64::new(1.0, 0.0)) * c;
        } else {
            z = (z + Complex64::new(1.0, 0.0)) * c;
        }
        i += 1;
        cyc.observe(&mut i, p.iteration_max, [z.re, z.im]);
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn magnet_julia(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Magnet1j_Iteration: z_{n+1} = ((z_n² + c - 1) / (2*z_n + c - 2))²
    let mut z = z_pixel;
    let mut i = 0u32;
    let mut cyc = super::cycle::CycleCut::<2>::new();
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
        cyc.observe_each(&mut i, p.iteration_max, [z.re, z.im]);
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn magnet_mandelbrot(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Magnet1m_Iteration: z_{n+1} = ((z_n² + c - 1) / (2*z_n + c - 2))²
    let c = z_pixel;
    let mut z = Complex64::new(0.0, 0.0);
    let mut i = 0u32;
    let mut cyc = super::cycle::CycleCut::<2>::new();
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
        cyc.observe_each(&mut i, p.iteration_max, [z.re, z.im]);
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn burning_ship(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // BurningShip_Iteration
    let mut z = p.seed;
    let mut i = 0u32;
    let bailout_sqr = p.bailout * p.bailout;

    let mut orbit_data = if p.channels.enable_orbit_traps {
        Some(OrbitData::new(p.channels.orbit_trap_type))
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
    let mut cyc = super::cycle::CycleCut::<2>::new();
    let bailout_sqr = p.bailout * p.bailout;

    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        // z^8 via multiplications successives
        let mut z_temp = z * z;      // z^2
        z_temp = z_temp * z_temp;    // z^4
        z_temp = z_temp * z_temp;    // z^8
        z = z_temp + z_pixel;
        i += 1;
        cyc.observe(&mut i, p.iteration_max, [z.re, z.im]);
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
    let mut cyc = super::cycle::CycleCut::<2>::new();
    let bailout_sqr = p.bailout * p.bailout;

    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        let z_sq = z * z;
        let m = z_sq + z_pixel;
        let m_sq = m * m;
        z = z_sq + m_sq + z_pixel;
        i += 1;
        cyc.observe(&mut i, p.iteration_max, [z.re, z.im]);
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn pickover_stalks(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // PickoverStalks_Iteration (trap based)
    let mut z = p.seed;
    let mut i = 0u32;
    let mut cyc = super::cycle::CycleCut::<2>::new();
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
        cyc.observe(&mut i, p.iteration_max, [z.re, z.im]);
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
    let mut cyc = super::cycle::CycleCut::<2>::new();
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
        cyc.observe_each(&mut i, p.iteration_max, [z.re, z.im]);
    }

    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[inline]
fn multibrot(p: &FractalParams, z_pixel: Complex64) -> FractalResult {
    // Multibrot_Iteration (puissance configurable, défaut 2.5)
    let mut z = p.seed;
    let mut i = 0u32;
    let mut cyc = super::cycle::CycleCut::<2>::new();
    let d = p.formula.multibrot_power;
    let bailout_sqr = p.bailout * p.bailout;

    // Check once if power is integer for fast-path
    let d_round = d.round();
    let is_integer = (d - d_round).abs() < 1e-10;
    let int_power = if is_integer { d_round as i32 } else { 0 };
    let half = if is_integer { None } else { half_integer_power(d) };

    match int_power {
        2 => while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            z = z * z + z_pixel;
            i += 1;
            cyc.observe_each(&mut i, p.iteration_max, [z.re, z.im]);
        },
        3 => while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            let z_sq = z * z;
            z = z_sq * z + z_pixel;
            i += 1;
            cyc.observe_each(&mut i, p.iteration_max, [z.re, z.im]);
        },
        4 => while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            let z2 = z * z;
            z = z2 * z2 + z_pixel;
            i += 1;
            cyc.observe_each(&mut i, p.iteration_max, [z.re, z.im]);
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
            cyc.observe_each(&mut i, p.iteration_max, [z.re, z.im]);
        },
        _ if half.is_some() => while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            let z_pow = cpow_half_int(z, half.unwrap_or(0));
            if !z_pow.re.is_finite() || !z_pow.im.is_finite() { break; }
            z = z_pow + z_pixel;
            i += 1;
            cyc.observe_each(&mut i, p.iteration_max, [z.re, z.im]);
        },
        _ => while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            let z_pow = z.powf(d);
            if !z_pow.re.is_finite() || !z_pow.im.is_finite() { break; }
            z = z_pow + z_pixel;
            i += 1;
            cyc.observe_each(&mut i, p.iteration_max, [z.re, z.im]);
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
    let mut cyc = super::cycle::CycleCut::<2>::new();
    let d = p.formula.multibrot_power;
    let bailout_sqr = p.bailout * p.bailout;

    // Check once if power is integer for fast-path
    let d_round = d.round();
    let is_integer = (d - d_round).abs() < 1e-10;
    let int_power = if is_integer { d_round as i32 } else { 0 };
    let half = if is_integer { None } else { half_integer_power(d) };

    match int_power {
        2 => while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            z = z * z + p.seed;
            i += 1;
            cyc.observe_each(&mut i, p.iteration_max, [z.re, z.im]);
        },
        3 => while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            let z_sq = z * z;
            z = z_sq * z + p.seed;
            i += 1;
            cyc.observe_each(&mut i, p.iteration_max, [z.re, z.im]);
        },
        4 => while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            let z2 = z * z;
            z = z2 * z2 + p.seed;
            i += 1;
            cyc.observe_each(&mut i, p.iteration_max, [z.re, z.im]);
        },
        n if is_integer && n > 4 => while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            let mut z_pow = z;
            for _ in 1..n {
                z_pow = z_pow * z;
            }
            if !z_pow.re.is_finite() || !z_pow.im.is_finite() { break; }
            z = z_pow + p.seed;
            i += 1;
            cyc.observe_each(&mut i, p.iteration_max, [z.re, z.im]);
        },
        _ if half.is_some() => while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            let z_pow = cpow_half_int(z, half.unwrap_or(0));
            if !z_pow.re.is_finite() || !z_pow.im.is_finite() { break; }
            z = z_pow + p.seed;
            i += 1;
            cyc.observe_each(&mut i, p.iteration_max, [z.re, z.im]);
        },
        _ => while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
            let z_pow = z.powf(d);
            if !z_pow.re.is_finite() || !z_pow.im.is_finite() { break; }
            z = z_pow + p.seed;
            i += 1;
            cyc.observe_each(&mut i, p.iteration_max, [z.re, z.im]);
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
    let mut cyc = super::cycle::CycleCut::<2>::new();
    let bailout_sqr = p.bailout * p.bailout;
    while i < p.iteration_max && z.norm_sqr() < bailout_sqr {
        let z_sq = z * z;
        let m = z_sq + p.seed;
        let m_sq = m * m;
        z = z_sq + m_sq + p.seed;
        i += 1;
        cyc.observe(&mut i, p.iteration_max, [z.re, z.im]);
    }
    FractalResult { iteration: i, z, orbit: None, distance: None }
}

#[cfg(test)]
mod fast_complex_tests {
    use super::*;

    fn rel(a: Complex64, b: Complex64) -> f64 {
        (a - b).norm() / b.norm().max(f64::MIN_POSITIVE)
    }

    fn samples() -> Vec<Complex64> {
        let mut v = Vec::new();
        for i in -40..=40 {
            for j in -40..=40 {
                v.push(Complex64::new(i as f64 * 0.0973, j as f64 * 0.0517));
            }
        }
        for &(x, y) in &[(-1.5, 0.0), (-1.5, -0.0), (-2.0, 1e-300), (-2.0, -1e-300),
                         (1e-200, 1e-200), (-1e-200, 0.0), (0.0, -3.0), (3.9, -3.9)] {
            v.push(Complex64::new(x, y));
        }
        v
    }

    /// Les composantes exposées au suivi de fiabilité recomposent `csin` au
    /// bit près (signes des zéros, raccourci |Im z| < 2⁻²⁸ compris).
    #[test]
    fn csin_parts_rebuild_csin_bit_for_bit() {
        let mut zs = samples();
        for &y in &[0.0, -0.0, 1e-9, -1e-9, 3.7e-9, 2e-8, -0.3, 5.0] {
            for &x in &[0.0, -0.0, 1.0, -2.5, 3.14159, 1e-300] {
                zs.push(Complex64::new(x, y));
            }
        }
        for z in zs {
            let (s, c, sh, ch) = csin_parts(z);
            let (a, b) = (csin(z), Complex64::new(s * ch, c * sh));
            assert_eq!(a.re.to_bits(), b.re.to_bits(), "{z}");
            assert_eq!(a.im.to_bits(), b.im.to_bits(), "{z}");
        }
    }

    #[test]
    fn newton_overflowing_quotients_match_mpc() {
        use crate::fractal::default_params_for_type;
        use crate::fractal::gmp::{iterate_point_mpc, MpcParams};
        use rug::Complex;
        let mut p = default_params_for_type(FractalType::Newton, 1, 1);
        p.iteration_max = 1;
        p.engine.precision_bits = 256;
        for (degree, start) in [
            (384, Complex64::new(3.0, 0.0)),  // carré du dénominateur infini
            (512, Complex64::new(-2.5, 1.25)),
            (1024, Complex64::new(3.0, -0.0)), // la puissance elle-même déborde
            (2, Complex64::new(1e105, 1e104)), // produits croisés infinis
        ] {
            p.seed.re = f64::from(degree);
            p.bailout = (start.norm() * 2.0).max(4.0);
            let (n, expected) = iterate_point_mpc(
                &MpcParams::from_params(&p),
                &Complex::with_val(256, (start.re, start.im)),
            );
            let actual = newton(&p, start);
            assert_eq!((actual.iteration, n), (1, 1));
            let expected = Complex64::new(expected.real().to_f64(), expected.imag().to_f64());
            assert!(rel(actual.z, expected) <= 4.0 * f64::EPSILON, "d={degree} z={start}");
        }
        p.seed.re = 1024.0;
        p.bailout = 4.0;
        p.iteration_max = 2048;
        let basin = newton(&p, Complex64::new(3.0, 0.000125));
        assert_eq!(basin.iteration, p.iteration_max);
        assert_eq!(basin.z, Complex64::new(1.0, 0.0));
    }

    /// `csin` = `Complex::sin` à quelques ulp près, signes des zéros compris.
    #[test]
    fn csin_matches_complex_sin() {
        for z in samples() {
            let (a, b) = (csin(z), z.sin());
            assert!(rel(a, b) < 4.0 * f64::EPSILON, "{z:?}: {a:?} vs {b:?}");
            if b.im == 0.0 {
                assert_eq!(a.im.is_sign_negative(), b.im.is_sign_negative(), "{z:?}");
            }
        }
    }

    /// `z^(k+½)` algébrique = `powf` (branche principale, y compris de part
    /// et d'autre de la coupure le long des réels négatifs).
    #[test]
    fn half_integer_power_matches_powf_on_principal_branch() {
        for k in [0u32, 1, 2, 3, 7] {
            let d = k as f64 + 0.5;
            assert_eq!(half_integer_power(d), Some(k));
            for z in samples() {
                let (a, b) = (cpow_half_int(z, k), z.powf(d));
                if b.norm() == 0.0 {
                    assert_eq!(a.norm(), 0.0);
                    continue;
                }
                assert!(rel(a, b) < 1e-12, "z={z:?} d={d}: {a:?} vs {b:?}");
            }
        }
        for d in [2.0, 2.7, -0.5, 0.25, 65.5] {
            assert_eq!(half_integer_power(d), None, "{d}");
        }
    }
}

#[cfg(test)]
mod cycle_cut_tests {
    use super::*;
    use crate::fractal::cycle::FORCE_OFF;
    use crate::fractal::definitions::default_params_for_type;

    fn with_cut<T>(cut: bool, f: impl FnOnce() -> T) -> T {
        FORCE_OFF.with(|c| c.set(!cut));
        let r = f();
        FORCE_OFF.with(|c| c.set(false));
        r
    }

    /// Grille de points couvrant la vue par défaut du type.
    fn grid(p: &FractalParams, nx: usize, ny: usize) -> Vec<Complex64> {
        let mut pts = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                pts.push(Complex64::new(
                    p.center_x + p.span_x * ((i as f64 + 0.5) / nx as f64 - 0.5),
                    p.center_y + p.span_y * ((j as f64 + 0.5) / ny as f64 - 0.5),
                ));
            }
        }
        pts
    }

    /// Verrou : pour chaque boucle scalaire équipée du raccourci des cycles
    /// exacts, le résultat (compte d'itérations ET z final, au bit près) est
    /// celui de l'itération complète, sur toute la vue par défaut et pour des
    /// plafonds tombant à des positions variées du cycle.
    #[test]
    fn cycle_cut_preserves_every_scalar_loop_bit_for_bit() {
        let mut cases: Vec<(String, FractalParams)> = [
            FractalType::JuliaSin,
            FractalType::MandelbrotSin,
            FractalType::Newton,
            FractalType::Phoenix,
            FractalType::BarnsleyJulia,
            FractalType::BarnsleyMandelbrot,
            FractalType::MagnetJulia,
            FractalType::MagnetMandelbrot,
            FractalType::Mandelbulb,
            FractalType::AlphaMandelbrot,
            FractalType::PickoverStalks,
            FractalType::Multibrot,
            FractalType::MultibrotJulia,
            FractalType::AlphaMandelbrotJulia,
            FractalType::Nova,
        ]
        .into_iter()
        .map(|ty| (format!("{ty:?}"), default_params_for_type(ty, 64, 48)))
        .collect();
        // Newton degrés 2 et 5 (branches dédiées), Multibrot entiers par les
        // boucles historiques (`--no-bytecode`).
        for degree in [2.0, 5.0] {
            let mut p = default_params_for_type(FractalType::Newton, 64, 48);
            p.seed.re = degree;
            cases.push((format!("Newton degré {degree}"), p));
        }
        for power in [0.5, 3.5, 2.7] {
            for ty in [FractalType::Multibrot, FractalType::MultibrotJulia] {
                let mut p = default_params_for_type(ty, 64, 48);
                p.formula.multibrot_power = power;
                cases.push((format!("{ty:?}^{power}"), p));
            }
        }
        for power in [2.0, 3.0, 4.0, 6.0] {
            for ty in [FractalType::Multibrot, FractalType::MultibrotJulia] {
                let mut p = default_params_for_type(ty, 64, 48);
                p.formula.multibrot_power = power;
                p.engine.use_bytecode_engine = false;
                cases.push((format!("{ty:?}^{power} historique"), p));
            }
        }
        // Path scalaire bytecode : hybrides multi-phase (2 et 3 phases) et
        // opcodes à rotation — comparés aussi à l'interpréteur de référence.
        for phases in [
            vec![FractalType::Mandelbrot, FractalType::BurningShip],
            vec![FractalType::Mandelbrot, FractalType::Mandelbrot, FractalType::Celtic],
            vec![FractalType::Tricorn, FractalType::Mandelbrot],
        ] {
            let mut p = default_params_for_type(FractalType::Mandelbrot, 64, 48);
            p.formula.hybrid_phases = Some(phases.clone());
            cases.push((format!("hybride {phases:?}"), p));
        }
        let mut shortened = 0usize;
        for (label, base) in &cases {
            let pts = grid(base, 24, 18);
            for max in [1u32, 8, 9, 100, 1000, 3001] {
                let mut p = base.clone();
                p.iteration_max = max;
                for &z in &pts {
                    let a = with_cut(true, || iterate_point(&p, z));
                    let b = with_cut(false, || iterate_point(&p, z));
                    assert!(
                        a.iteration == b.iteration
                            && a.z.re.to_bits() == b.z.re.to_bits()
                            && a.z.im.to_bits() == b.z.im.to_bits(),
                        "{label} {z:?} max={max} : raccourci ({}, {:?}) ≠ complet ({}, {:?})",
                        a.iteration,
                        a.z,
                        b.iteration,
                        b.z
                    );
                }
            }
            // Le raccourci raccourcit : au moins un saut sur la grille dès
            // qu'elle contient des points qui n'en sortent pas.
            let mut p = base.clone();
            p.iteration_max = 20_000;
            let before = crate::fractal::cycle::CUTS.with(|c| c.get());
            for &z in &pts {
                with_cut(true, || iterate_point(&p, z));
            }
            if crate::fractal::cycle::CUTS.with(|c| c.get()) > before {
                shortened += 1;
            }
        }
        assert!(shortened >= 12, "raccourci inopérant : {shortened} cas raccourcis sur {}", cases.len());
    }
}
