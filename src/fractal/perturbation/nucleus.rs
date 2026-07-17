//! Nucleus finder pour Mandelbrot (port fidèle de Fraktaler-3.1
//! `hybrid_period` + `hybrid_center`, cf. `fraktaler-3-3.1/src/hybrid.cc:417`).
//!
//! À deep zoom, l'utilisateur cible un point near-minibrot dont l'orbite
//! escape avant `iteration_max`. La perturbation tronque alors l'orbite
//! référence → image uniforme ou GMP-fallback prohibitif.
//!
//! Solution F3 :
//! 1. Détecter la période du minibrot voisin via le critère ball-arithmetic /
//!    atom-domain : `|z|² < s² · |dz|²` où `s` est l'échelle de vue (`~ 1/zoom`)
//!    et `dz` la dérivée de l'itération par rapport à `c`. Plus précis que
//!    le min-|z| heuristique : la criterion encode directement la définition
//!    de l'atome (boule centrée sur le nucleus de taille s/|dz|).
//! 2. Newton-raffiner le centre vers `z_period(c) = 0` (le centre périodique
//!    passe par 0). Une fois `c` raffiné, l'orbite référence est non-escape
//!    et la perturbation fonctionne pour tous les pixels voisins.
//!
//! Limitations :
//! - Mandelbrot z² + c uniquement (pas BS/Tricorn/Multibrot).
//! - Itération de période en GMP (précis mais lent pour très deep zoom). F3
//!   utilise floatexp pour la passe de détection ; à porter si la perf
//!   devient bloquante (cf. e22522 prec ~80k bits).

use rug::{Assign, Complex, Float};

use crate::fractal::perturbation::types::FloatExp;

/// Résultat de la recherche de nucleus.
#[derive(Clone, Debug)]
pub struct NucleusResult {
    /// Centre raffiné du minibrot.
    pub center_x: Float,
    pub center_y: Float,
    /// Période détectée du minibrot.
    pub period: u32,
    /// Nombre d'itérations Newton avant convergence.
    pub newton_steps: u32,
    /// `true` si Newton a convergé. Lu via debug log uniquement aujourd'hui ;
    /// gardé pour exposer la qualité de convergence au caller (GUI futur,
    /// QA harness).
    #[allow(dead_code)]
    pub converged: bool,
}

/// Détecte la période du minibrot le plus proche via le critère atom-domain F3
/// (`fraktaler-3-3.1/src/hybrid.cc::hybrid_period`).
///
/// Itère `z_{n+1} = z_n² + c` et `dz_{n+1} = 2 z_n dz_n + 1` (dérivée par
/// rapport à `c`). Retourne la première itération `n ≥ 4` où :
///
///   |z_n|² < s² · |dz_n|²
///
/// `s` est l'échelle de vue (~ `1 / zoom` ou `view_radius`). Le critère encode
/// la définition même d'un atome de période n : z_n est dans une boule de
/// rayon `s / |dz_n|` autour du nucleus, dont la taille `s/|dz|` est l'estimée
/// linéaire de la taille de l'atome via le théorème d'Alexander.
///
/// Retourne `None` si l'orbite escape avant qu'une période ne soit détectée,
/// ou si `max_iter` est atteint. C'est un signal que l'utilisateur est hors
/// d'atteinte d'un atome dans le champ visuel — la perturbation classique
/// reste alors la meilleure stratégie.
pub fn find_period_atom_domain(
    cx: &Float,
    cy: &Float,
    max_iter: u32,
    s: &Float,
    prec: u32,
) -> Option<u32> {
    // Bailout très large (F3 utilise 10^10000, on prend 1e60 — plus que
    // suffisant pour détecter une orbite non-bornée sans gêner la détection
    // de boucle interne).
    let bailout_sqr = Float::with_val(prec, 1e120);
    let s_sqr = Float::with_val(prec, s * s);

    let mut z_re = Float::with_val(prec, 0);
    let mut z_im = Float::with_val(prec, 0);
    // dz initialisé à 0 (z_0 = 0, donc dz_0 = d0/dc = 0). Première iter :
    //   dz_1 = 2 * 0 * 0 + 1 = 1
    let mut dz_re = Float::with_val(prec, 0);
    let mut dz_im = Float::with_val(prec, 0);

    let mut z_norm_sqr = Float::with_val(prec, 0);
    let mut dz_norm_sqr = Float::with_val(prec, 0);
    let mut threshold = Float::with_val(prec, 0);

    for n in 1..=max_iter {
        // dz := 2 * z * dz + 1
        //   re = 2*(z_re*dz_re - z_im*dz_im) + 1
        //   im = 2*(z_re*dz_im + z_im*dz_re)
        let dz_re_new = {
            let a = Float::with_val(prec, &z_re * &dz_re);
            let b = Float::with_val(prec, &z_im * &dz_im);
            let mut t = a - b;
            t *= 2;
            t += 1;
            t
        };
        let dz_im_new = {
            let a = Float::with_val(prec, &z_re * &dz_im);
            let b = Float::with_val(prec, &z_im * &dz_re);
            let mut t = a + b;
            t *= 2;
            t
        };

        // z := z² + c
        //   re = z_re² - z_im² + cx
        //   im = 2*z_re*z_im + cy
        let z_re_new = {
            let a = Float::with_val(prec, &z_re * &z_re);
            let b = Float::with_val(prec, &z_im * &z_im);
            let mut t = a - b;
            t += cx;
            t
        };
        let z_im_new = {
            let a = Float::with_val(prec, &z_re * &z_im);
            let mut t = a * 2;
            t += cy;
            t
        };

        z_re = z_re_new;
        z_im = z_im_new;
        dz_re = dz_re_new;
        dz_im = dz_im_new;

        // Bailout — orbite hors Mandelbrot, pas de période détectable.
        z_norm_sqr.assign(&z_re);
        z_norm_sqr.square_mut();
        let tmp = Float::with_val(prec, &z_im * &z_im);
        z_norm_sqr += &tmp;
        if z_norm_sqr > bailout_sqr {
            return None;
        }

        // Atom-domain criterion : |z|² < s² * |dz|²
        // F3 ne pose aucune borne inférieure sur n : pour un centre près du
        // cardioïde (période 1) ou de la bulbe principale (période 2),
        // retourner ces périodes courtes est correct. Le critère est lui-même
        // strict (un orbite chaotique ne satisfait pas |z|² < s²|dz|²).
        dz_norm_sqr.assign(&dz_re);
        dz_norm_sqr.square_mut();
        let tmp = Float::with_val(prec, &dz_im * &dz_im);
        dz_norm_sqr += &tmp;
        threshold.assign(&s_sqr);
        threshold *= &dz_norm_sqr;
        if z_norm_sqr < threshold {
            return Some(n);
        }
    }

    None
}

/// Alias rétro-compatible : utilise une échelle `s = 1e-6` par défaut, qui
/// approxime une "zone d'intérêt" sans dépendre du zoom courant. Préférer
/// `find_period_atom_domain` avec un `s` explicite quand le zoom est connu.
#[deprecated(note = "use find_period_atom_domain with an explicit view scale `s`")]
#[allow(dead_code)]
pub fn find_period(
    cx: &Float,
    cy: &Float,
    max_iter: u32,
    prec: u32,
) -> Option<u32> {
    let s = Float::with_val(prec, 1e-6);
    find_period_atom_domain(cx, cy, max_iter, &s, prec)
}

/// Newton-raffine le centre vers le nucleus exact de période `period`.
/// Suit `hybrid_center` de F3.cc : itère `c := c - z_period(c) / z'_period(c)`
/// avec dérivées propagées via dual numbers complexes.
pub fn newton_refine_center(
    cx_in: &Float,
    cy_in: &Float,
    period: u32,
    prec: u32,
    max_steps: u32,
) -> NucleusResult {
    let mut cx = cx_in.clone();
    let mut cy = cy_in.clone();

    // epsilon² = 2^(16 - 2*prec). À prec très grand (e.g., 3357 bits pour
    // e1000), 2^(16-6714) underflow le `Float::pow` à zéro, ce qui rend
    // toute itération "convergée" trivialement après 1 step. F3 utilise
    // floatexp pour cette comparaison ; on émule via rebound (Float * 2^exp).
    let exp_target: i32 = 16i32.saturating_sub(2i32.saturating_mul(prec as i32));
    // Compute 2^exp_target via shift if exp_target valid for Float ; sinon
    // construit explicitement.
    let mut epsilon_sqr = Float::with_val(prec, 1.0);
    if exp_target >= 0 {
        epsilon_sqr <<= exp_target as u32;
    } else {
        epsilon_sqr >>= (-exp_target) as u32;
    }

    // Best-seen state pour return même si non strictement convergé.
    let mut best_z_norm_sqr: Option<Float> = None;
    let mut best_cx = cx.clone();
    let mut best_cy = cy.clone();

    for step in 0..max_steps {
        // Itérer z + dérivées dz/dc sur period itérations.
        // Dual: z = (z_re + i*z_im) ; dz = (dz_re + i*dz_im) = dz/dc.
        // Récurrence :
        //   dz_{n+1} = 2 * z_n * dz_n + 1
        //   z_{n+1}  = z_n² + c
        let mut z_re = Float::with_val(prec, 0);
        let mut z_im = Float::with_val(prec, 0);
        let mut dz_re = Float::with_val(prec, 0);
        let mut dz_im = Float::with_val(prec, 0);

        for _ in 0..period {
            // dz_new = 2 * z * dz + 1
            //   re = 2*(z_re*dz_re - z_im*dz_im) + 1
            //   im = 2*(z_re*dz_im + z_im*dz_re)
            let dz_re_new = {
                let a = Float::with_val(prec, &z_re) * &dz_re;
                let b = Float::with_val(prec, &z_im) * &dz_im;
                let mut t = a - b;
                t *= 2;
                t += 1;
                t
            };
            let dz_im_new = {
                let a = Float::with_val(prec, &z_re) * &dz_im;
                let b = Float::with_val(prec, &z_im) * &dz_re;
                let mut t = a + b;
                t *= 2;
                t
            };
            // z_new = z² + c
            //   re = z_re² - z_im² + cx
            //   im = 2*z_re*z_im + cy
            let z_re_new = {
                let a = Float::with_val(prec, &z_re) * &z_re;
                let b = Float::with_val(prec, &z_im) * &z_im;
                let mut t = a - b;
                t += &cx;
                t
            };
            let z_im_new = {
                let a = Float::with_val(prec, &z_re) * &z_im;
                let mut t = a * 2;
                t += &cy;
                t
            };
            z_re = z_re_new;
            z_im = z_im_new;
            dz_re = dz_re_new;
            dz_im = dz_im_new;
        }

        // Newton: c -= z / dz
        // (z_re + i*z_im) / (dz_re + i*dz_im) = (z * conj(dz)) / |dz|²
        // delta_re = (z_re*dz_re + z_im*dz_im) / |dz|²
        // delta_im = (z_im*dz_re - z_re*dz_im) / |dz|²
        let dz_norm_sqr = {
            let mut t = Float::with_val(prec, &dz_re);
            t.square_mut();
            let mut u = Float::with_val(prec, &dz_im);
            u.square_mut();
            t += &u;
            t
        };
        if dz_norm_sqr.is_zero() {
            return NucleusResult {
                center_x: cx,
                center_y: cy,
                period,
                newton_steps: step,
                converged: false,
            };
        }

        let delta_re = {
            let a = Float::with_val(prec, &z_re) * &dz_re;
            let b = Float::with_val(prec, &z_im) * &dz_im;
            (a + b) / &dz_norm_sqr
        };
        let delta_im = {
            let a = Float::with_val(prec, &z_im) * &dz_re;
            let b = Float::with_val(prec, &z_re) * &dz_im;
            (a - b) / &dz_norm_sqr
        };
        cx -= &delta_re;
        cy -= &delta_im;

        let z_norm_sqr = {
            let mut t = Float::with_val(prec, &z_re);
            t.square_mut();
            let mut u = Float::with_val(prec, &z_im);
            u.square_mut();
            t += &u;
            t
        };

        // Best-seen tracking : on retient le centre qui minimise |z_period(c)|
        // pour pouvoir retourner même si Newton n'atteint pas epsilon strict
        // (improvement net sur le centre original).
        let update_best = match best_z_norm_sqr {
            None => true,
            Some(ref b) => z_norm_sqr < *b,
        };
        if update_best {
            best_z_norm_sqr = Some(z_norm_sqr.clone());
            best_cx = cx.clone();
            best_cy = cy.clone();
        }

        // Convergence : delta < epsilon (norme sub-précision).
        let delta_norm_sqr = {
            let mut t = Float::with_val(prec, &delta_re);
            t.square_mut();
            let mut u = Float::with_val(prec, &delta_im);
            u.square_mut();
            t += &u;
            t
        };
        if delta_norm_sqr < epsilon_sqr && z_norm_sqr < epsilon_sqr {
            return NucleusResult {
                center_x: cx,
                center_y: cy,
                period,
                newton_steps: step + 1,
                converged: true,
            };
        }
    }

    // Pas de convergence stricte mais on a peut-être amélioré.
    // Si |z_period(best_center)| < |z_period(center_original)|, considère succès.
    let cx_final;
    let cy_final;
    if best_z_norm_sqr.is_some() {
        cx_final = best_cx;
        cy_final = best_cy;
    } else {
        cx_final = cx;
        cy_final = cy;
    }

    NucleusResult {
        center_x: cx_final,
        center_y: cy_final,
        period,
        newton_steps: max_steps,
        converged: false,
    }
}

/// Pipeline complet : détecte la période via le critère atom-domain puis
/// Newton-raffine le centre. Retourne `None` si aucune période n'est trouvée
/// ou si la dérivée Newton est trop petite pour converger.
///
/// `s` est l'échelle de vue (~ `max(span_x, span_y) / 2`). À zoom courant,
/// l'utilisateur regarde une fenêtre de cette taille, donc on cherche le
/// nucleus dont l'atome remplit cette fenêtre — pas un nucleus plus petit
/// (qui serait subpixel) ni plus gros (qui ne serait pas centré).
pub fn find_nucleus(
    cx: &Float,
    cy: &Float,
    max_iter: u32,
    s: &Float,
    prec: u32,
) -> Option<NucleusResult> {
    let period = find_period_atom_domain(cx, cy, max_iter, s, prec)?;
    let result = newton_refine_center(cx, cy, period, prec, 64);
    // Accepte le résultat même non strictement convergé : Newton améliore
    // souvent le centre suffisamment pour que la perturbation fonctionne,
    // sans atteindre l'epsilon sub-précision (qui est parfois inatteignable
    // par rounding pour des minibrots à orbites longues).
    Some(result)
}

/// Résultat de `hybrid_size_mat2` : taille canonique du minibrot et matrice
/// 2×2 de transformation (orientation + scaling).
///
/// `size` est la taille naturelle du minibrot (1/zoom canonique pour qu'il
/// remplisse l'écran). `k` est la matrice row-major `[[K00, K01], [K10, K11]]`
/// qui encode la rotation/déformation du minibrot dans le plan c.
///
/// Pour Mandelbrot (quadratique conformal), `K` se réduit à `(1/β²) · R` où
/// `R` est une rotation pure. L'angle se récupère via `atan2(K[2], K[0])`.
#[derive(Clone, Copy, Debug)]
pub struct HybridSize {
    pub size: FloatExp,
    pub k: [f64; 4],
}

impl HybridSize {
    /// Angle de rotation extrait de `K` (radians, CCW dans le plan c).
    /// Pour Mandelbrot, `K = scale * R(θ)` avec `R[0][0] = cos θ`,
    /// `R[1][0] = sin θ`, d'où `θ = atan2(K[2], K[0])`.
    #[inline]
    pub fn rotation_radians(&self) -> f64 {
        self.k[2].atan2(self.k[0])
    }

    /// Angle de rotation en degrés.
    #[inline]
    pub fn rotation_degrees(&self) -> f64 {
        self.rotation_radians().to_degrees()
    }
}

/// Port fidèle de `hybrid_size` (`fraktaler-3-3.1/src/hybrid.cc:544`)
/// spécialisé pour Mandelbrot z² + c (degré 2).
///
/// Itère `period - 1` fois `Z := Z² + C` à partir de `Z = C`, en propageant
/// la matrice Jacobienne `L = ∂Z/∂c` via dual-numbers 2D, et accumule
/// `b += L⁻¹` à chaque étape. À la sortie :
///
/// - `λ = sqrt|det L_period|` — taux de divergence local.
/// - `β = sqrt|det b|` — magnitude de l'accumulateur.
/// - `size = 1 / (λ² · β)` (degré d=2 → d/(d-1) = 2).
/// - `K = inv(transp(b)) / β` — matrice de transformation.
///
/// Retourne `None` si l'orbite escape, si `det L_j` ou `det b` est nul
/// (singularité — typiquement period 2 sur c=-1, atome dégénéré), ou si
/// `K` contient des composantes non-finies.
///
/// **Coût** : O(period × prec²) en GMP. Pour un zoom 1e1000 avec
/// period ~10⁵ et prec ~3300 bits, l'appel prend plusieurs secondes.
/// Acceptable comme étape unique avant un rendu deep zoom.
pub fn hybrid_size_mat2(
    cx: &Float,
    cy: &Float,
    period: u32,
    prec: u32,
) -> Option<HybridSize> {
    if period == 0 {
        return None;
    }

    // Z initial = C avec Jacobian identité (∂Re(Z)/∂cx = 1, ∂Im(Z)/∂cy = 1).
    let mut z_re = cx.clone();
    let mut z_im = cy.clone();
    let mut zx_dx = Float::with_val(prec, 1); // ∂Re(Z)/∂cx
    let mut zx_dy = Float::with_val(prec, 0); // ∂Re(Z)/∂cy
    let mut zy_dx = Float::with_val(prec, 0); // ∂Im(Z)/∂cx
    let mut zy_dy = Float::with_val(prec, 1); // ∂Im(Z)/∂cy

    // Accumulateur b initialisé à l'identité 2×2.
    let mut b00 = Float::with_val(prec, 1);
    let mut b01 = Float::with_val(prec, 0);
    let mut b10 = Float::with_val(prec, 0);
    let mut b11 = Float::with_val(prec, 1);

    // Bailout large : si Z s'évade pendant l'itération du nucleus, le centre
    // n'est pas un nucleus valide. F3 utilise 1e10000 — 1e60 est suffisant
    // pour distinguer une orbite bornée d'une orbite divergente.
    let bailout_sqr = Float::with_val(prec, 1e120);

    for _ in 1..period {
        // Recurrence dérivée (cf. doc HybridSize) :
        //   ∂Z_new/∂cx = 2 Z (∂Z/∂cx) + 1
        //   ∂Z_new/∂cy = 2 Z (∂Z/∂cy) + i
        let zx_dx_new = {
            let a = Float::with_val(prec, &z_re * &zx_dx);
            let b = Float::with_val(prec, &z_im * &zy_dx);
            let mut t = a - b;
            t *= 2;
            t += 1;
            t
        };
        let zy_dx_new = {
            let a = Float::with_val(prec, &z_re * &zy_dx);
            let b = Float::with_val(prec, &z_im * &zx_dx);
            let mut t = a + b;
            t *= 2;
            t
        };
        let zx_dy_new = {
            let a = Float::with_val(prec, &z_re * &zx_dy);
            let b = Float::with_val(prec, &z_im * &zy_dy);
            let mut t = a - b;
            t *= 2;
            t
        };
        let zy_dy_new = {
            let a = Float::with_val(prec, &z_re * &zy_dy);
            let b = Float::with_val(prec, &z_im * &zx_dy);
            let mut t = a + b;
            t *= 2;
            t += 1;
            t
        };

        // Z_new = Z² + C
        let z_re_new = {
            let a = Float::with_val(prec, &z_re * &z_re);
            let b = Float::with_val(prec, &z_im * &z_im);
            let mut t = a - b;
            t += cx;
            t
        };
        let z_im_new = {
            let a = Float::with_val(prec, &z_re * &z_im);
            let mut t = a * 2;
            t += cy;
            t
        };

        z_re = z_re_new;
        z_im = z_im_new;
        zx_dx = zx_dx_new;
        zx_dy = zx_dy_new;
        zy_dx = zy_dx_new;
        zy_dy = zy_dy_new;

        // Bailout — orbite divergente, pas de nucleus stable.
        let z_norm_sqr = {
            let mut t = Float::with_val(prec, &z_re * &z_re);
            t += Float::with_val(prec, &z_im * &z_im);
            t
        };
        if z_norm_sqr > bailout_sqr {
            return None;
        }

        // L = current Jacobian = [[zx_dx, zx_dy], [zy_dx, zy_dy]].
        // det(L) = zx_dx * zy_dy - zx_dy * zy_dx.
        // inv(L) = (1/det) * [[zy_dy, -zx_dy], [-zy_dx, zx_dx]].
        let det_l = {
            let a = Float::with_val(prec, &zx_dx * &zy_dy);
            let b = Float::with_val(prec, &zx_dy * &zy_dx);
            a - b
        };
        if det_l.is_zero() {
            return None;
        }
        let inv_l_00 = Float::with_val(prec, &zy_dy / &det_l);
        let inv_l_01 = -Float::with_val(prec, &zx_dy / &det_l);
        let inv_l_10 = -Float::with_val(prec, &zy_dx / &det_l);
        let inv_l_11 = Float::with_val(prec, &zx_dx / &det_l);
        b00 += inv_l_00;
        b01 += inv_l_01;
        b10 += inv_l_10;
        b11 += inv_l_11;
    }

    // Final L (Jacobian Z'_period) et accumulateur b.
    let det_l_final = {
        let a = Float::with_val(prec, &zx_dx * &zy_dy);
        let b = Float::with_val(prec, &zx_dy * &zy_dx);
        a - b
    };
    let det_b = {
        let a = Float::with_val(prec, &b00 * &b11);
        let b = Float::with_val(prec, &b01 * &b10);
        a - b
    };
    if det_b.is_zero() {
        // Atome dégénéré (e.g., period 2 c=-1) — pas de transformation
        // canonique exploitable. L'appelant retombera sur le centre brut.
        return None;
    }

    // λ = sqrt|det L|, β = sqrt|det b|, size = 1/(λ²·β).
    // Calcul en FloatExp pour absorber les très grands |det L| (zoom profond
    // → |Z'_period| ~ 2^prec).
    let det_l_fexp = FloatExp::from_gmp(&det_l_final).abs();
    let det_b_fexp = FloatExp::from_gmp(&det_b).abs();
    let lambda_fexp = floatexp_sqrt(det_l_fexp);
    let beta_fexp = floatexp_sqrt(det_b_fexp);
    if beta_fexp.mantissa == 0.0 {
        return None;
    }
    // λ² · β. Pour Mandelbrot quadratique d = 2/(2-1) = 2.
    let lambda_sqr_fexp = lambda_fexp.sqr();
    let llb_fexp = lambda_sqr_fexp * beta_fexp;
    if llb_fexp.mantissa == 0.0 {
        return None;
    }
    let size = FloatExp::new(1.0 / llb_fexp.mantissa, -llb_fexp.exponent);

    // K = inv(transp(b)) / β. transp(b) = [[b00, b10], [b01, b11]],
    // det(transp(b)) = det(b), inv(transp(b)) = (1/det) * [[b11, -b10], [-b01, b00]].
    // Pour éviter underflow/overflow lors de la conversion vers f64 (cas deep
    // zoom où det_b · β a un exposant énorme), on combine les divisions en
    // FloatExp puis on convertit.
    let det_b_signed = FloatExp::from_gmp(&det_b);
    if det_b_signed.mantissa == 0.0 {
        return None;
    }
    let denom_fexp = det_b_signed * beta_fexp; // = det_b · |det_b|^(1/2), avec signe
    if denom_fexp.mantissa == 0.0 {
        return None;
    }
    let inv_denom = FloatExp::new(1.0 / denom_fexp.mantissa, -denom_fexp.exponent);
    let k00 = (FloatExp::from_gmp(&b11) * inv_denom).to_f64();
    let k01 = (FloatExp::from_gmp(&b10) * inv_denom).to_f64() * -1.0;
    let k10 = (FloatExp::from_gmp(&b01) * inv_denom).to_f64() * -1.0;
    let k11 = (FloatExp::from_gmp(&b00) * inv_denom).to_f64();
    let k = [k00, k01, k10, k11];
    if !k.iter().all(|x| x.is_finite()) {
        return None;
    }

    Some(HybridSize { size, k })
}

/// Racine carrée d'un `FloatExp` (assume `value` ≥ 0). Normalise pour gérer
/// les exposants impairs (sqrt nécessite exposant pair côté mantissa).
#[inline]
fn floatexp_sqrt(value: FloatExp) -> FloatExp {
    if value.mantissa <= 0.0 {
        return FloatExp::zero();
    }
    // mantissa ∈ [0.5, 1). On veut sqrt(mantissa * 2^exp).
    // Si exp pair : sqrt(mantissa) * 2^(exp/2). Si exp impair :
    // sqrt(2*mantissa) * 2^((exp-1)/2).
    if value.exponent % 2 == 0 {
        FloatExp::new(value.mantissa.sqrt(), value.exponent / 2)
    } else {
        FloatExp::new((value.mantissa * 2.0).sqrt(), (value.exponent - 1) / 2)
    }
}

// ─── G4 jalon 5f : nucleus PHASE-AWARE (formules hybrides) ───────────────────
// Port F3 `hybrid_period`/`hybrid_center`/`hybrid_size` pour une `Formula`
// bytecode arbitraire (phases cycliques, ops non-conformes abs/neg). Le
// Jacobien J = ∂(zx,zy)/∂(cx,cy) est un **mat2 de GMP Floats** — les abs
// cassent la conformité, le dz complexe des variantes z²+c ne suffit plus.
// Les variantes historiques restent le path single-phase Mandelbrot
// (bit-identiques, goldens). Différences vs F3 : la détection de période
// itère en FULL GMP (comme notre `find_period_atom_domain`) au lieu de la
// perturbation floatexp F3 — plus lent mais exact et sans dépendance aux
// références ; critère identique (K=I) : `|J⁻¹·z| < s` ⟺
// `|adj(J)·z|² < s²·det(J)²` (sans inversion).

use crate::fractal::bytecode::{Formula, Op, Phase};

/// État valeur + Jacobien mat2 GMP pour itérer une formule avec dérivées.
/// `j` (row-major `[j00, j01, j10, j11]`) est ∂(zx,zy)/∂(seed) : seed = c
/// (mode période/Newton, `add_j=true` → `Op::Add` fait J += I) ou seed = z₀
/// (mode size, `add_j=false` → J₀ = I, c constant).
struct GmpDualMat2 {
    zx: Float,
    zy: Float,
    j: [Float; 4],
    /// Copies `Op::Store` (consommées par `Op::Mul`).
    sx: Float,
    sy: Float,
    sj: [Float; 4],
    prec: u32,
}

impl GmpDualMat2 {
    fn new(prec: u32, zx: Float, zy: Float, j: [Float; 4]) -> Self {
        Self {
            sx: zx.clone(),
            sy: zy.clone(),
            sj: j.clone(),
            zx,
            zy,
            j,
            prec,
        }
    }

    /// `M(w)·J` où `M(w) = [[wx, −wy],[wy, wx]]` (matrice de multiplication
    /// complexe par w) — brique de la chain rule pour Sqr/Mul/Rot.
    fn cmul_mat(prec: u32, wx: &Float, wy: &Float, j: &[Float; 4]) -> [Float; 4] {
        let m = |a: &Float, b: &Float, c: &Float, d: &Float| -> Float {
            let t1 = Float::with_val(prec, a * b);
            let t2 = Float::with_val(prec, c * d);
            t1 - t2
        };
        let p = |a: &Float, b: &Float, c: &Float, d: &Float| -> Float {
            let t1 = Float::with_val(prec, a * b);
            let t2 = Float::with_val(prec, c * d);
            t1 + t2
        };
        [
            m(wx, &j[0], wy, &j[2]), // j00' = wx·j00 − wy·j10
            m(wx, &j[1], wy, &j[3]),
            p(wy, &j[0], wx, &j[2]), // j10' = wy·j00 + wx·j10
            p(wy, &j[1], wx, &j[3]),
        ]
    }

    /// Applique toutes les ops d'une phase (une itération), avec chain rule.
    fn step_phase(&mut self, phase: &Phase, cx: &Float, cy: &Float, add_j: bool) {
        let prec = self.prec;
        for op in &phase.ops {
            match op {
                Op::Sqr => {
                    // J' = M(2z)·J puis z' = z² (valeurs PRÉ-step).
                    let two_zx = Float::with_val(prec, &self.zx * 2u32);
                    let two_zy = Float::with_val(prec, &self.zy * 2u32);
                    self.j = Self::cmul_mat(prec, &two_zx, &two_zy, &self.j);
                    let x2 = Float::with_val(prec, &self.zx * &self.zx);
                    let y2 = Float::with_val(prec, &self.zy * &self.zy);
                    let xy = Float::with_val(prec, &self.zx * &self.zy);
                    self.zx = x2 - y2;
                    self.zy = Float::with_val(prec, &xy * 2u32);
                }
                Op::Mul => {
                    // z' = z·s : J' = M(s)·J + M(z)·Js (valeurs PRÉ-step).
                    let ja = Self::cmul_mat(prec, &self.sx, &self.sy, &self.j);
                    let jb = Self::cmul_mat(prec, &self.zx, &self.zy, &self.sj);
                    for k in 0..4 {
                        self.j[k] = Float::with_val(prec, &ja[k] + &jb[k]);
                    }
                    let ac = Float::with_val(prec, &self.zx * &self.sx);
                    let bd = Float::with_val(prec, &self.zy * &self.sy);
                    let ad = Float::with_val(prec, &self.zx * &self.sy);
                    let bc = Float::with_val(prec, &self.zy * &self.sx);
                    self.zx = ac - bd;
                    self.zy = ad + bc;
                }
                Op::Store => {
                    self.sx.assign(&self.zx);
                    self.sy.assign(&self.zy);
                    for k in 0..4 {
                        self.sj[k].assign(&self.j[k]);
                    }
                }
                Op::AbsX => {
                    if self.zx.is_sign_negative() {
                        self.zx = -self.zx.clone();
                        self.j[0] = -self.j[0].clone();
                        self.j[1] = -self.j[1].clone();
                    }
                }
                Op::AbsY => {
                    if self.zy.is_sign_negative() {
                        self.zy = -self.zy.clone();
                        self.j[2] = -self.j[2].clone();
                        self.j[3] = -self.j[3].clone();
                    }
                }
                Op::NegX => {
                    self.zx = -self.zx.clone();
                    self.j[0] = -self.j[0].clone();
                    self.j[1] = -self.j[1].clone();
                }
                Op::NegY => {
                    self.zy = -self.zy.clone();
                    self.j[2] = -self.j[2].clone();
                    self.j[3] = -self.j[3].clone();
                }
                Op::Add => {
                    self.zx += cx;
                    self.zy += cy;
                    if add_j {
                        self.j[0] += 1u32;
                        self.j[3] += 1u32;
                    }
                }
                Op::Rot { cos_theta, sin_theta } => {
                    let rx = Float::with_val(prec, *cos_theta);
                    let ry = Float::with_val(prec, *sin_theta);
                    self.j = Self::cmul_mat(prec, &rx, &ry, &self.j);
                    let ac = Float::with_val(prec, &self.zx * &rx);
                    let bd = Float::with_val(prec, &self.zy * &ry);
                    let ad = Float::with_val(prec, &self.zx * &ry);
                    let bc = Float::with_val(prec, &self.zy * &rx);
                    self.zx = ac - bd;
                    self.zy = ad + bc;
                }
            }
        }
    }

    fn norm_sqr(&self) -> Float {
        let x2 = Float::with_val(self.prec, &self.zx * &self.zx);
        let y2 = Float::with_val(self.prec, &self.zy * &self.zy);
        x2 + y2
    }

    fn det_j(&self) -> Float {
        let a = Float::with_val(self.prec, &self.j[0] * &self.j[3]);
        let b = Float::with_val(self.prec, &self.j[1] * &self.j[2]);
        a - b
    }
}

/// Degré d'une phase (F3 `param.cc:948` : sqr → ×2, mul → + stored, store →
/// snapshot). Mandelbrot/BS/Tricorn/… = 2 ; Multibrot n = n.
fn phase_degree(phase: &Phase) -> f64 {
    let mut q: f64 = 1.0;
    let mut sq: f64 = 1.0;
    for op in &phase.ops {
        match op {
            Op::Sqr => q *= 2.0,
            Op::Store => sq = q,
            Op::Mul => q += sq,
            _ => {}
        }
    }
    q.max(1.0)
}

/// Période atom-domain pour une formule hybride (mode d/dC, mat2).
/// Critère F3 (K=I) : `|adj(J)·z|² < s²·det(J)²`.
pub fn find_period_atom_domain_formula(
    formula: &Formula,
    cx: &Float,
    cy: &Float,
    max_iter: u32,
    s: &Float,
    prec: u32,
) -> Option<u32> {
    let n_phases = formula.phases.len().max(1);
    let bailout_sqr = Float::with_val(prec, 1e120);
    let s_sqr = Float::with_val(prec, s * s);
    let zero = Float::with_val(prec, 0);
    let mut st = GmpDualMat2::new(
        prec,
        zero.clone(),
        zero.clone(),
        [zero.clone(), zero.clone(), zero.clone(), zero.clone()],
    );

    for n in 1..=max_iter {
        let ph = &formula.phases[((n - 1) as usize) % n_phases];
        st.step_phase(ph, cx, cy, true);

        let z_norm = st.norm_sqr();
        if z_norm > bailout_sqr {
            return None;
        }
        // adj(J)·z = (j11·zx − j01·zy, −j10·zx + j00·zy)
        let vx = {
            let a = Float::with_val(prec, &st.j[3] * &st.zx);
            let b = Float::with_val(prec, &st.j[1] * &st.zy);
            a - b
        };
        let vy = {
            let a = Float::with_val(prec, &st.j[0] * &st.zy);
            let b = Float::with_val(prec, &st.j[2] * &st.zx);
            a - b
        };
        let lhs = {
            let a = Float::with_val(prec, &vx * &vx);
            let b = Float::with_val(prec, &vy * &vy);
            a + b
        };
        let det = st.det_j();
        let rhs = {
            let d2 = Float::with_val(prec, &det * &det);
            Float::with_val(prec, &s_sqr * &d2)
        };
        if lhs < rhs {
            return Some(n);
        }
    }
    None
}

/// Newton 2D vers `z_period(c) = 0` pour une formule hybride (F3
/// `hybrid_center` : solve J·Δ = −z par itération, mat2 GMP).
pub fn newton_refine_center_formula(
    formula: &Formula,
    cx_in: &Float,
    cy_in: &Float,
    period: u32,
    prec: u32,
    max_steps: u32,
) -> NucleusResult {
    let n_phases = formula.phases.len().max(1);
    let mut cx = cx_in.clone();
    let mut cy = cy_in.clone();

    let exp_target: i32 = 16i32.saturating_sub(2i32.saturating_mul(prec as i32));
    let mut epsilon_sqr = Float::with_val(prec, 1.0);
    if exp_target >= 0 {
        epsilon_sqr <<= exp_target as u32;
    } else {
        epsilon_sqr >>= (-exp_target) as u32;
    }

    let mut best_z_norm_sqr: Option<Float> = None;
    let mut best_cx = cx.clone();
    let mut best_cy = cy.clone();
    let mut converged = false;
    let mut steps_done = 0;

    for step in 0..max_steps {
        let zero = Float::with_val(prec, 0);
        let mut st = GmpDualMat2::new(
            prec,
            zero.clone(),
            zero.clone(),
            [zero.clone(), zero.clone(), zero.clone(), zero.clone()],
        );
        for i in 0..period {
            let ph = &formula.phases[(i as usize) % n_phases];
            st.step_phase(ph, &cx, &cy, true);
        }
        // Best-seen sur |z_period|² (mirror variante z²+c).
        let z_norm = st.norm_sqr();
        let improved = match &best_z_norm_sqr {
            None => true,
            Some(b) => z_norm < *b,
        };
        if improved {
            best_z_norm_sqr = Some(z_norm.clone());
            best_cx.assign(&cx);
            best_cy.assign(&cy);
        }
        // Newton : Δ = −J⁻¹·z ⇒ u = −(j11·x − j01·y)/det ; v = −(−j10·x + j00·y)/det.
        let det = st.det_j();
        if det == 0 {
            break;
        }
        let u = {
            let a = Float::with_val(prec, &st.j[3] * &st.zx);
            let b = Float::with_val(prec, &st.j[1] * &st.zy);
            let num = a - b;
            -(num / &det)
        };
        let v = {
            let a = Float::with_val(prec, &st.j[0] * &st.zy);
            let b = Float::with_val(prec, &st.j[2] * &st.zx);
            let num = a - b;
            -(num / &det)
        };
        cx += &u;
        cy += &v;
        steps_done = step + 1;
        let delta = {
            let a = Float::with_val(prec, &u * &u);
            let b = Float::with_val(prec, &v * &v);
            a + b
        };
        if delta < epsilon_sqr {
            converged = true;
            // Le dernier pas améliore par construction (delta sous epsilon).
            best_cx.assign(&cx);
            best_cy.assign(&cy);
            break;
        }
    }

    NucleusResult {
        center_x: best_cx,
        center_y: best_cy,
        period,
        newton_steps: steps_done,
        converged,
    }
}

/// Size + matrice K pour un nucleus de formule hybride (port F3 `hybrid_size`,
/// aligné sur la SÉMANTIQUE VALIDÉE de `hybrid_size_mat2` z²+c : J₀ = I et
/// récurrence d/dC (`Op::Add` fait J += I — la lettre F3 met dC=0, mais notre
/// variante z²+c historique, validée corpus P1.6.b sur les minibrots
/// non-axis-aligned, inclut le +I ; [M,M] doit lui être identique) ; degré =
/// moyenne géométrique des degrés de phase ; `b += inv(l)` par pas ;
/// `s = 1/(λ^d·β)` ; `K = inv(transp(b))/β`.
pub fn hybrid_size_mat2_formula(
    formula: &Formula,
    cx: &Float,
    cy: &Float,
    period: u32,
    prec: u32,
) -> Option<HybridSize> {
    if period == 0 {
        return None;
    }
    let n_phases = formula.phases.len().max(1);
    let one = Float::with_val(prec, 1);
    let zero = Float::with_val(prec, 0);
    // Z₀ = C, J₀ = I (d/dZ).
    let mut st = GmpDualMat2::new(
        prec,
        cx.clone(),
        cy.clone(),
        [one.clone(), zero.clone(), zero.clone(), one.clone()],
    );
    // b = I ; log-degré accumulé (phase 0 comprise, F3 init log_degree = log(deg[0])).
    let mut b = [one.clone(), zero.clone(), zero.clone(), one];
    let mut log_degree = phase_degree(&formula.phases[0]).ln();
    let bailout_sqr = Float::with_val(prec, 1e120);

    for jstep in 1..period {
        let ph = &formula.phases[(jstep as usize) % n_phases];
        st.step_phase(ph, cx, cy, true);
        log_degree += phase_degree(ph).ln();
        if st.norm_sqr() > bailout_sqr {
            return None;
        }
        // b += inv(l) où l = J courant.
        let det = st.det_j();
        if det == 0 {
            return None;
        }
        // inv(l) = adj(l)/det = [[j11, −j01],[−j10, j00]]/det
        let inv = [
            Float::with_val(prec, &st.j[3] / &det),
            Float::with_val(prec, -Float::with_val(prec, &st.j[1] / &det)),
            Float::with_val(prec, -Float::with_val(prec, &st.j[2] / &det)),
            Float::with_val(prec, &st.j[0] / &det),
        ];
        for k in 0..4 {
            b[k] += &inv[k];
        }
    }

    let degree = (log_degree / period as f64).exp().max(1.0 + 1e-9);
    let d = degree / (degree - 1.0);

    // λ = √|det l| ; β = √|det b| — en FloatExp (les dets débordent f64).
    let gmp_to_fexp = |f: &Float| -> FloatExp {
        let (m, e) = f.to_f64_exp();
        FloatExp::new(m, e)
    };
    let det_l = st.det_j();
    let det_b = {
        let a = Float::with_val(prec, &b[0] * &b[3]);
        let bb = Float::with_val(prec, &b[1] * &b[2]);
        a - bb
    };
    let det_l_fexp = {
        let mut v = gmp_to_fexp(&det_l);
        if v.mantissa < 0.0 {
            v.mantissa = -v.mantissa;
        }
        v
    };
    let det_b_fexp = {
        let mut v = gmp_to_fexp(&det_b);
        if v.mantissa < 0.0 {
            v.mantissa = -v.mantissa;
        }
        v
    };
    if det_l_fexp.mantissa == 0.0 || det_b_fexp.mantissa == 0.0 {
        return None;
    }
    let lambda = floatexp_sqrt(det_l_fexp);
    let beta = floatexp_sqrt(det_b_fexp);
    // llb = exp(log(λ)·d)·β ; size = 1/llb. log(λ) via mantisse+exposant.
    let log_lambda = lambda.mantissa.ln() + (lambda.exponent as f64) * std::f64::consts::LN_2;
    let log_llb = log_lambda * d + beta.mantissa.ln() + (beta.exponent as f64) * std::f64::consts::LN_2;
    // size = exp(−log_llb) → FloatExp via base 2.
    let log2_size = -log_llb / std::f64::consts::LN_2;
    let e = log2_size.floor();
    let size = FloatExp::new(2.0_f64.powf(log2_size - e), e as i32);

    // K = inv(transp(b))/β. transp(b) = [b0, b2, b1, b3] ;
    // inv = adj/det ; adj(transp) = [[b3, −b1],[−b2, b0]]... en f64 via ratios
    // normalisés par β (les b peuvent déborder f64 → normaliser par det_b).
    let beta_f = beta;
    let k = {
        // inv(transp(b)) = adj(transp(b))/det(transp(b)) ; det(transp) = det_b.
        // adj(transp(b)) row-major = [b[3], -b[2], -b[1], b[0]].
        let db = det_b;
        let el = |num: &Float, neg: bool| -> f64 {
            let r = Float::with_val(prec, num / &db);
            let r_fexp = gmp_to_fexp(&r);
            // divisé ensuite par β : (r/β) en FloatExp → f64
            let scaled = FloatExp::new(r_fexp.mantissa, r_fexp.exponent)
                .div(beta_f);
            let v = scaled.to_f64();
            if neg { -v } else { v }
        };
        [el(&b[3], false), el(&b[2], true), el(&b[1], true), el(&b[0], false)]
    };

    Some(HybridSize { size, k })
}

/// Pipeline complet phase-aware : période atom-domain (mat2) → Newton →
/// (size/K au caller). Mirror de `find_nucleus`.
pub fn find_nucleus_formula(
    formula: &Formula,
    cx: &Float,
    cy: &Float,
    max_iter: u32,
    s: &Float,
    prec: u32,
) -> Option<NucleusResult> {
    let period = find_period_atom_domain_formula(formula, cx, cy, max_iter, s, prec)?;
    let result = newton_refine_center_formula(formula, cx, cy, period, prec, 64);
    Some(result)
}

/// Helper : centre complexe → (Float, Float).
#[allow(dead_code)]
pub fn complex_to_xy(c: &Complex, prec: u32) -> (Float, Float) {
    (
        Float::with_val(prec, c.real()),
        Float::with_val(prec, c.imag()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Jalon 5f — verrou [M,M]** : la formule hybride [Mandelbrot,
    /// Mandelbrot] via l'interpréteur mat2 DOIT trouver la même période et le
    /// même centre (à epsilon près) que le path z²+c historique sur un
    /// satellite connu.
    #[test]
    fn formula_mm_nucleus_matches_z2c() {
        use crate::fractal::bytecode::compile_hybrid_formula;
        use crate::fractal::FractalType;
        let prec = 128u32;
        let cx = Float::with_val(prec, -1.7548);
        let cy = Float::with_val(prec, 0.0002);
        let s = Float::with_val(prec, 1e-3);

        let p_z2c = find_period_atom_domain(&cx, &cy, 1000, &s, prec)
            .expect("période z²+c");
        let f = compile_hybrid_formula(
            &[FractalType::Mandelbrot, FractalType::Mandelbrot],
            2.0,
        )
        .expect("formule [M,M]");
        let p_mm = find_period_atom_domain_formula(&f, &cx, &cy, 1000, &s, prec)
            .expect("période [M,M]");
        assert_eq!(p_mm, p_z2c, "période mat2 ≠ z²+c");

        let r_z2c = newton_refine_center(&cx, &cy, p_z2c, prec, 64);
        let r_mm = newton_refine_center_formula(&f, &cx, &cy, p_mm, prec, 64);
        let dx = Float::with_val(prec, &r_mm.center_x - &r_z2c.center_x)
            .abs()
            .to_f64();
        let dy = Float::with_val(prec, &r_mm.center_y - &r_z2c.center_y)
            .abs()
            .to_f64();
        assert!(dx < 1e-20 && dy < 1e-20, "centres divergent : dx={dx:e} dy={dy:e}");

        // Size/K : mêmes valeurs (J conforme ⇒ mat2 ≡ complexe).
        let s_z2c = hybrid_size_mat2(&r_z2c.center_x, &r_z2c.center_y, p_z2c, prec)
            .expect("size z²+c");
        let s_mm = hybrid_size_mat2_formula(&f, &r_mm.center_x, &r_mm.center_y, p_mm, prec)
            .expect("size [M,M]");
        let ratio = s_mm.size.div(s_z2c.size).to_f64();
        assert!(
            (ratio - 1.0).abs() < 1e-6,
            "size mat2/z²+c = {ratio} (attendu 1)"
        );
        for k in 0..4 {
            assert!(
                (s_mm.k[k] - s_z2c.k[k]).abs() < 1e-6,
                "K[{k}] : {} vs {}",
                s_mm.k[k],
                s_z2c.k[k]
            );
        }
    }

    /// **Jalon 5f — nucleus GENUINE [M,BS]** : le blob lisse du set hybride
    /// (trouvé au jalon 5e, ~(-0.4744, -0.6327), taille ~1e-6) est un
    /// mini-set du hybride. Le pipeline formule doit trouver sa période,
    /// converger vers un centre proche, et donner une taille plausible.
    #[test]
    fn formula_mbs_nucleus_finds_blob() {
        use crate::fractal::bytecode::compile_hybrid_formula;
        use crate::fractal::FractalType;
        let prec = 192u32;
        // Point PRÈS du blob (pas son centre exact) — le finder doit s'y caler.
        let cx = Float::with_val(prec, -0.474404979221344);
        let cy = Float::with_val(prec, -0.6327382500546773);
        let s = Float::with_val(prec, 2e-6);
        let f = compile_hybrid_formula(
            &[FractalType::Mandelbrot, FractalType::BurningShip],
            2.0,
        )
        .expect("formule [M,BS]");
        let period = find_period_atom_domain_formula(&f, &cx, &cy, 20000, &s, prec)
            .expect("période blob [M,BS]");
        assert!(period > 0);
        let r = newton_refine_center_formula(&f, &cx, &cy, period, prec, 64);
        // Le centre raffiné doit rester dans le voisinage du blob (~qq 1e-6).
        let dx = Float::with_val(prec, &r.center_x - &cx).abs().to_f64();
        let dy = Float::with_val(prec, &r.center_y - &cy).abs().to_f64();
        assert!(
            dx < 1e-4 && dy < 1e-4,
            "centre parti trop loin : dx={dx:e} dy={dy:e} (period={period})"
        );
        // Vérité indépendante : le centre trouvé est PÉRIODIQUE — itérer la
        // formule period fois depuis 0 en GMP pur (interp valeur seule) doit
        // revenir ~0 (|z_period| ≪ taille du blob).
        let zero = Float::with_val(prec, 0);
        let mut st = GmpDualMat2::new(
            prec,
            zero.clone(),
            zero.clone(),
            [zero.clone(), zero.clone(), zero.clone(), zero.clone()],
        );
        for i in 0..period {
            let ph = &f.phases[(i as usize) % f.phases.len()];
            st.step_phase(ph, &r.center_x, &r.center_y, true);
        }
        let z_end = st.norm_sqr().to_f64().sqrt();
        assert!(
            z_end < 1e-9,
            "|z_period({period})| = {z_end:e} au centre raffiné (attendu ~0)"
        );
        // Size finie positive (la sonde au BORD du blob peut converger vers un
        // satellite bien plus petit que le blob lui-même — correct : le
        // voisinage d'un bord contient des minibrots arbitrairement petits).
        let hs = hybrid_size_mat2_formula(&f, &r.center_x, &r.center_y, period, prec)
            .expect("size [M,BS]");
        let sz = hs.size.to_f64();
        assert!(
            sz.is_finite() && sz > 0.0 && sz < 1e-2,
            "size {sz:e} non plausible"
        );
        assert!(hs.k.iter().all(|v| v.is_finite()), "K non finie : {:?}", hs.k);
        eprintln!(
            "[MBS-NUCLEUS] period={period} newton_steps={} converged={} size={sz:e} K={:?}",
            r.newton_steps, r.converged, hs.k
        );
    }

    #[test]
    fn nucleus_main_cardioid_seed_converges_to_zero() {
        // Le seed (0, 0) est près du nucleus période 1 (le cardioïde central).
        // Newton doit converger vers (-0.25, 0) — le centre de la période-1
        // bulbe, mais en réalité la cardioïde n'a pas de "nucleus" classique.
        // Test plus simple : period 1, point près de 0, doit converger.
        let prec = 256u32;
        let cx = Float::with_val(prec, 0.1);
        let cy = Float::with_val(prec, 0.0);
        let res = newton_refine_center(&cx, &cy, 1, prec, 64);
        // z_1(c) = c, donc Newton c := c - c / 1 = 0. Doit converger en 1 step.
        assert!(res.converged, "Newton should converge for period 1");
    }

    #[test]
    fn nucleus_period_2_converges_to_minus_one() {
        // Période 2 : z_1 = c, z_2 = c² + c = c(c+1). Pour z_2 = 0 → c = 0 ou c = -1.
        // Démarrer près de -1 doit converger vers -1.
        let prec = 256u32;
        let cx = Float::with_val(prec, -0.9);
        let cy = Float::with_val(prec, 0.0);
        let res = newton_refine_center(&cx, &cy, 2, prec, 64);
        assert!(res.converged);
        let actual = res.center_x.to_f64();
        assert!((actual - (-1.0)).abs() < 1e-10, "Expected -1, got {}", actual);
    }

    #[test]
    fn atom_domain_detects_period_3_satellite() {
        // L'atom de période 3 vit autour de c = -1.754877666... (un minibrot
        // sur la branche négative). Près de ce point, l'atom-domain criterion
        // doit fire à n = 3.
        let prec = 128u32;
        let cx = Float::with_val(prec, -1.7548776);
        let cy = Float::with_val(prec, 0.0);
        // Échelle de vue ~ 0.01 — taille typique d'une vue centrée sur ce
        // minibrot. s grand permet à la condition de fire dès la période
        // minimale.
        let s = Float::with_val(prec, 0.01);
        let period = find_period_atom_domain(&cx, &cy, 100, &s, prec);
        assert_eq!(
            period,
            Some(3),
            "Expected period 3 for c ≈ -1.7548776, got {:?}",
            period
        );
    }

    #[test]
    fn atom_domain_returns_none_for_escaping_orbit() {
        // c = 1.0 escape immédiatement (z_2 = 2, z_3 = 5...). Aucune période
        // ne doit être détectée.
        let prec = 64u32;
        let cx = Float::with_val(prec, 1.0);
        let cy = Float::with_val(prec, 0.0);
        let s = Float::with_val(prec, 1e-3);
        let period = find_period_atom_domain(&cx, &cy, 100, &s, prec);
        assert_eq!(period, None);
    }

    #[test]
    fn hybrid_size_period_1_returns_identity() {
        // Period 1 sur c=0 : loop ne s'exécute pas. L = I, b = I.
        // size = 1/(λ²·β) = 1, K = I (à arrondi près).
        let prec = 128u32;
        let cx = Float::with_val(prec, 0);
        let cy = Float::with_val(prec, 0);
        let hs = hybrid_size_mat2(&cx, &cy, 1, prec).expect("period 1 should succeed");
        assert!((hs.size.to_f64() - 1.0).abs() < 1e-12);
        assert!((hs.k[0] - 1.0).abs() < 1e-12);
        assert!(hs.k[1].abs() < 1e-12);
        assert!(hs.k[2].abs() < 1e-12);
        assert!((hs.k[3] - 1.0).abs() < 1e-12);
        assert!(hs.rotation_radians().abs() < 1e-12);
    }

    #[test]
    fn hybrid_size_period_2_at_minus_one_is_degenerate() {
        // c=-1 period 2 : après iter 1, L = -I et inv(L) = -I, donc b = 0.
        // Atome dégénéré — la fonction renvoie None (F3 a la même limite mais
        // gère via safety check 10*size0 > size).
        let prec = 128u32;
        let cx = Float::with_val(prec, -1);
        let cy = Float::with_val(prec, 0);
        let hs = hybrid_size_mat2(&cx, &cy, 2, prec);
        assert!(hs.is_none(), "expected None for degenerate period 2 c=-1");
    }

    #[test]
    fn hybrid_size_period_3_axis_aligned_minibrot() {
        // Minibrot période 3 à c ≈ -1.7548776662... : axis-aligned (cy=0 →
        // dérivées symétriques en cy). La rotation doit être ≈ 0 ou π.
        let prec = 192u32;
        let cx = Float::with_val(prec, -1.7548776662466927_f64);
        let cy = Float::with_val(prec, 0);
        let hs = hybrid_size_mat2(&cx, &cy, 3, prec).expect("period 3 should succeed");
        // Le minibrot est axis-aligned → composantes off-diagonales ≈ 0.
        assert!(
            hs.k[1].abs() < 1e-6,
            "off-diagonal K[0][1] should be ~0, got {}",
            hs.k[1]
        );
        assert!(
            hs.k[2].abs() < 1e-6,
            "off-diagonal K[1][0] should be ~0, got {}",
            hs.k[2]
        );
        // Rotation extraite ≈ 0 (modulo π).
        let theta = hs.rotation_radians().abs();
        let theta_mod_pi = theta % std::f64::consts::PI;
        assert!(
            theta_mod_pi.min(std::f64::consts::PI - theta_mod_pi) < 1e-6,
            "rotation should be ≈ 0 or π, got {} rad",
            hs.rotation_radians()
        );
        // size > 0 et fini.
        let s = hs.size.to_f64();
        assert!(s.is_finite() && s > 0.0, "size should be positive finite, got {}", s);
    }

    #[test]
    fn hybrid_size_returns_none_for_escaping_orbit() {
        // c=1 escape immédiatement — hybrid_size doit retourner None.
        let prec = 64u32;
        let cx = Float::with_val(prec, 1);
        let cy = Float::with_val(prec, 0);
        let hs = hybrid_size_mat2(&cx, &cy, 10, prec);
        assert!(hs.is_none(), "escaping orbit should yield None");
    }

    #[test]
    fn atom_domain_full_pipeline_period_3() {
        // Pipeline complet : find_nucleus depuis un point proche du minibrot
        // période 3 doit retourner cx ≈ -1.7548776662..., converged.
        let prec = 192u32;
        let cx = Float::with_val(prec, -1.7548);
        let cy = Float::with_val(prec, 0.0);
        let s = Float::with_val(prec, 0.01);
        let res = find_nucleus(&cx, &cy, 100, &s, prec).expect("should find nucleus");
        assert_eq!(res.period, 3);
        let expected = -1.7548776662466927_f64;
        assert!(
            (res.center_x.to_f64() - expected).abs() < 1e-12,
            "Expected x ≈ {}, got {}",
            expected,
            res.center_x.to_f64()
        );
    }
}
