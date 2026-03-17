struct Params {
    center_x: f32,
    center_y: f32,
    span_x: f32,
    span_y: f32,
    cref_x: f32,
    cref_y: f32,
    width: u32,
    height: u32,
    iter_max: u32,
    bailout: f32,
    bla_levels: u32,
    fractal_kind: u32,
    glitch_tolerance: f32,
    series_order: u32,
    series_threshold: f32,
    _pad0: u32,
};

struct ZRef {
    re: f32,
    im: f32,
};

struct PixelOut {
    iter: u32,
    z_re: f32,
    z_im: f32,
    flags: u32,
};

struct BlaNode {
    a_re: f32,
    a_im: f32,
    b_re: f32,
    b_im: f32,
    c_re: f32,
    c_im: f32,
    validity: f32,
    _pad: f32,
};

const MAX_LEVELS: u32 = 17u;
// Extended rescaling thresholds for better precision at deep zooms
// FloatExp-style: keep mantissa in reasonable range while tracking scale separately
const RESCALE_HI: f32 = 1.0e6;
const RESCALE_LO: f32 = 1.0e-6;
// ============================================================================

// Adaptive glitch tolerance is pre-computed on CPU and passed via params.glitch_tolerance

struct BlaMeta {
    level_offsets: array<u32, MAX_LEVELS>,
    level_lengths: array<u32, MAX_LEVELS>,
    _pad: vec2<u32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out_pixels: array<PixelOut>;
@group(0) @binding(2) var<storage, read> bla_meta: BlaMeta;
@group(0) @binding(3) var<storage, read> bla_nodes: array<BlaNode>;
@group(0) @binding(4) var<storage, read> z_ref: array<ZRef>;
@group(0) @binding(5) var<storage, read> reuse_mask: array<u32>;

// ============================================================================
// Cache BLA en mémoire partagée (workgroup memory)
// Réduit les accès à la mémoire globale pour les nœuds BLA fréquemment utilisés
// ============================================================================

const BLA_CACHE_SIZE: u32 = 256u;
var<workgroup> shared_bla_cache: array<BlaNode, 256>;
var<workgroup> bla_cache_base: u32;
var<workgroup> bla_cache_level: u32;
var<workgroup> bla_cache_valid: bool;

// Charge les nœuds BLA dans le cache partagé
// Appelé par chaque thread du workgroup, avec synchronisation via workgroupBarrier
fn load_bla_cache(level: u32, base_n: u32, local_idx: u32) {
    // Seuls les premiers BLA_CACHE_SIZE threads chargent les données
    if (local_idx < BLA_CACHE_SIZE) {
        let offset = bla_meta.level_offsets[level];
        let len = bla_meta.level_lengths[level];
        let load_idx = base_n + local_idx;
        if (load_idx < len) {
            shared_bla_cache[local_idx] = bla_nodes[offset + load_idx];
        }
    }
}

// Récupère un nœud BLA depuis le cache ou la mémoire globale
fn get_bla_node(level: u32, n: u32) -> BlaNode {
    // Vérifier si le nœud est dans le cache
    if (bla_cache_valid && level == bla_cache_level) {
        let cache_idx = n - bla_cache_base;
        if (cache_idx < BLA_CACHE_SIZE) {
            return shared_bla_cache[cache_idx];
        }
    }
    // Fallback: accès direct à la mémoire globale
    let offset = bla_meta.level_offsets[level];
    return bla_nodes[offset + n];
}

// ============================================================================

fn complex_mul(a_re: f32, a_im: f32, b_re: f32, b_im: f32) -> vec2<f32> {
    return vec2<f32>(a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re);
}

// Stable computation of |c + d| - |c| avoiding catastrophic cancellation.
// Inspired by rust-fractal-core's diff_abs() function.
// Used for Burning Ship perturbation where we need the variation of |x|.
fn diffabs(c: f32, d: f32) -> f32 {
    let cd = c + d;
    let c2d = 2.0 * c + d;
    if (c >= 0.0) {
        if (cd >= 0.0) {
            return d;
        } else {
            return -c2d;
        }
    } else {
        if (cd > 0.0) {
            return c2d;
        } else {
            return -d;
        }
    }
}

fn rescale_delta(re: f32, im: f32, scale: f32) -> vec3<f32> {
    let abs_max = max(abs(re), abs(im));
    // Fast path: no rescaling needed
    if (abs_max <= RESCALE_HI && abs_max >= RESCALE_LO) {
        return vec3<f32>(re, im, scale);
    }
    if (abs_max == 0.0) {
        return vec3<f32>(re, im, scale);
    }

    var new_re = re;
    var new_im = im;
    var new_scale = scale;
    let max_scale = 1.0e30;
    let min_scale = 1.0e-30;

    if (abs_max > RESCALE_HI && new_scale < max_scale) {
        let k = floor(log2(abs_max / RESCALE_HI));
        let factor = exp2(min(k, 20.0));
        new_re = new_re / factor;
        new_im = new_im / factor;
        new_scale = min(new_scale * factor, max_scale);
    } else if (abs_max < RESCALE_LO && new_scale > min_scale) {
        let k = floor(log2(RESCALE_LO / abs_max));
        let factor = exp2(min(k, 20.0));
        new_re = new_re * factor;
        new_im = new_im * factor;
        new_scale = max(new_scale / factor, min_scale);
    }
    return vec3<f32>(new_re, new_im, new_scale);
}

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32
) {
    // Initialiser le cache BLA (niveau 0 par défaut)
    // Tous les threads participent au chargement initial
    if (params.bla_levels > 0u) {
        if (local_idx == 0u) {
            bla_cache_base = 0u;
            bla_cache_level = 0u;
            bla_cache_valid = true;
        }
        workgroupBarrier();
        load_bla_cache(0u, 0u, local_idx);
        workgroupBarrier();
    } else {
        if (local_idx == 0u) {
            bla_cache_valid = false;
        }
        workgroupBarrier();
    }

    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let idx = gid.y * params.width + gid.x;
    // IMPORTANT: Si reuse_mask est rempli de 1 partout (reuse désactivé), cette vérification
    // est toujours vraie et ne skip rien. Mais si le buffer n'est pas correctement initialisé,
    // tous les pixels seraient skippés. Pour debug, on peut commenter cette ligne temporairement.
    // En production, on garde cette vérification pour supporter le reuse quand activé.
    if (reuse_mask[idx] == 0u) {
        return;
    }
    
    // Calcul de la position du pixel en f32 standard
    // IMPORTANT: Utiliser la même formule que le code CPU pour éviter les erreurs de précision:
    // dc = (pixel_index/dimension - 0.5) * range
    // Cette formule divise d'abord, puis multiplie, ce qui évite la perte de précision
    // lors de la multiplication de grands nombres avec de très petits nombres aux zooms profonds
    let pixel_size = params.span_x / f32(params.width);
    let inv_width = 1.0 / f32(params.width);
    let inv_height = 1.0 / f32(params.height);
    let x_ratio = (f32(gid.x) + 0.5) * inv_width - 0.5;
    let y_ratio = (f32(gid.y) + 0.5) * inv_height - 0.5;
    let dc_re = x_ratio * params.span_x;
    let dc_im = y_ratio * params.span_y;

    let is_julia = params.fractal_kind == 1u;
    let is_burning_ship = params.fractal_kind == 2u;
    let is_tricorn = params.fractal_kind == 3u;
    let is_tricorn_julia = params.fractal_kind == 4u;
    var delta_re = select(0.0, dc_re, is_julia);
    var delta_im = select(0.0, dc_im, is_julia);
    var delta_scale = 1.0;
    var n: u32 = 0u;
    let bailout_sqr = params.bailout * params.bailout;
    
    // Adaptive glitch tolerance is pre-computed on CPU
    let glitch_tolerance_sqr = params.glitch_tolerance * params.glitch_tolerance;

    loop {
        if (n >= params.iter_max) {
            break;
        }

        var stepped = false;
        if (!is_burning_ship && !is_tricorn && !is_tricorn_julia && params.bla_levels > 0u) {
            var level: i32 = i32(params.bla_levels);
            loop {
                level = level - 1;
                if (level < 0) {
                    break;
                }
                let lvl = u32(level);
                let len = bla_meta.level_lengths[lvl];
                if (n >= len) {
                    continue;
                }
                // Utiliser le cache BLA en mémoire partagée si disponible
                let node = get_bla_node(lvl, n);
                let delta_actual_re = delta_re * delta_scale;
                let delta_actual_im = delta_im * delta_scale;
                let delta_norm_sqr = delta_actual_re * delta_actual_re + delta_actual_im * delta_actual_im;
                if (delta_norm_sqr < node.validity * node.validity) {
                    let mul1 = complex_mul(node.a_re, node.a_im, delta_actual_re, delta_actual_im);
                    let mul2 = complex_mul(node.b_re, node.b_im, dc_re, dc_im);
                    if (params.series_order >= 2u && delta_norm_sqr < params.series_threshold * params.series_threshold) {
                        let delta_sq = complex_mul(delta_actual_re, delta_actual_im, delta_actual_re, delta_actual_im);
                        let mul3 = complex_mul(node.c_re, node.c_im, delta_sq.x, delta_sq.y);
                        // BLA: mul2 contient le terme dc. Pour Mandelbrot, on doit l'ajouter. Pour Julia, non.
                        // WGSL: select(false_val, true_val, cond) => cond ? true_val : false_val
                        let next_re = mul1.x + select(0.0, mul2.x, !is_julia) + mul3.x;
                        let next_im = mul1.y + select(0.0, mul2.y, !is_julia) + mul3.y;
                        let scaled = rescale_delta(next_re, next_im, 1.0);
                        delta_re = scaled.x;
                        delta_im = scaled.y;
                        delta_scale = scaled.z;
                    } else {
                        // BLA: mul2 contient le terme dc. Pour Mandelbrot, on doit l'ajouter. Pour Julia, non.
                        // WGSL: select(false_val, true_val, cond) => cond ? true_val : false_val
                        let next_re = mul1.x + select(0.0, mul2.x, !is_julia);
                        let next_im = mul1.y + select(0.0, mul2.y, !is_julia);
                        let scaled = rescale_delta(next_re, next_im, 1.0);
                        delta_re = scaled.x;
                        delta_im = scaled.y;
                        delta_scale = scaled.z;
                    }
                    n = n + (1u << lvl);
                    stepped = true;
                    break;
                }
            }
        }

        if (!stepped) {
            let z = z_ref[n];
            if (is_tricorn || is_tricorn_julia) {
                // Tricorn perturbation: z' = conj(z)² + c
                // Inspired by rust-fractal-core's non-conformal perturbation approach.
                // Since Tricorn uses conjugation (anti-conformal), we compute z_curr = z_ref + delta,
                // apply conj(z_curr)² + c, then subtract the next z_ref to get the new delta.
                let delta_actual_re = delta_re * delta_scale;
                let delta_actual_im = delta_im * delta_scale;
                let z_re = z.re + delta_actual_re;
                let z_im = z.im + delta_actual_im;
                // conj(z) = (z_re, -z_im), then square: (z_re² - z_im², -2*z_re*z_im)
                let z_sq = complex_mul(z_re, -z_im, z_re, -z_im);
                var z_next_re: f32;
                var z_next_im: f32;
                if (is_tricorn_julia) {
                    z_next_re = z_sq.x + params.cref_x;
                    z_next_im = z_sq.y + params.cref_y;
                } else {
                    z_next_re = z_sq.x + (params.cref_x + dc_re);
                    z_next_im = z_sq.y + (params.cref_y + dc_im);
                }
                n = n + 1u;
                if (n >= params.iter_max) {
                    break;
                }
                let z_next_ref = z_ref[n];
                let next_re = z_next_re - z_next_ref.re;
                let next_im = z_next_im - z_next_ref.im;
                let scaled = rescale_delta(next_re, next_im, 1.0);
                delta_re = scaled.x;
                delta_im = scaled.y;
                delta_scale = scaled.z;
            } else if (is_burning_ship) {
                // Burning Ship perturbation using diffabs (inspired by rust-fractal-core).
                // Uses the scaled delta approach for better precision, matching the CPU path.
                // Key insight: diffabs(c, d) = |c + d| - |c| computed stably.
                //
                // Formula from rust-fractal-core:
                //   delta_re' = (2*Z_re + d_re*sf) * d_re - (2*Z_im + d_im*sf) * d_im + dc_re
                //   delta_im' = 2 * diffabs(Z_re*Z_im/sf, Z_re*d_im + d_re*(Z_im + d_im*sf)) + dc_im
                let sf = delta_scale;
                let inv_sf = 1.0 / max(sf, 1.0e-38);
                let d_re = delta_re;
                let d_im = delta_im;
                let temp_re = d_re;
                let new_re = (2.0 * z.re * inv_sf + temp_re * sf * inv_sf) * temp_re
                           - (2.0 * z.im * inv_sf + d_im * sf * inv_sf) * d_im
                           + dc_re * inv_sf;
                let new_im = 2.0 * diffabs(
                    z.re * z.im * inv_sf,
                    z.re * d_im + temp_re * (z.im * inv_sf + d_im * sf * inv_sf)
                ) + dc_im * inv_sf;
                let scaled = rescale_delta(new_re, new_im, sf);
                delta_re = scaled.x;
                delta_im = scaled.y;
                delta_scale = scaled.z;
                n = n + 1u;
            } else {
                // Mandelbrot/Julia perturbation using scaled delta approach
                // (inspired by rust-fractal-core's perturb_function).
                // Keeps delta as mantissa * scale_factor, doing arithmetic in scaled space
                // for better precision. Formula:
                //   delta' = delta * (2*z_ref + delta*sf) + dc
                // In scaled space (dividing by sf):
                //   d' = (2*z_ref/sf)*d + sf*(d*d) + dc/sf
                let sf = delta_scale;
                let inv_sf = 1.0 / max(sf, 1.0e-38);
                let d_re = delta_re;
                let d_im = delta_im;
                let two_zr_re = 2.0 * z.re * inv_sf;
                let two_zr_im = 2.0 * z.im * inv_sf;
                let new_re = two_zr_re * d_re - two_zr_im * d_im + sf * (d_re * d_re - d_im * d_im)
                           + select(0.0, dc_re * inv_sf, !is_julia);
                let new_im = two_zr_re * d_im + two_zr_im * d_re + sf * (2.0 * d_re * d_im)
                           + select(0.0, dc_im * inv_sf, !is_julia);
                let scaled = rescale_delta(new_re, new_im, sf);
                delta_re = scaled.x;
                delta_im = scaled.y;
                delta_scale = scaled.z;
                n = n + 1u;
            }
        }

        // ====================================================================================
        // REBASING + BAILOUT + GLITCH DETECTION (merged to compute delta_actual once)
        // ====================================================================================
        // Compute delta_actual once and reuse for rebase check, bailout, and glitch detection.
        // small_delta flag skips rebase and glitch checks when delta is tiny (no precision loss).
        if (n > 0u && n < params.iter_max) {
            let z_check = z_ref[n];
            let delta_actual_re = delta_re * delta_scale;
            let delta_actual_im = delta_im * delta_scale;
            let z_curr_re = z_check.re + delta_actual_re;
            let z_curr_im = z_check.im + delta_actual_im;
            let z_curr_norm_sqr = z_curr_re * z_curr_re + z_curr_im * z_curr_im;
            let delta_norm_sqr = delta_actual_re * delta_actual_re + delta_actual_im * delta_actual_im;
            let small_delta = (delta_scale < 1.0e-15);

            // Rebasing with hysteresis (inspired by rust-fractal-core):
            // Only rebase if z_curr is meaningfully smaller than delta (factor 0.5),
            // preventing ping-pong rebasing that wastes iterations.
            // Skip rebasing when delta is tiny (no precision loss) or z_ref is very small.
            if (!small_delta) {
                let z_ref_ns = z_check.re * z_check.re + z_check.im * z_check.im;
                if (z_curr_norm_sqr > 0.0 && delta_norm_sqr > 0.0 && z_curr_norm_sqr < delta_norm_sqr * 0.5 && z_ref_ns > 1.0e-20) {
                    let scaled = rescale_delta(z_curr_re, z_curr_im, 1.0);
                    delta_re = scaled.x;
                    delta_im = scaled.y;
                    delta_scale = scaled.z;
                    n = 0u;
                    continue;
                }
            }

            // Glitch detection: NaN/Inf always invalid, delta_too_large only when delta is not tiny
            let nan_re = z_curr_re != z_curr_re;
            let nan_im = z_curr_im != z_curr_im;
            let inf_re = abs(z_curr_re) > 1e30;
            let inf_im = abs(z_curr_im) > 1e30;
            let z_ref_norm_sqr = z_check.re * z_check.re + z_check.im * z_check.im;
            let glitch_scale = max(z_ref_norm_sqr, 1.0e-6);
            let delta_too_large = delta_norm_sqr > glitch_tolerance_sqr * glitch_scale;
            let glitched = (nan_re || nan_im || inf_re || inf_im || (!small_delta && delta_too_large));

            if (glitched) {
                out_pixels[idx].iter = n;
                out_pixels[idx].z_re = z_check.re;
                out_pixels[idx].z_im = z_check.im;
                out_pixels[idx].flags = 1u;
                return;
            }
            if (z_curr_norm_sqr > bailout_sqr) {
                out_pixels[idx].iter = n;
                out_pixels[idx].z_re = z_curr_re;
                out_pixels[idx].z_im = z_curr_im;
                out_pixels[idx].flags = 0u;
                return;
            }
        } else if (n >= params.iter_max) {
            break;
        }
    }

    // Clamp n to valid range to avoid out-of-bounds access
    // If n >= iter_max, the point is in the set (will be colored black by color_for_pixel)
    let final_n = min(n, params.iter_max - 1u);
    let z = z_ref[final_n];
    let delta_actual_re = delta_re * delta_scale;
    let delta_actual_im = delta_im * delta_scale;
    // Store iter_max if n >= iter_max to indicate point is in set
    out_pixels[idx].iter = select(final_n, params.iter_max, n >= params.iter_max);
    out_pixels[idx].z_re = z.re + delta_actual_re;
    out_pixels[idx].z_im = z.im + delta_actual_im;
    out_pixels[idx].flags = 0u;
}
