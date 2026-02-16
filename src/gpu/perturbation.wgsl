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
const DEFAULT_GLITCH_TOLERANCE: f32 = 1.0e-4;
// ============================================================================

// Calcule la tolérance de glitch adaptative basée sur le niveau de zoom
fn compute_adaptive_glitch_tolerance(pixel_size: f32, user_tolerance: f32) -> f32 {
    // Si l'utilisateur a défini une tolérance personnalisée, la respecter
    if (abs(user_tolerance - DEFAULT_GLITCH_TOLERANCE) > 1.0e-10) {
        return user_tolerance;
    }
    
    // Calculer le niveau de zoom: log10(4 / pixel_size)
    let zoom_level = log(4.0 / max(pixel_size, 1.0e-38)) / log(10.0);
    
    // Tolérance adaptative selon le niveau de zoom
    if (zoom_level < 7.0) {
        return 1.0e-5;      // Zoom peu profond : strict
    } else if (zoom_level < 15.0) {
        return 1.0e-4;      // Moyen : standard
    } else if (zoom_level < 31.0) {
        return 1.0e-3;      // Profond : relaxé
    } else if (zoom_level < 51.0) {
        return 1.0e-2;      // Très profond
    }
    return 1.0e-1;          // Extrême
}

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

fn rescale_delta(re: f32, im: f32, scale: f32) -> vec3<f32> {
    var new_re = re;
    var new_im = im;
    var new_scale = scale;
    let abs_max = max(abs(new_re), abs(new_im));

    // Clamp scale to avoid overflow/underflow
    // f32 exponent range is roughly 2^-126 to 2^127
    let max_scale = 1.0e30;
    let min_scale = 1.0e-30;

    if (abs_max > RESCALE_HI && new_scale < max_scale) {
        let k = floor(log2(abs_max / RESCALE_HI));
        let factor = exp2(min(k, 20.0)); // Limit single rescale step
        new_re = new_re / factor;
        new_im = new_im / factor;
        new_scale = min(new_scale * factor, max_scale);
    } else if (abs_max > 0.0 && abs_max < RESCALE_LO && new_scale > min_scale) {
        let k = floor(log2(RESCALE_LO / abs_max));
        let factor = exp2(min(k, 20.0)); // Limit single rescale step
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
    var delta_re = select(0.0, dc_re, is_julia);
    var delta_im = select(0.0, dc_im, is_julia);
    var delta_scale = 1.0;
    var n: u32 = 0u;
    let bailout_sqr = params.bailout * params.bailout;
    
    // Tolérance adaptative basée sur le niveau de zoom
    let adaptive_tolerance = compute_adaptive_glitch_tolerance(pixel_size, params.glitch_tolerance);
    let glitch_tolerance_sqr = adaptive_tolerance * adaptive_tolerance;

    loop {
        if (n >= params.iter_max) {
            break;
        }

        var stepped = false;
        if (!is_burning_ship && params.bla_levels > 0u) {
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
            if (is_burning_ship) {
                let delta_actual_re = delta_re * delta_scale;
                let delta_actual_im = delta_im * delta_scale;
                let z_re = z.re + delta_actual_re;
                let z_im = z.im + delta_actual_im;
                let re_abs = abs(z_re);
                let im_abs = abs(z_im);
                let z_sq = complex_mul(re_abs, im_abs, re_abs, im_abs);
                let z_next_re = z_sq.x + (params.cref_x + dc_re);
                let z_next_im = z_sq.y + (params.cref_y + dc_im);
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
            } else {
                let delta_actual_re = delta_re * delta_scale;
                let delta_actual_im = delta_im * delta_scale;
                let linear = complex_mul(2.0 * z.re, 2.0 * z.im, delta_actual_re, delta_actual_im);
                let nonlinear = complex_mul(delta_actual_re, delta_actual_im, delta_actual_re, delta_actual_im);
                // IMPORTANT: Pour Mandelbrot, on ajoute dc à chaque itération. Pour Julia, dc est déjà dans delta initial.
                // WGSL: select(false_val, true_val, cond) => cond ? true_val : false_val
                // Donc ici on veut dc si !is_julia, sinon 0.
                let next_re = linear.x + nonlinear.x + select(0.0, dc_re, !is_julia);
                let next_im = linear.y + nonlinear.y + select(0.0, dc_im, !is_julia);
                let scaled = rescale_delta(next_re, next_im, 1.0);
                delta_re = scaled.x;
                delta_im = scaled.y;
                delta_scale = scaled.z;
                n = n + 1u;
            }
        }

        // ====================================================================================
        // REBASING: Critical for avoiding glitches in perturbation
        // ====================================================================================
        // When |Z_m + z_n| < |z_n|, replace z_n with Z_m + z_n and reset m to 0.
        // This prevents delta from diverging and causing widespread glitches.
        // Without rebasing, delta grows uncontrollably at deeper zooms.
        if (n > 0u && n < params.iter_max) {
            let z_check = z_ref[n];
            let delta_actual_re = delta_re * delta_scale;
            let delta_actual_im = delta_im * delta_scale;
            let z_curr_re = z_check.re + delta_actual_re;
            let z_curr_im = z_check.im + delta_actual_im;
            let z_curr_norm_sqr = z_curr_re * z_curr_re + z_curr_im * z_curr_im;
            let delta_norm_sqr = delta_actual_re * delta_actual_re + delta_actual_im * delta_actual_im;

            // Rebasing condition: |z_curr| < |delta| and both are non-zero
            if (z_curr_norm_sqr > 0.0 && delta_norm_sqr > 0.0 && z_curr_norm_sqr < delta_norm_sqr) {
                // Rebase: delta = z_curr, restart from iteration 0
                let scaled = rescale_delta(z_curr_re, z_curr_im, 1.0);
                delta_re = scaled.x;
                delta_im = scaled.y;
                delta_scale = scaled.z;
                n = 0u;
                continue;
            }
        }

        if (n >= params.iter_max) {
            break;
        }

        let z = z_ref[n];
        let delta_actual_re = delta_re * delta_scale;
        let delta_actual_im = delta_im * delta_scale;
        let z_re = z.re + delta_actual_re;
        let z_im = z.im + delta_actual_im;
        let z_ref_norm_sqr = z.re * z.re + z.im * z.im;
        let delta_norm_sqr = delta_actual_re * delta_actual_re + delta_actual_im * delta_actual_im;
        let nan_re = z_re != z_re;
        let nan_im = z_im != z_im;
        let inf_re = abs(z_re) > 1e30;
        let inf_im = abs(z_im) > 1e30;

        // Glitch detection: with rebasing enabled, delta-based detection is now stable
        // Detect glitches when:
        // 1. NaN or Inf values (always invalid)
        // 2. Delta is too large relative to z_ref (indicates precision loss)
        let glitch_scale = z_ref_norm_sqr + 1.0;
        let delta_too_large = delta_norm_sqr > glitch_tolerance_sqr * glitch_scale;
        let glitched = (nan_re || nan_im || inf_re || inf_im || delta_too_large);
        
        if (glitched) {
            // IMPORTANT: Ne pas stocker les valeurs invalides (NaN/Inf) qui causent des artefacts visuels
            // Utiliser des valeurs par défaut valides qui seront remplacées par la correction CPU
            // Utiliser z_ref comme valeur par défaut pour éviter les artefacts circulaires
            out_pixels[idx].iter = n;
            out_pixels[idx].z_re = z.re;  // Utiliser z_ref au lieu de z_re invalide
            out_pixels[idx].z_im = z.im;  // Utiliser z_ref au lieu de z_im invalide
            out_pixels[idx].flags = 1u;
            return;
        }
        if (z_re * z_re + z_im * z_im > bailout_sqr) {
            out_pixels[idx].iter = n;
            out_pixels[idx].z_re = z_re;
            out_pixels[idx].z_im = z_im;
            out_pixels[idx].flags = 0u;
            return;
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
