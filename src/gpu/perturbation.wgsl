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
const RESCALE_HI: f32 = 1.0e4;
const RESCALE_LO: f32 = 1.0e-4;
const DEFAULT_GLITCH_TOLERANCE: f32 = 1.0e-4;
const DS_THRESHOLD: f32 = 1.0e-6;  // Seuil pour basculer en mode double-single

// ============================================================================
// Arithmétique Double-Single (DS) - Émule f64 avec deux f32
// Étend la précision GPU de ~10^7 à ~10^14
// ============================================================================

// Structure double-single: hi contient la partie principale, lo la correction
struct DS {
    hi: f32,
    lo: f32,
}

// Crée un DS à partir d'un f32
fn ds_from_f32(x: f32) -> DS {
    return DS(x, 0.0);
}

// Convertit un DS en f32 (perd la précision supplémentaire)
fn ds_to_f32(a: DS) -> f32 {
    return a.hi + a.lo;
}

// Quick two-sum: calcule s = a + b avec erreur e, supposant |a| >= |b|
fn two_sum_quick(a: f32, b: f32) -> DS {
    let s = a + b;
    let e = b - (s - a);
    return DS(s, e);
}

// Two-sum: calcule s = a + b avec erreur e, sans hypothèse sur les magnitudes
fn two_sum(a: f32, b: f32) -> DS {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    return DS(s, e);
}

// Two-product: calcule p = a * b avec erreur e, utilisant fma si disponible
fn two_prod(a: f32, b: f32) -> DS {
    let p = a * b;
    let e = fma(a, b, -p);
    return DS(p, e);
}

// Addition double-single: (a.hi + a.lo) + (b.hi + b.lo)
fn ds_add(a: DS, b: DS) -> DS {
    // Somme des parties hautes avec propagation d'erreur
    let s = two_sum(a.hi, b.hi);
    // Ajouter les parties basses et l'erreur
    let t = a.lo + b.lo + s.lo;
    // Renormaliser
    let hi = s.hi + t;
    let lo = t - (hi - s.hi);
    return DS(hi, lo);
}

// Soustraction double-single
fn ds_sub(a: DS, b: DS) -> DS {
    return ds_add(a, DS(-b.hi, -b.lo));
}

// Multiplication double-single: (a.hi + a.lo) * (b.hi + b.lo)
fn ds_mul(a: DS, b: DS) -> DS {
    // Produit des parties hautes avec erreur
    let p = two_prod(a.hi, b.hi);
    // Termes croisés
    let t = a.hi * b.lo + a.lo * b.hi + p.lo;
    // Renormaliser
    let hi = p.hi + t;
    let lo = t - (hi - p.hi);
    return DS(hi, lo);
}

// Multiplication DS par f32
fn ds_mul_f32(a: DS, b: f32) -> DS {
    let p = two_prod(a.hi, b);
    let t = a.lo * b + p.lo;
    let hi = p.hi + t;
    let lo = t - (hi - p.hi);
    return DS(hi, lo);
}

// Norme au carré d'un complexe DS: |z|² = re² + im²
fn ds_norm_sqr(re: DS, im: DS) -> f32 {
    let re_sq = ds_mul(re, re);
    let im_sq = ds_mul(im, im);
    let sum = ds_add(re_sq, im_sq);
    return ds_to_f32(sum);
}

// Structure pour un complexe en double-single
struct ComplexDS {
    re: DS,
    im: DS,
}

fn cds_from_f32(re: f32, im: f32) -> ComplexDS {
    return ComplexDS(ds_from_f32(re), ds_from_f32(im));
}

fn cds_to_f32(z: ComplexDS) -> vec2<f32> {
    return vec2<f32>(ds_to_f32(z.re), ds_to_f32(z.im));
}

// Addition de complexes DS
fn cds_add(a: ComplexDS, b: ComplexDS) -> ComplexDS {
    return ComplexDS(ds_add(a.re, b.re), ds_add(a.im, b.im));
}

// Soustraction de complexes DS
fn cds_sub(a: ComplexDS, b: ComplexDS) -> ComplexDS {
    return ComplexDS(ds_sub(a.re, b.re), ds_sub(a.im, b.im));
}

// Multiplication de complexes DS: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
fn cds_mul(a: ComplexDS, b: ComplexDS) -> ComplexDS {
    let ac = ds_mul(a.re, b.re);
    let bd = ds_mul(a.im, b.im);
    let ad = ds_mul(a.re, b.im);
    let bc = ds_mul(a.im, b.re);
    return ComplexDS(ds_sub(ac, bd), ds_add(ad, bc));
}

// Multiplication complexe DS par f32
fn cds_mul_f32(a: ComplexDS, b: f32) -> ComplexDS {
    return ComplexDS(ds_mul_f32(a.re, b), ds_mul_f32(a.im, b));
}

// Carré d'un complexe DS: z² = (re² - im²) + 2·re·im·i
fn cds_sqr(z: ComplexDS) -> ComplexDS {
    let re_sq = ds_mul(z.re, z.re);
    let im_sq = ds_mul(z.im, z.im);
    let two_re_im = ds_mul_f32(ds_mul(z.re, z.im), 2.0);
    return ComplexDS(ds_sub(re_sq, im_sq), two_re_im);
}

// Norme au carré d'un complexe DS
fn cds_norm_sqr(z: ComplexDS) -> f32 {
    return ds_norm_sqr(z.re, z.im);
}

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
    if (abs_max > RESCALE_HI) {
        let k = floor(log2(abs_max / RESCALE_HI));
        let factor = exp2(k);
        new_re = new_re / factor;
        new_im = new_im / factor;
        new_scale = new_scale * factor;
    } else if (abs_max > 0.0 && abs_max < RESCALE_LO) {
        let k = floor(log2(RESCALE_LO / abs_max));
        let factor = exp2(k);
        new_re = new_re * factor;
        new_im = new_im * factor;
        new_scale = new_scale / factor;
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
    if (reuse_mask[idx] == 0u) {
        return;
    }
    
    // Calculer le pixel_size pour déterminer si on utilise le mode DS
    let pixel_size = params.span_x / f32(params.width);
    let use_ds = pixel_size < DS_THRESHOLD;
    
    // Calcul de la position du pixel avec précision étendue si nécessaire
    var dc_re: f32;
    var dc_im: f32;
    
    if (use_ds) {
        // Mode double-single pour une meilleure précision sur les zooms profonds
        // Calcul: ((gid.x / width) - 0.5) * span_x
        // En DS: position = (gid.x * span_x) / width - span_x * 0.5
        let gid_x_ds = ds_from_f32(f32(gid.x));
        let gid_y_ds = ds_from_f32(f32(gid.y));
        let span_x_ds = ds_from_f32(params.span_x);
        let span_y_ds = ds_from_f32(params.span_y);
        let width_inv = ds_from_f32(1.0 / f32(params.width));
        let height_inv = ds_from_f32(1.0 / f32(params.height));
        let half = ds_from_f32(0.5);
        
        // dx = gid.x * span_x / width - span_x * 0.5
        let x_frac = ds_mul(gid_x_ds, width_inv);
        let x_centered = ds_sub(x_frac, half);
        let dx_ds = ds_mul(x_centered, span_x_ds);
        
        // dy = gid.y * span_y / height - span_y * 0.5
        let y_frac = ds_mul(gid_y_ds, height_inv);
        let y_centered = ds_sub(y_frac, half);
        let dy_ds = ds_mul(y_centered, span_y_ds);
        
        dc_re = ds_to_f32(dx_ds);
        dc_im = ds_to_f32(dy_ds);
    } else {
        // Mode standard f32
        let dx = (f32(gid.x) * params.span_x / f32(params.width)) - params.span_x * 0.5;
        let dy = (f32(gid.y) * params.span_y / f32(params.height)) - params.span_y * 0.5;
        dc_re = dx;
        dc_im = dy;
    }

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
                        let next_re = mul1.x + select(mul2.x, 0.0, is_julia) + mul3.x;
                        let next_im = mul1.y + select(mul2.y, 0.0, is_julia) + mul3.y;
                        let scaled = rescale_delta(next_re, next_im, delta_scale);
                        delta_re = scaled.x;
                        delta_im = scaled.y;
                        delta_scale = scaled.z;
                    } else {
                        let next_re = mul1.x + select(mul2.x, 0.0, is_julia);
                        let next_im = mul1.y + select(mul2.y, 0.0, is_julia);
                        let scaled = rescale_delta(next_re, next_im, delta_scale);
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
                let scaled = rescale_delta(next_re, next_im, delta_scale);
                delta_re = scaled.x;
                delta_im = scaled.y;
                delta_scale = scaled.z;
            } else {
                let delta_actual_re = delta_re * delta_scale;
                let delta_actual_im = delta_im * delta_scale;
                let linear = complex_mul(2.0 * z.re, 2.0 * z.im, delta_actual_re, delta_actual_im);
                let nonlinear = complex_mul(delta_actual_re, delta_actual_im, delta_actual_re, delta_actual_im);
                let next_re = linear.x + nonlinear.x + select(dc_re, 0.0, is_julia);
                let next_im = linear.y + nonlinear.y + select(dc_im, 0.0, is_julia);
                let scaled = rescale_delta(next_re, next_im, delta_scale);
                delta_re = scaled.x;
                delta_im = scaled.y;
                delta_scale = scaled.z;
                n = n + 1u;
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
        let glitched = (nan_re || nan_im || inf_re || inf_im)
            || (z_ref_norm_sqr > 0.0 && delta_norm_sqr > glitch_tolerance_sqr * z_ref_norm_sqr);
        if (glitched) {
            out_pixels[idx].iter = n;
            out_pixels[idx].z_re = z_re;
            out_pixels[idx].z_im = z_im;
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

    let z = z_ref[min(n, params.iter_max - 1u)];
    let delta_actual_re = delta_re * delta_scale;
    let delta_actual_im = delta_im * delta_scale;
    out_pixels[idx].iter = n;
    out_pixels[idx].z_re = z.re + delta_actual_re;
    out_pixels[idx].z_im = z.im + delta_actual_im;
    out_pixels[idx].flags = 0u;
}
