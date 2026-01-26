// ============================================================================
// Mandelbrot avec arithmétique Double-Single (DS)
// Émule f64 avec deux f32 pour une précision étendue (~10^14)
// ============================================================================

struct Params {
    center_x: f32,
    center_y: f32,
    span_x: f32,
    span_y: f32,
    seed_re: f32,
    seed_im: f32,
    width: u32,
    height: u32,
    iter_max: u32,
    _pad: u32,
    bailout: f32,
    _pad2: vec3<f32>,
};

struct PixelOut {
    iter: u32,
    z_re: f32,
    z_im: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out_pixels: array<PixelOut>;

// ============================================================================
// Arithmétique Double-Single (DS) - Émule f64 avec deux f32
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
    let s = two_sum(a.hi, b.hi);
    let t = a.lo + b.lo + s.lo;
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
    let p = two_prod(a.hi, b.hi);
    let t = a.hi * b.lo + a.lo * b.hi + p.lo;
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

// ============================================================================
// Arithmétique complexe Double-Single
// ============================================================================

struct ComplexDS {
    re: DS,
    im: DS,
}

fn cds_from_f32(re: f32, im: f32) -> ComplexDS {
    return ComplexDS(ds_from_f32(re), ds_from_f32(im));
}

// Addition de complexes DS
fn cds_add(a: ComplexDS, b: ComplexDS) -> ComplexDS {
    return ComplexDS(ds_add(a.re, b.re), ds_add(a.im, b.im));
}

// Carré d'un complexe DS: z² = (re² - im²) + 2·re·im·i
fn cds_sqr(z: ComplexDS) -> ComplexDS {
    let re_sq = ds_mul(z.re, z.re);
    let im_sq = ds_mul(z.im, z.im);
    let two_re_im = ds_mul_f32(ds_mul(z.re, z.im), 2.0);
    return ComplexDS(ds_sub(re_sq, im_sq), two_re_im);
}

// Norme au carré d'un complexe DS (retourne f32 pour comparaison)
fn cds_norm_sqr(z: ComplexDS) -> f32 {
    let re_sq = ds_mul(z.re, z.re);
    let im_sq = ds_mul(z.im, z.im);
    let sum = ds_add(re_sq, im_sq);
    return ds_to_f32(sum);
}

// ============================================================================
// Kernel principal
// ============================================================================

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let idx = gid.y * params.width + gid.x;
    
    // Calcul des coordonnées avec précision DS
    // Compute offset from center directly to avoid precision loss
    let fx = f32(gid.x) / f32(params.width);
    let fy = f32(gid.y) / f32(params.height);
    
    // Span en DS
    let span_x = ds_from_f32(params.span_x);
    let span_y = ds_from_f32(params.span_y);
    
    // Position c = center + (fraction - 0.5) * span
    let offset_x = fx - 0.5;
    let offset_y = fy - 0.5;
    let c_re = ds_add(ds_from_f32(params.center_x), ds_mul_f32(span_x, offset_x));
    let c_im = ds_add(ds_from_f32(params.center_y), ds_mul_f32(span_y, offset_y));
    let c = ComplexDS(c_re, c_im);
    
    // z commence à 0
    var z = cds_from_f32(0.0, 0.0);
    var i: u32 = 0u;
    let bailout_sqr = params.bailout * params.bailout;

    loop {
        if (i >= params.iter_max) {
            break;
        }
        
        let norm_sqr = cds_norm_sqr(z);
        if (norm_sqr > bailout_sqr) {
            break;
        }
        
        // z = z² + c
        z = cds_add(cds_sqr(z), c);
        i = i + 1u;
    }

    // Sortie: convertir z en f32 pour le stockage
    out_pixels[idx].iter = i;
    out_pixels[idx].z_re = ds_to_f32(z.re);
    out_pixels[idx].z_im = ds_to_f32(z.im);
    out_pixels[idx]._pad = 0u;
}
