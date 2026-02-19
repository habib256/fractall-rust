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
    plane_transform: u32,
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

fn apply_plane_transform(z: vec2<f32>, mode: u32) -> vec2<f32> {
    // IDs must match PlaneTransform::id() in Rust
    // 0: μ, 1: 1/μ, 2: 1/(μ+0.25), 3: λ, 4: 1/λ, 5: 1/λ-1, 6: 1/(μ-1.40115)
    if (mode == 0u) {
        return z;
    }
    let re = z.x;
    let im = z.y;
    if (mode == 1u) {
        let denom = re * re + im * im;
        if (denom < 1e-20) {
            return vec2<f32>(1e10, 0.0);
        }
        return vec2<f32>(re / denom, -im / denom);
    }
    if (mode == 2u) {
        let re_s = re + 0.25;
        let denom = re_s * re_s + im * im;
        if (denom < 1e-20) {
            return vec2<f32>(1e10, 0.0);
        }
        return vec2<f32>(re_s / denom, -im / denom);
    }

    // λ = 4μ(1-μ)
    // complex multiply helper: a*b
    let one_minus_re = 1.0 - re;
    let one_minus_im = -im;
    // μ * (1-μ)
    let mul_re = re * one_minus_re - im * one_minus_im;
    let mul_im = re * one_minus_im + im * one_minus_re;
    let lam_re = 4.0 * mul_re;
    let lam_im = 4.0 * mul_im;

    if (mode == 3u) {
        return vec2<f32>(lam_re, lam_im);
    }

    // 1/λ (and variants)
    let denom_l = lam_re * lam_re + lam_im * lam_im;
    if (denom_l < 1e-20) {
        return vec2<f32>(1e10, 0.0);
    }
    var inv_re = lam_re / denom_l;
    var inv_im = -lam_im / denom_l;

    if (mode == 4u) {
        return vec2<f32>(inv_re, inv_im);
    }
    if (mode == 5u) {
        inv_re = inv_re - 1.0;
        return vec2<f32>(inv_re, inv_im);
    }
    if (mode == 6u) {
        let re_s = re - 1.40115;
        let denom = re_s * re_s + im * im;
        if (denom < 1e-20) {
            return vec2<f32>(1e10, 0.0);
        }
        return vec2<f32>(re_s / denom, -im / denom);
    }
    return z;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let idx = gid.y * params.width + gid.x;
    let fx = (f32(gid.x) + 0.5) / f32(params.width);
    let fy = (f32(gid.y) + 0.5) / f32(params.height);
    // Compute offset from center directly to avoid precision loss
    var x = params.center_x + (fx - 0.5) * params.span_x;
    var y = params.center_y + (fy - 0.5) * params.span_y;
    let c = apply_plane_transform(vec2<f32>(x, y), params.plane_transform);
    x = c.x;
    y = c.y;

    var z_re = 0.0;
    var z_im = 0.0;
    var i: u32 = 0u;
    let bailout_sqr = params.bailout * params.bailout;

    loop {
        if (i >= params.iter_max) {
            break;
        }
        let z_re_sq = z_re * z_re;
        let z_im_sq = z_im * z_im;
        if (z_re_sq + z_im_sq > bailout_sqr) {
            break;
        }
        let z_im_new = 2.0 * z_re * z_im + y;
        let z_re_new = z_re_sq - z_im_sq + x;
        z_re = z_re_new;
        z_im = z_im_new;
        i = i + 1u;
    }

    out_pixels[idx].iter = i;
    out_pixels[idx].z_re = z_re;
    out_pixels[idx].z_im = z_im;
    out_pixels[idx]._pad = 0u;
}
