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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let idx = gid.y * params.width + gid.x;
    if (reuse_mask[idx] == 0u) {
        return;
    }
    let dx = (f32(gid.x) * params.span_x / f32(params.width)) - params.span_x * 0.5;
    let dy = (f32(gid.y) * params.span_y / f32(params.height)) - params.span_y * 0.5;
    let dc_re = dx;
    let dc_im = dy;

    let is_julia = params.fractal_kind == 1u;
    let is_burning_ship = params.fractal_kind == 2u;
    var delta_re = select(0.0, dc_re, is_julia);
    var delta_im = select(0.0, dc_im, is_julia);
    var delta_scale = 1.0;
    var n: u32 = 0u;
    let bailout_sqr = params.bailout * params.bailout;
    let glitch_tolerance_sqr = params.glitch_tolerance * params.glitch_tolerance;

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
                let offset = bla_meta.level_offsets[lvl];
                let node = bla_nodes[offset + n];
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
