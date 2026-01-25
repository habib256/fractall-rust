struct Params {
    xmin: f32,
    xmax: f32,
    ymin: f32,
    ymax: f32,
    cref_x: f32,
    cref_y: f32,
    width: u32,
    height: u32,
    iter_max: u32,
    bailout: f32,
    bla_levels: u32,
    fractal_kind: u32,
    glitch_tolerance: f32,
    _pad0: vec4<u32>,
    _pad: vec4<u32>,
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
    validity: f32,
    _pad: f32,
};

const MAX_LEVELS: u32 = 17u;

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

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let idx = gid.y * params.width + gid.x;
    if (reuse_mask[idx] == 0u) {
        return;
    }
    let fx = f32(gid.x) / f32(params.width);
    let fy = f32(gid.y) / f32(params.height);
    let x = params.xmin + (params.xmax - params.xmin) * fx;
    let y = params.ymin + (params.ymax - params.ymin) * fy;
    let dc_re = x - params.cref_x;
    let dc_im = y - params.cref_y;

    let is_julia = params.fractal_kind == 1u;
    let is_burning_ship = params.fractal_kind == 2u;
    var delta_re = select(0.0, dc_re, is_julia);
    var delta_im = select(0.0, dc_im, is_julia);
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
                let delta_norm_sqr = delta_re * delta_re + delta_im * delta_im;
                if (delta_norm_sqr < node.validity * node.validity) {
                    let mul1 = complex_mul(node.a_re, node.a_im, delta_re, delta_im);
                    let mul2 = complex_mul(node.b_re, node.b_im, dc_re, dc_im);
                    delta_re = mul1.x + select(mul2.x, 0.0, is_julia);
                    delta_im = mul1.y + select(mul2.y, 0.0, is_julia);
                    n = n + (1u << lvl);
                    stepped = true;
                    break;
                }
            }
        }

        if (!stepped) {
            let z = z_ref[n];
            if (is_burning_ship) {
                let z_re = z.re + delta_re;
                let z_im = z.im + delta_im;
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
                delta_re = z_next_re - z_next_ref.re;
                delta_im = z_next_im - z_next_ref.im;
            } else {
                let linear = complex_mul(2.0 * z.re, 2.0 * z.im, delta_re, delta_im);
                let nonlinear = complex_mul(delta_re, delta_im, delta_re, delta_im);
                delta_re = linear.x + nonlinear.x + select(dc_re, 0.0, is_julia);
                delta_im = linear.y + nonlinear.y + select(dc_im, 0.0, is_julia);
                n = n + 1u;
            }
        }

        if (n >= params.iter_max) {
            break;
        }

        let z = z_ref[n];
        let z_re = z.re + delta_re;
        let z_im = z.im + delta_im;
        let z_ref_norm_sqr = z.re * z.re + z.im * z.im;
        let delta_norm_sqr = delta_re * delta_re + delta_im * delta_im;
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
    out_pixels[idx].iter = n;
    out_pixels[idx].z_re = z.re + delta_re;
    out_pixels[idx].z_im = z.im + delta_im;
    out_pixels[idx].flags = 0u;
}
