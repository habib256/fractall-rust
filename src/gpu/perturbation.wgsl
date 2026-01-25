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
    _pad: u32,
    _pad2: vec3<f32>,
};

struct ZRef {
    re: f32,
    im: f32,
};

struct PixelOut {
    iter: u32,
    z_re: f32,
    z_im: f32,
    _pad: u32,
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

fn complex_mul(a_re: f32, a_im: f32, b_re: f32, b_im: f32) -> vec2<f32> {
    return vec2<f32>(a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let idx = gid.y * params.width + gid.x;
    let fx = f32(gid.x) / f32(params.width);
    let fy = f32(gid.y) / f32(params.height);
    let x = params.xmin + (params.xmax - params.xmin) * fx;
    let y = params.ymin + (params.ymax - params.ymin) * fy;
    let dc_re = x - params.cref_x;
    let dc_im = y - params.cref_y;

    var delta_re = 0.0;
    var delta_im = 0.0;
    var n: u32 = 0u;
    let bailout_sqr = params.bailout * params.bailout;

    loop {
        if (n >= params.iter_max) {
            break;
        }

        var stepped = false;
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
                delta_re = mul1.x + mul2.x;
                delta_im = mul1.y + mul2.y;
                n = n + (1u << lvl);
                stepped = true;
                break;
            }
        }

        if (!stepped) {
            let z = z_ref[n];
            let linear = complex_mul(2.0 * z.re, 2.0 * z.im, delta_re, delta_im);
            let nonlinear = complex_mul(delta_re, delta_im, delta_re, delta_im);
            delta_re = linear.x + nonlinear.x + dc_re;
            delta_im = linear.y + nonlinear.y + dc_im;
            n = n + 1u;
        }

        if (n >= params.iter_max) {
            break;
        }

        let z = z_ref[n];
        let z_re = z.re + delta_re;
        let z_im = z.im + delta_im;
        if (z_re * z_re + z_im * z_im > bailout_sqr) {
            out_pixels[idx].iter = n;
            out_pixels[idx].z_re = z_re;
            out_pixels[idx].z_im = z_im;
            out_pixels[idx]._pad = 0u;
            return;
        }
    }

    let z = z_ref[min(n, params.iter_max - 1u)];
    out_pixels[idx].iter = n;
    out_pixels[idx].z_re = z.re + delta_re;
    out_pixels[idx].z_im = z.im + delta_im;
    out_pixels[idx]._pad = 0u;
}
