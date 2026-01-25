enable f64;

struct Params {
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    seed_re: f64,
    seed_im: f64,
    width: u32,
    height: u32,
    iter_max: u32,
    _pad: u32,
    bailout: f64,
    _pad2: f64,
};

struct PixelOut {
    iter: u32,
    z_re: f32,
    z_im: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out_pixels: array<PixelOut>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let idx = gid.y * params.width + gid.x;
    let fx = f64(gid.x) / f64(params.width);
    let fy = f64(gid.y) / f64(params.height);
    let x = params.xmin + (params.xmax - params.xmin) * fx;
    let y = params.ymin + (params.ymax - params.ymin) * fy;

    var z_re = x;
    var z_im = y;
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
        let z_im_new = 2.0 * z_re * z_im + params.seed_im;
        let z_re_new = z_re_sq - z_im_sq + params.seed_re;
        z_re = z_re_new;
        z_im = z_im_new;
        i = i + 1u;
    }

    out_pixels[idx].iter = i;
    out_pixels[idx].z_re = f32(z_re);
    out_pixels[idx].z_im = f32(z_im);
    out_pixels[idx]._pad = 0u;
}
