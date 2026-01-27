// f64 support is enabled via SHADER_F64 feature on the device

struct Params {
    center_x: f64,
    center_y: f64,
    span_x: f64,
    span_y: f64,
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
    let fx: f64 = f64(gid.x) / f64(params.width);
    let fy: f64 = f64(gid.y) / f64(params.height);
    // Compute offset from center directly to avoid precision loss
    let x: f64 = params.center_x + (fx - f64(0.5)) * params.span_x;
    let y: f64 = params.center_y + (fy - f64(0.5)) * params.span_y;

    var z_re: f64 = f64(0.0);
    var z_im: f64 = f64(0.0);
    var i: u32 = 0u;
    let bailout_sqr: f64 = params.bailout * params.bailout;

    loop {
        if (i >= params.iter_max) {
            break;
        }
        let z_re_sq: f64 = z_re * z_re;
        let z_im_sq: f64 = z_im * z_im;
        if (z_re_sq + z_im_sq > bailout_sqr) {
            break;
        }
        let z_im_new: f64 = f64(2.0) * z_re * z_im + y;
        let z_re_new: f64 = z_re_sq - z_im_sq + x;
        z_re = z_re_new;
        z_im = z_im_new;
        i = i + 1u;
    }

    out_pixels[idx].iter = i;
    out_pixels[idx].z_re = f32(z_re);
    out_pixels[idx].z_im = f32(z_im);
    out_pixels[idx]._pad = 0u;
}
