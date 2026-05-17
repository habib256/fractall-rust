// Bytecode-driven escape-time kernel (P3.1 #7 prototype).
//
// Remplace mandelbrot_f32.wgsl, julia_f32.wgsl, burning_ship_f32.wgsl par un
// shader unique qui interprète un bytecode 8-opcodes uploadé via storage
// buffer. Mirror direct du jeu d'opcodes de bytecode/mod.rs::Op et de
// l'interpréteur de bytecode/interp.rs::iterate_bytecode_f64.
//
// Limitations actuelles (prototype) :
// - Mono-phase seulement (pas de cycle phase). À étendre en passant
//   `phase_offsets[]` quand on supportera les hybrides.
// - f32 uniquement (suffit pour zoom < ~1e7 ; sinon GPU perturbation requise).
// - Pas de plane transform (à câbler depuis mandelbrot_f32.wgsl si besoin).
// - L'orchestration Rust (Bind groups, encoding bytecode, dispatch) reste
//   à faire — c'est le travail de Session de #7 final.
//
// Opcodes (doivent matcher Rust src/fractal/bytecode/mod.rs::Op) :
//   0 = Sqr     z := z * z
//   1 = Mul     z := z * stored
//   2 = Store   stored := z
//   3 = AbsX    z.re := |z.re|
//   4 = AbsY    z.im := |z.im|
//   5 = NegX    z.re := -z.re
//   6 = NegY    z.im := -z.im
//   7 = Add     z := z + c (fin de phase, +1 itération)

struct Params {
    center_x: f32,
    center_y: f32,
    span_x: f32,
    span_y: f32,
    // Seed pour Mandelbrot-like (utilisé comme z0) ou pour Julia (utilisé comme c).
    seed_re: f32,
    seed_im: f32,
    width: u32,
    height: u32,
    iter_max: u32,
    bailout: f32,
    // 0 = Mandelbrot-like (z0 = seed, c = pixel),
    // 1 = Julia-like (z0 = pixel, c = seed)
    is_julia: u32,
    // Longueur du bytecode (nombre d'opcodes dans la phase).
    bytecode_len: u32,
};

struct PixelOut {
    iter: u32,
    z_re: f32,
    z_im: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out_pixels: array<PixelOut>;
@group(0) @binding(2) var<storage, read> bytecode: array<u32>;

// Multiplication complexe : (a + ib) * (c + id) = (ac-bd) + i(ad+bc)
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x,
    );
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let idx = gid.y * params.width + gid.x;
    let fx = (f32(gid.x) + 0.5) / f32(params.width);
    let fy = (f32(gid.y) + 0.5) / f32(params.height);
    let pixel_re = params.center_x + (fx - 0.5) * params.span_x;
    let pixel_im = params.center_y + (fy - 0.5) * params.span_y;

    var z: vec2<f32>;
    var c: vec2<f32>;
    if (params.is_julia != 0u) {
        z = vec2<f32>(pixel_re, pixel_im);
        c = vec2<f32>(params.seed_re, params.seed_im);
    } else {
        z = vec2<f32>(params.seed_re, params.seed_im);
        c = vec2<f32>(pixel_re, pixel_im);
    }

    var stored: vec2<f32> = z;
    var iter: u32 = 0u;
    let bailout_sqr = params.bailout * params.bailout;

    // Test bailout initial pour cohérence avec l'interpréteur CPU.
    if (dot(z, z) >= bailout_sqr) {
        out_pixels[idx].iter = 0u;
        out_pixels[idx].z_re = z.x;
        out_pixels[idx].z_im = z.y;
        out_pixels[idx]._pad = 0u;
        return;
    }

    loop {
        if (iter >= params.iter_max) {
            break;
        }
        // Exécute toutes les opcodes de la phase mono-phase.
        var op_idx: u32 = 0u;
        loop {
            if (op_idx >= params.bytecode_len) {
                break;
            }
            let op = bytecode[op_idx];
            if (op == 0u) {
                // Sqr
                z = cmul(z, z);
            } else if (op == 1u) {
                // Mul
                z = cmul(z, stored);
            } else if (op == 2u) {
                // Store
                stored = z;
            } else if (op == 3u) {
                // AbsX
                z.x = abs(z.x);
            } else if (op == 4u) {
                // AbsY
                z.y = abs(z.y);
            } else if (op == 5u) {
                // NegX
                z.x = -z.x;
            } else if (op == 6u) {
                // NegY
                z.y = -z.y;
            } else if (op == 7u) {
                // Add (fin de phase)
                z = z + c;
                iter = iter + 1u;
            }
            op_idx = op_idx + 1u;
        }

        // Bailout check après la phase (équivalent CPU).
        if (!(z.x == z.x) || !(z.y == z.y)) {
            // NaN
            break;
        }
        if (dot(z, z) >= bailout_sqr) {
            break;
        }
    }

    out_pixels[idx].iter = iter;
    out_pixels[idx].z_re = z.x;
    out_pixels[idx].z_im = z.y;
    out_pixels[idx]._pad = 0u;
}
