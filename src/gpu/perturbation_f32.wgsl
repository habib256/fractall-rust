// Kernel perturbation GPU en MANTISSE f32 + détecteur de fiabilité (G9.6).
//
// Destiné aux GPU sans `SHADER_F64` (Metal / Apple Silicon, une partie des
// iGPU) où le kernel f64 natif (`perturbation.wgsl`) n'existe pas et où toute
// la perturbation retombait sur le CPU.
//
// Pourquoi f32 suffit (souvent) : en perturbation, la mantisse requise pour
// une frame centrée est ~log2(diagonale en pixels) ≈ 10-13 b (modèle F3
// `render.cc:219`, cf. `wisdom.rs`) ; c'est la SENSIBILITÉ de certains pixels
// (cancellation au rebase, amplification de Lyapunov) qui exige davantage.
// Fraktaler-3 tourne en float 24 b par défaut à zoom modéré et le paie en
// pixels faux sans le savoir (mesuré 3-voies vs GMP, cf. TODO G3). Ici chaque
// pixel porte le traqueur de shadowing de `bytecode/reliability.rs` (même
// récurrence, `u = 2⁻²⁴`) : un pixel dont l'erreur, ramenée en espace-c,
// dépasse `kappa` pixel est marqué `flags = 2` et RE-CALCULÉ sur le CPU par
// l'hôte. Le GPU fait le gros du travail, le CPU garantit la justesse.
//
// Sémantique de boucle identique au kernel f64 (n/m séparés, rebasing F3
// strict, garde anti-over-skip BLA, références tronquées périodiques /
// atom-domain). Mandelbrot et Julia seulement (le traqueur est écrit pour
// z²+c ; Burning Ship reste sur CPU/f64).
//
// Plage : δ ~ pixel doit rester normal en f32 (≥ 1.2e-38) et |dz/dc| sous
// 3.4e38 — l'hôte borne le span (`GPU_F32_SPAN_MIN`).

struct Params {
    offset_x_hi: f32,
    offset_x_lo: f32,
    offset_y_hi: f32,
    offset_y_lo: f32,
    span_x_hi: f32,
    span_x_lo: f32,
    span_y_hi: f32,
    span_y_lo: f32,
    width: u32,
    height: u32,
    iter_max: u32,
    bailout: f32,
    bla_levels: u32,
    fractal_kind: u32,
    ref_len: u32,
    series_order: u32,
    series_threshold: f32,
    cycle_start: u32,
    cycle_period: u32,
    atom_truncated: u32,
    aa_sample: u32,
    aa_scale: f32,
    // Seuil de shadowing (fraction de pixel) et epsilon de validité BLA.
    kappa: f32,
    bla_epsilon: f32,
};

struct PixelOut {
    iter: u32,
    z_re: f32,
    z_im: f32,
    flags: u32,
};

struct BlaNode {
    a: vec2<f32>,
    b: vec2<f32>,
    c: vec2<f32>,
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
@group(0) @binding(4) var<storage, read> z_ref: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> reuse_mask: array<u32>;

// u = 2⁻²⁴ (epsilon f32) ; constante d'arrondi local d'un pas complexe.
const U: f32 = 5.9604645e-8;
const GAMMA: f32 = 2.3841858e-7;
const F32_BIG: f32 = 1.0e37;

fn burtle_hash(value: u32) -> u32 {
    var a = value;
    a = a + 0x7ed55d16u + (a << 12u);
    a = (a ^ 0xc761c23cu) ^ (a >> 19u);
    a = a + 0x165667b1u + (a << 5u);
    a = (a + 0xd3a2646cu) ^ (a << 9u);
    a = a + 0xfd7046c5u + (a << 3u);
    return (a ^ 0xb55a4f09u) ^ (a >> 16u);
}
fn radical_inverse(value: u32, base: u32) -> f32 {
    var a = value; var reversed = 0u; var inv = 1.0; let bi = 1.0 / f32(base);
    loop { if (a == 0u) { break; } let next = a / base; reversed = reversed * base + a - base * next; inv = inv * bi; a = next; }
    return min(f32(reversed) * inv, 0.99999994);
}
fn tent(v: f32) -> f32 {
    let orig = v * 2.0 - 1.0; if (orig == 0.0) { return 0.0; }
    return max(orig / sqrt(abs(orig)), -1.0) - select(-1.0, 1.0, orig >= 0.0);
}
fn aa_offset(idx: u32) -> vec2<f32> {
    if (params.aa_scale == 0.0) { return vec2<f32>(); }
    let h = f32(burtle_hash(idx)) / 4294967296.0;
    return vec2<f32>(tent(fract(radical_inverse(params.aa_sample, 2u) + h)), tent(fract(radical_inverse(params.aa_sample, 3u) + h))) * params.aa_scale;
}

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn norm_sqr(a: vec2<f32>) -> f32 {
    return a.x * a.x + a.y * a.y;
}

fn zref_at(m: u32) -> vec2<f32> {
    // ⚠️ clamp CLAUDE.md : après un pas, m peut valoir ref_len.
    return z_ref[min(m, params.ref_len - 1u)];
}

// Verdict de shadowing (mirror `ShadowTracker::shadow_ratio`) : erreur
// ramenée en espace-c, en pixels, comparée à κ. Borne débordée ou NaN →
// non fiable ; dérivée débordée avec erreur finie → expansion énorme, fiable.
fn unreliable(err: f32, deriv: vec2<f32>, pixel_size: f32) -> bool {
    if (err != err || err > F32_BIG) {
        return true;
    }
    if (err == 0.0) {
        return false;
    }
    let d = length(deriv);
    if (d != d || d > F32_BIG) {
        return false;
    }
    if (d == 0.0) {
        return true;
    }
    return err / d > params.kappa * pixel_size;
}

fn write_pixel(idx: u32, iter: u32, z: vec2<f32>, flags: u32) {
    out_pixels[idx].iter = iter;
    out_pixels[idx].z_re = z.x;
    out_pixels[idx].z_im = z.y;
    out_pixels[idx].flags = flags;
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

    // Mapping pixel→dc : la paire hi/lo garde span/offset à ~2⁻⁴⁸ ; le produit
    // ratio·span en f32 pur a une erreur relative 2⁻²⁴ de |dc| ≤ span/2, soit
    // ≪ 1 pixel pour toute image < 10⁶ px de côté.
    let span_x = params.span_x_hi + params.span_x_lo;
    let span_y = params.span_y_hi + params.span_y_lo;
    let jitter = aa_offset(idx);
    let x_ratio = (f32(gid.x) + 0.5 + jitter.x) / f32(params.width) - 0.5;
    let y_ratio = (f32(gid.y) + 0.5 + jitter.y) / f32(params.height) - 0.5;
    let dc = vec2<f32>(
        x_ratio * params.span_x_hi + (x_ratio * params.span_x_lo + params.offset_x_hi + params.offset_x_lo),
        y_ratio * params.span_y_hi + (y_ratio * params.span_y_lo + params.offset_y_hi + params.offset_y_lo),
    );
    let pixel_size = max(span_x / f32(params.width), span_y / f32(params.height));

    let is_julia = params.fractal_kind == 1u;
    var delta = vec2<f32>();
    // Traqueur : borne d'erreur absolue `err` sur z, dérivée `deriv` = dz/dc
    // (Mandelbrot, D₀ = 0) ou dz/dz₀ (Julia, D₀ = 1).
    var err: f32 = 0.0;
    var deriv = vec2<f32>();
    if (is_julia) {
        delta = dc;
        deriv = vec2<f32>(1.0, 0.0);
    }
    let dc_abs = length(dc);
    var n: u32 = 0u;
    var m: u32 = 0u;
    let bailout_sqr = params.bailout * params.bailout;
    let use_bla = params.bla_levels > 0u;
    let use_series = params.series_order >= 2u;
    let series_threshold_sqr = params.series_threshold * params.series_threshold;

    while (n < params.iter_max) {
        let z_m = zref_at(m);
        let z_abs = z_m + delta;
        if (norm_sqr(z_abs) >= bailout_sqr) {
            write_pixel(idx, n, z_abs, select(0u, 2u, unreliable(err, deriv, pixel_size)));
            return;
        }

        var stepped = false;
        if (use_bla) {
            let delta_norm_sqr = norm_sqr(delta);
            var level: i32 = i32(params.bla_levels);
            loop {
                level = level - 1;
                if (level < 0) {
                    break;
                }
                let lvl = u32(level);
                if (m >= bla_meta.level_lengths[lvl]) {
                    continue;
                }
                let node = bla_nodes[bla_meta.level_offsets[lvl] + m];
                if (delta_norm_sqr >= node.validity * node.validity) {
                    continue;
                }
                let skip = 1u << lvl;
                let new_n = n + skip;
                let new_m = m + skip;
                if (new_n > params.iter_max || new_m >= params.ref_len) {
                    continue;
                }
                if (params.atom_truncated != 0u && new_m + 1u >= params.ref_len) {
                    continue;
                }
                let a_delta = cmul(node.a, delta);
                var b_dc = vec2<f32>();
                if (!is_julia) {
                    b_dc = cmul(node.b, dc);
                }
                var cand = a_delta + b_dc;
                var c_term = vec2<f32>();
                if (use_series && delta_norm_sqr < series_threshold_sqr) {
                    c_term = cmul(node.c, cmul(delta, delta));
                    cand = cand + c_term;
                }
                let z_end = zref_at(new_m) + cand;
                if (skip >= 2u && norm_sqr(z_end) >= bailout_sqr) {
                    break;
                }
                // Traqueur, saut BLA : D' = (A + 2Cδ)·D (+ B), A conforme →
                // σ₁(A) = |A| ; troncature ε par saut + arrondi local.
                var jac = node.a;
                if (use_series && delta_norm_sqr < series_threshold_sqr) {
                    jac = jac + 2.0 * cmul(node.c, delta);
                }
                deriv = cmul(jac, deriv);
                if (!is_julia) {
                    deriv = deriv + node.b;
                }
                err = length(jac) * err
                    + (params.bla_epsilon + GAMMA) * (length(a_delta) + length(b_dc) + length(c_term));
                delta = cand;
                n = new_n;
                m = new_m;
                stepped = true;
                break;
            }
        }

        if (stepped) {
            continue;
        }

        // Pas direct δ' = (2Z + δ)·δ (+ dc). Traqueur : D' = 2zD (+ 1),
        // E' = E·(2|z| + E) + γ·(|2Z + δ|·|δ| + |dc|).
        let two_z_delta = 2.0 * z_m + delta;
        let z_norm = length(z_abs);
        deriv = 2.0 * cmul(z_abs, deriv);
        if (!is_julia) {
            deriv = deriv + vec2<f32>(1.0, 0.0);
        }
        err = err * (2.0 * z_norm + err) + GAMMA * (length(two_z_delta) * length(delta) + dc_abs);
        var next = cmul(two_z_delta, delta);
        if (!is_julia) {
            next = next + dc;
        }
        delta = next;
        n = n + 1u;
        m = m + 1u;

        if (delta.x != delta.x || delta.y != delta.y
            || abs(delta.x) > F32_BIG || abs(delta.y) > F32_BIG) {
            // Débordement f32 : état inexploitable → au CPU.
            write_pixel(idx, n, zref_at(m) + delta, 2u);
            return;
        }

        if (m + 1u >= params.ref_len) {
            if (params.cycle_period > 0u && m >= params.cycle_start) {
                m = params.cycle_start + (m - params.cycle_start) % params.cycle_period;
            } else if (params.atom_truncated != 0u) {
                let z_end = zref_at(m);
                delta = z_end + delta;
                err = err + U * (length(z_end) + length(delta));
                m = 0u;
            } else {
                write_pixel(idx, n, zref_at(m) + delta, 1u);
                return;
            }
        } else {
            let z_ref_m = zref_at(m);
            let z_curr = z_ref_m + delta;
            if (norm_sqr(z_curr) < norm_sqr(delta)) {
                delta = z_curr;
                err = err + U * (length(z_ref_m) + length(z_curr));
                m = 0u;
            }
        }
    }

    write_pixel(
        idx,
        params.iter_max,
        zref_at(m) + delta,
        select(0u, 2u, unreliable(err, deriv, pixel_size)),
    );
}
