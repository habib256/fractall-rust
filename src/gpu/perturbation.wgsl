// Kernel perturbation GPU — port F3-strict du pixel loop CPU en F64 NATIF.
//
// Sémantique alignée sur le chemin CPU (`bytecode/pixel_loop.rs::
// iterate_pixel_unified_mandelbrot`, jugé par `fractall-quality gpu-suite`
// contre le GMP pur) :
// - Compteur d'itération `n` et index de référence `m` SÉPARÉS : le rebase
//   remet `m := 0` mais préserve `n` (le kernel legacy confondait les deux —
//   chaque rebase faussait le compte d'itérations).
// - Rebasing F3 strict : après un pas direct, si `|Z[m]+δ|² < |δ|²` alors
//   `δ := Z[m]+δ, m := 0`. Pas d'hystérésis, pas de glitch detection
//   Pauldelbrot (remplacée par le rebasing proactif, cf. hybrid.cc:295-308).
// - Garde anti-over-skip BLA : un saut l ≥ 2 dont le point d'arrivée
//   `Z[m']+δ'` est déjà échappé est rejeté (single-step pour trouver
//   l'itération d'évasion exacte, cf. pixel_loop.rs).
// - Bailout en tête de boucle sur `|Z[m]+δ|²` (mêmes comptes que le CPU).
//
// Précision : f64 NATIF (SHADER_F64, requis — sans le feature le host
// retombe sur le CPU). Le double-float 2×f32 (Dekker/two_sum) est IMPOSSIBLE
// sur la stack WGSL→naga→SPIR-V actuelle : sans décorations NoContraction/
// precise, `fma(a,b,-(a·b))` peut être évalué non-fusionné (→ 0) et les
// transformations sans-erreur sont réassociées — mesuré sur RTX 4060 Ti
// (NVIDIA) ET llvmpipe : le df64 s'effondre silencieusement en f32 en
// contexte de boucle (div 0.067 vs GMP = stats f32 pures), alors que chaque
// primitive isolée passe. Diagnostic reproductible : `cargo run --release
// --bin df64_gpu_probe`. Revisiter 2×f32 (perf ~10× f64 NVIDIA) via
// passthrough SPIR-V décoré — cf. TODO G9.4.
//
// La mantisse f32 seule (24 b) est insuffisante : p99 146-237 vs GMP sur
// l'échelle seahorse (le kernel legacy ne tenait WARN qu'en re-rendant les
// pixels flagués en GMP côté CPU).
//
// Le caller Rust garantit `ref_len-1 >= iter_max` (sinon fallback CPU), donc
// `m <= n < iter_max <= ref_len-1` — pas de rebase-at-end nécessaire.
// Le tier HDR (exposant par pixel, deep zoom > ~1e290) = TODO G9.4.

struct Params {
    // Offset centre-vue − centre-référence (center − cref) et span, en paires
    // hi/lo f32 (reconstruction f64 exacte à 2^-48, suffisant pour le mapping
    // pixel→c ; évite les f64 en uniform buffer). L'offset est calculé en f64
    // côté host (soustraction catastrophique en f32).
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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct PixelOut {
    iter: u32,
    z_re: f32,
    z_im: f32,
    flags: u32,
};

// Nœud BLA conforme (f64 natif) : δ' = A·δ + B·dc + C·δ².
struct BlaNode {
    a: vec2<f64>,
    b: vec2<f64>,
    c: vec2<f64>,
    validity: f64,
    _pad: f64,
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
@group(0) @binding(4) var<storage, read> z_ref: array<vec2<f64>>;
@group(0) @binding(5) var<storage, read> reuse_mask: array<u32>;

fn cmul(a: vec2<f64>, b: vec2<f64>) -> vec2<f64> {
    return vec2<f64>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn norm_sqr(a: vec2<f64>) -> f64 {
    return a.x * a.x + a.y * a.y;
}

// |c + d| - |c| stable — perturbation Burning Ship (cf. delta_form.rs).
fn diffabs(c: f64, d: f64) -> f64 {
    let cd = c + d;
    let c2d = 2.0 * c + d;
    if (c >= 0.0) {
        if (cd >= 0.0) {
            return d;
        } else {
            return -c2d;
        }
    } else {
        if (cd > 0.0) {
            return c2d;
        } else {
            return -d;
        }
    }
}

fn zref_at(m: u32) -> vec2<f64> {
    // ⚠️ clamp CLAUDE.md : après un pas, m peut valoir ref_len.
    return z_ref[min(m, params.ref_len - 1u)];
}

fn write_pixel(idx: u32, iter: u32, z: vec2<f64>) {
    out_pixels[idx].iter = iter;
    out_pixels[idx].z_re = f32(z.x);
    out_pixels[idx].z_im = f32(z.y);
    out_pixels[idx].flags = 0u;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let idx = gid.y * params.width + gid.x;
    // reuse_mask == 0 : pixel déjà rempli par la passe précédente (progressive).
    if (reuse_mask[idx] == 0u) {
        return;
    }

    // Mapping pixel→dc en f64 : ratio d'abord puis multiplication (même
    // formule que le CPU), span/offset reconstruits depuis les paires hi/lo.
    let span_x = f64(params.span_x_hi) + f64(params.span_x_lo);
    let span_y = f64(params.span_y_hi) + f64(params.span_y_lo);
    let offset_x = f64(params.offset_x_hi) + f64(params.offset_x_lo);
    let offset_y = f64(params.offset_y_hi) + f64(params.offset_y_lo);
    let x_ratio = (f64(gid.x) + 0.5) / f64(params.width) - 0.5;
    let y_ratio = (f64(gid.y) + 0.5) / f64(params.height) - 0.5;
    let dc = vec2<f64>(x_ratio * span_x + offset_x, y_ratio * span_y + offset_y);

    let is_julia = params.fractal_kind == 1u;
    let is_burning_ship = params.fractal_kind == 2u;
    // Julia : le pixel entre par δ₀ = dc (z₀ = pixel, c = seed constant).
    // Mandelbrot-like : δ₀ = 0, dc entre par la formule.
    var delta = vec2<f64>();
    if (is_julia) {
        delta = dc;
    }
    var n: u32 = 0u;
    var m: u32 = 0u;
    let bailout_sqr = f64(params.bailout) * f64(params.bailout);
    let use_bla = !is_burning_ship && params.bla_levels > 0u;
    let use_series = params.series_order >= 2u;
    let series_threshold_sqr = f64(params.series_threshold) * f64(params.series_threshold);

    while (n < params.iter_max) {
        // Bailout en tête : |Z[m] + δ|² ≥ bailout² → escape à l'itération n.
        let z_m = zref_at(m);
        let z_abs = z_m + delta;
        if (norm_sqr(z_abs) >= bailout_sqr) {
            write_pixel(idx, n, z_abs);
            return;
        }

        // Étape 1 : essai BLA (Mandelbrot/Julia, table conforme par niveaux,
        // skip 2^level, nœud indexé par m). Niveaux décroissants = plus grand
        // saut valide d'abord.
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
                // δ' = A·δ (+ B·dc pour Mandelbrot) (+ C·δ² série ordre 2)
                var cand = cmul(node.a, delta);
                if (!is_julia) {
                    cand = cand + cmul(node.b, dc);
                }
                if (use_series && delta_norm_sqr < series_threshold_sqr) {
                    cand = cand + cmul(node.c, cmul(delta, delta));
                }
                // Garde anti-over-skip : saut multi-pas linéarisé autour de la
                // référence, aveugle à l'évasion propre du pixel. Si le point
                // d'arrivée est déjà échappé, rejeter et single-stepper.
                let z_end = zref_at(new_m) + cand;
                if (skip >= 2u && norm_sqr(z_end) >= bailout_sqr) {
                    break;
                }
                delta = cand;
                n = new_n;
                m = new_m;
                stepped = true;
                break;
            }
        }

        if (stepped) {
            // Saut BLA accepté : pas de check de rebase (mirror CPU), le
            // bailout de tête gère l'état d'arrivée.
            continue;
        }

        // Étape 2 : pas perturbation direct.
        if (is_burning_ship) {
            // δ_re' = (2·Z_re + δ_re)·δ_re − (2·Z_im + δ_im)·δ_im + dc_re
            // δ_im' = 2·diffabs(Z_re·Z_im, Z_re·δ_im + δ_re·(Z_im + δ_im)) + dc_im
            let new_re = (2.0 * z_m.x + delta.x) * delta.x
                - (2.0 * z_m.y + delta.y) * delta.y + dc.x;
            let new_im = 2.0 * diffabs(
                z_m.x * z_m.y,
                z_m.x * delta.y + delta.x * (z_m.y + delta.y),
            ) + dc.y;
            delta = vec2<f64>(new_re, new_im);
        } else {
            // Mandelbrot/Julia : δ' = (2·Z + δ)·δ (+ dc pour Mandelbrot).
            var next = cmul(2.0 * z_m + delta, delta);
            if (!is_julia) {
                next = next + dc;
            }
            delta = next;
        }
        n = n + 1u;
        m = m + 1u;

        // Garde NaN/Inf (mirror CPU : sortie avec l'état courant, pas de flag).
        if (delta.x != delta.x || delta.y != delta.y
            || abs(delta.x) > 1.0e300 || abs(delta.y) > 1.0e300) {
            write_pixel(idx, n, zref_at(m) + delta);
            return;
        }

        // Étape 3 : rebase F3 strict — |Z[m]+δ|² < |δ|² → δ := Z[m]+δ, m := 0.
        let z_curr = zref_at(m) + delta;
        if (norm_sqr(z_curr) < norm_sqr(delta)) {
            delta = z_curr;
            m = 0u;
        }
    }

    // n == iter_max : pixel intérieur.
    write_pixel(idx, params.iter_max, zref_at(m) + delta);
}
