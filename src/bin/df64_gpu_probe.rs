//! Sonde df64 GPU (diagnostic G9.4) : mesure l'EXACTITUDE réelle de two_sum,
//! fma() et du split de Dekker sur l'adaptateur wgpu de la machine.
//!
//! `cargo run --release --bin df64_gpu_probe`
//!
//! Attendus si l'arithmétique est saine :
//! - two_sum(1, 2^-30)   → err = 2^-30 (≠ 0)
//! - fma-two_prod        → err = 2^-40 exact
//! - split-two_prod      → err = 2^-40 exact (cassé si le driver contracte)

const PROBE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> zref: array<vec4<f32>>;

fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    return vec2<f32>(s, err);
}

fn two_prod_fma(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    let err = fma(a, b, -p);
    return vec2<f32>(p, err);
}

fn df_split(a: f32) -> vec2<f32> {
    let t = 4097.0 * a;
    let hi = t - (t - a);
    return vec2<f32>(hi, a - hi);
}

fn two_prod_split(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    let aa = df_split(a);
    let bb = df_split(b);
    let err = ((aa.x * bb.x - p) + aa.x * bb.y + aa.y * bb.x) + aa.y * bb.y;
    return vec2<f32>(p, err);
}

// Barrière anti-optimisation : round-trip bitcast u32.
fn opaque(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x));
}

fn two_sum_opaque(a: f32, b: f32) -> vec2<f32> {
    let s = opaque(a + b);
    let bb = opaque(s - a);
    let err = (a - opaque(s - bb)) + (b - bb);
    return vec2<f32>(s, err);
}

// Garde anti-réassociation : multiplication par u (== 1.0 au runtime, opaque
// pour le compilateur) — (a+b)*u n'a plus d'identité algébrique.
fn two_sum_guard(a: f32, b: f32, u: f32) -> vec2<f32> {
    let s = (a + b) * u;
    let bb = (s - a) * u;
    let err = (a - (s - bb)) + (b - bb);
    return vec2<f32>(s, err);
}

fn qts(a: f32, b: f32, u: f32) -> vec2<f32> {
    let s = (a + b) * u;
    return vec2<f32>(s, b - (s - a));
}
fn df_add(a: vec2<f32>, b: vec2<f32>, u: f32) -> vec2<f32> {
    let s = two_sum_guard(a.x, b.x, u);
    return qts(s.x, s.y + a.y + b.y, u);
}
fn df_mul(a: vec2<f32>, b: vec2<f32>, u: f32) -> vec2<f32> {
    let p = two_prod_fma(a.x, b.x);
    return qts(p.x, p.y + a.x * b.y + a.y * b.x, u);
}
fn df_mul_f32(a: vec2<f32>, b: f32, u: f32) -> vec2<f32> {
    let p = two_prod_fma(a.x, b);
    return qts(p.x, p.y + a.y * b, u);
}
fn cdf_add(a: vec4<f32>, b: vec4<f32>, u: f32) -> vec4<f32> {
    return vec4<f32>(df_add(a.xy, b.xy, u), df_add(a.zw, b.zw, u));
}
fn cdf_mul(a: vec4<f32>, b: vec4<f32>, u: f32) -> vec4<f32> {
    let m = df_mul(a.zw, b.zw, u);
    let re = df_add(df_mul(a.xy, b.xy, u), vec2<f32>(-m.x, -m.y), u);
    let im = df_add(df_mul(a.xy, b.zw, u), df_mul(a.zw, b.xy, u), u);
    return vec4<f32>(re, im);
}
fn cdf_ns(a: vec4<f32>) -> f32 {
    let re = a.x + a.y;
    let im = a.z + a.w;
    return re * re + im * im;
}

@compute @workgroup_size(1)
fn main() {
    // Valeurs runtime (empêche le constant folding naga) : out[8]=1.0,
    // out[9]=2^-30, out[10]=1+2^-20 écrits par le host avant dispatch.
    let one = out[8];
    let tiny = out[9];
    let x = out[10];

    let ts = two_sum(one, tiny);
    out[0] = ts.x;
    out[1] = ts.y;

    let pf = two_prod_fma(x, x);
    out[2] = pf.x;
    out[3] = pf.y;

    let ps = two_prod_split(x, x);
    out[4] = ps.x;
    out[5] = ps.y;

    // fma brut : fma(x,x,-x*x) — 0.0 si le driver l'évalue non-fusionné.
    out[6] = fma(x, x, -(x * x));

    let tso = two_sum_opaque(one, tiny);
    out[7] = tso.y;

    let u = out[11];
    let tsg = two_sum_guard(one, tiny, u);
    out[12] = tsg.y;

    // ---- Boucle perturbation df64 complète (mirror du kernel, garde u) ----
    let dc = vec4<f32>(out[16], out[17], out[18], out[19]);
    let iter_max = u32(out[20]);
    let ref_len = u32(out[21]);
    var delta = vec4<f32>(0.0);
    var n: u32 = 0u;
    var m: u32 = 0u;
    let bailout_sqr = 625.0;
    loop {
        if (n >= iter_max) { break; }
        let z_m = zref[min(m, ref_len - 1u)];
        let z_abs = cdf_add(z_m, delta, u);
        if (cdf_ns(z_abs) >= bailout_sqr) { break; }
        let two_z = vec4<f32>(df_mul_f32(z_m.xy, 2.0, u), df_mul_f32(z_m.zw, 2.0, u));
        delta = cdf_add(cdf_mul(cdf_add(two_z, delta, u), delta, u), dc, u);
        n = n + 1u;
        m = m + 1u;
        let z_curr = cdf_add(zref[min(m, ref_len - 1u)], delta, u);
        if (cdf_ns(z_curr) < cdf_ns(delta)) {
            delta = z_curr;
            m = 0u;
        }
        if (n <= 8u) {
            let base = 40u + (n - 1u) * 4u;
            out[base] = delta.x;
            out[base + 1u] = delta.y;
            out[base + 2u] = delta.z;
            out[base + 3u] = delta.w;
        }
    }
    // Helpers isolés : mêmes fonctions que la boucle.
    // df_add((1, 2^-30), (2^-10, 2^-42)) : lo attendu ≈ 2^-30 + 2^-42.
    let da = df_add(vec2<f32>(one, tiny), vec2<f32>(out[13], out[14]), u);
    out[29] = da.y;
    // df_mul((1+2^-20, 2^-30), (1+2^-20, 2^-30)) : lo ≈ 2^-40 + 2^-29.
    let dm = df_mul(vec2<f32>(x, tiny), vec2<f32>(x, tiny), u);
    out[30] = dm.y;
    // two_prod_fma seul, en aval d'une valeur de boucle (anti-folding check).
    let tp2 = two_prod_fma(x + tiny, x);
    out[31] = tp2.y;

    // Un PAS de perturbation complet hors boucle, entrées de la référence :
    // δ' = (2·Z[3] + dc)·dc + dc (état après δ₀=0 → δ₁=dc).
    let z3 = zref[3];
    let tz = vec4<f32>(df_mul_f32(z3.xy, 2.0, u), df_mul_f32(z3.zw, 2.0, u));
    let one_step = cdf_add(cdf_mul(cdf_add(tz, dc, u), dc, u), dc, u);
    out[32] = one_step.x;
    out[33] = one_step.y;
    out[34] = one_step.z;
    out[35] = one_step.w;

    out[24] = f32(n);
    out[25] = delta.x;
    out[26] = delta.y;
    out[27] = delta.z;
    out[28] = delta.w;
}
"#;

const F64_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> outv: array<f32>;
@group(0) @binding(1) var<storage, read> zref: array<vec2<f64>>;

@compute @workgroup_size(1)
fn main() {
    // Boucle perturbation en f64 natif — mêmes entrées que la boucle df64.
    let dc = vec2<f64>(f64(outv[16]) + f64(outv[17]), f64(outv[18]) + f64(outv[19]));
    let iter_max = u32(outv[20]);
    let ref_len = u32(outv[21]);
    var delta = vec2<f64>();
    var n: u32 = 0u;
    var m: u32 = 0u;
    let bailout_sqr = f64(625.0);
    loop {
        if (n >= iter_max) { break; }
        let z_m = zref[min(m, ref_len - 1u)];
        let z_abs = z_m + delta;
        if (z_abs.x * z_abs.x + z_abs.y * z_abs.y >= bailout_sqr) { break; }
        let t = 2.0 * z_m + delta;
        delta = vec2<f64>(t.x * delta.x - t.y * delta.y, t.x * delta.y + t.y * delta.x) + dc;
        n = n + 1u;
        m = m + 1u;
        let z_curr = zref[min(m, ref_len - 1u)] + delta;
        if (z_curr.x * z_curr.x + z_curr.y * z_curr.y < delta.x * delta.x + delta.y * delta.y) {
            delta = z_curr;
            m = 0u;
        }
    }
    outv[24] = f32(n);
    outv[25] = f32(delta.x);
    outv[26] = f32(delta.y);
}
"#;

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: std::env::var("PROBE_FALLBACK").is_ok(),
        })
        .await
        .expect("pas d'adaptateur GPU");
    let info = adapter.get_info();
    println!("[probe] adaptateur : {} ({:?})", info.name, info.backend);
    println!("[probe] SHADER_F64 : {}", adapter.features().contains(wgpu::Features::SHADER_F64));

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .expect("device");

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("df64-probe"),
        source: wgpu::ShaderSource::Wgsl(PROBE_WGSL.into()),
    });

    // Orbite de référence f64 (centre seahorse ; chaotique-mais-partagée,
    // seule compte l'identité GPU vs mirror CPU).
    let (cx, cy) = (-0.743643887037158f64, 0.131825904205311f64);
    let iter_max = 5000u32;
    let mut zref64 = Vec::with_capacity(iter_max as usize + 1);
    let (mut zr, mut zi) = (0.0f64, 0.0f64);
    zref64.push((zr, zi));
    for _ in 0..iter_max {
        let nr = zr * zr - zi * zi + cx;
        let ni = 2.0 * zr * zi + cy;
        zr = nr;
        zi = ni;
        zref64.push((zr, zi));
    }
    let split = |v: f64| -> (f32, f32) {
        let hi = v as f32;
        (hi, (v - hi as f64) as f32)
    };
    let zref_flat: Vec<f32> = zref64
        .iter()
        .flat_map(|&(r, i)| {
            let (rh, rl) = split(r);
            let (ih, il) = split(i);
            [rh, rl, ih, il]
        })
        .collect();
    let dc64 = (1.3e-6f64, -7.7e-7f64);
    let (dcxh, dcxl) = split(dc64.0);
    let (dcyh, dcyl) = split(dc64.1);

    let mut init = [0.0f32; 80];
    init[8] = 1.0;
    init[9] = (2.0f32).powi(-30);
    init[10] = 1.0 + (2.0f32).powi(-20);
    init[11] = 1.0;
    init[16] = dcxh;
    init[17] = dcxl;
    init[18] = dcyh;
    init[19] = dcyl;
    init[20] = 50.0; // court : juge l'arithmétique, pas le chaos
    init[13] = (2.0f32).powi(-10);
    init[14] = (2.0f32).powi(-42);
    init[21] = (zref64.len()) as f32;

    let buf = wgpu::util::DeviceExt::create_buffer_init(
        &device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("probe-out"),
            contents: bytemuck::cast_slice(&init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        },
    );
    let zref_buf = wgpu::util::DeviceExt::create_buffer_init(
        &device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("probe-zref"),
            contents: bytemuck::cast_slice(&zref_flat),
            usage: wgpu::BufferUsages::STORAGE,
        },
    );
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe-read"),
        size: 320,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: zref_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&buf, 0, &readback, 0, 320);
    queue.submit(Some(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let vals: &[f32] = bytemuck::cast_slice(&data);

    let exp_tiny = (2.0f32).powi(-30);
    let exp_err = (2.0f64).powi(-40);
    println!("two_sum(1, 2^-30)      : s={:e} err={:e} (attendu {:e}) → {}",
        vals[0], vals[1], exp_tiny, if vals[1] == exp_tiny { "OK" } else { "CASSÉ" });
    println!("two_prod fma (1+2^-20)²: p={:e} err={:e} (attendu {:e}) → {}",
        vals[2], vals[3], exp_err, if (vals[3] as f64 - exp_err).abs() < exp_err * 0.01 { "OK" } else { "CASSÉ" });
    println!("two_prod split         : p={:e} err={:e} (attendu {:e}) → {}",
        vals[4], vals[5], exp_err, if (vals[5] as f64 - exp_err).abs() < exp_err * 0.01 { "OK" } else { "CASSÉ" });
    println!("fma(x,x,-(x*x))        : {:e} (0.0 = fma non fusionné)", vals[6]);
    println!("two_sum bitcast-opaque : err={:e} (attendu {:e}) → {}",
        vals[7], exp_tiny, if vals[7] == exp_tiny { "OK" } else { "CASSÉ" });
    println!("two_sum garde u=1.0    : err={:e} (attendu {:e}) → {}",
        vals[12], exp_tiny, if vals[12] == exp_tiny { "OK" } else { "CASSÉ" });

    // ---- Mirror CPU de la boucle df64 (mêmes primitives, f32 IEEE) ----
    let (n_cpu, d_cpu) = cpu_df64_loop(&zref_flat, [dcxh, dcxl, dcyh, dcyl], 50);
    for k in 1..=8u32 {
        let (_, dk) = cpu_df64_loop(&zref_flat, [dcxh, dcxl, dcyh, dcyl], k);
        let b = (40 + (k - 1) * 4) as usize;
        let gk = [vals[b], vals[b + 1], vals[b + 2], vals[b + 3]];
        let rel = {
            let gr = gk[0] as f64 + gk[1] as f64;
            let gi = gk[2] as f64 + gk[3] as f64;
            let cr = dk[0] as f64 + dk[1] as f64;
            let ci = dk[2] as f64 + dk[3] as f64;
            (((gr - cr).powi(2) + (gi - ci).powi(2)).sqrt()) / (cr * cr + ci * ci).sqrt()
        };
        println!("iter {} : GPU {:?} CPU {:?} rel={:.2e}", k, gk, dk, rel);
    }
    // Vérité f64 de la même récurrence (référence df64-splittée re-fusionnée).
    let mut dre = 0.0f64;
    let mut dim = 0.0f64;
    let mut mm = 0usize;
    for _ in 0..50 {
        let i = mm.min(zref_flat.len() / 4 - 1) * 4;
        let zr = zref_flat[i] as f64 + zref_flat[i + 1] as f64;
        let zi = zref_flat[i + 2] as f64 + zref_flat[i + 3] as f64;
        let tr = 2.0 * zr + dre;
        let ti = 2.0 * zi + dim;
        let nre = tr * dre - ti * dim + dc64.0;
        let nim = tr * dim + ti * dre + dc64.1;
        dre = nre;
        dim = nim;
        mm += 1;
        let i2 = mm.min(zref_flat.len() / 4 - 1) * 4;
        let zr2 = zref_flat[i2] as f64 + zref_flat[i2 + 1] as f64;
        let zi2 = zref_flat[i2 + 2] as f64 + zref_flat[i2 + 3] as f64;
        let (cr, ci) = (zr2 + dre, zi2 + dim);
        if cr * cr + ci * ci < dre * dre + dim * dim {
            dre = cr;
            dim = ci;
            mm = 0;
        }
    }
    let n_gpu = vals[24] as u32;
    let d_gpu = [vals[25], vals[26], vals[27], vals[28]];
    println!("boucle df64 5000 iters : GPU n={} δ={:?}", n_gpu, d_gpu);
    println!("                         CPU n={} δ={:?} → {}", n_cpu, d_cpu,
        if n_cpu == n_gpu && d_cpu == d_gpu { "IDENTIQUE" } else { "DIVERGE" });
    let g_re = d_gpu[0] as f64 + d_gpu[1] as f64;
    let g_im = d_gpu[2] as f64 + d_gpu[3] as f64;
    let c_re = d_cpu[0] as f64 + d_cpu[1] as f64;
    let c_im = d_cpu[2] as f64 + d_cpu[3] as f64;
    let mag = (dre * dre + dim * dim).sqrt();
    println!("df_add lo : {:e} (attendu ~{:e})", vals[29], (2.0f64).powi(-30) + (2.0f64).powi(-42));
    println!("df_mul lo : {:e} (attendu ~{:e})", vals[30], (2.0f64).powi(-40) + (2.0f64).powi(-29));
    println!("two_prod_fma aval : {:e} (0 = foldé)", vals[31]);
    {
        // Mirror CPU du pas isolé.
        let i = 3 * 4;
        let z3 = [zref_flat[i], zref_flat[i + 1], zref_flat[i + 2], zref_flat[i + 3]];
        let dcv = [dcxh, dcxl, dcyh, dcyl];
        let ts = |a: f32, b: f32| -> (f32, f32) {
            let s = a + b;
            let bb = s - a;
            (s, (a - (s - bb)) + (b - bb))
        };
        let qts = |a: f32, b: f32| -> (f32, f32) { let s = a + b; (s, b - (s - a)) };
        let tp = |a: f32, b: f32| -> (f32, f32) { let p = a * b; (p, f32::mul_add(a, b, -p)) };
        let df_add = |a: (f32, f32), b: (f32, f32)| -> (f32, f32) { let s = ts(a.0, b.0); qts(s.0, s.1 + a.1 + b.1) };
        let df_mul = |a: (f32, f32), b: (f32, f32)| -> (f32, f32) { let p = tp(a.0, b.0); qts(p.0, p.1 + a.0 * b.1 + a.1 * b.0) };
        let df_mul_s = |a: (f32, f32), b: f32| -> (f32, f32) { let p = tp(a.0, b); qts(p.0, p.1 + a.1 * b) };
        let cadd = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
            let re = df_add((a[0], a[1]), (b[0], b[1]));
            let im = df_add((a[2], a[3]), (b[2], b[3]));
            [re.0, re.1, im.0, im.1]
        };
        let cmul = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
            let m = df_mul((a[2], a[3]), (b[2], b[3]));
            let re = df_add(df_mul((a[0], a[1]), (b[0], b[1])), (-m.0, -m.1));
            let im = df_add(df_mul((a[0], a[1]), (b[2], b[3])), df_mul((a[2], a[3]), (b[0], b[1])));
            [re.0, re.1, im.0, im.1]
        };
        let tzr = df_mul_s((z3[0], z3[1]), 2.0);
        let tzi = df_mul_s((z3[2], z3[3]), 2.0);
        let os = cadd(cmul(cadd([tzr.0, tzr.1, tzi.0, tzi.1], dcv), dcv), dcv);
        let g = [vals[32], vals[33], vals[34], vals[35]];
        println!("pas isolé : GPU {:?}", g);
        println!("            CPU {:?} → {}", os, if g == os { "IDENTIQUE" } else { "DIVERGE" });
    }
    println!("erreur rel vs f64 après 50 iters : GPU {:.3e} | CPU-mirror {:.3e} (f32 pur serait ~1e-7)",
        (((g_re - dre).powi(2) + (g_im - dim).powi(2)).sqrt()) / mag,
        (((c_re - dre).powi(2) + (c_im - dim).powi(2)).sqrt()) / mag);

    // ---- Boucle f64 natif (si SHADER_F64) ----
    if adapter.features().contains(wgpu::Features::SHADER_F64) {
        let (dev64, q64) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("probe-f64"),
                required_features: wgpu::Features::SHADER_F64,
                ..Default::default()
            })
            .await
            .expect("device f64");
        let module = dev64.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("f64-probe"),
            source: wgpu::ShaderSource::Wgsl(F64_WGSL.into()),
        });
        let zref64_flat: Vec<f64> = zref64.iter().flat_map(|&(r, i)| [r, i]).collect();
        let zbuf = wgpu::util::DeviceExt::create_buffer_init(
            &dev64,
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&zref64_flat),
                usage: wgpu::BufferUsages::STORAGE,
            },
        );
        let obuf = wgpu::util::DeviceExt::create_buffer_init(
            &dev64,
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            },
        );
        let rb = dev64.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 320,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lay = dev64.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl = dev64.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&lay)],
            immediate_size: 0,
        });
        let pipe = dev64.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bg = dev64.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &lay,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: obuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: zbuf.as_entire_binding() },
            ],
        });
        let mut enc = dev64.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        enc.copy_buffer_to_buffer(&obuf, 0, &rb, 0, 320);
        q64.submit(Some(enc.finish()));
        let sl = rb.slice(..);
        let (tx2, rx2) = std::sync::mpsc::channel();
        sl.map_async(wgpu::MapMode::Read, move |r| tx2.send(r).unwrap());
        let _ = dev64.poll(wgpu::PollType::wait_indefinitely());
        rx2.recv().unwrap().unwrap();
        let d2 = sl.get_mapped_range();
        let v2: &[f32] = bytemuck::cast_slice(&d2);
        let gr = v2[25] as f64;
        let gi = v2[26] as f64;
        println!(
            "boucle f64 natif 50 iters : n={} δ=({:e},{:e}) rel vs f64-CPU = {:.3e} (attendu ~1e-7 : sortie castée f32)",
            v2[24] as u32, gr, gi,
            (((gr - dre).powi(2) + (gi - dim).powi(2)).sqrt()) / mag
        );
    } else {
        println!("boucle f64 natif : SHADER_F64 indisponible sur cet adaptateur");
    }
}

// Mirror CPU exact de la boucle WGSL (df64 gardé, u=1.0 littéral :
// le compilateur Rust ne réassocie pas les flottants).
fn cpu_df64_loop(zref_flat: &[f32], dc: [f32; 4], iter_max: u32) -> (u32, [f32; 4]) {
    let ts = |a: f32, b: f32| -> (f32, f32) {
        let s = a + b;
        let bb = s - a;
        ((s), (a - (s - bb)) + (b - bb))
    };
    let qts = |a: f32, b: f32| -> (f32, f32) {
        let s = a + b;
        (s, b - (s - a))
    };
    let tp = |a: f32, b: f32| -> (f32, f32) {
        let p = a * b;
        (p, f32::mul_add(a, b, -p))
    };
    let df_add = |a: (f32, f32), b: (f32, f32)| -> (f32, f32) {
        let s = ts(a.0, b.0);
        qts(s.0, s.1 + a.1 + b.1)
    };
    let df_mul = |a: (f32, f32), b: (f32, f32)| -> (f32, f32) {
        let p = tp(a.0, b.0);
        qts(p.0, p.1 + a.0 * b.1 + a.1 * b.0)
    };
    let df_mul_s = |a: (f32, f32), b: f32| -> (f32, f32) {
        let p = tp(a.0, b);
        qts(p.0, p.1 + a.1 * b)
    };
    let cadd = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
        let re = df_add((a[0], a[1]), (b[0], b[1]));
        let im = df_add((a[2], a[3]), (b[2], b[3]));
        [re.0, re.1, im.0, im.1]
    };
    let cmul = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
        let m = df_mul((a[2], a[3]), (b[2], b[3]));
        let re = df_add(df_mul((a[0], a[1]), (b[0], b[1])), (-m.0, -m.1));
        let im = df_add(df_mul((a[0], a[1]), (b[2], b[3])), df_mul((a[2], a[3]), (b[0], b[1])));
        [re.0, re.1, im.0, im.1]
    };
    let ns = |a: [f32; 4]| -> f32 {
        let re = a[0] + a[1];
        let im = a[2] + a[3];
        re * re + im * im
    };
    let zref = |m: usize| -> [f32; 4] {
        let i = m.min(zref_flat.len() / 4 - 1) * 4;
        [zref_flat[i], zref_flat[i + 1], zref_flat[i + 2], zref_flat[i + 3]]
    };
    let mut delta = [0.0f32; 4];
    let mut n = 0u32;
    let mut m = 0usize;
    while n < iter_max {
        let z_m = zref(m);
        if ns(cadd(z_m, delta)) >= 625.0 {
            break;
        }
        let tzr = df_mul_s((z_m[0], z_m[1]), 2.0);
        let tzi = df_mul_s((z_m[2], z_m[3]), 2.0);
        let two_z = [tzr.0, tzr.1, tzi.0, tzi.1];
        delta = cadd(cmul(cadd(two_z, delta), delta), dc);
        n += 1;
        m += 1;
        let z_curr = cadd(zref(m), delta);
        if ns(z_curr) < ns(delta) {
            delta = z_curr;
            m = 0;
        }
    }
    (n, delta)
}
