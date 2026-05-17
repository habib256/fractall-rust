//! Export EXR raw au format Fraktaler-3 (N + NF channels, IterationsBias=1024).
//!
//! Permet la comparaison apples-to-apples avec F3 via `scripts/compare_f3.py`.
//! Le format imite `fraktaler-3-3.1/src/image_raw.cc` :
//!
//! - **Channel `N`** (UINT) : `iter + Nbias` si pixel échappé, sinon `0xFFFFFFFF`.
//! - **Channel `NF`** (FLOAT) : smooth fraction `1 - log(log(|Z|²) / log(ER²)) / log(degree)`
//!   clampé à `[0, 1]`, mis à 0 pour pixels intérieurs.
//! - **Attribut `IterationsBias`** (Int) : `Nbias = 1024`.
//! - **Attribut `Iterations`** (Int) : `iter_max`.
//!
//! Suppose `iter_max + Nbias < u32::MAX` (vrai pour tout notre corpus actuel).
//! Si on dépasse, on tronquera silencieusement à `u32::MAX` (TODO: support N0+N1).
//!
//! Référence F3 : `hybrid.cc:350` pour NF, `image_raw.cc:166` pour le layout EXR.

use std::path::Path;

use exr::prelude::{
    AnyChannel, AnyChannels, AttributeValue, Encoding, FlatSamples, Image, Layer,
    LayerAttributes, SmallVec, Text, Vec2, WritableImage,
};
use num_complex::Complex64;

pub const NBIAS: u32 = 1024;

/// Calcule la valeur NF (smooth fraction) façon F3 hybrid.cc:350.
///
/// `bailout_sq` est le rayon d'échappement au carré (ex: 625.0 pour ER=25).
/// `degree` est le degré polynomial de la dernière phase (2 pour Mandelbrot/Burning Ship).
pub fn nf_f3(z: Complex64, iter: u32, iter_max: u32, bailout_sq: f64, degree: f64) -> f32 {
    if iter >= iter_max {
        return 0.0;
    }
    let z2 = z.norm_sqr();
    if !z2.is_finite() || z2 < bailout_sq {
        return 0.0;
    }
    let num = z2.ln();
    let den = bailout_sq.ln();
    if !num.is_finite() || den <= 0.0 {
        return 0.0;
    }
    let r = num / den;
    if r <= 0.0 || !r.is_finite() {
        return 0.0;
    }
    let nf = 1.0 - (r.ln() / degree.ln());
    if !nf.is_finite() {
        return 0.0;
    }
    nf.clamp(0.0, 1.0) as f32
}

/// Écrit un EXR au format F3 (channels N + NF + attributs Iterations / IterationsBias).
pub fn save_iterations_exr(
    path: &Path,
    width: usize,
    height: usize,
    iterations: &[u32],
    zs: &[Complex64],
    iter_max: u32,
    bailout_sq: f64,
    degree: f64,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    debug_assert_eq!(iterations.len(), width * height);
    debug_assert_eq!(zs.len(), width * height);

    // Construit les deux buffers — pixel order = ligne par ligne, identique à F3.
    let mut n_buf: Vec<u32> = Vec::with_capacity(width * height);
    let mut nf_buf: Vec<f32> = Vec::with_capacity(width * height);
    for i in 0..(width * height) {
        let iter = iterations[i];
        let n_val: u32 = if iter >= iter_max {
            u32::MAX
        } else {
            let biased = iter as u64 + NBIAS as u64;
            if biased >= u32::MAX as u64 {
                u32::MAX - 1
            } else {
                biased as u32
            }
        };
        n_buf.push(n_val);
        nf_buf.push(nf_f3(zs[i], iter, iter_max, bailout_sq, degree));
    }

    let size = Vec2(width, height);

    let n_channel = AnyChannel::new(
        "N",
        FlatSamples::U32(n_buf),
    );
    let nf_channel = AnyChannel::new(
        "NF",
        FlatSamples::F32(nf_buf),
    );

    let mut attrs = LayerAttributes::default();
    attrs.other.insert(
        Text::from("Iterations"),
        AttributeValue::I32(iter_max as i32),
    );
    attrs.other.insert(
        Text::from("IterationsBias"),
        AttributeValue::I32(NBIAS as i32),
    );
    attrs.other.insert(
        Text::from("fraktall_source"),
        AttributeValue::Text(Text::from("fractall-rust --export-iterations")),
    );

    let layer = Layer::new(
        size,
        attrs,
        Encoding::default(),
        AnyChannels::sort({
            let mut v: SmallVec<[AnyChannel<FlatSamples>; 4]> = SmallVec::new();
            v.push(n_channel);
            v.push(nf_channel);
            v
        }),
    );

    let image = Image::from_layer(layer);
    image.write().to_file(path)?;
    Ok(())
}
