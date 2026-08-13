//! Assembleur vidéo (G12 jalon 3) — interpolation des frames entre keyframes.
//!
//! Zoom continu `z(t) = 2^(v·t)` : la frame à la position `p = k + log2(z)`
//! échantillonne la keyframe `k` (fenêtre centrale 1/z) et blende la keyframe
//! `k+1` (fenêtre 2/z, là où elle couvre) avec un poids `z − 1` — le schéma
//! trilinear de DeepDrill (`shaders/scalers/trilinear.glsl`). Bilinéaire CPU
//! + rayon : déterministe, testable en CI, largement assez rapide (2 taps par
//! pixel de sortie).
//!
//! Sorties : pipe stdin vers ffmpeg (`-f rawvideo -pix_fmt rgb24`) ou dossier
//! de frames PNG (`--frames-dir`, fallback sans ffmpeg).
//!
//! Verrous jalon 3 : à `z = 1` (positions entières, supersample 1) la frame
//! émise est la keyframe colorisée **pixel-exacte** ; continuité au raccord
//! (`z → 2⁻` converge vers la keyframe suivante) ; assemblage déterministe.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rayon::prelude::*;

use super::spline::Dynamic;
use super::{
    keyframe_params, keyframe_path, lighting, map_fingerprint, render_dimensions, Manifest,
    validate_project_keyframes,
};
use crate::fractal::OutColoringMode;
use crate::io::fmap::{load_fmap, FractalMap};
use crate::io::png::colorize_to_rgb_with_extras;

pub struct AssembleOptions {
    /// Fichier vidéo de sortie (encodé par ffmpeg). Exclusif avec frames_dir.
    pub output: Option<PathBuf>,
    /// Dossier de frames PNG numérotées (fallback sans ffmpeg).
    pub frames_dir: Option<PathBuf>,
    /// Binaire ffmpeg (défaut "ffmpeg").
    pub ffmpeg: String,
}

#[derive(Clone, Copy, Debug)]
pub struct AssembleStats {
    pub frames: usize,
    pub duration_s: f64,
}

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

/// Garde-fou mémoire/temps pour un manifest accidentel ou hostile. Dix
/// millions de frames représentent déjà plus de 23 h à 120 fps.
const MAX_TIMELINE_SAMPLES: usize = 10_000_000;

/// Valide la timeline et retourne son nombre de frames sans allouer le
/// vecteur des positions. Utilisé dès `plan` pour échouer avant le rendu des
/// keyframes.
pub(crate) fn timeline_sample_count(
    n: u32,
    fps: u32,
    velocity: &Dynamic,
) -> Result<usize, String> {
    if fps == 0 {
        return Err("fps doit être > 0".into());
    }
    let samples = if let Some(v) = velocity.as_constant() {
        if v <= 0.0 {
            return Err(format!(
                "vélocité constante {v} ≤ 0 : la vidéo n'avancerait jamais (utilisez une spline pour les segments à reculons)"
            ));
        }
        let intervals = n as f64 * fps as f64 / v;
        if !intervals.is_finite() {
            return Err("durée de timeline non finie".into());
        }
        let total = intervals.floor();
        let reaches_end = total / fps as f64 * v >= n as f64;
        total + if reaches_end { 1.0 } else { 2.0 }
    } else {
        let t_end = velocity.end_time().ok_or("spline sans fin temporelle")?;
        let frames = t_end * fps as f64;
        if !frames.is_finite() || frames < 0.0 {
            return Err("durée de timeline non finie ou négative".into());
        }
        frames.round() + 1.0
    };
    if samples > MAX_TIMELINE_SAMPLES as f64 {
        return Err(format!(
            "timeline trop longue: {samples:.0} frames (maximum {MAX_TIMELINE_SAMPLES})"
        ));
    }
    Ok(samples as usize)
}

/// Positions `p ∈ [0, n]` (keyframe fractionnaire) de chaque frame de sortie.
///
/// * vélocité CONSTANTE `v` (> 0) : forme fermée `p = v·t`, durée `n/v`
///   (chemin exact — verrou spline plate == constante) ;
/// * vélocité SPLINE : durée = dernier nœud, intégration point-milieu
///   `p += v(t+dt/2)·dt`, clamp [0, n] (une vélocité négative recule, façon
///   wobble DeepDrill).
pub fn timeline(n: u32, fps: u32, velocity: &Dynamic) -> Result<Vec<f64>, String> {
    let sample_count = timeline_sample_count(n, fps, velocity)?;
    let n_f = n as f64;
    if let Some(v) = velocity.as_constant() {
        if v <= 0.0 {
            return Err(format!(
                "vélocité constante {v} ≤ 0 : la vidéo n'avancerait jamais (utilisez une spline pour les segments à reculons)"
            ));
        }
        let total = (n_f * fps as f64 / v).floor() as usize;
        let mut positions: Vec<f64> = Vec::with_capacity(sample_count);
        positions.extend((0..=total)
            .map(|f| ((f as f64 / fps as f64) * v).min(n_f))
        );
        if positions.last().copied().unwrap_or(0.0) < n_f {
            positions.push(n_f);
        }
        return Ok(positions);
    }
    let t_end = velocity.end_time().expect("spline ⇒ end_time");
    let dt = 1.0 / fps as f64;
    let frames = (t_end * fps as f64).round() as usize;
    let mut positions = Vec::with_capacity(sample_count);
    let mut p = 0.0f64;
    for f in 0..=frames {
        positions.push(p.clamp(0.0, n_f));
        let t = f as f64 * dt;
        p += velocity.eval(t + dt * 0.5) * dt;
    }
    Ok(positions)
}

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------

#[inline]
fn bilinear(src: &[u8], w: usize, h: usize, cx: f64, cy: f64) -> [f64; 3] {
    // Coordonnée pixel continue (centre du texel à l'entier), clampée AVANT le
    // floor (clamp-to-edge) : pour sx légèrement < 0, clamper seulement
    // l'indice de base laisserait un poids fractionnaire ≈ 1 sur le texel
    // VOISIN au lieu du bord (bug raccord z→2, pixel (0, y)).
    let sx = (cx * w as f64 - 0.5).clamp(0.0, (w - 1) as f64);
    let sy = (cy * h as f64 - 0.5).clamp(0.0, (h - 1) as f64);
    let x0 = sx.floor();
    let y0 = sy.floor();
    let fx = sx - x0;
    let fy = sy - y0;
    let xi = x0 as usize;
    let yi = y0 as usize;
    let xi1 = (xi + 1).min(w - 1);
    let yi1 = (yi + 1).min(h - 1);
    let p = |x: usize, y: usize| -> [f64; 3] {
        let o = (y * w + x) * 3;
        [src[o] as f64, src[o + 1] as f64, src[o + 2] as f64]
    };
    let (p00, p10, p01, p11) = (p(xi, yi), p(xi1, yi), p(xi, yi1), p(xi1, yi1));
    let mut out = [0.0f64; 3];
    for c in 0..3 {
        let top = p00[c] * (1.0 - fx) + p10[c] * fx;
        let bot = p01[c] * (1.0 - fx) + p11[c] * fx;
        out[c] = top * (1.0 - fy) + bot * fy;
    }
    out
}

/// Frame interpolée entre `curr` (keyframe k, échantillonnée à l'échelle `z ∈
/// [1, 2)`) et `next` (keyframe k+1, là où sa fenêtre couvre), poids `z − 1`.
/// `curr`/`next` sont des buffers RGB `src_w × src_h` (résolution supersample) ;
/// la sortie fait `out_w × out_h`.
pub fn interpolate_frame(
    curr: &[u8],
    next: Option<&[u8]>,
    src_w: usize,
    src_h: usize,
    out_w: usize,
    out_h: usize,
    z: f64,
) -> Vec<u8> {
    assert!((1.0..2.0 + 1e-9).contains(&z), "z hors [1,2): {z}");
    assert_eq!(curr.len(), src_w * src_h * 3);

    // Fast path identité : z = 1 et résolutions égales → copie EXACTE de la
    // keyframe (verrou pixel-exact aux positions entières ; évite le bruit
    // d'ulp du round-trip (x+0.5)/W·W du path général).
    if z == 1.0 && src_w == out_w && src_h == out_h {
        return curr.to_vec();
    }

    let mut out = vec![0u8; out_w * out_h * 3];
    out.par_chunks_mut(out_w * 3).enumerate().for_each(|(y, row)| {
        let ty = (y as f64 + 0.5) / out_h as f64;
        let cy = 0.5 + (ty - 0.5) / z;
        for x in 0..out_w {
            let tx = (x as f64 + 0.5) / out_w as f64;
            let cx = 0.5 + (tx - 0.5) / z;
            let mut c = bilinear(curr, src_w, src_h, cx, cy);
            if z > 1.0 {
                if let Some(next) = next {
                    // Fenêtre de la keyframe suivante (span/2, même centre).
                    let c2x = 2.0 * cx - 0.5;
                    let c2y = 2.0 * cy - 0.5;
                    if (0.0..=1.0).contains(&c2x) && (0.0..=1.0).contains(&c2y) {
                        let c2 = bilinear(next, src_w, src_h, c2x, c2y);
                        let a = z - 1.0;
                        for i in 0..3 {
                            c[i] = c[i] * (1.0 - a) + c2[i] * a;
                        }
                    }
                }
            }
            let o = x * 3;
            for i in 0..3 {
                row[o + i] = c[i].round().clamp(0.0, 255.0) as u8;
            }
        }
    });
    out
}

// ---------------------------------------------------------------------------
// Colorisation des keyframes
// ---------------------------------------------------------------------------

/// Colorise une map de keyframe avec les couleurs du MANIFEST (pas celles
/// baked dans la map — changer la palette du manifest recolore sans re-rendre)
/// + décalage de palette dynamique + éclairage optionnel (jalon 5).
pub fn colorize_keyframe(map: &FractalMap, manifest: &Manifest, palette_offset: f64) -> Result<Vec<u8>, String> {
    let mut p = map.params.clone();
    p.color_mode = manifest.color.palette;
    p.color_repeat = manifest.color.color_repeat.max(1);
    p.color_space = manifest.color.color_space;
    p.out_coloring_mode = OutColoringMode::from_cli_name(&manifest.color.outcoloring)
        .ok_or_else(|| format!("outcoloring invalide: '{}'", manifest.color.outcoloring))?;
    p.color_offset = palette_offset;
    // Le canal `distances` de la map (rendu avec `[fractal] distance_estimation`)
    // alimente les modes Distance*/DistanceAO/Distance3D — sans lui, le
    // manifest pourrait activer l'estimation de distance et le rendu retomber
    // silencieusement sur Smooth.
    let mut rgb = colorize_to_rgb_with_extras(
        &p,
        &map.iterations,
        &map.zs,
        map.distances.as_deref().unwrap_or(&[]),
        &[],
    );
    if manifest.lighting.enable {
        lighting::shade_rgb(
            &mut rgb,
            &map.iterations,
            &map.zs,
            p.width as usize,
            p.height as usize,
            p.iteration_max,
            manifest.lighting.alpha,
            manifest.lighting.beta,
        );
    }
    Ok(rgb)
}

/// Cache des keyframes : maps brutes + RGB colorisé (invalide si le décalage
/// de palette dynamique a changé depuis la colorisation).
struct KeyframeCache<'a> {
    project: &'a Path,
    manifest: &'a Manifest,
    maps: HashMap<u32, FractalMap>,
    rgb: HashMap<u32, (u64, std::sync::Arc<Vec<u8>>)>,
}

impl<'a> KeyframeCache<'a> {
    fn new(project: &'a Path, manifest: &'a Manifest) -> Self {
        Self { project, manifest, maps: HashMap::new(), rgb: HashMap::new() }
    }

    fn colorized(&mut self, k: u32, palette_offset: f64) -> Result<std::sync::Arc<Vec<u8>>, String> {
        let off_bits = palette_offset.to_bits();
        if let Some((bits, rgb)) = self.rgb.get(&k) {
            if *bits == off_bits {
                return Ok(rgb.clone());
            }
        }
        if !self.maps.contains_key(&k) {
            let path = keyframe_path(self.project, k);
            let map = load_fmap(&path)
                .map_err(|e| format!("keyframe {k} illisible ({}) : {e} — lancez `fractall-video render`", path.display()))?;
            // Géométrie de la map ≠ géométrie du manifest (dimensions ou
            // supersample modifiés après le rendu) : `interpolate_frame`
            // assertait sur la taille du buffer → PANIC. On refuse proprement
            // et on renvoie vers `render` (le CLI `assemble` ne vérifie
            // aucune empreinte, contrairement au studio GUI).
            let (exp_w, exp_h) = render_dimensions(self.manifest)?;
            if map.params.width != exp_w || map.params.height != exp_h {
                return Err(format!(
                    "keyframe {k} rendue en {}×{} mais le manifest attend {exp_w}×{exp_h} \
                     (image.width/height/supersample modifiés depuis le rendu) — \
                     relancez `fractall-video render`",
                    map.params.width, map.params.height
                ));
            }
            let expected = keyframe_params(self.manifest, k)?;
            if map_fingerprint(&map.params) != map_fingerprint(&expected) {
                return Err(format!(
                    "keyframe {k} ne correspond plus au manifest (cible, type, itérations ou canaux de calcul modifiés) — relancez `fractall-video render`"
                ));
            }
            self.maps.insert(k, map);
        }
        let rgb = std::sync::Arc::new(colorize_keyframe(&self.maps[&k], self.manifest, palette_offset)?);
        self.rgb.insert(k, (off_bits, rgb.clone()));
        Ok(rgb)
    }

    /// Garde les keyframes [k−1, k+2] (fenêtre glissante), libère le reste.
    fn evict_before(&mut self, k: u32) {
        let keep = k.saturating_sub(1)..=k + 2;
        self.maps.retain(|kk, _| keep.contains(kk));
        self.rgb.retain(|kk, _| keep.contains(kk));
    }
}

// ---------------------------------------------------------------------------
// Sinks
// ---------------------------------------------------------------------------

enum FrameSink {
    /// Sous-processus ffmpeg + chemin du fichier de sortie (mémorisé pour
    /// pouvoir supprimer le .mp4 partiel en cas d'annulation — un x264 tué
    /// en cours d'écriture est illisible).
    Ffmpeg(Child, PathBuf),
    Frames(PathBuf),
}

impl FrameSink {
    fn write(&mut self, index: usize, frame: &[u8], w: u32, h: u32) -> Result<(), String> {
        match self {
            FrameSink::Ffmpeg(child, _) => child
                .stdin
                .as_mut()
                .expect("stdin ffmpeg piped")
                .write_all(frame)
                .map_err(|e| format!("écriture vers ffmpeg: {e}")),
            FrameSink::Frames(dir) => {
                let img = image::RgbImage::from_raw(w, h, frame.to_vec())
                    .expect("dimensions frame cohérentes");
                let path = dir.join(format!("frame_{index:06}.png"));
                img.save(&path).map_err(|e| format!("écriture {}: {e}", path.display()))
            }
        }
    }

    fn finish(self) -> Result<(), String> {
        match self {
            FrameSink::Ffmpeg(mut child, out) => {
                drop(child.stdin.take()); // EOF → ffmpeg finalise
                match child.wait() {
                    Ok(status) if status.success() => Ok(()),
                    Ok(status) => {
                        let _ = std::fs::remove_file(&out);
                        Err(format!("ffmpeg a échoué ({status})"))
                    }
                    Err(e) => {
                        let _ = std::fs::remove_file(&out);
                        Err(format!("attente ffmpeg: {e}"))
                    }
                }
            }
            FrameSink::Frames(_) => Ok(()),
        }
    }

    /// Arrêt sur annulation : tue ffmpeg et supprime le .mp4 partiel
    /// (illisible) ; les frames PNG déjà écrites restent (inoffensives).
    fn abort(self) {
        match self {
            FrameSink::Ffmpeg(mut child, out) => {
                drop(child.stdin.take());
                let _ = child.kill();
                let _ = child.wait();
                let _ = std::fs::remove_file(&out);
            }
            FrameSink::Frames(_) => {}
        }
    }
}

/// Après une réussite, le dossier doit représenter exactement cette vidéo.
/// On ne touche qu'aux fichiers produits par nous (`frame_<index>.png`) et
/// seulement à l'ancien suffixe situé après la dernière frame courante.
fn remove_stale_frame_suffix(dir: &Path, frame_count: usize) -> Result<(), String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("lecture du dossier frames {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("lecture d'une entrée frames: {e}"))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(index) = name
            .strip_prefix("frame_")
            .and_then(|s| s.strip_suffix(".png"))
            .filter(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|s| s.parse::<usize>().ok())
        else {
            continue;
        };
        if index >= frame_count {
            std::fs::remove_file(entry.path())
                .map_err(|e| format!("suppression de la frame obsolète {name}: {e}"))?;
        }
    }
    Ok(())
}

fn spawn_ffmpeg(opts: &AssembleOptions, out: &Path, w: u32, h: u32, fps: u32) -> Result<Child, String> {
    Command::new(&opts.ffmpeg)
        .args([
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", &format!("{w}x{h}"),
            "-r", &fps.to_string(),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
        ])
        .arg(out)
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|e| {
            format!(
                "impossible de lancer '{}' ({e}) — installez ffmpeg ou utilisez --frames-dir",
                opts.ffmpeg
            )
        })
}

// ---------------------------------------------------------------------------
// Assemblage
// ---------------------------------------------------------------------------

/// Progression d'assemblage : une frame écrite (`frame` 0-based sur `total`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AssembleFrameEvent {
    pub frame: usize,
    pub total: usize,
    pub keyframe: u32,
    pub z: f64,
}

/// Issue d'un assemblage annulable. Sur `Cancelled`, le .mp4 partiel a été
/// supprimé (illisible) ; les frames PNG déjà écrites restent.
#[derive(Debug)]
pub enum AssembleOutcome {
    Complete(AssembleStats),
    Cancelled { frames_written: usize },
}

/// Assemble la vidéo d'un projet dont les keyframes sont rendues.
///
/// Enveloppe no-cancel de `assemble_project_with_progress` reproduisant la
/// sortie console historique (CLI `fractall-video assemble`).
pub fn assemble_project(
    project: &Path,
    opts: &AssembleOptions,
) -> Result<AssembleStats, Box<dyn std::error::Error>> {
    let outcome = assemble_project_with_progress(
        project,
        opts,
        &Arc::new(AtomicBool::new(false)),
        &mut |ev: AssembleFrameEvent| {
            if ev.frame % 100 == 0 || ev.frame + 1 == ev.total {
                println!(
                    "[assemble] frame {}/{} (keyframe {}, z={:.3})",
                    ev.frame + 1,
                    ev.total,
                    ev.keyframe,
                    ev.z
                );
            }
        },
    )?;
    match outcome {
        AssembleOutcome::Complete(stats) => Ok(stats),
        // Inatteignable sans cancel externe — sémantique historique conservée.
        AssembleOutcome::Cancelled { .. } => Err("assemblage annulé".into()),
    }
}

/// Version annulable + progression (studio GUI, G12 jalon 6). `cancel` est
/// vérifié à chaque frame ; `progress` est appelé après CHAQUE frame écrite
/// (le consommateur throttle s'il veut). Sur annulation ou erreur, le sink
/// est proprement abandonné (ffmpeg tué + .mp4 partiel supprimé).
pub fn assemble_project_with_progress(
    project: &Path,
    opts: &AssembleOptions,
    cancel: &Arc<AtomicBool>,
    progress: &mut dyn FnMut(AssembleFrameEvent),
) -> Result<AssembleOutcome, Box<dyn std::error::Error>> {
    let manifest = Manifest::load(&project.join("manifest.toml"))?;
    let n = validate_project_keyframes(&manifest)
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    let fps = manifest.video.fps;
    if fps == 0 {
        return Err("video.fps doit être > 0".into());
    }
    if !manifest.color.palette_offset.is_finite() {
        return Err("color.palette_offset doit être fini".into());
    }
    if !manifest.lighting.alpha.is_finite() || !manifest.lighting.beta.is_finite() {
        return Err("lighting.alpha/beta doivent être finis".into());
    }
    let (out_w, out_h) = (manifest.image.width, manifest.image.height);
    let (src_w, src_h) = render_dimensions(&manifest)
        .map(|(w, h)| (w as usize, h as usize))?;
    keyframe_params(&manifest, 0)
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    let velocity = Dynamic::parse(&manifest.video.velocity)
        .map_err(|e| format!("video.velocity: {e}"))?;
    let palette_offset: Dynamic = match &manifest.dynamics.palette_offset {
        Some(s) => Dynamic::parse(s).map_err(|e| format!("dynamics.palette_offset: {e}"))?,
        None => Dynamic::Constant(manifest.color.palette_offset),
    };
    let positions = timeline(n, fps, &velocity)?;

    let mut sink = match (&opts.output, &opts.frames_dir) {
        (Some(out), None) => {
            FrameSink::Ffmpeg(spawn_ffmpeg(opts, out, out_w, out_h, fps)?, out.clone())
        }
        (None, Some(dir)) => {
            std::fs::create_dir_all(dir)?;
            FrameSink::Frames(dir.clone())
        }
        _ => return Err("exactement une sortie requise : -o video.mp4 OU --frames-dir DIR".into()),
    };

    let mut cache = KeyframeCache::new(project, &manifest);
    let total = positions.len();
    let mut written = 0usize;
    for (f, &p) in positions.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            sink.abort();
            return Ok(AssembleOutcome::Cancelled { frames_written: written });
        }
        let k = (p.floor() as u32).min(n);
        let frac = p - k as f64;
        let z = 2f64.powf(frac); // frac = 0 → z = 1 exactement
        let t = f as f64 / fps as f64;
        let off = palette_offset.eval(t);

        // Erreurs de colorisation/écriture : abandonner le sink AVANT de
        // propager (sinon un ffmpeg orphelin attend son stdin indéfiniment).
        let curr = match cache.colorized(k, off) {
            Ok(c) => c,
            Err(e) => {
                sink.abort();
                return Err(e.into());
            }
        };
        let next = if z > 1.0 && k + 1 <= n {
            match cache.colorized(k + 1, off) {
                Ok(c) => Some(c),
                Err(e) => {
                    sink.abort();
                    return Err(e.into());
                }
            }
        } else {
            None
        };
        cache.evict_before(k);

        let frame = interpolate_frame(
            &curr,
            next.as_deref().map(|v| v.as_slice()),
            src_w,
            src_h,
            out_w as usize,
            out_h as usize,
            z,
        );
        if let Err(e) = sink.write(f, &frame, out_w, out_h) {
            sink.abort();
            return Err(e.into());
        }
        written += 1;
        progress(AssembleFrameEvent { frame: f, total, keyframe: k, z });
    }
    sink.finish()?;
    if let Some(dir) = &opts.frames_dir {
        remove_stale_frame_suffix(dir, total)?;
    }
    Ok(AssembleOutcome::Complete(AssembleStats {
        frames: total,
        duration_s: total.saturating_sub(1) as f64 / fps as f64,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::{default_params_for_type, ColorSpace, FractalType};

    fn checker(w: usize, h: usize, seed: u8) -> Vec<u8> {
        (0..w * h * 3)
            .map(|i| ((i as u32).wrapping_mul(97).wrapping_add(seed as u32) % 256) as u8)
            .collect()
    }

    /// Verrou jalon 3 : à z = 1 (même résolution), frame == keyframe
    /// colorisée, pixel-exact.
    #[test]
    fn identity_at_z1() {
        let (w, h) = (32usize, 24usize);
        let curr = checker(w, h, 7);
        let frame = interpolate_frame(&curr, None, w, h, w, h, 1.0);
        assert_eq!(frame, curr);
    }

    /// La map ne fige que les canaux de calcul : l'espace du gradient vient
    /// du manifest courant, afin qu'un ré-assemblage puisse le modifier sans
    /// recalculer les keyframes.
    #[test]
    fn colorize_keyframe_uses_manifest_color_space() {
        let mut params = default_params_for_type(FractalType::Mandelbrot, 3, 1);
        params.iteration_max = 100;
        params.color_space = ColorSpace::Rgb;
        let map = FractalMap {
            params: params.clone(),
            iterations: vec![1, 15, 80],
            zs: vec![num_complex::Complex64::new(2.0, 1.0); 3],
            distances: None,
        };
        let mut manifest = Manifest::default();
        manifest.color.color_space = ColorSpace::Lch;

        let actual = colorize_keyframe(&map, &manifest, 0.0).unwrap();
        params.color_mode = manifest.color.palette;
        params.color_repeat = manifest.color.color_repeat;
        params.color_space = ColorSpace::Lch;
        params.out_coloring_mode = OutColoringMode::Smooth;
        let expected =
            colorize_to_rgb_with_extras(&params, &map.iterations, &map.zs, &[], &[]);
        assert_eq!(actual, expected);
    }

    /// Verrou jalon 3 : continuité au raccord — à z → 2⁻ la frame converge
    /// vers la keyframe suivante (celle qu'on prendra à z = 1 juste après).
    #[test]
    fn continuity_at_keyframe_boundary() {
        let (w, h) = (32usize, 24usize);
        let curr = checker(w, h, 3);
        let next = checker(w, h, 111);
        let frame = interpolate_frame(&curr, Some(&next), w, h, w, h, 2.0 - 1e-9);
        let max_diff = frame
            .iter()
            .zip(next.iter())
            .map(|(&a, &b)| (a as i32 - b as i32).abs())
            .max()
            .unwrap();
        assert!(max_diff <= 2, "raccord discontinu : max diff {max_diff}");
    }

    /// Verrou jalon 3 : déterminisme — deux exécutions bit-identiques
    /// (rayon ne change pas les valeurs, juste l'ordre).
    #[test]
    fn interpolation_is_deterministic() {
        let (w, h) = (48usize, 32usize);
        let curr = checker(w, h, 42);
        let next = checker(w, h, 130);
        let a = interpolate_frame(&curr, Some(&next), w, h, w, h, 1.37);
        let b = interpolate_frame(&curr, Some(&next), w, h, w, h, 1.37);
        assert_eq!(a, b);
    }

    /// Timeline constante : n/v secondes, dernière position == n, monotone.
    #[test]
    fn timeline_constant_velocity() {
        let v = Dynamic::parse("1.0").unwrap();
        let pos = timeline(3, 30, &v).unwrap();
        assert_eq!(pos.len(), 91); // 3 s à 30 fps + frame initiale
        assert_eq!(*pos.last().unwrap(), 3.0);
        assert!(pos.windows(2).all(|w| w[0] <= w[1]), "positions monotones");
        assert_eq!(pos[0], 0.0);

        // Vélocité 0.5 : deux fois plus long.
        let v2 = Dynamic::parse("0.5").unwrap();
        assert_eq!(timeline(3, 30, &v2).unwrap().len(), 181);
        // Vélocité nulle/négative constante : refus.
        assert!(timeline(3, 30, &Dynamic::parse("0").unwrap()).is_err());
        assert!(timeline(3, 30, &Dynamic::parse("-1").unwrap()).is_err());
        assert!(
            timeline(3, 120, &Dynamic::parse("0.000000001").unwrap()).is_err(),
            "une timeline démesurée doit être refusée avant allocation"
        );
    }

    /// Bout en bout (jalons 2+3) : plan→render→assemble sur un projet réel
    /// minuscule. Verrous : la frame 0 (z = 1, supersample 1) == la keyframe 0
    /// colorisée PIXEL-EXACTE ; nombre de frames de la timeline ; assemblage
    /// DÉTERMINISTE (deux runs bit-identiques).
    #[test]
    fn assemble_end_to_end_frames_dir() {
        let dir = std::env::temp_dir()
            .join(format!("fractall_assemble_e2e_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut m = crate::video::Manifest::default();
        m.image.width = 24;
        m.image.height = 18;
        m.fractal.iterations = 150;
        m.location.zoom = "4".into(); // 2 segments → 3 keyframes
        m.video.keyframes = crate::video::keyframe_count(&m.location.zoom).unwrap();
        m.video.fps = 5;
        m.video.velocity = "1.0".into(); // 2 s → 11 frames
        m.save(&dir.join("manifest.toml")).unwrap();
        crate::video::render_project(&dir).unwrap();

        let frames_a = dir.join("frames_a");
        let opts = AssembleOptions {
            output: None,
            frames_dir: Some(frames_a.clone()),
            ffmpeg: "ffmpeg".into(),
        };
        let stats = assemble_project(&dir, &opts).unwrap();
        assert_eq!(stats.frames, 11, "2 s à 5 fps + frame initiale");
        assert_eq!(stats.duration_s, 2.0, "durée = intervalles/fps");

        // Frame 0 == keyframe 0 colorisée (couleurs du manifest), pixel-exact.
        let map0 = load_fmap(&keyframe_path(&dir, 0)).unwrap();
        let expected = colorize_keyframe(&map0, &m, 0.0).unwrap();
        let frame0 = image::open(frames_a.join("frame_000000.png")).unwrap().to_rgb8();
        assert_eq!(frame0.as_raw(), &expected, "frame 0 doit être la keyframe 0 exacte");
        // Dernière frame == keyframe finale exacte (p termine à n entier).
        let map_last = load_fmap(&keyframe_path(&dir, m.video.keyframes)).unwrap();
        let expected_last = colorize_keyframe(&map_last, &m, 0.0).unwrap();
        let last = image::open(frames_a.join("frame_000010.png")).unwrap().to_rgb8();
        assert_eq!(last.as_raw(), &expected_last, "dernière frame == keyframe finale");

        // Déterminisme : second assemblage bit-identique.
        let frames_b = dir.join("frames_b");
        let opts_b = AssembleOptions {
            output: None,
            frames_dir: Some(frames_b.clone()),
            ffmpeg: "ffmpeg".into(),
        };
        std::fs::create_dir_all(&frames_b).unwrap();
        std::fs::write(frames_b.join("frame_999999.png"), b"ancienne frame").unwrap();
        std::fs::write(frames_b.join("notes.txt"), b"keep").unwrap();
        assemble_project(&dir, &opts_b).unwrap();
        assert!(!frames_b.join("frame_999999.png").exists(), "suffixe obsolète supprimé");
        assert!(frames_b.join("notes.txt").exists(), "fichier étranger conservé");
        for f in [0usize, 4, 10] {
            let a = std::fs::read(frames_a.join(format!("frame_{f:06}.png"))).unwrap();
            let b = std::fs::read(frames_b.join(format!("frame_{f:06}.png"))).unwrap();
            assert_eq!(a, b, "frame {f} non déterministe");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn failed_encoder_removes_partial_output() {
        let out = std::env::temp_dir()
            .join(format!("fractall_failed_encoder_{}.mp4", std::process::id()));
        std::fs::write(&out, b"partial").unwrap();
        let child = Command::new("false").stdin(Stdio::piped()).spawn().unwrap();
        let err = FrameSink::Ffmpeg(child, out.clone()).finish().unwrap_err();
        assert!(err.contains("échoué"));
        assert!(!out.exists(), "sortie partielle supprimée");
    }

    /// Régression : un manifest dont les dimensions ont changé APRÈS le rendu
    /// des keyframes faisait paniquer `interpolate_frame` (assert sur la
    /// taille du buffer). Le CLI `assemble` ne vérifie aucune empreinte
    /// (contrairement au studio GUI) — l'erreur doit être propre et renvoyer
    /// vers `render`.
    #[test]
    fn assemble_rejects_keyframes_of_wrong_size() {
        let dir = tiny_rendered_project("wrongsize");
        // Maps rendues en 16×12 ; on double la géométrie dans le manifest.
        let mut m = crate::video::Manifest::load(&dir.join("manifest.toml")).unwrap();
        m.image.width = 32;
        m.image.height = 24;
        m.save(&dir.join("manifest.toml")).unwrap();

        let opts = AssembleOptions {
            output: None,
            frames_dir: Some(dir.join("frames")),
            ffmpeg: "ffmpeg".into(),
        };
        let err = match assemble_project(&dir, &opts) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("assemblage avec des keyframes de mauvaise taille doit échouer"),
        };
        assert!(err.contains("render"), "erreur explicite attendue, eu : {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn assemble_rejects_stale_keyframe_parameters() {
        let dir = tiny_rendered_project("staleparams");
        let mut m = crate::video::Manifest::load(&dir.join("manifest.toml")).unwrap();
        m.fractal.iterations += 100;
        m.save(&dir.join("manifest.toml")).unwrap();

        let opts = AssembleOptions {
            output: None,
            frames_dir: Some(dir.join("frames")),
            ffmpeg: "ffmpeg".into(),
        };
        let err = assemble_project(&dir, &opts).unwrap_err().to_string();
        assert!(err.contains("ne correspond plus"), "erreur explicite: {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn assemble_rejects_desynchronized_keyframe_count() {
        let dir = tiny_rendered_project("badcount");
        let mut m = crate::video::Manifest::load(&dir.join("manifest.toml")).unwrap();
        m.video.keyframes = 99;
        m.save(&dir.join("manifest.toml")).unwrap();
        let opts = AssembleOptions {
            output: None,
            frames_dir: Some(dir.join("frames")),
            ffmpeg: "ffmpeg".into(),
        };
        let err = assemble_project(&dir, &opts).unwrap_err().to_string();
        assert!(err.contains("attendu 2"), "erreur explicite: {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn assemble_rejects_invalid_presentation_values() {
        for (tag, mutate) in [("fps", 0u8), ("offset", 1u8), ("light", 2u8)] {
            let dir = tiny_rendered_project(tag);
            let mut m = crate::video::Manifest::load(&dir.join("manifest.toml")).unwrap();
            match mutate {
                0 => m.video.fps = 0,
                1 => m.color.palette_offset = f64::NAN,
                2 => m.lighting.alpha = f64::INFINITY,
                _ => unreachable!(),
            }
            m.save(&dir.join("manifest.toml")).unwrap();
            let frames = dir.join("frames");
            let opts = AssembleOptions {
                output: None,
                frames_dir: Some(frames.clone()),
                ffmpeg: "ffmpeg".into(),
            };
            assert!(assemble_project(&dir, &opts).is_err(), "devait refuser {tag}");
            assert!(!frames.exists(), "aucun sink créé pour {tag}");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// Timeline spline : durée = dernier nœud ; une spline PLATE donne la
    /// même durée que la constante équivalente (chemin exact partagé).
    #[test]
    fn timeline_spline_duration_and_flat_equivalence() {
        let sp = Dynamic::parse("0/1,3/1").unwrap(); // plate → constante 1.0
        let flat = timeline(3, 30, &sp).unwrap();
        let cst = timeline(3, 30, &Dynamic::parse("1.0").unwrap()).unwrap();
        assert_eq!(flat, cst, "spline plate == constante, bit-identique");

        // Spline non plate : durée du dernier nœud, clamp [0, n].
        let wob = Dynamic::parse("0/2,1/2,2/0.5,4/0.5").unwrap();
        let pos = timeline(3, 10, &wob).unwrap();
        assert_eq!(pos.len(), 41); // 4 s à 10 fps + initiale
        assert!(pos.iter().all(|&p| (0.0..=3.0).contains(&p)));
        assert!(pos.windows(2).all(|w| w[1] >= w[0]), "vélocité > 0 ⇒ monotone");
    }

    /// Petit projet réel rendu (2 segments, 16×12, 5 fps → 11 frames).
    fn tiny_rendered_project(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("fractall_asm_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut m = crate::video::Manifest::default();
        m.image.width = 16;
        m.image.height = 12;
        m.fractal.iterations = 80;
        m.location.zoom = "4".into();
        m.video.keyframes = crate::video::keyframe_count("4").unwrap();
        m.video.fps = 5;
        m.save(&dir.join("manifest.toml")).unwrap();
        crate::video::render_project(&dir).unwrap();
        dir
    }

    /// Verrou hooks (G12 jalon 6) : `progress` est appelé pour CHAQUE frame,
    /// indices 0-based contigus, total cohérent avec les stats.
    #[test]
    fn assemble_with_progress_reports_every_frame() {
        let dir = tiny_rendered_project("progress");
        let frames = dir.join("frames");
        let opts = AssembleOptions {
            output: None,
            frames_dir: Some(frames),
            ffmpeg: "ffmpeg".into(),
        };
        let mut events: Vec<AssembleFrameEvent> = Vec::new();
        let outcome = assemble_project_with_progress(
            &dir,
            &opts,
            &Arc::new(AtomicBool::new(false)),
            &mut |ev| events.push(ev),
        )
        .unwrap();
        let AssembleOutcome::Complete(stats) = outcome else {
            panic!("Complete attendu");
        };
        assert_eq!(events.len(), stats.frames);
        assert!(events
            .iter()
            .enumerate()
            .all(|(i, e)| e.frame == i && e.total == stats.frames));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Verrou annulation : cancel après la 3e frame → Cancelled{3}, les 3 PNG
    /// écrits restent, la suite n'existe pas, pas de panic.
    #[test]
    fn assemble_with_progress_cancel_stops_cleanly() {
        let dir = tiny_rendered_project("cancel");
        let frames = dir.join("frames");
        let opts = AssembleOptions {
            output: None,
            frames_dir: Some(frames.clone()),
            ffmpeg: "ffmpeg".into(),
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let trigger = cancel.clone();
        let mut count = 0usize;
        let outcome = assemble_project_with_progress(&dir, &opts, &cancel, &mut |_| {
            count += 1;
            if count == 3 {
                trigger.store(true, Ordering::Relaxed);
            }
        })
        .unwrap();
        let AssembleOutcome::Cancelled { frames_written } = outcome else {
            panic!("Cancelled attendu");
        };
        assert_eq!(frames_written, 3);
        for f in 0..3 {
            assert!(frames.join(format!("frame_{f:06}.png")).exists(), "frame {f}");
        }
        assert!(!frames.join("frame_000003.png").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
