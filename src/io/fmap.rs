//! Format map persistant `.fmap` (G12 jalon 1) — sidecar calcul/rendu.
//!
//! Sépare le CALCUL (buffers bruts du dispatcher : itérations, z finaux,
//! distances) du RENDU (colorisation). Un `.fmap` permet de recoloriser une
//! vue sans la recalculer (`fractall-cli --from-map`), et sert de format
//! d'échange des keyframes du pipeline vidéo (G12 jalons 2-3).
//!
//! Inspiration : le `DrillMap` de DeepDrill (calcul une fois, colorise à
//! volonté). Contenu minimal ici : exactement les buffers que
//! `colorize_to_rgb` consomme (`io/png.rs:23`), en f64/u32 pleins — le
//! round-trip est **bit-identique**, c'est le verrou du format. Les canaux
//! sont compressés indépendamment (zlib via flate2).
//!
//! Layout binaire (little-endian) :
//! ```text
//! magic  b"FMAP"
//! u32    version (= 1)
//! u32    longueur du JSON params
//! bytes  JSON FractalParams (même sérialisation que le chunk tEXt PNG,
//!        coordonnées HP incluses)
//! u32    nombre de canaux
//! par canal :
//!   u8   id (0 = iterations u32, 1 = zs 2×f64, 2 = distances f64)
//!   u64  taille décompressée (octets)
//!   u64  taille compressée (octets)
//!   bytes flux zlib
//! ```
//!
//! `iterations` et `zs` sont obligatoires ; `distances` est optionnel (rempli
//! seulement quand le rendu a produit une estimation de distance). L'AA
//! multi-sample est hors périmètre : sa sortie est une moyenne RGB, pas des
//! canaux itération.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use num_complex::Complex64;

use crate::fractal::FractalParams;
use crate::io::atomic::write_atomic;

const MAGIC: &[u8; 4] = b"FMAP";
const VERSION: u32 = 1;

const CHANNEL_ITERATIONS: u8 = 0;
const CHANNEL_ZS: u8 = 1;
const CHANNEL_DISTANCES: u8 = 2;

/// Map persistée : paramètres complets + buffers bruts du dispatcher.
pub struct FractalMap {
    /// Paramètres de la vue. Les coordonnées HP doivent être renseignées par
    /// l'appelant (comme pour le PNG : `center_x_hp` etc. remplies au save).
    pub params: FractalParams,
    /// Compte d'itérations par pixel (row-major, `width × height`).
    pub iterations: Vec<u32>,
    /// z final par pixel (à l'échappement).
    pub zs: Vec<Complex64>,
    /// Distances estimées (si le rendu les a produites — path perturbation
    /// avec `enable_distance_estimation`). `None` sinon.
    pub distances: Option<Vec<f64>>,
}

fn compress(data: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::fast());
    enc.write_all(data)?;
    enc.finish()
}

fn decompress(data: &[u8], expected_len: usize) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let dec = ZlibDecoder::new(data);
    let mut out = Vec::new();
    out.try_reserve_exact(expected_len)
        .map_err(|e| format!("canal trop grand pour être décompressé: {e}"))?;
    // `take` empêche un flux zlib mensonger de faire croître le Vec au-delà
    // de la taille annoncée avant que nous puissions le refuser.
    dec.take((expected_len as u64).saturating_add(1)).read_to_end(&mut out)?;
    if out.len() != expected_len {
        return Err(format!(
            "canal corrompu : {} octets décompressés, {} attendus",
            out.len(),
            expected_len
        )
        .into());
    }
    Ok(out)
}

fn u32s_to_bytes(v: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn f64s_to_bytes(v: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 8);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn zs_to_bytes(v: &[Complex64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 16);
    for z in v {
        out.extend_from_slice(&z.re.to_le_bytes());
        out.extend_from_slice(&z.im.to_le_bytes());
    }
    out
}

fn bytes_to_u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bytes_to_f64s(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

fn bytes_to_zs(b: &[u8]) -> Vec<Complex64> {
    let f = bytes_to_f64s(b);
    f.chunks_exact(2).map(|p| Complex64::new(p[0], p[1])).collect()
}

fn write_channel(w: &mut impl Write, id: u8, raw: &[u8]) -> std::io::Result<()> {
    let compressed = compress(raw)?;
    w.write_all(&[id])?;
    w.write_all(&(raw.len() as u64).to_le_bytes())?;
    w.write_all(&(compressed.len() as u64).to_le_bytes())?;
    w.write_all(&compressed)?;
    Ok(())
}

/// Sauvegarde une map. Les buffers doivent être cohérents avec
/// `params.width × params.height` ; une incohérence retourne `Err` plutôt que
/// de faire paniquer un appelant de bibliothèque.
pub fn save_fmap(map: &FractalMap, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if map.params.width == 0 || map.params.height == 0 {
        return Err("fmap : dimensions nulles".into());
    }
    let n = (map.params.width as usize)
        .checked_mul(map.params.height as usize)
        .ok_or("fmap : dimensions trop grandes")?;
    if map.iterations.len() != n {
        return Err("fmap : taille iterations invalide".into());
    }
    if map.zs.len() != n {
        return Err("fmap : taille zs invalide".into());
    }
    if let Some(ref d) = map.distances {
        if d.len() != n {
            return Err("fmap : taille distances invalide".into());
        }
    }

    let params_json = serde_json::to_vec(&map.params)?;
    let n_channels: u32 = 2 + map.distances.is_some() as u32;

    write_atomic(path, |file| {
        let mut w = std::io::BufWriter::new(file);
        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        w.write_all(&(params_json.len() as u32).to_le_bytes())?;
        w.write_all(&params_json)?;
        w.write_all(&n_channels.to_le_bytes())?;
        write_channel(&mut w, CHANNEL_ITERATIONS, &u32s_to_bytes(&map.iterations))?;
        write_channel(&mut w, CHANNEL_ZS, &zs_to_bytes(&map.zs))?;
        if let Some(ref d) = map.distances {
            write_channel(&mut w, CHANNEL_DISTANCES, &f64s_to_bytes(d))?;
        }
        w.flush()?;
        Ok(())
    })
}

fn read_exact_vec(r: &mut impl Read, len: usize) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    buf.try_reserve_exact(len).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::OutOfMemory, format!("allocation fmap: {e}"))
    })?;
    buf.resize(len, 0);
    r.read_exact(&mut buf)?;
    Ok(buf)
}

/// Charge une map. Refus propre (Err) si le fichier n'est pas un `.fmap`
/// valide : magic/version inconnus, JSON illisible, canaux manquants ou de
/// taille incohérente avec `width × height`.
pub fn load_fmap(path: &Path) -> Result<FractalMap, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut r = std::io::BufReader::new(file);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(format!("{} : pas un fichier fmap (magic invalide)", path.display()).into());
    }
    let mut b4 = [0u8; 4];
    r.read_exact(&mut b4)?;
    let version = u32::from_le_bytes(b4);
    if version != VERSION {
        return Err(format!(
            "{} : version fmap {} non supportée (attendu {})",
            path.display(),
            version,
            VERSION
        )
        .into());
    }
    r.read_exact(&mut b4)?;
    let json_len = u32::from_le_bytes(b4) as usize;
    if json_len as u64 > file_len {
        return Err("fmap : longueur JSON supérieure au fichier".into());
    }
    let params_json = read_exact_vec(&mut r, json_len)?;
    let params: FractalParams = serde_json::from_slice(&params_json)?;
    if params.width == 0 || params.height == 0 {
        return Err("fmap : dimensions nulles".into());
    }
    let n = (params.width as usize)
        .checked_mul(params.height as usize)
        .ok_or("fmap : dimensions trop grandes")?;

    r.read_exact(&mut b4)?;
    let n_channels = u32::from_le_bytes(b4);
    if n_channels > 64 {
        return Err(format!("fmap : trop de canaux ({n_channels})").into());
    }

    let mut iterations: Option<Vec<u32>> = None;
    let mut zs: Option<Vec<Complex64>> = None;
    let mut distances: Option<Vec<f64>> = None;
    for _ in 0..n_channels {
        let mut id = [0u8; 1];
        r.read_exact(&mut id)?;
        let mut b8 = [0u8; 8];
        r.read_exact(&mut b8)?;
        let raw_len_u64 = u64::from_le_bytes(b8);
        r.read_exact(&mut b8)?;
        let comp_len_u64 = u64::from_le_bytes(b8);
        if comp_len_u64 > file_len {
            return Err("fmap : canal compressé plus grand que le fichier".into());
        }
        let comp_len = usize::try_from(comp_len_u64)
            .map_err(|_| "fmap : canal compressé trop grand pour cette plateforme")?;
        let expected = match id[0] {
            CHANNEL_ITERATIONS => Some(n.checked_mul(4).ok_or("fmap : canal iterations trop grand")?),
            CHANNEL_ZS => Some(n.checked_mul(16).ok_or("fmap : canal zs trop grand")?),
            CHANNEL_DISTANCES => Some(n.checked_mul(8).ok_or("fmap : canal distances trop grand")?),
            _ => None,
        };
        let Some(raw_len) = expected else {
            // Compatibilité future : ignorer le flux compressé inconnu sans
            // l'allouer ni le décompresser.
            let copied = std::io::copy(&mut r.by_ref().take(comp_len_u64), &mut std::io::sink())?;
            if copied != comp_len_u64 {
                return Err("fmap : canal inconnu tronqué".into());
            }
            continue;
        };
        if raw_len_u64 != raw_len as u64 {
            return Err(format!("fmap : taille déclarée invalide pour le canal {}", id[0]).into());
        }
        let compressed = read_exact_vec(&mut r, comp_len)?;
        let raw = decompress(&compressed, raw_len)?;
        match id[0] {
            CHANNEL_ITERATIONS => {
                if iterations.is_some() {
                    return Err("fmap : canal iterations dupliqué".into());
                }
                iterations = Some(bytes_to_u32s(&raw));
            }
            CHANNEL_ZS => {
                if zs.is_some() {
                    return Err("fmap : canal zs dupliqué".into());
                }
                zs = Some(bytes_to_zs(&raw));
            }
            CHANNEL_DISTANCES => {
                if distances.is_some() {
                    return Err("fmap : canal distances dupliqué".into());
                }
                distances = Some(bytes_to_f64s(&raw));
            }
            _ => unreachable!(),
        }
    }

    let iterations = iterations.ok_or("fmap : canal iterations manquant")?;
    let zs = zs.ok_or("fmap : canal zs manquant")?;
    Ok(FractalMap { params, iterations, zs, distances })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::definitions::default_params_for_type;
    use crate::fractal::FractalType;
    use crate::io::png::colorize_to_rgb;
    use crate::render::render_escape_time;

    fn tmp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("fractall_fmap_{}_{}.fmap", tag, std::process::id()))
    }

    /// Verrou jalon 1 : round-trip **bit-identique** de tous les canaux, params
    /// inclus (comparés via leur JSON, la même représentation que le fichier).
    #[test]
    fn fmap_round_trip_bit_identical() {
        let mut params = default_params_for_type(FractalType::Mandelbrot, 17, 13);
        params.center_x_hp = Some("-0.5".into());
        params.center_y_hp = Some("0.0".into());
        let n = 17 * 13;
        // Buffers variés, avec NaN/infini dans distances (INFINITY = pas
        // d'estimation, cf. escape_time.rs) : le format doit les préserver.
        let iterations: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(2654435761)).collect();
        let zs: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(i as f64 * 0.37 - 3.0, -(i as f64) * 0.11))
            .collect();
        let mut distances: Vec<f64> = (0..n).map(|i| (i as f64).sqrt()).collect();
        distances[0] = f64::INFINITY;
        distances[1] = f64::NAN;

        let map = FractalMap {
            params: params.clone(),
            iterations: iterations.clone(),
            zs: zs.clone(),
            distances: Some(distances.clone()),
        };
        let path = tmp_path("roundtrip");
        save_fmap(&map, &path).expect("save");
        let loaded = load_fmap(&path).expect("load");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.iterations, iterations);
        // Comparaison par bits (NaN ≠ NaN en ==).
        let bits = |v: &[Complex64]| -> Vec<(u64, u64)> {
            v.iter().map(|z| (z.re.to_bits(), z.im.to_bits())).collect()
        };
        assert_eq!(bits(&loaded.zs), bits(&zs));
        let dbits = |v: &[f64]| -> Vec<u64> { v.iter().map(|x| x.to_bits()).collect() };
        assert_eq!(dbits(loaded.distances.as_ref().expect("distances présentes")), dbits(&distances));
        assert_eq!(
            serde_json::to_string(&loaded.params).unwrap(),
            serde_json::to_string(&params).unwrap(),
            "params doivent survivre au round-trip"
        );
    }

    /// Verrou jalon 1 : recoloriser depuis une map rechargée == coloriser le
    /// rendu direct, **pixel-exact** (rendu réel 48×36, path f64 standard).
    #[test]
    fn fmap_recolor_matches_direct_render() {
        let params = default_params_for_type(FractalType::Mandelbrot, 48, 36);
        let (iterations, zs) = render_escape_time(&params);
        let direct_rgb = colorize_to_rgb(&params, &iterations, &zs);

        let path = tmp_path("recolor");
        save_fmap(
            &FractalMap { params: params.clone(), iterations, zs, distances: None },
            &path,
        )
        .expect("save");
        let loaded = load_fmap(&path).expect("load");
        let _ = std::fs::remove_file(&path);

        // Recolorisation avec une AUTRE palette : les deux chemins doivent
        // rester d'accord (même buffers → mêmes pixels), y compris après
        // override couleur comme le fait `--from-map`.
        let mut recolor_params = loaded.params.clone();
        recolor_params.color_mode = (params.color_mode + 3) % 27;
        let mut direct_params = params.clone();
        direct_params.color_mode = recolor_params.color_mode;

        let from_map = colorize_to_rgb(&recolor_params, &loaded.iterations, &loaded.zs);
        let (it2, zs2) = render_escape_time(&direct_params);
        let direct2 = colorize_to_rgb(&direct_params, &it2, &zs2);
        assert_eq!(from_map, direct2, "recolor from map doit être pixel-exact");

        // Et avec la palette d'origine : identique au premier rendu colorisé.
        let same = colorize_to_rgb(&loaded.params, &loaded.iterations, &loaded.zs);
        assert_eq!(same, direct_rgb);
    }

    /// Refus propre : fichier non-fmap et version inconnue.
    #[test]
    fn fmap_rejects_invalid_files() {
        let path = tmp_path("invalid");
        std::fs::write(&path, b"PNG\x0dnot a map at all").unwrap();
        assert!(load_fmap(&path).is_err(), "magic invalide doit être refusé");

        // Version future : magic OK, version 999.
        let mut bad = Vec::new();
        bad.extend_from_slice(MAGIC);
        bad.extend_from_slice(&999u32.to_le_bytes());
        bad.extend_from_slice(&0u32.to_le_bytes());
        std::fs::write(&path, &bad).unwrap();
        let err = match load_fmap(&path) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("version inconnue doit être refusée"),
        };
        assert!(err.contains("version"), "erreur explicite attendue, eu : {err}");

        // Une map 0×0 a des canaux formellement cohérents mais ferait
        // sous-déborder l'échantillonnage des miniatures (`h - 1`).
        let params = default_params_for_type(FractalType::Mandelbrot, 0, 0);
        let params_json = serde_json::to_vec(&params).unwrap();
        let mut zero = Vec::new();
        zero.extend_from_slice(MAGIC);
        zero.extend_from_slice(&VERSION.to_le_bytes());
        zero.extend_from_slice(&(params_json.len() as u32).to_le_bytes());
        zero.extend_from_slice(&params_json);
        zero.extend_from_slice(&0u32.to_le_bytes());
        std::fs::write(&path, zero).unwrap();
        assert!(load_fmap(&path).is_err(), "dimensions nulles refusées au chargement");

        // Une taille brute mensongère est refusée avant toute allocation ou
        // décompression du canal.
        let params = default_params_for_type(FractalType::Mandelbrot, 1, 1);
        let params_json = serde_json::to_vec(&params).unwrap();
        let mut huge = Vec::new();
        huge.extend_from_slice(MAGIC);
        huge.extend_from_slice(&VERSION.to_le_bytes());
        huge.extend_from_slice(&(params_json.len() as u32).to_le_bytes());
        huge.extend_from_slice(&params_json);
        huge.extend_from_slice(&1u32.to_le_bytes());
        huge.push(CHANNEL_ITERATIONS);
        huge.extend_from_slice(&u64::MAX.to_le_bytes());
        huge.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(&path, huge).unwrap();
        assert!(load_fmap(&path).is_err(), "taille brute mensongère refusée");

        let bad_map = FractalMap {
            params,
            iterations: vec![],
            zs: vec![Complex64::new(0.0, 0.0)],
            distances: None,
        };
        assert!(save_fmap(&bad_map, &path).is_err(), "buffers incohérents: Err, pas panic");
        let _ = std::fs::remove_file(&path);
    }
}
