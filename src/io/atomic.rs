//! Écriture transactionnelle de petits/grands artefacts persistants.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_SERIAL: AtomicU64 = AtomicU64::new(0);

fn temp_path(path: &Path, serial: u64) -> Result<PathBuf, std::io::Error> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path.file_name().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "chemin sans nom de fichier")
    })?;
    Ok(parent.join(format!(
        ".{}.tmp-{}-{serial}",
        name.to_string_lossy(),
        std::process::id()
    )))
}

/// Écrit dans un fichier temporaire du même dossier, synchronise, puis
/// remplace la cible par rename atomique. L'ancienne cible reste intacte si
/// l'écriture échoue avant le rename.
pub(crate) fn write_atomic(
    path: &Path,
    write: impl FnOnce(&mut File) -> Result<(), Box<dyn std::error::Error>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut last_collision = None;
    for _ in 0..32 {
        let serial = TEMP_SERIAL.fetch_add(1, Ordering::Relaxed);
        let tmp = temp_path(path, serial)?;
        let opened = OpenOptions::new().write(true).create_new(true).open(&tmp);
        let mut file = match opened {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                last_collision = Some(e);
                continue;
            }
            Err(e) => return Err(e.into()),
        };
        let result = write(&mut file)
            .and_then(|_| file.flush().map_err(Into::into))
            .and_then(|_| file.sync_all().map_err(Into::into));
        drop(file);
        if let Err(e) = result {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        if let Err(e) = std::fs::rename(&tmp, path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e.into());
        }
        return Ok(());
    }
    Err(last_collision
        .unwrap_or_else(|| std::io::Error::new(std::io::ErrorKind::AlreadyExists, "collision temporaire"))
        .into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_atomic_write_preserves_existing_target() {
        let path = std::env::temp_dir().join(format!(
            "fractall_atomic_preserve_{}.txt",
            std::process::id()
        ));
        std::fs::write(&path, b"old").unwrap();
        let err = write_atomic(&path, |file| {
            file.write_all(b"new")?;
            Err("simulated failure".into())
        });
        assert!(err.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"old");
        let _ = std::fs::remove_file(path);
    }
}
