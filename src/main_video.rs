//! Binaire `fractall-video` (G12) — MINCE enveloppe du module `video/` :
//! parsing des sous-commandes, la logique vit dans la bibliothèque
//! (cf. src/lib.rs, même convention que les trois autres binaires).
//!
//! Pipeline : `plan` (config → manifest + géométrie keyframes) → `render`
//! (keyframes .fmap, reprise auto) → `assemble` (vidéo ffmpeg ou frames PNG).

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use fractall_cli::video;
use video::assemble::{assemble_project, AssembleOptions};

#[derive(Parser)]
#[command(
    name = "fractall-video",
    about = "Pipeline vidéo zoom fractall : plan → render → assemble",
    version,
    author = "Arnaud Verhille et contributeurs"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Prépare un projet : lit une config TOML (location/image/fractal/color/
    /// video), calcule le nombre de keyframes (ceil(log2(zoom))) et écrit
    /// PROJECT/manifest.toml.
    Plan {
        /// Config TOML d'entrée (mêmes sections que le manifest).
        config: PathBuf,
        /// Dossier du projet (créé si besoin).
        #[arg(short, long)]
        project: PathBuf,
    },
    /// Rend les keyframes manquantes du projet en maps .fmap (reprise :
    /// les maps valides sont skippées ; un changement de palette du manifest
    /// n'invalide PAS les maps).
    Render {
        /// Dossier du projet (contenant manifest.toml).
        project: PathBuf,
    },
    /// Assemble la vidéo : interpole les frames entre keyframes et encode via
    /// ffmpeg (-o video.mp4) ou écrit des frames PNG (--frames-dir).
    Assemble {
        /// Dossier du projet (keyframes rendues).
        project: PathBuf,
        /// Fichier vidéo de sortie (encodé par ffmpeg).
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Dossier de frames PNG numérotées (fallback sans ffmpeg).
        #[arg(long)]
        frames_dir: Option<PathBuf>,
        /// Binaire ffmpeg à utiliser.
        #[arg(long, default_value = "ffmpeg")]
        ffmpeg: String,
    },
}

fn main() {
    let cli = Cli::parse();
    let result: Result<(), Box<dyn std::error::Error>> = match cli.cmd {
        Cmd::Plan { config, project } => video::plan_project(&config, &project).map(|m| {
            println!(
                "Manifest écrit : {} ({} keyframes → zoom {})",
                project.join("manifest.toml").display(),
                m.video.keyframes,
                m.location.zoom
            );
        }),
        Cmd::Render { project } => video::render_project(&project).map(|(rendered, skipped)| {
            println!("Keyframes : {rendered} rendues, {skipped} réutilisées");
        }),
        Cmd::Assemble { project, output, frames_dir, ffmpeg } => {
            let opts = AssembleOptions { output, frames_dir, ffmpeg };
            assemble_project(&project, &opts).map(|stats| {
                println!(
                    "Vidéo assemblée : {} frames ({:.1} s)",
                    stats.frames, stats.duration_s
                );
            })
        }
    };
    if let Err(e) = result {
        eprintln!("Erreur : {e}");
        std::process::exit(1);
    }
}
