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
    about = "fractall zoom-video pipeline: plan → render → assemble",
    version,
    author = "Arnaud Verhille and contributors"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Prepare a project: read a TOML config (location/image/fractal/color/
    /// video), compute the number of keyframes (ceil(log2(zoom))) and write
    /// PROJECT/manifest.toml.
    Plan {
        /// Input TOML config (same sections as the manifest).
        config: PathBuf,
        /// Project directory (created if needed).
        #[arg(short, long)]
        project: PathBuf,
    },
    /// Render the project's missing keyframes as .fmap maps (resumable:
    /// valid maps are skipped; changing the manifest's palette does NOT
    /// invalidate the maps).
    Render {
        /// Project directory (containing manifest.toml).
        project: PathBuf,
    },
    /// Assemble the video: interpolate frames between keyframes and encode
    /// with ffmpeg (-o video.mp4) or write PNG frames (--frames-dir).
    Assemble {
        /// Project directory (rendered keyframes).
        project: PathBuf,
        /// Output video file (encoded by ffmpeg).
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Directory of numbered PNG frames (fallback without ffmpeg).
        #[arg(long)]
        frames_dir: Option<PathBuf>,
        /// ffmpeg binary to use.
        #[arg(long, default_value = "ffmpeg")]
        ffmpeg: String,
    },
}

fn main() {
    let cli = Cli::parse();
    let result: Result<(), Box<dyn std::error::Error>> = match cli.cmd {
        Cmd::Plan { config, project } => video::plan_project(&config, &project).map(|m| {
            println!(
                "Manifest written: {} ({} keyframes → zoom {})",
                project.join("manifest.toml").display(),
                m.video.keyframes,
                m.location.zoom
            );
        }),
        Cmd::Render { project } => video::render_project(&project).map(|(rendered, skipped)| {
            println!("Keyframes: {rendered} rendered, {skipped} reused");
        }),
        Cmd::Assemble { project, output, frames_dir, ffmpeg } => {
            let opts = AssembleOptions { output, frames_dir, ffmpeg };
            assemble_project(&project, &opts).map(|stats| {
                println!(
                    "Video assembled: {} frames ({:.1} s)",
                    stats.frames, stats.duration_s
                );
            })
        }
    };
    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
