//! Binaire `fractall-video-gui` (G12 jalon 6) — MINCE enveloppe eframe du
//! studio vidéo (`src/video_gui/`). Toute la logique vit dans la lib.

use fractall_cli::video_gui::VideoStudioApp;

fn main() {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Fractall Video Studio")
            .with_inner_size([1200.0, 800.0])
            .with_drag_and_drop(true),
        hardware_acceleration: eframe::HardwareAcceleration::Preferred,
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    if let Err(e) = eframe::run_native(
        "Fractall Video Studio",
        options,
        Box::new(|cc| Ok(Box::new(VideoStudioApp::new(cc)))),
    ) {
        eprintln!("Erreur lors du lancement du studio vidéo: {e}");
        std::process::exit(1);
    }
}
