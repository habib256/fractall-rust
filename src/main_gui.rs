mod fractal;
mod color;
mod render;
mod io;
mod gui;

use gui::FractallApp;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Fractall - Visualiseur de fractales")
            .with_inner_size([1024.0, 768.0]),
        ..Default::default()
    };
    
    eframe::run_native(
        "Fractall",
        options,
        Box::new(|cc| Box::new(FractallApp::new(cc))),
    )
}
