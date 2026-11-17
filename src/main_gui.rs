use fractall_cli::gui;

use gui::FractallApp;

fn main() {
    // Configurer le panic hook pour afficher un message plus informatif
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let msg = panic_info.payload().downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| {
                panic_info.payload().downcast_ref::<String>()
                    .map(|s| s.clone())
            })
            .unwrap_or_else(|| "Unknown panic".to_string());

        eprintln!("\n❌ Fatal error while initializing the GUI:");
        eprintln!("   {}", msg);
        
        // Afficher la localisation si disponible
        if let Some(location) = panic_info.location() {
            eprintln!("   File: {}:{}:{}", location.file(), location.line(), location.column());
        }
        
        if msg.contains("BadAccess") || msg.contains("egl") || msg.contains("wgpu") || msg.contains("EGL") {
            eprintln!("\n💡 Possible solutions:");
            eprintln!("   1. Check that an X11 display is available:");
            eprintln!("      echo $DISPLAY");
            eprintln!("   2. Over SSH, use X11 forwarding:");
            eprintln!("      ssh -X user@server");
            eprintln!("   3. Check GPU access permissions:");
            eprintln!("      ls -la /dev/dri/");
            eprintln!("   4. Try setting the environment variable to force a backend:");
            eprintln!("      WGPU_BACKEND=vulkan cargo run --release --bin fractall-gui");
            eprintln!("      or");
            eprintln!("      WGPU_BACKEND=gl cargo run --release --bin fractall-gui");
            eprintln!("      or");
            eprintln!("      WGPU_BACKEND=metal cargo run --release --bin fractall-gui");
            eprintln!("   5. On a headless system, try:");
            eprintln!("      export DISPLAY=:0");
            eprintln!("      or use xvfb-run:");
            eprintln!("      xvfb-run -a cargo run --release --bin fractall-gui");
        }
        
        // Appeler le hook par défaut pour afficher le backtrace si RUST_BACKTRACE est défini
        default_hook(panic_info);
    }));

    // Configuration pour éviter les problèmes EGL sur NVIDIA
    // Utiliser l'accélération matérielle avec préférence GPU haute performance
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Fractall - Fractal viewer")
            .with_inner_size([800.0, 600.0])
            .with_drag_and_drop(true),
        // Préférer l'accélération matérielle
        // Activer le rendu GPU
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    
    if let Err(e) = eframe::run_native(
        "Fractall",
        options,
        Box::new(|cc| Ok(Box::new(FractallApp::new(cc)))),
    ) {
        eprintln!("Failed to launch the application: {}", e);
        std::process::exit(1);
    }
}
