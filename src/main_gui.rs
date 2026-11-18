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

    // Reprise de session : la vue (et l'historique) de la dernière fermeture,
    // sauf `--fresh` (vue par défaut ; la session est réécrite à la sortie).
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("Usage: fractall-gui [--fresh]");
        println!("  --fresh   ignore the saved session and start from the default view");
        return;
    }
    let fresh = args.iter().any(|a| a == "--fresh");
    let session = if fresh {
        None
    } else {
        gui::view_history::session_file_path().and_then(|p| gui::view_history::load_session(&p))
    };
    let inner_size = session
        .as_ref()
        .and_then(|s| s.window_size)
        .unwrap_or([800.0, 600.0]);

    // Configuration pour éviter les problèmes EGL sur NVIDIA
    // Utiliser l'accélération matérielle avec préférence GPU haute performance
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Fractall - Fractal viewer")
            .with_inner_size(inner_size)
            .with_drag_and_drop(true),
        // Préférer l'accélération matérielle
        // Activer le rendu GPU
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    
    if let Err(e) = eframe::run_native(
        "Fractall",
        options,
        Box::new(move |cc| Ok(Box::new(FractallApp::with_session(cc, session)))),
    ) {
        eprintln!("Failed to launch the application: {}", e);
        std::process::exit(1);
    }
}
