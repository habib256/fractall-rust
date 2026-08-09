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
            .unwrap_or_else(|| "Panic inconnu".to_string());

        eprintln!("\n❌ Erreur fatale lors de l'initialisation de la GUI:");
        eprintln!("   {}", msg);
        
        // Afficher la localisation si disponible
        if let Some(location) = panic_info.location() {
            eprintln!("   Fichier: {}:{}:{}", location.file(), location.line(), location.column());
        }
        
        if msg.contains("BadAccess") || msg.contains("egl") || msg.contains("wgpu") || msg.contains("EGL") {
            eprintln!("\n💡 Solutions possibles:");
            eprintln!("   1. Vérifiez que vous avez un affichage X11 disponible:");
            eprintln!("      echo $DISPLAY");
            eprintln!("   2. Si vous êtes en SSH, utilisez X11 forwarding:");
            eprintln!("      ssh -X utilisateur@serveur");
            eprintln!("   3. Vérifiez les permissions d'accès au GPU:");
            eprintln!("      ls -la /dev/dri/");
            eprintln!("   4. Essayez de définir la variable d'environnement pour forcer un backend:");
            eprintln!("      WGPU_BACKEND=vulkan cargo run --release --bin fractall-gui");
            eprintln!("      ou");
            eprintln!("      WGPU_BACKEND=gl cargo run --release --bin fractall-gui");
            eprintln!("      ou");
            eprintln!("      WGPU_BACKEND=metal cargo run --release --bin fractall-gui");
            eprintln!("   5. Si vous êtes sur un système headless, essayez:");
            eprintln!("      export DISPLAY=:0");
            eprintln!("      ou utilisez xvfb-run:");
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
        hardware_acceleration: eframe::HardwareAcceleration::Preferred,
        // Activer le rendu GPU
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    
    if let Err(e) = eframe::run_native(
        "Fractall",
        options,
        Box::new(|cc| Ok(Box::new(FractallApp::new(cc)))),
    ) {
        eprintln!("Erreur lors du lancement de l'application: {}", e);
        std::process::exit(1);
    }
}
