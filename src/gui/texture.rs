use image::RgbImage;
use egui::ColorImage;

/// Convertit une RgbImage en ColorImage egui pour affichage.
pub fn rgb_image_to_color_image(img: &RgbImage) -> ColorImage {
    let width = img.width() as usize;
    let height = img.height() as usize;
    let pixels = img.as_raw();
    
    // RgbImage stocke les pixels comme [R, G, B, R, G, B, ...]
    // ColorImage attend [R, G, B, A, R, G, B, A, ...]
    let mut rgba = Vec::with_capacity(width * height * 4);
    for chunk in pixels.chunks_exact(3) {
        rgba.push(chunk[0]);
        rgba.push(chunk[1]);
        rgba.push(chunk[2]);
        rgba.push(255); // Alpha
    }
    
    ColorImage::from_rgba_unmultiplied([width, height], &rgba)
}
