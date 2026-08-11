//! Cœur de fractall : moteur de rendu, couleur, I/O, GPU, GUI et QA.
//!
//! Les binaires (`fractall-cli`, `fractall-gui`, `fractall-quality`) sont de
//! MINCES enveloppes au-dessus de cette bibliothèque : parsing d'arguments,
//! boucle d'événements, orchestration. Toute la logique vit ici.
//!
//! Why : avant l'extraction de cette lib, chaque binaire déclarait son propre
//! `mod fractal; mod color; …` et compilait donc l'arbre ENTIER, trois fois.
//! Deux conséquences payées quotidiennement :
//!   1. temps de compilation triplé ;
//!   2. un item consommé par un seul binaire (ou par les seuls tests) paraissait
//!      MORT dans les autres — d'où une série d'exemptions `#[allow(dead_code)]`
//!      qui masquaient la vraie question. Ici, `pub` = API de la bibliothèque :
//!      plus de faux positifs, et un item réellement mort est de nouveau signalé.
//!
//! ⚠️ Les tests unitaires des modules appartiennent désormais à la cible **lib**
//! (`cargo test --lib`), plus aux cibles `--bin` où ils tournaient en triple.

pub mod color;
pub mod fractal;
pub mod gpu;
pub mod gui;
pub mod io;
pub mod quality;
pub mod render;
pub mod video;
