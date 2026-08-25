//! Coût de la colorisation (`io::png::colorize_buffers`) — `cargo run
//! --release --example bench_colorize`.
//!
//! Sert à arbitrer les optimisations de l'étage couleur, qui est partagé par
//! TOUS les paths de sortie (PNG CLI, passes GUI, tuiles G10.5, assemblage
//! vidéo, samples AA). Trois mesures :
//!
//! 1. colorisation d'une frame pleine, à `color_offset` VARIABLE — le régime
//!    du défilement de palette vidéo (`[dynamics] palette_offset`), où le
//!    cache RGB de l'assembleur ne peut par construction jamais toucher ;
//! 2. construction d'un `PaletteLut` neuf (mémoïsé par `PaletteLut::cached`,
//!    sinon payé une fois par tuile dans le path streaming GUI) ;
//! 3. tuile 64×64 seule ET passe complète depuis un pool rayon SATURÉ — le
//!    seul régime représentatif du sink G10.5 (une tuile mesurée à vide voit
//!    tous les cœurs libres, ce qui fausse la comparaison).

use rayon::prelude::*;
use std::time::Instant;

use fractall_cli::io::png::colorize_to_rgb_with_extras;
use fractall_cli::fractal::{default_params_for_type, FractalType, OutColoringMode};
use fractall_cli::render::render_escape_time;

fn main() {
    for (w, h) in [(1280u32, 720u32), (2560, 1440)] {
        let mut p = default_params_for_type(FractalType::Mandelbrot, w, h);
        p.iteration_max = 1000;
        p.color.out_coloring_mode = OutColoringMode::Smooth;
        let t0 = Instant::now();
        let out = render_escape_time(&p);
        let (it, zs) = (out.iterations, out.zs);
        let render_s = t0.elapsed().as_secs_f64();

        // Colorisation complète, répétée (mesure stable).
        const N: usize = 20;
        let t1 = Instant::now();
        for k in 0..N {
            p.color.color_offset = k as f64 * 0.01; // offset variable = cas défilement
            std::hint::black_box(colorize_to_rgb_with_extras(&p, &it, &zs, &[], &[]));
        }
        let colorize_s = t1.elapsed().as_secs_f64() / N as f64;

        // Coût seul de la construction du LUT (fixe par appel).
        let t2 = Instant::now();
        for _ in 0..N {
            std::hint::black_box(fractall_cli::color::PaletteLut::new(p.color.color_mode, p.color.color_space));
        }
        let lut_s = t2.elapsed().as_secs_f64() / N as f64;

        println!(
            "{w}x{h}: rendu={render_s:.3}s  colorisation={:.1}ms (LUT neuf {:.3}ms)  \
             ratio colorisation/rendu={:.1}%",
            colorize_s * 1e3,
            lut_s * 1e3,
            100.0 * colorize_s / render_s
        );

        // Path TUILES de la GUI : colorisation d'une tuile 64×64 (taille
        // choisie par TileGrid pour une image de cette taille).
        let mut tp = p.clone();
        tp.width = 64;
        tp.height = 64;
        let tile_n = 64 * 64;
        let (tit, tzs) = (it[..tile_n].to_vec(), zs[..tile_n].to_vec());
        let tiles_per_frame = ((w as f64 / 64.0).ceil() * (h as f64 / 64.0).ceil()) as usize;

        // (a) une tuile SEULE sur une machine au repos (mesure trompeuse :
        //     rayon peut y consacrer tous les cœurs).
        let t3 = Instant::now();
        const NT: usize = 2000;
        for _ in 0..NT {
            std::hint::black_box(colorize_to_rgb_with_extras(&tp, &tit, &tzs, &[], &[]));
        }
        let tile_solo = t3.elapsed().as_secs_f64() / NT as f64;

        // (b) RÉGIME RÉEL du sink G10.5 : toutes les tuiles d'une passe
        //     colorisées depuis des workers rayon déjà saturés. C'est le
        //     débit qui compte pour la réactivité de la GUI.
        let t4 = Instant::now();
        const NP: usize = 20;
        for _ in 0..NP {
            let all: Vec<Vec<u8>> = (0..tiles_per_frame)
                .into_par_iter()
                .map(|_| colorize_to_rgb_with_extras(&tp, &tit, &tzs, &[], &[]))
                .collect();
            std::hint::black_box(all);
        }
        let pass_s = t4.elapsed().as_secs_f64() / NP as f64;
        println!(
            "  tuile 64x64 : solo={:.3}ms  |  passe complète ({tiles_per_frame} tuiles, \
             pool saturé)={:.1}ms",
            tile_solo * 1e3,
            pass_s * 1e3
        );
    }
}
