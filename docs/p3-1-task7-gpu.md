# P3.1 #7 — GPU shader unifié bytecode-driven

État au 2026-05-17 sur branche `bytecode`. Prototype WGSL livré, intégration
Rust à faire.

## Approche

Remplacer les 3 shaders d'escape-time non-perturbation
(`mandelbrot_f32.wgsl`, `julia_f32.wgsl`, `burning_ship_f32.wgsl` — ~380
lignes total, 95% identiques) par un shader unique `bytecode_kernel.wgsl`
qui interprète un bytecode 8-opcodes uploadé via storage buffer.

L'avantage : un nouveau type escape-time = un changement de bytecode côté
Rust, **zéro modification GPU**. Préalable aux hybrides multi-phases
(Mandelbrot×3 + BS×2 etc., feature unique vs Kalles Fraktaler).

## Prototype livré

`src/gpu/bytecode_kernel.wgsl` (~140 lignes) :
- `Params` : ajoute `is_julia: u32` et `bytecode_len: u32`.
- Binding 2 : `array<u32>` du bytecode.
- Boucle compute : initialise (z0, c) selon Mandelbrot vs Julia, puis loop
  - inner loop : exécute toutes les opcodes du bytecode (switch sur opcode)
  - outer loop : bailout check après chaque phase (équivalent au CPU)

Opcodes mappés sur les valeurs Rust de `bytecode::Op` (0=Sqr … 7=Add).

## Validation actuelle

`src/gpu/bytecode_kernel_test.rs` : deux tests qui parsent + valident
le WGSL via naga (déjà dans l'arbre wgpu) sans avoir besoin de GPU. Tourne
dans `cargo test --release --bin fractall-cli gpu::bytecode_kernel_test`.
Dev-dep `naga = { version = "22.1", features = ["wgsl-in"] }`.

## Intégration à faire (Session future de #7)

1. Côté `src/gpu/mod.rs` :
   - Nouvelle entrée `pipeline_bytecode` à côté des 3 existants.
   - Encoder le bytecode au render-time : un `Vec<u32>` de longueur
     `phase.ops.len()`, uploadé dans un storage buffer.
   - Bind group avec 3 bindings (Params, output, bytecode buffer).
   - Dispatcher selon `compile_formula(fractal_type, multibrot_power)` :
     - Si `Some(formula)` et formula mono-phase : utiliser le pipeline
       bytecode.
     - Sinon : fallback sur les pipelines actuels (Mandelbrot/Julia/BS).
2. Tests d'iso-pixel GPU vs CPU sur Mandelbrot/Julia/BurningShip à
   plusieurs zooms, ajoutés au harness `tests/golden_images.rs` avec
   un nouvel exécutable `--gpu`.
3. Une fois validé : supprimer les 3 shaders dupliqués + leurs pipelines
   dans `gpu/mod.rs` (~200 lignes nettes en moins).

## Extensions ultérieures

- **Multi-phases (hybrides)** : ajouter `phase_offsets: array<u32>` qui
  indexe `bytecode[]` pour chaque phase, et un `phase_counter` qui cycle.
- **Perturbation** : adapter `perturbation.wgsl` pour exécuter le bytecode
  au lieu du switch fractal_kind hardcodé. Plus gros chantier (~400 lignes
  WGSL avec BLA lookup + rebasing).
- **f64** : actuellement f32 seulement (suffit pour zoom < ~1e7). Pour
  zooms plus profonds, perturbation GPU est la voie (cf. ci-dessus) ;
  un kernel bytecode f64 standalone n'apporte pas grand-chose.

## Pourquoi pas plus loin maintenant

L'intégration runtime (étape 1 ci-dessus) demande de jongler avec wgpu :
- créer un nouveau `BindGroupLayout` (qui diffère des 3 actuels)
- gérer un buffer de bytecode dynamique (taille varie selon le type)
- dispatcher correctement
- tester sur device réel (les tests naga ne couvrent pas le runtime)

C'est ~150-200 lignes Rust + débogage GPU. Mieux fait en session dédiée.
