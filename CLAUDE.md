# CLAUDE.md

> Carte rapide du repo pour reprendre une session sans relire tout le code.
> Référence algorithmique : **Fraktaler-3.1** (`fraktaler-3-3.1/src/`,
> `fraktaler-3-3.1/fraktaler-3-analysis.md`).
> **Boucle d'auto-amélioration** : protocole dans `HARNESS.md`, outil
> `scripts/harness.py`, itération via le skill `/improve`, dernier état dans
> `SCORECARD.md` + `harness/history/`.

## Build & Run

```bash
cargo build --release
cargo run --release --bin fractall-cli -- --type 3 --output out.png
cargo run --release --bin fractall-gui
cargo run --release --bin fractall-quality -- suite
```

Prérequis natifs : GMP / MPFR / MPC (pour `rug`).

## Tests

- **Unit tests** : `cargo test --release --bin fractall-cli` (~271 tests).
  Couvre `perturbation/{bla,delta,series,nonconformal,distance,interior,
  glitch,orbit,types,nucleus,mod}`, `jitter` (AA), `lyapunov`, `progressive`,
  `wisdom` (select_algorithm/device, plan), et tout le `bytecode/` (`compile`
  + `compile_hybrid_formula`, `interp{,_gmp}`, `bla_dual`, `delta_form`,
  `pixel_loop{,_exp}`).
- **Golden image tests** (`tests/golden_images.rs`) : 24 cas pixel-exact
  (goldens forcent `--no-gpu` pour déterminisme cross-machine) — paths f64
  standard, perturbation, deep zoom GMP, types non-bytecode, zooms
  intermédiaires (1e10/1e15/1e20, path perturbation par défaut), atom-domain,
  et `mandelbrot_interior_ref_640` (SEUL cas >512², exerce l'escalade dd).
  - Lancer : `cargo test --release --test golden_images`
  - Régénérer : `FRACTALL_UPDATE_GOLDENS=1 cargo test --release --test
    golden_images`. **Toujours** vérifier visuellement les nouveaux PNG.
  - Note : les goldens ER=25 (2026-05-20) ; `mandelbrot_e10/e15/e20` (G6,
    2026-07-12) comblent le trou 1e8→1e50 (revus visuellement, e10 PASS
    pixel-exact vs GMP, e15/e20 bruit de bord dispersé p99=0).
- **QA perturbation vs GMP** (`fractall-quality`) : compare le chemin
  perturbation au rendu GMP pur pixel-par-pixel. Voir §Quality plus bas.

CI : `.github/workflows/ci.yml` (unit + golden sur push/PR, ubuntu,
gmp/mpfr/mpc). Extension du corpus golden à venir (cf. TODO G6).

## Harness d'auto-amélioration (HARNESS.md)

Trois axes mesurés contre Fraktaler-3 : **vitesse** (wall-clock head-to-head,
geomean des ratios), **génération/parité** (`compare_f3.py`, EXR N0/NF),
**qualité** (`fractall-quality` vs ground truth GMP + goldens).

```bash
python3 scripts/harness.py score --tier quick   # cycle interne (~10 cas, 256²)
python3 scripts/harness.py score --tier standard # 3 runs médiane, avant commit perf
python3 scripts/harness.py baseline              # fige la référence (explicite)
```

Sorties : `harness/history/<date>-<sha>.json` (mémoire longue, versionnée),
`SCORECARD.md` (dernier score + gaps triés), artefacts lourds dans `bench/`
(gitignoré). Baseline **par machine**. Binaire F3 : auto-détection
(`fraktaler-3-3.1/fraktaler-3` Linux, `.macos` sinon, override `F3_BIN`).
Priorités de gap : correction > robustesse > vitesse > qualité. Détail et
invariants : `HARNESS.md`.

## Architecture

```
src/
├── main.rs              # CLI fractall-cli (clap)
├── main_gui.rs          # GUI fractall-gui (egui/eframe)
├── main_quality.rs      # QA fractall-quality (clap subcommands)
├── fractal/
│   ├── mod.rs           # exports + default_params_for_type()
│   ├── types.rs         # FractalType, FractalParams, AlgorithmMode, …
│   ├── definitions.rs   # constantes par type + LyapunovPreset
│   ├── iterations.rs    # escape-time f64 + dispatch bytecode unifié
│   ├── jitter.rs        # AA per-frame : radical_inverse + triangle (port F3)
│   ├── xaos.rs          # G10.4 réutilisation pixels inter-frame (XaoS)
│   ├── gmp.rs           # précision arbitraire (rug / mpc)
│   ├── lyapunov.rs      # Lyapunov exponent
│   ├── buddhabrot.rs    # Buddhabrot / Nebulabrot / Anti-Buddhabrot
│   ├── vectorial.rs     # Von Koch, Dragon
│   ├── orbit_traps.rs   # Point, Line, Cross, Circle
│   ├── bytecode/        # Moteur unifié Fraktaler-3 (P3.1, défaut)
│   │   ├── mod.rs          # Op, Phase, Formula
│   │   ├── compile.rs      # compile_formula + compile_hybrid_formula + formula_for_params (G4)
│   │   ├── interp.rs       # iterate_bytecode_f64
│   │   ├── interp_gmp.rs   # interpréteur GMP pour orbite référence
│   │   ├── delta_form.rs   # DeltaState f64 + DeltaStateExp ComplexExp
│   │   ├── bla_dual.rs     # BLA mat2 unifié via dual-numbers
│   │   ├── pixel_loop.rs   # BLA + delta-form + rebasing F3 (f64)
│   │   ├── pixel_loop_exp.rs # idem ComplexExp (deep zoom > 1e13)
│   │   ├── pixel_loop_dd.rs # tier double-double ~106b (float128-like), opt-in
│   │   └── bla_dd.rs       # table BLA à coefficients dd (skip sans perte 106b)
│   └── perturbation/    # Path GMP + ponts vers bytecode
│       ├── mod.rs          # render_perturbation_cancellable_with_reuse
│       ├── types.rs        # ComplexExp, FloatExp (mantisse + exposant)
│       ├── orbit.rs        # ReferenceOrbit{,Cache}, HybridBlaReferences
│       ├── nucleus.rs      # Atom-domain period + Newton (port F3)
│       ├── bla.rs          # BlaTable conformal (path GMP deep zoom)
│       ├── nonconformal.rs # BLA matriciel (path GMP)
│       ├── delta.rs        # iterate_pixel{,_gmp}
│       ├── series.rs       # Taylor series approximation
│       └── glitch.rs       # Clustering Pauldelbrot (legacy)
├── render/
│   ├── escape_time.rs    # dispatcher unique CLI↔GUI (sélection algo)
│   └── tiles.rs          # G10.5 file de tuiles priorité-centre + sink streaming
├── gpu/                  # wgpu Vulkan/Metal/DX12
│   ├── mod.rs              # GpuRenderer + pipeline bytecode
│   ├── *_f32.wgsl          # paths legacy par type
│   ├── perturbation.wgsl   # kernel perturbation F3-strict f64 (SHADER_F64)
│   └── bytecode_kernel.wgsl # runtime bytecode unifié (P3.1)
├── gui/                  # FractallApp (egui)
├── color/                # 27 palettes + RGB/HSB/LCH + 15 modes
├── io/png.rs             # save/load avec métadonnées JSON
└── quality/              # fractall-quality QA suite
```

## Dépendances principales

| Crate | Version | Rôle |
|-------|---------|------|
| clap | 4.5 | CLI parsing (derive) |
| image | 0.25 | PNG support |
| num-complex | 0.4 | Arithmétique complexe |
| rayon | 1.11 | Parallélisme multi-thread |
| eframe / egui | 0.34 | GUI (wrapper egui, backend wgpu) |
| wgpu | 22.1 | GPU (Vulkan/Metal/DX12) |
| pollster | 0.3 | Runtime async GPU |
| bytemuck | 1.15 | Sérialisation buffers GPU |
| rug | 1.24 | Précision arbitraire (GMP/MPFR/MPC) |
| serde / serde_json | 1.0 | Métadonnées PNG |
| png | 0.17 | Accès chunks PNG bas niveau |
| toml | 0.8 | Loader paramètres (`toml/`) |
| naga (dev) | 22.1 | Validation WGSL en tests |

## Système de coordonnées

`center + span` (pas `xmin/xmax/ymin/ymax`) :

```rust
pub struct FractalParams {
    pub center_x: f64,           // f64 GPU/CPU standard
    pub center_y: f64,
    pub span_x: f64,
    pub span_y: f64,
    pub center_x_hp: Option<String>,  // HP GMP pour zooms > 10^15
    pub center_y_hp: Option<String>,
    pub span_x_hp: Option<String>,
    pub span_y_hp: Option<String>,
    pub rotation: f64,            // degrés, R = mat2(cos,-sin,sin,cos)
    pub find_nucleus: bool,       // opt-in Mandelbrot only
    pub transform_k: Option<[f64; 4]>, // mat2 row-major (F3 param.transform)
    // …
}
```

`transform_matrix()` est le point d'entrée pour le mapping pixel→c : K si
`transform_k` est `Some` (et fini), sinon retombe sur `rotation_matrix()`.
Drop-in compatible — quasi-tous les callsites consomment un tuple
`(m00, m01, m10, m11)` ou `None`.

Helpers HP-aware (`perturbation/mod.rs`) :
- `effective_spans_fexp(params)` → `(FloatExp, FloatExp)` survie f64
  underflow (zoom > 1e308).
- `effective_pixel_size(params)` → utilisé par les dispatchers de path.

## Métadonnées PNG

Chunk `tEXt` clé `fractall-params` = JSON sérialisé de `FractalParams`
(coordonnées HP incluses). API : `save_png_with_metadata()` /
`load_png_metadata()`. **Drag-and-drop** GUI restaure l'état exact.

## Dispatch rendu (`render/escape_time.rs`)

### ⚠️ Chemin de rendu UNIQUE CLI ↔ GUI (invariant)

**Le CLI et la GUI DOIVENT suivre exactement le même chemin de rendu CPU.**
Il existe **un seul dispatcher** : `render_escape_time_cancellable_with_reuse(
params, cancel, reuse, orbit_cache, xaos, tiles)`. Toute la sélection
type→algorithme (perturbation / GMP / f64 / spéciaux) et l'appel des renderers
vivent là. `xaos` (G10.4) et `tiles` (G10.5) sont les options GUI de
réutilisation inter-frame et de scheduling par tuiles — `None` partout ailleurs.

- **CLI** (`main.rs`) → `render_escape_time(params)` = thin wrapper sur le
  dispatcher (no cancel, no reuse, `&mut None` cache).
- **GUI** (`gui/app.rs`, chaque passe progressive + HQ + AA) → appelle le MÊME
  dispatcher. `orbit_cache: &mut Option<Arc<ReferenceOrbitCache>>` est threadé
  in/out pour réutiliser l'orbite référence entre passes (perturbation only).
- **`fractall-quality`** → idem (`&mut None`).

Le path perturbation passe par `render_perturbation_with_cache` (cœur commun) ;
le cache est géré DANS le dispatcher, pas par un appel parallèle côté GUI.

**Ne JAMAIS** réintroduire une logique de dispatch dans `gui/app.rs` ou
dupliquer `render_escape_time*` : une divergence GUI/CLI = bug.

**Dispatch GPU UNIFIÉ** (2026-05-21) : `GpuRenderer::render_dispatch(params,
cancel, reuse, orbit_cache) -> Option<GpuDispatchResult>` (`gpu/mod.rs`) est le
point d'entrée unique partagé par le CLI (`main.rs`) et le thread GUI. Il calcule
`use_perturbation` (algorithm_mode + plane_transform), dispatche par type
(perturbation Mandelbrot/Julia/BurningShip, sinon shader f32), thread l'orbite
référence, et renvoie `None` quand le GPU ne peut pas → le caller fait le
fallback CPU via le dispatcher unique. **Ne plus dupliquer ce choix dans
`main.rs`/`gui/app.rs`.**

**Auto-device (G9.5)** : `wisdom::select_device(params, gpu_available) ->
Device` (`fractal/wisdom.rs`) arbitre CPU/GPU par débit benché machine
(`fractall-cli --wisdom-bench` → `~/.config/fractall/wisdom.toml`, clés dont
`gpu_perturb_f64`) SOUS garde-fou correction : GPU routé UNIQUEMENT dans la
plage deep both-perturbation (les deux devices font de la perturbation, ~1e12
à 4e37) ; JAMAIS sur les shaders std f32 (24 b = faux). Arbitrage final
`arbitrate_device` : GPU si `gpu_thr > cpu_thr · GPU_SPEED_MARGIN` (les deux
benchés). Sur GPU grand public (f64 1:64) l'auto reste CPU. Consommé par
`main.rs` (`use_gpu`), `gui/app.rs` (menu « Tech: 🔄 Auto »), CLI overrides
`--gpu`/`--no-gpu`. Goldens forcent `--no-gpu`.

```
AlgorithmMode::Auto :
  - Spéciaux : VonKoch/Dragon → vectoriel ; Buddha* → densité ;
               Lyapunov → exposant ; Mandelbulb → 3D
  - Escape-time (Mandelbrot, Julia, BS, Tricorn, Celtic, Buffalo, PerpBS,
    Multibrot puiss. entière, + variantes Julia) :
      should_use_perturbation() (zoom > ~1e13) → perturbation + BLA
      should_use_gmp_reference() (zoom > 1e16) → GMP reference
      sinon                                    → CPU f64 standard
  - Perturbation incompatible avec plane_transform != Mu → fallback f64/GMP

Modes forcés : StandardF64 | Perturbation | ReferenceGmp
```

### Moteur bytecode (défaut depuis P3.1 Session E)

Si `params.use_bytecode_engine` ET `compile_formula(type, power)` réussit :

- **CPU f64 standard** : `iterations.rs::iterate_via_bytecode`. Supporte
  distance estimation, interior detection, orbit traps via tracking de `dz`.
- **CPU perturbation** : `perturbation/delta.rs::try_bytecode_unified_path`
  → `bytecode/pixel_loop.rs` (f64) ou `pixel_loop_exp.rs` (ComplexExp,
  deep zoom > 1e13). **Rebasing F3 strict** (`|Z[m] + δ|² < |δ|²` **OU bout de
  référence**) remplace la glitch detection Pauldelbrot legacy. Le **rebase-at-end**
  (F3 `hybrid.cc:301` : `z := Z+δ, m := 0` quand la réf est épuisée) garde les
  centres escape-time profonds sur le path perturbation au lieu de tomber en GMP
  par-pixel — c'est ce qui a débloqué la perf deep-zoom (e50 544→1.6 s, e1000
  742→0.5 s, dragon ~6 h→6.5 s à 256², cf. TODO G2). ⚠️ Lire `z_ref[m]` avec `m`
  clampé à `ref_len-1` (après un pas, `m` peut valoir `ref_len`).
  **Fallback correction** (Mandelbrot bytecode, `perturbation/mod.rs`) : quand
  `glitch_ratio > GLITCH_FALLBACK_THRESHOLD` (0.30), on **escalade au tier dd**
  (`[DD-ESCALATION]`, re-render pixel-exact GMP ~4-8× plus rapide) au lieu du
  backstop full-GMP per-pixel. Les blocs de résolution glitch récursive
  (neighbor-pass, secondary-refs, réf-intérieure `l.1623`) sont tous gatés
  `!bytecode_path` : sans le gate, la 2e passe réf-intérieure supprimait le
  fallback GMP à >512² → ~3.4 % de structure spurious.
- **GPU** : `gpu/mod.rs::try_render_bytecode` → `bytecode_kernel.wgsl`.

Fallback legacy si : type non compilable, `--no-bytecode`, ou GMP deep zoom
+ features non encore portées sur pixel_loop.

### Nucleus finder (P1.6.a + P1.6.b, 2026-05-19)

`perturbation/nucleus.rs` : port de Fraktaler-3.1 `hybrid_period` +
`hybrid_center` + `hybrid_size` (`hybrid.cc:417,493,544`). Mandelbrot
uniquement, opt-in via CLI `--find-nucleus` ou `params.find_nucleus = true`.

Pipeline :
1. `find_period_atom_domain(cx, cy, max_iter, s, prec)` — critère atom-domain
   `|z|² < s²·|dz|²` avec `s` = échelle de vue. Plus fiable que min-|z|.
2. `newton_refine_center(…)` — Newton complexe via dual-numbers GMP,
   tracking best-seen pour retourner amélioration locale même si convergence
   stricte échoue.
3. `hybrid_size_mat2(…)` — itère `period-1` fois via dual-numbers 2D et
   accumule `b += L⁻¹`. Renvoie `size = 1/(λ²·β)` (taille canonique) et la
   matrice `K = inv(transp(b))/β` (orientation + scale). La rotation extraite
   via `atan2(K[2], K[0])` aligne le frame de rendu sur celui du minibrot —
   indispensable pour les minibrots non-axis-aligned (seahorse valley,
   flake, olbaid*). F3 `out.transform = K; out.p.transform.rotate = 0`
   (`engine.cc:208`) : on écrase de même `params.rotation` par la valeur
   dérivée de K.

**Why** : à deep zoom, l'orbite référence escape avant `iteration_max` →
perturbation tronque. Le nucleus refine vers un centre périodique exact
dont l'orbite ne s'évade pas. Sans K, un minibrot rotated apparaît noir
ou en bandes diagonales car les pixels échantillonnent à travers les
branches voisines.

Reste pour la parité/perf F3 (cf. TODO G2/G4) : **auto-firing dd par
sensibilité** (l'échelle wisdom est livrée, cf. §Wisdom ci-dessous) et
**nucleus phase-aware** pour les hybrides. La rotation de vue est gérée au
pixel→c (CPU + GPU bytecode) ; `Op::Rot` reste un opcode CPU dormant, utile
seulement pour une rotation *par phase* (TODO G4).

### Hybrides multi-phase (G4 jalon 1-2, `bytecode/compile.rs`)

Chaîner des formules par phase, itérées cycliquement (`phases[n % len]`). CLI
**`--phases mandelbrot,burning_ship`** → `params.hybrid_phases:
Option<Vec<FractalType>>`. `formula_for_params(params)` = source UNIQUE de la
formule (hybride via `compile_hybrid_formula` si `hybrid_phases`, sinon
`compile_formula`). **Rendu par le path f64 standard** (`iterate_bytecode_f64`
cycle déjà les phases) : `select_algorithm` **force `StandardF64`** pour un
hybride, le GMP par-pixel `iterate_point_mpc` étant z²+c hardcodé (ne cycle
pas). `render_dispatch` renvoie `None` (GPU ne cycle pas → fallback CPU).
**Deep (jalon 3)** : dans la bande f64-perturbation (`pixel ∈
[exp_threshold, perturb_threshold]`, ~zoom 1e10–1e13) `select_algorithm` route
l'hybride vers **Perturbation** ; `delta.rs::try_bytecode_unified_path` route
le multi-phase vers `iterate_pixel_unified_multi_phase` (pas directs f64 +
rebasing, SANS BLA ; dd/exp gatés single-phase). Orbite référence itérée avec
la formule hybride (`orbit.rs` → `formula_for_params`). Verrous : `[M,M] ==
Mandelbrot` pixel-exact — unit test (f64-std) + render-level deep-perturbation
(`hybrid_mm_equals_mandelbrot_deep_perturbation`) — + golden
`mandelbrot_hybrid_burningship`. **Reste (jalon 4)** : exp multi-phase
(deep > 1e13) + BLA par phase (perf) + nucleus phase-aware + éditeur GUI.
`fractal_type` sert la convention d'appel (Mandelbrot-like : δ₀=0, dc=pixel).

### Wisdom auto-dispatch (`fractal/wisdom.rs`, 2026-07-12 · G9.1 2026-07-15)

Source UNIQUE de la sélection **algorithme + tier + variantes** :
- `wisdom::select_algorithm(params, Device::Cpu|Gpu)` → `StandardF64 |
  Perturbation | ReferenceGmp`. Consommée par le dispatcher CPU
  (`render/escape_time.rs`, famille `perturbation_family` = M/J/BS/Tricorn),
  par `GpuRenderer::render_dispatch` (device Gpu : seuil perturbation f32 ~1e5
  vs ~1e12 CPU) et par les labels/passes GUI (`effective_cpu_mode`,
  `will_use_perturbation`). Couvre modes forcés, fallback plane≠Mu, Auto.
- `wisdom::number_tier(params)` → `F64 | Exp | Dd` (ordre dd demandé > exp
  `pixel_size<1e-280` > f64), consommée par `bytecode_path_label` ET le dispatch
  d'exécution (`try_bytecode_unified_path`).
- `wisdom::variants(params)` → `{compression, harmonic}` : partie STATIQUE des
  prédicats de routage. Compression : opt-in `FRACTALL_COMPRESS_REF`
  (`compression_active`, per-pixel-safe : gate d'abord). **Harmonic : routé
  AUTO par défaut (G9.3)** — `FRACTALL_HARMONIC_LA` tri-état (unset/`auto` →
  Auto ; `1|lla|mla` → forcé ; `0|off|bla` → kill switch) ; candidat
  `harmonic_candidate` (per-render), décision finale au build de l'entrée
  cache BLA : probe `detect_period0` + `route_harmonic_auto` (route si
  `1 ≤ period0 ≤ 100`, calibré corpus — gagne 1.2-5.9× jusqu'à p78, perd dès
  p112 ; la longueur d'orbite est hors de cause : super_dense p9/695 k gagne
  1.74×). Per-pixel : routage sur la PRÉSENCE de la table dans l'entrée.

`wisdom::plan(params)` / `plan_for(params, device)` renvoie un `WisdomPlan`
inspectable (device + algorithme + tier + variantes + **débit benché machine**
+ exposant/mantisse requis F3-style + précision GMP orbite) ; `FRACTALL_WISDOM=1`
logue la ligne `[WISDOM]` (`bench=` = iters/s mesurés, `-` si pas de fichier
wisdom). **Benchmarks machine (G9.2)** : `fractal/wisdom_bench.rs` +
`fractall-cli --wisdom-bench` → `~/.config/fractall/wisdom.toml` (override
`FRACTALL_WISDOM_FILE`) ; débit effectif par technique (cpu_std_f64,
cpu_perturb_{f64,exp,dd}, gpu_std_f32) sur rendus réels, modèle F3
`wisdom.cc:393`. Consommateur d'arbitrage device = jalon G9.5. Modèle F3 (`wisdom.cc:240` /
`render.cc:219`) : un type de mantisse `M` / exposant `E` est viable si
`req_exp+16 < 2^E/2` ET `req_prec < M` ; pour une frame **centrée**
`req_prec ≈ log2(hypot(w,h))` (~8-13 b), donc **f64 (53 b) suffit toujours sur la
mantisse** — l'escalade f64→exp se fait sur l'**exposant** (profondeur), jamais
la mantisse (vérifié : à 1e300, `req_exp=1003` mais `req_prec=8`). Le tier **dd
(~106 b)** reste opt-in (`--dd-tier`) : son besoin vient d'une sensibilité pixel
non captée par un détecteur cheap fiable (proxy `cbits` réfuté, cf. TODO G3).
Seuils calibrés **préservés** (271 unit + 24 golden pixel-exact + sweep-lock).

### Précision GMP perturbation

Formule C++ Fraktaler-3 : `bits = max(24, 24 + floor(log2(zoom * height)))`,
clamp `[128, 65536]`. Cf. `compute_perturbation_precision_bits()` ; le champ
utilisateur `precision_bits` sert de plancher.

**Cache** (`ReferenceOrbitCache`) : orbite + BLA réutilisées si même
centre / type / précision. ⚠️ **Régime atom-domain** (`is_valid_for` +
`can_subset_reuse` → `atom_regime_scale_mismatch`) : la troncature atom-domain
de la référence dépend de l'ÉCHELLE de vue (`atom_radius_sqr = span_vue²`), donc
une référence est baked à son span de construction. En zoom profond
(`pixel_size < 1e-13`) une référence bâtie à une échelle différente (±1/16) est
INVALIDÉE (rebuild) — sinon sa troncature ne correspond pas à un build frais →
bruit sel-et-poivre (~1.7 % px) sur le rendu inter-frame (bug corrigé 2026-07-16 ;
le pan à profondeur fixe garde span constant → réutilisation préservée). N'affecte
que la réutilisation GUI multi-frame ; les rendus single-shot (CLI/quality/harness,
`cache=None`) ne consultent jamais ces prédicats.

## Paramètres perturbation

| Champ | Description | Défaut |
|-------|-------------|--------|
| `bla_threshold` | seuil delta BLA | 1/2²⁴ ≈ 5.96e-8 |
| `bla_validity_scale` | multiplicateur rayon BLA | 1.0 |
| `glitch_tolerance` | tolerance Pauldelbrot (legacy) | 1e-4 |
| `series_order` | ordre série (0 = off) | 2 |
| `max_secondary_refs` | refs secondaires (legacy) | 3 |
| `min_glitch_cluster_size` | taille min cluster (legacy) | 100 |
| `max_perturb_iterations` | cap pas DIRECTS par pixel (⚠️ voir note) | 1024 |
| `max_bla_steps` | cap pas BLA par pixel | 1024 |
| `use_reference_precision_formula` | formule C++ F3 | true |
| `use_bytecode_engine` | path unifié BLA mat2 + rebasing F3 | true |
| `use_dd_tier` | tier double-double ~106b Mandelbrot deep (float128-like, sans BLA) | false |
| `find_nucleus` | nucleus Mandelbrot avant orbit | false |
| `jitter_scale` | amplitude AA sous-pixel (px) | 0.0 |
| `aa_subpixel_offset` | offset AA transitoire (`#[serde(skip)]`, posé par la boucle multi-sample) | `[0,0]` |
| `rotation` | degrés CCW, mat2(cos,-sin,sin,cos) | 0.0 |

⚠️ **`max_perturb_iterations` / `max_bla_steps` : clampés à `≥ iteration_max`**
dans `render_perturbation_with_cache` (chemin commun). Comme `iters_ptb ≤ n <
iteration_max`, un cap < iteration_max ne fait que tronquer les pas directs tôt →
compte d'itération ~radial ⇒ **anneaux concentriques** (cf. cusp -0.75 : défaut
1024 < ~1700 requis). Le défaut 1024 reste le champ utilisateur, mais le clamp
garantit l'absence de troncature parasite. F3 met `maximum_perturb_iterations =
iterations`. Le loader TOML faisait déjà `= iters` ; le clamp couvre GUI + CLI
non-TOML.

**Escape radius** : champ `bailout`. Défaut **25** (`bailout_sqr=625`,
`const ESCAPE_TIME_BAILOUT`, `definitions.rs`) pour la famille escape-time
bytecode+perturbation — aligné F3 (`escape_radius=625`). Les types à sémantique
d'évasion particulière (Newton, Magnet, Sin, Nova, Pickover, densité, vectoriel,
AlphaMandelbrot) gardent leur propre bailout (souvent 4). Configurable par
pixel via `--bailout`/PNG.

**Anti-aliasing multi-sample** (`fractal/jitter.rs`) : chaque sample décale la
grille d'un offset sous-pixel low-discrepancy (Halton `radical_inverse` + tente
`triangle`, port F3 `hybrid.h`) posé dans `aa_subpixel_offset`, appliqué au
mapping pixel→c des 4 paths (f64/GMP/perturbation + cancellables) ; les rendus
colorisés sont moyennés en RGB. CLI `--aa-samples N`/`--jitter-scale` (boucle
dans `main.rs` via `io::png::colorize_to_rgb` + `save_png_rgb_with_metadata`) ;
GUI : dropdown **AA** (accumulation après les passes, `RenderMessage::AaProgress`).
CPU uniquement.

## Couleur

**27 palettes** (index 0-26) — Fire, Ocean, Forest, Violet, Rainbow, Sunset,
**Plasma** (défaut), Ice, Cosmic, Neon, Twilight, Emboss, Waves, SynthRed,
LightYears, Blues, Coffee, Classic, Dimensions, Earth, FireIce, Habs, Jays,
Slice, Stardust, Strobe, SynthBlue.

**3 espaces couleur** (`color_models.rs`) : RGB, HSB (circulaire),
LCH (CIE Lab, perceptuellement uniforme).

**15 modes coloring** (`OutColoringMode`) : Iter, IterPlus{Real,Imag,
RealImag,All}, BinaryDecomposition, Biomorphs, Potential,
ColorDecomposition, **Smooth** (défaut), OrbitTraps, Wings, Distance,
DistanceAO, Distance3D.

**4 orbit traps** (`orbit_traps.rs`) : Point, Line, Cross, Circle.

## Transformations de plan (XaoS-style)

| ID | Nom | Formule |
|----|-----|---------|
| 0 | Mu | c |
| 1 | Inversion | 1/c |
| 2 | InversionShifted | 1/(c + 0.25) |
| 3 | Lambda | 4·c·(1-c) |
| 4 | InversionLambda | 1/(4·c·(1-c)) |
| 5 | InversionLambdaMinus1 | 1/(4·c·(1-c)) - 1 |
| 6 | InversionSpecial | 1/(c - 1.40115) |

## Types de fractales (`--type N`)

| ID | Type | Algo |
|----|------|------|
| 1  | Von Koch | vectoriel |
| 2  | Dragon | vectoriel |
| 3  | Mandelbrot | **bytecode + perturbation** |
| 4  | Julia | **bytecode + perturbation** |
| 5  | Julia Sin | f64 / GMP |
| 6  | Newton | f64 / GMP |
| 7  | Phoenix | f64 / GMP |
| 8  | Buffalo | bytecode |
| 9–12 | Barnsley/Magnet (Julia + Mandelbrot) | f64 / GMP |
| 13 | Burning Ship | **bytecode + perturbation** |
| 14 | Tricorn | **bytecode + perturbation** |
| 15 | Mandelbulb | f64 / GMP (3D) |
| 16 | Buddhabrot | densité |
| 17 | Lyapunov | spécial (6 presets) |
| 18 | Perpendicular Burning Ship | bytecode |
| 19 | Celtic | bytecode |
| 20 | Alpha Mandelbrot | f64 / GMP |
| 21 | Pickover Stalks | f64 / GMP |
| 22 | Nova | f64 / GMP |
| 23 | Multibrot | bytecode (puiss. entières) sinon f64/GMP |
| 24 | Nebulabrot | densité |
| 25–30 | Julia variants (BS/Tricorn/Celtic/Buffalo/Multibrot/PerpBS) | bytecode |
| 31 | Alpha Mandelbrot Julia | f64 / GMP |
| 32 | Mandelbrot Sin | f64 / GMP |
| 33 | Anti-Buddhabrot | densité |

**Paires Mandelbrot ↔ Julia** (preview + touche `J`) : Mandelbrot↔Julia,
Barnsley↔Barnsley Julia, Magnet↔Magnet Julia, Burning Ship↔Burning Ship
Julia, Tricorn↔Tricorn Julia, Celtic↔Celtic Julia, Buffalo↔Buffalo Julia,
Multibrot↔Multibrot Julia, PerpBS↔PerpBS Julia, Alpha Mandelbrot↔Alpha
Mandelbrot Julia, Mandelbrot Sin↔Julia Sin.

## CLI (résumé — voir README pour la liste complète)

```
fractall-cli --type N --output FILE [OPTIONS]

  --type / --width / --height
  --center-x / --center-y (f64) ou --center-x-hp / --center-y-hp (string HP)
  --zoom EXPR (notation scientifique OK, ex: 1.41e219)
  --iterations N
  --algorithm auto|f64|perturbation|gmp
  --precision-bits N
  --palette 0-26 / --color-repeat N / --outcoloring MODE
  --aa-samples N          # AA multi-sample jitteré (1 = off), CPU only
  --jitter-scale F        # amplitude jitter sous-pixel (px, défaut 1.0)
  --plane N (0-6)
  --rotation DEG          # F3-style mat2 rotation, degrés CCW (CPU + GPU bytecode)
  --find-nucleus          # Mandelbrot nucleus refine (atom-domain)
  --dd-tier               # tier double-double ~106b (spirales deep sensibles)
  --no-bytecode           # désactive bytecode (debug)
  --gpu / --no-gpu        # override device ; sinon AUTO (wisdom::select_device,
                          #   G9.5 : benchmark machine + garde-fou correction —
                          #   GPU seulement plage deep both-perturbation)
  --enable-distance-estimation / --enable-interior-detection
  --multibrot-power F / --lyapunov-preset NAME
  --toml FILE             # charge real/imag/zoom/iterations/rotate
                          # format léger rust-fractal-core
  --wisdom-bench          # bench machine (G9.2) : débits par technique →
                          # ~/.config/fractall/wisdom.toml (--output non requis)
```

## GPU (wgpu)

- `mandelbrot_f32.wgsl`, `julia_f32.wgsl`, `burning_ship_f32.wgsl` — paths
  legacy f32 par type
- `perturbation.wgsl` — **kernel perturbation F3-strict en f64 natif**
  (2026-07-15, G9.4) : port de `pixel_loop.rs::iterate_pixel_unified_
  mandelbrot` (n/m séparés, rebasing strict, anti-over-skip BLA, PAS de
  glitch detection), buffers zref/BLA f64, span/offset en paires hi/lo f32.
  Exige `Features::SHADER_F64` (NVIDIA/AMD/Intel/llvmpipe OK ; **Metal NON**
  → pipeline absente → fallback CPU). ⚠️ Le double-float 2×f32 est
  IMPOSSIBLE en WGSL : naga/drivers réassocient les EFT (two_sum → 0, fma
  non-fusionné) — diagnostic : `cargo run --release --bin df64_gpu_probe`.
  Gate host : `ref_len-1 ≥ iter_max` sinon fallback CPU (réfs tronquées).
- `bytecode_kernel.wgsl` — runtime bytecode unifié (P3.1 Task 7). Applique la
  matrice K (rotation/transform) au mapping pixel→c, parité CPU/F3.

**Rotation/transform** : seul le path bytecode applique K sur GPU. Les autres
paths GPU (perturbation, shaders f32 dédiés) retombent sur le CPU quand
`transform_matrix().is_some()` (garde-fou anti-sortie-non-tournée).

Le path perturbation GPU (f64) est vérifié par `gpu-suite` jusqu'à **1e30**
(G9.4b, 2026-07-15) : le kernel gère les réfs tronquées comme le CPU (wrap
périodique + rebase-at-end atom-domain + guard BLA `lands_on_ref_end`) ; seule
la réf tronquée par ESCAPE retombe en CPU (per-pixel GMP requis). Range borné
à zoom ≲ 4e37 par le transport span f32 hi/lo (gate `GPU_SPAN_F32_MIN`). Perf
e30 1024² : GPU ~2× plus lent que CPU f64 16t (GeForce f64 1:64) — l'intérêt
du kernel deep est la CORRECTION (div 3e-4 là où le CPU f64 fait 0.034 sur
scène ultra-sensible, cf. TODO 9.4/9.6), pas la vitesse. **Backend** : macOS
Metal (sans SHADER_F64 → perturbation sur CPU) ; Linux Vulkan (prioritaire),
puis OpenGL ; Windows DX12 / Vulkan.

## GUI (`FractallApp`)

### Menu Type
- Racine : Mandelbrot, Barnsley Mandelbrot, Magnet Mandelbrot, Burning
  Ship, Perp. Burning Ship, Tricorn, Celtic, Buffalo, Multibrot, Alpha
  Mandelbrot, Mandelbrot Sin.
- **Julia all** (dossier) : toutes les variantes Julia.
- Mandelbulb, Julia Sin, Newton, Phoenix, Pickover Stalks, Nova après séparateur.
- **Densité** : Buddhabrot, Nebulabrot, Anti-Buddhabrot.
- **Lyapunov** : 6 presets.
- Von Koch / Dragon : CLI uniquement (`--type 1/2`).

### Fonctionnalités
- Rendu progressif multi-passes (preview → full).
- Recolorisation asynchrone (versioning pour ignorer les résultats obsolètes).
- Cache orbite + BLA entre re-rendus.
- **Réutilisation pixels inter-frame XaoS** (G10.4, `fractal/xaos.rs`) : en
  pan/zoom sans rotation, les colonnes/lignes de la frame précédente
  matchées à ≤ 0.5 px (positions vraies trackées `col_err`/`row_err`, aucune
  dérive cumulée) sont copiées au lieu d'être recalculées (~×40 en pan) ;
  param `xaos: Option<&XaosMap>` du dispatcher unique (CLI/quality/HQ/AA =
  `None`). **Zoom (2026-07-16)** : matching INJECTIF par axe (une colonne
  source → au plus une cible) — garantit ≥ (1−a)·n colonnes fraîches par axe
  en zoom-in (fin de l'« écho pur » du zoom ×2 aligné qui ne calculait RIEN
  et retardait l'image exacte), no-op en pan/zoom-out/previews. Raffinement
  exact silencieux à l'idle (400 ms, label `≈XaoS`, déclenché seulement si
  erreur réelle > ε) via `build_refine_map` : map UNION identité
  (`keep_union`) qui conserve tout pixel dont un axe est ENTIÈREMENT exact
  (`col_exact`/`row_exact` — ≠ « aligné » : une ligne copiée alignée peut
  être décalée par l'axe colonne, cf. pan horizontal) et ne recalcule que
  les approximations → cycle zoom ×2 écho+refine ≈ 107 % d'un rendu frais
  (image visible à ~0.8×, vs 148 % en refine total et ~200 % pré-injectivité).
  Frame source stockée par passe CPU uniquement (jamais GPU f32, jamais une
  passe écho-pur — aucune information nouvelle, dégraderait la source en
  copies de copies). Compatibilité = fingerprint JSON des params
  non-géométriques. Boucles pixel : point d'entrée unique
  `XaosMap::source_index(i, j)` (sémantique produit vs union).
  ⚠️ **Invariant : écho XaoS et reuse basse-résolution inter-passes sont
  mutuellement EXCLUSIFS** (dispatcher + `render_perturbation_with_cache`) :
  le `reuse` copie des centres décalés de (ratio−1)/2 px, ce qui
  contaminerait les axes que le map déclare FRAIS/exacts, consommés par le
  refine union (verrou `echo_pass_ignores_coarse_pass_reuse`). Les passes
  intermédiaires écho-pur sont SAUTÉES (le warp G10.1 affiche déjà le même
  contenu, en plus net — supprime le pompage flou preview→full en
  navigation) ; la passe finale tourne toujours. Diagnostics :
  `xaos_pan_speedup_diagnostic`, `xaos_zoom_cycle_diagnostic` (`--ignored`).
- **File de tuiles priorité-centre** (G10.5, `render/tiles.rs`) : les 4
  boucles pixel CPU (f64/GMP/perturbation/perturbation-GMP) tournent sur une
  work-queue de tuiles (16/32/64 px, ≥ 8/thread) ordonnée par distance au
  curseur (`hover_norm`, centre sinon) — file atomique `fetch_add` (l'ordre
  rayon natif disperse les fronts). Param `tiles: Option<&TileOpts>` du
  dispatcher unique (CLI/quality/HQ/AA/refine = `None` ; l'ordre ne change
  AUCUN pixel, verrou `tiled_render_identical_across_priorities`). Streaming
  intra-passe GUI : le sink colorise chaque tuile, la blitte sur la passe
  précédente colorisée et envoie `TileProgress` throttlé 100 ms → la zone
  sous le curseur devient nette en premier. Sans `unsafe` : segments de
  lignes disjoints par tuile (`TileGrid::split`). Gates sink : refine
  silencieux, OrbitTraps/Wings, 1re passe (pas de base). Cancel : poll par
  tuile + par ligne (GMP/perturbation) + re-check final anti-buffer-troué.
- Coordonnées HP synchronisées vers `FractalParams`. ⚠️ L'arithmétique HP des
  zooms (`zoom_hp`/`zoom_anchored_hp`/`zoom_rect_hp`/`zoom_out_hp`) utilise `hp_arith_precision()`
  (≈ `-log2(span)+96` bits, **dynamique**), PAS le `HP_PRECISION` fixe (256 b) :
  sinon le centre est arrondi au zoom (à 1e235 il faut ~783 b) → vue fausse →
  **image uniforme** après zoom. Les sync HP↔f64 gardent 256 b (f64 = 53 b).
- Sélection rectangle pour zoom (méthode préférée), switch CPU/GPU, stats.
  Clic gauche = zoom in re-centré sur le point ; clic droit = zoom out re-centré
  sur le curseur (symétrique).
- Preview Julia au survol (touche `J` pour basculer).
- Rendu haute résolution asynchrone (Window / 4K / 8K).
- Drag-and-drop PNG pour restaurer l'état. Save (S) embed metadata JSON.

### Raccourcis clavier

| Touche | Action |
|--------|--------|
| F1–F12 | Type fractale (F1=Mandelbrot, F2=Julia, F3=JuliaSin, …, F12=Tricorn) |
| C | Cycler palette |
| R | Cycler color_repeat |
| J | Bascule Julia / preview |
| S | Screenshot PNG (avec metadata) |
| +/= | Zoom avant (×1.5) |
| - | Zoom arrière (×1.5) |
| 0 | Reset vue (ignoré si focus sur champ itérations) |
| Enter | Valider champ itérations |
| Molette | Zoom in/out continu ANCRÉ au curseur (`zoom_anchored_hp`, ≈×1.2/cran) |
| Clic gauche + drag | Sélection rectangle zoom |
| Clic droit | Zoom arrière |
| Clic milieu + drag | Pan |

+/=/-/0 désactivés en mode preview Julia.

## Threading

- **Rendu progressif** : thread dédié par passe, mpsc channels.
- **Recolorisation** : thread séparé + versioning pour annuler les résultats
  obsolètes au glissement rapide du slider.
- **Rendu HQ** : thread dédié, messages `Progress/Done/Error`.
- **Parallélisme pixel** : `rayon::par_chunks_mut`. `AtomicBool` pour
  l'annulation propre.

## Quality (`fractall-quality`)

Compare le chemin perturbation à un rendu GMP pur pixel-par-pixel. Garde-fou
de régression à chaque modification de `perturbation/`.

```bash
cargo run --release --bin fractall-quality -- list
cargo run --release --bin fractall-quality -- preset seahorse-valley
cargo run --release --bin fractall-quality -- suite --width 256 --height 256
cargo run --release --bin fractall-quality -- compare --type 3 \
    --center-x-hp "-0.7436..." --center-y-hp "0.1318..." --zoom 1e10
cargo run --release --bin fractall-quality -- gpu-suite   # parité GPU↔CPU (G9.5)
```

**Parité GPU↔CPU** (`gpu-suite` / `gpu-compare`, 2026-07-15) : rend la même
vue via `GpuRenderer::render_dispatch` et la juge contre le **GMP pur**
(⚠️ PAS le CPU Auto : le f64-std diverge lui-même de ~6 % de la vérité à
5000 iters sur bord chaotique), mêmes métriques/verdicts, rapports sous
`quality-reports/gpu/` (`pert.*` = GPU, `gmp.*` = juge). Presets
`GPU_PRESETS` (échelle seahorse 1e2→1e8). **État (kernel perturbation
F3-strict f64 natif, 2026-07-15) : WARN p99=0 div ≈ 0.001 sur tout le range
perturbation 1e4→1e8 (= niveau CPU-perturbation) ; FAIL 1e2/1e3 (shaders std
f32, gap restant G9.5)**. C'est le harnais d'acceptance du kernel deep G9.4.
Les tests unitaires du bin quality tournent en CI.

**Sorties** dans `quality-reports/<preset>/` :
- `pert.png` / `gmp.png` (metadata pour drag-and-drop fractall-gui).
- `diff.png` heatmap (noir=match, rouge-jaune=divergence).
- `report.md` métriques + top 10 pixels divergents.
- `suite-summary.md` global PASS/WARN/FAIL.

**Métriques** : `|iter_pert - iter_gmp|` (max, mean, rms, p50/p95/p99),
ratio divergence (>1), `|z_pert - z_gmp|`, erreur relative, désaccord
d'échappement, temps pert vs GMP.

**Seuils défaut (recalibrés G6, 2026-07-10 — robustes au bruit de bord)** :
- **PASS** : `max_iter_diff ≤ 1` ET `div_ratio ≤ 0.001` (quasi-exact, strict).
- **FAIL** : `p99_iter_diff > 1` (divergence LARGE : > 1 % des pixels divergent
  de > 1 → vrai bug) **OU** `div_ratio > 0.01` (SYSTÉMATIQUE : offset uniforme,
  signature over-skip BLA…).
- **WARN** : sinon (divergence éparse — quelques pixels de bord au plancher f64,
  cf. e13/e17/seahorse-valley : `max` grand mais `p99=0`, `div_ratio` minuscule).

⚠️ Le gate ne FAIL PLUS sur le `max` outlier seul (ancien comportement) : un `max`
élevé sur quelques pixels DISPERSÉS = bruit inhérent, pas une régression. `max`
reste rapporté dans `report.md`. Override : `--pass-max-iter-diff`,
`--fail-p99-iter-diff`, `--warn-divergence-ratio`.

**14 presets** (`src/quality/presets.rs`) : Mandelbrot (seahorse 1e8,
activation 1e13, GMP perturbation 1e17, Misiurewicz 1e12, minibrot 1e18,
spirales profondes e30/e50/e100), Julia (seed -0.8+0.156i à 1e10), Burning
Ship antenna 1e9 (non-conformal BLA), Tricorn spiral 1e8, et frontières
lisses hors axes de pliage Celtic/Buffalo/PerpBS 1e9 (G3 2026-07-13 — les
frontières hirsutes de ces familles sont à sensibilité de précision extrême,
GMP-256 non convergé, donc non comparables).

**Perf** : GMP pur O(1e3-1e4) plus lent que perturbation, d'où la résolution
défaut 256×256. La suite peut prendre plusieurs minutes au-delà de 1e15.

**Boucle d'auto-amélioration** :
1. Après toute modif `perturbation/` : `fractall-quality suite`.
2. Lire `suite-summary.md`, identifier FAIL.
3. Pour chaque FAIL, lire `report.md` + top 10 divergents.
4. Localiser via coordonnées pixel ↔ modules `bla/delta/nonconformal/glitch/orbit`.
5. Patcher, relancer, vérifier l'amélioration.

## Référence Fraktaler-3.1

Source de vérité algorithmique. Submodule présent dans
`fraktaler-3-3.1/src/`. Analyse complète dans `docs/fraktaler-3-analysis.md` ;
roadmap parité + perf dans `TODO.md` (goals G1/G2).

Fichiers F3 clés à consulter :
- `hybrid.h` / `hybrid.cc` — hybrid_period (atom-domain), hybrid_center
  (Newton), hybrid_size (matrice K), hybrid_render, opcodes
- `bla.h` / `bla.cc` — blaR2 multi-niveaux, merge F3, dual-number BLA
- `engine.cc` — pipeline complet newton_thread + render_thread
- `floatexp.h` / `softfloat.h` / `float128.h` — hiérarchie de types
- `wisdom.cc` / `wisdom.h` — sélection auto de précision (TOML persistant)
- `dual.h` — propagation Jacobienne par dual-numbers
