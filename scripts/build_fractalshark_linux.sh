#!/usr/bin/env bash
# Build FractalSharkCli (Linux expérimental, CUDA) comme 2e référence VITESSE
# Mandelbrot-deep du harness (cf. TODO §G8.2, 2026-07-14).
#
# Prérequis :
#   - clone : ~/src/FractalShark (GIT_LFS_SKIP_SMUDGE=1 ok pour le CLI)
#   - CUDA toolkit (nvcc) — install utilisateur : sudo apt install cuda-toolkit
#   - clang/clang++, cmake, libgmp-dev (déjà là pour rug/F3)
#   - GPU NVIDIA (cette machine : RTX 4060 Ti 16 GB, driver 595)
#
# Sortie : ~/src/FractalShark/build-release/.../FractalSharkCli
# Smoke test inclus (list-render-algorithms + rendu 128² CPU + GPU).
#
# NB vitesse harness : FractalSharkCli ne sort que du PNG (pas d'EXR N0/NF) →
# utilisable pour l'axe speed head-to-head uniquement ; la correction reste
# jugée par notre GMP. Sémantique de --zoom à calibrer au premier run
# (comparer le champ de vue vs fractall sur une vue connue).

set -euo pipefail

FS_DIR="${FS_DIR:-$HOME/src/FractalShark}"
CUDA_BIN="${CUDA_BIN:-/usr/local/cuda/bin}"

die() { echo "ERREUR : $*" >&2; exit 1; }

[ -d "$FS_DIR" ] || die "clone absent : $FS_DIR (git clone https://github.com/mattsaccount364/FractalShark.git)"

# --- Gate CUDA (l'install apt peut prendre du temps) ---
if command -v nvcc >/dev/null 2>&1; then
    NVCC="$(command -v nvcc)"
elif [ -x "$CUDA_BIN/nvcc" ]; then
    NVCC="$CUDA_BIN/nvcc"
    export PATH="$CUDA_BIN:$PATH"
else
    die "nvcc introuvable — installer le CUDA toolkit d'abord (sudo apt install cuda-toolkit), puis relancer."
fi
echo "nvcc : $NVCC ($("$NVCC" --version | tail -1))"

for tool in clang clang++ cmake; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool introuvable (sudo apt install clang cmake)"
done

cd "$FS_DIR"

# LFS best-effort : les assets (Pics/) servent surtout au GUI ; le CLI peut
# builder sans. On tente si git-lfs est là, sinon on continue.
if command -v git-lfs >/dev/null 2>&1; then
    git lfs install --local >/dev/null 2>&1 || true
    git lfs pull || echo "warn: git lfs pull a échoué — on continue (CLI seulement)"
else
    echo "warn: git-lfs absent — on continue (CLI seulement)"
fi

# Release only (leur build_linux.sh fait Debug+Release : inutile ici).
cmake -S . -B build-release \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CUDA_HOST_COMPILER=g++ \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
    -DCMAKE_CUDA_FLAGS_RELEASE="-O3 -DNDEBUG"
cmake --build build-release --parallel --target FractalSharkCli

CLI="$(find build-release -name FractalSharkCli -type f -executable | head -1)"
[ -n "$CLI" ] || die "binaire FractalSharkCli introuvable après build"
echo "binaire : $FS_DIR/$CLI"

# --- Smoke tests ---
echo "--- algorithmes disponibles (extrait) ---"
"$CLI" --list-render-algorithms | head -20

OUT=$(mktemp -d)
echo "--- rendu CPU 128² (seahorse 1e8) ---"
time "$CLI" --render-algorithm Cpu64PerturbedBLAV2HDR \
    --center-x "-0.743643887037158704752191506114774" \
    --center-y 0.131825904205311970493132056385139 \
    --zoom 1e8 --iterations 4096 --width 128 --height 128 \
    --out "$OUT/fs-cpu.png" || echo "warn: rendu CPU KO (vérifier noms d'algos/format --center-x négatif)"

echo "--- rendu GPU 128² (même vue) ---"
time "$CLI" --render-algorithm Gpu1x32PerturbedLAv2 \
    --center-x "-0.743643887037158704752191506114774" \
    --center-y 0.131825904205311970493132056385139 \
    --zoom 1e8 --iterations 4096 --width 128 --height 128 \
    --out "$OUT/fs-gpu.png" || echo "warn: rendu GPU KO (driver/algo — voir --list-render-algorithms)"

ls -la "$OUT"
echo
echo "OK. Prochaines étapes (TODO §G8.2) :"
echo "  1. calibrer la sémantique --zoom vs fractall (même vue → même cadrage ?)"
echo "  2. valider visuellement les PNG ($OUT)"
echo "  3. câbler la colonne speed Mandelbrot-deep dans scripts/harness.py (FS_BIN=$FS_DIR/$CLI)"
