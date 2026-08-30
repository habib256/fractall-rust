#!/bin/bash
# Build d'un AppImage fractall pour Linux x86_64.
#
# ⚠️ À BÂTIR SOUS UBUNTU BIONIC (18.04, glibc 2.27) — c'est la contrainte
# structurante d'un AppImage : la glibc n'est JAMAIS embarquée, donc le binaire
# tourne au mieux sur une glibc ≥ celle de la machine de build. Le job
# `linux-glibc227` de .github/workflows/release.yml fait exactement ça dans un
# conteneur `ubuntu:18.04`. En local :
#
#   docker run --rm -v "$PWD":/w -w /w ubuntu:18.04 \
#       bash -c '.github/scripts/bionic-deps.sh && packaging/linux/build_appimage.sh'
#
# Stratégie (calquée sur packaging/linux/build_appimage.sh de POM1) :
#   1. `cargo build --release --bins` (sautable : FRACTALL_APPIMAGE_SKIP_BUILD=1).
#   2. Layout AppDir miroir d'une install /usr :
#        usr/bin/{fractall-gui,fractall-cli,fractall-quality}
#        usr/lib/                      libs déployées par linuxdeploy
#        usr/share/applications/fractall.desktop
#        usr/share/icons/hicolor/256x256/apps/fractall.png
#        usr/share/doc/fractall/README.md
#   3. AppRun (packaging/linux/AppRun) : dispatch GUI/CLI/quality + LD_LIBRARY_PATH.
#   4. linuxdeploy bundle les libs non-blacklist, appimagetool emballe.
#
# L'ICÔNE EST UN RENDU FRAIS produit par le `fractall-cli` qu'on vient de bâtir
# (256², Mandelbrot, palette par défaut) : pas d'asset binaire versionné à
# maintenir, et une icône qui ne peut pas mentir sur ce que produit le moteur.
#
# Sortie : dist/fractall-<version>-x86_64.AppImage

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=.github/scripts/version.sh
source "${REPO_ROOT}/.github/scripts/version.sh"
VERSION="${FRACTALL_VERSION:-$(fractall_version)}"

DIST="${REPO_ROOT}/dist"
WORK="${REPO_ROOT}/build-appimage"
APPDIR="${WORK}/AppDir"
TOOLS="${WORK}/tools"
BINS=(fractall-gui fractall-cli fractall-quality)

# 1. Build — TOUJOURS recompiler par défaut : un simple test d'existence
#    laisserait un binaire périmé entrer dans l'AppImage (piège vécu côté POM1).
if [ "${FRACTALL_APPIMAGE_SKIP_BUILD:-0}" != "1" ]; then
    echo "[appimage] cargo build --release --locked --bins…"
    cargo build --release --locked --bins
fi
for bin in "${BINS[@]}"; do
    if [ ! -f "target/release/${bin}" ]; then
        echo "[appimage] ERREUR : target/release/${bin} absent" \
             "(FRACTALL_APPIMAGE_SKIP_BUILD=1 sans build préalable ?)." >&2
        exit 1
    fi
done

# 2. Outils : téléchargés PUIS EXTRAITS — tourner sans FUSE (conteneur, CI,
#    sandbox) sinon `--appimage-extract-and-run` recopie l'archive à chaque appel.
mkdir -p "${TOOLS}"
fetch_extract() {
    local url="$1" name="$2"
    [ -d "${TOOLS}/${name}.AppDir" ] && return 0
    echo "[appimage] Téléchargement de ${name}…"
    if command -v wget >/dev/null 2>&1; then
        wget -q "${url}" -O "${TOOLS}/${name}.AppImage"
    else
        curl -sSL "${url}" -o "${TOOLS}/${name}.AppImage"
    fi
    chmod +x "${TOOLS}/${name}.AppImage"
    (cd "${TOOLS}" && "./${name}.AppImage" --appimage-extract >/dev/null \
        && mv squashfs-root "${name}.AppDir")
}
fetch_extract "https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage" linuxdeploy
fetch_extract "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage" appimagetool

# 3. AppDir from scratch.
rm -rf "${APPDIR}"
mkdir -p "${APPDIR}/usr/bin" \
         "${APPDIR}/usr/lib" \
         "${APPDIR}/usr/share/applications" \
         "${APPDIR}/usr/share/icons/hicolor/256x256/apps" \
         "${APPDIR}/usr/share/doc/fractall"

for bin in "${BINS[@]}"; do
    cp "target/release/${bin}" "${APPDIR}/usr/bin/${bin}"
done
# Le profil release fait déjà `strip = true` ; on ne re-strippe pas.

cp "${REPO_ROOT}/packaging/linux/AppRun" "${APPDIR}/AppRun"
chmod +x "${APPDIR}/AppRun"
cp "${REPO_ROOT}/packaging/linux/fractall.desktop" "${APPDIR}/fractall.desktop"
cp "${REPO_ROOT}/packaging/linux/README.txt" "${APPDIR}/README.txt"
cp "${REPO_ROOT}/README.md" "${APPDIR}/usr/share/doc/fractall/README.md"

# 3a. Icône = rendu frais du moteur qu'on emballe. `FRACTALL_APPIMAGE_ICON`
#     fournit un PNG tout fait à la place — utile quand la machine qui empaquette
#     ne peut pas EXÉCUTER le binaire produit (cross-build, ou binaire bâti pour
#     un CPU plus récent que l'hôte).
ICON="${APPDIR}/fractall.png"
if [ -n "${FRACTALL_APPIMAGE_ICON:-}" ]; then
    echo "[appimage] Icône fournie : ${FRACTALL_APPIMAGE_ICON}"
    cp "${FRACTALL_APPIMAGE_ICON}" "${ICON}"
else
    echo "[appimage] Rendu de l'icône (fractall-cli, 256²)…"
    "${APPDIR}/usr/bin/fractall-cli" \
        --type 3 --width 256 --height 256 --iterations 500 --no-gpu \
        --output "${ICON}"
fi
cp "${ICON}" "${APPDIR}/usr/share/icons/hicolor/256x256/apps/fractall.png"
ln -sf fractall.png "${APPDIR}/.DirIcon"

# 4. linuxdeploy : bundle les libs (rpath = $ORIGIN/../lib) + recopie
#    desktop/icône aux emplacements standards. Les trois binaires sont passés
#    en --executable pour que les deps de la GUI **et** des outils console
#    soient déployées. NO_STRIP=1 : le profil release a déjà strippé.
#    ⚠️ libGL/libEGL/libvulkan restent HORS de l'AppImage (blacklist
#    linuxdeploy) : ces libs sont l'ABI du pilote graphique de l'hôte, les
#    embarquer casse l'accélération — wgpu les dlopen depuis le système.
EXEC_ARGS=()
for bin in "${BINS[@]}"; do
    EXEC_ARGS+=("--executable=${APPDIR}/usr/bin/${bin}")
done
NO_STRIP=1 "${TOOLS}/linuxdeploy.AppDir/AppRun" \
    --appdir="${APPDIR}" \
    "${EXEC_ARGS[@]}" \
    --desktop-file="${APPDIR}/fractall.desktop" \
    --icon-file="${APPDIR}/fractall.png"

# 5. appimagetool : assemble et compresse.
mkdir -p "${DIST}"
OUT="${DIST}/fractall-${VERSION}-x86_64.AppImage"
rm -f "${OUT}"
PATH="${TOOLS}/appimagetool.AppDir/usr/bin:${PATH}" \
ARCH=x86_64 \
VERSION="${VERSION}" \
"${TOOLS}/appimagetool.AppDir/usr/bin/appimagetool" "${APPDIR}" "${OUT}"

echo
echo "[appimage] OK → ${OUT}"
ls -lh "${OUT}"
