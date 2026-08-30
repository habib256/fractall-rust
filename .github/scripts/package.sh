#!/usr/bin/env bash
# Assemble les binaires release dans `dist/` puis produit une archive.
#
# Usage : .github/scripts/package.sh <label-cible>
#   ex.  .github/scripts/package.sh linux-x86_64-glibc2.27
#
# Sortie : dist/fractall-<version>-<label>/            (répertoire déployable)
#          dist/fractall-<version>-<label>.tar.gz|.zip (archive uploadée)
#
# Partagé par les 4 jobs de release.yml pour que le contenu d'une distribution
# soit défini à UN seul endroit (binaires, doc, DLL Windows).
set -euo pipefail

LABEL="${1:?usage: package.sh <label-cible>}"
BINS=(fractall-cli fractall-gui fractall-quality)

# Version : logique partagée avec appimage.sh (tag > Cargo.toml + sha).
# shellcheck source=.github/scripts/version.sh
source "$(dirname "${BASH_SOURCE[0]}")/version.sh"
VERSION="$(fractall_version)"

NAME="fractall-${VERSION}-${LABEL}"
OUT="dist/${NAME}"
rm -rf "${OUT}"
mkdir -p "${OUT}"

# Extension exécutable : cible Windows uniquement.
EXT=""
case "${LABEL}" in windows-*) EXT=".exe" ;; esac

for bin in "${BINS[@]}"; do
  cp "target/release/${bin}${EXT}" "${OUT}/"
done

# Le profil release fait déjà `strip = true` ; on ne re-strippe pas (le faire
# sur macOS casserait la signature ad-hoc des binaires arm64).

cp README.md "${OUT}/" 2>/dev/null || true

# Windows : la toolchain GNU laisse des dépendances au runtime MinGW. Sans ces
# DLL à côté des .exe, le lancement échoue sur une machine sans MSYS2.
if [[ "${LABEL}" == windows-* ]]; then
  for dll in libgcc_s_seh-1.dll libwinpthread-1.dll libstdc++-6.dll; do
    if [[ -f "/mingw64/bin/${dll}" ]]; then
      cp "/mingw64/bin/${dll}" "${OUT}/"
    fi
  done
fi

# Archive : zip côté Windows, tar.gz ailleurs (préserve le bit exécutable).
if [[ "${LABEL}" == windows-* ]]; then
  (cd dist && zip -qr "${NAME}.zip" "${NAME}")
else
  for bin in "${BINS[@]}"; do
    chmod +x "${OUT}/${bin}"
  done
  tar -czf "dist/${NAME}.tar.gz" -C dist "${NAME}"
fi

echo "── contenu de dist/ ──"
ls -lh dist/
