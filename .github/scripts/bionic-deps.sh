#!/usr/bin/env bash
# Prépare un conteneur Ubuntu 18.04 (bionic, glibc 2.27) pour bâtir fractall
# ET son AppImage. Utilisé par le job `linux-glibc227` de release.yml, et
# reproductible en local à l'identique :
#
#   docker run --rm -v "$PWD":/w -w /w \
#       -e CARGO_HOME=/w/.cargo-home -e RUSTUP_HOME=/w/.rustup-home \
#       -e RUSTFLAGS="-C target-cpu=x86-64-v2" \
#       ubuntu:18.04 bash -eux -c \
#       '.github/scripts/bionic-deps.sh && . "$CARGO_HOME/env" \
#        && packaging/linux/build_appimage.sh'
#
# Pourquoi bionic : la glibc n'est jamais embarquée dans un AppImage ni dans le
# tar.gz ; celle de la machine de build devient donc le PLANCHER de
# compatibilité. 18.04 = glibc 2.27, soit « toute distribution depuis 2018 ».
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update

# build-essential/m4/make/diffutils : exigés par le build source de GMP/MPFR/MPC
#   (rug → gmp-mpfr-sys compile ces libs depuis les sources).
# libx11/xcursor/xrandr/xi/xkbcommon/wayland/GL : dépendances du binaire GUI
#   (eframe features x11 + wayland).
# wget/file : outillage AppImage (téléchargement + inspection par linuxdeploy).
# desktop-file-utils : `desktop-file-validate`, appelé par appimagetool sur le
#   .desktop de l'AppDir.
apt-get install -y --no-install-recommends \
    build-essential m4 make diffutils pkg-config \
    curl ca-certificates wget file desktop-file-utils \
    libx11-dev libxcursor-dev libxrandr-dev libxi-dev \
    libxkbcommon-dev libwayland-dev libgl1-mesa-dev

curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable
