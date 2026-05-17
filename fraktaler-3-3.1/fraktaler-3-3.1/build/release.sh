#!/bin/sh
# Fraktaler 3 -- fast deep escape time fractals
# Copyright (C) 2021-2025 Claude Heiland-Allen
# SPDX-License-Identifier: AGPL-3.0-only

make -B VERSION.txt
make -j $(nproc) SYSTEM=i686-w64-mingw32 "$@"
make -j $(nproc) SYSTEM=x86_64-w64-mingw32 "$@"
make -j $(nproc) SYSTEM=armv7-w64-mingw32 FLOAT128= "$@"
make -j $(nproc) SYSTEM=aarch64-w64-mingw32 FLOAT128= "$@"
emmake make -j $(nproc) SYSTEM=emscripten FLOAT128= "$@"
make -j $(nproc) SYSTEM=native-gcc "$@"
make -j $(nproc) SYSTEM=native-clang "$@"
make -j $(nproc) SYSTEM=docs "$@"
make release
