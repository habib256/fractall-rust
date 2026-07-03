# Fraktaler 3 -- Linux batch-only build (no GUI, no OpenCL) with OpenEXR.
# GMP/MPFR/MPC dev packages are NOT installed system-wide; we reuse the
# static libs + headers that the fractall-rust `rug`/gmp-mpfr-sys crate built
# under target/release/build. OpenEXR 3.1 comes from system libopenexr-dev.
# Made for benchmarking against fractall-rust (scripts/compare_f3.py).

COMPILER = g++

# rug-built GMP/MPFR/MPC (static). Overrides the missing system libmpfr-dev.
RUG := /home/gistarcade/src/fractall-rust/target/release/build/gmp-mpfr-sys-45ab1725c6e1aa16/out

# Vendored header-only deps (no libglm-dev / libtoml11-dev on this host).
VENDOR := /home/gistarcade/src/fractall-rust/fraktaler-3-3.1/vendor
# toml11 v3.8.1 => API 3 (single-header at vendor/toml11-src/toml.hpp).
override TOML11 := 3

# Batch-only: no GUI (gui.cc stubbed via batch_stubs.cc), no OpenCL.
override CL :=
override FLOAT128 :=
# Drop mpfr from the pkg-config LIBS list (mpfr has no .pc here);
# mpfr/gmp come in via -L$(RUG)/lib below. libpng/libjpeg/zlib/sdl2 have .pc.
# sdl2 stays: version.cc includes <SDL.h> unconditionally even in batch mode
# (gui.cc itself is stubbed, so no SDL runtime symbols are actually used).
# OpenEXR 3.1 + libdeflate (system dev pkgs) => HAVE_EXR=3, enables the batch
# .exr output read by scripts/compare_f3.py (same wiring as macos-batch.mk).
override LIBS := libpng libjpeg zlib sdl2 OpenEXR libdeflate
override LIBS_GUI :=
override LIBS_CL :=
override LIBS_IMGUI :=

CFLAGS += -std=$(STDCXX) -Wall -Wextra -pedantic -O3 -march=native -MMD -DHAVE_EXR=3 -I$(RUG)/include -I$(VENDOR)/glm-src -I$(VENDOR)/toml11-src -I$(VENDOR)
CPPFLAGS += -I$(RUG)/include -I$(VENDOR)/glm-src -I$(VENDOR)/toml11-src -I$(VENDOR)
LDFLAGS += -L$(RUG)/lib -lstdc++ -lstdc++fs -lm
# mpfr must precede gmp for static link resolution.
CLFLAGS += -lmpfr -lgmp

OEXT = .linux.o
EXEEXT = .linux
TARGETS = fraktaler-3-$(VERSION)$(EXEEXT)
