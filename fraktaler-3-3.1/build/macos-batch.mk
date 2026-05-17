# Fraktaler 3 -- macOS batch-only build (no GUI, no OpenCL, no float128)
# Avec OpenEXR activé (HAVE_EXR=3) pour permettre la sortie raw EXR utilisée
# par le harness de parité de fractall-rust (scripts/compare_f3.py).
# Made for benchmarking against fractall-rust.

COMPILER = clang++
# Batch-only: no GUI (imgui forces GLES3 which doesn't ship on macOS), no OpenCL.
# OpenEXR 3.x est requis (brew install openexr) — détecté via pkg-config.
PKG_CONFIG_PATH := /opt/homebrew/lib/pkgconfig:$(PKG_CONFIG_PATH)
export PKG_CONFIG_PATH
# macOS clang n'a pas __float128 / libquadmath — désactive complètement.
FLOAT128 :=
override FLOAT128 :=
CFLAGS += -std=$(STDCXX) -Wall -Wextra -pedantic -O3 -mtune=native -MMD -I/opt/homebrew/include -DHAVE_EXR=3
CPPFLAGS += -I/opt/homebrew/include
LDFLAGS += -L/opt/homebrew/lib -lm -lstdc++
LIBS += OpenEXR
LIBS_IMGUI +=
LIBS_CL +=
OEXT = .macos.o
EXEEXT = .macos
TARGETS = fraktaler-3-$(VERSION)$(EXEEXT)
