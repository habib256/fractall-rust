# Fraktaler 3 -- fast deep escape time fractals
# Copyright (C) 2021-2025 Claude Heiland-Allen
# SPDX-License-Identifier: AGPL-3.0-only

EMSCRIPTEN_HOME ?= $(HOME)/opt/emscripten/pthreads-exceptions
PKG_CONFIG_PATH = $(EMSCRIPTEN_HOME)/lib/pkgconfig
COMPILER = em++
CFLAGS += -std=$(STDCXX) -Wall -Wextra -pedantic -fexceptions -O3 -MMD -s USE_SDL=2 -s USE_PTHREADS
CPPFLAGS += -I$(EMSCRIPTEN_HOME)/include -I$(EMSCRIPTEN_HOME)/include/Imath -I../emscripten-browser-clipboard -I../emscripten-browser-file -DHAVE_GUI -DHAVE_EXR=3 -DEMSCRIPTEN_BROWSER_CLIPBOARD_GIT_VERSION_STRING="\"$(shell cd ../emscripten-browser-clipboard && git describe --always --tags --dirty=+)\"" -DEMSCRIPTEN_BROWSER_FILE_GIT_VERSION_STRING="\"$(shell cd ../emscripten-browser-file && git describe --always --tags --dirty=+)\""
LDFLAGS += -fexceptions -L$(EMSCRIPTEN_HOME)/lib -lgmp -lmpfr -lOpenEXR-3_3 -lOpenEXRCore-3_3 -lIex-3_3 -lIlmThread-3_3 -lImath-3_1 -ldeflate -ljpeg -lpng -lz -lidbfs.js -s USE_SDL=2 -s ALLOW_MEMORY_GROWTH=1 -s USE_PTHREADS -s PTHREAD_POOL_SIZE="(navigator.hardwareConcurrency+2)" -s MIN_WEBGL_VERSION=2 -s MAX_WEBGL_VERSION=2 -s ASYNCIFY=1 -s EXPORTED_RUNTIME_METHODS=ccall -s EXPORTED_FUNCTIONS="['_main','_malloc','_free']"
OEXT = .emscripten.o
EXEEXT =
TARGETS = live/$(VERSION)/index.html
