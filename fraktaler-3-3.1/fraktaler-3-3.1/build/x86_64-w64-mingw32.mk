# Fraktaler 3 -- fast deep escape time fractals
# Copyright (C) 2021-2025 Claude Heiland-Allen
# SPDX-License-Identifier: AGPL-3.0-only

WIN ?= $(HOME)/opt/windows/posix/x86_64
CLEWPREFIX ?= ../clew
CLFLAGS = -I$(CLEWPREFIX)/include -Dclew_STATIC $(CLEWPREFIX)/src/clew.c -Wno-cast-function-type
PKG_CONFIG_PATH = $(WIN)/lib/pkgconfig
PKG_CONFIG_FLAGS = --static
PKG_CONFIG_SED = s/-pthread/-Wl,-Bstatic -lstdc++ -lstdc++fs -lpthread -Wl,-Bdynamic/g
COMPILER = x86_64-w64-mingw32-g++
STRIP = x86_64-w64-mingw32-strip
WINDRES = x86_64-w64-mingw32-windres
GWINDRES = /usr/bin/x86_64-w64-mingw32-windres
CFLAGS += -std=$(STDCXX) -Wall -Wextra -pedantic -O3 -MMD
CPPFLAGS += -D__USE_MINGW_ANSI_STDIO=1 -DWINVER=0x501 -D_WIN32_WINNT=0x501 -I$(WIN)/include -I$(WIN)/include/OpenEXR -DHAVE_EXR=$(EXR) -I$(CLEWPREFIX)/include -Dclew_STATIC -DHAVE_CLEW $(CL) -DHAVE_GUI
LDFLAGS += -static -static-libgcc -static-libstdc++ -static -L$(WIN)/lib -lopengl32
OEXT = .win32-x86_64.o
EXEEXT = .x86_64.exe
TARGETS = fraktaler-3-$(VERSION)$(EXEEXT)
RESOURCES = src/resource$(OEXT)
