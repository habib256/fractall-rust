# Fraktaler 3 -- fast deep escape time fractals
# Copyright (C) 2021-2025 Claude Heiland-Allen
# SPDX-License-Identifier: AGPL-3.0-only

COMPILER = g++
CFLAGS += -std=$(STDCXX) -Wall -Wextra -pedantic -O3 -march=native -MMD -DHAVE_GUI -DHAVE_ICON -DHAVE_EXR=$(EXR) $(CL)
LDFLAGS += -lstdc++ -lstdc++fs -lm
LIBS_IMGUI += -ldl
LIBS_CL += OpenCL
OEXT = .native-gcc.o
EXEEXT = .gcc
TARGETS = fraktaler-3-$(VERSION)$(EXEEXT)
