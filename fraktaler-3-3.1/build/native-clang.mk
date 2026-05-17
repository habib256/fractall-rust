# Fraktaler 3 -- fast deep escape time fractals
# Copyright (C) 2021-2025 Claude Heiland-Allen
# SPDX-License-Identifier: AGPL-3.0-only

COMPILER = clang
CFLAGS += -std=$(STDCXX) -Wall -Wextra -pedantic -O3 -mtune=native -MMD -DHAVE_GUI -DHAVE_ICON -DHAVE_EXR=$(EXR) $(CL)
LDFLAGS += -lstdc++ -lstdc++fs -lm
LIBS_IMGUI += -ldl
LIBS_CL += OpenCL
OEXT = .native-clang.o
EXEEXT = .clang
TARGETS = fraktaler-3-$(VERSION)$(EXEEXT)
