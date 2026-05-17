# Fraktaler 3 -- fast deep escape time fractals
# Copyright (C) 2021-2025 Claude Heiland-Allen
# SPDX-License-Identifier: AGPL-3.0-only

VERSION ?= $(shell test -d .git && git describe --always --dirty=+ || (cat VERSION.txt | head -n 1))
DATE ?= $(shell test -d .git && date --iso || (cat VERSION.txt | tail -n+1 | head -n 1))

SOURCE := $(shell cat INDEX.txt)

OPENEXR_VERSION_MAJOR := $(shell (PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --modversion OpenEXR 2>/dev/null || echo 0) | sed "s/[.].*//g")

IMGUI_GIT_VERSION_STRING := $(shell test -d ../imgui && cd ../imgui && git describe --tags --always --dirty=+ || echo none)
IMGUI_FILEBROWSER_GIT_VERSION_STRING := $(shell test -d ../imgui-filebrowser && cd ../imgui-filebrowser && git describe --tags --always --dirty=+ || echo none)
IMPLOT_GIT_VERSION_STRING := $(shell test -d ../implot && cd ../implot && git describe --tags --always --dirty=+ || echo none)
CLEW_GIT_VERSION_STRING := $(shell test -d ../clew && cd ../clew && git describe --tags --always --dirty=+ || echo none)

# features
STDCXX ?= c++17
CL ?= -DHAVE_CL
EXR ?= $(OPENEXR_VERSION_MAJOR)
FS ?= -DHAVE_FS
DEBUG ?= -g -gdwarf -ggdb
TOML11 ?= 4
FLOAT128 ?= -DHAVE_FLOAT128 -DMPFR_WANT_FLOAT128 -lquadmath

SYSTEM ?= native-gcc
COMPILER = false
OEXT = .o
include build/$(SYSTEM).mk

STRIP ?= strip

VERSIONS += \
-DFRAKTALER_3_VERSION_STRING="\"$(VERSION)\"" \
-DIMGUI_GIT_VERSION_STRING="\"$(IMGUI_GIT_VERSION_STRING)\"" \
-DIMGUI_FILEBROWSER_GIT_VERSION_STRING="\"$(IMGUI_FILEBROWSER_GIT_VERSION_STRING)\"" \
-DIMPLOT_GIT_VERSION_STRING="\"$(IMPLOT_GIT_VERSION_STRING)\"" \
-DCLEW_GIT_VERSION_STRING="\"$(CLEW_GIT_VERSION_STRING)\"" \
-DTOML11_API="$(TOML11)"

LIBS += libpng mpfr zlib
# Removed for macOS batch build: libdeflate OpenEXR (EXR=0), libjpeg (only used by gui.cc).
LIBS_GUI += sdl2

CFLAGS_IMGUI += -Isrc -I../imgui -I../imgui/backends -I../imgui/misc/cpp -I../imgui-filebrowser -I../implot -DIMGUI_USER_CONFIG="\"f3imconfig.h\"" $(FS)
LIBS_IMGUI +=

CFLAGS += $(FLOAT128) $(DEBUG)
LDFLAGS += $(FLOAT128) $(DEBUG)

COMPILE := $(COMPILER) $(CPPFLAGS) $(CFLAGS) $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config $(PKG_CONFIG_FLAGS) --cflags $(LIBS) $(LIBS_GUI) $(LIBS_CL) | sed "$(PKG_CONFIG_SED)") $(CFLAGS_IMGUI) $(VERSIONS)
COMPILE_WEB := $(COMPILER) $(CPPFLAGS) $(CFLAGS) $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config $(PKG_CONFIG_FLAGS) --cflags $(LIBS)) $(CFLAGS_IMGUI) $(VERSIONS)

LINK := $(COMPILER) $(CFLAGS)
LINK_FLAGS := $(LDFLAGS) $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config $(PKG_CONFIG_FLAGS) --libs $(LIBS) $(LIBS_GUI) $(LIBS_CL) | sed "$(PKG_CONFIG_SED)") $(LIBS_IMGUI) $(CLFLAGS)

SOURCES_CC += \
src/batch.cc \
src/bla.cc \
src/colour.cc \
src/display_gles.cc \
src/engine.cc \
src/gles2.cc \
src/glutil.cc \
src/gui.cc \
src/histogram.cc \
src/hybrid.cc \
src/image_raw.cc \
src/image_rgb.cc \
src/jpeg.cc \
src/main.cc \
src/opencl.cc \
src/param.cc \
src/png.cc \
src/render.cc \
src/source.cc \
src/version.cc \
src/wisdom.cc \

SOURCES_IMGUI_CC += \
../imgui/imgui.cpp \
../imgui/imgui_draw.cpp \
../imgui/imgui_tables.cpp \
../imgui/imgui_widgets.cpp \
../imgui/backends/imgui_impl_sdl2.cpp \
../imgui/backends/imgui_impl_opengl3.cpp \
../imgui/misc/cpp/imgui_stdlib.cpp \
../implot/implot.cpp \
../implot/implot_items.cpp \

OBJECTS = \
$(patsubst %.cc,%$(OEXT),$(SOURCES_CC)) \
$(patsubst %.cpp,%$(OEXT),$(SOURCES_IMGUI_CC)) \

DEPENDS = \
$(patsubst %.o,%.d,$(OBJECTS)) \

default: $(TARGETS)

main: fraktaler-3-$(VERSION)$(EXEEXT)

web: live/$(VERSION)/index.html

clean:
	-rm -f src/fraktaler-3-source.7z.h
	-rm -f src/cl-pre.h src/cl-post.h
	-rm -f src/colour.vert.h src/colour.frag.h src/colour_default.frag.h
	-rm -f $(OBJECTS)
	-rm -f $(DEPENDS)

headers:
	@echo "'make headers' is no longer needed, just run 'make'"

VERSION.txt: $(shell cat INDEX.txt | grep -v VERSION.txt)
	echo "$(VERSION)" > VERSION.txt
	date --iso >> VERSION.txt

# distribution

fraktaler-3-$(VERSION)$(EXEEXT): fraktaler-3$(EXEEXT)
	cp -avf $< $@
	$(STRIP) --strip-unneeded $@

fraktaler-3-$(VERSION).7z: fraktaler-3-source.7z
	cp -avf $< $@

fraktaler-3-source.7z: $(shell cat INDEX.txt)
	-rm -rf fraktaler-3-source.7z "fraktaler-3-$(VERSION)"
	mkdir fraktaler-3-$(VERSION)
	cat INDEX.txt |	cpio -pd "fraktaler-3-$(VERSION)"
	7za a -t7z -m0=lzma -mx=9 -mfb=64 -md=32m -ms=on -bb0 -bd "fraktaler-3-source.7z" "fraktaler-3-$(VERSION)/"

fraktaler-3-$(VERSION).html: README.md fraktaler-3-$(VERSION).css
	pandoc README.md --metadata="title=fraktaler-3-$(VERSION)" --metadata="date=$(DATE)" --standalone -c "fraktaler-3-$(VERSION).css" --toc --mathjax='/mathjax/MathJax.js?config=TeX-AMS-MML_HTMLorMML' -o "fraktaler-3-$(VERSION).html"
	sed -i "s|<head>|<head profile='http://www.w3.org/2005/10/profile'>|g" "fraktaler-3-$(VERSION).html"
	sed -i "s|href=.fraktaler-3.css.|href='fraktaler-3-$(VERSION).css'|g" "fraktaler-3-$(VERSION).html"

fraktaler-3-$(VERSION).css: fraktaler-3.css
	cp -avf $< $@

fraktaler-3-$(VERSION).pdf: README.md
	pandoc README.md --metadata="title=fraktaler-3-$(VERSION)" --metadata="date=$(DATE)" -o fraktaler-3-$(VERSION).pdf

LICENSE.pdf: LICENSE.md
	pandoc LICENSE.md --toc --toc-depth=4 -V geometry="margin=1in" --metadata="author=Free Software Foundation, Inc." --metadata="title=GNU Affero General Public License" --metadata="date=19 November 2007" -o LICENSE.pdf

src/fraktaler-3-source.7z.h: fraktaler-3-source.7z
	xxd -i $< | sed "s/unsigned/const unsigned/g" > $@

fraktaler-3.png: fraktaler-3.ico
	icotool -x fraktaler-3.ico -o fraktaler-3.png

fraktaler-3.bmp: fraktaler-3.png
	convert fraktaler-3.png fraktaler-3.bmp

src/icon.h: fraktaler-3.bmp
	xxd -i $< | sed "s/unsigned/const unsigned/g" > $@

src/cl-pre.h: src/cl-pre.cl
	xxd -i $< | sed "s/unsigned/const unsigned/g" > $@

src/cl-post.h: src/cl-post.cl
	xxd -i $< | sed "s/unsigned/const unsigned/g" > $@

src/colour_default.frag.h: examples/default.glsl
	xxd -i $< | sed "s/unsigned/const/g" | sed "s/};/,0x00};/g" > $@

src/colour.frag.h: src/colour.frag.glsl
	xxd -i $< | sed "s/unsigned/const/g" | sed "s/};/,0x00};/g" > $@

src/colour.vert.h: src/colour.vert.glsl
	xxd -i $< | sed "s/unsigned/const/g" | sed "s/};/,0x00};/g" > $@

# link

fraktaler-3$(EXEEXT): $(OBJECTS) $(RESOURCES)
	$(LINK) -o $@ $(OBJECTS) $(RESOURCES) $(LINK_FLAGS)

live/$(VERSION)/index.html: $(OBJECTS) fraktaler-3-$(VERSION).7z fraktaler-3.ico
	mkdir -p live/$(VERSION)
	cp -avf src/index.html fraktaler-3-$(VERSION).7z fraktaler-3.ico live/$(VERSION)
	sed -i "s/VERSION/$(VERSION)/g" "live/$(VERSION)/index.html"
	$(LINK) -o live/$(VERSION)/fraktaler-3.html $(OBJECTS) $(LDFLAGS)
	rm live/$(VERSION)/fraktaler-3.html
	gzip -9 -k -f live/$(VERSION)/fraktaler-3.ico
	gzip -9 -k -f live/$(VERSION)/fraktaler-3.js
	gzip -9 -k -f live/$(VERSION)/fraktaler-3.wasm
	gzip -9 -k -f live/$(VERSION)/index.html

%.gz: %
	gzip -k -9 -f $<

# compile

%$(OEXT): %.cc
	$(COMPILE) -o $@ -c $< $(EXTRA_CFLAGS)

%$(OEXT): %.cpp
	$(COMPILE) -o $@ -c $< $(EXTRA_CFLAGS)

%$(OEXT): %.c
	$(COMPILE) -o $@ -c $< $(EXTRA_CFLAGS)

%.web$(OEXT): %.cc
	$(COMPILE_WEB) -o $@ -c $< $(EXTRA_CFLAGS)

%.web$(OEXT): %.cpp
	$(COMPILE_WEB) -o $@ -c $< $(EXTRA_CFLAGS)

# windows

%$(OEXT): %.res
	$(WINDRES) -J res -i $< -o $@

src/resource.res: src/resource.rc fraktaler-3.ico
	 $(GWINDRES) -i src/resource.rc -o src/resource.res

# distribution

release:
	mkdir -p fraktaler-3-$(VERSION)-windows
	cp -avft fraktaler-3-$(VERSION)-windows fraktaler-3-$(VERSION)*.exe fraktaler-3-$(VERSION).pdf LICENSE.pdf
	7za a -t7z -m0=lzma -mx=9 -mfb=64 -md=32m -ms=on fraktaler-3-$(VERSION)-windows.7z fraktaler-3-$(VERSION)-windows/*

# dependencies

.PHONY: default clean headers
.SUFFIXES:

src/colour$(OEXT): src/colour.vert.h src/colour.frag.h src/colour_default.frag.h

# macos-batch: skip icon/source/cl deps (no GUI, no OpenCL, no embedded source).
# src/gui$(OEXT): src/icon.h
# src/opencl$(OEXT): src/cl-pre.h src/cl-post.h
# src/source$(OEXT): src/fraktaler-3-source.7z.h

-include \
$(DEPENDS) \
