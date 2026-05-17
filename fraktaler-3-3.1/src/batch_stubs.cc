// Stubs for batch-only macOS build (no HAVE_GUI).
// These symbols normally live in gui.cc and colour.cc but those translation
// units are compiled to empty when HAVE_GUI is undefined.

#include <cstdio>

#include "floatexp.h"

// gui.cc real signature is 3 args (main.cc:412 calls it with 3).
extern "C++" int gui(const char *progname, const char *persistence_str, bool do_batch)
{
  (void) persistence_str;
  (void) do_batch;
  std::fprintf(stderr, "%s: error: built without GUI support\n", progname);
  return 1;
}

// Referenced from engine.cc via extern declaration. Lives in gui.cc normally.
floatexp<float, int> newton_relative_start = 1;

// Referenced from param.cc (pcolour::pcolour()). Normally points to the
// embedded examples/default.glsl source via colour.cc.
const char *colour_default_glsl = "";
