// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#include <thread>

#include <getopt.h>
#include <SDL.h>

#ifdef HAVE_CLEW
#include "clew.h"
#endif

#ifdef __EMSCRIPTEN__
#include <string.h>
#include "emscripten.h"
#include "emscripten/html5.h"
#endif

#if defined(__ANDROID__) && ! defined(__TERMUX__)
#include <filesystem>
#endif

#include "engine.h"
#include "main.h"
#include "source.h"
#include "version.h"
#include "wisdom.h"

param par;
wisdom wdom;

std::string pref_path, default_source_path, default_persistence_path, default_wisdom_path;

void initialize_paths()
{
  pref_path = "";
  if (! SDL_Init(0))
  {
    char *p = SDL_GetPrefPath("uk.co.mathr", "fraktaler-3");
    if (p)
    {
      pref_path = std::string(p);
      SDL_free(p);
    }
#if defined(__ANDROID__) && ! defined(__TERMUX__)
    const char *default_path = SDL_AndroidGetExternalStoragePath();
    if (default_path)
    {
      // can be accessed by other apps
      std::filesystem::current_path(std::string(default_path));
    }
    else
    {
      // last resort, cannot be accessed outside this app
      std::filesystem::current_path(pref_path);
    }
#endif
    SDL_Quit();
  }
  default_source_path = source_filename;
  default_persistence_path = pref_path + "persistence.f3.toml";
  default_wisdom_path = pref_path + "wisdom.toml";
}

int load_wisdom(const char *wisdom)
{
  try
  {
    bool success = false;
    wdom = wisdom_load(std::string(wisdom), success);
    return success ? 0 : 1;
  }
  catch (...)
  {
    wisdom_default(wdom);
    return 1;
  }
}

int generate_wisdom(const char *wisdom)
{
  try
  {
    wdom = wisdom_enumerate(true);
    wisdom_save(wdom, std::string(wisdom));
    return 0;
  }
  catch (...)
  {
    return 1;
  }
}

int benchmark_wisdom(const char *wisdom)
{
  try
  {
    volatile progress_t progress = 0;
    volatile bool running = true;
    wdom = wisdom_benchmark(wdom, &progress, &running);
    wisdom_save(wdom, std::string(wisdom));
    return 0;
  }
  catch (...)
  {
    return 1;
  }
}

int print_mode_error(const char *progname)
{
  std::fprintf(stderr, "%s: error: exactly one mode of operation must be specified\n", progname);
  return 1;
}

int print_help(const char *progname)
{
  std::fprintf
    ( stdout
    , "usage:\n"
      "  %s [mode] [flags ...] [inputfile [inputfile ...]]\n"
      "modes of operation:\n"
      "  -h, --help                print this message and exit\n"
      "  -V, --version             print version information and exit\n"
      "  -i, --interactive         interactive graphical user interface\n"
      "  -b, --batch               command line batch processing\n"
      "  -W, --generate-wisdom     generate initial hardware configuration\n"
      "  -B, --benchmark-wisdom    benchmark hardware for optimal efficiency\n"
      "  -S, --export-source       export this program's source code\n"
      "    default location: %s\n"
      "flags:\n"
      "  -v, --verbose             increase verbosity\n"
      "  -q, --quiet               decrease verbosity\n"
      "  -p, --persistence file    path to persist state\n"
      "    default location: %s\n"
      "  -P, --no-persistence      don't persist state\n"
      "  -w, --wisdom file         path to wisdom\n"
      "    default location: %s\n"
      "input files are merged in command line order\n"
    , progname
    , default_source_path.c_str()
    , default_persistence_path.c_str()
    , default_wisdom_path.c_str()
    );
  return 0;
}

int print_version()
{
  std::fprintf(stdout, "%s", version().c_str());
  return 0;
}

int export_source()
{
  if (write_source(source_filename))
  {
    return 0;
  }
  else
  {
    return 1;
  }
}

int print_generate_wisdom_error(const char *progname, const char *wisdom)
{
  std::fprintf(stderr, "%s: error: could not generate wisdom file '%s'\n", progname, wisdom);
  return 1;
}

int print_benchmark_wisdom_error(const char *progname, const char *wisdom)
{
  std::fprintf(stderr, "%s: error: could not benchmark wisdom file '%s'\n", progname, wisdom);
  return 1;
}

int print_load_wisdom_error(const char *progname, const char *wisdom)
{
  std::fprintf(stderr, "%s: error: could not load wisdom file '%s'\n", progname, wisdom);
  return 1;
}

int print_load_parameter_error(const char *progname, const char *parameter)
{
  std::fprintf(stderr, "%s: error: could not load parameter file '%s'\n", progname, parameter);
  return 1;
}

int main0(int argc, char **argv)
{
  // initialize environment
  initialize_paths();
  // modes of operation  
  bool
    do_help = false,
    do_version = false,
    do_interactive = false,
    do_batch = false,
    do_generate_wisdom = false,
    do_benchmark_wisdom = false,
    do_source = false;
  int verbosity = 1;
  const char *wisdom = ""; // empty string -> use default app path
  const char *persistence = ""; // empty string -> use default app path
  // parse arguments
  while (1)
  {
    static struct option long_options[] =
      { { "version", no_argument, 0, 'V' }
      , { "help", no_argument, 0, 'h' }
      , { "verbose", no_argument, 0, 'v' }
      , { "quiet", no_argument, 0, 'q' }
      , { "interactive", no_argument, 0, 'i' }
      , { "batch", no_argument, 0, 'b' }
      , { "persistence", required_argument, 0, 'p' }
      , { "no-persistence", no_argument, 0, 'P' }
      , { "wisdom", required_argument, 0, 'w' }
      , { "generate-wisdom", no_argument, 0, 'W' }
      , { "benchmark-wisdom", no_argument, 0, 'B' }
      , { "export-source", no_argument, 0, 'S' }
      , { 0, 0, 0, 0 }
      };
    int option_index = 0;
    int c = getopt_long(argc, argv, "Vhvqibp:Pw:WBS", long_options, &option_index);
    if (c == -1)
    {
      break;
    }
    switch (c)
    {
      case 'V':
        do_version = true;
        break;
      case 'h':
        do_help = true;
        break;
      case 'v':
        verbosity++;
        break;
      case 'q':
        verbosity--;
        break;
      case 'i':
        do_interactive = true;
        break;
      case 'b':
        do_batch = true;
        break;
      case 'p':
        persistence = optarg;
        break;
      case 'P':
        persistence = nullptr;
        break;
      case 'w':
        wisdom = optarg;
        break;
      case 'W':
        do_generate_wisdom = true;
        break;
      case 'B':
        do_benchmark_wisdom = true;
        break;
      case 'S':
        do_source = true;
        break;
    }
  }
  int count = int(do_help) + int(do_version) + int(do_source) + int(do_generate_wisdom) + int(do_benchmark_wisdom) + int(do_interactive) + int(do_batch);
  if (count == 0)
  {
    do_interactive = true;
    count++;
  }
  if (count != 1)
  {
    return print_mode_error(argv[0]);
  }
  if (std::string(wisdom) == std::string(""))
  {
    wisdom = default_wisdom_path.c_str();
  }
  if (persistence && std::string(persistence) == std::string(""))
  {
    persistence = default_persistence_path.c_str();
  }
  if (do_help)
  {
    return print_help(argv[0]);
  }
  if (do_version)
  {
    return print_version();
  }
  if (do_source)
  {
    return export_source();
  }
  // initialize rendering engine
#ifdef HAVE_CLEW
  if (clewInit())
  {
    if (verbosity > 0)
    {
      std::fprintf(stderr, "%s: warning: clewInit() failed\n", argv[0]);
    }
  }
#endif
  bool loaded_wisdom = false;
  if (0 == load_wisdom(wisdom))
  {
    loaded_wisdom = true;
    if (verbosity > 0)
    {
      std::fprintf(stdout, "loaded wisdom %s\n", wisdom);
    }
  }
  else
  {
    if (verbosity > 0)
    {
      std::fprintf(stderr, "%s: warning: could not load wisdom %s\n", argv[0], wisdom);
    }
  }
  if (do_generate_wisdom)
  {
    if (generate_wisdom(wisdom))
    {
      return print_generate_wisdom_error(argv[0], wisdom);
    }
    else
    {
      return 0;
    }
  }
  if (do_benchmark_wisdom)
  {
    if (benchmark_wisdom(wisdom))
    {
      return print_benchmark_wisdom_error(argv[0], wisdom);
    }
    else
    {
      return 0;
    }
  }
  if (do_batch || do_interactive)
  {
    // load wisdom
    if (! loaded_wisdom)
    {
      std::fprintf(stderr, "%s: warning: using default wisdom\n", argv[0]);
      wisdom_default(wdom);
    }
    // load persistence
    if (persistence)
    {
      try
      {
        par.load_toml(persistence);
        if (verbosity > 0)
        {
          std::fprintf(stdout, "loaded persistence %s\n", persistence);
        }
      }
      catch (...)
      {
        if (verbosity > 0)
        {
          std::fprintf(stderr, "%s: warning: failed to load persistnce %s\n", argv[0], persistence);
        }
      }
    }
    // load parameters
    for (int arg = optind; arg < argc; ++arg)
    {
      try
      {
        par.load_any(argv[arg]);
      }
      catch (...)
      {
        return print_load_parameter_error(argv[0], argv[arg]);
      }
    }
    // main program is graphical
    if (do_batch && ! (par.p.render.save_toml || par.p.render.save_exr || par.p.render.save_png || par.p.render.save_jpg))
    {
      std::fprintf(stderr, "%s: batch mode: no save formats enabled\n", argv[0]);
      return 1;
    }
    if (do_batch && par.p.render.save_exr && ((par.exr_channels & ~Channels_RGB) != 0) && par.p.image.subframes != 1)
    {
      std::fprintf(stderr, "%s: batch mode: can only save raw EXR channels with 1 subframe\n", argv[0]);
      return 1;
    }
    if (do_batch && ! (par.p.image.subframes > 0))
    {
      std::fprintf(stderr, "%s: batch mode: continuous render mode requested\n", argv[0]);
      return 1;
    }
    if (do_batch && ! par.p.render.save_png && ! par.p.render.save_jpg && (par.p.render.save_exr ? (par.exr_channels & Channels_RGB) == 0 : true))
    {
      // non-RGB batch mode works without GUI
      return batch(verbosity, par);
    }
    else
    {
      // interactive and RGB batch requires GUI
      return gui(argv[0], persistence, do_batch);
    }
  }
  // unreachable
  return 1;
}

#ifdef __EMSCRIPTEN__

int web_argc = 0;
char **web_argv = nullptr;

extern "C" int EMSCRIPTEN_KEEPALIVE web(void)
{
  return main0(web_argc, web_argv);
}

int main(int argc, char **argv)
{
  // copy arguments
  web_argc = argc;
  web_argv = new char *[argc + 1];
  web_argv[0] = strdup("fraktaler-3");
  for (int arg = 1; arg < argc; ++arg)
  {
    web_argv[arg] = strdup(argv[arg]);
  }
  web_argv[argc] = nullptr;
  // initialize file systen
  pref_path = "/libsdl/uk.co.mathr/fraktaler-3/";
  EM_ASM(
    FS.mkdir('/libsdl');
    FS.mkdir('/libsdl/uk.co.mathr');
    FS.mkdir('/libsdl/uk.co.mathr/fraktaler-3');
    FS.mount(IDBFS, {}, '/libsdl/uk.co.mathr/fraktaler-3');
    FS.syncfs(true, function (err) {
      assert(! err);
      ccall('web', 'number');
    });
  );
  return 0;
}

#else

int main(int argc, char **argv)
{
  return main0(argc, argv);
}

#endif
