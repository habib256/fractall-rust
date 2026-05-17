// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#ifndef HAVE_GUI

#include <cstdio>

int gui(const char *progname, const char *persistence_str)
{
  (void) persistence_str;
  std::fprintf(stderr, "%s: error: built without GUI support\n", progname);
  return 1;
}

#else

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <map>
#include <thread>
#include <vector>

#include "types.h"

#include <SDL.h>
#include "gles2.h"

#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_opengl3.h>
#include <imgui_stdlib.h>
#include <implot.h>
#if HAVE_FS
#include <imfilebrowser.h>
#endif
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_transform_2d.hpp>
#include <mpreal.h>
#include <toml.hpp>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten_browser_clipboard.h>
#include <emscripten_browser_file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#include <SDL_syswm.h>
#else
#ifdef HAVE_ICON
#include "icon.h"
#endif
#endif

#include "colour.h"
#include "display_gles.h"
#include "engine.h"
#include "floatexp.h"
#include "histogram.h"
#include "image_raw.h"
#include "image_rgb.h"
#include "main.h"
#include "param.h"
#include "render.h"
#include "version.h"
#include "wisdom.h"

#if defined(__EMSCRIPTEN__) || (defined(__ANDROID__) && ! defined(__TERMUX__))
#define FULLSCREEN_OPTION 0
#else
#define FULLSCREEN_OPTION 1
#endif

#if defined(__EMSCRIPTEN__) || (defined(__ANDROID__) && ! defined(__TERMUX__))
#define WISDOM_DIALOGS 0
#else
#define WISDOM_DIALOGS HAVE_FS
#endif

#if defined(__EMSCRIPTEN__)
#define LOADSAVE_DIALOGS 0
#else
#define LOADSAVE_DIALOGS HAVE_FS
#endif

// rendering state machine
std::vector<progress_t> progress;
progress_t newton_progress[4];
bool quit = false;
bool running = false;
bool ended = true;
bool restart = false;
time_t timestamp = 0;
bool recolour = false;
bool continue_subframe_rendering = false;
int subframes_rendered = 0;
std::chrono::duration<double> duration = std::chrono::duration<double>::zero();

floatexp<float, int> batch_zoom;
count_t batch_frame;
count_t batch_frame_start;
count_t batch_frame_end;
bool batch_running = false;
bool batch_ended = false;
bool batch_really_stop = false;
bool batch_quit_when_done = false;

#ifdef __EMSCRIPTEN__
#define STOP \
  running = false; \
  while (! ended) \
    emscripten_sleep(1);
#else
#define STOP \
  running = false; \
  while (! ended) \
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif

extern param par;

void clipboard_copy();
void clipboard_paste();

#ifdef __EMSCRIPTEN__

// <https://github.com/Armchair-Software/emscripten-browser-clipboard?tab=readme-ov-file#use-with-imgui>

std::string clipboard_content;

const char *get_content_for_imgui(void *user_data)
{
  (void) user_data;
  return clipboard_content.c_str();
}

void set_content_from_imgui(void *user_data, const char *text)
{
  (void) user_data;
  clipboard_content = text;
  emscripten_browser_clipboard::copy(clipboard_content);
}

void initialize_clipboard(void)
{
  emscripten_browser_clipboard::paste([](std::string &&paste_data, void *user_data) {
    (void) user_data;
    clipboard_content = paste_data;
    clipboard_paste();
  });
  ImGuiIO &imgui_io = ImGui::GetIO();
  imgui_io.GetClipboardTextFn = get_content_for_imgui;
  imgui_io.SetClipboardTextFn = set_content_from_imgui;
}

void gui_post_load(param &par);

void upload_handler(const std::string &filename, const std::string &mimetype, std::string_view buffer, void *user_data)
{
  (void) user_data;
  (void) mimetype;
  try
  {
    std::string tmpname = "UPLOAD" + std::filesystem::path(filename).extension().string(); // FIXME
    FILE *file = std::fopen(tmpname.c_str(), "wb");
    if (file)
    {
      bool ok = 1 == std::fwrite(buffer.data(), buffer.size(), 1, file);
      std::fclose(file);
      if (ok)
      {
        STOP
        par.load_any(tmpname);
        gui_post_load(par);
        restart = true;
      }
      unlink(tmpname.c_str());
      syncfs();
    }
  }
  catch (...)
  {
    // FIXME ignored
  }
}

void download_handler(const std::filesystem::path &path)
{
  std::string mimetype = "application/octet-stream";
  if (ends_with(path, ".exr"))
  {
    mimetype = "image/x-exr";
  }
  else if (ends_with(path, ".png"))
  {
    mimetype = "image/png";
  }
  else if (ends_with(path, ".jpg") || ends_with(path, ".jpeg"))
  {
    mimetype = "image/jpeg";
  }
  else if (ends_with(path, ".toml"))
  {
    mimetype = "application/toml";
  }
  else if (ends_with(path, ".glsl"))
  {
    mimetype = "text/plain"; // FIXME check if there's anything better
  }
  struct stat sb;
  if (stat(path.string().c_str(), &sb) != -1)
  {
    FILE *file = std::fopen(path.string().c_str(), "rb");
    if (file)
    {
      void *data = std::malloc(sb.st_size);
      if (data)
      {
        bool ok = 1 == std::fread(data, sb.st_size, 1, file);
        std::fclose(file);
        if (ok)
        {
          emscripten_browser_file::download(path.filename().string().c_str(), mimetype.c_str(), data, sb.st_size);
        }
        std::free(data);
      }
      else
      {
        std::fclose(file);
      }
    }
  }
}

#else

void initialize_clipboard(void)
{
  // nop
}

#endif

#ifdef HAVE_GLDEBUG
#ifdef _WIN32
__attribute__((stdcall))
#endif
static void opengl_debug_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, const void *userParam)
{
#ifdef _WIN32
  // don't do anything in Windows
  // in Wine printing happens in multiple threads at once (?) and crashes
  return;
#endif
  (void) userParam;
  (void) length;
  if (source == 33350 && type == 33361 && id == 131185 && severity == 33387)
  {
    // silence extremely verbose message from NVIDIA driver about buffer memory
    return;
  }
  if (source == GL_DEBUG_SOURCE_SHADER_COMPILER && type == GL_DEBUG_TYPE_OTHER && severity == GL_DEBUG_SEVERITY_NOTIFICATION)
  {
    // silence verbose messages from AMDGPU driver about shader compilation
    return;
  }
  const char *source_name = "unknown";
  const char *type_name = "unknown";
  const char *severity_name = "unknown";
  switch (source)
  {
    case GL_DEBUG_SOURCE_API: source_name = "API"; break;
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM: source_name = "Window System"; break;
    case GL_DEBUG_SOURCE_SHADER_COMPILER: source_name = "Shader Compiler"; break;
    case GL_DEBUG_SOURCE_THIRD_PARTY: source_name = "Third Party"; break;
    case GL_DEBUG_SOURCE_APPLICATION: source_name = "Application"; break;
    case GL_DEBUG_SOURCE_OTHER: source_name = "Other"; break;
  }
  switch (type)
  {
    case GL_DEBUG_TYPE_ERROR: type_name = "Error"; break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: type_name = "Deprecated Behaviour"; break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR: type_name = "Undefined Behaviour"; break;
    case GL_DEBUG_TYPE_PORTABILITY: type_name = "Portability"; break;
    case GL_DEBUG_TYPE_PERFORMANCE: type_name = "Performance"; break;
    case GL_DEBUG_TYPE_MARKER: type_name = "Marker"; break;
    case GL_DEBUG_TYPE_PUSH_GROUP: type_name = "Push Group"; break;
    case GL_DEBUG_TYPE_POP_GROUP: type_name = "Pop Group"; break;
    case GL_DEBUG_TYPE_OTHER: type_name = "Other"; break;
  }
  switch (severity)
  {
    case GL_DEBUG_SEVERITY_HIGH: severity_name = "High"; break;
    case GL_DEBUG_SEVERITY_MEDIUM: severity_name = "Medium"; break;
    case GL_DEBUG_SEVERITY_LOW: severity_name = "Low"; break;
    case GL_DEBUG_SEVERITY_NOTIFICATION: severity_name = "Notification"; break;
  }
  std::fprintf(stderr, "OpenGL : %s  %s  %s : %d : %s\n", severity_name, type_name, source_name, id, message);
}
#endif

bool persistence = true;
std::string persistence_path = "persistence.f3.toml";

const char *gl_version = "unknown";
GLint maximum_texture_size = 1;
int srgb_conversion = 0;

// for quality window
int image_width = 0;
int image_height = 0;
int image_dpi = 0;
bool image_lock_dpi = false;
int image_width_aspect_locked = 1;
int image_height_aspect_locked = 1;
bool image_lock_aspect = false;
// for algorithm window
int tile_width = 0;
int tile_height = 0;

// global state
SDL_Window* window = nullptr;
bool fullscreen = false;
display_gles *dsp = nullptr;
image_raw *raw = nullptr;
image_rgb *rgb = nullptr;
colour *clr = nullptr;
std::thread *bg = nullptr;
std::chrono::time_point<std::chrono::steady_clock> program_start_time, start_time;
std::atomic<int> needs_redraw {0};
std::atomic<int> needs_dopost {0};
std::atomic<int> started {0};

std::string display_error_string;

// touch events
SDL_TouchID finger_device;
std::map<SDL_FingerID, std::pair<vec3, vec3>> fingers;
mat3 finger_transform(1.0f);
mat3 finger_transform_started(1.0f);

std::mutex tile_queue_mutex;
struct tile_spec
{
  int platform, device, x, y, subframe;
  tile *data;
};
std::vector<tile_spec> tile_queue;
std::vector<tile_spec> tile_cache;
int tile_cache_subframes;

struct gui_hooks : public hooks
{
  gui_hooks()
  {
  }
  virtual ~gui_hooks()
  {
  }
  virtual void tile(int platform, int device, int x, int y, int subframe, const struct tile *data)
  {
    if (raw && subframe == 0)
    {
      raw->blit(x, y, data);
    }
    tile_spec spec = { platform, device, x, y, subframe, tile_copy(data) };
    std::lock_guard<std::mutex> lock(tile_queue_mutex);
    tile_queue.push_back(spec);
  }
};

void reset(param &par)
{
  home(par);
  par.p.formula = phybrid();
  post_edit_formula(par);
}

wlookup lookup;

void render_thread(progress_t *progress, volatile bool *running, volatile bool *ended)
{
  gui_hooks h;
  count_t pixel_spacing_exp, pixel_precision_exp;
  get_required_precision(par, pixel_spacing_exp, pixel_precision_exp);
  std::set<number_type> available = { nt_float, nt_double, nt_longdouble, nt_floatexp, nt_doubleexp, nt_softfloat // FIXME
#ifdef HAVE_FLOAT128
  , nt_float128
#endif
  };
  lookup = wisdom_lookup(wdom, available, pixel_spacing_exp, pixel_precision_exp);
  if (! reference_can_be_reused(lookup, par))
  {
    set_reference_to_image_center(par);
    get_required_precision(par, pixel_spacing_exp, pixel_precision_exp);
    lookup = wisdom_lookup(wdom, available, pixel_spacing_exp, pixel_precision_exp);
  }
  param par2 = par;
  par2.p.image.subframes = 1;
  par2.p.render.prng_seed = par.p.render.prng_seed + subframes_rendered;
  render(lookup, par2, &h, true, progress, running);
  *ended = true;
}

void render_subframe_thread(progress_t *progress, volatile bool *running, volatile bool *ended)
{
  gui_hooks h;
  param par2 = par;
  par2.p.image.subframes = 1;
  par2.p.render.prng_seed = par.p.render.prng_seed + subframes_rendered;
  render(lookup, par2, &h, false, progress, running);
  *ended = true;
}

// zoom by mouse drag
bool drag = false;
double drag_start_x = 0;
double drag_start_y = 0;

// last mouse coordinates
int mouse_x = 0;
int mouse_y = 0;

void finger_to_image(coord_t win_width, coord_t win_height, vec2 &p);

bool matrix_ok(const mat3 &m)
{
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j)
  {
    if (std::isnan(m[i][j]))
    {
      return false;
    }
    if (std::isinf(m[i][j]))
    {
      return false;
    }
  }
  if (std::abs(glm::determinant(m)) < 1.0e-6f)
  {
    return false;
  }
  return true;
}

void update_finger_transform(coord_t win_width, coord_t win_height)
{
  switch (fingers.size())
  {
    case 0: // identity
      {
        mat3 T = mat3(1.0f);
        if (matrix_ok(T))
        {
          finger_transform = T * finger_transform;
        }
      }
      break;
    case 1: // translate
      {
        const std::pair<vec3, vec3> &finger = (*fingers.begin()).second;
        const vec3 start = finger.first;
        const vec3 end = finger.second;
        mat3 T = mat3(1.0f);
        vec2 p1 = vec2(start[0], start[1]) / start.z;
        vec2 q1 = vec2(end[0], end[1]) / end.z;
        finger_to_image(win_width, win_height, p1);
        finger_to_image(win_width, win_height, q1);
        T = glm::translate(T, -p1);
        T = glm::translate(T, q1);
        if (matrix_ok(T))
        {
          finger_transform = T * finger_transform;
        }
      }
      break;
    case 2: // translate, rotate, scale
      {
        const std::pair<vec3, vec3> &finger1 = (*fingers.begin()).second;
        const std::pair<vec3, vec3> &finger2 = (*++fingers.begin()).second;
        const vec3 start1 = finger1.first;
        const vec3 end1 = finger1.second;
        const vec3 start2 = finger2.first;
        const vec3 end2 = finger2.second;
        vec2 p1 = vec2(start1[0], start1[1]) / start1.z;
        vec2 q1 = vec2(end1[0], end1[1]) / end1.z;
        vec2 p2 = vec2(start2[0], start2[1]) / start2.z;
        vec2 q2 = vec2(end2[0], end2[1]) / end2.z;
        finger_to_image(win_width, win_height, p1);
        finger_to_image(win_width, win_height, q1);
        finger_to_image(win_width, win_height, p2);
        finger_to_image(win_width, win_height, q2);
        const vec2 dp = p1 - p2;
        const vec2 dq = q1 - q2;
        const vec2 mp = (p1 + p2) * 0.5f;
        const vec2 mq = (q1 + q2) * 0.5f;
        const float ap = std::atan2(dp.y, dp.x);
        const float aq = std::atan2(dq.y, dq.x);
        const float sp = std::hypot(dp.y, dp.x);
        const float sq = std::hypot(dq.y, dq.x);
        const float t = aq - ap;
        const float s = sq / sp;
        const float co = s * std::cos(t);
        const float si = s * std::sin(t);
        const float px = mp[0];
        const float py = mp[1];
        const float qx = mq[0];
        const float qy = mq[1];
        mat3 N(1.0f, 0.0f, -px,   0.0f, 1.0f, -py,   0.0f, 0.0f, 1.0f);
        mat3 M(co,   -si,  0.0f,  si,   co,   0.0f,  0.0f, 0.0f, 1.0f);
        mat3 L(1.0f, 0.0f, qx,    0.0f, 1.0f, qy,    0.0f, 0.0f, 1.0f);
        mat3 T = transpose(L) * transpose(M) * transpose(N);
        if (matrix_ok(T))
        {
          finger_transform = T * finger_transform;
        }
      }
      break;
    default: // overdetermined system, just use first 3 fingers....
    case 3: // translate, rotate, scale, skew
      {
        const std::pair<vec3, vec3> &finger1 = (*fingers.begin()).second;
        const std::pair<vec3, vec3> &finger2 = (*++fingers.begin()).second;
        const std::pair<vec3, vec3> &finger3 = (*++++fingers.begin()).second;
        vec2 p1 = vec2(finger1.first) / finger1.first.z;
        vec2 q1 = vec2(finger2.first) / finger2.first.z;
        vec2 r1 = vec2(finger3.first) / finger3.first.z;
        vec2 p2 = vec2(finger1.second) / finger1.second.z;
        vec2 q2 = vec2(finger2.second) / finger2.second.z;
        vec2 r2 = vec2(finger3.second) / finger3.second.z;
        finger_to_image(win_width, win_height, p1);
        finger_to_image(win_width, win_height, q1);
        finger_to_image(win_width, win_height, r1);
        finger_to_image(win_width, win_height, p2);
        finger_to_image(win_width, win_height, q2);
        finger_to_image(win_width, win_height, r2);
        const mat3 start(vec3(p1, 1.0f), vec3(q1, 1.0f), vec3(r1, 1.0f));
        const mat3 end(vec3(p2, 1.0f), vec3(q2, 1.0f), vec3(r2, 1.0f));
        mat3 T = end * inverse(start);
        if (matrix_ok(T))
        {
          finger_transform = T * finger_transform;
        }
      }
      break;
  }
  for (auto & kfinger : fingers)
  {
    kfinger.second.first = kfinger.second.second;
  }
}

// imgui state

bool show_windows = true;

struct window
{
  bool show = false;
  int x = -1, y = -1;
  double w = 16, h = 6;
};

struct windows
{
  struct window
    io = { false, -1, -1, 32, 5 },
    formula = { false, -1, -1, 32, 10 },
    status = { false, -1, -1, 12, 13 },
    location = { false, -1, -1, -1, 8 },
    reference = { false, -1, -1, -1, 8 },
    algorithm = { false, -1, -1, 24, 14 },
    bailout = { false, -1, -1, 24, 13 },
    transform = { false, -1, -1, 24, 14 },
    information = { false, -1, -1, 24, 20 },
    quality = { false, -1, -1, 24, 25 },
    render = { false, -1, -1, 24, 25 },
    colours = { false, -1, -1, 32, 32 },
    postprocessing = { false, -1, -1, 24, 25 },
    newton = { false, -1, -1, 24, 16 },
    wisdom = { false, -1, -1, 32, 19 },
    preferences = { false, -1, -1, 32, 13 },
#ifdef HAVE_IMGUI_DEMO
    demo = { false, -1, -1, -1, -1 },
#endif
    about = { false, -1, -1, -1, -1 };
};

struct windowss
{
  struct windows windowed, fullscreen;
  double cache_size = 1.0;
  double ui_scale = 100.0;
  bool iterations_limited = true;
  int64_t iterations_limit = count_t(1) << 27;
  int64_t perturb_iterations_limit = count_t(1) << 12;
  int64_t bla_steps_limit = count_t(1) << 12;
  double megapixels_limit = 16.0;
};

std::istream &operator>>(std::istream &ifs, windowss &w)
{
  auto t = toml::parse(ifs);
#define LOAD1(m,a) \
  w.m.a.show = toml::find_or(t, #m, #a, "show", w.m.a.show); \
  w.m.a.x = toml::find_or(t, #m, #a, "x", w.m.a.x); \
  w.m.a.y = toml::find_or(t, #m, #a, "y", w.m.a.y); \
  w.m.a.w = toml::find_or(t, #m, #a, "w", w.m.a.w); \
  w.m.a.h = toml::find_or(t, #m, #a, "h", w.m.a.h);
#define LOAD(a) \
  LOAD1(windowed,a) \
  LOAD1(fullscreen,a)
  LOAD(io)
  LOAD(formula)
  LOAD(status)
  LOAD(location)
  LOAD(reference)
  LOAD(algorithm)
  LOAD(bailout)
  LOAD(transform)
  LOAD(information)
  LOAD(quality)
  LOAD(render)
  LOAD(colours)
  LOAD(postprocessing)
  LOAD(newton)
  LOAD(wisdom)
  LOAD(preferences)
#ifdef HAVE_IMGUI_DEMO
  LOAD(demo)
#endif
  LOAD(about)
#undef LOAD
#undef LOAD1
#define LOAD(a) w.a = toml::find_or(t, #a, w.a);
  LOAD(cache_size)
  LOAD(ui_scale)
  LOAD(iterations_limited)
  LOAD(iterations_limit)
  LOAD(perturb_iterations_limit)
  LOAD(bla_steps_limit)
#undef LOAD
  return ifs;
}

std::ostream &operator<<(std::ostream &ofs, const windowss &p)
{
  const windowss q;
#define SAVE3(m,a,b) \
  if (p.m.a.b != q.m.a.b) \
  { \
    ofs << #m << "." << #a << "." << #b << " = " << std::setw(70) << toml::value(p.m.a.b) << "\n"; \
  }
#define SAVE2(a,b) SAVE3(windowed,a,b) SAVE3(fullscreen,a,b)
#define SAVE(a) SAVE2(a, show) SAVE2(a, x) SAVE2(a, y) SAVE2(a, w) SAVE2(a, h)
  SAVE(io)
  SAVE(formula)
  SAVE(status)
  SAVE(location)
  SAVE(reference)
  SAVE(algorithm)
  SAVE(bailout)
  SAVE(transform)
  SAVE(information)
  SAVE(quality)
  SAVE(render)
  SAVE(colours)
  SAVE(postprocessing)
  SAVE(newton)
  SAVE(wisdom)
  SAVE(preferences)
#ifdef HAVE_IMGUI_DEMO
  SAVE(demo)
#endif
  SAVE(about)
#undef SAVE
#undef SAVE2
#undef SAVE3
#define SAVE(a) \
  if (p.a != q.a) \
  { \
    ofs << #a << " = " << std::setw(70) << toml::value(p.a) << "\n"; \
  }
  SAVE(cache_size)
  SAVE(ui_scale)
  SAVE(iterations_limited)
  SAVE(iterations_limit)
  SAVE(perturb_iterations_limit)
  SAVE(bla_steps_limit)
#undef SAVE
  return ofs;
}

windowss window_state;

enum user_event_codes
{
  code_persist = 1
};

Uint32 one_minute = 60 * 1000;
Uint32 persistence_timer_callback(Uint32 interval, void *p)
{
  (void) p;
  SDL_Event event;
  SDL_UserEvent userevent;
  userevent.type = SDL_USEREVENT;
  userevent.code = code_persist;
  userevent.data1 = 0;
  userevent.data2 = 0;
  event.type = SDL_USEREVENT;
  event.user = userevent;
  SDL_PushEvent(&event);
  return interval;
}

int mouse_action = 0;

const SDL_TouchID multitouch_device = 147;
SDL_FingerID multitouch_finger = 0;
std::map<SDL_FingerID, std::pair<coord_t, coord_t>> multitouch_fingers;

void multitouch_move_finger(SDL_FingerID finger, coord_t x, coord_t y)
{
  multitouch_fingers[finger] = std::pair<coord_t, coord_t>(x, y);
}

SDL_FingerID multitouch_add_finger(coord_t x, coord_t y)
{
  float md2 = 1.0f/0.0f;
  SDL_FingerID finger = 0;
  for (const auto & idp : multitouch_fingers)
  {
    float dx = x - idp.second.first;
    float dy = y - idp.second.second;
    float d2 = dx * dx + dy * dy;
    if (d2 < md2)
    {
      md2 = d2;
      finger = idp.first;
    }
  }
  if (md2 > 16 * 16) // FIXME hardcoded sensitivity
  {
    // none nearby, add one
    finger = 1;
    for (const auto & idv : multitouch_fingers)
    {
      if (idv.first == finger)
      {
        finger++;
      }
      else
      {
        break;
      }
    }
    multitouch_fingers[finger] = std::pair<coord_t, coord_t>(x, y);
  }
  else
  {
  }
  return finger;
}

SDL_FingerID multitouch_remove_finger(coord_t &x, coord_t &y)
{
  float md2 = 1.0f/0.0f;
  coord_t mx = x;
  coord_t my = y;
  SDL_FingerID finger = 0;
  for (const auto & idp : multitouch_fingers)
  {
    float dx = x - idp.second.first;
    float dy = y - idp.second.second;
    float d2 = dx * dx + dy * dy;
    if (d2 < md2)
    {
      md2 = d2;
      finger = idp.first;
      mx = idp.second.first;
      my = idp.second.second;
    }
  }
  if (finger)
  {
    x = mx;
    y = my;
    multitouch_fingers.erase(finger);
  }
  return finger;
}

void resize(coord_t super, coord_t sub)
{
  auto width = (par.p.image.width * super + sub - 1) / sub;
  auto height = (par.p.image.height * super + sub - 1) / sub;
  delete rgb;
  rgb = new image_rgb(width, height);
  delete raw;
  raw = new image_raw(width, height,
    (1 << Channel_DEX) |
    (1 << Channel_DEY) |
    (1 << Channel_N0)  |
    (1 << Channel_N1)  |
    (1 << Channel_NF)  |
    (1 << Channel_T)   |
    (1 << Channel_BLA) |
    (1 << Channel_PTB) |
    0);
  dsp->resize(width, height, sub);
  colour_set_image_size(clr, width, height);
  const int bytes_per_pixel = 11 * 4; // FIXME
  double bytes = window_state.cache_size * 1024 * 1024 * 1024;
  tile_cache_subframes = bytes / (bytes_per_pixel * width * height);
}

void window_to_image1(coord_t win_width, coord_t win_height, double x, double y, double *cx, double *cy)
{
  dsp->image_coord(win_width, win_height, &x, &y);
  *cx = (x - dsp->width / 2.0) / (dsp->width / 2.0);
  *cy = (y - dsp->height / 2.0) / (dsp->height / 2.0);
}

void window_to_image(coord_t win_width, coord_t win_height, double x, double y, double *cx, double *cy)
{
  dsp->image_coord(win_width, win_height, &x, &y);
  *cx = x;
  *cy = y;
}

void finger_to_image(coord_t win_width, coord_t win_height, vec2 &p)
{
  double cx = p.x;
  double cy = p.y;
  window_to_image(win_width, win_height, cx, cy, &cx, &cy);
  p.x = float(cx);
  p.y = float(cy);
}

void gui_pre_save(param &par)
{
  par.p.colour.shader = colour_get_shader(clr);
  par.p.colour.uniforms = colour_get_uniforms(clr);
}

void limit_image_size(param &par)
{
  par.p.image.width = std::max(par.p.image.width, 1);
  par.p.image.height = std::max(par.p.image.height, 1);
  par.p.image.supersampling = std::min(std::max(par.p.image.supersampling, 1), 32); // FIXME
  par.p.image.subsampling = std::min(std::max(par.p.image.subsampling, 1), 32); // FIXME
  double megapixels = maximum_texture_size * maximum_texture_size / (1024.0 * 1024.0);
  window_state.megapixels_limit = std::min(window_state.megapixels_limit, megapixels);
  window_state.megapixels_limit = std::max(window_state.megapixels_limit, 0.1); // prevent infinite loop
  do
  {
    megapixels =
      par.p.image.width  * par.p.image.supersampling / (double) par.p.image.subsampling *
      par.p.image.height * par.p.image.supersampling / (double) par.p.image.subsampling /
      (1024.0 * 1024.0);
    if (megapixels > window_state.megapixels_limit)
    {
      if (par.p.image.supersampling > 1)
      {
        par.p.image.supersampling -= 1;
      }
      else if (par.p.image.subsampling < 32)
      {
        par.p.image.subsampling += 1;
      }
      else
      {
        par.p.image.width >>= 1;
        par.p.image.height >>= 1;
        par.p.image.width = std::max(1, par.p.image.width);
        par.p.image.height = std::max(1, par.p.image.height);
      }
    }
  } while (megapixels > window_state.megapixels_limit);
  if (par.p.image.width > maximum_texture_size)
  {
    par.p.image.height *= maximum_texture_size / (double) par.p.image.width;
    par.p.image.height = std::max(1, par.p.image.height);
    par.p.image.width = maximum_texture_size;
  }
  else if (par.p.image.height > maximum_texture_size)
  {
    par.p.image.width *= maximum_texture_size / (double) par.p.image.height;
    par.p.image.width = std::max(1, par.p.image.width);
    par.p.image.height = maximum_texture_size;
  }
}

void gui_post_load(param &par)
{
  if (window_state.iterations_limited)
  {
    par.p.bailout.iterations = std::min(par.p.bailout.iterations, window_state.iterations_limit);
    par.p.bailout.maximum_reference_iterations = std::min(par.p.bailout.maximum_reference_iterations, window_state.iterations_limit);
    par.p.bailout.maximum_bla_steps = std::min(par.p.bailout.maximum_bla_steps, window_state.bla_steps_limit);
    par.p.bailout.maximum_perturb_iterations = std::min(par.p.bailout.maximum_perturb_iterations, window_state.perturb_iterations_limit);
  }
  limit_image_size(par);
  resize(par.p.image.supersampling, par.p.image.subsampling);
  // upload colour
  colour_set_shader(clr, par.p.colour.shader);
  colour_set_uniforms(clr, par.p.colour.uniforms);
  colour_stale_cdf(clr);
  colour_upload(clr);
  // for quality window
  image_width = par.p.image.width;
  image_height = par.p.image.height;
  image_dpi = par.p.image.dpi;
  // for algorithm window
  tile_width = par.p.opencl.tile_width;
  tile_height = par.p.opencl.tile_height;
  // for bailout window (etc)
  restring_vals(par);
}

void persist_state()
{
  if (! persistence)
  {
    return;
  }
  try
  {
    gui_pre_save(par);
    par.save_toml(persistence_path);
    syncfs();
  }
  catch (const std::exception &e)
  {
    SDL_LogWarn(SDL_LOG_CATEGORY_APPLICATION, "saving persistence %s", e.what());
  }
  try
  {
    std::ofstream ofs;
    ofs.exceptions(std::ofstream::badbit | std::ofstream::failbit);
    ofs.open(pref_path + "gui-settings.toml", std::ios_base::binary);
    ofs << window_state;
  }
  catch (const std::exception &e)
  {
    SDL_LogWarn(SDL_LOG_CATEGORY_APPLICATION, "saving GUI settings %s", e.what());
  }
}

complex<floatexp<float, int>> newton_c(0);
floatexp<float, int> newton_r = 1;
bool start_newton = false;
bool newton_running = false;
bool newton_ended = false;
param newton_par;
bool newton_ok = false;

void handle_event(SDL_Window *window, SDL_Event &e, param &par)
{
  int win_width = 0;
  int win_height = 0;
  SDL_GetWindowSize(window, &win_width, &win_height);
  SDL_Keymod mods = SDL_GetModState();
  bool shift = mods & KMOD_SHIFT;
  bool ctrl = mods & KMOD_CTRL;
  bool multitouch_emulation = ctrl && shift;
  switch (e.type)
  {
    case SDL_QUIT:
      if (batch_running)
      {
        // FIXME ignored
      }
      else
      {
        STOP
        quit = true;
        persist_state();
      }
      break;

    case SDL_MOUSEMOTION:
      if (e.motion.which != SDL_TOUCH_MOUSEID)
      {
        if (multitouch_emulation)
        {
          if (e.motion.state & SDL_BUTTON_LMASK)
          {
            multitouch_move_finger(multitouch_finger, e.motion.x, e.motion.y);
            SDL_Event t;
            t.type = SDL_FINGERMOTION;
            t.tfinger.type = SDL_FINGERMOTION;
            t.tfinger.timestamp = e.motion.timestamp;
            t.tfinger.touchId = multitouch_device;
            t.tfinger.fingerId = multitouch_finger;
            t.tfinger.x = e.motion.x / float(win_width);
            t.tfinger.y = e.motion.y / float(win_height);
            t.tfinger.dx = e.motion.xrel / float(win_width);
            t.tfinger.dy = e.motion.yrel / float(win_height);
            t.tfinger.pressure = 0.5f; // FIXME
#if 0
            t.tfinger.windowID = e.motion.windowID;
#endif
            SDL_PushEvent(&t);
          }
        }
        else
        {
          mouse_x = e.motion.x;
          mouse_y = e.motion.y;
        }
      }
      break;

    case SDL_MOUSEBUTTONDOWN:
      if (e.button.which != SDL_TOUCH_MOUSEID)
      {
        if (multitouch_emulation)
        {
          if (e.button.button == SDL_BUTTON_LEFT)
          {
            multitouch_finger = multitouch_add_finger(e.button.x, e.button.y);
            SDL_Event t;
            t.type = SDL_FINGERDOWN;
            t.tfinger.type = SDL_FINGERDOWN;
            t.tfinger.timestamp = e.button.timestamp;
            t.tfinger.touchId = multitouch_device;
            t.tfinger.fingerId = multitouch_finger;
            t.tfinger.x = e.button.x / float(win_width);
            t.tfinger.y = e.button.y / float(win_height);
            t.tfinger.dx = 0.0f;
            t.tfinger.dy = 0.0f;
            t.tfinger.pressure = 0.5f; // FIXME
#if 0
            t.tfinger.windowID = e.button.windowID;
#endif
            SDL_PushEvent(&t);
          }
          else if (e.button.button == SDL_BUTTON_RIGHT)
          {
            coord_t x = e.button.x;
            coord_t y = e.button.y;
            multitouch_finger = multitouch_remove_finger(x, y);
            if (multitouch_finger)
            {
              SDL_Event t;
              t.type = SDL_FINGERUP;
              t.tfinger.type = SDL_FINGERUP;
              t.tfinger.timestamp = e.button.timestamp;
              t.tfinger.touchId = multitouch_device;
              t.tfinger.fingerId = multitouch_finger;
              t.tfinger.x = x / float(win_width);
              t.tfinger.y = y / float(win_height);
              t.tfinger.dx = 0.0f;
              t.tfinger.dy = 0.0f;
              t.tfinger.pressure = 0.5f; // FIXME
#if 0
              t.tfinger.windowID = e.button.windowID;
#endif
              SDL_PushEvent(&t);
            }
          }
        }
        else
        {
          switch (e.button.button)
          {
            case SDL_BUTTON_LEFT:
              drag = true;
              drag_start_x = e.button.x;
              drag_start_y = e.button.y;
              break;
            case SDL_BUTTON_RIGHT:
              if (drag)
              {
                drag = false;
              }
              else
              {
                STOP
                double cx, cy;
                window_to_image1(win_width, win_height, e.button.x, e.button.y, &cx, &cy);
                zoom(par, cx, cy, 0.5);
                window_to_image(win_width, win_height, e.button.x, e.button.y, &cx, &cy);
                mat3 T = mat3(1.0f);
                T = glm::translate(T, vec2(float(cx), float(dsp->height - cy)));
                T = glm::scale(T, vec2(float(0.5f), float(0.5f)));
                T = glm::translate(T, -vec2(float(cx), float(dsp->height - cy)));
                finger_transform_started = T * finger_transform_started;
                restart = true;
              }
              break;
            case SDL_BUTTON_MIDDLE:
              {
                STOP
                double cx, cy;
                window_to_image1(win_width, win_height, e.button.x, e.button.y, &cx, &cy);
                zoom(par, cx, cy, 1, false);
                window_to_image(win_width, win_height, e.button.x, e.button.y, &cx, &cy);
                mat3 T = mat3(1.0f);
                T = glm::translate(T, -vec2(float(cx - dsp->width / 2.0), float(dsp->height - cy - dsp->height / 2.0)));
                finger_transform_started = T * finger_transform_started;
                restart = true;
              }
            default:
              break;
          }
        }
      }
      break;

    case SDL_MOUSEBUTTONUP:
      if (e.button.which != SDL_TOUCH_MOUSEID)
      {
        if (multitouch_emulation)
        {
        }
        else
        {
          switch (e.button.button)
          {
            case SDL_BUTTON_LEFT:
              if (drag)
              {
                double drag_end_x = e.button.x;
                double drag_end_y = e.button.y;
                double cx, cy, mx, my;
                window_to_image1(win_width, win_height, drag_start_x, drag_start_y, &cx, &cy);
                window_to_image1(win_width, win_height, drag_end_x, drag_end_y, &mx, &my);
                double r = std::hypot(mx - cx, my - cy);
                double d = std::min(std::max(1 / r, 1.0/16.0), 16.0);
                drag = false;
                switch (mouse_action)
                {
                  case 0:
                    {
                      STOP
                      zoom(par, cx, cy, d, false);
                      window_to_image(win_width, win_height, drag_start_x, drag_start_y, &cx, &cy);
                      mat3 T = mat3(1.0f);
                      T = glm::translate(T, vec2(float(dsp->width / 2.0), float(dsp->height / 2.0)));
                      T = glm::scale(T, vec2(float(d), float(d)));
                      T = glm::translate(T, -vec2(float(dsp->width / 2.0), float(dsp->height / 2.0)));
                      T = glm::translate(T, -vec2(float(cx - dsp->width / 2.0), float(dsp->height - cy - dsp->height / 2.0)));
                      finger_transform_started = T * finger_transform_started;
                      restart = true;
                    }
                    break;
                  case 1:
                    {
                      STOP
                      newton_c = get_delta_c(par, cx, cy);
                      newton_r = d / par.zoom;
                      start_newton = true;
                    }
                    break;
                }
              }
              break;
            default:
              break;
          }
        }
      }
      break;

    case SDL_MOUSEWHEEL:
      if (e.wheel.which != SDL_TOUCH_MOUSEID)
      {
        if (e.wheel.y > 0)
        {
          double cx, cy;
          window_to_image1(win_width, win_height, mouse_x, mouse_y, &cx, &cy);
          STOP
          zoom(par, cx, cy, 2);
          window_to_image(win_width, win_height, mouse_x, mouse_y, &cx, &cy);
          mat3 T = mat3(1.0f);
          T = glm::translate(T, vec2(float(cx), float(dsp->height - cy)));
          T = glm::scale(T, vec2(float(2), float(2)));
          T = glm::translate(T, -vec2(float(cx), float(dsp->height - cy)));
          finger_transform_started = T * finger_transform_started;
          restart = true;
        }
        if (e.wheel.y < 0)
        {
          double cx, cy;
          window_to_image1(win_width, win_height, mouse_x, mouse_y, &cx, &cy);
          STOP
          zoom(par, cx, cy, 0.5);
          window_to_image(win_width, win_height, mouse_x, mouse_y, &cx, &cy);
          mat3 T = mat3(1.0f);
          T = glm::translate(T, vec2(float(cx), float(dsp->height - cy)));
          T = glm::scale(T, vec2(float(0.5f), float(0.5f)));
          T = glm::translate(T, -vec2(float(cx), float(dsp->height - cy)));
          finger_transform_started = T * finger_transform_started;
          restart = true;
        }
      }
      break;

    case SDL_FINGERDOWN:
      switch (mouse_action)
      {
        case 0:
          {
            if (fingers.size() == 0)
            {
              finger_device = e.tfinger.touchId;
            }
            if (finger_device == e.tfinger.touchId)
            {
              vec3 f = vec3(e.tfinger.x * win_width, (1 - e.tfinger.y) * win_height, 1.0f);
              fingers[e.tfinger.fingerId] = std::pair<vec3, vec3>(f, f);
              update_finger_transform(win_width, win_height);
            }
          }
          break;
        case 1:
          {
            STOP
            newton_c = get_delta_c(par, (e.tfinger.x - 0.5) * 2, (0.5 - e.tfinger.y) * 2);
            newton_r = 0.1 / par.zoom;
            start_newton = true;
          }
          break;
      }
      break;
    case SDL_FINGERUP:
      if (finger_device == e.tfinger.touchId)
      {
        vec3 f = vec3(e.tfinger.x * win_width, (1 - e.tfinger.y) * win_height, 1.0f);
        fingers[e.tfinger.fingerId].second = f;
        update_finger_transform(win_width, win_height);
        fingers.erase(e.tfinger.fingerId);
        if (fingers.size() == 0)
        {
          STOP
          mat3 S = mat3(1.0f);
          // [0..w] x [0..h]
          S = glm::scale(S, vec2(float(dsp->width), float(dsp->height)));
          S = glm::scale(S, vec2(0.5f, 0.5f));
          S = glm::translate(S, vec2(1.0f));
          // [-1..1] x [-1..1]
          S = glm::inverse(S) * finger_transform * S;
          zoom(par, glm::inverse(S), finger_transform);
          finger_transform_started = finger_transform * finger_transform_started;
          finger_transform = mat3(1.0f);
          restart = true;
        }
      }
      break;

    case SDL_FINGERMOTION:
      if (finger_device == e.tfinger.touchId)
      {
        vec3 f = vec3(e.tfinger.x * win_width, (1 - e.tfinger.y) * win_height, 1.0f);
        fingers[e.tfinger.fingerId].second = f;
        update_finger_transform(win_width, win_height);
      }
      break;

    case SDL_KEYDOWN:
      switch (e.key.keysym.sym)
      {
        case SDLK_ESCAPE:
          STOP
          break;

        case SDLK_F5:
          STOP
          restart = true;
          break;

        case SDLK_F10:
          show_windows = ! show_windows;
          break;

#if FULLSCREEN_OPTION
        case SDLK_F11:
           fullscreen = ! fullscreen;
           SDL_SetWindowFullscreen(window, fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
           break;
#endif

        case SDLK_KP_PLUS:
          STOP
          if (ctrl)
          {
            if (par.p.bailout.maximum_bla_steps < count_t(1) << 48)
            {
              par.p.bailout.maximum_bla_steps <<= 1;
              if (window_state.iterations_limited)
              {
                par.p.bailout.maximum_bla_steps = std::min(par.p.bailout.maximum_bla_steps, window_state.bla_steps_limit);
              }
            }
          }
          else if (shift)
          {
            if (par.p.bailout.maximum_perturb_iterations < count_t(1) << 48)
            {
              par.p.bailout.maximum_perturb_iterations <<= 1;
              if (window_state.iterations_limited)
              {
                par.p.bailout.maximum_perturb_iterations = std::min(par.p.bailout.maximum_perturb_iterations, window_state.perturb_iterations_limit);
              }
            }
          }
          else
          {
            if (par.p.bailout.iterations < count_t(1) << 48)
            {
              par.p.bailout.iterations <<= 1;
              par.p.bailout.maximum_reference_iterations <<= 1;
              if (window_state.iterations_limited)
              {
                par.p.bailout.iterations = std::min(par.p.bailout.iterations, window_state.iterations_limit);
                par.p.bailout.maximum_reference_iterations = std::min(par.p.bailout.maximum_reference_iterations, window_state.iterations_limit);
              }
            }
          }
          restring_vals(par);
          restart = true;
          break;
        case SDLK_KP_MINUS:
          STOP
          if (ctrl)
          {
            if (par.p.bailout.maximum_bla_steps > count_t(1) << 8)
            {
              par.p.bailout.maximum_bla_steps >>= 1;
            }
          }
          else if (shift)
          {
            if (par.p.bailout.maximum_perturb_iterations > count_t(1) << 8)
            {
              par.p.bailout.maximum_perturb_iterations >>= 1;
            }
          }
          else
          {
            if (par.p.bailout.iterations > count_t(1) << 8)
            {
              par.p.bailout.iterations >>= 1;
              par.p.bailout.maximum_reference_iterations >>= 1;
            }
          }
          restring_vals(par);
          restart = true;
          break;

        case SDLK_PAGEUP:
        case SDLK_KP_0:
        {
          STOP
          zoom(par, 0, 0, 0.5);
          float x = dsp->width * 2 / 4.0;
          float y = dsp->height * 2 / 4.0;
          mat3 T = mat3(1.0f);
          T = glm::translate(T, vec2(float(x), float(dsp->height - y)));
          T = glm::scale(T, vec2(float(0.5f), float(0.5f)));
          T = glm::translate(T, -vec2(float(x), float(dsp->height - y)));
          finger_transform_started = T * finger_transform_started;
          restart = true;
          break;
        }

#define TRANSFORM \
          mat3 T = mat3(1.0f); \
          T = glm::translate(T, vec2(float(x), float(dsp->height - y))); \
          T = glm::scale(T, vec2(float(2), float(2))); \
          T = glm::translate(T, -vec2(float(x), float(dsp->height - y))); \
          finger_transform_started = T * finger_transform_started;
        case SDLK_KP_1:
        {
          STOP
          zoom(par, -1, 1, 2);
          float x = 0;
          float y = dsp->height;
          TRANSFORM
          restart = true;
          break;
        }
        case SDLK_KP_2:
        {
          STOP
          zoom(par, 0, 1, 2);
          float x = dsp->width * 0.5;
          float y = dsp->height;
          TRANSFORM
          restart = true;
          break;
        }
        case SDLK_KP_3:
        {
          STOP
          zoom(par, 1, 1, 2);
          float x = dsp->width;
          float y = dsp->height;
          TRANSFORM
          restart = true;
          break;
        }
        case SDLK_KP_4:
        {
          STOP
          zoom(par, -1, 0, 2);
          float x = 0;
          float y = dsp->height * 0.5;
          TRANSFORM
          restart = true;
          break;
        }
        case SDLK_PAGEDOWN:
        case SDLK_KP_5:
        {
          STOP
          zoom(par, 0, 0, 2);
          float x = dsp->width * 0.5;
          float y = dsp->height * 0.5;
          TRANSFORM
          restart = true;
          break;
        }
        case SDLK_KP_6:
        {
          STOP
          zoom(par, 1, 0, 2);
          float x = dsp->width;
          float y = dsp->height * 0.5;
          TRANSFORM
          restart = true;
          break;
        }
        case SDLK_KP_7:
        {
          STOP
          zoom(par, -1, -1, 2);
          float x = 0;
          float y = 0;
          TRANSFORM
          restart = true;
          break;
        }
        case SDLK_KP_8:
        {
          STOP
          zoom(par, 0, -1, 2);
          float x = dsp->width * 0.5;
          float y = 0;
          TRANSFORM
          restart = true;
          break;
        }
        case SDLK_KP_9:
        {
          STOP
          zoom(par, 1, -1, 2);
          float x = dsp->width;
          float y = 0;
          TRANSFORM
          restart = true;
          break;
        }
#undef TRANSFORM

        case SDLK_HOME:
        {
          if (shift)
          {
            STOP
            home(par);
            restart = true;
          }
          else if (ctrl)
          {
            STOP
            reset(par);
            restart = true;
          }
          break;
        }

        case SDLK_q:
          if (ctrl)
          {
            quit = true;
          }
          break;

        case SDLK_c:
          if (ctrl)
          {
            clipboard_copy();
          }
          break;
        case SDLK_v:
          if (ctrl)
          {
            clipboard_paste();
          }
          break;

        default:
          break;
      }
      break;

    case SDL_USEREVENT:
      switch (e.user.code)
      {
        case code_persist:
          persist_state();
          break;
      }
      break;
  }
}

void display_background(SDL_Window *window, display_gles &dsp)
{
  int win_width = 0;
  int win_height = 0;
  SDL_GetWindowSize(window, &win_width, &win_height);
  int display_w = 0, display_h = 0;
  SDL_GL_GetDrawableSize(window, &display_w, &display_h);
  // draw
  dsp.draw(display_w, display_h, finger_transform * finger_transform_started, srgb_conversion, false);
  if (drag)
  {
    double cx, cy, mx, my;
    window_to_image1(win_width, win_height, drag_start_x, drag_start_y, &cx, &cy);
    window_to_image1(win_width, win_height, mouse_x, mouse_y, &mx, &my);
    double r = std::hypot(mx - cx, my - cy);
    double x0 = cx - r;
    double x1 = cx + r;
    double y0 = cy - r;
    double y1 = cy + r;
    dsp.draw_rectangle(display_w, display_h, x0, y0, x1, y1, srgb_conversion);
  }
  if (fingers.size() > 0)
  {
    std::vector<glm::vec4> circles;
    float rx = 0.01;
    float ry = rx * win_width / win_height;
    for (const auto &f : fingers)
    {
      double cx = (f.second.second[0] - win_width / 2.0) / (win_width / 2.0);
      double cy = (f.second.second[1] - win_height / 2.0) / (win_height / 2.0);
      circles.push_back(glm::vec4(float(cx), float(cy), rx, ry));
    }
    dsp.draw_circles(display_w, display_h, circles, srgb_conversion);
  }
}

void capture_background(SDL_Window *window, display_gles &dsp)
{
  int display_w = 0, display_h = 0;
  SDL_GL_GetDrawableSize(window, &display_w, &display_h);
  dsp.draw(display_w, display_h, finger_transform * finger_transform_started, srgb_conversion, true);
}

void display_window_window()
{
  ImGui::SetNextWindowPos(ImVec2(16, 16), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(16 * ImGui::GetFontSize(), 36 * ImGui::GetFontSize()), ImGuiCond_FirstUseEver);
  ImGui::Begin("Fraktaler 3");
//  ImGui::Combo("##MouseAction", &mouse_action, "Navigate\0");// "Newton\0");
#define W(w) (fullscreen ? &window_state.fullscreen.w.show : &window_state.windowed.w.show)
  ImGui::Checkbox("Input/Ouput", W(io));
  ImGui::Checkbox("Formula", W(formula));
  ImGui::Checkbox("Status", W(status));
  ImGui::Checkbox("Location", W(location));
  ImGui::Checkbox("Reference", W(reference));
  ImGui::Checkbox("Bailout", W(bailout));
  ImGui::Checkbox("Transform", W(transform));
  ImGui::Checkbox("Algorithm", W(algorithm));
  ImGui::Checkbox("Information", W(information));
  ImGui::Checkbox("Quality", W(quality));
  ImGui::Checkbox("Render", W(render));
  ImGui::Checkbox("Colours", W(colours));
  ImGui::Checkbox("Postprocessing", W(postprocessing));
  ImGui::Checkbox("Newton Zooming", W(newton));
  ImGui::Checkbox("Wisdom", W(wisdom));
  ImGui::Checkbox("Preferences", W(preferences));
  ImGui::Checkbox("About", W(about));
#ifdef HAVE_IMGUI_DEMO
  ImGui::Checkbox("ImGui Demo", W(demo));
#endif
#undef W
  ImGui::Text("Press F10 to toggle all");
  ImGui::End();
}

#if LOADSAVE_DIALOGS
ImGui::FileBrowser *load_dialog = nullptr;
ImGui::FileBrowser *save_dialog = nullptr;
ImGui::FileBrowser *render_dialog = nullptr;
#endif

#if WISDOM_DIALOGS
ImGui::FileBrowser *wisdom_load_dialog = nullptr;
ImGui::FileBrowser *wisdom_save_dialog = nullptr;
#else
bool wisdom_load_unlocked = false;
bool wisdom_save_unlocked = false;
#endif

void display_set_window_dims(const struct window &w)
{
  const auto &io = ImGui::GetIO();
  int width = w.w > 0 ? w.w * ImGui::GetFontSize() : io.DisplaySize.x - 16 * 2;
  int height = w.h > 0 ? w.h * ImGui::GetFontSize() : io.DisplaySize.y - 16 * 2;
  int x = w.x > 0 ? w.x : (io.DisplaySize.x - width) / 2;
  int y = w.y > 0 ? w.y : (io.DisplaySize.y - height) / 2;
  ImGui::SetNextWindowPos(ImVec2(x, y), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(width, height), ImGuiCond_FirstUseEver);
}

void display_get_window_dims(struct window &w)
{
  ImVec2 p = ImGui::GetWindowPos();
  ImVec2 s = ImGui::GetWindowSize();
  w.x = p.x;
  w.y = p.y;
  w.w = s.x / (float) ImGui::GetFontSize();
  w.h = s.y / (float) ImGui::GetFontSize();
}

void clipboard_copy()
{
  gui_pre_save(par);
  std::string s = par.to_string();
#ifdef __EMSCRIPTEN__
  clipboard_content = s;
  emscripten_browser_clipboard::copy(s);
#else
  SDL_SetClipboardText(s.c_str());
#endif
}

bool load_colouring_only = false;
bool save_raw_exr = false;

void clipboard_paste()
{
#ifdef __EMSCRIPTEN__
  try
  {
    if (load_colouring_only)
    {
      param par2;
      par2.from_string(clipboard_content);
      colour_set_shader(clr, par2.p.colour.shader);
      colour_set_uniforms(clr, par2.p.colour.uniforms);
      recolour = true;
    }
    else
    {
      STOP
      par.from_string(clipboard_content);
      gui_post_load(par);
      restart = true;
    }
  }
  catch (...)
  {
    // FIXME
  }
#else
  char *t = SDL_GetClipboardText();
  try
  {
    if (load_colouring_only)
    {
      param par2;
      par2.from_string(std::string(t));
      colour_set_shader(clr, par2.p.colour.shader);
      colour_set_uniforms(clr, par2.p.colour.uniforms);
      recolour = true;
    }
    else
    {
      STOP
      par.from_string(std::string(t));
      gui_post_load(par);
      restart = true;
    }
  }
  catch (...)
  {
    // FIXME
  }
  SDL_free(t);
#endif
}

bool reset_unlocked = false;
bool home_unlocked = false;

#define WINDOW(title,w) \
  display_set_window_dims(fullscreen ? window_state.fullscreen.w : window_state.windowed.w); \
  ImGui::Begin(title, open); \
  display_get_window_dims(fullscreen ? window_state.fullscreen.w : window_state.windowed.w);

#ifdef __EMSCRIPTEN__
int download_type = 4; // toml
#endif

bool save_any(const std::string &filename)
{
  bool result = false;
  if (ends_with(filename, ".exr"))
  {
    int threads = std::thread::hardware_concurrency();
#ifdef __EMSCRIPTEN__
    if (download_type == 1)
    {
      if (subframes_rendered > 0 && raw)
      {
        result = image_raw(*raw, ! par.p.transform.vertical_flip).save_exr(filename, par.exr_channels, par.p.bailout.iterations, threads, par.to_string(), par.p.image.dpi);
      }
    }
    else
#endif
    {
      result = image_rgb(*rgb, ! par.p.transform.vertical_flip).save_exr(filename, threads, par.to_string(), par.p.image.dpi);
    }
  }
  else if (ends_with(filename, ".png"))
  {
    const bool dither = true; // FIXME
    result = image_rgb8(*rgb, ! par.p.transform.vertical_flip, dither).save_png(filename, par.to_string(), par.p.image.dpi);
  }
  else if (ends_with(filename, ".jpg") || ends_with(filename, ".jpeg"))
  {
    const int jpeg_quality = 97; // FIXME
    const bool dither = true; // FIXME
    result = image_yuv8(*rgb, ! par.p.transform.vertical_flip, dither).save_jpeg(filename, par.to_string(), jpeg_quality, par.p.image.dpi);
  }
  else if (ends_with(filename, ".glsl"))
  {
    result = par.save_glsl(filename);
  }
  else if (ends_with(filename, ".csv"))
  {
    result = colour_save_csv(clr, filename);
  }
  else
  {
    result = par.save_toml(filename);
  }
#ifdef __EMSCRIPTEN__
  download_handler(filename);
  unlink(filename.c_str());
#endif
  syncfs();
  return result;
}

void display_io_window(bool *open)
{
  WINDOW("Input/Output", io)
  ImGui::Checkbox("##ResetUnlocked", &reset_unlocked);
  ImGui::SameLine();
  if (ImGui::Button("Reset") && reset_unlocked)
  {
    STOP
    reset_unlocked = false;
    reset(par);
    restart = true;
  }
  ImGui::SameLine();
  ImGui::Checkbox("##HomeUnlocked", &home_unlocked);
  ImGui::SameLine();
  if (ImGui::Button("Home") && home_unlocked)
  {
    STOP
    home_unlocked = false;
    home(par);
    restart = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("Copy"))
  {
    clipboard_copy();
  }
#ifndef __EMSCRIPTEN__
  ImGui::SameLine();
  if (ImGui::Button("Paste"))
  {
    clipboard_paste();
  }
#endif
#ifdef __EMSCRIPTEN__
  ImGui::SameLine();
  if (ImGui::Button("Upload"))
  {
    emscripten_browser_file::upload(".toml,.jpg,.jpeg,.png,.exr,.glsl,.csv,.kfr", upload_handler);
  }
  ImGui::SameLine();
  if (ImGui::Button("Download"))
  {
    try
    {
      const char *download_type_extension[] = {"exr", "exr", "jpg", "png", "toml", "glsl", "csv"};
      tm lt = *localtime(&timestamp);
      std::ostringstream filename;
      filename << "/tmp/" << std::setfill('0')
        << std::setw(4) << (lt.tm_year + 1900) << "-"
        << std::setw(2) << (lt.tm_mon + 1) << "-"
        << std::setw(2) << lt.tm_mday << "T"
        << std::setw(2) << lt.tm_hour << "-"
        << std::setw(2) << lt.tm_min << "-"
        << std::setw(2) << lt.tm_sec << ".f3."
        << download_type_extension[download_type];
      gui_pre_save(par);
      bool ok = save_any(filename.str());
      if (! ok)
      {
        display_error_string = std::string("could not save file ") + filename.str();
      }
    }
    catch (const std::exception &e)
    {
      SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "saving error: %s", e.what());
    }
  }
  ImGui::SameLine();
  ImGui::PushItemWidth(6 * ImGui::GetFontSize());
  ImGui::Combo("##DownloadType", &download_type, "exr\0" "exr (raw)\0" "jpg\0" "png\0" "toml\0" "glsl\0" "csv\0");
  ImGui::PopItemWidth();
#endif
#if LOADSAVE_DIALOGS
  if (load_dialog)
  {
    ImGui::SameLine();
    if (ImGui::Button("Load"))
    {
      load_dialog->Open();
    }
  }
  if (save_dialog)
  {
    ImGui::SameLine();
    if (ImGui::Button("Save"))
    {
      save_dialog->Open();
    }
  }
#endif
  ImGui::Checkbox("Load Colouring Only", &load_colouring_only);
#ifndef __EMSCRIPTEN__
  ImGui::SameLine();
  ImGui::Checkbox("Save RAW EXR", &save_raw_exr);
#endif
  ImGui::End();
#if LOADSAVE_DIALOGS
  if (load_dialog)
  {
    load_dialog->Display();
    if (load_dialog->HasSelected())
    {
      std::string filename = load_dialog->GetSelected().string();
      try
      {
        if (ends_with(filename, ".csv"))
        {
          colour_load_csv(clr, filename);
          recolour = true;
        }
        else if (ends_with(filename, ".glsl"))
        {
          par.load_glsl(filename);
          colour_set_shader(clr, par.p.colour.shader);
          recolour = true;
        }
        else
        {
          if (load_colouring_only)
          {
            param par2;
            par2.load_any(filename);
            colour_set_shader(clr, par2.p.colour.shader);
            colour_set_uniforms(clr, par2.p.colour.uniforms);
            par.p.colour.background = par2.p.colour.background;
            par.p.colour.uses_histogram = par2.p.colour.uses_histogram;
            recolour = true;
          }
          else
          {
            STOP
            par.load_any(filename);
            gui_post_load(par);
            restart = true;
          }
        }
      }
      catch (std::exception &e)
      {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "loading \"%s\": %s", filename.c_str(), e.what());
      }
      load_dialog->ClearSelected();
    }
  }
  if (save_dialog)
  {
    save_dialog->Display();
    if (save_dialog->HasSelected())
    {
      std::string filename = save_dialog->GetSelected().string();
      bool ok = false;
      try
      {
        if (ends_with(filename, ".csv"))
        {
          ok = colour_save_csv(clr, filename);
        }
        else if (ends_with(filename, ".exr") && save_raw_exr)
        {
          if (subframes_rendered > 0 && raw)
          {
            gui_pre_save(par);
            int threads = std::thread::hardware_concurrency();
            ok = image_raw(*raw, ! par.p.transform.vertical_flip).save_exr(filename, par.exr_channels, par.p.bailout.iterations, threads, par.to_string(), par.p.image.dpi);
          }
        }
        else
        {
          gui_pre_save(par);
          ok = save_any(filename);
        }
      }
      catch (const std::exception &e)
      {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "saving \"%s\": %s", filename.c_str(), e.what());
      }
      save_dialog->ClearSelected();
      if (! ok)
      {
        display_error_string = "could not save file " + filename;
      }
    }
  }
#endif
}

void display_status_window(bool *open)
{
  const count_t count = par.p.formula.per.size();
  const char *status = "Status: unknown";
  if (subframes_rendered >= par.p.image.subframes && par.p.image.subframes > 0)
  {
    status = "Completed";
  }
  else if (! running)
  {
    status = "Cancelled";
  }
  else
  {
    status = "Working...";
  }
  WINDOW("Status", status)
  ImGui::Text("%s", status);
  float r = 0;
  for (int i = 0; i < count; ++i)
  {
    r += progress[i];
  }
  r /= count;
  char ref[20];
  std::snprintf(ref, sizeof(ref), "Ref: %3d%%", (int)(r * 100));
  ImGui::ProgressBar(r, ImVec2(-1.f, 0.f), ref);
  float a = 0;
  for (int i = 0; i < count; ++i)
  {
    a += progress[count + i];
  }
  a /= count;
  char apx[20];
  std::snprintf(apx, sizeof(apx), "BLA: %3d%%", (int)(a * 100));
  ImGui::ProgressBar(a, ImVec2(-1.f, 0.f), apx);
  char sub[20];
  float p = subframes_rendered / progress_t(par.p.image.subframes <= 0 ? subframes_rendered + 1 : par.p.image.subframes);
  std::snprintf(sub, sizeof(sub), "Sub: %3d%%", (int)(p * 100));
  ImGui::ProgressBar(p, ImVec2(-1.f, 0.f), sub);
  char pix[20];
  p = progress[2 * count + 1];
  std::snprintf(pix, sizeof(pix), "Pix: %3d%%", (int)(p * 100));
  ImGui::ProgressBar(p, ImVec2(-1.f, 0.f), pix);
  count_t ms = std::ceil(1000 * duration.count());
  count_t s = ms / 1000;
  count_t m = s / 60;
  count_t h = m / 60;
  count_t d = h / 24;
  if (d > 0)
  {
    ImGui::Text("T: %dd%02dh%02dm%02ds%03dms", int(d), int(h % 24), int(m % 60), int(s % 60), int(ms % 1000));
  }
  else if (h > 0)
  {
    ImGui::Text("T: %dh%02dm%02ds%03dms", int(h), int(m % 60), int(s % 60), int(ms % 1000));
  }
  else if (m > 0)
  {
    ImGui::Text("T: %dm%02ds%03dms", int(m), int(s % 60), int(ms % 1000));
  }
  else if (s > 0)
  {
    ImGui::Text("T: %ds%03dms", int(s), int(ms % 1000));
  }
  else
  {
    ImGui::Text("T: %dms", int(ms));
  }
  ImGui::End();
}

bool InputFloatExp(const char *label, floatexp<float, int> *x, std::string *str)
{

  if (ImGui::InputText(label, str, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CharsScientific))
  {
    mpfr_t z;
    mpfr_init2(z, 53);
    mpfr_set_str(z, str->c_str(), 10, MPFR_RNDN);
    long e = 0;
    double m = mpfr_get_d_2exp(&e, z, MPFR_RNDN);
    mpfr_clear(z);
    *x = floatexp<float, int>(m, e);
    return true;
  }
  return false;
}

const char * const op_string_gui[op_count] = { "(delete)", "store", "mul", "sqr", "absx", "absy", "negx", "negy", "rot" };

void display_formula_window(bool *open)
{
  WINDOW("Formula", formula)
  auto f = par.p.formula.per;
  count_t count = f.size();
  bool changed = false;
  for (count_t i = 0; i < count; ++i)
  {
    ImGui::PushID(i);
    count_t nopcodes = f[i].opcodes.size();
    if (nopcodes)
    {
      // advanced mode
      for (count_t j = 0; j < nopcodes; ++j)
      {
        ImGui::PushID(j);
        if (ImGui::Button("+"))
        {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
          f[i].opcodes.insert(f[i].opcodes.begin() + j, { op_sqr } );
#pragma GCC diagnostic pop
          changed |= true;
          ++nopcodes;
        }
        ImGui::SameLine();
        switch (f[i].opcodes[j].op)
        {
          case op_add:
            ImGui::TextUnformatted("add");
            break;
          default:
            {
              int item = f[i].opcodes[j].op;
              ImGui::PushItemWidth(4 * ImGui::GetFontSize());
              if (ImGui::Combo("", &item, op_string_gui, op_count))
              {
                if (item == 0)
                {
                  f[i].opcodes.erase(f[i].opcodes.begin() + j);
                  --nopcodes;
                }
                else
                {
                  f[i].opcodes[j].op = opcode_tag(item);
                  if (f[i].opcodes[j].op == op_rot)
                  {
                    f[i].opcodes[j].u.rot.x = 0;
                    f[i].opcodes[j].u.rot.y = 1;
                  }
                }
                changed |= true;
              }
              if (f[i].opcodes[j].op == op_rot)
              {
                ImGui::SameLine();
                float degrees = std::atan2(f[i].opcodes[j].u.rot.y, f[i].opcodes[j].u.rot.x) * 180.0f / float(M_PI);
                if (ImGui::InputFloat("##RotDegrees", &degrees))
                {
                  float radians = degrees / 180.0f * float(M_PI);
                  f[i].opcodes[j].u.rot.x = std::cos(radians);
                  f[i].opcodes[j].u.rot.y = std::sin(radians);
                  changed |= true;
                }
              }
              ImGui::PopItemWidth();
            }
        }
        ImGui::SameLine();
        ImGui::PopID();
      }
      if (ImGui::Button("simple"))
      {
        f[i].power = par.degrees[i];
        f[i].opcodes = std::vector<opcode>();
        changed |= true;
      }
      ImGui::SameLine();
    }
    else
    {
      // simple mode
      changed |= ImGui::Checkbox("|X|", &f[i].abs_x); ImGui::SameLine();
      changed |= ImGui::Checkbox("|Y|", &f[i].abs_y); ImGui::SameLine();
      changed |= ImGui::Checkbox("-X", &f[i].neg_x); ImGui::SameLine();
      changed |= ImGui::Checkbox("-Y", &f[i].neg_y); ImGui::SameLine();
      ImGui::PushItemWidth(6 * ImGui::GetFontSize());
      changed |= ImGui::InputInt("P", &f[i].power, 1, 5); ImGui::SameLine();
      if (f[i].power < 2)
      {
        f[i].power = 2;
      }
      ImGui::PopItemWidth();
      if (ImGui::Button("advanced"))
      {
        f[i].opcodes = par.opss[i];
        changed |= true;
      }
      ImGui::SameLine();
    }
    if (ImGui::Button("+"))
    {
      f.insert(f.begin() + i, f[i]);
      ++count;
      changed |= true;
    }
    if (1 < count)
    {
      ImGui::SameLine();
      if (ImGui::Button("-"))
      {
        f.erase(f.begin() + i);
        --count;
        changed |= true;
      }
    }
    ImGui::PopID();
  }
  if (changed)
  {
    STOP
    par.p.formula.per = f;
    post_edit_formula(par);
    restart = true;
  }
  ImGui::End();
}

void display_location_window(bool *open)
{
  WINDOW("Location", location)
  ImGui::Text("Zoom");
  ImGui::SameLine();
  ImGui::PushItemWidth(-FLT_MIN);
  if (ImGui::InputText("##Zoom", &par.p.location.zoom, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CharsScientific))
  {
    STOP
    mpfr_t zoom;
    mpfr_init2(zoom, 53);
    mpfr_set_str(zoom, par.p.location.zoom.c_str(), 10, MPFR_RNDN);
    long e = 0;
    double m = mpfr_get_d_2exp(&e, zoom, MPFR_RNDN);
    mpfr_clear(zoom);
    par.zoom = floatexp<float, int>(m, e);
    // adjust (increase only) precision of center and reference,
    // so that entering new coordinates works as expected,
    // and zoom can be increased again after being decreased
    mpfr_prec_t prec = std::max
      ( (mpfr_prec_t)(std::max(24, 24 + (par.zoom * par.p.image.height).exp))
      , std::max
          ( std::max(mpfr_get_prec(par.reference.x.mpfr_ptr()), mpfr_get_prec(par.reference.y.mpfr_ptr()))
          , std::max(mpfr_get_prec(par.center.x.mpfr_ptr()), mpfr_get_prec(par.center.y.mpfr_ptr()))
          )
      );
    mpfr_prec_round(par.center.x.mpfr_ptr(), prec, MPFR_RNDN);
    mpfr_prec_round(par.center.y.mpfr_ptr(), prec, MPFR_RNDN);
    mpfr_prec_round(par.reference.x.mpfr_ptr(), prec, MPFR_RNDN);
    mpfr_prec_round(par.reference.y.mpfr_ptr(), prec, MPFR_RNDN);
    restring_vals(par);
    restart = true;
  }
  ImGui::PopItemWidth();
  ImGui::Text("Real");
  ImGui::SameLine();
  ImGui::PushItemWidth(-FLT_MIN);
  if (ImGui::InputText("##Real", &par.p.location.real, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CharsScientific))
  {
    STOP
    mpfr_set_str(par.center.x.mpfr_ptr(), par.p.location.real.c_str(), 10, MPFR_RNDN);
    restring_locs(par);
    restart = true;
  }
  ImGui::PopItemWidth();
  ImGui::Text("Imag");
  ImGui::SameLine();
  ImGui::PushItemWidth(-FLT_MIN);
  if (ImGui::InputText("##Imag", &par.p.location.imag, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CharsScientific))
  {
    STOP
    mpfr_set_str(par.center.y.mpfr_ptr(), par.p.location.imag.c_str(), 10, MPFR_RNDN);
    restring_locs(par);
    restart = true;
  }
  ImGui::PopItemWidth();
  ImGui::End();
}

void display_reference_window(bool *open)
{
  WINDOW("Reference", reference)
  ImGui::Text("Period");
  ImGui::SameLine();
  ImGui::PushItemWidth(-FLT_MIN);
  if (ImGui::InputText("##Period", &par.s_period, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CharsScientific))
  {
    STOP
    unstring_vals(par);
    restart = true;
  }
  ImGui::PopItemWidth();
  ImGui::Text("Real");
  ImGui::SameLine();
  ImGui::PushItemWidth(-FLT_MIN);
  if (ImGui::InputText("##Real", &par.p.reference.real, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CharsScientific))
  {
    STOP
    unstring_locs(par);
    restart = true;
  }
  ImGui::PopItemWidth();
  ImGui::Text("Imag");
  ImGui::SameLine();
  ImGui::PushItemWidth(-FLT_MIN);
  if (ImGui::InputText("##Imag", &par.p.reference.imag, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CharsScientific))
  {
    STOP
    unstring_locs(par);
    restart = true;
  }
  ImGui::PopItemWidth();
  ImGui::End();
}

void display_bailout_window(bool *open)
{
  WINDOW("Bailout", bailout)
  ImGui::Text("Iterations   ");
  ImGui::SameLine();
  if (ImGui::Button("-##IterationsDown"))
  {
    STOP
    par.p.bailout.iterations >>= 1;
    par.p.bailout.iterations = std::max(par.p.bailout.iterations, count_t(1) << 6);
    par.p.bailout.maximum_reference_iterations = par.p.bailout.iterations;
    restring_vals(par);
    restart = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("+##IterationsUp"))
  {
    STOP
    par.p.bailout.iterations <<= 1;
    par.p.bailout.iterations = std::min(par.p.bailout.iterations, count_t(1) << 60);
    if (window_state.iterations_limited)
    {
      par.p.bailout.iterations = std::min(par.p.bailout.iterations, window_state.iterations_limit);
    }
    par.p.bailout.maximum_reference_iterations = par.p.bailout.iterations;
    restring_vals(par);
    restart = true;
  }
  ImGui::SameLine();
  ImGui::PushItemWidth(-FLT_MIN);
  if (ImGui::InputText("##Iterations", &par.s_iterations, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CharsDecimal))
  {
    try
    {
      count_t tmp = std::stoll(par.s_iterations);
      if (tmp > 0)
      {
        STOP
        par.p.bailout.iterations = tmp;
        par.p.bailout.iterations = std::min(par.p.bailout.iterations, count_t(1) << 60);
        if (window_state.iterations_limited)
        {
          par.p.bailout.iterations = std::min(par.p.bailout.iterations, window_state.iterations_limit);
        }
        par.p.bailout.maximum_reference_iterations = par.p.bailout.iterations;
        restring_vals(par);
        restart = true;
      }
      else
      {
        restring_vals(par);
      }
    }
    catch (std::invalid_argument &e)
    {
      restring_vals(par);
    }
    catch (std::out_of_range &e)
    {
      restring_vals(par);
    }
  }
  ImGui::PopItemWidth();
  ImGui::Text("Max Ptb Iters");
  ImGui::SameLine();
  if (ImGui::Button("-##MaxPtbItersDown"))
  {
    STOP
    par.p.bailout.maximum_perturb_iterations >>= 1;
    par.p.bailout.maximum_perturb_iterations = std::max(par.p.bailout.maximum_perturb_iterations, count_t(1) << 6);
    restring_vals(par);
    restart = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("+##MaxPtbItersUp"))
  {
    STOP
    par.p.bailout.maximum_perturb_iterations <<= 1;
    par.p.bailout.maximum_perturb_iterations = std::min(par.p.bailout.maximum_perturb_iterations, count_t(1) << 60);
    if (window_state.iterations_limited)
    {
      par.p.bailout.maximum_perturb_iterations = std::min(par.p.bailout.maximum_perturb_iterations, window_state.perturb_iterations_limit);
    }
    restring_vals(par);
    restart = true;
  }
  ImGui::SameLine();
  ImGui::PushItemWidth(-FLT_MIN);
  if (ImGui::InputText("##MaxPtbIters", &par.s_maximum_perturb_iterations, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CharsDecimal))
  {
    try
    {
      count_t tmp = std::stoll(par.s_maximum_perturb_iterations);
      if (tmp > 0)
      {
        STOP
        par.p.bailout.maximum_perturb_iterations = tmp;
        par.p.bailout.maximum_perturb_iterations = std::min(par.p.bailout.maximum_perturb_iterations, count_t(1) << 60);
        if (window_state.iterations_limited)
        {
          par.p.bailout.maximum_perturb_iterations = std::min(par.p.bailout.maximum_perturb_iterations, window_state.perturb_iterations_limit);
        }
        restring_vals(par);
        restart = true;
      }
      else
      {
        restring_vals(par);
      }
    }
    catch (std::invalid_argument &e)
    {
      restring_vals(par);
    }
    catch (std::out_of_range &e)
    {
      restring_vals(par);
    }
  }
  ImGui::PopItemWidth();
  ImGui::Text("Max BLA Steps");
  ImGui::SameLine();
  if (ImGui::Button("-##MaxBLAStepsDown"))
  {
    STOP
    par.p.bailout.maximum_bla_steps >>= 1;
    par.p.bailout.maximum_bla_steps = std::max(par.p.bailout.maximum_bla_steps, count_t(1) << 6);
    restring_vals(par);
    restart = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("+##MaxBLAStepsUp"))
  {
    STOP
    par.p.bailout.maximum_bla_steps <<= 1;
    par.p.bailout.maximum_bla_steps = std::min(par.p.bailout.maximum_bla_steps, count_t(1) << 60);
    if (window_state.iterations_limited)
    {
      par.p.bailout.maximum_bla_steps = std::min(par.p.bailout.maximum_bla_steps, window_state.bla_steps_limit);
    }
    restring_vals(par);
    restart = true;
  }
  ImGui::SameLine();
  ImGui::PushItemWidth(-FLT_MIN);
  if (ImGui::InputText("##MaxBLASteps", &par.s_maximum_bla_steps, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CharsDecimal))
  {
    try
    {
      count_t tmp = std::stoll(par.s_maximum_bla_steps);
      if (tmp > 0)
      {
        STOP
        par.p.bailout.maximum_bla_steps = tmp;
        par.p.bailout.maximum_bla_steps = std::min(par.p.bailout.maximum_bla_steps, count_t(1) << 60);
        if (window_state.iterations_limited)
        {
          par.p.bailout.maximum_bla_steps = std::min(par.p.bailout.maximum_bla_steps, window_state.bla_steps_limit);
        }
        restring_vals(par);
        restart = true;
      }
      else
      {
        restring_vals(par);
      }
    }
    catch (std::invalid_argument &e)
    {
      restring_vals(par);
    }
    catch (std::out_of_range &e)
    {
      restring_vals(par);
    }
  }
  ImGui::PopItemWidth();

  ImGui::Text("Escape Radius");
  ImGui::SameLine();
  if (ImGui::Button("-##EscapeRadiusDown"))
  {
    STOP
    par.p.bailout.escape_radius /= 2;
    clamp_escape_radius(par);
    restart = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("+##EscapeRadiusUp"))
  {
    STOP
    par.p.bailout.escape_radius *= 2;
    clamp_escape_radius(par);
    restart = true;
  }
  ImGui::SameLine();
  ImGui::PushItemWidth(-FLT_MIN);
  float escape_radius = par.p.bailout.escape_radius;
  if (ImGui::InputFloat("##EscapeRadius", &escape_radius))
  {
    STOP
    par.p.bailout.escape_radius = escape_radius;
    clamp_escape_radius(par);
    restart = true;
  }
  ImGui::PopItemWidth();

  ImGui::Text("Inscape Radius");
  ImGui::SameLine();
  if (ImGui::Button("-##InscapeRadiusDown"))
  {
    STOP
    par.p.bailout.inscape_radius /= 2;
    par.p.bailout.inscape_radius = std::max(par.p.bailout.inscape_radius, 1.0 / (1 << 20));
    restring_vals(par);
    restart = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("+##InscapeRadiusUp"))
  {
    STOP
    par.p.bailout.inscape_radius *= 2;
    par.p.bailout.inscape_radius = std::min(par.p.bailout.inscape_radius, 1.0 / (1 << 0));
    restring_vals(par);
    restart = true;
  }
  ImGui::SameLine();
  ImGui::PushItemWidth(-FLT_MIN);
  if (ImGui::InputText("##InscapeRadius", &par.s_inscape_radius, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CharsScientific))
  {
    try
    {
      double tmp = std::stod(par.s_inscape_radius);
      if (tmp <= 1)
      {
        STOP
        par.p.bailout.inscape_radius = tmp;
        restring_vals(par);
        restart = true;
      }
      else
      {
        restring_vals(par);
      }
    }
    catch (std::invalid_argument &e)
    {
      restring_vals(par);
    }
    catch (std::out_of_range &e)
    {
      restring_vals(par);
    }
  }
  ImGui::PopItemWidth();
  ImGui::End();
}

void display_transform_window(bool *open)
{
  WINDOW("Transform", transform)
  bool reflect = par.p.transform.reflect;
  if (ImGui::Checkbox("Reflect", &reflect))
  {
    STOP
    mat2<double> T (1, 0, 0, reflect != par.p.transform.reflect ? -1 : 1);
    par.transform = par.transform * T;
    unstring_vals(par);
    restart = true;
  }
  {
    float rotate = par.p.transform.rotate;
    bool changed = false;
    if (ImGui::Button("-##RotateDown"))
    {
      rotate -= 5;
      if (rotate < -360) rotate += 720;
      changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("0##RotateZero"))
    {
      rotate = 0;
      changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("+##RotateUp"))
    {
      rotate += 5;
      if (rotate > 360) rotate -= 720;
      changed = true;
    }
    ImGui::SameLine();
    ImGui::PushItemWidth(12 * ImGui::GetFontSize());
    if (ImGui::SliderFloat("Rotate", &rotate, -360.f, 360.f, "%.2f") || changed)
    {
      STOP
      float a = (rotate - par.p.transform.rotate) * 2.f * 3.14159265358979f / 360.f;
      mat2<double> T (cos(a), -sin(a), sin(a), cos(a));
      par.transform = par.transform * T;
      unstring_vals(par);
      restart = true;
    }
    ImGui::PopItemWidth();
  }
  ImGui::BeginDisabled(subframes_rendered == 0 || ! raw);
  if (ImGui::Button("Auto Stretch (DE)"))
  {
    if (subframes_rendered > 0 && raw)
    {
      STOP
      mat2<double> T = unskew_de(*raw);
      double d = determinant(T);
      if (! (std::isnan(d) || std::isinf(d) || d == 0))
      {
        par.transform = par.transform * T;
        unstring_vals(par);
        restart = true;
      }
    }
  }
  ImGui::EndDisabled();
  {
    float stretch_amount = par.p.transform.stretch_amount;
    bool changed = false;
    if (ImGui::Button("-##AmountDown"))
    {
      stretch_amount -= 5;
      if (stretch_amount < -1000) stretch_amount = -1000;
      changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("0##AmountZero"))
    {
      stretch_amount = 0;
      changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("+##AmountUp"))
    {
      stretch_amount += 5;
      if (stretch_amount > 1000) stretch_amount = 1000;
      changed = true;
    }
    ImGui::SameLine();
    ImGui::PushItemWidth(12 * ImGui::GetFontSize());
    if (ImGui::SliderFloat("Stretch Amount", &stretch_amount, -1000.f, 1000.f, "%.2f") || changed) // FIXME
    {
      STOP
      par.p.transform.stretch_amount = stretch_amount;
      restring_vals(par);
      restart = true;
    }
    ImGui::PopItemWidth();
  }
  {
    float stretch_angle = par.p.transform.stretch_angle;
    bool changed = false;
    if (ImGui::Button("-##AngleDown"))
    {
      stretch_angle -= 5;
      if (stretch_angle < -360) stretch_angle += 720;
      changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("0##AngleZero"))
    {
      stretch_angle = 0;
      changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("+##AngleUp"))
    {
      stretch_angle += 5;
      if (stretch_angle > 360) stretch_angle -= 720;
      changed = true;
    }
    ImGui::SameLine();
    ImGui::PushItemWidth(12 * ImGui::GetFontSize());
    if (ImGui::SliderFloat("Stretch Angle", &stretch_angle, -360.f, 360.f, "%.2f") || changed)
    {
      STOP
      par.p.transform.stretch_angle = stretch_angle;
      restring_vals(par);
      restart = true;
    }
    ImGui::PopItemWidth();
  }
  bool exponential_map = par.p.transform.exponential_map;
  if (ImGui::Checkbox("Exponential Map", &exponential_map))
  {
    STOP
    par.p.transform.exponential_map = exponential_map;
    restring_vals(par);
    restart = true;
  }
  if (ImGui::Checkbox("Vertical Flip", &par.p.transform.vertical_flip))
  {
    // nop
  }
  ImGui::End();
}

void display_algorithm_window(bool *open)
{
  WINDOW("Algorithm", algorithm)
  bool lock_maximum_reference_iterations_to_period = par.p.algorithm.lock_maximum_reference_iterations_to_period;
  if (ImGui::Checkbox("Lock Max Ref Iters to Period", &lock_maximum_reference_iterations_to_period))
  {
    STOP
    par.p.algorithm.lock_maximum_reference_iterations_to_period = lock_maximum_reference_iterations_to_period;
    restart = true;
  }
  bool reuse_reference = par.p.algorithm.reuse_reference;
  if (ImGui::Checkbox("Reuse Reference", &reuse_reference))
  {
    STOP
    par.p.algorithm.reuse_reference = reuse_reference;
    restart = true;
  }
  bool reuse_bilinear_approximation = par.p.algorithm.reuse_bilinear_approximation;
  if (ImGui::Checkbox("Reuse Bilinear Approximation", &reuse_bilinear_approximation))
  {
    STOP
    par.p.algorithm.reuse_bilinear_approximation = reuse_bilinear_approximation;
    restart = true;
  }
  int bla_skip_levels = par.p.algorithm.bla_skip_levels;
  if (ImGui::InputInt("BLA Skip Levels", &bla_skip_levels))
  {
    STOP
    par.p.algorithm.bla_skip_levels = std::min(std::max(bla_skip_levels, 0), 64);
    restart = true;
  }
  ImGui::Text("Tile Size");
  ImGui::PushItemWidth(6 * ImGui::GetFontSize());
  ImGui::SameLine();
  if (tile_width == 0) tile_width = par.p.opencl.tile_width;
  if (tile_height == 0) tile_height = par.p.opencl.tile_height;
  ImGui::InputInt("x", &tile_width);
  ImGui::SameLine();
  ImGui::InputInt("##y", &tile_height);
  ImGui::PopItemWidth();
  if (ImGui::Button("Apply"))
  {
    STOP
    par.p.opencl.tile_width = std::min(std::max(16, tile_width), rgb ? (int) rgb->width : 1024);
    par.p.opencl.tile_height = std::min(std::max(16, tile_height), rgb ? (int) rgb->height : 576);
    restart = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("Cancel"))
  {
    tile_width = par.p.opencl.tile_width;
    tile_height = par.p.opencl.tile_height;
  }
  ImGui::End();
}

histogram3 hist_pre = {{ { 0, 0, false, 0, { }, false }, { 0, 0, false, 0, { }, false }, { 0, 0, false, 0, { }, false } }};
bool hist_pre_log = true;
histogram3 hist_post = {{ { 0, 0, false, 0, { }, false }, { 0, 0, false, 0, { }, false }, { 0, 0, false, 0, { }, false } }};
bool hist_post_log = true;

void display_postprocessing_window(bool *open)
{
  WINDOW("Postprocessing", postprocessing)

  ImGui::Checkbox("Pre##PreLog", &hist_pre_log);
  if (hist_pre_log)
  {
    histogram3_log2(hist_pre);
  }
  else
  {
    histogram3_exp2(hist_pre);
  }
  ImGui::PlotHistogram("R##PreR", &hist_pre.h[0].data[0], hist_pre.h[0].data.size());
  ImGui::PlotHistogram("G##PreG", &hist_pre.h[1].data[0], hist_pre.h[1].data.size());
  ImGui::PlotHistogram("B##PreB", &hist_pre.h[2].data[0], hist_pre.h[2].data.size());

  ImGui::Checkbox("Post##PostLog", &hist_post_log);
  if (hist_post_log)
  {
    histogram3_log2(hist_post);
  }
  else
  {
    histogram3_exp2(hist_post);
  }
  ImGui::PlotHistogram("R##PostR", &hist_post.h[0].data[0], hist_post.h[0].data.size());
  ImGui::PlotHistogram("G##PostG", &hist_post.h[1].data[0], hist_post.h[1].data.size());
  ImGui::PlotHistogram("B##PostB", &hist_post.h[2].data[0], hist_post.h[2].data.size());

  {
    float brightness = par.p.postprocessing.brightness;
    bool changed = false;
    if (ImGui::Button("-##BrightnessDown"))
    {
      brightness -= 1;
      changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("0##BrightnessZero"))
    {
      brightness = 0;
      changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("+##BrightnessUp"))
    {
      brightness += 1;
      changed = true;
    }
    ImGui::SameLine();
    ImGui::PushItemWidth(12 * ImGui::GetFontSize());
    if (ImGui::SliderFloat("Brightness", &brightness, -16.0f, 16.f, "%.2f") || changed)
    {
      par.p.postprocessing.brightness = brightness;
      needs_dopost = true;
    }
    ImGui::PopItemWidth();
  }

  {
    float contrast = par.p.postprocessing.contrast;
    bool changed = false;
    if (ImGui::Button("-##ContrastDown"))
    {
      contrast -= 1;
      changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("0##ContrastZero"))
    {
      contrast = 0;
      changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("+##ContrastUp"))
    {
      contrast += 1;
      changed = true;
    }
    ImGui::SameLine();
    ImGui::PushItemWidth(12 * ImGui::GetFontSize());
    if (ImGui::SliderFloat("Contrast", &contrast, -16.f, 16.f, "%.2f") || changed)
    {
      par.p.postprocessing.contrast = contrast;
      needs_dopost = true;
    }
    ImGui::PopItemWidth();
  }

  {
    float gamma = par.p.postprocessing.gamma;
    bool changed = false;
    if (ImGui::Button("-##GammaDown"))
    {
      gamma -= 1;
      changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("1##GammaOne"))
    {
      gamma = 1;
      changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("+##GammaUp"))
    {
      gamma += 1;
      changed = true;
    }
    ImGui::SameLine();
    ImGui::PushItemWidth(12 * ImGui::GetFontSize());
    if (ImGui::SliderFloat("Gamma", &gamma, 0.0f, 16.f, "%.2f") || changed)
    {
      par.p.postprocessing.gamma = gamma;
      needs_dopost = true;
    }
    ImGui::PopItemWidth();
  }

  {
    float exposure = par.p.postprocessing.exposure;
    bool changed = false;
    if (ImGui::Button("-##ExposureDown"))
    {
      exposure -= 1;
      changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("0##ExposureZero"))
    {
      exposure = 0;
      changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("+##ExposureUp"))
    {
      exposure += 1;
      changed = true;
    }
    ImGui::SameLine();
    ImGui::PushItemWidth(12 * ImGui::GetFontSize());
    if (ImGui::SliderFloat("Exposure", &exposure, -16.f, 16.f, "%.2f") || changed)
    {
      par.p.postprocessing.exposure = exposure;
      needs_dopost = true;
    }
    ImGui::PopItemWidth();
  }

  ImGui::End();
}

void display_colours_window(bool *open)
{
  WINDOW("Colours", colours)
  bool modified = false;
  float background[3] = { float(par.p.colour.background.r), float(par.p.colour.background.g), float(par.p.colour.background.b) };
  if (ImGui::ColorEdit3("Background", &background[0]))
  {
    par.p.colour.background.r = background[0];
    par.p.colour.background.g = background[1];
    par.p.colour.background.b = background[2];
    // don't need to recolour
  }
  if (ImGui::Checkbox("Shader Uses Histogram", &par.p.colour.uses_histogram))
  {
    modified |= par.p.colour.uses_histogram;
  }
  modified |= colour_display(clr, *open);
  ImGui::End();
  if (modified)
  {
    recolour = true;
  }
}

histogramcdf cdf_n = { 0, 0, 0, { } };

histogram2d hist_de = { 1, 1, 0.0f, { 0.0f }, 0.0f, false };
histogram hist_n = { 0, 0, false, 0, { }, false };
histogram hist_bla = { 0, 0, false, 0, { }, false };
histogram hist_ptb = { 0, 0, false, 0, { }, false };
bool hist_de_log = false;
bool hist_n_log = true;
bool hist_bla_log = true;
bool hist_ptb_log = true;

void display_information_window(bool *open)
{
  WINDOW("Information", information)
  ImGui::Checkbox("##IterationsLog", &hist_n_log);
  if (hist_n_log)
  {
    histogram_log2(hist_n);
  }
  else
  {
    histogram_exp2(hist_n);
  }
  ImGui::SameLine();
  ImGui::PlotHistogram("Iterations", &hist_n.data[0], hist_n.data.size());
  ImGui::Checkbox("##BLAStepsLog", &hist_bla_log);
  if (hist_bla_log)
  {
    histogram_log2(hist_bla);
  }
  else
  {
    histogram_exp2(hist_bla);
  }
  ImGui::SameLine();
  ImGui::PlotHistogram("BLA steps", &hist_bla.data[0], hist_bla.data.size());
  ImGui::Checkbox("##PTBItersLog", &hist_ptb_log);
  if (hist_ptb_log)
  {
    histogram_log2(hist_ptb);
  }
  else
  {
    histogram_exp2(hist_ptb);
  }
  ImGui::SameLine();
  ImGui::PlotHistogram("PTB iters", &hist_ptb.data[0], hist_ptb.data.size());

  if (hist_de.data.size() > 0)
  {
    ImGui::Checkbox("##DistanceLog", &hist_de_log);
    if (hist_de_log)
    {
      histogram2d_log2(hist_de);
    }
    else
    {
      histogram2d_exp2(hist_de);
    }
    ImGui::SameLine();
    const ImPlotHeatmapFlags hm_flags = 0;//ImPlotHeatmapFlags_ColMajor;
    const ImPlotAxisFlags axes_flags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks | ImPlotAxisFlags_NoTickLabels;
    ImGui::BeginChild("##DistanceHM");
    ImPlot::PushColormap(ImPlotColormap_Viridis);
    if (ImPlot::BeginPlot("Distance",ImVec2(32 * 8, 18 * 8), ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText))
    {
      ImPlot::SetupAxes(nullptr, nullptr, axes_flags, axes_flags);
      ImPlot::PlotHeatmap("##DistanceHMPlot", &hist_de.data[0], hist_de.height, hist_de.width, 0.0f, hist_de.peak, "", ImPlotPoint(0,0), ImPlotPoint(1,1), hm_flags);
      ImPlot::EndPlot();
    }
    ImGui::SameLine();
    ImPlot::ColormapScale("##HeatScale", 0.0f, hist_de.peak, ImVec2(32, 18 * 8), "");
    ImPlot::PopColormap();
    ImGui::EndChild();
  }

#if 0
  double count = sta.iiters.s0 + sta.uiters.s0 + sta.iters.s0;
  ImGui::Text("Speedup       %.1fx", sta.iters.mean() / sta.steps.mean());
  ImGui::Text("Escaped       %.1f%%", 100.0 * sta.iters.s0 / count);
  ImGui::Text("Unscaped      %.1f%%", 100.0 * sta.uiters.s0 / count);
  ImGui::Text("Inscaped      %.1f%%", 100.0 * sta.iiters.s0 / count);
  ImGui::Text("Ex. Steps     %.1f (min %.1f, max %.1f, stddev %.1f)", sta.steps.mean(), sta.steps.mi, sta.steps.ma, sta.steps.stddev());
  ImGui::Text("Ex. Steps BLA %.1f (min %.1f, max %.1f, stddev %.1f)", sta.steps_bla.mean(), sta.steps_bla.mi, sta.steps_bla.ma, sta.steps_bla.stddev());
  ImGui::Text("Ex. Steps Ptb %.1f (min %.1f, max %.1f, stddev %.1f)", sta.steps_ptb.mean(), sta.steps_ptb.mi, sta.steps_ptb.ma, sta.steps_ptb.stddev());
  ImGui::Text("Ex. Iters     %.1f (min %.1f, max %.1f, stddev %.1f)", sta.iters.mean(), sta.iters.mi, sta.iters.ma, sta.iters.stddev());
  ImGui::Text("Ex. Iters BLA %.1f (min %.1f, max %.1f, stddev %.1f)", sta.iters_bla.mean(), sta.iters_bla.mi, sta.iters_bla.ma, sta.iters_bla.stddev());
  ImGui::Text("Ex. Iters Ptb %.1f (min %.1f, max %.1f, stddev %.1f)", sta.iters_ptb.mean(), sta.iters_ptb.mi, sta.iters_ptb.ma, sta.iters_ptb.stddev());
  ImGui::Text("Un. Steps     %.1f (min %.1f, max %.1f, stddev %.1f)", sta.usteps.mean(), sta.usteps.mi, sta.usteps.ma, sta.usteps.stddev());
  ImGui::Text("Un. Steps BLA %.1f (min %.1f, max %.1f, stddev %.1f)", sta.usteps_bla.mean(), sta.usteps_bla.mi, sta.usteps_bla.ma, sta.usteps_bla.stddev());
  ImGui::Text("Un. Steps Ptb %.1f (min %.1f, max %.1f, stddev %.1f)", sta.usteps_ptb.mean(), sta.usteps_ptb.mi, sta.usteps_ptb.ma, sta.usteps_ptb.stddev());
  ImGui::Text("Un. Iters     %.1f (min %.1f, max %.1f, stddev %.1f)", sta.uiters.mean(), sta.uiters.mi, sta.uiters.ma, sta.uiters.stddev());
  ImGui::Text("Un. Iters BLA %.1f (min %.1f, max %.1f, stddev %.1f)", sta.uiters_bla.mean(), sta.uiters_bla.mi, sta.uiters_bla.ma, sta.uiters_bla.stddev());
  ImGui::Text("Un. Iters Ptb %.1f (min %.1f, max %.1f, stddev %.1f)", sta.uiters_ptb.mean(), sta.uiters_ptb.mi, sta.uiters_ptb.ma, sta.uiters_ptb.stddev());
  ImGui::Text("In. Steps     %.1f (min %.1f, max %.1f, stddev %.1f)", sta.isteps.mean(), sta.isteps.mi, sta.isteps.ma, sta.isteps.stddev());
  ImGui::Text("In. Steps BLA %.1f (min %.1f, max %.1f, stddev %.1f)", sta.isteps_bla.mean(), sta.isteps_bla.mi, sta.isteps_bla.ma, sta.isteps_bla.stddev());
  ImGui::Text("In. Steps Ptb %.1f (min %.1f, max %.1f, stddev %.1f)", sta.isteps_ptb.mean(), sta.isteps_ptb.mi, sta.isteps_ptb.ma, sta.isteps_ptb.stddev());
  ImGui::Text("In. Iters     %.1f (min %.1f, max %.1f, stddev %.1f)", sta.iiters.mean(), sta.iiters.mi, sta.iiters.ma, sta.iiters.stddev());
  ImGui::Text("In. Iters BLA %.1f (min %.1f, max %.1f, stddev %.1f)", sta.iiters_bla.mean(), sta.iiters_bla.mi, sta.iiters_bla.ma, sta.iiters_bla.stddev());
  ImGui::Text("In. Iters Ptb %.1f (min %.1f, max %.1f, stddev %.1f)", sta.iiters_ptb.mean(), sta.iiters_ptb.mi, sta.iiters_ptb.ma, sta.iiters_ptb.stddev());
  ImGui::Text("Iters Ref Max %.1f (min %.1f, max %.1f, stddev %.1f)", sta.iters_ref.mean(), sta.iters_ref.mi, sta.iters_ref.ma, sta.iters_ref.stddev());
  ImGui::Text("Rebases       %.1f (min %.1f, max %.1f, stddev %.1f)", sta.rebases.mean(), sta.rebases.mi, sta.rebases.ma, sta.rebases.stddev());
  ImGui::Text("Rebases Small %.1f (min %.1f, max %.1f, stddev %.1f)", sta.rebases_small.mean(), sta.rebases_small.mi, sta.rebases_small.ma, sta.rebases_small.stddev());
  ImGui::Text("Rebases NoRef %.1f (min %.1f, max %.1f, stddev %.1f)", sta.rebases_noref.mean(), sta.rebases_noref.mi, sta.rebases_noref.ma, sta.rebases_noref.stddev());
#endif
  ImGui::End();
}

void display_quality_window(bool *open)
{
  WINDOW("Quality", quality)
  double megapixels =
    par.p.image.width * par.p.image.supersampling / (double) par.p.image.subsampling *
    par.p.image.height * par.p.image.supersampling / (double) par.p.image.subsampling /
    (1024.0 * 1024.0);
  ImGui::Text("%.2f Mpixels", megapixels);
  ImGui::PushItemWidth(8 * ImGui::GetFontSize());
  if (image_width == 0) image_width = par.p.image.width;
  if (image_height == 0) image_height = par.p.image.height;
  if (image_dpi == 0) image_dpi = par.p.image.dpi;
  if (! image_lock_aspect)
  {
    image_width_aspect_locked = image_width;
    image_height_aspect_locked = image_height;
  }
  if (ImGui::InputInt("x##xpixels", &image_width))
  {
    if (image_lock_aspect)
    {
      image_height = image_width * image_height_aspect_locked / image_width_aspect_locked;
    }
  }
  ImGui::SameLine();
  if (ImGui::InputInt("px##ypixels", &image_height))
  {
    if (image_lock_aspect)
    {
      image_width = image_height * image_width_aspect_locked / image_height_aspect_locked;
    }
  }
  float image_width_in = image_width / (float) image_dpi;
  float image_height_in = image_height / (float) image_dpi;
  if (ImGui::InputFloat("x##xinches", &image_width_in))
  {
    if (image_lock_dpi)
    {
      image_width = round(image_width_in * image_dpi);
    }
    else
    {
      image_dpi = round(image_width / image_width_in);
    }
    if (image_lock_aspect)
    {
      image_height = image_width * image_height_aspect_locked / image_width_aspect_locked;
    }
  }
  ImGui::SameLine();
  if (ImGui::InputFloat("in##yinches", &image_height_in))
  {
    if (image_lock_dpi)
    {
      image_height = round(image_height_in * image_dpi);
    }
    else
    {
      image_dpi = round(image_height / image_height_in);
    }
    if (image_lock_aspect)
    {
      image_width = image_height * image_width_aspect_locked / image_height_aspect_locked;
    }
  }
  float image_width_cm = image_width / (float) image_dpi * 2.54;
  float image_height_cm = image_height / (float) image_dpi * 2.54;
  if (ImGui::InputFloat("x##xcm", &image_width_cm))
  {
    if (image_lock_dpi)
    {
      image_width = round((image_width_cm  / 2.54) * image_dpi);
    }
    else
    {
      image_dpi = round(image_width / (image_width_cm / 2.54));
    }
    if (image_lock_aspect)
    {
      image_height = image_width * image_height_aspect_locked / image_width_aspect_locked;
    }
  }
  ImGui::SameLine();
  if (ImGui::InputFloat("cm##ycm", &image_height_cm))
  {
    if (image_lock_dpi)
    {
      image_height = round((image_height_cm / 2.54) * image_dpi);
    }
    else
    {
      image_dpi = round(image_height / (image_height_cm / 2.54));
    }
    if (image_lock_aspect)
    {
      image_width = image_height * image_width_aspect_locked / image_height_aspect_locked;
    }
  }
  if (ImGui::InputInt("dpi", &image_dpi))
  {
    if (image_lock_dpi)
    {
      image_width = round(image_width_in * image_dpi);
      image_height = round(image_height_in * image_dpi);
    }
  }
  ImGui::SameLine();
  ImGui::Checkbox("Lock##lockdpi", &image_lock_dpi);
  ImGui::Checkbox("Lock Aspect", &image_lock_aspect);
  ImGui::SameLine();
  ImGui::Text("%f", image_width_aspect_locked / (double) image_height_aspect_locked);
  image_dpi = std::min(std::max(image_dpi, 1), 65535); // upper limit from JPEG API (uint16)
  ImGui::PopItemWidth();
  if (ImGui::Button("Apply"))
  {
    STOP
    par.p.image.width = image_width;
    par.p.image.height = image_height;
    par.p.image.dpi = image_dpi;
    limit_image_size(par);
    resize(par.p.image.supersampling, par.p.image.subsampling);
    restart = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("Cancel"))
  {
    image_width = par.p.image.width;
    image_height = par.p.image.height;
    image_dpi = par.p.image.dpi;
  }
  int supersampling = par.p.image.supersampling;
  if (ImGui::InputInt("Super", &supersampling))
  {
    STOP
    par.p.image.supersampling = supersampling;
    limit_image_size(par);
    resize(par.p.image.supersampling, par.p.image.subsampling);
    restart = true;
  }
  int subsampling = par.p.image.subsampling;
  if (ImGui::InputInt("Sub", &subsampling))
  {
    STOP
    par.p.image.subsampling = subsampling;
    limit_image_size(par);
    resize(par.p.image.supersampling, par.p.image.subsampling);
    restart = true;
  }
  int subframes = par.p.image.subframes;
  if (ImGui::InputInt("Samples", &subframes))
  {
    if (subframes <= 0 || subframes > par.p.image.subframes)
    {
      continue_subframe_rendering = true;
    }
    par.p.image.subframes = std::min(std::max(subframes, 0), 65536); // FIXME
  }
  float cache_size = window_state.cache_size;
  if (ImGui::InputFloat("Cache Size (GiB)", &cache_size))
  {
    window_state.cache_size = cache_size;
    const int bytes_per_pixel = 11 * 4; // FIXME
    double bytes = window_state.cache_size * 1024 * 1024 * 1024;
    tile_cache_subframes = bytes / (bytes_per_pixel * rgb->width * rgb->height);
  }
  ImGui::Text("Cache Samples: %d", tile_cache_subframes);
  ImGui::End();
}

void start_batch(void)
{
  floatexp<float, int> Zoom = par.zoom;
  floatexp<float, int> ZoomedOut = 1 / 65536.0;
  count_t nframes = std::ceil(double(log(Zoom / ZoomedOut) / log(par.p.render.zoom_out_factor)));
  count_t start_frame = par.p.render.start_frame;
  count_t end_frame = par.p.render.frame_count == 0 ? nframes : par.p.render.start_frame + par.p.render.frame_count;
  nframes = end_frame - start_frame;
  if (nframes == 0)
  {
    nframes = 1;
  }
  end_frame = start_frame + nframes;
  if (! par.p.render.zoom_out_sequence)
  {
    start_frame = 0;
    end_frame = 1;
    nframes = 1;
  }
  batch_zoom = par.zoom;
  batch_frame = start_frame;
  batch_frame_end = end_frame;
  batch_running = true;
}

bool render_unlocked = false;
void display_render_window(bool *open)
{
  WINDOW("Render", render)
  std::string warning = "";
  if (par.p.render.save_exr && (par.exr_channels & ~ Channels_RGB) != 0 && par.p.image.subframes != 1)
  {
    warning = "raw EXR requires samples 1";
  }
  else if (par.p.render.save_exr && par.exr_channels == 0)
  {
    warning = "no EXR channels selected";
  }
  else if (par.p.render.save_exr && (par.exr_channels & Channels_RGB) != 0 && (par.exr_channels & ~ Channels_RGB) != 0)
  {
    warning = "EXR can be either RGB or raw, not both"; // FIXME
  }
  else if (par.p.render.save_exr && (par.exr_channels & Channels_RGB) != 0 && (par.exr_channels & Channels_RGB) != Channels_RGB)
  {
    warning = "non-raw EXR requires all three RGB channels"; // FIXME
  }
  else if (par.p.image.subframes <= 0)
  {
    warning = "samples is 0";
  }
  else if (! (par.p.render.save_toml || par.p.render.save_exr || par.p.render.save_png || par.p.render.save_jpg))
  {
    warning = "no save formats selected";
  }
  if (warning != "")
  {
    ImGui::TextUnformatted(warning.c_str());
  }
  else
  {
    ImGui::Checkbox("##unlock", &render_unlocked);
    ImGui::SameLine();
    if (ImGui::Button("Render"))
    {
      if (render_unlocked)
      {
        render_unlocked = false;
        STOP
        start_batch();
        restart = true;
      }
    }
#if ! (defined(__ANDROID__) && ! defined(__TERMUX__)) && ! defined(__EMSCRIPTEN__)
    ImGui::SameLine();
    ImGui::Checkbox("Quit When Done", &batch_quit_when_done);
#endif
  }
#if LOADSAVE_DIALOGS
  if (render_dialog)
  {
    if (ImGui::Button("Output Stem"))
    {
      render_dialog->Open();
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(par.p.render.filename.c_str());
  }
  else
#endif
  {
    ImGui::InputText("Output Stem", &par.p.render.filename);
  }
  ImGui::Checkbox("Zoom Out Sequence", &par.p.render.zoom_out_sequence);
  float zoom_out_factor = par.p.render.zoom_out_factor;
  ImGui::InputFloat("Zoom Out Factor", &zoom_out_factor);
  par.p.render.zoom_out_factor = zoom_out_factor;
  ImGui::InputInt("Numbering Offset", &par.p.render.numbering_offset);
  ImGui::InputInt("Start Frame", &par.p.render.start_frame);
  ImGui::InputInt("Frame Count", &par.p.render.frame_count);
  ImGui::InputInt("PRNG Seed", &par.p.render.prng_seed);
  ImGui::Text("Save Formats:");
  ImGui::Checkbox("TOML", &par.p.render.save_toml);
  ImGui::SameLine();
  ImGui::Checkbox("JPG", &par.p.render.save_jpg);
  ImGui::SameLine();
  ImGui::Checkbox("PNG", &par.p.render.save_png);
  ImGui::SameLine();
  ImGui::Checkbox("EXR", &par.p.render.save_exr);
  ImGui::Text("EXR Channels:");
  bool modified = false;
#define CHANNEL(C) \
  { \
    channel_mask_t m = channel_mask_t(1) << Channel_ ## C; \
    bool channel = (par.exr_channels & m) == m; \
    std::string name = std::string(#C "   ").substr(0, 3); \
    if (ImGui::Checkbox(name.c_str(), &channel)) \
    { \
      modified |= true; \
      if (channel) \
      { \
        par.exr_channels |= m; \
      } \
      else \
      { \
        par.exr_channels &= ~ m; \
      } \
    } \
  }
  CHANNEL(R)
  ImGui::SameLine();
  CHANNEL(G)
  ImGui::SameLine();
  CHANNEL(B)
  CHANNEL(N0)
  ImGui::SameLine();
  CHANNEL(N1)
  ImGui::SameLine();
  CHANNEL(NF)
  ImGui::SameLine();
  CHANNEL(T)
  CHANNEL(DEX)
  ImGui::SameLine();
  CHANNEL(DEY)
  ImGui::SameLine();
  CHANNEL(BLA)
  ImGui::SameLine();
  CHANNEL(PTB)
#undef CHANNEL
#if LOADSAVE_DIALOGS
  if (render_dialog)
  {
    render_dialog->Display();
    if (render_dialog->HasSelected())
    {
      par.p.render.filename = render_dialog->GetSelected().string();
      render_dialog->ClearSelected();
    }
  }
#endif

  if (modified)
  {
    unstring_vals(par);
  }
  ImGui::End();
}

wisdom benchmark_wisdom;
bool benchmark_ok = false;
bool start_benchmark = false;
volatile bool benchmark_running = false;
volatile bool benchmark_ended = false;
volatile progress_t benchmark_progress = 0;

void benchmark_thread(volatile progress_t *progress, volatile bool *running, volatile bool *ended)
{
  try
  {
    benchmark_wisdom = wisdom_benchmark(benchmark_wisdom, progress, running);
    benchmark_ok = *running;
  }
  catch (...)
  {
    benchmark_ok = false;
  }
  *ended = true;
}


extern std::string default_wisdom_path;

bool enumerate_unlocked = false;
bool benchmark_unlocked = false;

void display_wisdom_window(bool *open)
{
  WINDOW("Wisdom", wisdom)
  ImGui::Checkbox("##EnumerateUnlocked", &enumerate_unlocked);
  ImGui::SameLine();
  if (ImGui::Button("Enumerate") && enumerate_unlocked)
  {
    STOP
    enumerate_unlocked = false;
    wdom = wisdom_enumerate(true);
    restart = true;
  }
  ImGui::SameLine();
  ImGui::Checkbox("##BenchmarkUnlocked", &benchmark_unlocked);
  ImGui::SameLine();
  if (ImGui::Button("Benchmark") && benchmark_unlocked)
  {
    STOP
    benchmark_unlocked = false;
    start_benchmark = true;
  }
#if WISDOM_DIALOGS
  if (wisdom_load_dialog)
  {
    ImGui::SameLine();
    if (ImGui::Button("Load"))
    {
      wisdom_load_dialog->Open();
    }
  }
  if (wisdom_save_dialog)
  {
    ImGui::SameLine();
    if (ImGui::Button("Save"))
    {
      wisdom_save_dialog->Open();
    }
  }
#else
  ImGui::SameLine();
  ImGui::Checkbox("##LoadUnlocked", &wisdom_load_unlocked);
  ImGui::SameLine();
  if (ImGui::Button("Load") && wisdom_load_unlocked)
  {
    wisdom_load_unlocked = false;
    std::string filename = default_wisdom_path;
    try
    {
      STOP
      bool success = false;
      wdom = wisdom_load(filename, success);
      restart = true;
    }
    catch (std::exception &e)
    {
      SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "loading \"%s\": %s", filename.c_str(), e.what());
    }
  }
  ImGui::SameLine();
  ImGui::Checkbox("##SaveUnlocked", &wisdom_save_unlocked);
  ImGui::SameLine();
  if (ImGui::Button("Save") && wisdom_save_unlocked)
  {
    wisdom_save_unlocked = false;
    std::string filename = default_wisdom_path;
    bool ok = false;
    try
    {
      ok = wisdom_save(wdom, filename);
      syncfs();
    }
    catch (const std::exception &e)
    {
      SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "saving \"%s\": %s", filename.c_str(), e.what());
    }
    if (! ok)
    {
      display_error_string = std::string("could not save file ") + filename;
    }
  }
#endif
  int columns = 4;
  for (const auto & [ group, hardware ] : wdom.hardware)
  {
    columns += hardware.size();
  }
  bool changed = false;
  int id = 0;
  if (ImGui::BeginTable("Devices", columns))
  {
    ImGui::TableSetupColumn("Type");
    ImGui::TableSetupColumn("Mantissa");
    ImGui::TableSetupColumn("Exponent");
    ImGui::TableSetupColumn("Bytes");
    for (const auto & [ group, hardware ] : wdom.hardware)
    {
      for (const auto & device : hardware)
      {
        (void) device;
        ImGui::TableSetupColumn("Device");
      }
    }
    ImGui::TableHeadersRow();
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TableNextColumn();
    ImGui::TableNextColumn();
    ImGui::TableNextColumn();
    bool exit = false;
    for (auto & [ group, hardware ] : wdom.hardware)
    {
      for (auto dev = hardware.begin(); dev != hardware.end(); ++dev)
      {
        if (ImGui::TableNextColumn())
        {
          ImGui::PushID(++id);
          std::string groupname = group;
          if (ImGui::InputText("##Group", &groupname, ImGuiInputTextFlags_EnterReturnsTrue))
          {
            if (groupname != group)
            {
              if (wdom.hardware.count(groupname))
              {
                wdom.hardware[groupname].push_back(*dev);
              }
              else
              {
                wdom.hardware[groupname] = std::vector<whardware>{ *dev };
              }
              dev = hardware.erase(dev);
              if (! wdom.hardware[group].size())
              {
                wdom.hardware.erase(wdom.hardware.find(group));
              }
              changed |= true;
              exit = true;
            }
          }
          ImGui::PopID();
        }
        if (exit) break;
      }
      if (exit) break;
    }
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TableNextColumn();
    ImGui::TableNextColumn();
    ImGui::TableNextColumn();
    for (auto & [ group, hardware ] : wdom.hardware)
    {
      for (auto & [ name, platform, device, enabled ] : hardware)
      {
        if (ImGui::TableNextColumn())
        {
          ImGui::PushID(++id);
          if (ImGui::Checkbox("##Enabled", &enabled))
          {
            changed = true;
          }
          ImGui::SameLine();
          if (ImGui::InputText("##Name", &name, ImGuiInputTextFlags_EnterReturnsTrue))
          {
            changed |= true;
          }
          ImGui::PopID();
        }
      }
    }
    for (auto & [ tname, type ] : wdom.type)
    {
      auto & [ mantissa, exponent, bytes, devices ] = type;
      ImGui::TableNextRow();
      if (ImGui::TableNextColumn())
      {
        ImGui::TextUnformatted(tname.c_str());
      }
      if (ImGui::TableNextColumn())
      {
        ImGui::Text("%d", mantissa);
      }
      if (ImGui::TableNextColumn())
      {
        ImGui::Text("%d", exponent);
      }
      if (ImGui::TableNextColumn())
      {
        ImGui::Text("%d", bytes);
      }
      for (const auto & [ group, hardware ] : wdom.hardware)
      {
        for (const auto & [ name, platform, device, enabled ] : hardware)
        {
          if (ImGui::TableNextColumn())
          {
            for (auto & [ dplatform, ddevice, denabled, speed ] : devices)
            {
              if (platform == dplatform && device == ddevice)
              {
                ImGui::PushID(++id);
                bool in_use = false;
                if (std::string(nt_string[lookup.nt]) == tname)
                {
                  for (const auto & [ lplatform, ldevice, lenabled, lspeed ] : lookup.device)
                  {
                    in_use |= platform == lplatform && device == ldevice;
                    if (in_use) break;
                  }
                }
                auto & colors = ImGui::GetStyle().Colors;
                auto frame_bg = colors[ImGuiCol_FrameBg];
                auto checkmark = colors[ImGuiCol_CheckMark];
                if (in_use)
                {
                  colors[ImGuiCol_FrameBg] = colors[ImGuiCol_PlotHistogram];
                  colors[ImGuiCol_CheckMark] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
                }
                if (ImGui::Checkbox("##Enabled", &denabled))
                {
                  changed |= true;
                }
                if (in_use)
                {
                  colors[ImGuiCol_FrameBg] = frame_bg;
                  colors[ImGuiCol_CheckMark] = checkmark;
                }
                ImGui::SameLine();
                ImGui::Text("%.2f", std::max(0.0, std::log2(speed)));
                ImGui::PopID();
                break;
              }
            }
          }
        }
      }
    }
    ImGui::EndTable();
  }
  ImGui::End();
#if WISDOM_DIALOGS
  if (wisdom_load_dialog)
  {
    wisdom_load_dialog->Display();
    if (wisdom_load_dialog->HasSelected())
    {
      std::string filename = wisdom_load_dialog->GetSelected().string();
      try
      {
        STOP
        bool success = false;
        wdom = wisdom_load(filename, success);
        restart = true;
      }
      catch (std::exception &e)
      {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "loading \"%s\": %s", filename.c_str(), e.what());
      }
      wisdom_load_dialog->ClearSelected();
    }
  }
  if (wisdom_save_dialog)
  {
    wisdom_save_dialog->Display();
    if (wisdom_save_dialog->HasSelected())
    {
      std::string filename = wisdom_save_dialog->GetSelected().string();
      bool ok = false;
      try
      {
        ok = wisdom_save(wdom, filename);
        syncfs();
      }
      catch (const std::exception &e)
      {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "saving \"%s\": %s", filename.c_str(), e.what());
      }
      wisdom_save_dialog->ClearSelected();
      if (! ok)
      {
        display_error_string = "could not save file " + filename;
      }
    }
  }
#endif
  if (changed)
  {
    STOP
    restart = true;
  }
}

void display_preferences_window(bool *open)
{
  WINDOW("Preferences", preferences)
  float interface_scale = window_state.ui_scale;
  if (ImGui::InputFloat("UI Scale", &interface_scale))
  {
    window_state.ui_scale = std::min(std::max(interface_scale, 100.0f), 1000.0f);
    ImGui::GetIO().FontGlobalScale = window_state.ui_scale / 100.0f;
  }
  float megapixels = window_state.megapixels_limit;
  if (ImGui::InputFloat("Limit Megapixels", &megapixels))
  {
    window_state.megapixels_limit = std::min(std::max((double) megapixels, 0.1), (double) maximum_texture_size * maximum_texture_size);
  }
  ImGui::Checkbox("Limit Iterations", &window_state.iterations_limited);
  if (window_state.iterations_limited)
  {
    {
      ImGui::Text("Iterations");
      ImGui::SameLine();
      if (ImGui::Button("-##IterationsDown"))
      {
        window_state.iterations_limit >>= 1;
        window_state.iterations_limit = std::max(window_state.iterations_limit, count_t(1) << 6);
      }
      ImGui::SameLine();
      if (ImGui::Button("+##IterationsUp"))
      {
        window_state.iterations_limit <<= 1;
        window_state.iterations_limit = std::min(window_state.iterations_limit, count_t(1) << 60);
      }
      ImGui::SameLine();
      ImGui::PushItemWidth(-FLT_MIN);
      std::ostringstream s;
      s << window_state.iterations_limit;
      std::string s_iterations = s.str();
      if (ImGui::InputText("##Iterations", &s_iterations, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CharsDecimal))
      {
        try
        {
          count_t tmp = std::stoll(s_iterations);
          if (tmp > 0)
          {
            window_state.iterations_limit = tmp;
            window_state.iterations_limit = std::max(window_state.iterations_limit, count_t(1) << 6);
            window_state.iterations_limit = std::min(window_state.iterations_limit, count_t(1) << 60);
          }
        }
        catch (...)
        {
          // FIXME ignored
        }
      }
      ImGui::PopItemWidth();
    }
    {
      ImGui::Text("Ptb Iters ");
      ImGui::SameLine();
      if (ImGui::Button("-##PerturbDown"))
      {
        window_state.perturb_iterations_limit >>= 1;
        window_state.perturb_iterations_limit = std::max(window_state.perturb_iterations_limit, count_t(1) << 6);
      }
      ImGui::SameLine();
      if (ImGui::Button("+##PerturbUp"))
      {
        window_state.perturb_iterations_limit <<= 1;
        window_state.perturb_iterations_limit = std::min(window_state.perturb_iterations_limit, count_t(1) << 60);
      }
      ImGui::SameLine();
      ImGui::PushItemWidth(-FLT_MIN);
      std::ostringstream s;
      s << window_state.perturb_iterations_limit;
      std::string s_iterations = s.str();
      if (ImGui::InputText("##Perturb", &s_iterations, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CharsDecimal))
      {
        try
        {
          count_t tmp = std::stoll(s_iterations);
          if (tmp > 0)
          {
            window_state.perturb_iterations_limit = tmp;
            window_state.perturb_iterations_limit = std::max(window_state.perturb_iterations_limit, count_t(1) << 6);
            window_state.perturb_iterations_limit = std::min(window_state.perturb_iterations_limit, count_t(1) << 60);
          }
        }
        catch (...)
        {
          // FIXME ignored
        }
      }
      ImGui::PopItemWidth();
    }
    {
      ImGui::Text("BLA Steps ");
      ImGui::SameLine();
      if (ImGui::Button("-##BLADown"))
      {
        window_state.bla_steps_limit >>= 1;
        window_state.bla_steps_limit = std::max(window_state.bla_steps_limit, count_t(1) << 6);
      }
      ImGui::SameLine();
      if (ImGui::Button("+##BLAUp"))
      {
        window_state.bla_steps_limit <<= 1;
        window_state.bla_steps_limit = std::min(window_state.bla_steps_limit, count_t(1) << 60);
      }
      ImGui::SameLine();
      ImGui::PushItemWidth(-FLT_MIN);
      std::ostringstream s;
      s << window_state.bla_steps_limit;
      std::string s_iterations = s.str();
      if (ImGui::InputText("##BLA", &s_iterations, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CharsDecimal))
      {
        try
        {
          count_t tmp = std::stoll(s_iterations);
          if (tmp > 0)
          {
            window_state.bla_steps_limit = tmp;
            window_state.bla_steps_limit = std::max(window_state.bla_steps_limit, count_t(1) << 6);
            window_state.bla_steps_limit = std::min(window_state.bla_steps_limit, count_t(1) << 60);
          }
        }
        catch (...)
        {
          // FIXME ignored
        }
      }
      ImGui::PopItemWidth();
    }
  }
  ImGui::End();
}

bool newton_zoom_enabled = false;
bool newton_zoom_keep_enabled = false;
int newton_action = 0;
int newton_zoom_mode = 0;

bool newton_zoom_auto_capture = false;
int newton_relative_preset = 1;
floatexp<float, int> newton_relative_start = 1;
std::string newton_relative_start_str = "1";
float newton_relative_fold = 0.5;

bool newton_power_preset_custom = false;
bool newton_factor_preset_custom = false;

int newton_absolute_mini_preset = 1;
float newton_absolute_mini_power = 0.5;

int newton_absolute_domain_preset = 1;
float newton_absolute_domain_power = 1.0;

int newton_size_factor_preset = 2;
float newton_size_factor = 4;

void display_newton_window(bool *open)
{
  WINDOW("Newton Zooming", newton)
  ImGui::Checkbox("Activate", &newton_zoom_enabled);
  if (newton_zoom_enabled)
  {
    ImGui::SameLine();
    ImGui::Checkbox("Keep Activated", &newton_zoom_keep_enabled);
  }
  mouse_action = newton_zoom_enabled ? 1 : 0;
  ImGui::Combo("Action", &par.p.newton.action, "Period\0" "Center\0" "Zoom\0" "Transform\0");
  int absolute = par.p.newton.absolute;
  ImGui::Combo("Zoom Mode", &absolute, "Relative\0" "Absolute\0");
  par.p.newton.absolute = absolute;
  if (! par.p.newton.absolute)
  {
      if (InputFloatExp("Relative Start", &newton_relative_start, &newton_relative_start_str))
      {
        std::ostringstream s;
        s << newton_relative_start;
        newton_relative_start_str = s.str();
      }
      if (ImGui::Button("Capture"))
      {
        newton_relative_start = par.zoom;
        std::ostringstream s;
        s << newton_relative_start;
        newton_relative_start_str = s.str();
      }
      ImGui::SameLine();
      ImGui::Checkbox("Auto Capture", &newton_zoom_auto_capture);
  }

  int formula_power = par.degrees[0];
  // generate zoom presets for given formula power
  float power_presets[6];
  char power_item_array[6][64];
  const char* power_items[6];
  for (int i = 0; i < 6; ++i)
  {
    power_items[i] = &power_item_array[i][0];
  }
  power_presets[0] = par.p.newton.power;
  std::snprintf(power_item_array[0], sizeof(power_item_array[0]), "Custom");
  for (int i = 1; i < 5; ++i)
  {
    int fold = std::round(std::pow(formula_power, i));
    power_presets[i] = 1.0f - 1.0f / fold;
    std::snprintf(power_item_array[i], sizeof(power_item_array[i]), "%.4f (%dx)", power_presets[i], fold);
  }
  power_presets[5] = 1.0f;
  std::snprintf(power_item_array[5], sizeof(power_item_array[5]), "%.4f (Mini)", power_presets[5]);

  // look up current preset
  int power_preset = 0;
  if (! newton_power_preset_custom)
  {
    for (int i = 1; i < 6; ++i)
    {
      if (par.p.newton.power == power_presets[i])
      {
        power_preset = i;
        break;
      }
    }
  }

  // display power entry
  if (ImGui::Combo("Power", &power_preset, power_items, 6))
  {
    par.p.newton.power = power_presets[power_preset];
  }
  if (power_preset == 0)
  {
    newton_power_preset_custom = true;
    ImGui::SameLine();
    ImGui::InputFloat("##PowerCustom", &par.p.newton.power);
  }
  else
  {
    newton_power_preset_custom = false;
  }

#if 0
    case nr_mode_absolute_domain:
      if (ImGui::Combo("Absolute Power##DomainAbsolutePower", &newton_absolute_domain_preset, "Custom\0" "1.0 (Domain)\0" "1.125 (Morph)\0"))
      {
        switch (newton_absolute_domain_preset)
        {
          case 1: newton_absolute_domain_power = 1.0; break;
          case 2: newton_absolute_domain_power = 1.125; break;
        }
      }
      if (newton_absolute_domain_preset == 0)
      {
        ImGui::SameLine();
        ImGui::InputFloat("##DomainAbsolutePowerCustom", &newton_absolute_domain_power);
      }
      break;
#endif
  int factor_preset = 0;
  float factor_presets[6] = { par.p.newton.factor, 10.0f, 4.0f, 1.0f, 0.25f, 0.1f };
  if (! newton_factor_preset_custom)
  {
    for (int i = 1; i < 6; ++i)
    {
      if (par.p.newton.factor == factor_presets[i])
      {
        factor_preset = i;
        break;
      }
    }
  }
  if (ImGui::Combo("Factor", &factor_preset, "Custom\0" "10/1 (zoomed out)\0" "4/1\0" "1/1 (actual size)\0" "1/4\0" "1/10 (zoomed in)\0"))
  {
    par.p.newton.factor = factor_presets[factor_preset];
    switch (newton_size_factor_preset)
    {
      case 1: newton_size_factor = 10; break;
      case 2: newton_size_factor = 4; break;
      case 3: newton_size_factor = 1; break;
      case 4: newton_size_factor = 1./4; break;
      case 5: newton_size_factor = 1./10; break;
    }
  }
  if (factor_preset == 0)
  {
    newton_factor_preset_custom = true;
    ImGui::SameLine();
    ImGui::InputFloat("##FactorCustom", &par.p.newton.factor);
  }
  else
  {
    newton_factor_preset_custom = false;
  }
  ImGui::End();
}

void display_error_modal(bool &open)
{
  if (open)
  {
    ImGui::OpenPopup("Error");
  }
  ImVec2 center(ImGui::GetIO().DisplaySize.x * 0.5f, ImGui::GetIO().DisplaySize.y * 0.5f);
  ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (ImGui::BeginPopupModal("Error", NULL, ImGuiWindowFlags_AlwaysAutoResize))
  {
    ImGui::TextUnformatted(display_error_string.c_str());
    if (ImGui::Button("Ok"))
    {
      display_error_string = "";
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

bool newton_really_stop = false;
void display_newton_modal(bool &open)
{
  if (open)
  {
    ImGui::OpenPopup("Newton Zooming Progress");
  }
  ImVec2 center(ImGui::GetIO().DisplaySize.x * 0.5f, ImGui::GetIO().DisplaySize.y * 0.5f);
  ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (ImGui::BeginPopupModal("Newton Zooming Progress", NULL, ImGuiWindowFlags_AlwaysAutoResize))
  {
    if (par.p.newton.action >= newton_action_period)
    {
      char str[30];
      std::snprintf(str, 30, "Period: %d%%", int(100 * newton_progress[0]));
      ImGui::ProgressBar(float(newton_progress[0]), ImVec2(200,0), &str[0]);
    }
    if (par.p.newton.action >= newton_action_center)
    {
      char str[30];
      std::snprintf(str, 30, "Steps: %d%%", int(100 * newton_progress[1]));
      ImGui::ProgressBar(float(newton_progress[1]), ImVec2(200,0), &str[0]);
      std::snprintf(str, 30, "Center: %d%%", int(100 * newton_progress[2]));
      ImGui::ProgressBar(float(newton_progress[2]), ImVec2(200,0), &str[0]);
    }
    if (par.p.newton.action >= newton_action_zoom)
    {
      char str[30];
      std::snprintf(str, 30, "Size: %d%%", int(100 * newton_progress[3]));
      ImGui::ProgressBar(float(newton_progress[3]), ImVec2(200,0), &str[0]);
    }
    ImGui::Checkbox("##ReallyStop", &newton_really_stop);
    ImGui::SameLine();
    if (ImGui::Button("Stop") && newton_really_stop)
    {
      newton_running = false;
    }
    if (newton_ended)
    {
      open = false;
      if (! newton_zoom_keep_enabled)
      {
        newton_zoom_enabled = false;
      }
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

bool benchmarking_really_stop = false;
void display_benchmarking_modal(bool &open)
{
  if (open)
  {
    ImGui::OpenPopup("Benchmarking Wisdom");
  }
  ImVec2 center(ImGui::GetIO().DisplaySize.x * 0.5f, ImGui::GetIO().DisplaySize.y * 0.5f);
  ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (ImGui::BeginPopupModal("Benchmarking Wisdom", NULL, ImGuiWindowFlags_AlwaysAutoResize))
  {
    ImGui::Text("Wisdom calculation in progress.");
    ImGui::Text("This may take a few minutes.");
    char percent[20];
    std::snprintf(percent, sizeof(percent), "%3d%%", (int)(benchmark_progress * 100));
    ImGui::ProgressBar(benchmark_progress, ImVec2(-1.f, 0.f), percent);
    ImGui::Checkbox("##ReallyStop", &benchmarking_really_stop);
    ImGui::SameLine();
    if (ImGui::Button("Stop") && benchmarking_really_stop)
    {
      benchmark_running = false;
    }
    if (benchmark_ended)
    {
      open = false;
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

void display_batch_modal(bool &open)
{
  if (open)
  {
    ImGui::OpenPopup("Batch Render");
  }
  ImVec2 center(ImGui::GetIO().DisplaySize.x * 0.5f, ImGui::GetIO().DisplaySize.y * 0.5f);
  ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (ImGui::BeginPopupModal("Batch Render", NULL, ImGuiWindowFlags_AlwaysAutoResize))
  {
    ImGui::Text("Batch rendering in progress.");
    double batch_progress = (batch_frame + 1.0) / (batch_frame_end - batch_frame_start);
    char percent[20];
    std::snprintf(percent, sizeof(percent), "%3d%%", (int)(batch_progress * 100));
    ImGui::ProgressBar(batch_progress, ImVec2(-1.f, 0.f), percent);
    ImGui::Checkbox("##ReallyStop", &batch_really_stop);
    ImGui::SameLine();
    if (ImGui::Button("Stop") && batch_really_stop)
    {
      batch_quit_when_done = false;
      STOP
      batch_ended = true;
    }
    if (batch_ended)
    {
      // control flow must pass through here when batch mode finishes
      // otherwise the modal popup stays open with bad consequences
      batch_ended = false;
      batch_running = false;
      quit = batch_quit_when_done;
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

std::string version_text = "";
std::string license_text = "";

void display_about_window(bool *open)
{
  if (version_text == "")
  {
    std::ostringstream s;
    s << version(gl_version)
      << "GL_EXT_color_buffer_float " << (GLAD_GL_EXT_color_buffer_float ? "found" : "not found") << "\n"
      << "GL_MAX_TEXTURE_SIZE = " << maximum_texture_size << "\n";
    version_text = s.str();
    license_text = "\n\n\n" + license();
  }
  WINDOW("About", about)
  ImGui::TextUnformatted(version_text.c_str());
  const auto &io = ImGui::GetIO();
  ImGui::Text("average frame time %.2fms (%.2ffps)\n", (1000.0f / io.Framerate), io.Framerate);
  ImGui::TextUnformatted(license_text.c_str());
  ImGui::End();
}

void display_gui(SDL_Window *window, display_gles &dsp
#if 0
  , stats &sta
#endif
  , bool newton_modal = false
  , bool benchmarking_modal = false)
{
  int win_screen_width = 0;
  int win_screen_height = 0;
  int win_pixel_width = 0;
  int win_pixel_height = 0;
  SDL_GetWindowSize(window, &win_screen_width, &win_screen_height);
  SDL_GL_GetDrawableSize(window, &win_pixel_width, &win_pixel_height);
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplSDL2_NewFrame();
  ImGui::NewFrame();
  bool error_modal = display_error_string != "";
  if (show_windows && ! benchmarking_modal && ! error_modal)
  {
    display_window_window();
#define W(w) \
    if (fullscreen ? window_state.fullscreen.w.show : window_state.windowed.w.show) \
    { \
      display_ ## w ## _window(fullscreen ? &window_state.fullscreen.w.show : &window_state.windowed.w.show); \
    }
    W(io)
    W(status)
    W(formula)
    W(location)
    W(reference)
    W(bailout)
    W(transform)
    W(algorithm)
    W(information)
    W(quality)
    W(render)
    W(colours)
    W(postprocessing)
    W(newton)
    W(wisdom)
    W(preferences)
    W(about)
#ifdef HAVE_IMGUI_DEMO
    W(demo)
#endif
#undef W
  }
  if (! error_modal)
  {
    if (! batch_running)
    {
      if (! benchmarking_modal)
      {
        display_newton_modal(newton_modal);
      }
      display_benchmarking_modal(benchmarking_modal);
    }
    display_batch_modal(batch_running);
  }
  display_error_modal(error_modal);

  ImGui::Render();
  glViewport(0, 0, win_pixel_width, win_pixel_height);
  glClearColor(par.p.colour.background.r, par.p.colour.background.g, par.p.colour.background.b, 1);
  glClear(GL_COLOR_BUFFER_BIT);
  display_background(window, dsp);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
#ifndef __EMSCRIPTEN__
  SDL_GL_SwapWindow(window);
#endif
}

bool want_capture(int type)
{
  ImGuiIO& io = ImGui::GetIO();
  return
    (io.WantCaptureMouse && (
      type == SDL_MOUSEBUTTONDOWN ||
      type == SDL_MOUSEBUTTONUP ||
      type == SDL_MOUSEWHEEL ||
      type == SDL_MOUSEMOTION ||
      type == SDL_FINGERDOWN ||
      type == SDL_FINGERUP ||
      type == SDL_FINGERMOTION)) ||
    (io.WantCaptureKeyboard && (
      type == SDL_KEYDOWN ||
      type == SDL_KEYUP)) ;
}

enum { st_start, st_render_start, st_subframe_start, st_render, st_render_end, st_idle, st_quit, st_newton_start, st_newton, st_newton_end, st_benchmark_start, st_benchmark, st_benchmark_end } state = st_start;

#define GUI_BUSY 10
int gui_busy = GUI_BUSY;

void main1()
{
  // colour tiles resulting from calculations
  {
    std::lock_guard<std::mutex> lock(tile_queue_mutex);
    if (recolour)
    {
      colour_upload(clr);
      for (auto & spec : tile_cache)
      {
        tile_queue.push_back(spec);
      }
      tile_cache.clear();
      rgb->clear();
      recolour = false;
    }
    for (auto & spec : tile_queue)
    {
      colour_tile(clr, spec.x, spec.y, spec.subframe, spec.data);
      rgb->blit(spec.x, spec.y, spec.data);
      if (spec.subframe < tile_cache_subframes)
      {
        tile_cache.push_back(spec);
      }
      else
      {
        tile_delete(spec.data);
      }
      needs_redraw = 1;
    }
    tile_queue.clear();
  }

  const count_t count = par.p.formula.per.size();
  switch (state)
  {
    case st_start:
      if (quit)
      {
        state = st_quit;
      }
      else
      {
        if (batch_running)
        {
          par.zoom = batch_zoom / pow(floatexp<float, int>(par.p.render.zoom_out_factor), batch_frame);
          restring_vals(par);
        }
        progress.resize(2 * count + 3);
        for (int i = 0; i < 2 * count + 3; ++i)
        {
          progress[i] = 0;
        }
        running = true;
        ended = false;
        restart = false;
        recolour = false;
        just_did_newton = false;
        start_time = std::chrono::steady_clock::now();
        timestamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        started = 0;
        needs_redraw = 0;
        rgb->clear();
        raw->clear();
        {
          std::lock_guard<std::mutex> lock(tile_queue_mutex);
          for (auto & spec : tile_cache)
          {
            tile_delete(spec.data);
          }
          tile_cache.clear();
          for (auto & spec : tile_queue)
          {
            tile_delete(spec.data);
          }
          tile_queue.clear();
        }
        {
          using namespace std::chrono_literals;
          colour_set_time(clr, (start_time - program_start_time) / 1.0s);
        }
        colour_set_zoom_log_2(clr, float(log2(par.zoom)));
        colour_stale_cdf(clr);
        state = st_render_start;
      }
      break;
    case st_render_start:
      if (quit)
      {
        state = st_quit;
      }
      else
      {
        capture_background(window, *dsp);
        running = true;
        ended = false;
        subframes_rendered = 0;
        bg = new std::thread(render_thread, &progress[0], &running, &ended);
        state = st_render;
      }
      break;
    case st_subframe_start:
      if (quit)
      {
        state = st_quit;
      }
      else
      {
        running = true;
        ended = false;
        bg = new std::thread(render_subframe_thread, &progress[0], &running, &ended);
        state = st_render;
      }
      break;
    case st_render:
      if (quit)
      {
        running = false;
        state = st_quit;
      }
      else if (restart)
      {
        running = false;
        if (ended)
        {
          state = st_render_end;
        }
        else
        {
          state = st_render;
        }
      }
      else
      {
        auto current_time = std::chrono::steady_clock::now();
        duration = current_time - start_time;
        int redraw = needs_redraw.exchange(0);
        int dopost = needs_dopost.exchange(0);
        if (redraw)
        {
          finger_transform_started = mat3(1.0f);
        }
        if (redraw || dopost)
        {
          dsp->plot(*rgb, par.p.postprocessing);
          hist_post = dsp->hist;
        }
        display_gui(window, *dsp /* , sta */);
        SDL_Event e;
        while (SDL_PollEvent(&e))
        {
          ImGui_ImplSDL2_ProcessEvent(&e);
          if (! want_capture(e.type))
          {
            handle_event(window, e, par);
          }
          gui_busy = GUI_BUSY;
        }
        if (ended)
        {
          state = st_render_end;
        }
        else
        {
          state = st_render;
        }
      }
      break;
    case st_render_end:
      if (! ended)
      {
        break;
      }
      else
      {
        bg->join();
        delete bg;
        bg = nullptr;
        state = st_idle;
        if (running)
        {
          subframes_rendered++;
          if (subframes_rendered == 1)
          {
            // update information window // FIXME background thread
            hist_de = histogram_logde(*raw);
            hist_n = histogram_n(*raw, 100, 1024, par.p.bailout.iterations + 1024); // FIXME bias
            hist_bla = histogram_bla(*raw, 100, par.p.bailout.maximum_bla_steps);
            hist_ptb = histogram_ptb(*raw, 100, par.p.bailout.maximum_perturb_iterations);
            histogram hist_n2 = histogram_n(*raw, 4096, count_t(~0), count_t(0)); // FIXME bias
            histogramcdf cdf_n2 = histogram_cdf(hist_n2);
            cdf_n = histogram_n_warp(*raw, 4096, cdf_n2.minimum, cdf_n2.median, cdf_n2.maximum);
            colour_upload_cdf(clr, cdf_n);
            if (par.p.colour.uses_histogram)
            {
              recolour = true;
            }
          }
          hist_pre = histogram_rgb(*rgb, 256);
          if (par.p.image.subframes == 0 || subframes_rendered < par.p.image.subframes)
          {
            state = st_subframe_start;
          }
        }
      }
      break;
    case st_idle:
      if (quit)
      {
        state = st_quit;
      }
      else if (start_newton)
      {
        state = st_newton_start;
      }
      else if (start_benchmark)
      {
        state = st_benchmark_start;
      }
      else if (restart)
      {
        state = st_start;
      }
      else if (continue_subframe_rendering)
      {
        continue_subframe_rendering = false;
        state = st_subframe_start;
      }
      else if (batch_running && ! batch_ended)
      {
        std::ostringstream filename;
        filename << par.p.render.filename;
        if (par.p.render.zoom_out_sequence)
        {
          filename << "." << std::setfill('0') << std::setw(8) << (par.p.render.numbering_offset + batch_frame);
        }
        bool ok = true;
        std::string failed_files = "";
        if (par.p.render.save_toml)
        {
          bool ok1 = save_any(filename.str() + ".f3.toml");
          if (! ok1)
          {
            failed_files += "- " + filename.str() + ".f3.toml\n";
          }
          ok &= ok1;
        }
        if (par.p.render.save_exr)
        {
          if ((par.exr_channels & Channels_RGB) != 0)
          {
            bool ok1 = save_any(filename.str() + ".exr");
            if (! ok1)
            {
              failed_files += "- " + filename.str() + ".exr\n";
            }
            ok &= ok1;
          }
          else
          {
            int threads = std::thread::hardware_concurrency();
            bool ok1 = image_raw(*raw, ! par.p.transform.vertical_flip).save_exr(filename.str() + ".exr", par.exr_channels, par.p.bailout.iterations, threads, par.to_string(), par.p.image.dpi);
            if (! ok1)
            {
              failed_files += "- " + filename.str() + ".exr\n";
            }
            ok &= ok1;
#ifdef __EMSCRIPTEN__
            if (ok1)
            {
              download_handler(filename.str() + ".exr");
              unlink((filename.str() + ".exr").c_str());
            }
#endif
          }
        }
        if (par.p.render.save_png)
        {
          bool ok1 = save_any(filename.str() + ".png");
          if (! ok1)
          {
            failed_files += "- " + filename.str() + ".png\n";
          }
          ok &= ok1;
        }
        if (par.p.render.save_jpg)
        {
          bool ok1 = save_any(filename.str() + ".jpg");
          if (! ok1)
          {
            failed_files += "- " + filename.str() + ".jpg\n";
          }
          ok &= ok1;
        }
        // continue
        if (ok)
        {
          batch_frame++;
          if (batch_frame >= batch_frame_end)
          {
            batch_ended = true;
          }
          else
          {
            state = st_start;
          }
        }
        else
        {
          display_error_string = "could not save files:\n" + failed_files;
          batch_quit_when_done = false;
          batch_ended = true;
          state = st_idle;
        }
      }
      else
      {
        int redraw = needs_redraw.exchange(0);
        int dopost = needs_dopost.exchange(0);
        if (redraw)
        {
          finger_transform_started = mat3(1.0f);
        }
        if (redraw || dopost)
        {
          dsp->plot(*rgb, par.p.postprocessing);
          hist_post = dsp->hist;
        }
        display_gui(window, *dsp /* , sta */);
        bool got_event = false;
        do
        {
          got_event = false;
          gui_busy--;
          if (gui_busy < 0)
          {
            gui_busy = 0;
          }
          SDL_Event e;
          if (gui_busy ? (got_event = SDL_PollEvent(&e)) : SDL_WaitEvent(&e))
          {
            ImGui_ImplSDL2_ProcessEvent(&e);
            if (! want_capture(e.type))
            {
              handle_event(window, e, par);
            }
            gui_busy = GUI_BUSY;
          }
        }
        while (got_event);
      }
      break;
    case st_quit:
      {
      }
      break;
    case st_newton_start:
      {
        newton_running = true;
        newton_ended = false;
        start_newton = false;
        newton_progress[0] = 0;
        newton_progress[1] = 0;
        newton_progress[2] = 0;
        newton_progress[3] = 0;
        newton_c.x = newton_c.x - floatexp<float, int>(par.reference.x - par.center.x);
        newton_c.y = newton_c.y - floatexp<float, int>(par.reference.y - par.center.y);
        bg = new std::thread (newton_thread, std::ref(newton_par), std::ref(newton_ok), std::cref(par), std::cref(newton_c), std::cref(newton_r), &newton_progress[0], &newton_running, &newton_ended);
        state = st_newton;
      }
      break;
    case st_newton:
      if (! quit)
      {
        int redraw = needs_redraw.exchange(0);
        int dopost = needs_dopost.exchange(0);
        if (redraw)
        {
          finger_transform_started = mat3(1.0f);
        }
        if (redraw || dopost)
        {
          dsp->plot(*rgb, par.p.postprocessing);
          hist_post = dsp->hist;
        }
        display_gui(window, *dsp /*, sta*/ , true);
        SDL_Event e;
        while (SDL_PollEvent(&e))
        {
          ImGui_ImplSDL2_ProcessEvent(&e);
          if (! want_capture(e.type))
          {
            handle_event(window, e, par);
          }
        }
        gui_busy = GUI_BUSY;
      }
      if (quit || newton_ended)
      {
        state = st_newton_end;
      }
      break;
    case st_newton_end:
      {
        bg->join();
        delete bg;
        bg = nullptr;
        if (newton_ok)
        {
          par = newton_par;
          just_did_newton = true;
          state = st_start;
          if (newton_zoom_auto_capture)
          {
            newton_relative_start = par.zoom;
            std::ostringstream s;
            s << newton_relative_start;
            newton_relative_start_str = s.str();
          }
        }
        else
        {
          state = st_idle;
        }
      }
      break;
    case st_benchmark_start:
      {
        start_benchmark = false;
        benchmark_ended = false;
        benchmark_running = true;
        benchmark_progress = 0;
        benchmark_wisdom = wdom;
        bg = new std::thread (benchmark_thread, &benchmark_progress, &benchmark_running, &benchmark_ended);
        state = st_benchmark;
      }
      break;
    case st_benchmark:
      if (! quit)
      {
        int redraw = needs_redraw.exchange(0);
        int dopost = needs_dopost.exchange(0);
        if (redraw)
        {
          finger_transform_started = mat3(1.0f);
        }
        if (redraw || dopost)
        {
          dsp->plot(*rgb, par.p.postprocessing);
          hist_post = dsp->hist;
        }
        display_gui(window, *dsp /*, sta*/ , false, true);
        SDL_Event e;
        while (SDL_PollEvent(&e))
        {
          ImGui_ImplSDL2_ProcessEvent(&e);
          if (! want_capture(e.type))
          {
            handle_event(window, e, par);
          }
        }
        gui_busy = GUI_BUSY;
      }
      if (quit || benchmark_ended)
      {
        state = st_benchmark_end;
      }
      break;
    case st_benchmark_end:
      {
        bg->join();
        delete bg;
        bg = nullptr;
        if (benchmark_ok)
        {
          wdom = benchmark_wisdom;
          state = st_start;
        }
        else
        {
          state = st_idle;
        }
      }
      break;
  }

}

GLADapiproc get_proc_address(void *userptr, const char *name)
{
  (void) userptr;
  return (GLADapiproc) SDL_GL_GetProcAddress(name);
}

int gui(const char *progname, const char *persistence_str, bool batchmode)
{
  program_start_time = std::chrono::steady_clock::now();
  persistence = persistence_str;
  if (persistence_str)
  {
    persistence_path = persistence_str;
  }
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)
  {
    const std::string message = "SDL_Init: " + std::string(SDL_GetError());
    if (0 != SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Fraktaler 3", message.c_str(), nullptr))
    {
      std::cerr << progname << ": " << message << std::endl;
    }
    return 1;
  }

#if defined(__ANDROID__) && ! defined(__TERMUX__)
  SDL_DisplayMode mode;
  if (SDL_GetDesktopDisplayMode(0, &mode) != 0)
  {
    const std::string message = "SDL_GetDesktopDisplayMode: " + std::string(SDL_GetError());
    if (0 != SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Fraktaler 3", message.c_str(), nullptr))
    {
      SDL_LogCritical(SDL_LOG_CATEGORY_APPLICATION, "%s", message.c_str());
    }
    SDL_Quit();
    return 1;
  }
  int win_screen_width = mode.w;
  int win_screen_height = mode.h;
#else
  int win_screen_width = 1024;
  int win_screen_height = 576;
#endif

  // decide GL+GLSL versions
  // see f3imconfig.h
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, OPENGL_FLAGS);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, OPENGL_PROFILE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, OPENGL_MAJOR);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, OPENGL_MINOR);

  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
  SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE /* | SDL_WINDOW_ALLOW_HIGHDPI */);
  window = SDL_CreateWindow("Fraktaler 3", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, win_screen_width, win_screen_height, window_flags);
  if (! window)
  {
    const std::string message = "SDL_CreateWindow: " + std::string(SDL_GetError());
    if (0 != SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Fraktaler 3", message.c_str(), nullptr))
    {
      SDL_LogCritical(SDL_LOG_CATEGORY_APPLICATION, "%s", message.c_str());
    }
    SDL_Quit();
    return 1;
  }
  SDL_GLContext gl_context = SDL_GL_CreateContext(window);
  if (! gl_context)
  {
    const std::string message = "SDL_GL_CreateContext: " + std::string(SDL_GetError());
    if (0 != SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Fraktaler 3", message.c_str(), window))
    {
      SDL_LogCritical(SDL_LOG_CATEGORY_APPLICATION, "%s", message.c_str());
    }
    SDL_Quit();
    return 1;
  }
  SDL_GL_MakeCurrent(window, gl_context);

#ifdef WANT_EXT_COLOR_BUFFER_FLOAT
#ifdef __EMSCRIPTEN__
  bool EXT_color_buffer_float = emscripten_webgl_enable_extension(emscripten_webgl_get_current_context(), "EXT_color_buffer_float");
  (void) EXT_color_buffer_float;
#endif
#endif
  gladLoadGLES2UserPtr(get_proc_address, nullptr);
#ifndef WANT_EXT_COLOR_BUFFER_FLOAT
  GLAD_GL_EXT_color_buffer_float = 1; // in core gl
#endif

#ifdef HAVE_GLDEBUG
  if (glDebugMessageCallback)
  {
    glDebugMessageCallback(opengl_debug_callback, nullptr);
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
  }
#endif

  gl_version = (const char *) glGetString(GL_VERSION);
#ifdef NEED_EXT_COLOR_BUFFER_FLOAT
  if (! EXT_color_buffer_float)
  {
    const char *extensions = (const char *) glGetString(GL_EXTENSIONS);
    const std::string message = "could not enable OpenGL extension EXT_color_buffer_float\nversion: " + std::string(gl_version) + "\nextensions: " + std::string(extensions);
    if (0 != SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Fraktaler 3", message.c_str(), window))
    {
      SDL_LogCritical(SDL_LOG_CATEGORY_APPLICATION, "%s", message.c_str());
    }
    SDL_Quit();
    return 1;
  }
#endif
#ifdef __EMSCRIPTEN__
  srgb_conversion = 1; // FIXME, should check if framebuffer is linear or sRGB
#endif

  SDL_GL_SetSwapInterval(1);
  int win_pixel_width = 0;
  int win_pixel_height = 0;
  SDL_GL_GetDrawableSize(window, &win_pixel_width, &win_pixel_height);

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glClearColor(par.p.colour.background.r, par.p.colour.background.g, par.p.colour.background.b, 1);
  glClear(GL_COLOR_BUFFER_BIT);

  glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maximum_texture_size);

  // set window icon
#ifdef _WIN32
  {
    HINSTANCE instance = GetModuleHandle(nullptr);
    HICON icon = LoadIcon(instance, "IDI_MAIN_ICON");
    if (icon)
    {
      SDL_SysWMinfo info;
      SDL_VERSION(&info.version);
      if (SDL_TRUE == SDL_GetWindowWMInfo(window, &info))
      {
        HWND wnd = info.info.win.window;
        SetClassLong(wnd, GCLP_HICON, PtrToLong(icon));
      }
    }
  }
#else
#ifdef HAVE_ICON
  {
    SDL_Surface *surface = SDL_LoadBMP_RW(SDL_RWFromConstMem(fraktaler_3_bmp, sizeof(fraktaler_3_bmp)), 1);
    if (surface)
    {
      SDL_SetWindowIcon(window, surface);
      SDL_FreeSurface(surface);
    }
  }
#endif
#endif

  // setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImGui::StyleColorsDark();
  ImGui::GetIO().IniFilename = nullptr;
#if 0
  // for slow machine where FPS is slower than default 0.30f
  ImGui::GetIO().MouseDoubleClickTime = 2.0f;
#endif
  ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
  ImGui_ImplOpenGL3_Init(OPENGL_GLSL_VERSION);
  initialize_clipboard();

  try {
#if LOADSAVE_DIALOGS
    load_dialog = new ImGui::FileBrowser(ImGuiFileBrowserFlags_CloseOnEsc | ImGuiFileBrowserFlags_SkipItemsCausingError);
    load_dialog->SetTitle("Load...");
    load_dialog->SetTypeFilters({ ".toml", ".exr", ".png", ".jpg", ".jpeg", ".glsl", ".kfr" });
    save_dialog = new ImGui::FileBrowser(ImGuiFileBrowserFlags_CloseOnEsc | ImGuiFileBrowserFlags_SkipItemsCausingError | ImGuiFileBrowserFlags_EnterNewFilename | ImGuiFileBrowserFlags_CreateNewDir);
    save_dialog->SetTitle("Save...");
    save_dialog->SetTypeFilters({ ".toml", ".exr", ".png", ".jpg", ".jpeg", ".glsl" });
    render_dialog = new ImGui::FileBrowser(ImGuiFileBrowserFlags_CloseOnEsc | ImGuiFileBrowserFlags_SkipItemsCausingError | ImGuiFileBrowserFlags_EnterNewFilename | ImGuiFileBrowserFlags_CreateNewDir);
    render_dialog->SetTitle("Render...");
#endif
#if WISDOM_DIALOGS
    wisdom_load_dialog = new ImGui::FileBrowser(ImGuiFileBrowserFlags_CloseOnEsc | ImGuiFileBrowserFlags_SkipItemsCausingError);
    wisdom_load_dialog->SetTitle("Load Wisdom...");
    wisdom_load_dialog->SetTypeFilters({ ".toml" });
    wisdom_load_dialog->SetPwd(pref_path);
    wisdom_save_dialog = new ImGui::FileBrowser(ImGuiFileBrowserFlags_CloseOnEsc | ImGuiFileBrowserFlags_SkipItemsCausingError | ImGuiFileBrowserFlags_EnterNewFilename | ImGuiFileBrowserFlags_CreateNewDir);
    wisdom_save_dialog->SetTitle("Save Wisdom...");
    wisdom_save_dialog->SetTypeFilters({ ".toml" });
    wisdom_save_dialog->SetPwd(pref_path);
#endif
  }
  catch (const std::exception &e)
  {
    SDL_LogWarn(SDL_LOG_CATEGORY_APPLICATION, "creating file browsers: %s", e.what());
  }

  {
    try
    {
      std::ifstream ifs(pref_path + "gui-settings.toml", std::ios_base::binary);
      ifs.exceptions(std::ifstream::badbit);
      ifs >> window_state;
    }
    catch (const std::exception &e)
    {
      SDL_LogWarn(SDL_LOG_CATEGORY_APPLICATION, "loading GUI settings: %s", e.what());
    }
    dsp = new display_gles();
    clr = colour_new();
    gui_post_load(par);
//      reset(sta);
  }
  SDL_AddTimer(one_minute, persistence_timer_callback, nullptr);
  ImGui::GetIO().FontGlobalScale = window_state.ui_scale / 100.0f;

  if (batchmode)
  {
    batch_quit_when_done = true;
    start_batch();
  }
  state = st_start;

#ifdef __EMSCRIPTEN__
  emscripten_set_main_loop(main1, 0, false);
  return 0;
#else
  while (! quit)
  {
    main1();
  }
#if 0
  {
    std::lock_guard<std::mutex> lock(tile_queue_mutex);
    for (auto & spec : tile_cache)
    {
      tile_delete(spec.data);
    }
    tile_cache.clear();
    for (auto & spec : tile_queue)
    {
      tile_delete(spec.data);
    }
    tile_queue.clear();
  }
#endif
#endif

#if LOADSAVE_DIALOGS
  if (load_dialog)
  {
    delete load_dialog;
    load_dialog = nullptr;
  }
  if (save_dialog)
  {
    delete save_dialog;
    save_dialog = nullptr;
  }
  if (render_dialog)
  {
    delete render_dialog;
    render_dialog = nullptr;
  }
#endif
#if WISDOM_DIALOGS
  if (wisdom_load_dialog)
  {
    delete wisdom_load_dialog;
    wisdom_load_dialog = nullptr;
  }
  if (wisdom_save_dialog)
  {
    delete wisdom_save_dialog;
    wisdom_save_dialog = nullptr;
  }
#endif

  // cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();
  SDL_GL_DeleteContext(gl_context);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}

#endif
