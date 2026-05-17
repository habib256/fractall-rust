// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#include <atomic>

#include "engine.h"
#include "floatexp.h"
#include "hybrid.h"
#include "opencl.h"
#include "param.h"
#include "parallel.h"
#include "render.h"
#include "softfloat.h"
#include "wisdom.h"

struct tile_queue
{
  std::atomic<coord_t> index;
  std::atomic<coord_t> done;
  coord_t count;
  coord_t width;
  coord_t height;
  coord_t subframes;
  progress_t *progress;
};

bool get_tile(tile_queue &queue, coord_t &x, coord_t &y, coord_t &subframe)
{
  coord_t index = queue.index++;
  if (queue.count == 0 || index < queue.count)
  {
    x = index % queue.width;
    y = (index / queue.width) % queue.height;
    subframe = (index / queue.width) / queue.height;
    if (queue.count == 0)
    {
      queue.progress[0] = index / progress_t(index + 1);
    }
    else
    {
      queue.progress[0] = index / progress_t(queue.count);
    }
    return true;
  }
  else
  {
    queue.progress[0] = 1;
  }
  return false;
}

bool done_tile(tile_queue &queue)
{
  coord_t done = ++queue.done;
  if (queue.count == 0 || done < queue.count)
  {
    if (queue.count == 0)
    {
      queue.progress[1] = done / progress_t(done + 1);
    }
    else
    {
      queue.progress[1] = done / progress_t(queue.count);
    }
    return true;
  }
  else
  {
    queue.progress[1] = 1;
  }
  return false;
}

bool render_tile(coord_t frame, coord_t tx, coord_t ty, coord_t subframe, tile *data, const param &par, number_type nt, volatile bool *running)
{
  coord_t x0 = tx * data->width;
  coord_t x1 = std::min(x0 + data->width,  coord_t(par.p.image.width  * par.p.image.supersampling + par.p.image.subsampling - 1) / par.p.image.subsampling);
  coord_t y0 = ty * data->height;
  coord_t y1 = std::min(y0 + data->height, coord_t(par.p.image.height * par.p.image.supersampling + par.p.image.subsampling - 1) / par.p.image.subsampling);
  switch (nt)
  {
    case nt_float: return hybrid_render(frame, x0, y0, x1, y1, subframe, data, par, Zf, Bf, running);
    case nt_double: return hybrid_render(frame, x0, y0, x1, y1, subframe, data, par, Zd, Bd, running);
    case nt_longdouble: return hybrid_render(frame, x0, y0, x1, y1, subframe, data, par, Zld, Bld, running);
    case nt_floatexp: return hybrid_render(frame, x0, y0, x1, y1, subframe, data, par, Zfe, Bfe, running);
    case nt_doubleexp: return hybrid_render(frame, x0, y0, x1, y1, subframe, data, par, Zde, Bde, running);
    case nt_softfloat: return hybrid_render(frame, x0, y0, x1, y1, subframe, data, par, Zsf, Bsf, running);
#ifdef HAVE_FLOAT128
    case nt_float128: return hybrid_render(frame, x0, y0, x1, y1, subframe, data, par, Zq, Bq, running);
#endif
    default: return false;
  }
}

void render_device(tile_queue &queue, number_type nt, const wdevice &device, const param &par, hooks *h, bool ref_recalculated, bool bla_recalculated, volatile bool *running)
{
  hooks default_hooks;
  if (! h)
  {
    h = &default_hooks;
  }
  h->pre_device(device.platform, device.device);
  if (device.platform == -1 && device.device == 0)
  {
    h->pre_compile(device.platform, device.device);
    // regular CPU code
    tile data;
    std::memset(&data, 0, sizeof(data));
    data.width = par.p.opencl.tile_width;
    data.height = par.p.opencl.tile_height;
    data.RGB = new float[3 * data.width * data.height];
    data.N0  = new uint32_t[data.width * data.height];
    data.N1  = new uint32_t[data.width * data.height];
    data.NF  = new float[data.width * data.height];
    data.T   = new float[data.width * data.height];
    data.DEX = new float[data.width * data.height];
    data.DEY = new float[data.width * data.height];
    data.BLA = new uint32_t[data.width * data.height];
    data.PTB = new uint32_t[data.width * data.height];
    h->post_compile(device.platform, device.device);
    h->pre_upload(device.platform, device.device);
    h->post_upload(device.platform, device.device);
    // main rendering loop
    const coord_t frame = 0; // FIXME
    bool ok = true;
    coord_t x, y, subframe;
    while (get_tile(queue, x, y, subframe) && *running && ok)
    {
      subframe += par.p.render.prng_seed;
      h->pre_tile(device.platform, device.device, x, y, subframe);
      ok &= render_tile(frame, x, y, subframe, &data, par, nt, running);
      h->post_tile(device.platform, device.device, x, y, subframe);
      if (! *running || ! ok)
      {
        break;
      }
      h->pre_download(device.platform, device.device, x, y, subframe);
      h->tile(device.platform, device.device, x, y, subframe, &data);
      h->post_download(device.platform, device.device, x, y, subframe);
      done_tile(queue);
    }
    delete[] data.RGB;
    delete[] data.N0;
    delete[] data.N1;
    delete[] data.NF;
    delete[] data.T;
    delete[] data.DEX;
    delete[] data.DEY;
    delete[] data.BLA;
    delete[] data.PTB;
  }
#ifdef HAVE_CL
  else
  {
    // may retrieve already-created context from cache
    opencl_context *context = opencl_get_context(device.platform, device.device);
    if (context)
    {
      h->pre_compile(device.platform, device.device);
      // may retrieve already-compiled kernel from cache or compile new one
      opencl_kernel *kernel = opencl_get_kernel(context, nt, par);
      h->post_compile(device.platform, device.device);
      if (kernel)
      {
        size_t ref_bytes = opencl_ref_layout(opencl_kernel_config_host(kernel), nt);
        size_t bla_bytes = opencl_bla_layout(opencl_kernel_config_host(kernel), nt);
        // may retrieve already-allocated buffers from cache or allocate new one
        coord_t tile_width = par.p.opencl.tile_width; // FIXME
        coord_t tile_height = par.p.opencl.tile_height; // FIXME
        h->pre_upload(device.platform, device.device);
        if (opencl_get_buffers(context, ref_bytes, bla_bytes, tile_width, tile_height, true))
        {
          // may reuse already-uploaded data if par and nt unchanged
          opencl_upload_config(context, kernel);
          if (ref_recalculated)
          {
            opencl_upload_ref(context, kernel);
          }
          if (bla_recalculated)
          {
            opencl_upload_bla(context, kernel);
          }
          opencl_set_kernel_arguments(context, kernel);
          h->post_upload(device.platform, device.device);
          // main rendering loop
          coord_t x, y, subframe;
          while (get_tile(queue, x, y, subframe) && *running)
          {
            subframe += par.p.render.prng_seed;
            h->pre_tile(device.platform, device.device, x, y, subframe);
            opencl_render_tile(context, kernel, x, y, subframe);
            h->post_tile(device.platform, device.device, x, y, subframe);
            h->pre_download(device.platform, device.device, x, y, subframe);
            tile *data = opencl_map_tile(context);
            h->tile(device.platform, device.device, x, y, subframe, data);
            data = nullptr;
            opencl_unmap_tile(context);
            h->post_download(device.platform, device.device, x, y, subframe);
            done_tile(queue);
          }
        }
        else
        {
          h->post_upload(device.platform, device.device);
        }
        opencl_release_kernel(context, kernel);
      }
      opencl_release_context(context);
    }
  }
#else
  (void) ref_recalculated;
  (void) bla_recalculated;
#endif
  h->post_device(device.platform, device.device);
}

void get_required_precision(const param &par, count_t &pixel_spacing_exp, count_t &pixel_precision_exp)
{
  using std::max;
  floatexp pixel_spacing =
    4 / (par.zoom * ((par.p.image.height * par.p.image.supersampling + par.p.image.subsampling - 1) / par.p.image.subsampling));
  complex<mpreal> offset;
  offset.x.set_prec(par.center.x.get_prec());
  offset.y.set_prec(par.center.y.get_prec());
  offset = par.center - par.reference;
  floatexp<float, int> pixel_precision = max
    ( max(abs(floatexp<float, int>(offset.x) / pixel_spacing)
        , abs(floatexp<float, int>(offset.y) / pixel_spacing))
    , hypot(floatexp<float, int>((par.p.image.width  * par.p.image.supersampling + par.p.image.subsampling - 1) / par.p.image.subsampling)
          , floatexp<float, int>((par.p.image.height * par.p.image.supersampling + par.p.image.subsampling - 1) / par.p.image.subsampling))
    );
  pixel_spacing_exp = std::abs(pixel_spacing.exp);
  pixel_precision_exp = pixel_precision.exp;
}

bool reference_can_be_reused(const wlookup &l, const param &par)
{
  using std::max;
  count_t pixel_spacing_exp, pixel_precision_exp;
  get_required_precision(par, pixel_spacing_exp, pixel_precision_exp);
  return nt_ref == l.nt &&
    pixel_spacing_exp < (count_t(1) << l.exponent) >> 1 &&
    max(count_t(24), 24 - pixel_precision_exp) <= l.mantissa &&
    par.p.algorithm.reuse_reference &&
    ! just_did_newton;
}

bool bla_can_be_reused(const wlookup &l, const param &par) // FIXME
{
  return nt_bla == l.nt &&
    par.p.algorithm.reuse_bilinear_approximation ;
}

// std::vector<progress_t> progress(par.opss.size() * 2 + 2), 0);
void render(const wlookup &l, const param &par, hooks *h, bool first, progress_t *progress, volatile bool *running)
{
  hooks default_hooks;
  if (! h)
  {
    h = &default_hooks;
  }
  h->start();
  bool ref_recalculated = false;
  bool bla_recalculated = false;
  if (first)
  {
    // reference
    if (*running)
    {
      bool recalc_ref = ! reference_can_be_reused(l, par);
      if (h->pre_reference(recalc_ref) || recalc_ref)
      {
        calculate_reference(l.nt, par, &progress[0], running);
        ref_recalculated = true;
      }
      h->post_reference(ref_recalculated);
    }
    nt_ref = l.nt;
    // bla
    if (*running)
    {
      bool recalc_bla = ref_recalculated || ! bla_can_be_reused(l, par);
      if (h->pre_bla(recalc_bla) || recalc_bla)
      {
        calculate_bla(l.nt, par, &progress[par.opss.size()], running);
        bla_recalculated = true;
      }
      h->post_bla(bla_recalculated);
    }
    nt_bla = l.nt; // FIXME
  }
  // tiles
  coord_t width  = (par.p.image.width  * par.p.image.supersampling + par.p.image.subsampling - 1) / par.p.image.subsampling;
  coord_t height = (par.p.image.height * par.p.image.supersampling + par.p.image.subsampling - 1) / par.p.image.subsampling;
  coord_t tiling_width = (width + par.p.opencl.tile_width - 1) / par.p.opencl.tile_width;
  coord_t tiling_height = (height + par.p.opencl.tile_height - 1) / par.p.opencl.tile_height;
  coord_t tile_count = tiling_width * tiling_height * par.p.image.subframes;
  progress[par.opss.size() * 2 + 0] = 0;
  progress[par.opss.size() * 2 + 1] = 0;
  tile_queue queue { {0}, {0}, tile_count, tiling_width, tiling_height, par.p.image.subframes, &progress[par.opss.size() * 2] };
  h->pre_render();
  if (*running)
  {
    parallel1d(l.device.size(), 0, l.device.size(), 1, running, [&](coord_t device) -> void
    {
      if (l.device[device].platform == -1 && l.device[device].device == 0)
      {
        int threads = std::thread::hardware_concurrency();
        parallel1d(threads, 0, threads, 1, running, [&](coord_t thread) -> void
        {
          (void) thread;
          render_device(queue, l.nt, l.device[device], par, h, ref_recalculated, bla_recalculated, running);
        });
      }
      else
      {
        render_device(queue, l.nt, l.device[device], par, h, ref_recalculated, bla_recalculated, running);
      }
    });
  }
  h->post_render();
  h->stop();
}

tile *tile_copy(const tile *src)
{
  tile *dst = new tile();
  dst->width = src->width;
  dst->height = src->height;
  dst->RGB = new float[3 * dst->width * dst->height];
  dst->N0  = new uint32_t[dst->width * dst->height];
  dst->N1  = new uint32_t[dst->width * dst->height];
  dst->NF  = new float[dst->width * dst->height];
  dst->T   = new float[dst->width * dst->height];
  dst->DEX = new float[dst->width * dst->height];
  dst->DEY = new float[dst->width * dst->height];
  dst->BLA = new uint32_t[dst->width * dst->height];
  dst->PTB = new uint32_t[dst->width * dst->height];
  memcpy(dst->RGB, src->RGB, sizeof(*dst->RGB) * 3 * dst->width * dst->height);
  memcpy(dst->N0,  src->N0,  sizeof(*dst->N0 ) * dst->width * dst->height);
  memcpy(dst->N1,  src->N1,  sizeof(*dst->N1 ) * dst->width * dst->height);
  memcpy(dst->NF,  src->NF,  sizeof(*dst->NF ) * dst->width * dst->height);
  memcpy(dst->T,   src->T,   sizeof(*dst->T  ) * dst->width * dst->height);
  memcpy(dst->DEX, src->DEX, sizeof(*dst->DEX) * dst->width * dst->height);
  memcpy(dst->DEY, src->DEY, sizeof(*dst->DEY) * dst->width * dst->height);
  memcpy(dst->BLA, src->BLA, sizeof(*dst->BLA) * dst->width * dst->height);
  memcpy(dst->PTB, src->PTB, sizeof(*dst->PTB) * dst->width * dst->height);
  return dst;
}

void tile_delete(tile *t)
{
  delete[] t->RGB;
  delete[] t->N0;
  delete[] t->N1;
  delete[] t->NF;
  delete[] t->T;
  delete[] t->DEX;
  delete[] t->DEY;
  delete[] t->BLA;
  delete[] t->PTB;
  delete t;
}
