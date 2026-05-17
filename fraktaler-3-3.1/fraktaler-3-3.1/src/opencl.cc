// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#ifdef HAVE_CL

#include <mutex>

#define CL_TARGET_OPENCL_VERSION 200
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef HAVE_CLEW
#include "clew.h"
#else
#include <CL/cl.h>
#endif

#include "bla.h"
#include "engine.h"
#include "hybrid.h"
#include "opencl.h"
#include "param.h"
#include "softfloat.h"

#include "cl-pre.h"
#include "cl-post.h"

#define MAX_PHASES 64
#define MAX_LEVELS 64

// used to find maximum possible size of config structure
union ufloat
{
  float f;
  double d;
  floatexp<float, int> fe;
  floatexp<double, int> de;
  softfloat sf;
};

template <typename real>
struct mat2_cl
{
  real a, b, c, d;
};

template <typename real>
struct blaR2_cl
{
  struct mat2_cl<real> A, B;
  real r2;
  cl_long l;
};

template <typename real>
struct config_cl
{
  cl_long size;
  cl_long number_type;
  /* shape */
  cl_long height;
  cl_long width;
  cl_long supersampling;
  cl_long subsampling;
  cl_long tile_height;
  cl_long tile_width;
  cl_long subframes;
  cl_long frame;
  /* bailout */
  cl_long Iterations;
  real ER2;
  cl_long PerturbIterations;
  cl_long BLASteps;
  /* transform */
  cl_int transform_exponential_map;
  cl_int transform_vertical_flip;
  struct mat2_cl<real> transform_K;
  real pixel_spacing;
  real offset_x;
  real offset_y;
  /* ref layout */
  cl_long number_of_phases;
  cl_long degree[MAX_PHASES];
  cl_long ref_size[MAX_PHASES];
  cl_long ref_start[MAX_PHASES];
  /* bla layout */
  cl_long bla_size[MAX_PHASES];
  cl_long bla_levels[MAX_PHASES];
  cl_long bla_start[MAX_PHASES][MAX_LEVELS];
  cl_long bla_size_level[MAX_PHASES][MAX_LEVELS];
};

struct opencl_buffers
{
  size_t ref_bytes;
  cl_mem ref_device;
  size_t bla_bytes;
  cl_mem bla_device;
  coord_t tile_width;
  coord_t tile_height;
  size_t RGB_bytes;
  cl_mem RGB_device;
  size_t N0_bytes;
  cl_mem N0_device;
  size_t N1_bytes;
  cl_mem N1_device;
  size_t NF_bytes;
  cl_mem NF_device;
  size_t T_bytes;
  cl_mem T_device;
  size_t DEX_bytes;
  cl_mem DEX_device;
  size_t DEY_bytes;
  cl_mem DEY_device;
  size_t BLA_bytes;
  cl_mem BLA_device;
  size_t PTB_bytes;
  cl_mem PTB_device;
  tile tile_host;
};

struct opencl_context
{
  int platform;
  int device;
  cl_platform_id platform_id;
  cl_device_id device_id;
  bool supports_double;
  cl_context context;
  cl_command_queue command_queue;
  cl_event ready;
  std::vector<opencl_kernel*> kernels;
  opencl_buffers buffers;
  int reference_count;
};

struct opencl_kernel
{
  number_type nt;
  phybrid formula;
  cl_program program;
  cl_kernel kernel;
  size_t config_bytes;
  cl_mem config_device;
  void *config_host;
  int reference_count;
};

std::mutex cache_mutex;
std::vector<opencl_context*> cache;

opencl_context *opencl_get_context(int platform, int device)
{
  std::lock_guard<std::mutex> lock(cache_mutex);
  for (auto context : cache)
  {
    if (context->platform == platform && context->device == device)
    {
      if (context->reference_count > 0)
      {
        return nullptr;
      }
      else
      {
        context->reference_count++;
        return context;
      }
    }
  }
  opencl_context *context = new opencl_context();
  context->platform = platform;
  context->device = device;
  cl_platform_id platform_id[64];
  cl_uint platform_ids;
  cl_int err;
  if (CL_SUCCESS == (err = clGetPlatformIDs(64, &platform_id[0], &platform_ids)))
  {
    if (0 <= platform && platform < (int) platform_ids)
    {
      context->platform_id = platform_id[platform];
      cl_device_id device_id[64];
      cl_uint device_ids;
      if (CL_SUCCESS == (err = clGetDeviceIDs(platform_id[platform], CL_DEVICE_TYPE_ALL, 64, &device_id[0], &device_ids)))
      {
        if (0 <= device && device < (int) device_ids)
        {
          context->device_id = device_id[device];
          cl_uint dvecsize = 0;
          if (CL_SUCCESS != clGetDeviceInfo(device_id[device], CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(dvecsize), &dvecsize, 0))
          {
            dvecsize = 0;
          }
          context->supports_double = dvecsize > 0;
          cl_context_properties properties[] =
          {
            CL_CONTEXT_PLATFORM, (cl_context_properties) platform_id[platform]
          , 0
          };
          context->context = clCreateContext(properties, 1, &device_id[device], NULL, NULL, &err);
          if (context->context)
          {
            context->command_queue = clCreateCommandQueue(context->context, device_id[device], 0, &err);
            if (context->command_queue)
            {
              context->reference_count = 1;
              cache.push_back(context);
              return context;
            }
            else
            {
              clReleaseContext(context->context);
            }
          }
          else
          {
            std::fprintf(stderr, "CL ERROR %d %s %d\n", err, __FUNCTION__, __LINE__);
          }
        }
      }
      else
      {
        std::fprintf(stderr, "CL ERROR %d %s %d\n", err, __FUNCTION__, __LINE__);
      }
    }
  }
  else
  {
    std::fprintf(stderr, "CL ERROR %d %s %d\n", err, __FUNCTION__, __LINE__);
  }
  delete context;
  return nullptr;
}

void opencl_release_context(opencl_context *context)
{
  std::lock_guard<std::mutex> lock(cache_mutex);
  context->reference_count--;
}

template <typename T>
bool opencl_initialize_config(config_cl<T> *config_host, number_type nt, const param &par)
{
  complex<mpreal> offset;
  offset.x.set_prec(par.center.x.get_prec());
  offset.y.set_prec(par.center.y.get_prec());
  offset = par.center - par.reference;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
  config_cl<T> config_host_init = // ignore -Wmissing-field-initializers
    { sizeof(config_host_init)
    , nt
    , (par.p.image.height * par.p.image.supersampling + par.p.image.subsampling - 1) / par.p.image.subsampling
    , (par.p.image.width  * par.p.image.supersampling + par.p.image.subsampling - 1) / par.p.image.subsampling
    , par.p.image.supersampling
    , par.p.image.subsampling
    , par.p.opencl.tile_height
    , par.p.opencl.tile_width
    , par.p.image.subframes
    , 0 // FIXME animation frame is used for PRNG/hash seed
    , par.p.bailout.iterations
    , T(par.p.bailout.escape_radius * par.p.bailout.escape_radius)
    , par.p.bailout.maximum_perturb_iterations
    , par.p.bailout.maximum_bla_steps
    , par.p.transform.exponential_map
    , par.p.transform.vertical_flip
    , { T(par.transform.x[0][0]), T(par.transform.x[0][1]), T(par.transform.x[1][0]), T(par.transform.x[1][1]) }
    , T(4 / par.zoom / ((par.p.image.height * par.p.image.supersampling + par.p.image.subsampling - 1) / par.p.image.subsampling))
    , T(offset.x)
    , T(offset.y)
    , cl_long(par.opss.size())
    // ...
    };
#pragma GCC diagnostic pop
  for (size_t i = 0; i < par.degrees.size(); ++i)
  {
    config_host_init.degree[i] = par.degrees[i];
  }
  *config_host = config_host_init;
  return true;
}

opencl_kernel *opencl_get_kernel(opencl_context *context, number_type nt, const param &par)
{
  // supported types in OpenCL
  if (! (nt == nt_float || nt == nt_double || nt == nt_floatexp || nt == nt_doubleexp || nt == nt_softfloat))
  {
    return nullptr;
  }
  // some devices don't support double
  if ((nt == nt_double || nt == nt_doubleexp) && ! context->supports_double)
  {
    return nullptr;
  }
  // retrieve from cache
  for (auto kernel : context->kernels)
  {
    if (kernel->nt == nt && kernel->formula == par.p.formula)
    {
      if (kernel->reference_count > 0)
      {
        // already in use
        return nullptr;
      }
      else
      {
        // non-formula parts of the parameter may have changed
        switch (nt)
        {
          case nt_float: opencl_initialize_config((config_cl<float>*)(kernel->config_host), nt, par); break;
          case nt_double: opencl_initialize_config((config_cl<double>*)(kernel->config_host), nt, par); break;
          case nt_floatexp: opencl_initialize_config((config_cl<floatexp<float, int>>*)(kernel->config_host), nt, par); break;
          case nt_doubleexp: opencl_initialize_config((config_cl<floatexp<double, int>>*)(kernel->config_host), nt, par); break;
          case nt_softfloat: opencl_initialize_config((config_cl<softfloat>*)(kernel->config_host), nt, par); break;
          default: /* unreachable */ break;
        }
        kernel->reference_count = 1;
        return kernel;
      }
    }
  }
  opencl_kernel *kernel = new opencl_kernel();
  kernel->nt = nt;
  kernel->formula = par.p.formula;
  // prepare kernel source code
  const std::string body = hybrid_perturb_opencl(par.opss, par.degrees);
  unsigned int src_cl_body_cl_len = strlen(body.c_str());
  unsigned int source_len = src_cl_pre_cl_len + src_cl_body_cl_len + src_cl_post_cl_len + 1;
  char *source = new char[source_len];
  std::memcpy(source, src_cl_pre_cl, src_cl_pre_cl_len);
  std::memcpy(source + src_cl_pre_cl_len, body.c_str(), src_cl_body_cl_len);
  std::memcpy(source + src_cl_pre_cl_len + src_cl_body_cl_len, src_cl_post_cl, src_cl_post_cl_len);
  source[source_len - 1] = 0;
  // compile program
  cl_int err;
  kernel->program = clCreateProgramWithSource(context->context, 1, const_cast<const char **>(&source), 0, &err);
  if (kernel->program)
  {
    std::ostringstream optionss;
    optionss << "-DNUMBER_TYPE=" << int(nt);
    optionss << " -DHAVE_DOUBLE=" << int(context->supports_double);
    err = clBuildProgram(kernel->program, 1, &context->device_id, optionss.str().c_str(), 0, 0);
    if (err == CL_SUCCESS)
    {
      kernel->kernel = clCreateKernel(kernel->program, "fraktaler3", &err);
      if (kernel->kernel)
      {
        kernel->config_bytes = sizeof(config_cl<ufloat>);
        kernel->config_device = clCreateBuffer(context->context, CL_MEM_READ_ONLY, kernel->config_bytes, 0, &err);
        if (kernel->config_device)
        {
          kernel->config_host = std::calloc(1, kernel->config_bytes);
          if (kernel->config_host)
          {
            bool ok = true;
            switch (nt)
            {
              case nt_float: opencl_initialize_config((config_cl<float>*)(kernel->config_host), nt, par); break;
              case nt_double: opencl_initialize_config((config_cl<double>*)(kernel->config_host), nt, par); break;
              case nt_floatexp: opencl_initialize_config((config_cl<floatexp<float, int>>*)(kernel->config_host), nt, par); break;
              case nt_doubleexp: opencl_initialize_config((config_cl<floatexp<double, int>>*)(kernel->config_host), nt, par); break;
              case nt_softfloat: opencl_initialize_config((config_cl<softfloat>*)(kernel->config_host), nt, par); break;
              default: ok = false; break; // unreachable
            }
            if (ok)
            {
              kernel->reference_count = 1;
              context->kernels.push_back(kernel);
              delete[] source;
              return kernel;
            }
            else
            {
              std::free(kernel->config_host);
              kernel->config_host = 0;
            }
          }
          else
          {
            clReleaseMemObject(kernel->config_device);
            kernel->config_device = 0;
          }
        }
        else
        {
          std::fprintf(stderr, "CL ERROR %d %s %d\n", err, __FUNCTION__, __LINE__);
          clReleaseKernel(kernel->kernel);
          kernel->kernel = 0;
        }
      }
      else
      {
        std::fprintf(stderr, "CL ERROR %d %s %d\n", err, __FUNCTION__, __LINE__);
        clReleaseProgram(kernel->program);
        kernel->program = 0;
      }
    }
    else
    {
      char *error_log = new char[1000000];
      error_log[0] = 0;
      err = clGetProgramBuildInfo(kernel->program, context->device_id, CL_PROGRAM_BUILD_LOG, 1000000, &error_log[0], 0);
      std::fprintf(stderr, "error: OpenCL %d.%d source code:\n%s\n", context->platform, context->device, source);
      if (err == CL_SUCCESS)
      {
        std::fprintf(stderr, "error: OpenCL %d.%d compile failed:\n%s\n", context->platform, context->device, error_log);
      }
      else
      {
        std::fprintf(stderr, "error: OpenCL %d.%d compile failed:\n%s\n", context->platform, context->device, "(could not retrieve error log)");
      }
      delete[] error_log;
    }
  }
  delete[] source;
  delete kernel;
  return nullptr;
}

void opencl_release_kernel(opencl_context *context, opencl_kernel *kernel)
{
  (void) context;
  std::lock_guard<std::mutex> lock(cache_mutex);
  kernel->reference_count--;
}

void *opencl_kernel_config_host(opencl_kernel *kernel)
{
  return kernel->config_host;
}

bool opencl_set_kernel_arguments(opencl_context *context, opencl_kernel *kernel)
{
  bool ok = true;
  ok &= CL_SUCCESS == clSetKernelArg(kernel->kernel, 0, sizeof(cl_mem), &kernel->config_device);
  ok &= CL_SUCCESS == clSetKernelArg(kernel->kernel, 1, sizeof(cl_mem), &context->buffers.ref_device);
  ok &= CL_SUCCESS == clSetKernelArg(kernel->kernel, 2, sizeof(cl_mem), &context->buffers.bla_device);
  ok &= CL_SUCCESS == clSetKernelArg(kernel->kernel, 3, sizeof(cl_mem), context->buffers.RGB_device ? &context->buffers.RGB_device : nullptr);
  ok &= CL_SUCCESS == clSetKernelArg(kernel->kernel, 4, sizeof(cl_mem), context->buffers.N0_device  ? &context->buffers.N0_device  : nullptr);
  ok &= CL_SUCCESS == clSetKernelArg(kernel->kernel, 5, sizeof(cl_mem), context->buffers.N1_device  ? &context->buffers.N1_device  : nullptr);
  ok &= CL_SUCCESS == clSetKernelArg(kernel->kernel, 6, sizeof(cl_mem), context->buffers.NF_device  ? &context->buffers.NF_device  : nullptr);
  ok &= CL_SUCCESS == clSetKernelArg(kernel->kernel, 7, sizeof(cl_mem), context->buffers.T_device   ? &context->buffers.T_device   : nullptr);
  ok &= CL_SUCCESS == clSetKernelArg(kernel->kernel, 8, sizeof(cl_mem), context->buffers.DEX_device ? &context->buffers.DEX_device : nullptr);
  ok &= CL_SUCCESS == clSetKernelArg(kernel->kernel, 9, sizeof(cl_mem), context->buffers.DEY_device ? &context->buffers.DEY_device : nullptr);
  ok &= CL_SUCCESS == clSetKernelArg(kernel->kernel,10, sizeof(cl_mem), context->buffers.BLA_device ? &context->buffers.BLA_device : nullptr);
  ok &= CL_SUCCESS == clSetKernelArg(kernel->kernel,11, sizeof(cl_mem), context->buffers.PTB_device ? &context->buffers.PTB_device : nullptr);
  if (! ok)
  {
    std::fprintf(stderr, "CL ERROR %d %s %d\n", -1, __FUNCTION__, __LINE__);
  }
  return ok;
}

template<typename T>
bool opencl_upload_config(cl_command_queue command_queue, config_cl<T> *config_host, cl_event &ready, cl_mem config_device)
{
  cl_int err = clEnqueueWriteBuffer(command_queue, config_device, CL_FALSE, 0, sizeof(*config_host), config_host, 0, 0, &ready);
  if (err)
  {
    std::fprintf(stderr, "CL ERROR %d %s %d\n", err, __FUNCTION__, __LINE__);
  }
  return CL_SUCCESS == err;
}

bool opencl_upload_config(opencl_context *context, opencl_kernel *kernel)
{
  switch (kernel->nt)
  {
    case nt_float: return opencl_upload_config(context->command_queue, (config_cl<float> *) kernel->config_host, context->ready, kernel->config_device);
    case nt_double: return opencl_upload_config(context->command_queue, (config_cl<double> *) kernel->config_host, context->ready, kernel->config_device);
    case nt_floatexp: return opencl_upload_config(context->command_queue, (config_cl<floatexp<float, int>> *) kernel->config_host, context->ready, kernel->config_device);
    case nt_doubleexp: return opencl_upload_config(context->command_queue, (config_cl<floatexp<double, int>> *) kernel->config_host, context->ready, kernel->config_device);
    case nt_softfloat: return opencl_upload_config(context->command_queue, (config_cl<softfloat> *) kernel->config_host, context->ready, kernel->config_device);
    default: return false;
  }
}

template <typename T>
size_t opencl_ref_layout(config_cl<T> *config_host, const std::vector<std::vector<complex<T>>> &Z)
{
  cl_long ref_size = 0;
  for (int phase = 0; phase < config_host->number_of_phases; ++phase)
  {
    ref_size += config_host->ref_size[phase] = Z[phase].size();
  }
  cl_long ref_bytes = 2 * ref_size * sizeof(T);
  config_host->ref_start[0] = 0;
  for (int phase = 1; phase < config_host->number_of_phases; ++phase)
  {
    config_host->ref_start[phase] = config_host->ref_start[phase - 1] + 2 * config_host->ref_size[phase - 1];
  }
  return ref_bytes;
}

size_t opencl_ref_layout(void *config_host, number_type nt)
{
  switch (nt)
  {
    case nt_float: return opencl_ref_layout((config_cl<float>*) config_host, Zf);
    case nt_double: return opencl_ref_layout((config_cl<double>*) config_host, Zd);
    case nt_floatexp: return opencl_ref_layout((config_cl<floatexp<float, int>>*) config_host, Zfe);
    case nt_doubleexp: return opencl_ref_layout((config_cl<floatexp<double, int>>*) config_host, Zde);
    case nt_softfloat: return opencl_ref_layout((config_cl<softfloat>*) config_host, Zsf);
    default: return 0;
  }
}

template <typename T>
bool opencl_upload_ref(cl_command_queue command_queue, config_cl<T> *config_host, cl_event &ready, cl_mem ref_device, const std::vector<std::vector<complex<T>>> &Z)
{
  for (int phase = 0; phase < config_host->number_of_phases; ++phase)
  {
    const cl_long start_bytes = config_host->ref_start[phase] * sizeof(T);
    const cl_long size_bytes = config_host->ref_size[phase] * 2 * sizeof(T);
    const void *ptr = &Z[phase][0];
    cl_event done;
    cl_int err;
    if (CL_SUCCESS != (err = clEnqueueWriteBuffer(command_queue, ref_device, CL_FALSE, start_bytes, size_bytes, ptr, 1, &ready, &done)))
    {
      std::fprintf(stderr, "CL ERROR %d %s %d\n", err, __FUNCTION__, __LINE__);
      return false;
    }
    clReleaseEvent(ready);
    ready = done;
  }
 return true;
}

bool opencl_upload_ref(cl_command_queue command_queue, void *config_host, cl_event &ready, cl_mem ref_device, number_type nt)
{
  switch (nt)
  {
    case nt_float: return opencl_upload_ref(command_queue, (config_cl<float>*) config_host, ready, ref_device, Zf);
    case nt_double: return opencl_upload_ref(command_queue, (config_cl<double>*) config_host, ready, ref_device, Zd);
    case nt_floatexp: return opencl_upload_ref(command_queue, (config_cl<floatexp<float, int>>*) config_host, ready, ref_device, Zfe);
    case nt_doubleexp: return opencl_upload_ref(command_queue, (config_cl<floatexp<double, int>>*) config_host, ready, ref_device, Zde);
    case nt_softfloat: return opencl_upload_ref(command_queue, (config_cl<softfloat>*) config_host, ready, ref_device, Zsf);
    default: return false;
  }
}

bool opencl_upload_ref(opencl_context *context, opencl_kernel *kernel)
{
  return opencl_upload_ref(context->command_queue, kernel->config_host, context->ready, context->buffers.ref_device, kernel->nt);
}

template <typename T>
size_t opencl_bla_layout(config_cl<T> *config_host, const std::vector<blasR2<T>> &B)
{
  for (int phase = 0; phase < config_host->number_of_phases; ++phase)
  {
    config_host->bla_size[phase] = B[phase].M;
    config_host->bla_levels[phase] = B[phase].L;
  }
  cl_long bla_start = 0;
  for (int phase = 0; phase < config_host->number_of_phases; ++phase)
  {
    for (int level = 0; level < config_host->bla_levels[phase]; ++level)
    {
      config_host->bla_start[phase][level] = bla_start;
      bla_start += config_host->bla_size_level[phase][level] = B[phase].b[level].size();
    }
  }
  cl_long bla_bytes = bla_start * sizeof(struct blaR2_cl<T>);
  return bla_bytes;
}

size_t opencl_bla_layout(void *config_host, number_type nt)
{
  switch (nt)
  {
    case nt_float: return opencl_bla_layout((config_cl<float>*) config_host, Bf);
    case nt_double: return opencl_bla_layout((config_cl<double>*) config_host, Bd);
    case nt_floatexp: return opencl_bla_layout((config_cl<floatexp<float, int>>*) config_host, Bfe);
    case nt_doubleexp: return opencl_bla_layout((config_cl<floatexp<double, int>>*) config_host, Bde);
    case nt_softfloat: return opencl_bla_layout((config_cl<softfloat>*) config_host, Bsf);
    default: return 0;
  }
}

template <typename T>
bool opencl_upload_bla(cl_command_queue command_queue, config_cl<T> *config_host, cl_event &ready, cl_mem bla_device, const std::vector<blasR2<T>> &B)
{
  for (int phase = 0; phase < config_host->number_of_phases; ++phase)
  {
    for (int level = 0; level < config_host->bla_levels[phase]; ++level)
    {
      const cl_long start_bytes = config_host->bla_start[phase][level] * sizeof(blaR2_cl<T>);
      const cl_long size_bytes = B[phase].b[level].size() * sizeof(blaR2_cl<T>);
      const void *ptr = &B[phase].b[level][0];
      if (size_bytes)
      {
        cl_event done;
        cl_int err;
        if (CL_SUCCESS != (err = clEnqueueWriteBuffer(command_queue, bla_device, CL_FALSE, start_bytes, size_bytes, ptr, 1, &ready, &done)))
        {
          std::fprintf(stderr, "CL ERROR %d %s %d\n", err, __FUNCTION__, __LINE__);
          return false;
        }
        clReleaseEvent(ready);
        ready = done;
      }
    }
  }
  return true;
}

bool opencl_upload_bla(cl_command_queue command_queue, void *config_host, cl_event &ready, cl_mem bla_device, number_type nt)
{
  switch (nt)
  {
    case nt_float: return opencl_upload_bla(command_queue, (config_cl<float>*) config_host, ready, bla_device, Bf);
    case nt_double: return opencl_upload_bla(command_queue, (config_cl<double>*) config_host, ready, bla_device, Bd);
    case nt_floatexp: return opencl_upload_bla(command_queue, (config_cl<floatexp<float, int>>*) config_host, ready, bla_device, Bfe);
    case nt_doubleexp: return opencl_upload_bla(command_queue, (config_cl<floatexp<double, int>>*) config_host, ready, bla_device, Bde);
    case nt_softfloat: return opencl_upload_bla(command_queue, (config_cl<softfloat>*) config_host, ready, bla_device, Bsf);
    default: return false;
  }
}

bool opencl_upload_bla(opencl_context *context, opencl_kernel *kernel)
{
  return opencl_upload_bla(context->command_queue, kernel->config_host, context->ready, context->buffers.bla_device, kernel->nt);
}

bool opencl_get_buffers(opencl_context *context, size_t ref_bytes, size_t bla_bytes, coord_t tile_width, coord_t tile_height, bool raw)
{
  bool no_raw = ! raw;
  size_t RGB_bytes = sizeof(float) * tile_width * tile_height * 3;
  size_t raw_bytes = no_raw ? 0 : sizeof(float) * tile_width * tile_height;
  size_t
    N0_bytes = raw_bytes,
    N1_bytes = raw_bytes,
    NF_bytes = raw_bytes,
    T_bytes = raw_bytes,
    DEX_bytes = raw_bytes,
    DEY_bytes = raw_bytes,
    BLA_bytes = raw_bytes,
    PTB_bytes = raw_bytes;
  // release buffers if they are too big or small
#define FREE(what,when) \
  if (context->buffers.what##_bytes < what##_bytes || context->buffers.what##_bytes > what##_bytes * 2 || when) \
  { \
    if (context->buffers.what##_device) \
    { \
      clReleaseMemObject(context->buffers.what##_device); \
    } \
    context->buffers.what##_device = 0; \
    context->buffers.what##_bytes = 0; \
  }
  FREE(ref, false)
  FREE(bla, false)
  FREE(RGB, false)
  FREE(N0,  no_raw)
  FREE(N1,  no_raw)
  FREE(NF,  no_raw)
  FREE(T,   no_raw)
  FREE(DEX, no_raw)
  FREE(DEY, no_raw)
  FREE(BLA, no_raw)
  FREE(PTB, no_raw)
#undef FREE
  // allocate new buffers if necessary
#define ALLOC(what,mode) \
  if (! context->buffers.what##_device && what##_bytes > 0) \
  { \
    cl_int err; \
    context->buffers.what##_device = clCreateBuffer(context->context, mode, what##_bytes, 0, &err); \
    if (context->buffers.what##_device) \
    { \
      context->buffers.what##_bytes = what##_bytes; \
    } \
    else \
    { \
      return false; \
    } \
  }
  ALLOC(ref, CL_MEM_READ_ONLY)
  ALLOC(bla, CL_MEM_READ_ONLY)
  ALLOC(RGB, CL_MEM_WRITE_ONLY)
  ALLOC(N0,  CL_MEM_WRITE_ONLY)
  ALLOC(N1,  CL_MEM_WRITE_ONLY)
  ALLOC(NF,  CL_MEM_WRITE_ONLY)
  ALLOC(T,   CL_MEM_WRITE_ONLY)
  ALLOC(DEX, CL_MEM_WRITE_ONLY)
  ALLOC(DEY, CL_MEM_WRITE_ONLY)
  ALLOC(BLA, CL_MEM_WRITE_ONLY)
  ALLOC(PTB, CL_MEM_WRITE_ONLY)
#undef ALLOC
  context->buffers.tile_width = tile_width;
  context->buffers.tile_height = tile_height;
  return true;
}

void opencl_render_tile(opencl_context *context, opencl_kernel *kernel, coord_t x, coord_t y, coord_t subframe)
{
  cl_long y0 = y * context->buffers.tile_height;
  cl_long x0 = x * context->buffers.tile_width;
  cl_long sub = subframe;
  clSetKernelArg(kernel->kernel, 12, sizeof(cl_long), &y0);
  clSetKernelArg(kernel->kernel, 13, sizeof(cl_long), &x0);
  clSetKernelArg(kernel->kernel, 14, sizeof(cl_long), &sub);
  cl_event done;
  size_t global[2] = { (size_t) context->buffers.tile_height, (size_t) context->buffers.tile_width };
  cl_int err;
  if (CL_SUCCESS != (err = clEnqueueNDRangeKernel(context->command_queue, kernel->kernel, 2, 0, global, 0, 1, &context->ready, &done)))
  {
    std::fprintf(stderr, "CL ERROR %d %s %d\n", err, __FUNCTION__, __LINE__);
  }
  clReleaseEvent(context->ready);
  context->ready = done;
}

#define MAP_BUFFER 1

tile *opencl_map_tile(opencl_context *context)
{
  std::memset(&context->buffers.tile_host, 0, sizeof(context->buffers.tile_host));
  context->buffers.tile_host.width = context->buffers.tile_width;
  context->buffers.tile_host.height = context->buffers.tile_height;
#define MAP(c, t) \
  if (context->buffers.c##_device) \
  { \
    cl_event done; \
    cl_int err; \
    if (MAP_BUFFER) \
    { \
      context->buffers.tile_host.c = (t* ) clEnqueueMapBuffer(context->command_queue, context->buffers.c##_device, CL_FALSE, CL_MAP_READ, 0, context->buffers.c##_bytes, 1, &context->ready, &done, &err); \
    } \
    else \
    { \
      context->buffers.tile_host.c = (t *) std::malloc(context->buffers.c##_bytes); \
      err = clEnqueueReadBuffer(context->command_queue, context->buffers.c##_device, CL_FALSE, 0, context->buffers.c##_bytes, context->buffers.tile_host.c, 1, &context->ready, &done); \
    } \
    if (err != CL_SUCCESS) \
    { \
      std::fprintf(stderr, "CL ERROR %d %s %d\n", err, __FUNCTION__, __LINE__); \
    } \
    clReleaseEvent(context->ready); \
    context->ready = done; \
  }
  MAP(RGB, float)
  MAP(N0,  uint32_t)
  MAP(N1,  uint32_t)
  MAP(NF,  float)
  MAP(T,   float)
  MAP(DEX, float)
  MAP(DEY, float)
  MAP(BLA, uint32_t)
  MAP(PTB, uint32_t)
#undef MAP
  clWaitForEvents(1, &context->ready);
  return &context->buffers.tile_host;
}

void opencl_unmap_tile(opencl_context *context)
{
#define UNMAP(c) \
  if (context->buffers.c##_device && context->buffers.tile_host.c) \
  { \
    if (MAP_BUFFER) \
    { \
      cl_event done; \
      clEnqueueUnmapMemObject(context->command_queue, context->buffers.c##_device, context->buffers.tile_host.c, 1, &context->ready, &done); \
      clReleaseEvent(context->ready); \
      context->ready = done; \
    } \
    else \
    { \
      std::free(context->buffers.tile_host.c); \
      context->buffers.tile_host.c = nullptr; \
    } \
  }
  UNMAP(RGB)
  UNMAP(N0)
  UNMAP(N1)
  UNMAP(NF)
  UNMAP(T)
  UNMAP(DEX)
  UNMAP(DEY)
  UNMAP(BLA)
  UNMAP(PTB)
#undef UNMAP
  if (MAP_BUFFER)
  {
    clWaitForEvents(1, &context->ready);
  }
}

#undef MAP_BUFFER

void opencl_clear_cache() // FIXME
{
}

#endif
