// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#ifdef HAVE_CL

#include "render.h"

struct opencl_context;
struct opencl_kernel;

opencl_context *opencl_get_context(int platform, int device);
void opencl_release_context(opencl_context *context);

opencl_kernel *opencl_get_kernel(opencl_context *context, number_type nt, const param &par);
void opencl_release_kernel(opencl_context *context, opencl_kernel *kernel);

void *opencl_kernel_config_host(opencl_kernel *kernel);

size_t opencl_ref_layout(void *config_host, number_type nt);
size_t opencl_bla_layout(void *config_host, number_type nt);
bool opencl_get_buffers(opencl_context *context, size_t ref_bytes, size_t bla_bytes, coord_t tile_width, coord_t tile_height, bool raw); // FIXME raw should be channel mask

bool opencl_upload_config(opencl_context *context, opencl_kernel *kernel);
bool opencl_upload_ref(opencl_context *context, opencl_kernel *kernel);
bool opencl_upload_bla(opencl_context *context, opencl_kernel *kernel);

bool opencl_set_kernel_arguments(opencl_context *context, opencl_kernel *kernel);

void opencl_render_tile(opencl_context *context, opencl_kernel *kernel, coord_t x, coord_t y, coord_t subframe);

tile *opencl_map_tile(opencl_context *context);
void opencl_unmap_tile(opencl_context *context);

void opencl_clear_cache();

#endif
