// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#include "exr.h"

#ifdef HAVE_EXR

static const char kf2plus[] = "KallesFraktaler2+";
static const char fraktaler3[] = "Fraktaler3";

#endif

#include "image_raw.h"
#include "render.h"

image_raw::image_raw(coord_t width, coord_t height, channel_mask_t channels)
: image(width, height)
, channels(channels)
, R(nullptr)
, G(nullptr)
, B(nullptr)
, N0(nullptr)
, N1(nullptr)
, NF(nullptr)
, T(nullptr)
, DEX(nullptr)
, DEY(nullptr)
, BLA(nullptr)
, PTB(nullptr)
{
  if (channels & (1 << Channel_R))   R   = new float[width * height];
  if (channels & (1 << Channel_G))   G   = new float[width * height];
  if (channels & (1 << Channel_B))   B   = new float[width * height];
  if (channels & (1 << Channel_N0))  N0  = new uint32_t[width * height];
  if (channels & (1 << Channel_N1))  N1  = new uint32_t[width * height];
  if (channels & (1 << Channel_NF))  NF  = new float[width * height];
  if (channels & (1 << Channel_T))   T   = new float[width * height];
  if (channels & (1 << Channel_DEX)) DEX = new float[width * height];
  if (channels & (1 << Channel_DEY)) DEY = new float[width * height];
  if (channels & (1 << Channel_BLA)) BLA = new uint32_t[width * height];
  if (channels & (1 << Channel_PTB)) PTB = new uint32_t[width * height];
}

image_raw::image_raw(image_raw &source, bool vflip)
: image_raw(source.width, source.height, source.channels)
{
  std::lock_guard<std::mutex> lock(source.mutex);
#define COPY(c) \
  if (c && source.c) \
  { \
    for (coord_t y = 0; y < height; ++y) \
    { \
      for (coord_t x = 0; x < width; ++x) \
      { \
        coord_t z = (y * width + x); \
        coord_t w = vflip ? ((height - 1 - y) * width + x) : z; \
        c[z] = source.c[w]; \
      } \
    } \
  }
  COPY(R)
  COPY(G)
  COPY(B)
  COPY(N0)
  COPY(N1)
  COPY(NF)
  COPY(T)
  COPY(DEX)
  COPY(DEY)
  COPY(BLA)
  COPY(PTB)
#undef COPY
}

image_raw::~image_raw()
{
  delete[] R;
  delete[] G;
  delete[] B;
  delete[] N0;
  delete[] N1;
  delete[] NF;
  delete[] T;
  delete[] DEX;
  delete[] DEY;
  delete[] BLA;
  delete[] PTB;
}

void image_raw::clear()
{
#define CLEAR(c) \
  if (c) std::memset(c, 0, width * height * sizeof(*c));
  CLEAR(R)
  CLEAR(G)
  CLEAR(B)
  CLEAR(N0)
  CLEAR(N1)
  CLEAR(NF)
  CLEAR(T)
  CLEAR(DEX)
  CLEAR(DEY)
  CLEAR(BLA)
  CLEAR(PTB)
#undef CLEAR
}

void image_raw::blit(coord_t tx, coord_t ty, const struct tile *t)
{
  std::lock_guard<std::mutex> lock(mutex); // FIXME tile based locking
  if (t && t->RGB && (R || G || B))
  {
    for (coord_t j = 0; j < t->height; ++j)
    {
      coord_t y = ty * t->height + j;
      if (0 <= y && y < height)
      {
        for (coord_t i = 0; i < t->width; ++i)
        {
          coord_t x = tx * t->width + i;
          if (0 <= x && x < width)
          {
            coord_t k = 3 * (j * t->width + i);
            coord_t z = y * width + x;
            if (R) R[z] = t->RGB[k+0];
            if (G) G[z] = t->RGB[k+1];
            if (B) B[z] = t->RGB[k+2];
          }
        }
      }
    }
  }
#define BLIT(c) \
  if (t && t->c && c) \
  { \
    for (coord_t j = 0; j < t->height; ++j) \
    { \
      coord_t y = ty * t->height + j; \
      if (0 <= y && y < height) \
      { \
        for (coord_t i = 0; i < t->width; ++i) \
        { \
          coord_t x = tx * t->width + i; \
          if (0 <= x && x < width) \
          { \
            coord_t k = j * t->width + i; \
            coord_t z = y * width + x; \
            c[z] = t->c[k]; \
          } \
        } \
      } \
    } \
  }
  BLIT(N0)
  BLIT(N1)
  BLIT(NF)
  BLIT(T)
  BLIT(DEX)
  BLIT(DEY)
  BLIT(BLA)
  BLIT(PTB)
#undef BLIT
}

bool image_raw::save_exr(const std::string &filename, channel_mask_t save_channels, count_t maxiters, int threads, const std::string &metadata, int dpi)
{
  std::lock_guard<std::mutex> lock(mutex);
  if (save_channels & ~ channels) // FIXME requested channel cannot be saved
  {
    // ...
  }
#ifndef HAVE_EXR
  (void) filename;
  (void) maxiters;
  (void) threads;
  (void) metadata;
  (void) dpi;
#else
  try
  {
    setGlobalThreadCount(threads);
    Header header(width, height);
    if (metadata != "") header.insert(fraktaler3, StringAttribute(metadata));
    const count_t Nbias = 1024;
    bool twoN = maxiters + Nbias >= 0xFFffFFfeU;
    if (maxiters + Nbias < INT_MAX - 1)
    {
      header.insert("Iterations", IntAttribute(maxiters));
    }
    else
    {
      std::ostringstream s;
      s << maxiters;
      header.insert("Iterations", StringAttribute(s.str()));
    }
    if (Nbias < INT_MAX - 1)
    {
      header.insert("IterationsBias", IntAttribute(Nbias));
    }
    else
    {
      std::ostringstream s;
      s << Nbias;
      header.insert("IterationsBias", StringAttribute(s.str()));
    }
    header.insert("xDensity", FloatAttribute(dpi));
    FrameBuffer fb;
#define CHAN(c,ignored,t) \
    if (c && (save_channels & (1 << Channel_##c))) \
    { \
      header.channels().insert(#c, Channel(IMF::t)); \
      fb.insert(#c, Slice(IMF::t, (char *)(c), sizeof(*c), sizeof(*c) * width)); \
    }
    CHAN(R,   HALF,  FLOAT)
    CHAN(G,   HALF,  FLOAT)
    CHAN(B,   HALF,  FLOAT)
    if (twoN)
    {
      CHAN(N0,  UINT,  UINT)
      CHAN(N1,  UINT,  UINT)
    }
    else
    {
      if (N0 && (save_channels & (1 << Channel_N0)))
      {
        header.channels().insert("N", Channel(IMF::UINT));
        fb.insert("N", Slice(IMF::UINT, (char *)(N0), sizeof(*N0), sizeof(*N0) * width));
      }
    }
    CHAN(NF,  FLOAT, FLOAT)
    CHAN(T,   FLOAT, FLOAT)
    CHAN(DEX, FLOAT, FLOAT)
    CHAN(DEY, FLOAT, FLOAT)
    CHAN(BLA, UINT,  UINT)
    CHAN(PTB, UINT,  UINT)
#undef CHAN
    OutputFile of(filename.c_str(), header);
    of.setFrameBuffer(fb);
    of.writePixels(height);
    return true;
  }
  catch (...)
#endif
  {
    return false;
  }
}
