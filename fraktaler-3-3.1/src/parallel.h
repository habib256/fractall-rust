// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <atomic>
#include <thread>
#include <vector>

#include "types.h"

template <typename result, typename F>
result parallel1dr(int max_threads, coord_t x0, coord_t x1, coord_t xn, volatile bool *running, F function)
{
  const coord_t buckets = (x1 - x0 + xn - 1) / xn;
  const int threads = std::min(coord_t(max_threads), buckets);
  std::atomic<coord_t> next_bucket {0};
  std::vector<std::thread> workers;
  std::vector<result> results;
  for (int t = 0; t < threads; ++t)
  {
    results.push_back(result());
  }
  for (int t = 0; t < threads; ++t)
  {
    workers.push_back(std::thread([&, t](){
      result r;
      try
      {
        while (*running)
        {
          const coord_t bucket = next_bucket.fetch_add(1);
          if (bucket >= buckets)
          {
            break;
          }
          const coord_t my_x0 = x0 + bucket * xn;
          const coord_t my_x1 = std::min(my_x0 + xn, x1);
          for (coord_t x = my_x0; x < my_x1; ++x)
          {
            r += function(x);
          }
        }
      }
      catch (const std::exception &e)
      {
        std::cerr << "parallel1dr worker exception " << e.what() << std::endl;
      }
      results[t] = r;
    }));
  }
  for (int t = 0; t < threads; ++t)
  {
    workers[t].join();
  }
  result r;
  for (int t = 0; t < threads; ++t)
  {
    r += results[t];
  }
  return r;
}

template <typename result, typename F>
result parallel2dr(int max_threads, coord_t x0, coord_t x1, coord_t xn, coord_t y0, coord_t y1, coord_t yn, volatile bool *running, F function)
{
  const coord_t buckets_x = (x1 - x0 + xn - 1) / xn;
  const coord_t buckets_y = (y1 - y0 + yn - 1) / yn;
  const coord_t buckets = buckets_x * buckets_y;
  const int threads = std::min(coord_t(max_threads), buckets);
  std::atomic<coord_t> next_bucket {0};
  std::vector<std::thread> workers;
  std::vector<result> results;
  for (int t = 0; t < threads; ++t)
  {
    results.push_back(result());
  }
  for (int t = 0; t < threads; ++t)
  {
    workers.push_back(std::thread([&, t]() -> void{
      result r;
      try
      {
        while (*running)
        {
          const coord_t bucket = next_bucket.fetch_add(1);
          if (bucket >= buckets)
          {
            break;
          }
          const coord_t bucket_x = bucket / buckets_y;
          const coord_t bucket_y = bucket - buckets_y * bucket_x;
          const coord_t my_x0 = x0 + bucket_x * xn;
          const coord_t my_x1 = std::min(my_x0 + xn, x1);
          const coord_t my_y0 = y0 + bucket_y * yn;
          const coord_t my_y1 = std::min(my_y0 + yn, y1);
          for (coord_t y = my_y0; y < my_y1; ++y)
          {
            for (coord_t x = my_x0; x < my_x1; ++x)
            {
              r += function(x, y);
            }
          }
        }
      }
      catch (const std::exception &e)
      {
        std::cerr << "parallel2dr worker exception " << e.what() << std::endl;
      }
      results[t] = r;
    }));
  }
  for (auto & t : workers)
  {
    t.join();
  }
  result r;
  for (int t = 0; t < threads; ++t)
  {
    r += results[t];
  }
  return r;
}

template <typename F>
void parallel1d(int max_threads, coord_t x0, coord_t x1, coord_t xn, volatile bool *running, F function)
{
  const coord_t buckets = (x1 - x0 + xn - 1) / xn;
  const int threads = std::min(coord_t(max_threads), buckets);
  std::atomic<coord_t> next_bucket {0};
  std::vector<std::thread> workers;
  for (int t = 0; t < threads; ++t)
  {
    workers.push_back(std::thread([&](){
      try
      {
        while (*running)
        {
          const coord_t bucket = next_bucket.fetch_add(1);
          if (bucket >= buckets)
          {
            break;
          }
          const coord_t my_x0 = x0 + bucket * xn;
          const coord_t my_x1 = std::min(my_x0 + xn, x1);
          for (coord_t x = my_x0; x < my_x1; ++x)
          {
            function(x);
          }
        }
      }
      catch (const std::exception &e)
      {
        std::cerr << "parallel1d worker exception " << e.what() << std::endl;
      }
    }));
  }
  for (int t = 0; t < threads; ++t)
  {
    workers[t].join();
  }
}

template <typename F>
void parallel2d(int max_threads, coord_t x0, coord_t x1, coord_t xn, coord_t y0, coord_t y1, coord_t yn, volatile bool *running, F function)
{
  const coord_t buckets_x = (x1 - x0 + xn - 1) / xn;
  const coord_t buckets_y = (y1 - y0 + yn - 1) / yn;
  const coord_t buckets = buckets_x * buckets_y;
  const int threads = std::min(coord_t(max_threads), buckets);
  std::atomic<coord_t> next_bucket {0};
  std::vector<std::thread> workers;
  for (int t = 0; t < threads; ++t)
  {
    workers.push_back(std::thread([&](){
      try
      {
        while (*running)
        {
          const coord_t bucket = next_bucket.fetch_add(1);
          if (bucket >= buckets)
          {
            break;
          }
          const coord_t bucket_x = bucket / buckets_y;
          const coord_t bucket_y = bucket - buckets_y * bucket_x;
          const coord_t my_x0 = x0 + bucket_x * xn;
          const coord_t my_x1 = std::min(my_x0 + xn, x1);
          const coord_t my_y0 = y0 + bucket_y * yn;
          const coord_t my_y1 = std::min(my_y0 + yn, y1);
          for (coord_t y = my_y0; y < my_y1; ++y)
          {
            for (coord_t x = my_x0; x < my_x1; ++x)
            {
              function(x, y);
            }
          }
        }
      }
      catch (const std::exception &e)
      {
        std::cerr << "parallel2d worker exception " << e.what() << std::endl;
      }
    }));
  }
  for (int t = 0; t < threads; ++t)
  {
    workers[t].join();
  }
}
