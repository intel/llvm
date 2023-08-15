//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#pragma once
#include "external/BS_thread_pool.hpp"

// Macro to parallelize a loop.
#define PARALLEL_FOR(tp, func, lower, upper)                                   \
  {                                                                            \
    int NumWorkers = tp.get_thread_count();                                    \
    int Step = (upper - lower + 1) / NumWorkers;                               \
    for (int i = 0; i < NumWorkers; ++i) {                                     \
      int Min = lower + i * Step;                                              \
      int Max = std::min<int>((lower + (i + 1) * Step), upper);                \
      auto Ret = tp.submit(func, Min, Max);                                    \
    }                                                                          \
    tp.wait_for_tasks();                                                       \
  }
