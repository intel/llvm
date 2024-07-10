//==------------ lsc_usm_2d.cpp - DPC++ ESIMD on-device test ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "../esimd_test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

int main() {
  using namespace sycl;
  using namespace sycl::ext::intel::esimd;
  using namespace sycl::ext::intel::experimental::esimd;
  unsigned data_height = 4;
  unsigned data_width = 9;
  unsigned data_pitch = 16;
  unsigned x = 0;
  unsigned y = 0;
  unsigned size = data_height * data_pitch;

  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  auto device = q.get_device();
  std::cout << "Device name: " << device.get_info<sycl::info::device::name>()
            << std::endl;

  auto *input = malloc_shared<int>(size, q);
  std::iota(input, input + size, 0);

  constexpr unsigned Width = 4;
  constexpr unsigned Height = 4;
  constexpr unsigned NumBlocks = 1;
  auto *block_store = malloc_shared<int>(size, q);

  auto *ref = new int[size];
  // Fill dst and ref data which is untouched with random values
  for (int i = 0; i < size; i++)
    block_store[i] = ref[i] = rand() % 128;

  for (int i = 0; i < Height; i++) {
    for (int j = 0; j < Width; j++) {
      ref[y * data_pitch + i * data_pitch + x + j] =
          input[y * data_pitch + i * data_pitch + x + j];
    }
  }
  try {
    q.submit([&](handler &h) {
      h.parallel_for<class SimplestKernel>(
          range<1>{1}, [=](id<1> id) SYCL_ESIMD_KERNEL {
            lsc_prefetch_2d<int, Width, Height, NumBlocks, cache_hint::cached,
                            cache_hint::uncached>(
                input, (data_width * sizeof(int)) - 1, data_height - 1,
                (data_pitch * sizeof(int)) - 1, x, y);
            auto data = lsc_load_2d<int, Width, Height, NumBlocks, false, false,
                                    cache_hint::uncached, cache_hint::uncached>(
                input, (data_width * sizeof(int)) - 1, data_height - 1,
                (data_pitch * sizeof(int)) - 1, x, y);
            lsc_store_2d<int, Width, Height, cache_hint::uncached,
                         cache_hint::uncached>(
                block_store, (data_width * sizeof(int)) - 1, data_height - 1,
                (data_pitch * sizeof(int)) - 1, x, y, data);
          });
    });
    q.wait();
  } catch (sycl::exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    free(input, q);
    free(block_store, q);
    return 1;
  }

  auto error = 0;
  for (auto i = 0; i < size; ++i)
    error += std::abs(ref[i] - block_store[i]);
  free(input, q);
  free(block_store, q);
  std::cout << (error != 0 ? "FAILED" : "passed") << std::endl;
  return error;
}
