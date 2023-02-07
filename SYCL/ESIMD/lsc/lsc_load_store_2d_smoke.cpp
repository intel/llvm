//==------- lsc_load_store_2d_smoke.cpp - DPC++ ESIMD on-device test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Basic test for new lsc_load_2d/lsc_store_2d API.

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

int main() {
  using namespace sycl;
  using namespace sycl::ext::intel::esimd;
  using namespace sycl::ext::intel::experimental::esimd;
  constexpr uint32_t SurfaceHeight = 4;
  constexpr uint32_t SurfaceWidth = 9;
  constexpr uint32_t SurfacePitch = 16;
  constexpr uint32_t x = 0;
  constexpr uint32_t y = 0;
  constexpr uint32_t BlockWidth = 4;
  constexpr uint32_t BlockHeight = 4;
  constexpr uint32_t NumBlocks = 1;

  constexpr uint32_t Size = SurfaceHeight * SurfacePitch;

  queue q;
  auto device = q.get_device();
  std::cout << "Device name: " << device.get_info<sycl::info::device::name>()
            << std::endl;

  auto *input = malloc_shared<int>(Size, q);
  std::iota(input, input + Size, 0);

  auto *block_store = malloc_shared<int>(Size, q);

  auto *ref = new int[Size];

  for (int i = 0; i < Size; i++)
    block_store[i] = ref[i] = rand() % 128;

  for (int i = 0; i < BlockHeight; i++) {
    for (int j = 0; j < BlockWidth; j++) {
      ref[y * SurfacePitch + i * SurfacePitch + x + j] =
          input[y * SurfacePitch + i * SurfacePitch + x + j];
    }
  }
  try {
    q.submit([&](handler &h) {
      h.parallel_for<class SimplestKernel>(
          range<1>{1}, [=](id<1> id) SYCL_ESIMD_KERNEL {
            config_2d_mem_access<int, BlockWidth, BlockHeight, NumBlocks>
                payload(input, (SurfaceWidth * sizeof(int)) - 1,
                        SurfaceHeight - 1, (SurfacePitch * sizeof(int)) - 1, x,
                        y);

            auto data = lsc_load_2d(payload);

            payload.set_data_pointer(block_store);

            lsc_store_2d(payload, data);
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
  for (auto i = 0; i < Size; ++i)
    if (ref[i] != block_store[i])
      error++;

  free(input, q);
  free(block_store, q);
  std::cout << (error != 0 ? "FAILED" : "passed") << std::endl;
  return error;
}
