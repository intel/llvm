//==----- lsc_load_store_2d_compare.cpp - DPC++ ESIMD on-device test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// The tests makes sure old and new load_2d/store_2d API produce identical
// results.

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

int main() {
  constexpr uint32_t BlockWidth = 16;
  constexpr uint32_t BlockHeight = 4;
  constexpr uint32_t NumBlocks = 1;
  constexpr uint32_t SurfaceHeight = 16;
  constexpr uint32_t SurfaceWidth = 16;
  constexpr uint32_t SurfacePitch = 16;
  constexpr uint32_t x = 0;
  constexpr uint32_t y = 0;

  constexpr uint32_t Size = SurfacePitch * SurfaceHeight * NumBlocks;

  queue q;
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  auto *A = malloc_shared<float>(Size, q);
  auto *B = malloc_shared<float>(Size, q);
  auto *C = malloc_shared<float>(Size, q);
  auto *C1 = malloc_shared<float>(Size, q);

  for (auto i = 0; i != Size; i++) {
    A[i] = B[i] = 7;
    C[i] = C1[i] = 0;
  }

  auto e = q.submit([&](handler &cgh) {
    cgh.parallel_for<class Test>(range<1>{1}, [=](id<1> i) SYCL_ESIMD_KERNEL {
      constexpr uint32_t width = SurfaceWidth * sizeof(float) - 1;
      constexpr uint32_t height = SurfaceHeight - 1;
      constexpr uint32_t pitch = SurfacePitch * sizeof(float) - 1;
      auto data_a = lsc_load2d<float, BlockWidth, BlockHeight, NumBlocks>(
          A, width, height, pitch, x, y);
      auto data_b = lsc_load2d<float, BlockWidth, BlockHeight, NumBlocks>(
          B, width, height, pitch, x, y);

      auto data_c = data_a + data_b;

      lsc_store2d<float, BlockWidth, BlockHeight>(C, width, height, pitch, x, y,
                                                  data_c);
    });
  });
  e.wait();

  auto e1 = q.submit([&](handler &cgh) {
    cgh.parallel_for<class Test1>(range<1>{1}, [=](id<1> i) SYCL_ESIMD_KERNEL {
      constexpr uint32_t width = SurfaceWidth * sizeof(float) - 1;
      constexpr uint32_t height = SurfaceHeight - 1;
      constexpr uint32_t pitch = SurfacePitch * sizeof(float) - 1;

      config_2d_mem_access<float, BlockWidth, BlockHeight, NumBlocks> payload(
          A, width, height, pitch, 0, 0);
      auto data_a = lsc_load_2d(payload);

      payload.set_data_pointer(B);
      auto data_b = lsc_load_2d(payload);

      auto data_c = data_a + data_b;

      payload.set_data_pointer(C1);

      lsc_store_2d(payload, data_c);
    });
  });
  e1.wait();

  auto error = 0;
  for (auto i = 0; i < Size; ++i)
    error += std::abs(C[i] - C1[i]);

  free(A, q);
  free(B, q);
  free(C, q);
  free(C1, q);
  std::cout << (error != 0 ? "FAILED" : "passed") << std::endl;
  return 0;
}
