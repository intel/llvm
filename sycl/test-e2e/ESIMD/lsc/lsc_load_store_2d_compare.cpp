//==----- lsc_load_store_2d_compare.cpp - DPC++ ESIMD on-device test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The tests makes sure old and new load_2d/store_2d API produce identical
// results.

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

template <typename T> bool test() {
  constexpr uint32_t BlockWidth = 8;
  constexpr uint32_t BlockHeight = 8;
  constexpr uint32_t NumBlocks = 1;
  constexpr uint32_t SurfaceHeight = BlockHeight;
  constexpr uint32_t SurfaceWidth = BlockWidth * NumBlocks;
  constexpr uint32_t SurfacePitch = BlockWidth;
  constexpr uint32_t x = 0;
  constexpr uint32_t y = 0;

  constexpr uint32_t Size = SurfaceWidth * SurfaceHeight;

  queue q;
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  auto *A = malloc_shared<T>(Size, q);
  auto *B = malloc_shared<T>(Size, q);
  auto *C = malloc_shared<T>(Size, q);
  auto *C1 = malloc_shared<T>(Size, q);

  for (auto i = 0; i != Size; i++) {
    A[i] = B[i] = i;
    C[i] = C1[i] = 0;
  }

  auto e = q.submit([&](handler &cgh) {
    cgh.parallel_for(range<1>{1}, [=](id<1> i) SYCL_ESIMD_KERNEL {
      constexpr uint32_t width = SurfaceWidth * sizeof(T) - 1;
      constexpr uint32_t height = SurfaceHeight - 1;
      constexpr uint32_t pitch = SurfacePitch * sizeof(T) - 1;
      simd<T, Size> data_a = lsc_load_2d<T, BlockWidth, BlockHeight, NumBlocks>(
          A, width, height, pitch, x, y);
      simd<T, Size> data_b = lsc_load_2d<T, BlockWidth, BlockHeight, NumBlocks>(
          B, width, height, pitch, x, y);

      simd<T, Size> data_c = data_a + data_b;

      lsc_store_2d<T, BlockWidth, BlockHeight>(C, width, height, pitch, x, y,
                                               data_c);
    });
  });
  e.wait();

  auto e1 = q.submit([&](handler &cgh) {
    cgh.parallel_for(range<1>{1}, [=](id<1> i) SYCL_ESIMD_KERNEL {
      constexpr uint32_t width = SurfaceWidth * sizeof(T) - 1;
      constexpr uint32_t height = SurfaceHeight - 1;
      constexpr uint32_t pitch = SurfacePitch * sizeof(T) - 1;

      config_2d_mem_access<T, BlockWidth, BlockHeight, NumBlocks> payload(
          A, width, height, pitch, 0, 0);
      lsc_prefetch_2d<T, BlockWidth, BlockHeight, NumBlocks, false, false,
                      cache_hint::cached, cache_hint::cached>(payload);
      simd<T, Size> data_a = lsc_load_2d(payload);

      payload.set_data_pointer(B);
      lsc_prefetch_2d<T, BlockWidth, BlockHeight, NumBlocks, false, false,
                      cache_hint::cached, cache_hint::cached>(payload);
      simd<T, Size> data_b = lsc_load_2d(payload);

      simd<T, Size> data_c = data_a + data_b;

      payload.set_data_pointer(C1);

      lsc_store_2d(payload, data_c);
    });
  });
  e1.wait();

  bool error = false;
  for (auto i = 0; i < Size; ++i)
    error |= C[i] != C1[i];

  free(A, q);
  free(B, q);
  free(C, q);
  free(C1, q);
  return error;
}

int main() {
  bool result = false;
  result |= test<float>();
  result |= test<uint32_t>();
  result |= test<uint16_t>();
  result |= test<uint64_t>();
  result |= test<double>();
  result |= test<uint8_t>();
  result |= test<sycl::half>();

  std::cout << (result ? "FAILED" : "passed") << std::endl;
  return 0;
}
