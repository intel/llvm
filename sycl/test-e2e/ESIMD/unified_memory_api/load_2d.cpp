//==----- load_2d.cpp - DPC++ ESIMD on-device test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// The test verifies esimd::load_2d() function accepting USM pointer
// and optional compile-time esimd::properties.
#include "../esimd_test_utils.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;

template <typename T, int BlockWidth, unsigned SurfaceWidth,
          unsigned SurfaceHeight, bool CheckProperties,
          typename LoadPropertiesT>
bool test(LoadPropertiesT LoadProperties) {
  sycl::queue Q(sycl::gpu_selector_v);
  esimd_test::printTestLabel(Q);

  constexpr int BlockHeight = 1;
  constexpr int NBlocks = 1;
  constexpr bool Transposed = false;
  constexpr bool Transformed = false;

  constexpr unsigned SurfacePitch = SurfaceWidth;

  auto *A = malloc_shared<T>(SurfaceWidth * SurfaceHeight, Q);
  auto *B = malloc_shared<T>(SurfaceWidth * SurfaceHeight, Q);

  for (int i = 0; i < SurfaceWidth * SurfaceHeight; i++) {
    A[i] = static_cast<T>(i);
  }

  sycl::range<2> GlobalRange = {SurfaceWidth / BlockWidth,
                                SurfaceHeight / BlockHeight};
  sycl::range<2> LocalRange = {1, 1};

  Q.parallel_for(
       sycl::nd_range<2>(GlobalRange, LocalRange),
       [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
         const auto x = item.get_global_id(0);
         const auto y = item.get_global_id(1);
         simd<T, NBlocks * BlockWidth * BlockHeight> tmp;
         if constexpr (CheckProperties) {
           if (y % 5 == 0)
             tmp = load_2d<T, BlockWidth, BlockHeight, NBlocks, Transposed,
                           Transformed>(
                 A, SurfaceWidth * sizeof(T) - 1, SurfaceHeight - 1,
                 SurfacePitch * sizeof(T) - 1, x * BlockWidth, y * BlockHeight,
                 LoadProperties);
           else if (y % 5 == 1)
             tmp = load_2d<T, BlockWidth, BlockHeight, NBlocks, Transposed>(
                 A, SurfaceWidth * sizeof(T) - 1, SurfaceHeight - 1,
                 SurfacePitch * sizeof(T) - 1, x * BlockWidth, y * BlockHeight,
                 LoadProperties);
           else if (y % 5 == 2)
             tmp = load_2d<T, BlockWidth, BlockHeight, NBlocks>(
                 A, SurfaceWidth * sizeof(T) - 1, SurfaceHeight - 1,
                 SurfacePitch * sizeof(T) - 1, x * BlockWidth, y * BlockHeight,
                 LoadProperties);
           else if (y % 5 == 3)
             tmp = load_2d<T, BlockWidth, BlockHeight>(
                 A, SurfaceWidth * sizeof(T) - 1, SurfaceHeight - 1,
                 SurfacePitch * sizeof(T) - 1, x * BlockWidth, y * BlockHeight,
                 LoadProperties);
           else if (y % 5 == 4)
             tmp = load_2d<T, BlockWidth>(
                 A, SurfaceWidth * sizeof(T) - 1, SurfaceHeight - 1,
                 SurfacePitch * sizeof(T) - 1, x * BlockWidth, y * BlockHeight,
                 LoadProperties);
         } else {
           if (y % 5 == 0)
             tmp = load_2d<T, BlockWidth, BlockHeight, NBlocks, Transposed,
                           Transformed>(
                 A, SurfaceWidth * sizeof(T) - 1, SurfaceHeight - 1,
                 SurfacePitch * sizeof(T) - 1, x * BlockWidth, y * BlockHeight);
           else if (y % 5 == 1)
             tmp = load_2d<T, BlockWidth, BlockHeight, NBlocks, Transposed>(
                 A, SurfaceWidth * sizeof(T) - 1, SurfaceHeight - 1,
                 SurfacePitch * sizeof(T) - 1, x * BlockWidth, y * BlockHeight);
           else if (y % 5 == 2)
             tmp = load_2d<T, BlockWidth, BlockHeight, NBlocks>(
                 A, SurfaceWidth * sizeof(T) - 1, SurfaceHeight - 1,
                 SurfacePitch * sizeof(T) - 1, x * BlockWidth, y * BlockHeight);
           else if (y % 5 == 3)
             tmp = load_2d<T, BlockWidth, BlockHeight>(
                 A, SurfaceWidth * sizeof(T) - 1, SurfaceHeight - 1,
                 SurfacePitch * sizeof(T) - 1, x * BlockWidth, y * BlockHeight);
           else if (y % 5 == 4)
             tmp = load_2d<T, BlockWidth>(
                 A, SurfaceWidth * sizeof(T) - 1, SurfaceHeight - 1,
                 SurfacePitch * sizeof(T) - 1, x * BlockWidth, y * BlockHeight);
         }
         tmp.copy_to(B + x * BlockWidth + y * SurfaceWidth);
       })
      .wait();

  bool error = false;
  for (auto i = 0; i < SurfaceWidth * SurfaceHeight; ++i) {
    if (B[i] != A[i]) {
      error = true;
      std::cerr << "Error at index=" << i << ": Expected=" << A[i]
                << ", Computed=" << B[i] << std::endl;
    }
  }

  free(A, Q);
  free(B, Q);
  return error;
}

int main() {
  properties CacheProps{cache_hint_L1<cache_hint::streaming>,
                        cache_hint_L2<cache_hint::cached>};
  constexpr bool CheckProperties = true;
  bool result = false;
  result |= test<float, 16, 128, 128, CheckProperties>(CacheProps);
  result |= test<float, 16, 128, 128, !CheckProperties>(CacheProps);
  result |= test<int64_t, 8, 128, 128, CheckProperties>(CacheProps);
  result |= test<int64_t, 8, 128, 128, !CheckProperties>(CacheProps);

  result |= test<int32_t, 16, 128, 128, CheckProperties>(CacheProps);
  result |= test<int32_t, 16, 128, 128, !CheckProperties>(CacheProps);
  result |= test<int16_t, 8, 128, 128, CheckProperties>(CacheProps);
  result |= test<int16_t, 8, 128, 128, !CheckProperties>(CacheProps);
  result |= test<int8_t, 8, 128, 128, CheckProperties>(CacheProps);
  result |= test<int8_t, 8, 128, 128, !CheckProperties>(CacheProps);

  std::cout << (result ? "FAILED" : "passed") << std::endl;
  return 0;
}
