//==------------ lsc_usm.cpp - DPC++ ESIMD on-device test ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <sycl/ext/intel/esimd.hpp>

int main() {
  using namespace cl::sycl;
  using namespace sycl::ext::intel::esimd;
  using namespace sycl::ext::intel::experimental::esimd;
  auto size = size_t{128};
  auto constexpr SIMDSize = unsigned{4};

  auto q =
      queue{esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler()};
  auto device = q.get_device();
  std::cout << "Device name: " << device.get_info<info::device::name>()
            << std::endl;

  auto *vec_0 = malloc_shared<int>(size, q);
  auto *vec_1 = malloc_shared<int>(size, q);
  auto *vec_2 = malloc_shared<int>(size, q);
  auto *vec_3 = malloc_shared<int>(size, q);
  auto *vec_4 = malloc_shared<int>(size, q);
  std::iota(vec_0, vec_0 + size, 0);
  std::iota(vec_1, vec_1 + size, 0);
  std::iota(vec_2, vec_2 + size, 0);
  std::iota(vec_3, vec_3 + size, 0);
  std::iota(vec_4, vec_4 + size, 0);

  try {
    q.submit([&](handler &h) {
      h.parallel_for<class SimplestKernel>(
          range<1>{size / SIMDSize}, [=](id<1> id) SYCL_ESIMD_KERNEL {
            auto offset = id[0] * SIMDSize;
            auto offsets = simd<uint32_t, SIMDSize>(id * SIMDSize * sizeof(int),
                                                    sizeof(int));
            auto pred = simd_mask<SIMDSize>(1);
            auto add = simd<int, SIMDSize>(5);
            auto compare = simd<int, SIMDSize>(id * SIMDSize, 1);
            auto swap = compare * 2;

            lsc_prefetch<int, SIMDSize, lsc_data_size::default_size,
                         cache_hint::cached, cache_hint::uncached>(vec_0 +
                                                                   offset);
            auto data_0 = lsc_block_load<int, SIMDSize>(vec_0 + offset);
            lsc_block_store<int, SIMDSize>(vec_0 + offset, data_0 * 2);

            lsc_prefetch<int, 1, lsc_data_size::default_size,
                         cache_hint::cached, cache_hint::uncached>(vec_1,
                                                                   offsets);
            auto data_1 = lsc_gather<int>(vec_1, offsets);
            lsc_scatter<int>(vec_1, offsets, data_1 * 2);

            lsc_atomic_update<atomic_op::inc, int>(vec_2, offsets, pred);
            lsc_atomic_update<atomic_op::add, int>(vec_3, offsets, add, pred);
            lsc_atomic_update<atomic_op::cmpxchg, int>(vec_4, offsets, compare,
                                                       swap, pred);
          });
    });
    q.wait();
  } catch (sycl::exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    sycl::free(vec_0, q);
    sycl::free(vec_1, q);
    sycl::free(vec_2, q);
    sycl::free(vec_3, q);
    sycl::free(vec_4, q);
    return 1;
  }

  auto error = 0;
  for (auto i = 0; i != size; ++i) {
    error += std::abs(vec_0[i] - 2 * i);
    error += std::abs(vec_1[i] - 2 * i);
    error += std::abs(vec_2[i] - (i + 1));
    error += std::abs(vec_3[i] - (i + 5));
    error += std::abs(vec_4[i] - (i * 2));
  }
  sycl::free(vec_0, q);
  sycl::free(vec_1, q);
  sycl::free(vec_2, q);
  sycl::free(vec_3, q);
  sycl::free(vec_4, q);
  std::cout << (error != 0 ? "FAILED" : "passed") << std::endl;
  return error;
}
