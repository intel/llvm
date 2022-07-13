//==------------ lsc_surf.cpp - DPC++ ESIMD on-device test -----------------==//
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

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

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

  auto vec_0 = std::vector<int>(size);
  auto vec_1 = std::vector<int>(size);
  auto vec_2 = std::vector<int>(size);
  auto vec_3 = std::vector<int>(size);
  auto vec_4 = std::vector<int>(size);
  std::iota(vec_0.begin(), vec_0.end(), 0);
  std::iota(vec_1.begin(), vec_1.end(), 0);
  std::iota(vec_2.begin(), vec_2.end(), 0);
  std::iota(vec_3.begin(), vec_3.end(), 0);
  std::iota(vec_4.begin(), vec_4.end(), 0);
  auto buf_0 = buffer{vec_0};
  auto buf_1 = buffer{vec_1};
  auto buf_2 = buffer{vec_2};
  auto buf_3 = buffer{vec_3};
  auto buf_4 = buffer{vec_4};

  try {
    q.submit([&](handler &h) {
      auto access_0 = buf_0.template get_access<access::mode::read_write>(h);
      auto access_1 = buf_1.template get_access<access::mode::read_write>(h);
      auto access_2 = buf_2.template get_access<access::mode::read_write>(h);
      auto access_3 = buf_3.template get_access<access::mode::read_write>(h);
      auto access_4 = buf_4.template get_access<access::mode::read_write>(h);
      h.parallel_for<class SimplestKernel>(
          range<1>{size / SIMDSize}, [=](id<1> id) SYCL_ESIMD_KERNEL {
            auto offset = id[0] * SIMDSize * sizeof(int);
            auto offsets = simd<uint32_t, SIMDSize>(id * SIMDSize * sizeof(int),
                                                    sizeof(int));
            auto pred = simd_mask<SIMDSize>(1);
            auto add = simd<int, SIMDSize>(5);
            auto compare = simd<int, SIMDSize>(id * SIMDSize, 1);
            auto swap = compare * 2;

            lsc_prefetch<int, SIMDSize, lsc_data_size::default_size,
                         cache_hint::cached, cache_hint::uncached>(access_0,
                                                                   offset);
            auto data_0 = lsc_block_load<int, SIMDSize>(access_0, offset);
            lsc_block_store<int, SIMDSize>(access_0, offset, data_0 * 2);

            lsc_prefetch<int, 1, lsc_data_size::default_size,
                         cache_hint::cached, cache_hint::uncached>(access_1,
                                                                   offsets);
            auto data_1 = lsc_gather<int>(access_1, offsets);
            lsc_scatter<int>(access_1, offsets, data_1 * 2);

            lsc_atomic_update<atomic_op::inc, int>(access_2, offsets, pred);
            lsc_atomic_update<atomic_op::add, int>(access_3, offsets, add,
                                                   pred);
            lsc_atomic_update<atomic_op::cmpxchg, int>(access_4, offsets,
                                                       compare, swap, pred);
          });
    });
    q.wait();
    buf_0.template get_access<access::mode::read_write>();
    buf_1.template get_access<access::mode::read_write>();
    buf_2.template get_access<access::mode::read_write>();
    buf_3.template get_access<access::mode::read_write>();
    buf_4.template get_access<access::mode::read_write>();
  } catch (sycl::exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    return 1;
  }

  auto error = 0;
  for (auto i = 0; i != size; ++i) {
    error += std::abs(vec_0[i] - (i * 2));
    error += std::abs(vec_1[i] - (i * 2));
    error += std::abs(vec_2[i] - (i + 1));
    error += std::abs(vec_3[i] - (i + 5));
    error += std::abs(vec_4[i] - (i * 2));
  }
  std::cout << (error != 0 ? "FAILED" : "passed") << std::endl;
  return error;
}
