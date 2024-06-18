//==------------ lsc_slm.cpp - DPC++ ESIMD on-device test ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc || gpu-intel-dg2
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "../esimd_test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

int main() {
  using namespace sycl;
  using namespace sycl::ext::intel::esimd;
  using namespace sycl::ext::intel::experimental::esimd;
  auto size = size_t{128};
  auto constexpr SIMDSize = unsigned{4};

  auto q =
      queue{esimd_test::ESIMDSelector, esimd_test::createExceptionHandler()};
  auto device = q.get_device();
  std::cout << "Device name: " << device.get_info<sycl::info::device::name>()
            << std::endl;

  auto vec_0 = std::vector<int>(size);
  auto vec_1 = std::vector<int>(size);
  auto vec_2 = std::vector<int>(size);
  auto vec_3 = std::vector<int>(size);
  auto vec_4 = std::vector<int>(size);

  try {
    auto buf_0 = buffer{vec_0};
    auto buf_1 = buffer{vec_1};
    auto buf_2 = buffer{vec_2};
    auto buf_3 = buffer{vec_3};
    auto buf_4 = buffer{vec_4};
    q.submit([&](handler &h) {
      auto access_0 = buf_0.template get_access<access::mode::read_write>(h);
      auto access_1 = buf_1.template get_access<access::mode::read_write>(h);
      auto access_2 = buf_2.template get_access<access::mode::read_write>(h);
      auto access_3 = buf_3.template get_access<access::mode::read_write>(h);
      auto access_4 = buf_4.template get_access<access::mode::read_write>(h);
      h.parallel_for<class SimplestKernel>(
          range<1>{size / SIMDSize}, [=](id<1> id) SYCL_ESIMD_KERNEL {
            auto offset = id * SIMDSize * sizeof(int);
            auto offsets = simd<uint32_t, SIMDSize>(id * SIMDSize * sizeof(int),
                                                    sizeof(int));
            auto data = simd<int, SIMDSize>(id * SIMDSize, 1);
            auto pred = simd_mask<SIMDSize>(1);
            auto add = simd<int, SIMDSize>(5);
            auto compare = simd<int, SIMDSize>(id * SIMDSize, 1);
            auto swap = compare * 2;

            slm_init<4096>();
            lsc_slm_block_store<int, SIMDSize>(offset, data * 2);
            auto data_0 = lsc_slm_block_load<int, SIMDSize>(offset);
            lsc_block_store<int, SIMDSize>(access_0, offset, data_0);

            lsc_slm_scatter<int>(offsets, data * 2);
            auto data_1 = lsc_slm_gather<int>(offsets);
            lsc_block_store<int, SIMDSize>(access_1, offset, data_1);

            lsc_slm_block_store<int, SIMDSize>(offset, data);
            lsc_slm_atomic_update<atomic_op::inc, int>(offsets, pred);
            auto data_2 = lsc_slm_block_load<int, SIMDSize>(offset);
            lsc_block_store<int, SIMDSize>(access_2, offset, data_2);

            lsc_slm_block_store<int, SIMDSize>(offset, data);
            lsc_slm_atomic_update<atomic_op::add, int>(offsets, add, pred);
            auto data_3 = lsc_slm_block_load<int, SIMDSize>(offset);
            lsc_block_store<int, SIMDSize>(access_3, offset, data_3);

            lsc_slm_block_store<int, SIMDSize>(offset, data);
            lsc_slm_atomic_update<atomic_op::cmpxchg, int>(offsets, compare,
                                                           swap, pred);
            auto data_4 = lsc_slm_block_load<int, SIMDSize>(offset);
            lsc_block_store<int, SIMDSize>(access_4, offset, data_4);
          });
    });
    q.wait();
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
