// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//==- thread_id_test.cpp - Test to verify thread id functionlity-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This is basic test to validate thread id functions.

#include <cmath>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/simd.hpp>
#include <sycl/usm.hpp>
#include <sycl/usm/usm_allocator.hpp>
#include <vector>

int ErrCnt = 0;
template <typename DataT>
using shared_allocator = sycl::usm_allocator<DataT, sycl::usm::alloc::shared>;
template <typename DataT>
using shared_vector = std::vector<DataT, shared_allocator<DataT>>;

int test_thread_id() {
  sycl::queue Queue;
  shared_allocator<int32_t> Allocator(Queue);
  constexpr int32_t SIZE = 32;

  shared_vector<int32_t> VectorOutputThreadId(SIZE, -1, Allocator);
  shared_vector<int32_t> VectorOutputSubdeviceId(SIZE, -1, Allocator);

  auto GlobalRange = sycl::range<1>(SIZE);
  sycl::range<1> LocalRange{1};
  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  {
    Queue.submit([&](sycl::handler &cgh) {
      int32_t *VectorOutputThreadIdPtr = VectorOutputThreadId.data();
      int32_t *VectorOutputSubdeviceIdPtr = VectorOutputSubdeviceId.data();

      auto Kernel = ([=](sycl::nd_item<1> ndi) [[intel::sycl_explicit_simd]] {
        using namespace sycl::ext::intel::esimd;
        auto Idx = ndi.get_global_id(0);

        simd<int32_t, 1> VectorResultThreadId =
            sycl::ext::intel::experimental::esimd::get_hw_thread_id();
        simd<int32_t, 1> VectorResultSubdeviceId =
            sycl::ext::intel::experimental::esimd::get_subdevice_id();

        VectorResultThreadId.copy_to(VectorOutputThreadIdPtr + Idx);
        VectorResultSubdeviceId.copy_to(VectorOutputSubdeviceIdPtr + Idx);
      });

      cgh.parallel_for(Range, Kernel);
    });
    Queue.wait();
  }

  int Result = 0;

  // Check if all returned elements are non negative
  Result |=
      !std::all_of(VectorOutputSubdeviceId.begin(),
                   VectorOutputSubdeviceId.end(), [](int i) { return i >= 0; });
  Result |=
      !std::all_of(VectorOutputThreadId.begin(), VectorOutputThreadId.end(),
                   [](int i) { return i >= 0; });

  // Check if returned values are not the same
  std::sort(VectorOutputThreadId.begin(), VectorOutputThreadId.end());
  Result |=
      std::equal(VectorOutputThreadId.begin() + 1, VectorOutputThreadId.end(),
                 VectorOutputThreadId.begin());

  return Result;
}

int main() {

  int TestResult = 0;

  TestResult |= test_thread_id();

  if (!TestResult) {
    std::cout << "Pass" << std::endl;
  }
  return TestResult;
}
