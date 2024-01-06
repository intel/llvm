// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//==- rdtsc_sr0.cpp - Test to verify rdtsc0 and sr0 functionlity------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This is basic test to validate rdtsc and sr0 functions.

#include <cmath>
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/simd.hpp>
#include <sycl/sycl.hpp>
#include <vector>

int ErrCnt = 0;
template <typename DataT>
using shared_allocator = sycl::usm_allocator<DataT, sycl::usm::alloc::shared>;
template <typename DataT>
using shared_vector = std::vector<DataT, shared_allocator<DataT>>;

int test_rdtsc_sr0() {
  sycl::queue Queue;
  shared_allocator<uint64_t> Allocator(Queue);
  constexpr int32_t SIZE = 32;

  shared_vector<uint64_t> VectorOutputSR0(SIZE, -1, Allocator);
  shared_vector<uint64_t> VectorOutputRDTSC(SIZE, -1, Allocator);

  auto GlobalRange = sycl::range<1>(SIZE);
  sycl::range<1> LocalRange{1};
  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  {
    Queue.submit([&](sycl::handler &cgh) {
      uint64_t *VectorOutputSR0Ptr = VectorOutputSR0.data();
      uint64_t *VectorOutputRDTSCPtr = VectorOutputRDTSC.data();

      auto Kernel = ([=](sycl::nd_item<1> ndi) [[intel::sycl_explicit_simd]] {
        using namespace sycl::ext::intel::esimd;
        auto Idx = ndi.get_global_id(0);

        simd<uint64_t, 1> VectorResultSR0 =
            sycl::ext::intel::experimental::esimd::sr0();
        simd<uint64_t, 1> VectorResultRDTSC =
            sycl::ext::intel::experimental::esimd::rdtsc();

        VectorResultSR0.copy_to(VectorOutputSR0Ptr + Idx);
        VectorResultRDTSC.copy_to(VectorOutputRDTSCPtr + Idx);
      });

      cgh.parallel_for(Range, Kernel);
    });
    Queue.wait();
  }

  int Result = 0;

  // Check if returned values are not the same
  std::sort(VectorOutputRDTSC.begin(), VectorOutputRDTSC.end());
  std::sort(VectorOutputSR0.begin(), VectorOutputSR0.end());
  Result |= std::equal(VectorOutputRDTSC.begin() + 1, VectorOutputRDTSC.end(),
                       VectorOutputRDTSC.begin());
  Result |= std::equal(VectorOutputSR0.begin() + 1, VectorOutputSR0.end(),
                       VectorOutputSR0.begin());

  return Result;
}

int main() {

  int TestResult = 0;

  TestResult |= test_rdtsc_sr0();

  if (!TestResult) {
    std::cout << "Pass" << std::endl;
  }
  return TestResult;
}
