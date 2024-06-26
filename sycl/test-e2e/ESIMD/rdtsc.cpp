// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//==- rdtsc.cpp - Test to verify rdtsc0 and sr0 functionlity----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This is basic test to validate rdtsc function.

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

int test_rdtsc() {
  sycl::queue Queue;
  shared_allocator<uint64_t> Allocator(Queue);
  constexpr int32_t SIZE = 32;

  shared_vector<uint64_t> VectorOutputRDTSC(SIZE, 0, Allocator);

  auto GlobalRange = sycl::range<1>(SIZE);
  sycl::range<1> LocalRange{1};
  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  {
    Queue.submit([&](sycl::handler &cgh) {
      uint64_t *VectorOutputRDTSCPtr = VectorOutputRDTSC.data();

      auto Kernel = ([=](sycl::nd_item<1> ndi) [[intel::sycl_explicit_simd]] {
        using namespace sycl::ext::intel::esimd;
        auto Idx = ndi.get_global_id(0);
        uint64_t StartCounter =
            Idx % 2 == 0 ? __ESIMD_NS::rdtsc() : __ESIMD_ENS::rdtsc();
        simd<uint64_t, 1> VectorResultRDTSC(VectorOutputRDTSCPtr + Idx);
        uint64_t EndCounter =
            Idx % 2 == 0 ? __ESIMD_NS::rdtsc() : __ESIMD_ENS::rdtsc();
        VectorResultRDTSC += EndCounter > StartCounter;

        VectorResultRDTSC.copy_to(VectorOutputRDTSCPtr + Idx);
      });

      cgh.parallel_for(Range, Kernel);
    });
    Queue.wait();
  }

  return std::any_of(VectorOutputRDTSC.begin(), VectorOutputRDTSC.end(),
                     [](uint64_t v) { return v == 0; });
}

int main() {

  int TestResult = test_rdtsc();

  if (!TestResult) {
    std::cout << "Pass" << std::endl;
  }
  return TestResult;
}
