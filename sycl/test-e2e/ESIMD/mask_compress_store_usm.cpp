//=mask_compress_store_usm.cpp-Test to verify compressed store functionality=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// This is a basic test to validate the compressed store API.

#include "esimd_test_utils.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;

template <typename DataT>
using shared_allocator = sycl::usm_allocator<DataT, sycl::usm::alloc::shared>;
template <typename DataT>
using shared_vector = std::vector<DataT, shared_allocator<DataT>>;

template <int N> bool test(sycl::queue &Queue) {
  shared_allocator<uint32_t> Allocator(Queue);

  shared_vector<uint32_t> Output(N, 0, Allocator);
  shared_vector<uint32_t> OutputVector(N, 0, Allocator);
  std::vector<uint32_t> ExpectedOutput(Output.begin(), Output.end());

  int idx = 0;
  for (int I = 0; I < N; I++) {
    if ((I % 2) == 0)
      ExpectedOutput[idx++] = I + 1;
  }

  auto *OutputPtr = Output.data();
  auto *OutputVectorPtr = OutputVector.data();

  Queue.submit([&](sycl::handler &cgh) {
    auto Kernel = ([=]() SYCL_ESIMD_KERNEL {
      simd<uint32_t, N> Input(1, 1);
      simd<uint32_t, N> Result(0);
      simd_mask<N> Mask;
      for (int i = 0; i < N; i++)
        Mask[i] = (i % 2) == 0;
      mask_compress_store(OutputPtr, Input, Mask);
      mask_compress_store(Result, 0, Input, Mask);
      Result.copy_to(OutputVectorPtr);
    });
    cgh.single_task(Kernel);
  });
  Queue.wait();

  for (int I = 0; I < N; I++) {
    if (Output[I] != OutputVector[I] || Output[I] != ExpectedOutput[I]) {
      std::cout << "mask_compress_store: error at I = " << std::to_string(I)
                << ": " << std::to_string(ExpectedOutput[I])
                << " != " << std::to_string(OutputVector[I])
                << " != " << std::to_string(Output[I]) << std::endl;
      return false;
    }
  }

  return true;
}

int main() {

  bool Pass = true;
  auto Q = queue{gpu_selector_v};
  esimd_test::printTestLabel(Q);
  Pass &= test<8>(Q);
  Pass &= test<16>(Q);
  Pass &= test<32>(Q);

  if (Pass)
    std::cout << "Pass" << std::endl;

  return !Pass;
}
