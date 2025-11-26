//=----------clamp.cpp-Test to verify clamp functionality------------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// This is a basic test to validate the clamp API.

#include "esimd_test_utils.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;

template <typename DataT>
using shared_allocator = sycl::usm_allocator<DataT, sycl::usm::alloc::shared>;
template <typename DataT>
using shared_vector = std::vector<DataT, shared_allocator<DataT>>;
constexpr int N = 16;

bool test(sycl::queue &Queue) {
  shared_allocator<int32_t> Allocator(Queue);
  shared_vector<int32_t> Output(N, 0, Allocator);
  shared_vector<int32_t> OutputVector(N, 0, Allocator);
  std::vector<int32_t> ExpectedOutput(Output.begin(), Output.end());
  std::vector<int32_t> ExpectedVectorOutput(Output.begin(), Output.end());

  for (int I = 0; I < N; I++) {
    if (I + 1 < 3)
      ExpectedOutput[I] = 3;
    else if (I + 1 > 9)
      ExpectedOutput[I] = 9;
    else
      ExpectedOutput[I] = I + 1;
    if (I + 1 < 3 - I)
      ExpectedVectorOutput[I] = 3 - I;
    else if (I + 1 > 15 - I)
      ExpectedVectorOutput[I] = 15 - I;
    else
      ExpectedVectorOutput[I] = I + 1;
  }

  auto *OutputPtr = Output.data();
  auto *OutputVectorPtr = OutputVector.data();

  Queue.submit([&](sycl::handler &cgh) {
    auto Kernel = ([=]() SYCL_ESIMD_KERNEL {
      simd<int32_t, N> Input(1, 1);
      simd<int32_t, N> Min(3, -1);
      simd<int32_t, N> Max(15, -1);
      simd<int32_t, N> Result = clamp(Input, 3, 9);
      simd<int32_t, N> ResultVector = clamp(Input, Min, Max);

      Result.copy_to(OutputPtr);
      ResultVector.copy_to(OutputVectorPtr);
    });
    cgh.single_task(Kernel);
  });
  Queue.wait();

  for (int I = 0; I < N; I++) {
    if (ExpectedVectorOutput[I] != OutputVector[I] ||
        Output[I] != ExpectedOutput[I]) {
      std::cout << "clamp: error at I = " << I << ": " << ExpectedOutput[I]
                << " != " << Output[I] << " : " << ExpectedVectorOutput[I]
                << " != " << OutputVector[I] << std::endl;
      return false;
    }
  }

  return true;
}

int main() {

  auto Q = queue{gpu_selector_v};
  esimd_test::printTestLabel(Q);
  bool Pass = test(Q);

  if (Pass)
    std::cout << "Pass" << std::endl;

  return !Pass;
}
