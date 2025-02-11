//==-mask_expand_load.cpp - Test to verify expanded load functionality==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// This is a basic test to validate the expanded load API.

#include "esimd_test_utils.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;

template <typename DataT>
using shared_allocator = sycl::usm_allocator<DataT, sycl::usm::alloc::shared>;
template <typename DataT>
using shared_vector = std::vector<DataT, shared_allocator<DataT>>;

template <int N> bool test(sycl::queue &Queue) {
  shared_allocator<uint32_t> Allocator(Queue);

  shared_vector<uint32_t> Input(N / 2, 0, Allocator);
  shared_vector<uint32_t> Output(N, 0, Allocator);
  shared_vector<uint32_t> OutputAcc(N, 0, Allocator);
  std::vector<uint32_t> ExpectedOutput(Output.begin(), Output.end());

  int idx = 1;
  for (int I = 0; I < N; I++) {
    if (I < N / 2)
      Input[I] = I + 1;
    if ((I % 2) == 0)
      ExpectedOutput[I] = idx++;
  }

  auto *InputPtr = Input.data();
  auto *OutputPtr = Output.data();
  auto *OutputAccPtr = OutputAcc.data();
  buffer<uint32_t, 1> InputAcc_buffer(Input.data(), Input.size());

  Queue.submit([&](sycl::handler &cgh) {
    auto InputAcc = InputAcc_buffer.get_access<access::mode::read>(cgh);
    auto Kernel = ([=]() SYCL_ESIMD_KERNEL {
      simd_mask<N> Mask;
      for (int i = 0; i < N; i++)
        Mask[i] = (i % 2) == 0;

      simd<uint32_t, N> Result = mask_expand_load(InputPtr, Mask);
      simd<uint32_t, N> ResultAcc =
          mask_expand_load<uint32_t>(InputAcc, 0, Mask);
      Result.copy_to(OutputPtr);
      ResultAcc.copy_to(OutputAccPtr);
    });
    cgh.single_task(Kernel);
  });
  Queue.wait();

  for (int I = 0; I < N; I++) {
    if (OutputAcc[I] != Output[I] || Output[I] != ExpectedOutput[I]) {
      std::cout << "mask_compress_load: error at I = " << std::to_string(I)
                << ": " << std::to_string(ExpectedOutput[I])
                << " != " << std::to_string(OutputAcc[I])
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
