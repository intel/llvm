//=mask_compress_store_acc.cpp-Test to verify compressed store functionality=//
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

template <int N> bool test() {
  std::vector<uint32_t> OutputAcc(N);
  std::vector<uint32_t> ExpectedOutput(N);

  int idx = 0;
  for (int I = 0; I < N; I++) {
    if ((I % 2) == 0)
      ExpectedOutput[idx++] = I + 1;
  }
  {
    auto Queue = queue{gpu_selector_v};
    esimd_test::printTestLabel(Queue);

    sycl::buffer<uint32_t, 1> OutputAcc_buffer(OutputAcc.data(),
                                               OutputAcc.size());

    auto e = Queue.submit([&](sycl::handler &cgh) {
      auto OutputAcc_out =
          OutputAcc_buffer.get_access<access::mode::read_write>(cgh);

      auto kernel = ([=]() [[intel::sycl_explicit_simd]] {
        simd<uint32_t, N> Input(1, 1);
        simd_mask<N> Mask;
        for (int i = 0; i < N; i++)
          Mask[i] = (i % 2) == 0;
        mask_compress_store(OutputAcc_out, 0, Input, Mask);
      });
      cgh.single_task(kernel);
    });
    Queue.wait();
  }

  for (int I = 0; I < N; I++) {
    if (OutputAcc[I] != ExpectedOutput[I]) {
      std::cout << "mask_compress_store: error at I = " << std::to_string(I)
                << ": " << std::to_string(ExpectedOutput[I])
                << " != " << std::to_string(OutputAcc[I]) << std::endl;
      return false;
    }
  }

  return true;
}

int main() {

  bool Pass = true;

  Pass &= test<8>();
  Pass &= test<16>();
  Pass &= test<32>();

  if (Pass)
    std::cout << "Pass" << std::endl;

  return !Pass;
}
