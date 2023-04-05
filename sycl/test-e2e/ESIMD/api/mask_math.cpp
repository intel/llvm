//==---------------- mask_math.cpp  - DPC++ ESIMD on-device test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test checks basic arithmetic operations between simd and simd_mask
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: gpu-intel-gen9 && windows
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

namespace esimd = sycl::ext::intel::esimd;

int main(int, char **) {
  constexpr unsigned Size = 16;
  sycl::queue q;
  float *Input = sycl::malloc_shared<float>(Size, q);
  float *Output1 = sycl::malloc_shared<float>(Size, q);
  float *Output2 = sycl::malloc_shared<float>(Size, q);
  float *Output3 = sycl::malloc_shared<float>(Size, q);
  float *Output4 = sycl::malloc_shared<float>(Size, q);

  for (int i = 0; i < Size; ++i)
    Input[i] = i;

  q.submit([&](sycl::handler &cgh) {
     cgh.single_task<class Kernel>([=]() SYCL_ESIMD_KERNEL {
       esimd::simd<float, Size> x = 2;
       esimd::simd<float, Size> InputVector(Input);
       esimd::simd<float, Size> OutputVector1 = x * (InputVector < 8);
       esimd::simd<float, Size> OutputVector2 = (InputVector < 8) * x;
       esimd::simd<float, Size> OutputVector3 = 2 * (InputVector < 8);
       esimd::simd<float, Size> OutputVector4 = (InputVector < 8) * 2;
       OutputVector1.copy_to(Output1);
       OutputVector2.copy_to(Output2);
       OutputVector3.copy_to(Output3);
       OutputVector4.copy_to(Output4);
     });
   }).wait();

  bool Passed = true;
  float ExpectedValue = 0;
  for (int i = 0; i < Size; ++i) {
    if (i < 8) {
      ExpectedValue = 2;
    } else {
      ExpectedValue = 0;
    }

    if (Output1[i] != ExpectedValue || Output2[i] != ExpectedValue ||
        Output3[i] != ExpectedValue || Output4[i] != ExpectedValue) {
      Passed = false;
    }
  }
  if (!Passed) {
    std::cout << "Test failed." << std::endl;
  } else {
    std::cout << "Test passed." << std::endl;
  }
  free(Input, q);
  free(Output1, q);
  free(Output2, q);
  free(Output3, q);
  free(Output4, q);

  return Passed ? 1 : 0;
}
