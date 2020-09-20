//==---------------- reduction.cpp  - DPC++ ESIMD on-device test
//------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// TODO enable on Windows
// REQUIRES: linux
// REQUIRES: gpu
// RUN: %clangxx-esimd -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %ESIMD_RUN_PLACEHOLDER %t.out

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <iostream>

using namespace cl::sycl;

typedef short TYPE;

int main(void) {
  constexpr unsigned InputSize = 32;
  constexpr unsigned OutputSize = 4;
  constexpr unsigned VL = 32;
  constexpr unsigned GroupSize = 1;

  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  auto ctxt = q.get_context();
  TYPE *A =
      static_cast<TYPE *>(malloc_shared(InputSize * sizeof(TYPE), dev, ctxt));
  int *B =
      static_cast<int *>(malloc_shared(OutputSize * sizeof(int), dev, ctxt));

  for (unsigned i = 0; i < InputSize; ++i) {
    if (i == 19) {
      A[i] = 32767;
    } else {
      A[i] = i;
    }
  }

  {
    cl::sycl::range<1> GroupRange{InputSize / VL};
    cl::sycl::range<1> TaskRange{GroupSize};
    cl::sycl::nd_range<1> Range(GroupRange, TaskRange);

    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<class Test>(
          GroupRange * TaskRange, [=](id<1> i) SYCL_ESIMD_KERNEL {
            using namespace sycl::INTEL::gpu;

            simd<TYPE, VL> va = block_load<TYPE, VL>(A + i * VL);
            simd<int, OutputSize> vb;

            vb.select<1, 0>(0) = reduce<int>(va, std::plus<>());
            vb.select<1, 0>(1) = reduce<int>(va, std::multiplies<>());
            vb.select<1, 0>(2) = hmax<int>(va);
            vb.select<1, 0>(3) = hmin<int>(va);

            block_store<int, OutputSize>(B + i * VL, vb);
          });
    });
    e.wait();
  }

  auto compute_reduce_sum = [](TYPE A[InputSize]) -> int {
    int retv = A[0];
    for (int i = 1; i < InputSize; i++) {
      retv += A[i];
    }
    return retv;
  };

  auto compute_reduce_prod = [](TYPE A[InputSize]) -> int {
    int retv = A[0];
    for (int i = 1; i < InputSize; i++) {
      retv *= A[i];
    }
    return retv;
  };

  auto compute_reduce_max = [](TYPE A[InputSize]) -> int {
    int retv = A[0];
    for (int i = 1; i < InputSize; i++) {
      if (A[i] > retv) {
        retv = A[i];
      }
    }
    return retv;
  };

  auto compute_reduce_min = [](TYPE A[InputSize]) -> int {
    int retv = A[0];
    for (int i = 1; i < InputSize; i++) {
      if (A[i] < retv) {
        retv = A[i];
      }
    }
    return retv;
  };

  bool TestPass = true;
  int ref = compute_reduce_sum(A);
  if (B[0] != ref) {
    std::cout << "Incorrect sum " << B[0] << ", expected " << ref << "\n";
    TestPass = false;
  }

  ref = compute_reduce_prod(A);
  if (B[1] != ref) {
    std::cout << "Incorrect prod " << B[1] << ", expected " << ref << "\n";
    TestPass = false;
  }

  ref = compute_reduce_max(A);
  if (B[2] != ref) {
    std::cout << "Incorrect max " << B[2] << ", expected " << ref << "\n";
    TestPass = false;
  }

  ref = compute_reduce_min(A);
  if (B[3] != ref) {
    std::cout << "Incorrect min " << B[3] << ", expected " << ref << "\n";
    TestPass = false;
  }

  if (!TestPass) {
    std::cout << "Failed\n";
    return 1;
  }

  std::cout << "Passed\n";
  return 0;
}
