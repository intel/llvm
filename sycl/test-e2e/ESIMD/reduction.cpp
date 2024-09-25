//==---------------- reduction.cpp  - DPC++ ESIMD on-device test
//------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "esimd_test_utils.hpp"

using namespace sycl;
constexpr unsigned InputSize = 32;
constexpr unsigned OutputSize = 4;
constexpr unsigned VL = 32;
constexpr unsigned GroupSize = 1;

template <typename TIn, typename TOut> bool test(sycl::queue &Queue) {
  std::cout << "TIn= " << esimd_test::type_name<TIn>()
            << " TOut= " << esimd_test::type_name<TOut>() << std::endl;
  TIn *A = malloc_shared<TIn>(InputSize, Queue);
  TOut *B = malloc_shared<TOut>(OutputSize, Queue);

  for (unsigned i = 0; i < InputSize; ++i) {
    if (i == 19) {
      A[i] = 32767;
    } else {
      A[i] = i;
    }
  }

  try {
    range<1> GroupRange{InputSize / VL};
    range<1> TaskRange{GroupSize};
    nd_range<1> Range(GroupRange, TaskRange);

    auto e = Queue.submit([&](handler &cgh) {
      cgh.parallel_for(GroupRange * TaskRange, [=](id<1> i) SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;

        simd<TIn, VL> va;
        va.copy_from(A + i * VL);
        simd<TOut, OutputSize> vb;

        vb[0] = reduce<TOut>(va, std::plus<>());
        vb[1] = reduce<TOut>(va, std::multiplies<>());
        vb[2] = hmax<TOut>(va);
        vb[3] = hmin<TOut>(va);

        vb.copy_to(B + i * VL);
      });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(A, Queue);
    free(B, Queue);
    return 0;
  }

  auto compute_reduce_sum = [](TIn A[InputSize]) -> TOut {
    TOut retv = A[0];
    for (int i = 1; i < InputSize; i++) {
      retv += A[i];
    }
    return retv;
  };

  auto compute_reduce_prod = [](TIn A[InputSize]) -> TOut {
    TOut retv = A[0];
    for (int i = 1; i < InputSize; i++) {
      retv *= A[i];
    }
    return retv;
  };

  auto compute_reduce_max = [](TIn A[InputSize]) -> TOut {
    TOut retv = A[0];
    for (int i = 1; i < InputSize; i++) {
      if (A[i] > retv) {
        retv = A[i];
      }
    }
    return retv;
  };

  auto compute_reduce_min = [](TIn A[InputSize]) -> TOut {
    TOut retv = A[0];
    for (int i = 1; i < InputSize; i++) {
      if (A[i] < retv) {
        retv = A[i];
      }
    }
    return retv;
  };

  bool TestPass = true;
  TOut ref = compute_reduce_sum(A);
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

  free(A, Queue);
  free(B, Queue);
  return TestPass;
}

int main(void) {

  queue Queue(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  esimd_test::printTestLabel(Queue);

  bool TestPass = test<int16_t, int32_t>(Queue);
  TestPass &= test<uint16_t, uint32_t>(Queue);
  TestPass &= test<float, float>(Queue);

  if (!TestPass) {
    std::cout << "Failed\n";
    return 1;
  }

  std::cout << "Passed\n";
  return 0;
}
