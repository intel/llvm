//==---------------- popcount.cpp - DPC++ ESIMD on-device test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

#include "esimd_test_utils.hpp"

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl;

template <typename T> bool test(queue &q) {
  std::cout << "Running " << esimd_test::type_name<T>() << std::endl;
  constexpr unsigned VL = 16;
  constexpr unsigned Size =
      std::min(std::numeric_limits<T>::max() - VL + 1, 65536u);

  T *A = new T[Size];
  T *B = new T[Size];

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = i;
    B[i] = 0;
  }

  try {
    buffer<T, 1> bufa(A, range<1>(Size));
    buffer<T, 1> bufb(B, range<1>(Size));

    // We need that many workgroups
    range<1> GlobalRange{Size / VL};

    // We need that many threads in each group
    range<1> LocalRange{1};

    queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufa.template get_access<access::mode::read>(cgh);
      auto PB = bufb.template get_access<access::mode::write>(cgh);
      cgh.parallel_for(GlobalRange * LocalRange,
                       [=](id<1> i) SYCL_ESIMD_KERNEL {
                         using namespace sycl::ext::intel::esimd;
                         unsigned int offset = i * VL * sizeof(T);
                         simd<T, VL> va;
                         va.copy_from(PA, offset);
                         simd<T, VL> vb = __ESIMD_ENS::popcount(va);
                         vb.copy_to(PB, offset);
                       });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';

    delete[] A;
    delete[] B;
    return false;
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    int Expected = std::bitset<sizeof(T) * 8>(i).count();
    int Computed = B[i];
    if (Expected != Computed && ++err_cnt < 10)
      std::cout << "Failure at " << std::to_string(i)
                << ": Expected: " << std::to_string(Expected)
                << " Computed: " << std::to_string(Computed) << std::endl;
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }

  delete[] A;
  delete[] B;

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt == 0;
}

int main() {
  bool Passed = true;
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  esimd_test::printTestLabel(q);
  Passed &= test<uint8_t>(q);
  Passed &= test<int8_t>(q);
  Passed &= test<uint16_t>(q);
  Passed &= test<int16_t>(q);
  Passed &= test<uint32_t>(q);
  Passed &= test<int32_t>(q);
  return !Passed;
}
