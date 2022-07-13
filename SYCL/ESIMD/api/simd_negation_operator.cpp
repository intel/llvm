//==----- simd_negation_operator.cpp  - DPC++ ESIMD on-device test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// The test checks negation (operator!) of a simd object.
// The operation `!v` is equivalent to `v == 0`;

#include "../esimd_test_utils.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <iostream>
#include <limits>

using namespace cl::sycl;
using namespace sycl::ext::intel::esimd;

constexpr unsigned VL = 8;

template <class T>
bool test(queue q, const std::array<T, VL> &input,
          const std::array<unsigned short, VL> &gold) {
  std::cout << "Testing "
            << "T=" << typeid(T).name() << " ...\n";

  auto A = input;
  // mask elements are unsigned short
  std::array<unsigned short, VL> outputMask = {0, 0, 0, 0, 0, 0, 0, 0};

  try {
    buffer<T, 1> bufA(A.data(), range<1>(VL));
    buffer<unsigned short, 1> bufB(outputMask.data(), range<1>(VL));
    range<1> glob_range{1};

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufA.template get_access<access::mode::read>(cgh);
      auto PB = bufB.template get_access<access::mode::write>(cgh);
      cgh.parallel_for(glob_range, [=](id<1> i) SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;
        simd<T, VL> va;
        va.copy_from(PA, 0);
        auto mask = !va;
        mask.copy_to(PB, 0);
      });
    });
    q.wait_and_throw();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false; // not success
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < VL; ++i) {
    unsigned short val = outputMask[i];

    if (val != gold[i]) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ": " << val << " != " << gold[i]
                  << " (gold)\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: " << ((float)(VL - err_cnt) / (float)VL) * 100.0f
              << "% (" << (VL - err_cnt) << "/" << VL << ")\n";
  }

  std::cout << (err_cnt > 0 ? "  FAILED\n" : "  Passed\n");
  return err_cnt > 0 ? false : true;
}

int main(void) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  auto int32max = std::numeric_limits<int>::max();
  auto int32min = std::numeric_limits<int>::min();
  auto uint32max = std::numeric_limits<unsigned int>::max();
  auto int16max = std::numeric_limits<short>::max();
  auto int16min = std::numeric_limits<short>::min();
  auto uint16max = std::numeric_limits<unsigned short>::max();

  bool passed = true;
  passed &= test(
      q, std::array<int, VL>{0, 1, -4, int32max, 0, int32min, 0, 1}, // input
      std::array<unsigned short, VL>{1, 0, 0, 0, 1, 0, 1, 0});       // gold
  passed &=
      test(q, std::array<unsigned int, VL>{0, 1, 4, 0, uint32max, 0, 0, 1},
           std::array<unsigned short, VL>{1, 0, 0, 1, 0, 1, 1, 0});
  passed &=
      test(q, std::array<short, VL>{0, 1, -4, int16max, 0, int16min, 0, 1},
           std::array<unsigned short, VL>{1, 0, 0, 0, 1, 0, 1, 0});
  passed &=
      test(q, std::array<unsigned short, VL>{0, 1, 4, 0, uint16max, 0, 0, 1},
           std::array<unsigned short, VL>{1, 0, 0, 1, 0, 1, 1, 0});

  return passed ? 0 : 1;
}
