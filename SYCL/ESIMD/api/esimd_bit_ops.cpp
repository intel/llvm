//==------- esimd_bit_ops.cpp  - DPC++ ESIMD on-device test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// Smoke test for esimd bit operations APIs.

#include "../esimd_test_utils.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

using namespace cl::sycl;
using namespace sycl::ext::intel;
using namespace sycl::ext::intel::esimd;

struct bit_op {
  enum { cbit, fbl, fbh, num_ops };
};

template <typename T, int N, int Op> struct test_id;

template <typename T> struct char_to_int {
  using type = typename std::conditional<
      sizeof(T) == 1,
      typename std::conditional<std::is_signed<T>::value, int, unsigned>::type,
      T>::type;
};

// The main test routine.
template <typename T, int N, int Op> bool test(queue q) {
  const char *ops[bit_op::num_ops] = {"cbit", "fbl", "fbh"};
  std::cout << "Testing op=" << ops[Op] << " T=" << typeid(T).name()
            << ", N=" << N << "...\n";

  T val_all_zero{0};
  T val_all_one{static_cast<T>(~val_all_zero)};
  T val_two_one{static_cast<T>(T{1} << (sizeof(T) * 8 - 2) | 2)}; // 010...010

  T vals[] = {val_all_zero, val_all_one, val_two_one};
  constexpr size_t num_vals = sizeof(vals) / sizeof(vals[0]);

  constexpr size_t size = N * num_vals;
  unsigned int *A = sycl::malloc_shared<unsigned int>(num_vals, q);

  for (unsigned int i = 0; i < num_vals; ++i) {
    A[i] = 0xFFFFffff;
  }

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.single_task<test_id<T, N, Op>>([=]() SYCL_ESIMD_KERNEL {
        // TODO add test cases where each lane contains different value
        if constexpr (Op == bit_op::cbit) {
          A[0] = esimd::cbit(simd<T, N>{val_all_zero})[N / 2];
          A[1] = esimd::cbit(simd<T, N>{val_all_one})[N / 2];
          A[2] = esimd::cbit(simd<T, N>{val_two_one})[N / 2];
        } else if constexpr (Op == bit_op::fbl) {
          A[0] = esimd::fbl(simd<T, N>{val_all_zero})[N / 2];
          A[1] = esimd::fbl(simd<T, N>{val_all_one})[N / 2];
          A[2] = esimd::fbl(simd<T, N>{val_two_one})[N / 2];
        } else {
          static_assert(Op == bit_op::fbh);
          A[0] = esimd::fbh(simd<T, N>{val_all_zero})[N / 2];
          A[1] = esimd::fbh(simd<T, N>{val_all_one})[N / 2];
          A[2] = esimd::fbh(simd<T, N>{val_two_one})[N / 2];
        }
      });
    });
    e.wait_and_throw();
  } catch (cl::sycl::exception const &e) {
    std::cout << "  SYCL exception caught: " << e.what() << '\n';
    sycl::free(A, q);
    return false;
  }

  unsigned int Gold[size] = {
      // cbit:
      0,             // all zero
      sizeof(T) * 8, // all one
      2,             // two one
      // fbl:
      0xFFFFffff, // all zero
      0,          // all one
      1,          // two one
      // fbh:
      0xFFFFffff,                                // all zero
      std::is_signed<T>::value ? 0xFFFFffff : 0, // all one
      1                                          // two one
  };
  int err_cnt = 0;

  using ValTy = typename char_to_int<T>::type;

  for (unsigned i = 0; i < num_vals; ++i) {
    T gold = Gold[Op * num_vals + i];
    T val = A[i];
    std::cout << "  " << (ValTy)vals[i] << ": ";

    if (val != gold) {
      ++err_cnt;
      std::cout << "ERROR. " << (ValTy)val << " != " << (ValTy)gold
                << "(gold)\n";
    } else {
      std::cout << "ok\n";
    }
  }
  std::cout << (err_cnt > 0 ? "  FAILED\n" : "  Passed\n");
  sycl::free(A, q);
  return err_cnt > 0 ? false : true;
}

int main(int argc, char **argv) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;
  passed &= test<char, 32, bit_op::cbit>(q);
  passed &= test<unsigned char, 16, bit_op::cbit>(q);
  passed &= test<short, 32, bit_op::cbit>(q);
  passed &= test<short, 16, bit_op::cbit>(q);
  passed &= test<unsigned short, 8, bit_op::cbit>(q);
  passed &= test<int, 32, bit_op::cbit>(q);
  passed &= test<unsigned int, 32, bit_op::cbit>(q);
  // TODO uncomment when implemenation is fixed to support 64-bit ints ops:
  // passed &= test<int64_t, 32, bit_op::cbit>(q);
  // passed &= test<uint64_t, 32, bit_op::cbit>(q);

  passed &= test<int, 32, bit_op::fbl>(q);
  passed &= test<unsigned int, 32, bit_op::fbl>(q);

  passed &= test<int, 32, bit_op::fbh>(q);
  passed &= test<unsigned int, 32, bit_op::fbh>(q);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
