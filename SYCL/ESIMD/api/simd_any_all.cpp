//==------- esimd_any_all.cpp  - DPC++ ESIMD on-device test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// TODO: esimd_emulator fails due to unimplemented 'half' type
// XFAIL: esimd_emulator
// RUN: %clangxx -fsycl %s -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// Smoke test for esimd any/all operations APIs.

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

#include <iostream>

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental;
using namespace sycl::ext::intel::experimental::esimd;

struct bit_op {
  enum { any, all, num_ops };
};

template <typename T, int N, int Op> struct test_id;

template <typename T> struct char_to_int {
  using type = typename std::conditional<
      sizeof(T) == 1,
      typename std::conditional<std::is_signed<T>::value, int, unsigned>::type,
      T>::type;
};

// The main test routine.
template <typename T, int N, int Op> bool test_impl(queue q) {
  const char *ops[bit_op::num_ops] = {"any", "all"};
  std::cout << "Testing op=" << ops[Op] << " T=" << typeid(T).name()
            << ", N=" << N << "...\n";

  simd<T, N> all_zero((T)0);
  simd<T, N> all_one((T)1);
  if (std::is_signed_v<T>) {
    all_one[0] = -1;
  }
  simd<T, N> all_two((T)2); // check that non-zero with LSB=0 counts as 'set'
  if (std::is_signed_v<T>) {
    all_two[N - 1] = -2;
  }
  simd<T, N> zero_two((T)0);

  if (N > 1) {
    zero_two[1] = 2;
  }

  simd<T, N> test_vals_arr[] = {all_zero, all_one, all_two, zero_two};
  constexpr size_t num_vals = sizeof(test_vals_arr) / sizeof(test_vals_arr[0]);
  T *test_vals = sycl::malloc_shared<T>(num_vals * N, q);
  uint16_t *res = sycl::malloc_shared<uint16_t>(num_vals, q);

  for (unsigned int i = 0; i < num_vals; ++i) {
    res[i] = 0xFFff;
  }
  memcpy(test_vals, test_vals_arr, sizeof(test_vals_arr));

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.single_task<test_id<T, N, Op>>([=]() SYCL_ESIMD_KERNEL {
        for (int i = 0; i < num_vals; i++) {
          simd<T, N> src(test_vals + i * N);

          if constexpr (Op == bit_op::any) {
            res[i] = src.any();
          } else if constexpr (Op == bit_op::all) {
            res[i] = src.all();
          }
        }
      });
    });
    e.wait_and_throw();
  } catch (cl::sycl::exception const &e) {
    std::cout << "  SYCL exception caught: " << e.what() << '\n';
    sycl::free(res, q);
    sycl::free(test_vals, q);
    return false;
  }

  unsigned int Gold[num_vals * bit_op::num_ops] = {
      // any:
      0, // all zero
      1, // all one
      1, // all two
      1, // zero, two
      // all:
      0, // all zero
      1, // all one
      1, // all two
      0, // zero, two
  };
  int err_cnt = 0;

  using ValTy = typename char_to_int<T>::type;

  for (unsigned i = 0; i < num_vals; ++i) {
    if ((N == 1) && (i == 3)) {
      continue; // (zero, two) testcase not available for single element
    }
    T gold = Gold[Op * num_vals + i];
    T val = res[i];
    std::cout << "  " << ops[Op] << "(" << (simd<ValTy, N>)test_vals_arr[i]
              << ") = " << (ValTy)val;

    if (val != gold) {
      ++err_cnt;
      std::cout << " ERROR. " << (ValTy)val << " != " << (ValTy)gold
                << "(gold)\n";
    } else {
      std::cout << " (ok)\n";
    }
  }
  std::cout << (err_cnt > 0 ? "  FAILED\n" : "  Passed\n");
  sycl::free(res, q);
  sycl::free(test_vals, q);
  return err_cnt > 0 ? false : true;
}

template <class T1, class T2> bool test(queue q) {
  bool passed = true;
  passed &= test_impl<T1, 32, bit_op::any>(q);
  passed &= test_impl<T1, 8, bit_op::all>(q);
  passed &= test_impl<T2, 3, bit_op::any>(q);
  passed &= test_impl<T2, 1, bit_op::all>(q);
  return passed;
}

int main(int argc, char **argv) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;
  passed &= test<int8_t, uint8_t>(q);
  passed &= test<int16_t, uint16_t>(q);
  passed &= test<int32_t, uint32_t>(q);
  passed &= test<int64_t, uint64_t>(q);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
