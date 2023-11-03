// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_TRACE=2 %{run} %t.out 2>&1 | FileCheck %s

//==----------- spec_const_redefine.cpp ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The test checks that ESIMD kernels support specialization constants,
// particularly:
// - a specialization constant can be redifined and correct new value is used
//   after redefinition.
// - the program is JITted only once per a unique set of specialization
//   constants values.

#include "../esimd_test_utils.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <iostream>
#include <vector>

using namespace sycl;

int val = 0;

// Fetch a value at runtime.
int get_value() { return val; }

constexpr specialization_id<int32_t> SC0;
constexpr specialization_id<int32_t> SC1;

int main(int argc, char **argv) {
  val = argc;

  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  auto ctxt = q.get_context();

  bool passed = true;
  int x = get_value();

  const int sc_vals[][2] = {
      {1 + x, 2 + x},
      {2 + x, 3 + x},
      {1 + x, 2 + x}, // same as first - program in cache must be used
      {2 + x, 3 + x}  // same as second - program in cache must be used
  };
  constexpr int n_sc_sets = sizeof(sc_vals) / sizeof(sc_vals[0]);
  std::vector<int> vec(n_sc_sets);

  for (int i = 0; i < n_sc_sets; i++) {
    const int *sc_set = &sc_vals[i][0];
    try {
      sycl::buffer<int, 1> buf(vec.data(), vec.size());

      q.submit([&](sycl::handler &cgh) {
        cgh.set_specialization_constant<SC0>(-500);
        cgh.set_specialization_constant<SC1>(9999);
        cgh.set_specialization_constant<SC0>(sc_set[0]);
        cgh.set_specialization_constant<SC1>(sc_set[1]);
        auto acc = buf.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelAAA>(
            [=](kernel_handler kh) SYCL_ESIMD_KERNEL {
              auto SC0Val = kh.get_specialization_constant<SC0>();
              auto SC1Val = kh.get_specialization_constant<SC1>();
              sycl::ext::intel::esimd::scalar_store(acc, i * sizeof(int),
                                                    SC0Val + SC1Val);
            });
      });
    } catch (sycl::exception &e) {
      std::cout << "*** Exception caught: " << e.what() << "\n";
      return 1;
    }
    int val = vec[i];
    int gold = sc_set[0] + sc_set[1];

    std::cout << "val = " << val << " gold = " << gold << "\n";

    if (val != gold) {
      std::cout << "*** ERROR[" << i << "]: " << val << " != " << gold
                << "(gold)\n";
      passed = false;
    }
  }
  std::cout << (passed ? "passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}

// --- Check that only two JIT compilation happened:
// CHECK-COUNT-2: ---> piProgramBuild
// CHECK-NOT: ---> piProgramBuild
// --- Check that the test completed with expected results:
// CHECK: passed
