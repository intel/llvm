// UNSUPPORTED: cuda || hip
//
// FIXME Disable fallback assert so that it doesn't interferes with number of
// program builds at run-time
// RUN: %clangxx -DSYCL_DISABLE_FALLBACK_ASSERT -D__SYCL_INTERNAL_API -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out 2>&1 %CPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
//
//==----------- spec_const_redefine.cpp ------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The test checks that:
// - a specialization constant can be redifined and correct new value is used
//   after redefinition.
// - the program is JITted only once per a unique set of specialization
//   constants values.

#include <sycl/sycl.hpp>

#include <iostream>
#include <vector>

class SC0;
class SC1;
class KernelAAA;

using namespace sycl;

int val = 0;

// Fetch a value at runtime.
int get_value() { return val; }

int main(int argc, char **argv) {
  val = argc;

  cl::sycl::queue q(default_selector{}, [](exception_list l) {
    for (auto ep : l) {
      try {
        std::rethrow_exception(ep);
      } catch (cl::sycl::exception &e0) {
        std::cout << e0.what();
      } catch (std::exception &e1) {
        std::cout << e1.what();
      } catch (...) {
        std::cout << "*** catch (...)\n";
      }
    }
  });

  std::cout << "Running on " << q.get_device().get_info<info::device::name>()
            << "\n";
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
    cl::sycl::program program(q.get_context());
    const int *sc_set = &sc_vals[i][0];
    cl::sycl::ext::oneapi::experimental::spec_constant<int32_t, SC0> sc0 =
        program.set_spec_constant<SC0>(sc_set[0]);
    cl::sycl::ext::oneapi::experimental::spec_constant<int32_t, SC1> sc1 =
        program.set_spec_constant<SC1>(sc_set[1]);

    program.build_with_kernel_type<KernelAAA>();

    try {
      cl::sycl::buffer<int, 1> buf(vec.data(), vec.size());

      q.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::write>(cgh);
        cgh.single_task<KernelAAA>(program.get_kernel<KernelAAA>(),
                                   [=]() { acc[i] = sc0.get() + sc1.get(); });
      });
    } catch (cl::sycl::exception &e) {
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
// CHECK-NOT: ---> piProgramBuild
// CHECK: ---> piProgramBuild
// CHECK: ---> piProgramBuild
// CHECK-NOT: ---> piProgramBuild
// --- Check that the test completed with expected results:
// CHECK: passed
