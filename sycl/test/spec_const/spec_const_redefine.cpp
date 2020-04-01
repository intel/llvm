// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// TODO the test currently fails as program recompilation based on spec
//      constants set change is not complete yet.
// XFAIL: *
//
//==----------- spec_const_redefine.cpp ------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The test checks that the specialization constant can be redifined and correct
// new value is used after redefinition.

#include <CL/sycl.hpp>

#include <iostream>
#include <vector>

class MyInt32Const;

using namespace sycl;

class KernelAAAi;

int val = 10;

// Fetch a value at runtime.
int get_value() { return val; }

int main(int argc, char **argv) {
  val = argc + 16;

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
  std::cout << "val = " << val << "\n";
  bool passed = true;

  for (int i = 0; i < 2; i++) {
    cl::sycl::program program(q.get_context());
    int gold = (int)get_value() + i;
    // The same constant - MyInt32Const - is set to different value with the
    // same kernel each loop iteration. SYCL RT must rebuild the underlying
    // program.
    cl::sycl::experimental::spec_constant<int32_t, MyInt32Const> i32 =
        program.set_spec_constant<MyInt32Const>(gold);
    program.build_with_kernel_type<KernelAAAi>();
    std::vector<int> vec(1);

    try {
      cl::sycl::buffer<int, 1> buf(vec.data(), vec.size());

      q.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::write>(cgh);
        cgh.single_task<KernelAAAi>(
            program.get_kernel<KernelAAAi>(),
            [=]() {
              acc[0] = i32.get();
            });
      });
    } catch (cl::sycl::exception &e) {
      std::cout << "*** Exception caught: " << e.what() << "\n";
      return 1;
    }
    int val = vec[0];

    if (val != gold) {
      std::cout << "*** ERROR[" << i << "]: " << val << " != " << gold << "(gold)\n";
      passed = false;
    }
  }
  std::cout << (passed ? "passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
