// RUN: %clangxx -fsycl -D__SYCL_INTERNAL_API %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// Specialization constants are not supported on FPGA h/w and emulator.
// UNSUPPORTED: cuda || hip
//
//==----------- spec_const_hw.cpp ------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The test that specialization constant implementation throws exceptions when
// expected.

#include <sycl/sycl.hpp>

#include <iostream>
#include <vector>

class MyInt32Const;

using namespace sycl;

class KernelAAAi;

int main(int argc, char **argv) {
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
  cl::sycl::program program1(q.get_context());

  cl::sycl::ext::oneapi::experimental::spec_constant<int32_t, MyInt32Const>
      i32 = program1.set_spec_constant<MyInt32Const>(10);

  std::vector<int> veci(1);
  bool passed = false;

  program1.build_with_kernel_type<KernelAAAi>();

  try {
    // This is an attempt to set a spec constant after the program has been
    // built - spec_const_error should be thrown
    cl::sycl::ext::oneapi::experimental::spec_constant<int32_t, MyInt32Const>
        i32 = program1.set_spec_constant<MyInt32Const>(10);

    cl::sycl::buffer<int, 1> bufi(veci.data(), veci.size());

    q.submit([&](cl::sycl::handler &cgh) {
      auto acci = bufi.get_access<cl::sycl::access::mode::write>(cgh);
      cgh.single_task<KernelAAAi>(program1.get_kernel<KernelAAAi>(),
                                  [=]() { acci[0] = i32.get(); });
    });
  } catch (cl::sycl::ext::oneapi::experimental::spec_const_error &sc_err) {
    passed = true;
  } catch (cl::sycl::exception &e) {
    std::cout << "*** Exception caught: " << e.what() << "\n";
    return 1;
  }
  std::cout << (passed ? "passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
