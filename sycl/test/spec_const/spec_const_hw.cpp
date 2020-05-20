// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// TODO: re-enable after CI drivers are updated to newer which support spec
// constants:
// XFAIL: linux && opencl
// UNSUPPORTED: cuda
//
//==----------- spec_const_hw.cpp ------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The test checks that the specialization constant feature works correctly -
// tool chain processes them correctly and runtime can correctly execute the
// program.

#include <CL/sycl.hpp>

#include <iostream>
#include <vector>

class MyInt32Const;
class MyFloatConst;

using namespace sycl;

class KernelAAAi;
class KernelBBBf;

int val = 10;

// Fetch a value at runtime.
int get_value() { return val; }

float foo(
    const cl::sycl::experimental::spec_constant<float, MyFloatConst> &f32) {
  return f32;
}

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
  cl::sycl::program program1(q.get_context());
  cl::sycl::program program2(q.get_context());

  int goldi = (int)get_value();
  // TODO make this floating point once supported by the compiler
  float goldf = (float)get_value();

  cl::sycl::experimental::spec_constant<int32_t, MyInt32Const> i32 =
      program1.set_spec_constant<MyInt32Const>(goldi);

  cl::sycl::experimental::spec_constant<float, MyFloatConst> f32 =
      program2.set_spec_constant<MyFloatConst>(goldf);

  program1.build_with_kernel_type<KernelAAAi>();
  // Use an option (does not matter which exactly) to test different internal
  // SYCL RT execution path
  program2.build_with_kernel_type<KernelBBBf>("-cl-fast-relaxed-math");

  std::vector<int> veci(1);
  std::vector<float> vecf(1);
  try {
    cl::sycl::buffer<int, 1> bufi(veci.data(), veci.size());
    cl::sycl::buffer<float, 1> buff(vecf.data(), vecf.size());

    q.submit([&](cl::sycl::handler &cgh) {
      auto acci = bufi.get_access<cl::sycl::access::mode::write>(cgh);
      cgh.single_task<KernelAAAi>(
          program1.get_kernel<KernelAAAi>(),
          [=]() {
            acci[0] = i32.get();
          });
    });
    q.submit([&](cl::sycl::handler &cgh) {
      auto accf = buff.get_access<cl::sycl::access::mode::write>(cgh);
      cgh.single_task<KernelBBBf>(
          program2.get_kernel<KernelBBBf>(),
          [=]() {
            accf[0] = foo(f32);
          });
    });
  } catch (cl::sycl::exception &e) {
    std::cout << "*** Exception caught: " << e.what() << "\n";
    return 1;
  }
  bool passed = true;
  int vali = veci[0];

  if (vali != goldi) {
    std::cout << "*** ERROR: " << vali << " != " << goldi << "(gold)\n";
    passed = false;
  }
  int valf = vecf[0];

  if (valf != goldf) {
    std::cout << "*** ERROR: " << valf << " != " << goldf << "(gold)\n";
    passed = false;
  }
  std::cout << (passed ? "passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
