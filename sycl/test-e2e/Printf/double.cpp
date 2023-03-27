// This test is written with an aim to check that experimental::printf behaves
// in the same way as printf from C99/C11
//
// The test is written using conversion specifiers table from cppreference [1]
// [1]: https://en.cppreference.com/w/cpp/io/c/fprintf
//
// Temporarily disable test on Windows due to regressions in GPU driver.
// UNSUPPORTED: hip_amd, windows
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER
// FIXME: Remove dedicated constant address space testing once generic AS
//        support is considered stable.
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.constant.out \
// RUN: -DTEST_CONSTANT_AS
// RUN: %CPU_RUN_PLACEHOLDER %t.constant.out %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER %t.constant.out %GPU_CHECK_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER %t.constant.out %ACC_CHECK_PLACEHOLDER
//
// CHECK: double -6.813800e+00, -6.813800E+00
// CHECK: mixed 3.140000e+00, -6.813800E+00
// CHECK: double -0x1.b4154d8cccccdp+2, -0X1.B4154D8CCCCCDP+2
// CHECK: mixed 0x1.91eb86{{0*}}p+1, -0X1.B4154D8CCCCCDP+2
// CHECK: double -6.8138, -6.8138
// CHECK: mixed 3.14, -6.8138

#include <iostream>

#include <sycl/sycl.hpp>

#include "helper.hpp"

using namespace sycl;

void do_double_test() {
  float f = 3.14;
  double d = -f * 2.17;

  {
    // %e, %E floating-point, decimal exponent notation
    FORMAT_STRING(fmt_double) = "double %e, %E\n";
    FORMAT_STRING(fmt_mixed) = "mixed %e, %E\n";
    ext::oneapi::experimental::printf(fmt_double, d, d);
    ext::oneapi::experimental::printf(fmt_mixed, f, d);
  }

  {
    // %a, %A floating-point, hexadecimal exponent notation
    FORMAT_STRING(fmt_double) = "double %a, %A\n";
    FORMAT_STRING(fmt_mixed) = "mixed %a, %A\n";
    ext::oneapi::experimental::printf(fmt_double, d, d);
    ext::oneapi::experimental::printf(fmt_mixed, f, d);
  }

  {
    // %g, %G floating-point
    FORMAT_STRING(fmt_double) = "double %g, %G\n";
    FORMAT_STRING(fmt_mixed) = "mixed %g, %G\n";
    ext::oneapi::experimental::printf(fmt_double, d, d);
    ext::oneapi::experimental::printf(fmt_mixed, f, d);
  }
}

class DoubleTest;

int main() {
  queue q;

  if (q.get_device().has(aspect::fp64)) {
    q.submit([](handler &cgh) {
      cgh.single_task<DoubleTest>([]() { do_double_test(); });
    });
    q.wait();
  } else
    std::cout << "Skipping the actual test. "
                 "Printing hard-coded output from the host side:\n"
              << "double -6.813800e+00, -6.813800E+00\n"
                 "mixed 3.140000e+00, -6.813800E+00\n"
                 "double -0x1.b4154d8cccccdp+2, -0X1.B4154D8CCCCCDP+2\n"
                 "mixed 0x1.91eb86p+1, -0X1.B4154D8CCCCCDP+2\n"
                 "double -6.8138, -6.8138\n"
                 "mixed 3.14, -6.8138"
              << std::endl;
  return 0;
}
