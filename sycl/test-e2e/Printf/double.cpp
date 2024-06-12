// This test is written with an aim to check that experimental::printf behaves
// in the same way as printf from C99/C11
//
// The test is written using conversion specifiers table from cppreference [1]
// [1]: https://en.cppreference.com/w/cpp/io/c/fprintf
//
// REQUIRES: aspect-fp64
// Temporarily disable test on Windows due to regressions in GPU driver.
// UNSUPPORTED: hip_amd, windows
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out | FileCheck %s
// FIXME: Remove dedicated constant address space testing once generic AS
//        support is considered stable.
// RUN: %{build} -o %t.constant.out -DTEST_CONSTANT_AS
// RUN: %{run} %t.constant.out | FileCheck %s
//
// CHECK: double -6.813800e+00, -6.813800E+00
// CHECK: mixed 3.140000e+00, -6.813800E+00
// CHECK: double -0x1.b4154d8cccccdp+2, -0X1.B4154D8CCCCCDP+2
// CHECK: mixed 0x1.91eb86{{0*}}p+1, -0X1.B4154D8CCCCCDP+2
// CHECK: double -6.8138, -6.8138
// CHECK: mixed 3.14, -6.8138

#include <iostream>

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/builtins.hpp>

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

  q.submit([](handler &cgh) {
    cgh.single_task<DoubleTest>([]() { do_double_test(); });
  });
  q.wait();

  return 0;
}
