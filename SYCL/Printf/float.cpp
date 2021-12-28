// This test is written with an aim to check that experimental::printf behaves
// in the same way as printf from C99/C11
//
// The test is written using conversion specifiers table from cppreference [1]
// [1]: https://en.cppreference.com/w/cpp/io/c/fprintf
//
// UNSUPPORTED: hip_amd
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
// CHECK: float 3.140000e+00, 3.140000E+00
// CHECK: double -6.813800e+00, -6.813800E+00
// CHECK: mixed 3.140000e+00, -6.813800E+00
// CHECK: float 0x1.91eb86{{0*}}p+1, 0X1.91EB86{{0*}}P+1
// CHECK: double -0x1.b4154d8cccccdp+2, -0X1.B4154D8CCCCCDP+2
// CHECK: mixed 0x1.91eb86{{0*}}p+1, -0X1.B4154D8CCCCCDP+2
// CHECK: float 3.14, 3.14
// CHECK: double -6.8138, -6.8138
// CHECK: mixed 3.14, -6.8138

#include <CL/sycl.hpp>

#include "helper.hpp"

using namespace sycl;

void do_float_test() {
  {
    // %e, %E floating-point, decimal exponent notation
    FORMAT_STRING(fmt1) = "float %e, %E\n";
    FORMAT_STRING(fmt2) = "double %e, %E\n";
    FORMAT_STRING(fmt3) = "mixed %e, %E\n";

    float f = 3.14;
    double d = -f * 2.17;
    ext::oneapi::experimental::printf(fmt1, f, f);
    ext::oneapi::experimental::printf(fmt2, d, d);
    ext::oneapi::experimental::printf(fmt3, f, d);
  }

  {
    // %a, %A floating-point, hexadecimal exponent notation
    FORMAT_STRING(fmt1) = "float %a, %A\n";
    FORMAT_STRING(fmt2) = "double %a, %A\n";
    FORMAT_STRING(fmt3) = "mixed %a, %A\n";

    float f = 3.14;
    double d = -f * 2.17;
    ext::oneapi::experimental::printf(fmt1, f, f);
    ext::oneapi::experimental::printf(fmt2, d, d);
    ext::oneapi::experimental::printf(fmt3, f, d);
  }

  {
    // %g, %G floating-point
    FORMAT_STRING(fmt1) = "float %g, %G\n";
    FORMAT_STRING(fmt2) = "double %g, %G\n";
    FORMAT_STRING(fmt3) = "mixed %g, %G\n";

    float f = 3.14;
    double d = -f * 2.17;
    ext::oneapi::experimental::printf(fmt1, f, f);
    ext::oneapi::experimental::printf(fmt2, d, d);
    ext::oneapi::experimental::printf(fmt3, f, d);
  }
}

class FloatTest;

int main() {
  queue q;

  q.submit([](handler &cgh) {
    cgh.single_task<FloatTest>([]() { do_float_test(); });
  });
  q.wait();

  return 0;
}
