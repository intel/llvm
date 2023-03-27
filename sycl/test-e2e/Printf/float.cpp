// This test is written with an aim to check that experimental::printf behaves
// in the same way as printf from C99/C11
//
// The test is written using conversion specifiers table from cppreference [1]
// [1]: https://en.cppreference.com/w/cpp/io/c/fprintf
//
// UNSUPPORTED: hip_amd
// XFAIL: cuda && windows
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER
// FIXME: Remove dedicated non-variadic printf testing once the headers
//        enforce it by default.
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.nonvar.out \
// RUN: -D__SYCL_USE_NON_VARIADIC_SPIRV_OCL_PRINTF__
// RUN: %CPU_RUN_PLACEHOLDER %t.nonvar.out %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER %t.nonvar.out %GPU_CHECK_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER %t.nonvar.out %ACC_CHECK_PLACEHOLDER
// FIXME: Remove dedicated constant address space testing once generic AS
//        support is considered stable.
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.constant.out \
// RUN: -DTEST_CONSTANT_AS
// RUN: %CPU_RUN_PLACEHOLDER %t.constant.out %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER %t.constant.out %GPU_CHECK_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER %t.constant.out %ACC_CHECK_PLACEHOLDER
//
// CHECK: 3.140000e+00, 3.140000E+00
// CHECK: 0x1.91eb86{{0*}}p+1, 0X1.91EB86{{0*}}P+1
// CHECK: 3.14, 3.14

#include <iostream>

#include <sycl/sycl.hpp>

#include "helper.hpp"

using namespace sycl;

void do_float_test() {
  float f = 3.14;
  // %e, %E floating-point, decimal exponent notation
  FORMAT_STRING(fmt1) = "float %e, %E\n";
  ext::oneapi::experimental::printf(fmt1, f, f);
  // %a, %A floating-point, hexadecimal exponent notation
  FORMAT_STRING(fmt2) = "float %a, %A\n";
  ext::oneapi::experimental::printf(fmt2, f, f);
  // %g, %G floating-point
  FORMAT_STRING(fmt3) = "float %g, %G\n";
  ext::oneapi::experimental::printf(fmt3, f, f);
}

class FloatTest;

int main() {
  queue q;

#ifndef __SYCL_USE_NON_VARIADIC_SPIRV_OCL_PRINTF__
  if (!q.get_device().has(aspect::fp64)) {
    std::cout << "Skipping the actual test due to variadic argument promotion. "
                 "Printing hard-coded output from the host side:\n"
              << "3.140000e+00, 3.140000E+00\n"
                 "0x1.91eb86p+1, 0X1.91EB86P+1\n"
                 "3.14, 3.14"
              << std::endl;
    return 0;
  }
#endif // !__SYCL_USE_NON_VARIADIC_SPIRV_OCL_PRINTF__
  q.submit([](handler &cgh) {
    cgh.single_task<FloatTest>([]() { do_float_test(); });
  });
  q.wait();
  return 0;
}
