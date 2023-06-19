// RUN: %clangxx %s -S -o %t_opt.ll -fsycl-device-only -Xclang -verify -Xclang -verify-ignore-unexpected=note
// RUN: FileCheck %s --input-file %t_opt.ll --check-prefix=CHECK-OPT
// RUN: %clangxx %s -S -fno-sycl-early-optimizations -o %t_noopt.ll -fsycl-device-only -Xclang -verify -Xclang -verify-ignore-unexpected=note
// RUN: FileCheck %s --input-file %t_noopt.ll --check-prefix=CHECK-NOOPT

// Tests that an optimization that removes the use of double still produces a
// warning.

// CHECK-OPT-NOT: double
// CHECK-NOOPT: double

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  // expected-warning-re@+1 {{function '{{.*}}' uses aspect 'fp64' not listed in its 'sycl::device_has' attribute}}
  Q.single_task([=]() [[sycl::device_has()]] {
    // Double will be optimized out as LoweredFloat can be set directly to a
    // lowered value.
    double Double = 3.14;
    volatile float LoweredFloat = Double;
  });
  return 0;
}
