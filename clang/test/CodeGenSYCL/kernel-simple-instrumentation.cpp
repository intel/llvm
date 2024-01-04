// Check if start/finish ITT annotations are being added during compilation of
// SYCL device code

// RUN: %clang_cc1 -fsycl-is-device -fsycl-instrument-device-code -triple spir64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// CHECK: kernel_function
// CHECK-NEXT: entry:
// CHECK-NEXT: call spir_func void @__itt_offload_wi_start_wrapper()
// CHECK: call spir_func void @__itt_offload_wi_finish_wrapper()
// CHECK-NEXT: ret void

#include "Inputs/sycl.hpp"

int main() {
  sycl::accessor<int, 1, sycl::access::mode::read_write> accessorA;
  sycl::kernel_single_task<class kernel_function>(
      [=]() {
        accessorA.use();
      });
  return 0;
}
