// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -fsycl-int_footer=%t.h %s -emit-llvm -o %t.ll
// RUN: FileCheck -input-file=%t.h %s

// CHECK: // Integration Footer contents to go here.

#include "Inputs/sycl.hpp"

nt main() {
  cl::sycl::kernel_single_task<class first_kernel>([]() {});
}
