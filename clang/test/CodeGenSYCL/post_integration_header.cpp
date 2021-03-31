// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -fsycl-post-int-header=%t.h %s -emit-llvm -o %t.ll
// RUN: FileCheck -input-file=%t.h %s

// CHECK: // Post Integration Header contents to go here.

#include "Inputs/sycl.hpp"

int main() {
  cl::sycl::kernel_single_task<class first_kernel>([]() {});
}
