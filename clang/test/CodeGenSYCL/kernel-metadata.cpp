// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -emit-llvm %s -o - | FileCheck %s

// CHECK-NOT: define {{.*}}spir_kernel void @{{.*}}kernel_function{{.*}} !kernel_arg_addr_space {{.*}} !kernel_arg_access_qual {{.*}} !kernel_arg_type {{.*}} !kernel_arg_base_type {{.*}} !kernel_arg_type_qual {{.*}}

#include "Inputs/sycl.hpp"

int main() {
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write> accessorA;
  cl::sycl::kernel_single_task<class kernel_function>(
      [=]() {
        accessorA.use();
      });
  return 0;
}
