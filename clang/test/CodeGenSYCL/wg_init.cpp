// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

#include "Inputs/sycl.hpp"

template <typename KernelName, typename KernelType>
__attribute__((sycl_kernel)) void
kernel_parallel_for_work_group(const KernelType &KernelFunc) {
  cl::sycl::group<1> G;
  KernelFunc(G);
}

int main() {

  kernel_parallel_for_work_group<class kernel>([=](cl::sycl::group<1> G) {
    int WG_VAR = 10;
  });
  // CHECK: @{{.*}}WG_VAR = internal addrspace(3) global {{.*}} undef, {{.*}}

  return 0;
}
