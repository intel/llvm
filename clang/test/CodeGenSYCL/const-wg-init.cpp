// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

#include "Inputs/sycl.hpp"

template <typename KernelName, typename KernelType>
__attribute__((sycl_kernel)) void
kernel_parallel_for_work_group(const KernelType &KernelFunc) {
  sycl::group<1> G;
  KernelFunc(G);
}

int main() {

  kernel_parallel_for_work_group<class kernel>([=](sycl::group<1> G) {
    const int WG_CONST = 10;
  });
  // CHECK:  store i32 10, ptr addrspace(4) addrspacecast (ptr addrspace(3) @{{.*}}WG_CONST{{.*}} to ptr addrspace(4))
  return 0;
}
