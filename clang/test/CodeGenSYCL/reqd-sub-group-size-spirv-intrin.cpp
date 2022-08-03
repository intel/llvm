// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -internal-isystem %S/Inputs -fdeclare-spirv-builtins %s -emit-llvm -o - | FileCheck %s

// Test that when __spirv intrinsics are invoked from kernel functions
// that have a sub_group_size specified, that such invocations don't
// trigger the error diagnostic that the intrinsic routines must also
// marked with the same attribute.

#include "Inputs/sycl.hpp"

int main() {
  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
    auto kernel_ = [=](sycl::group<1> item) [[intel::sub_group_size(8)]] {
    };

    cgh.parallel_for_work_group<class kernel_class>(
        cl::sycl::range<1>(), cl::sycl::range<1>(), kernel_);
  });
  return 0;
}

// CHECK: define dso_local spir_kernel void @{{.*}}main{{.*}}kernel_class() {{.*}} !intel_reqd_sub_group_size ![[SUBGROUPSIZE:[0-9]+]]
// CHECK: tail call spir_func void @{{.*}}__spirv_ControlBarrier{{.*}}({{.*}})

// CHECK: declare spir_func void @{{.*}}__spirv_ControlBarrier{{.*}}({{.*}})

// CHECK: ![[SUBGROUPSIZE]] = !{i32 8}
