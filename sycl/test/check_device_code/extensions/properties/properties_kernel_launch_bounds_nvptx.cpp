// REQUIRES: cuda

// RUN: %clangxx -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -S -Xclang -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK-IR

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;

  constexpr auto Props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::min_work_groups_per_multiprocessor<8>,
      sycl::ext::oneapi::experimental::max_work_groups_per_cluster<4>,
  };

  // CHECK-IR: define{{.*}}void @[[LaunchBoundsKernelFn:.*LaunchBoundsKernel0]](){{.*}} #[[LaunchBoundsAttrs:[0-9]+]]
  Q.single_task<class LaunchBoundsKernel0>(Props, []() {});

  return 0;
}

// CHECK-IR: attributes #[[LaunchBoundsAttrs]] = {
// CHECK-IR-SAME: "sycl-max-work-groups-per-cluster"="4"
// CHECK-IR-SAME: "sycl-min-work-groups-per-multiprocessor"="8"

// CHECK-IR-DAG: !{ptr @[[LaunchBoundsKernelFn]], !"kernel", i32 1}
// CHECK-IR-DAG: !{ptr @[[LaunchBoundsKernelFn]], !"minctasm", i32 8}
// CHECK-IR-DAG: !{ptr @[[LaunchBoundsKernelFn]], !"maxclusterrank", i32 4}
