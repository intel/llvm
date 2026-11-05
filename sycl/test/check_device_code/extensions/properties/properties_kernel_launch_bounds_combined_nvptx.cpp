// REQUIRES: cuda

// RUN: %clangxx -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_90 -S -Xclang -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK-IR

#include <sycl/sycl.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

struct LaunchBoundsKernel {
  [[intel::min_work_groups_per_cu(2), intel::max_work_groups_per_mp(4)]] void
  operator()(sycl::nd_item<1>) const {}
  auto get(syclex::properties_tag) const {
    return syclex::properties{syclex::max_work_group_size<256>};
  }
};

int main() {
  sycl::queue Q;

  // CHECK-IR: define{{.*}}ptx_kernel{{.*}}void @[[CombinedKernelFn:.*CombinedKernel]]({{.*}} !min_work_groups_per_cu ![[MWGPC:[0-9]+]] !max_work_groups_per_mp ![[MWGPM:[0-9]+]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class CombinedKernel>(sycl::nd_range<1>(256, 256),
                                           LaunchBoundsKernel{});
  });

  // CHECK-IR: define{{.*}}ptx_kernel{{.*}}void @[[LambdaKernelFn:.*LambdaKernel]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class LambdaKernel>(
        sycl::nd_range<1>(128, 128),
        [=](sycl::nd_item<1>) [[intel::min_work_groups_per_cu(3),
                                intel::max_work_groups_per_mp(5)]] {});
  });

  return 0;
}

// CHECK-IR-DAG: !{ptr @[[CombinedKernelFn]], !"maxntidx", i32 256}
// CHECK-IR-DAG: !{ptr @[[LambdaKernelFn]], !"minctasm", i32 3}
// CHECK-IR-DAG: !{ptr @[[LambdaKernelFn]], !"maxclusterrank", i32 5}
// CHECK-IR-DAG: ![[MWGPC]] = !{i32 2}
// CHECK-IR-DAG: ![[MWGPM]] = !{i32 4}
