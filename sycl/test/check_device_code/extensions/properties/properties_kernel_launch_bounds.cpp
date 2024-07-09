// RUN: %clangxx -fsycl-device-only -S -Xclang -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK-IR
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;

  constexpr auto Props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::min_work_groups_per_multiprocessor<8>,
      sycl::ext::oneapi::experimental::max_work_groups_per_cluster<4>,
  };
  // CHECK-IR: spir_kernel void @{{.*}}LaunchBoundsKernel(){{.*}} #[[LaunchBoundsAttrs:[0-9]+]]
  // CHECK-IR-SAME: !max_work_groups_per_cluster ![[MaxWGsPerCMD:[0-9]+]]
  // CHECK-IR-SAME: !min_work_groups_per_multiprocessor ![[MinWGsPerMPMD:[0-9]+]]
  Q.single_task<class LaunchBoundsKernel>(Props, []() {});

  return 0;
}

// CHECK-IR: attributes #[[LaunchBoundsAttrs]] = {
// CHECK-IR-SAME: "sycl-max-work-groups-per-cluster"="4"
// CHECK-IR-SAME: "sycl-min-work-groups-per-multiprocessor"="8"

// CHECK-IR: ![[MaxWGsPerCMD]] = !{i32 4}
// CHECK-IR: ![[MinWGsPerMPMD]] = !{i32 8}
