// RUN: %clangxx -fsycl-device-only -S -Xclang -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK-IR
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;

  constexpr auto Props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::max_linear_work_group_size<4>,
  };
  // CHECK-IR: spir_kernel void @{{.*}}LaunchBoundsKernel(){{.*}} #[[LaunchBoundsAttrs:[0-9]+]]
  Q.single_task<class LaunchBoundsKernel>(Props, []() {});

  return 0;
}

// CHECK-IR: attributes #[[LaunchBoundsAttrs]] = {
// CHECK-IR-SAME: "sycl-max-linear-work-group-size"="4"
