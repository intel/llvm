// RUN: %clangxx -fsycl-device-only -S -Xclang -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK-IR
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

constexpr auto Props = sycl::ext::oneapi::experimental::properties{
    sycl::ext::oneapi::experimental::max_linear_work_group_size<4>,
};
struct TestKernelLaunchBounds {
  void operator()() const {}
  auto get(sycl::ext::oneapi::experimental::properties_tag) { return Props; }
};

int main() {
  sycl::queue Q;

  // CHECK-IR: spir_kernel void @{{.*}}LaunchBoundsKernel(){{.*}} #[[LaunchBoundsAttrs:[0-9]+]]
  Q.submit([&](sycl::handler &h) {
    h.single_task<class LaunchBoundsKernel>(TestKernelLaunchBounds{});
  });

  return 0;
}

// CHECK-IR: attributes #[[LaunchBoundsAttrs]] = {
// CHECK-IR-SAME: "sycl-max-linear-work-group-size"="4"
