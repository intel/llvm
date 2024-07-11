// REQUIRES: cuda

// RUN: %clangxx -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -S -Xclang -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK-IR

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;

  constexpr auto Props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::max_work_group_size<8, 4, 2>};

  // CHECK-IR: define{{.*}}void @[[MaxWGSizeKernelFn:.*MaxWGSizeKernel0]](){{.*}} #[[MaxWGSizeAttr1:[0-9]+]]
  Q.single_task<class MaxWGSizeKernel0>(Props, []() {});

  return 0;
}

// CHECK-IR: attributes #[[MaxWGSizeAttr1]] = {
// CHECK-IR-SAME: "sycl-max-work-group-size"="8,4,2"

// CHECK-IR-DAG: !{ptr @[[MaxWGSizeKernelFn]], !"kernel", i32 1}
// CHECK-IR-DAG: !{ptr @[[MaxWGSizeKernelFn]], !"maxntidx", i32 2}
// CHECK-IR-DAG: !{ptr @[[MaxWGSizeKernelFn]], !"maxntidy", i32 4}
// CHECK-IR-DAG: !{ptr @[[MaxWGSizeKernelFn]], !"maxntidz", i32 8}
