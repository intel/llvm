// REQUIRES: cuda

// RUN: %clangxx -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -S -Xclang -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK-IR

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;

  constexpr auto Props1 = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::max_work_group_size<8>};
  constexpr auto Props2 = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::max_work_group_size<8, 4>};
  constexpr auto Props3 = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::max_work_group_size<8, 4, 2>};

  // CHECK-IR: define{{.*}}void @[[MaxWGSizeKernelFn0:.*MaxWGSizeKernel0]](){{.*}} #[[MaxWGSizeAttr0:[0-9]+]]
  Q.single_task<class MaxWGSizeKernel0>(Props1, []() {});

  // CHECK-IR: define{{.*}}void @[[MaxWGSizeKernelFn1:.*MaxWGSizeKernel1]](){{.*}} #[[MaxWGSizeAttr1:[0-9]+]]
  Q.single_task<class MaxWGSizeKernel1>(Props2, []() {});

  // CHECK-IR: define{{.*}}void @[[MaxWGSizeKernelFn2:.*MaxWGSizeKernel2]](){{.*}} #[[MaxWGSizeAttr2:[0-9]+]]
  Q.single_task<class MaxWGSizeKernel2>(Props3, []() {});

  return 0;
}

// CHECK-IR: attributes #[[MaxWGSizeAttr0]] = {
// CHECK-IR-SAME: "sycl-max-work-group-size"="8"

// CHECK-IR: attributes #[[MaxWGSizeAttr1]] = {
// CHECK-IR-SAME: "sycl-max-work-group-size"="8,4"

// CHECK-IR: attributes #[[MaxWGSizeAttr2]] = {
// CHECK-IR-SAME: "sycl-max-work-group-size"="8,4,2"

// CHECK-IR:     !{ptr @[[MaxWGSizeKernelFn0]], !"kernel", i32 1}
// CHECK-IR:     !{ptr @[[MaxWGSizeKernelFn1]], !"kernel", i32 1}
// CHECK-IR:     !{ptr @[[MaxWGSizeKernelFn2]], !"kernel", i32 1}

// CHECK-IR:     !{ptr @[[MaxWGSizeKernelFn0]], !"maxntidx", i32 8}
// CHECK-IR-NOT: !{ptr @[[MaxWGSizeKernelFn0]], !"maxntidy",
// CHECK-IR-NOT: !{ptr @[[MaxWGSizeKernelFn0]], !"maxntidz",

// CHECK-IR:     !{ptr @[[MaxWGSizeKernelFn1]], !"maxntidx", i32 4}
// CHECK-IR:     !{ptr @[[MaxWGSizeKernelFn1]], !"maxntidy", i32 8}
// CHECK-IR-NOT: !{ptr @[[MaxWGSizeKernelFn1]], !"maxntidz",

// CHECK-IR:     !{ptr @[[MaxWGSizeKernelFn2]], !"maxntidx", i32 2}
// CHECK-IR:     !{ptr @[[MaxWGSizeKernelFn2]], !"maxntidy", i32 4}
// CHECK-IR:     !{ptr @[[MaxWGSizeKernelFn2]], !"maxntidz", i32 8}
