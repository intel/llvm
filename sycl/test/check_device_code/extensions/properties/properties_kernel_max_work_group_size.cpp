// RUN: %clangxx -fsycl-device-only -S -Xclang -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK-IR
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  sycl::event Ev;

  constexpr auto Props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::max_work_group_size<8, 4, 2>};

  // CHECK-IR: spir_kernel void @{{.*}}MaxWGSizeKernel0(){{.*}} #[[MaxWGSizeAttr1:[0-9]+]]
  // CHECK-IR-SAME: !max_work_group_size ![[MaxWGSizeMD1:[0-9]+]]
  Q.single_task<class MaxWGSizeKernel0>(Props, []() {});
  // CHECK-IR: spir_kernel void @{{.*}}MaxWGSizeKernel1(){{.*}} #[[MaxWGSizeAttr1]]
  // CHECK-IR-SAME: !max_work_group_size ![[MaxWGSizeMD1]]
  Q.single_task<class MaxWGSizeKernel1>(Ev, Props, []() {});
  // CHECK-IR: spir_kernel void @{{.*}}MaxWGSizeKernel2(){{.*}} #[[MaxWGSizeAttr1]]
  // CHECK-IR-SAME: !max_work_group_size ![[MaxWGSizeMD1]]
  Q.single_task<class MaxWGSizeKernel2>({Ev}, Props, []() {});

  return 0;
}

// CHECK-IR: attributes #[[MaxWGSizeAttr1]] = { {{.*}}"sycl-max-work-group-size"="8,4,2"

// CHECK-IR: ![[MaxWGSizeMD1]] = !{i64 2, i64 4, i64 8}
