// RUN: %clangxx -fsycl-device-only -S -Xclang -emit-llvm %s -o - | FileCheck %s
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics
#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  sycl::nd_range<1> ndr{6, 2};

  // CHECK: spir_kernel void @{{.*}}Kernel1()
  // CHECK-SAME: !intel_reqd_sub_group_size ![[SGSizeAttr:[0-9]+]]
  sycl::ext::oneapi::experimental::properties P1{
      sycl::ext::oneapi::experimental::sub_group_size_primary};
  q.parallel_for<class Kernel1>(ndr, P1, [=](auto id) {});

  // CHECK: spir_kernel void @{{.*}}Kernel2()
  // CHECK-NOT: intel_reqd_sub_group_size
  // CHECK-SAME: {
  sycl::ext::oneapi::experimental::properties P2{
      sycl::ext::oneapi::experimental::sub_group_size_automatic};
  q.parallel_for<class Kernel2>(ndr, P2, [=](auto id) {});
}

// CHECK: ![[SGSizeAttr]] = !{i32 0}