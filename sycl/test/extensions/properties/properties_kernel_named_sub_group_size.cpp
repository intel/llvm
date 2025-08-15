// RUN: %clangxx -fsycl-device-only -S -Xclang -emit-llvm %s -o - | FileCheck %s
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics
#include <sycl/sycl.hpp>

struct SGSizePrimaryKernelFunctor {
  SGSizePrimaryKernelFunctor() {}

  void operator()(sycl::nd_item<1>) const {}

  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size_primary};
  }
};

struct SGSizeAutoKernelFunctor {
  SGSizeAutoKernelFunctor() {}

  void operator()(sycl::nd_item<1>) const {}

  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size_automatic};
  }
};

int main() {
  sycl::queue Q;
  sycl::nd_range<1> NDRange{6, 2};

  // CHECK: spir_kernel void @{{.*}}SGSizePrimaryKernelFunctor()
  // CHECK-SAME: !intel_reqd_sub_group_size ![[SGSizeAttr:[0-9]+]]
  Q.parallel_for(NDRange, SGSizePrimaryKernelFunctor{});

  // CHECK: spir_kernel void @{{.*}}SGSizeAutoKernelFunctor()
  // CHECK-NOT: intel_reqd_sub_group_size
  // CHECK-SAME: {
  Q.parallel_for(NDRange, SGSizeAutoKernelFunctor{});
}

// CHECK: ![[SGSizeAttr]] = !{i32 -1}
