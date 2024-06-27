// RUN: %clangxx -fsycl-device-only -Xclang -fsycl-is-device -emit-llvm %s -S -o %t.ll -I %sycl_include
// RUN: FileCheck %s --input-file %t.ll

// Check copying of parallel_for kernel attributes to wrapper kernel.

#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  range<1> Size{10};
  {
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      cgh.parallel_for<class C>(Size, [=](item<1> ITEM)
                                          [[sycl::reqd_work_group_size(4)]] {});
    });
  }

  return 0;
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}__pf_kernel_wrapper{{.*}}reqd_work_group_size
