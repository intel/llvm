// RUN: %clangxx -fsycl -fsycl-device-only %s -O3 -S -o - | FileCheck %s
//
// This test checks that shift_group_left (which is _Z32__spirv_SubgroupShuffleDownINTELIiET_S0_S0_j in SPIRV)
// is called twice after O3 optimizations.
//
// Usually clang with SimplifyCFG pass optimizes constructs like:
// if (i % 2 == 0)
//   func();
// else
//   func();
//
// into one simple func() invocation.
// This behaviour might be wrong in cases when func's behaviour depends on
// a place where it is written.
// There is a relevant discussion about introducing
// a reliable tool for such cases: https://reviews.llvm.org/D85603

// CHECK: {{.*}} call spir_func i32 @_Z32__spirv_SubgroupShuffleDownINTELIiET_S0_S0_j
// CHECK: {{.*}} call spir_func i32 @_Z32__spirv_SubgroupShuffleDownINTELIiET_S0_S0_j


#include <sycl.hpp>

int main() {
  sycl::queue q;
  int* output = sycl::malloc_shared<int>(1, q);
  q.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1> it){
    int i = it.get_global_id(0);
    if (i % 2 == 0) {
      output[0] = sycl::shift_group_left(it.get_sub_group(), 1, 1);
    } else {
      output[0] = sycl::shift_group_left(it.get_sub_group(), 1, 1);
    }
  }).wait();
}
