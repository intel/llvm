// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/work_group_memory.hpp>

// Check that the work group memory object is mapped to exactly one backend
// kernel argument.

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
    sycl::ext::oneapi::experimental::work_group_memory<int[2]> data{cgh};
    cgh.parallel_for(sycl::nd_range<1>{1, 1},
                     [=](sycl::nd_item<1> it) { data[0] = 42; });
  });
}

// CHECK-COUNT-1: ---> urKernelSetArg
// CHECK-NOT: ---> urKernelSetArg
