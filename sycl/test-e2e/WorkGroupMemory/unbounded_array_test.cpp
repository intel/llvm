// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/work_group_memory.hpp>

// This test checks that work group memory objects can be used with unbounded
// arrays as template arguments.

// Unbounded array support is not yet implemented for work group memory
// due to a LLVM IR <-> SPIRV translation problem.
// XFAIL: *

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
    sycl::ext::oneapi::experimental::work_group_memory<int[]> data{16, cgh};
    cgh.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1> it) {
      for (int i = 0; i < 16; ++i)
        data[i] = 42;
    });
  });
}
