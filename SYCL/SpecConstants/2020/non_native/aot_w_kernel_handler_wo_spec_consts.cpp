// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

// This test checks correctness of compiling and running of application with
// kernel lambdas containing kernel_handler arguments and w/o usage of
// specialization constants in AOT mode

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class KernelSingleTaskWithKernelHandler>(
        [=](sycl::kernel_handler kh) {});
  });

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class KernelParallelForNDItemWithKernelHandler>(
        sycl::nd_range<3>(sycl::range<3>(4, 4, 4), sycl::range<3>(2, 2, 2)),
        [=](sycl::nd_item<3> item, sycl::kernel_handler kh) {});
  });

  // parallel_for_work_group with kernel_handler arg
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for_work_group<
        class KernelParallelForWorkGroupWithoutKernelHandler>(
        sycl::range<3>(2, 2, 2), sycl::range<3>(2, 2, 2),
        [=](sycl::group<3> myGroup, sycl::kernel_handler kh) {
          myGroup.parallel_for_work_item([&](sycl::h_item<3> myItem) {});
          myGroup.parallel_for_work_item([&](sycl::h_item<3> myItem) {});
        });
  });
}
