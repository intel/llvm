// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

// This test checks all possible scenarios of running single_task, parallel_for
// and parallel_for_work_group to verify that this code compiles and runs
// correctly with user's lambda with and without sycl::kernel_handler argument

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;

  // single_task w/o kernel_handler arg
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class KernelSingleTaskWOKernelHandler>([=]() {});
  });

  // single_task with kernel_handler arg
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class KernelSingleTaskWithKernelHandler>(
        [=](sycl::kernel_handler kh) {});
  });

  // parallel_for with id and w/o kernel_handler arg
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class KernelParallelForIdWithoutKernelHandler>(
        sycl::range<1>(1), [](sycl::id<1> i) {});
  });

  // parallel_for with id and kernel_handler args
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class KernelParallelForIdWithKernelHandler>(
        sycl::range<1>(1), [](sycl::id<1> i, sycl::kernel_handler kh) {});
  });

  // parallel_for with item and w/o kernel_handler arg
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class KernelParallelForItemWithoutKernelHandler>(
        sycl::range<3>(3, 3, 3), [](sycl::item<3> it) {});
  });

  // parallel_for with item and kernel_handler args
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class KernelParallelForItemWithKernelHandler>(
        sycl::range<3>(3, 3, 3),
        [](sycl::item<3> it, sycl::kernel_handler kh) {});
  });

  // parallel_for with nd_item and w/o kernel_handler arg
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class KernelParallelForNDItemWithoutKernelHandler>(
        sycl::nd_range<3>(sycl::range<3>(4, 4, 4), sycl::range<3>(2, 2, 2)),
        [=](sycl::nd_item<3> item) {});
  });

  // parallel_for with nd_item and kernel_handler args
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class KernelParallelForNDItemWithKernelHandler>(
        sycl::nd_range<3>(sycl::range<3>(4, 4, 4), sycl::range<3>(2, 2, 2)),
        [=](sycl::nd_item<3> item, sycl::kernel_handler kh) {});
  });

  // parallel_for with generic lambda w/o kernel_handler arg
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class KernelParallelForGenericLambdaWithoutKernelHandler>(
        sycl::range<3>(3, 3, 3), [](auto it) {});
  });

  // parallel_for with generic lambda with kernel_handler arg
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class KernelParallelForGenericLambdaWithKernelHandler>(
        sycl::range<3>(3, 3, 3), [](auto it, sycl::kernel_handler kh) {});
  });

  // parallel_for with integral type arg and w/o kernel_handler arg
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class KernelParallelForIntWithoutKernelHandler>(
        sycl::range<1>(1), [](int index) {});
  });

  // parallel_for with integral type and kernel_handler args
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class KernelParallelForIntWithKernelHandler>(
        sycl::range<1>(1), [](int index, sycl::kernel_handler kh) {});
  });

  // parallel_for_work_group without kernel_handler arg
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for_work_group<
        class KernelParallelForWorkGroupWithKernelHandler>(
        sycl::range<3>(2, 2, 2), sycl::range<3>(2, 2, 2),
        [=](sycl::group<3> myGroup) {
          myGroup.parallel_for_work_item([&](sycl::h_item<3> myItem) {});
          myGroup.parallel_for_work_item([&](sycl::h_item<3> myItem) {});
        });
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
