// RUN: %clangxx -fsycl  -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>

int main() {
  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class Kernel1>([=](sycl::kernel_handler kh) {});
  });

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class Kernel2>(
        sycl::range<3>(3, 3, 3),
        [](sycl::item<3> it, sycl::kernel_handler kh) {});
  });

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class Kernel2>(sycl::range<3>(3, 3, 3), [](auto it) {});
  });

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for_work_group<class Kernel3>(
        sycl::range<3>(2, 2, 2), sycl::range<3>(2, 2, 2),
        [=](sycl::group<3> myGroup, sycl::kernel_handler kh) {
          myGroup.parallel_for_work_item([&](sycl::h_item<3> myItem) {});
          myGroup.parallel_for_work_item([&](sycl::h_item<3> myItem) {});
        });
  });
}
