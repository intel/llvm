// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-device-only -c %s -o %t.temp

// This validates that the unnamed lambda logic in the library correctly works
// with a new implementation of __builtin_unique_stable_name, where
// instantiation order matters.  parallel_for instantiates the KernelInfo before
// the kernel itself, so this checks that example, which only happens when the
// named kernel is inside another lambda.

#include "CL/sycl.hpp"

void foo(cl::sycl::queue queue) {
  cl::sycl::event queue_event2 = queue.submit([&](cl::sycl::handler &cgh) {
    cgh.parallel_for<class K1>(cl::sycl::range<1>{1},
                                           [=](cl::sycl::item<1> id) {});
  });
}
