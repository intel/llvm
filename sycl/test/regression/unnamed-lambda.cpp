// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-device-only -c %s -o %t.temp

#include "CL/sycl.hpp"

void foo(cl::sycl::queue queue) {
  cl::sycl::event queue_event2 = queue.submit([&](cl::sycl::handler &cgh) {
    cgh.parallel_for<class K1>(cl::sycl::range<1>{1},
                                           [=](cl::sycl::item<1> id) {});
  });
}
