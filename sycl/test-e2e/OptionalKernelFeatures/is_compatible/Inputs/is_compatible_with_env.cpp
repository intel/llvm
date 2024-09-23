#include <sycl/detail/core.hpp>

int main() {
  sycl::device dev;
  if (sycl::is_compatible<class Kernel>(dev)) {
    sycl::queue q(dev);
    q.submit([&](sycl::handler &cgh) {
       cgh.parallel_for<class Kernel>(sycl::range<1>{1},
                                      [=](sycl::id<1> Id) { int x = Id[0]; });
     }).wait_and_throw();
    return 0;
  }
  return 1;
}
