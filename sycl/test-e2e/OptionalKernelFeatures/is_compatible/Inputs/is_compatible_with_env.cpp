#include <sycl/sycl.hpp>

int main() {
  sycl::device dev;
  // Should not throw any exception as it should only run on the specific
  // target device, defined during compilation.
  if (sycl::is_compatible<class Kernel>(dev)) {
    sycl::queue q(dev);
    q.submit([&](sycl::handler &cgh) {
       cgh.parallel_for<class Kernel>(sycl::range<1>{1},
                                      [=](sycl::id<1> Id) { int x = Id[0]; });
     }).wait_and_throw();
  }
  return 0;
}
