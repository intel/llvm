// RUN: %{build} %debug_option -O0 -o %t.out
// RUN: %{build} %debug_option -O1 -o %t.out
// RUN: %{build} %debug_option -O2 -o %t.out
// RUN: %{build} %debug_option -O3 -o %t.out

// NOTE: Tests that debugging information can be generated for all integral
// optimization levels.

#include <sycl/detail/core.hpp>

int main() {
  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class kernel_test>(100, [=](sycl::id<1> idx) {});
  });
  q.wait();

  return 0;
}
