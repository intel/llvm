// RUN: %{build} %debug_option -Ofast -o %t.out
// RUN: %{build} %debug_option -Os -o %t.out
// RUN: %{build} %debug_option -Oz -o %t.out
// RUN: %{build} %debug_option -Og -o %t.out
// RUN: %{build} %debug_option -O -o %t.out

// NOTE: Tests that debugging information can be generated for all special-name
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
