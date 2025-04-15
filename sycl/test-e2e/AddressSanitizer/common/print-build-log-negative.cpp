// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_asan_flags -DGPU -o %t_gpu.out
// RUN: %{build} %device_asan_flags -o %t.out
// RUN: %{run} not --crash %if gpu %{ %t_gpu.out %} %else %{ %t.out %} 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>

void test() {
  sycl::queue Q;
  sycl::buffer<int> A{1};
  Q.submit([&](sycl::handler &h) {
    sycl::accessor A_acc(A, h);

    h.single_task([=]() { A_acc[0] = 88; });
  });
}

// CHECK-NOT: <SANITIZER>[ERROR]: Printing build log for program

int main() {
  test();
  return 0;
}
