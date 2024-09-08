// End-to-End test for testing device image compression.
// RUN: %{build} -O0 -g -o %t_not_compress.out
// RUN: %{build} -O0 -g --offload-compress -o %t_compress.out
// RUN: %{run} %t_not_compress.out
// RUN: %{run} %t_compress.out
// RUN: not diff %t_not_compress.out %t_compress.out

#include <sycl/detail/core.hpp>

int main() {

  sycl::queue q0;
  int val = -1;
  {
    sycl::buffer<int, 1> buffer1(&val, sycl::range(1));

    q0.submit([&](sycl::handler &cgh) {
        auto acc = sycl::accessor(buffer1, cgh);
        cgh.single_task([=] { acc[0] = acc[0] + 1; });
      }).wait();
  }

  return !(val == 0);
}