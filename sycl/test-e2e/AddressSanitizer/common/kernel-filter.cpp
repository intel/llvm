// REQUIRES: linux
// RUN: %{build} %device_asan_flags -O2 -g -fsanitize-ignorelist=%p/ignorelist.txt -o %t
// RUN: %{run} not %t &> %t.txt ; FileCheck --input-file %t.txt %s
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

int main() {
  sycl::queue Q;
  constexpr std::size_t N = 16;
  auto *array = sycl::malloc_device<int>(N, Q);

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernel>(
        sycl::nd_range<1>(N + 1, 1),
        [=](sycl::nd_item<1> item) { ++array[item.get_global_id(0)]; });
  });
  Q.wait();

  sycl::free(array, Q);
  return 0;
}
