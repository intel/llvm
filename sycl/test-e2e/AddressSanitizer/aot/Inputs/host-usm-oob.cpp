#include <sycl/detail/core.hpp>

#include <sycl/usm.hpp>

int main() {
  sycl::queue Q;
  constexpr std::size_t N = 12;
  auto *array = sycl::malloc_host<int>(N, Q);
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernelR_4>(
        sycl::nd_range<1>(N + 1, 1),
        [=](sycl::nd_item<1> item) { ++array[item.get_global_id(0)]; });
  });
  Q.wait();
  // CHECK:   ERROR: DeviceSanitizer: out-of-bounds-access on Host USM
  // CHECK: {{READ of size 4 at kernel <.*MyKernelR_4> LID\(0, 0, 0\) GID\(12, 0, 0\)}}
  // CHECK: {{#0 .*}}[[@LINE-5]]

  sycl::free(array, Q);
  return 0;
}
