// REQUIRES: linux, cpu
// RUN: %{build} %device_asan_flags -O2 -g -o %t
// RUN: env SYCL_PREFER_UR=1 %{run} not %t &> %t.txt ; FileCheck --input-file %t.txt %s
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
  // CHECK: kernel <typeinfo name for main::{{.*\(sycl::_V1::handler&\).*}}::operator()(sycl::_V1::handler&) const::MyKernel>

  sycl::free(array, Q);
  return 0;
}
