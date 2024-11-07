// REQUIRES: linux
// RUN: %{build} %device_asan_flags -O2 -fsanitize-ignorelist=%p/ignorelist.txt -o %t
// RUN: %{run} not %t &> %t.txt ; FileCheck --input-file %t.txt %s
// RUN: %{build} %device_asan_flags %if cpu %{ -fsycl-targets=spir64_x86_64 %} %if gpu %{ -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %} -O2 -fsanitize-ignorelist=%p/ignorelist.txt -o %t2
// RUN: %{run} not %t2 &> %t2.txt ; FileCheck --input-file %t2.txt %s

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
  std::cout << "PASS" << std::endl;
  return 0;
}

// CHECK: PASS
