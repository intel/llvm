// REQUIRES: linux
// RUN: %{build} %device_asan_flags -O2 -fsanitize-ignorelist=%p/ignorelist.txt -o %t
// RUN: %{run} %t 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags %if cpu %{ -fsycl-targets=spir64_x86_64 %} %if gpu %{ -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %} -O2 -fsanitize-ignorelist=%p/ignorelist.txt -o %t2
// RUN: %{run} %t2 2>&1 | FileCheck %s

// XFAIL: arch-intel_gpu_pvc
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/16401

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
