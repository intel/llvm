// REQUIRES: native_cpu_ock
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %s -o %t
// RUN: env ONEAPI_DEVICE_SELECTOR="native_cpu:cpu" %t

// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -O0 -g %s -o %t_debug
// RUN: env ONEAPI_DEVICE_SELECTOR="native_cpu:cpu" %t_debug
#include <sycl/sycl.hpp>

using namespace sycl;

class Test;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;
int main() {
  queue q;
  constexpr unsigned N = 10;
  constexpr unsigned NumG = 2;
  range<1> localR{N};
  range<1> globalR{NumG * N};
  buffer<int, 1> Buffer(globalR);
  q.submit([&](handler &h) {
    auto acc = Buffer.get_access<sycl_write>(h);
    h.parallel_for<Test>(nd_range<1>{globalR, localR}, [=](nd_item<1> it) {
      acc[it.get_global_id(0)] = 0;
      it.barrier(access::fence_space::local_space);
      acc[it.get_global_id(0)]++;
    });
  });
  sycl::host_accessor HostAccessor{Buffer, sycl::read_only};
  for (unsigned i = 0; i < N * NumG; i++) {
    if (HostAccessor[i] != 1) {
      std::cout << "Error\n";
      return 1;
    }
  }
  std::cout << "Test passed\n";
  return 0;
}
