// REQUIRES: native_cpu_be
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -g -O0 -o %t %s
// RUN: env ONEAPI_DEVICE_SELECTOR="native_cpu:cpu" %t
// RUN: env ONEAPI_DEVICE_SELECTOR="native_cpu:cpu" SYCL_DEVICE_ALLOWLIST="BackendName:native_cpu" %t

#include "sycl.hpp"
class Test1;
int main() {
  const size_t N = 4;
  sycl::buffer<size_t, 1> Buffer(N);
  sycl::queue deviceQueue;
  sycl::range<1> r(N);
  deviceQueue
      .submit([&](sycl::handler &h) {
        auto Accessor = Buffer.get_access<sycl::access::mode::write>(h);
        h.parallel_for<Test1>(r, [=](sycl::id<1> id) { Accessor[id[0]] = 42; });
      })
      .wait();
  sycl::host_accessor HostAccessor{Buffer, sycl::read_only};
  for (unsigned I = 0; I < N; I++) {
    if (HostAccessor[I] != 42)
      return 1;
  }
  return 0;
}
