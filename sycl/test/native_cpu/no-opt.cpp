// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -O0 -o %t.bc %s

#include "sycl.hpp"
class Test1;
int main() {
  const size_t N = 4;
  sycl::buffer<size_t, 1> Buffer(N);
  sycl::queue deviceQueue;
  sycl::accessor<int, 1, sycl::access::mode::write> acc;
  sycl::range<1> r(1);
  deviceQueue
      .submit([&](sycl::handler &h) {
        sycl::accessor Accessor{Buffer, h, sycl::write_only};
        h.parallel_for<Test1>(r, [=](sycl::id<1> id) { acc[id[0]] = 42; });
      })
      .wait();
  sycl::host_accessor HostAccessor{Buffer, sycl::read_only};
  for (unsigned I = 0; I < N; I++) {
    if (HostAccessor[I] != 42)
      return 1;
  }
  return 0;
}
